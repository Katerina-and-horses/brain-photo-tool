"""Загрузка папки с фото и подбор лучшей выдержки (экспозиции)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

from .models import GroupConfig, Shot

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# ищет "exp" + число в имени файла, например ex730em810exp150angle112.tif -> exposure=150.
# В части папок лаборатории «p» пропущено: ex730em810ex10angle112.tif — «ex» сразу после
# цифры (не в начале, где ex730 — длина волны) тоже выдержка (сессия 11: 73 таких файла
# на данных с Яндекс.Диска, раньше каждый был отдельным «кадром» без выдержки)
_EXPOSURE_RE = re.compile(r"exp(\d+)|(?<=\d)ex(\d+)", re.IGNORECASE)

# снимок при внешнем свете: «Reference (Filt695 exp2000).tif», «Ref4000.tif», «ref 1_2exp.tif»
# (так их называют в лаборатории, 147 папок из 345 на Яндекс.Диске). Это не
# флуоресценция — анализировать его яркость бессмысленно, но форма черепов на нём
# видна целиком (сессия 10)
_REFERENCE_RE = re.compile(r"^\s*ref", re.IGNORECASE)


def is_reference_file(path: Path) -> bool:
    return bool(_REFERENCE_RE.match(Path(path).stem))

# доля пикселей на грани диапазона (>=250 из 255 / >=98% от макс. значения),
# после которой кадр считается «засвеченным»
SATURATION_FRACTION_LIMIT = 0.002
# засвет ВНУТРИ принятых масок, при котором выдержка не годится для анализа яркости
# (0,1% пикселей маски у предела шкалы; решение пользователя, сессия 7)
MASK_SATURATION_FRACTION_LIMIT = 0.001


def load_image(path: Path) -> np.ndarray:
    """Читает изображение как numpy-массив, сохраняя исходную битность.

    Кешируется (одни и те же файлы выдержек перечитываются при подборе выдержки,
    пересчёте таблицы и т.п.); массив из кеша помечен только-для-чтения — менять
    его на месте нельзя, только копию."""
    return _load_image_cached(str(Path(path).resolve()), Path(path).stat().st_mtime_ns)


@lru_cache(maxsize=48)
def _load_image_cached(path_str: str, _mtime_ns: int) -> np.ndarray:
    arr = _read_image(Path(path_str))
    arr.setflags(write=False)
    return arr


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(str(path))
    else:
        with Image.open(path) as img:
            arr = np.array(img)
    if arr.ndim == 3:
        # цветное фото — берём яркость (на случай обычных RGB-снимков, не флуоресценции)
        arr = np.mean(arr[..., :3], axis=-1).astype(arr.dtype)
    return arr


def load_reference(shot: Shot, shape: tuple[int, ...]) -> np.ndarray | None:
    """Снимок при внешнем свете для кадра, приведённый к размеру кадра `shape`, или
    None — если его нет, он не читается или у него другие пропорции (значит, снят
    не в том же кадре, и совмещать нельзя). Бывает снят в большем разрешении (5472×3672
    при кадрах 1368×918) — тогда уменьшается."""
    if shot.reference_file is None:
        return None
    try:
        ref = load_image(shot.reference_file)
    except Exception:  # noqa: BLE001
        return None
    h, w = shape[:2]
    rh, rw = ref.shape[:2]
    if (rh, rw) == (h, w):
        return ref
    if abs(rw / rh - w / h) > 0.01 * (w / h):
        return None
    return cv2.resize(ref.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


def contrast_stretch_to_uint8(arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> np.ndarray:
    """Растягивает контраст для показа на экране (не влияет на измерения)."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
    return out.astype(np.uint8)


def _saturation_fraction(arr: np.ndarray) -> float | None:
    """Доля пикселей у предела шкалы. Возвращает None для нецелочисленных (float)
    изображений — там нет известного "предела датчика", а сравнение с максимумом
    этого же кадра было бы самоотносительным и вводило бы в заблуждение (у любого
    кадра всегда есть хотя бы один пиксель, равный его собственному максимуму)."""
    if not np.issubdtype(arr.dtype, np.integer):
        return None
    max_possible = float(np.iinfo(arr.dtype).max)
    threshold = max_possible * 0.98
    return float(np.mean(arr >= threshold))


def scan_group_folder(group: GroupConfig) -> list[Shot]:
    """Находит все кадры в папке группы и группирует файлы по выдержкам.

    Файлы, чьё имя отличается только числом после "exp", считаются одним
    кадром с разными выдержками. Если в имени файла нет "exp<число>",
    файл становится отдельным кадром без выбора экспозиции. Снимки при внешнем
    свете («Reference…», «Ref…») кадрами не считаются — первый из них становится
    `reference_file` всех кадров группы.
    """
    files = sorted(
        p for p in group.folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    references = [p for p in files if is_reference_file(p)]
    reference = references[0] if references else None
    shots: dict[str, Shot] = {}
    for path in files:
        if path in references:
            continue
        match = _EXPOSURE_RE.search(path.stem)
        if match:
            exposure = int(match.group(1) or match.group(2))
            key = path.stem[: match.start()] + path.stem[match.end():]
        else:
            exposure = 0
            key = path.stem

        shot = shots.setdefault(key, Shot(group=group, shot_key=key, reference_file=reference))
        shot.exposure_files[exposure] = path

    return [shots[k] for k in sorted(shots.keys())]


def choose_best_exposure(shot: Shot) -> Path:
    """Выбирает не засвеченный кадр с максимальным сигналом среди доступных выдержек."""
    candidates = sorted(shot.exposure_files.items())  # (exposure, path), по возрастанию
    if len(candidates) == 1:
        shot.chosen_exposure = candidates[0][0]
        return candidates[0][1]

    best_ok: tuple[int, Path] | None = None
    fallback_best: tuple[int, Path, float] | None = None  # наименьшая засветка на всякий случай

    for exposure, path in candidates:
        arr = load_image(path)
        frac = _saturation_fraction(arr)
        if frac is None:
            # нецелочисленные данные — не можем оценить засветку, берём максимальную выдержку
            if best_ok is None or exposure > best_ok[0]:
                best_ok = (exposure, path)
            continue
        if frac <= SATURATION_FRACTION_LIMIT:
            if best_ok is None or exposure > best_ok[0]:
                best_ok = (exposure, path)
        if fallback_best is None or frac < fallback_best[2]:
            fallback_best = (exposure, path, frac)

    chosen = best_ok if best_ok is not None else fallback_best[:2]
    shot.chosen_exposure = chosen[0]
    return chosen[1]


def saturation_limit_value(arr: np.ndarray) -> float | None:
    """Значение, начиная с которого пиксель считается засвеченным (98% шкалы файла),
    или None для нецелочисленных изображений (предел датчика неизвестен)."""
    if not np.issubdtype(arr.dtype, np.integer):
        return None
    return float(np.iinfo(arr.dtype).max) * 0.98


def unsaturated_exposures(shot: Shot) -> list[int]:
    """Выдержки кадра без засвета по всему кадру (как в `choose_best_exposure`), от
    самой длинной к самой короткой."""
    ok: list[int] = []
    for exposure, path in sorted(shot.exposure_files.items(), reverse=True):
        frac = _saturation_fraction(load_image(path))
        if frac is None or frac <= SATURATION_FRACTION_LIMIT:
            ok.append(exposure)
    return ok


def choose_mask_exposure(shot: Shot) -> int:
    """Выдержка, на которой ищутся маски (сессия 7): ВТОРАЯ по длине незасвеченная —
    на самой длинной вокруг срезов больше свечения и мусора, а форма среза от выдержки
    не зависит. Если незасвеченная одна — она; если все засвечены — наименее
    засвеченная (как раньше в `choose_best_exposure`)."""
    ok = unsaturated_exposures(shot)
    if len(ok) >= 2:
        return ok[1]
    if ok:
        return ok[0]
    choose_best_exposure(shot)
    return shot.chosen_exposure


def mask_saturation_fraction(image: np.ndarray, union_mask: np.ndarray) -> float:
    """Доля засвеченных пикселей внутри маски (0 для float-изображений и пустой маски)."""
    limit = saturation_limit_value(image)
    n = int(union_mask.sum())
    if limit is None or n == 0:
        return 0.0
    return float(np.count_nonzero(image[union_mask] >= limit)) / n
