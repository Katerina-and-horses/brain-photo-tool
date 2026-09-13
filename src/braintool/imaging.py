"""Загрузка папки с фото и подбор лучшей выдержки (экспозиции)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from .models import GroupConfig, Shot

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# ищет "exp" + число в имени файла, например ex730em810exp150angle112.tif -> exposure=150
_EXPOSURE_RE = re.compile(r"exp(\d+)", re.IGNORECASE)

# доля пикселей на грани диапазона (>=250 из 255 / >=98% от макс. значения),
# после которой кадр считается «засвеченным»
SATURATION_FRACTION_LIMIT = 0.002


def load_image(path: Path) -> np.ndarray:
    """Читает изображение как numpy-массив, сохраняя исходную битность."""
    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(str(path))
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 3:
        # цветное фото — берём яркость (на случай обычных RGB-снимков, не флуоресценции)
        arr = np.mean(arr[..., :3], axis=-1).astype(arr.dtype)
    return arr


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
    файл становится отдельным кадром без выбора экспозиции.
    """
    files = sorted(
        p for p in group.folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    shots: dict[str, Shot] = {}
    for path in files:
        match = _EXPOSURE_RE.search(path.stem)
        if match:
            exposure = int(match.group(1))
            key = path.stem[: match.start()] + path.stem[match.end():]
        else:
            exposure = 0
            key = path.stem

        shot = shots.setdefault(key, Shot(group=group, shot_key=key))
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
