"""Автоматическое предложение масок: поиск светящихся пятен и их разбор по сетке
"животное x срез", плюс эвристика для отделения носовой части черепа.

Это только ПЕРВОЕ ПРИБЛИЖЕНИЕ — пользователь всегда может поправить результат
руками в окне проверки. Поэтому эвристики здесь намеренно простые и быстрые,
а не идеально точные.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import GroupConfig, MaskMode, SpecimenMask

# минимальная площадь пятна в пикселях, чтобы не путать шум/пыль/битые пиксели с животным
# (реальные срезы/черепа в образцах — от нескольких сотен пикселей, шумовые точки — до ~150)
MIN_BLOB_AREA_PX = 200

# во сколько раз разброс фона (устойчивая оценка через MAD) добавляется к медиане фона,
# чтобы получить порог яркости "это уже ткань, а не фон". Подобрано опытным путём на
# реальных фото: при Otsu на глобально нормализованном кадре порог получался завышенным
# и обрезал заметный глазом, но тусклый, диффузный край ткани вокруг яркого ядра пятна —
# из-за этого измеренная площадь среза выходила заметно меньше настоящей
BACKGROUND_SIGMA_MULTIPLIER = 4.0


@dataclass
class Blob:
    mask: np.ndarray          # bool, размер как у исходного изображения
    centroid: tuple[float, float]  # (x, y)
    area: int


def detect_blobs(
    image: np.ndarray,
    min_area_px: int = MIN_BLOB_AREA_PX,
    sigma_multiplier: float = BACKGROUND_SIGMA_MULTIPLIER,
) -> list[Blob]:
    """Находит светлые пятна на тёмном фоне.

    Порог считается от фактического фона кадра (медиана + устойчивый разброс через MAD),
    а не через Otsu на нормализованном 0-255 диапазоне: у флуоресцентных снимков сигнал
    часто представляет собой яркое ядро с плавно затухающим, но реальным диффузным краем
    ткани, а Otsu на глобально растянутом гистограммой кадре (искажённом яркими точками
    пыли/бликов) отсекал именно этот край, занижая измеряемую площадь.
    """
    arr = image.astype(np.float32)
    blurred = cv2.GaussianBlur(arr, (5, 5), 0)

    median = float(np.median(blurred))
    mad = float(np.median(np.abs(blurred - median)))
    robust_std = 1.4826 * mad if mad > 0 else float(np.std(blurred))
    threshold = median + sigma_multiplier * robust_std

    binary = (blurred >= threshold).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    blobs: list[Blob] = []
    for label in range(1, num_labels):  # label 0 = фон
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        mask = labels == label
        cx, cy = centroids[label]
        blobs.append(Blob(mask=mask, centroid=(float(cx), float(cy)), area=area))
    return blobs


def _split_into_columns(blobs: list[Blob], cols: int) -> list[list[Blob]]:
    """Делит пятна на `cols` групп по X-координате, используя самые большие разрывы."""
    if not blobs:
        return []
    ordered = sorted(blobs, key=lambda b: b.centroid[0])
    if cols <= 1 or len(ordered) <= cols:
        # либо одна колонка, либо пятен меньше/столько же, сколько ожидалось колонок —
        # каждое пятно получает свою колонку (сигнал, что что-то не совпало с ожиданием)
        return [[b] for b in ordered] if len(ordered) <= cols and cols > 1 else [ordered]

    gaps = [(ordered[i + 1].centroid[0] - ordered[i].centroid[0], i) for i in range(len(ordered) - 1)]
    gaps.sort(key=lambda g: g[0], reverse=True)
    split_after = sorted(i for _, i in gaps[: cols - 1])

    columns: list[list[Blob]] = []
    start = 0
    for idx in split_after:
        columns.append(ordered[start: idx + 1])
        start = idx + 1
    columns.append(ordered[start:])
    return columns


def assign_blobs_to_grid(
    blobs: list[Blob], cols: int, rows: int | None = None
) -> tuple[dict[tuple[int, int], Blob], int, str | None]:
    """Раскладывает найденные пятна по сетке (индекс животного, индекс среза).

    Если `rows` не задан — число срезов на животное определяется автоматически:
    берётся максимальное количество пятен, найденное в одном из столбцов
    (животных обычно проще посчитать точно, т.к. они разнесены по горизонтали
    сильнее, чем срезы одного животного по вертикали).

    Возвращает (словарь {(animal_index, slice_index): Blob}, итоговое число строк
    сетки, текст предупреждения либо None).
    """
    columns = _split_into_columns(blobs, cols)
    counts = [len(c) for c in columns]

    if rows is None:
        resolved_rows = max(counts) if counts else 1
    else:
        resolved_rows = rows

    warning: str | None = None
    if rows is None:
        if counts and len(set(counts)) > 1:
            details = ", ".join(f"животное {i + 1}: {c}" for i, c in enumerate(counts))
            warning = (
                f"На фото найдено разное число срезов у разных животных ({details}). "
                f"Взято максимальное ({resolved_rows}) — недостающие ячейки нужно доразметить вручную."
            )
    else:
        expected = rows * cols
        if len(blobs) != expected:
            warning = (
                f"Найдено пятен: {len(blobs)}, ожидалось по настройке сетки: {expected} "
                f"({cols} животных x {rows} срезов). Проверьте разметку на этом фото вручную."
            )

    assignment: dict[tuple[int, int], Blob] = {}
    for animal_index, column_blobs in enumerate(columns):
        if animal_index >= cols:
            break
        column_sorted = sorted(column_blobs, key=lambda b: b.centroid[1])
        for slice_index, blob in enumerate(column_sorted):
            if slice_index >= resolved_rows:
                break
            assignment[(animal_index, slice_index)] = blob

    return assignment, resolved_rows, warning


def _find_rostral_boundary_bin(profile: np.ndarray) -> int:
    """Находит границу рострального (носового) отдела в профиле ширины.

    `profile` ориентирован так, что индекс 0 — узкий "носовой" конец черепа.
    Череп сверху/снизу обычно расширяется от кончика носа к скулам/глазницам
    (первый локальный максимум ширины), затем слегка сужается ("талия" перед
    мозговой коробкой), а после этого сильно расширяется на саму мозговую
    коробку (глобальный максимум ширины). Границу рострального отдела
    анатомически естественно проводить именно по этой "талии" — самому
    выраженному локальному минимуму ширины между носом и мозговой коробкой.

    Старая версия резала там, где ширина впервые достигала заданной доли от
    максимума — это почти всегда попадало на самый подъём от кончика носа к
    первому расширению (скулы), т.е. в первую же четверть черепа, отрезая
    только крошечный кончик носа вместо всего носового/лицевого отдела.
    """
    n = len(profile)
    global_max_idx = int(np.argmax(profile))

    # "талия" может быть только между носом и мозговой коробкой (глобальным максимумом);
    # исключаем самый кончик носа и всё, что после мозговой коробки
    lo_bound = max(1, int(n * 0.12))
    hi_bound = min(global_max_idx, int(n * 0.75))
    if hi_bound <= lo_bound:
        # профиль нарастает монотонно, выраженной "талии" нет — берём консервативную
        # фиксированную долю длины черепа как ростральный отдел
        return max(1, int(n * 0.35))

    minima = [
        i for i in range(lo_bound, hi_bound)
        if profile[i] <= profile[i - 1] and profile[i] <= profile[i + 1]
    ]
    if not minima:
        return max(1, int(n * 0.35))

    # среди локальных минимумов берём тот, что глубже всего относительно
    # предшествующего ему локального максимума — самая выраженная "талия"
    best_idx, best_depth = minima[0], -1.0
    for i in minima:
        preceding_peak = float(np.max(profile[: i + 1]))
        depth = preceding_peak - profile[i]
        if depth > best_depth:
            best_depth = depth
            best_idx = i
    return best_idx


def compute_rostral_cut(
    blob_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]]]:
    """Отделяет "носовую" (переднюю, лицевую) часть черепа от мозговой коробки.

    Находим главную ось вытянутости пятна (PCA), строим профиль ширины пятна
    вдоль этой оси и ищем анатомическую "талию" между лицевым отделом и
    мозговой коробкой (см. `_find_rostral_boundary_bin`). Это только стартовое
    предложение — линию отреза можно будет перетащить в интерфейсе.
    """
    ys, xs = np.nonzero(blob_mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, int(np.argmax(eigvals))]

    proj = centered @ main_axis

    n_bins = min(60, max(10, len(proj) // 20))
    lo, hi = proj.min(), proj.max()
    if hi <= lo:
        hi = lo + 1.0
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(proj, bin_edges) - 1, 0, n_bins - 1)

    counts = np.zeros(n_bins)
    for bi in range(n_bins):
        counts[bi] = int(np.sum(bin_idx == bi))

    smooth_window = max(3, n_bins // 10)
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(counts, kernel, mode="same")

    tip_start_w = float(smoothed[: max(1, n_bins // 10)].mean())
    tip_end_w = float(smoothed[-max(1, n_bins // 10):].mean())

    # переориентируем профиль так, чтобы нос (более узкий конец) был слева (индекс 0)
    if tip_start_w <= tip_end_w:
        direction = 1
        profile = smoothed
    else:
        direction = -1
        profile = smoothed[::-1]

    boundary_in_profile = _find_rostral_boundary_bin(profile)
    cut_bin = boundary_in_profile if direction == 1 else (n_bins - 1 - boundary_in_profile)

    cut_proj_value = bin_edges[cut_bin] if direction == 1 else bin_edges[cut_bin + 1]

    rostral_sel = proj <= cut_proj_value if direction == 1 else proj >= cut_proj_value
    rostral_mask = np.zeros_like(blob_mask, dtype=bool)
    rostral_mask[ys[rostral_sel], xs[rostral_sel]] = True

    perp_axis = np.array([-main_axis[1], main_axis[0]])
    center_at_cut = mean + main_axis * cut_proj_value
    half_len = max(blob_mask.shape) * 0.35
    p1 = tuple(center_at_cut + perp_axis * half_len)
    p2 = tuple(center_at_cut - perp_axis * half_len)

    return rostral_mask, (p1, p2)


def build_masks_for_image(
    image: np.ndarray, group: GroupConfig
) -> tuple[list[SpecimenMask], str | None]:
    """Главная функция: находит животных на фото и строит для них маски по режиму группы.

    Для каждой ячейки сетки (животное x срез) всегда создаётся запись — даже если
    авто-детекция не нашла для неё пятно. В этом случае маска создаётся пустой,
    чтобы её можно было выбрать в интерфейсе и дорисовать кистью вручную, а не
    "потерять" ячейку молча.
    """
    blobs = detect_blobs(image)
    assignment, resolved_rows, warning = assign_blobs_to_grid(blobs, group.cols, group.rows)

    masks: list[SpecimenMask] = []
    for animal_index in range(group.cols):
        for slice_index in range(resolved_rows):
            blob = assignment.get((animal_index, slice_index))
            if blob is None:
                masks.append(
                    SpecimenMask(
                        animal_index=animal_index,
                        slice_index=slice_index,
                        mask=np.zeros(image.shape[:2], dtype=bool),
                    )
                )
            elif group.mode == MaskMode.WHOLE_BLOB:
                masks.append(
                    SpecimenMask(
                        animal_index=animal_index,
                        slice_index=slice_index,
                        mask=blob.mask,
                    )
                )
            else:  # ROSTRAL_CUT
                rostral_mask, cut_line = compute_rostral_cut(blob.mask)
                masks.append(
                    SpecimenMask(
                        animal_index=animal_index,
                        slice_index=slice_index,
                        mask=rostral_mask,
                        cut_line=cut_line,
                        source_blob=blob.mask,
                        rostral_anchor=_mask_centroid(rostral_mask),
                    )
                )

    masks.sort(key=lambda m: (m.animal_index, m.slice_index))
    return masks, warning


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def recompute_rostral_mask_from_line(
    source_blob: np.ndarray,
    line: tuple[tuple[float, float], tuple[float, float]],
    rostral_side_point: tuple[float, float],
) -> np.ndarray:
    """Пересчитывает маску носовой части после того, как пользователь подвинул линию отреза.

    `rostral_side_point` — точка, заведомо находящаяся на "носовой" стороне
    (центроид исходной маски до правки), чтобы понять, какую сторону от линии оставлять.
    """
    p1, p2 = np.array(line[0]), np.array(line[1])
    direction = p2 - p1
    normal = np.array([-direction[1], direction[0]])

    ys, xs = np.nonzero(source_blob)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    side = (pts - p1) @ normal

    ref_side = np.dot(np.array(rostral_side_point) - p1, normal)
    keep = side >= 0 if ref_side >= 0 else side <= 0

    out = np.zeros_like(source_blob, dtype=bool)
    out[ys[keep], xs[keep]] = True
    return out
