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
# чтобы получить порог яркости "это уже ткань, а не фон". Этим порогом находим все
# пятна и считаем по ним количество животных/срезов на фото — должен быть строгим,
# чтобы шум/пыль/блики не путались с животными.
SEED_SIGMA_MULTIPLIER = 4.0

# мягкий порог для ЛОКАЛЬНОГО расширения уже найденного пятна до его настоящей, но
# тусклой границы (у срезов мозга сигнал плавно спадает к фону, и заметная глазом
# ткань оставалась за пределами строгого порога — площадь занижалась, край обрезался).
GROW_SIGMA_MULTIPLIER = 1.5

# на сколько пикселей ядро пятна можно расширить при поиске тусклого края. Это ГЛАВНАЯ
# защита от "утечки" в посторонние яркие объекты (блики, пыль, соседние пятна) —
# расширение идёт не по связности через весь кадр (как при обычном гистерезисном
# пороге), а только в пределах этого фиксированного радиуса вокруг исходного ядра,
# даже если мягкий порог где-то дальше тоже пройден.
GROW_MARGIN_PX = 15


@dataclass
class Blob:
    mask: np.ndarray          # bool, размер как у исходного изображения
    centroid: tuple[float, float]  # (x, y)
    area: int


def _clean_binary(binary_bool: np.ndarray) -> np.ndarray:
    """Морфологическое открытие+закрытие — убирает единичные шумовые пиксели и
    мелкие дырки внутри пятна, не сдвигая заметно сам контур."""
    binary = binary_bool.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def detect_blobs(
    image: np.ndarray,
    min_area_px: int = MIN_BLOB_AREA_PX,
    sigma_multiplier: float = SEED_SIGMA_MULTIPLIER,
) -> list[Blob]:
    """Находит светлые пятна на тёмном фоне (единый строгий порог).

    Порог считается от фактического фона кадра (медиана + устойчивый разброс через MAD),
    а не через Otsu на нормализованном 0-255 диапазоне — так надёжнее при бликах/пыли на
    кадре. Даёт уверенное "ядро" каждого пятна; тусклый диффузный край (для срезов) можно
    затем безопасно добрать функцией `grow_blob_to_soft_edge`.
    """
    arr = image.astype(np.float32)
    blurred = cv2.GaussianBlur(arr, (5, 5), 0)

    median = float(np.median(blurred))
    mad = float(np.median(np.abs(blurred - median)))
    robust_std = 1.4826 * mad if mad > 0 else float(np.std(blurred))
    threshold = median + sigma_multiplier * robust_std

    binary = _clean_binary(blurred >= threshold)
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


def grow_blob_to_soft_edge(
    image: np.ndarray,
    blob: Blob,
    other_blobs: list[Blob] = (),
    grow_sigma_multiplier: float = GROW_SIGMA_MULTIPLIER,
    margin_px: int = GROW_MARGIN_PX,
) -> Blob:
    """Расширяет уже найденное пятно до его настоящей, но тусклой границы.

    Расширение ограничено фиксированным радиусом `margin_px` вокруг исходного ядра
    (дилатация ядра ∩ мягкий порог) — то есть не может "утечь" по связности через весь
    кадр в далёкий блик, пыль или соседнее пятно, даже если между ними и правда есть
    сплошная полоска пикселей выше мягкого порога. Используется только там, где реальная
    граница объекта размытая (срезы) — для черт с чёткой геометрией (например, носовая
    часть черепа, где форма пятна важна для поиска "талии") не применяется.

    `other_blobs` — все ОСТАЛЬНЫЕ найденные на этом фото пятна (например, соседние
    срезы того же животного). Их расширение на тот же радиус `margin_px` исключается
    из зоны роста — иначе если два среза сближены (расстояние между ними меньше
    ~2×margin_px, а тусклая "дымка" между ними и правда выше мягкого порога, что для
    соседних срезов одного препарата обычное дело), спорная полоска между ними попала
    бы в зону роста ОБОИХ одновременно, и одни и те же пиксели дублировались бы в двух
    масках сразу (исключать только чужое ядро недостаточно — спорная зона шире ядра).
    """
    arr = image.astype(np.float32)
    blurred = cv2.GaussianBlur(arr, (5, 5), 0)
    median = float(np.median(blurred))
    mad = float(np.median(np.abs(blurred - median)))
    robust_std = 1.4826 * mad if mad > 0 else float(np.std(blurred))
    grow_threshold = median + grow_sigma_multiplier * robust_std

    grow_binary = _clean_binary(blurred >= grow_threshold) > 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))

    # исключаем не только ядра соседей, а их расширение на ТОТ ЖЕ радиус margin_px —
    # иначе спорная зона между двумя близкими пятнами (тусклая "дымка" между ними,
    # выше мягкого порога, но не входящая ни в одно ядро) исключалась бы только из
    # чужого ядра, но не из чужой зоны роста, и одни и те же пиксели дублировались бы
    # сразу в обеих выращенных масках. Симметричное исключение оставляет спорную
    # полоску посередине НИЧЬЕЙ — консервативная недооценка с обеих сторон, зато без
    # пересечения и двойного счёта площади/яркости.
    others_mask = np.zeros_like(blob.mask)
    for ob in other_blobs:
        others_mask |= cv2.dilate(ob.mask.astype(np.uint8), kernel) > 0
    grow_binary &= ~others_mask

    dilated = cv2.dilate(blob.mask.astype(np.uint8), kernel) > 0

    grown = (dilated & grow_binary) | blob.mask
    ys, xs = np.nonzero(grown)
    return Blob(mask=grown, centroid=(float(xs.mean()), float(ys.mean())), area=int(grown.sum()))


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


def _split_by_largest_y_gap(blobs: list[Blob]) -> tuple[list[Blob], list[Blob]]:
    """Делит пятна на верхнюю и нижнюю полосу по самому большому разрыву по Y.

    Используется, чтобы отделить ряд черепов (сверху) от ряда целых мозгов (снизу) на
    фото «череп и мозг» по фактическому расположению на КОНКРЕТНОМ фото, а не считать,
    что достаточно взять "самое верхнее пятно в каждой колонке" — если у одного черепа
    носовая часть слишком тусклая и не находится вообще, "самым верхним" в этой колонке
    ошибочно окажется мозг снизу.

    Порог "заметности" самого большого разрыва сравнивается с типичным разрывом ВНУТРИ
    ряда (между соседними черепами или соседними мозгами) — он на реальных фото на
    порядок меньше разрыва МЕЖДУ рядами и не зависит от формы/вытянутости конкретного
    пятна (в отличие от высоты пятна — на практике черепа часто вытянуты почти на всю
    высоту своего ряда, вплотную к границе с рядом мозгов, так что сравнение с высотой
    пятна ненадёжно). При подсчёте типичного разрыва сам оцениваемый (самый большой)
    разрыв исключается — иначе при малом числе пятен медиана вырождается в сам этот
    разрыв, и он почти никогда не признаётся "достаточно большим".

    Если пятен всего два (напр. один череп + один его мозг — весь кадр с одним
    животным, `cols=1`) — сравнивать не с чем, и здесь мы просто делим по этому
    единственному разрыву: на фото «череп и мозг» два пятна почти всегда и есть ровно
    эта пара. Вызывающий код (`assign_rostral_blobs_to_grid`) в этом случае явно
    предупреждает пользователя о низкой уверенности.

    Если пятно всего одно — делить нечего, но и уверенности в том, что это череп,
    а не мозг, тоже нет (см. предупреждение там же).
    """
    if len(blobs) < 2:
        return list(blobs), []
    ordered = sorted(blobs, key=lambda b: b.centroid[1])
    gaps = [(ordered[i + 1].centroid[1] - ordered[i].centroid[1], i) for i in range(len(ordered) - 1)]
    gap, idx = max(gaps, key=lambda g: g[0])

    other_gaps = [g for g, i in gaps if i != idx]
    if other_gaps:
        threshold = max(float(np.median(other_gaps)) * 2.0, 30.0)
        if gap < threshold:
            return ordered, []

    return ordered[: idx + 1], ordered[idx + 1 :]


def assign_rostral_blobs_to_grid(
    blobs: list[Blob], cols: int
) -> tuple[dict[tuple[int, int], Blob], str | None]:
    """Раскладка пятен по животным для режима «носовая часть черепа».

    На фото «череп и мозг» под каждым черепом лежит ещё и целый мозг — в этом режиме
    он не анализируется вообще, это отдельный этап (срезы). Сначала все пятна на фото
    делятся на верхнюю полосу (черепа) и нижнюю (мозги) по самому большому разрыву по Y
    (см. `_split_by_largest_y_gap`) — это надёжнее, чем "самое верхнее пятно в колонке":
    если у одного черепа носовая часть не нашлась вообще, колонка просто останется
    пустой (доразметить вручную), а не подхватит мозг снизу как замену черепу.
    """
    top_band, bottom_band = _split_by_largest_y_gap(blobs)
    columns = _split_into_columns(top_band, cols)

    warning: str | None = None
    missing = [i + 1 for i, c in enumerate(columns) if not c]
    if len(columns) < cols or missing:
        counts = [len(c) for c in columns]
        warning = (
            f"Не для всех животных нашёлся череп на фото (ожидалось {cols}, "
            f"пятен в колонках: {counts}). Недостающие ячейки нужно доразметить вручную."
        )
    elif not bottom_band:
        # разрыв между рядом черепов и рядом мозгов не нашёлся (мало пятен на фото —
        # например, один череп вообще не даёт сигнала, и сравнивать не с чем) — не можем
        # быть уверены, что каждое найденное пятно действительно череп, а не мозг
        warning = (
            "Не удалось надёжно отличить ряд черепов от ряда мозгов на этом фото "
            "(слишком мало найденных пятен, чтобы сравнить их расположение). "
            "Автоматическая обводка ниже может ошибочно относиться к мозгу вместо "
            "черепа — обязательно проверьте эту разметку вручную."
        )

    assignment: dict[tuple[int, int], Blob] = {}
    for animal_index, column_blobs in enumerate(columns):
        if animal_index >= cols or not column_blobs:
            continue
        topmost = min(column_blobs, key=lambda b: b.centroid[1])
        assignment[(animal_index, 0)] = topmost

    return assignment, warning


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

    # переориентируем профиль так, чтобы нос был слева (индекс 0). По протоколу съёмки
    # черепа на фото «череп и мозг» всегда лежат носом вверх (к меньшим Y на кадре) —
    # используем это напрямую (знак Y-компоненты главной оси), а не пытаемся определить
    # направление по ширине профиля на кончиках. Ширина ненадёжна: у части черепов
    # противоположный (каудальный) конец среза тоже сужается (например, торчащий
    # обрубок спинного мозга) и тогда оказывается ещё уже носа — по ширине его и
    # приняли бы за нос, и в "носовую" часть попала бы почти вся мозговая коробка.
    if main_axis[1] >= 0:
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
    if group.mode == MaskMode.ROSTRAL_CUT:
        # фото «череп и мозг»: под каждым черепом на фото лежит ещё и целый мозг,
        # который в этом режиме не анализируется вообще — берём только черепа
        assignment, warning = assign_rostral_blobs_to_grid(blobs, group.cols)
        resolved_rows = 1
    else:
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
                other_blobs = [b for b in blobs if b is not blob]
                grown = grow_blob_to_soft_edge(image, blob, other_blobs)
                masks.append(
                    SpecimenMask(
                        animal_index=animal_index,
                        slice_index=slice_index,
                        mask=grown.mask,
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
