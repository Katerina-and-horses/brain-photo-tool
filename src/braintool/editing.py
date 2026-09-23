"""Структурная правка разметки кадра: добавить маску, перенести её к другому
животному / на другой номер среза, удалить (сессия 7).

Нумерация (с 0) всегда плотная: животные 0..N-1 слева направо, у каждого животного
срезы 0..K-1. Все функции меняют список `masks` НА МЕСТЕ (это тот же список, что
`ShotReview.masks` и `MaskCanvas.masks`) — так отмена (Ctrl+Z) и холст видят одно и то же.
"""
from __future__ import annotations

import numpy as np

from .models import SpecimenMask


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def animal_count(masks: list[SpecimenMask]) -> int:
    return max((m.animal_index for m in masks), default=-1) + 1


def slices_of(masks: list[SpecimenMask], animal_index: int) -> list[SpecimenMask]:
    return sorted((m for m in masks if m.animal_index == animal_index), key=lambda m: m.slice_index)


def _animal_x(masks: list[SpecimenMask]) -> dict[int, float]:
    """Положение животного по X — медиана центров его непустых масок."""
    xs: dict[int, list[float]] = {}
    for m in masks:
        c = mask_centroid(m.mask)
        if c is not None:
            xs.setdefault(m.animal_index, []).append(c[0])
    return {a: float(np.median(v)) for a, v in xs.items()}


def _mask_width(mask: np.ndarray) -> float:
    cols = np.flatnonzero(mask.any(axis=0))
    return float(cols[-1] - cols[0] + 1) if cols.size else 0.0


def guess_animal(masks: list[SpecimenMask], new_mask: np.ndarray) -> tuple[int, bool]:
    """К какому животному относится новая маска: (номер, новое_ли_животное).

    Ближайшее по X животное, если до его столбца не дальше ширины типичного среза;
    иначе — новое животное, которое встанет между соседями по X (номера правее
    сдвинутся). Ошибку человек поправит кнопкой «Переназначить».
    """
    c = mask_centroid(new_mask)
    positions = _animal_x(masks)
    if c is None or not positions:
        return animal_count(masks), True
    widths = [w for w in (_mask_width(m.mask) for m in masks) if w > 0]
    limit = float(np.median(widths)) if widths else 80.0
    nearest = min(positions, key=lambda a: abs(positions[a] - c[0]))
    if abs(positions[nearest] - c[0]) <= limit:
        return nearest, False
    # новое животное — место по X среди существующих
    left_of = sum(1 for x in positions.values() if x < c[0])
    return left_of, True


def _slice_position_by_y(masks: list[SpecimenMask], animal_index: int, new_mask: np.ndarray) -> int:
    c = mask_centroid(new_mask)
    existing = slices_of(masks, animal_index)
    if c is None:
        return len(existing)
    pos = 0
    for m in existing:
        mc = mask_centroid(m.mask)
        if mc is not None and mc[1] < c[1]:
            pos = m.slice_index + 1
    return pos


def _make_room_for_animal(masks: list[SpecimenMask], animal_index: int) -> None:
    for m in masks:
        if m.animal_index >= animal_index:
            m.animal_index += 1


def _make_room_for_slice(masks: list[SpecimenMask], animal_index: int, slice_index: int) -> None:
    for m in masks:
        if m.animal_index == animal_index and m.slice_index >= slice_index:
            m.slice_index += 1


def _compact(masks: list[SpecimenMask]) -> None:
    """Убирает дыры в нумерации после удаления/переноса: животные 0..N-1 (в прежнем
    порядке), у каждого срезы 0..K-1 (в прежнем порядке)."""
    animals = sorted({m.animal_index for m in masks})
    remap = {a: i for i, a in enumerate(animals)}
    for m in masks:
        m.animal_index = remap[m.animal_index]
    for a in range(len(animals)):
        for i, m in enumerate(slices_of(masks, a)):
            m.slice_index = i
    masks.sort(key=lambda m: (m.animal_index, m.slice_index))


def add_mask(
    masks: list[SpecimenMask],
    new_mask: np.ndarray,
    animal_index: int | None = None,
    new_animal: bool = False,
) -> SpecimenMask:
    """Добавляет маску. Без `animal_index` животное угадывается по X (`guess_animal`);
    номер среза — по положению по Y среди срезов этого животного (срез выше всех
    станет срезом 1, остальные сдвинутся). `new_animal=True` — вставить НОВОЕ животное
    на место `animal_index` (остальные правее сдвигаются)."""
    if animal_index is None:
        animal_index, new_animal = guess_animal(masks, new_mask)
    if new_animal:
        _make_room_for_animal(masks, animal_index)
        slice_index = 0
    else:
        slice_index = _slice_position_by_y(masks, animal_index, new_mask)
        _make_room_for_slice(masks, animal_index, slice_index)
    m = SpecimenMask(animal_index=animal_index, slice_index=slice_index, mask=new_mask)
    masks.append(m)
    masks.sort(key=lambda x: (x.animal_index, x.slice_index))
    return m


def move_mask(
    masks: list[SpecimenMask],
    m: SpecimenMask,
    animal_index: int,
    slice_index: int | None,
    new_animal: bool = False,
) -> None:
    """Переносит маску к животному `animal_index` на место среза `slice_index` (с 0;
    None — по положению по Y). Срезы с этим номером и дальше сдвигаются на один вниз.
    `new_animal=True` — маска становится отдельным новым животным на месте
    `animal_index`. Опустевшее животное исчезает, номера уплотняются."""
    masks.remove(m)
    # номер назначения задан в нумерации ДО переноса: если исходное животное
    # опустело и исчезнет при уплотнении, животные правее него сдвинутся на один
    old_animal = m.animal_index
    source_emptied = not any(o.animal_index == old_animal for o in masks)
    if source_emptied and animal_index > old_animal:
        animal_index -= 1
    _compact(masks)
    if new_animal:
        _make_room_for_animal(masks, animal_index)
        m.animal_index, m.slice_index = animal_index, 0
    else:
        if slice_index is None:
            slice_index = _slice_position_by_y(masks, animal_index, m.mask)
        slice_index = min(slice_index, len(slices_of(masks, animal_index)))
        _make_room_for_slice(masks, animal_index, slice_index)
        m.animal_index, m.slice_index = animal_index, slice_index
    m.accepted = False
    masks.append(m)
    _compact(masks)


def delete_mask(masks: list[SpecimenMask], m: SpecimenMask) -> None:
    masks.remove(m)
    if masks:
        _compact(masks)
