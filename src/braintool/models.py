"""Общие структуры данных программы."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class MaskMode(str, Enum):
    """Тип обводки, который нужно предложить на фото."""

    WHOLE_BLOB = "whole_blob"       # весь срез мозга целиком
    ROSTRAL_CUT = "rostral_cut"     # носовая (передняя) часть черепа


@dataclass
class GroupConfig:
    """Настройки одной группы (=одна папка с фото)."""

    name: str
    folder: Path
    mode: MaskMode
    cols: int = 5           # сколько животных на одном фото (по горизонтали)
    # сколько срезов/повторов на одно животное (по вертикали). None = определять
    # автоматически по количеству реально найденных пятен на каждом фото —
    # пользователю не нужно знать и вводить это число заранее.
    rows: int | None = None
    # фиксированная метка роли группы — для сравнения "контроль vs опыт" в результатах
    is_control: bool = False
    # произвольная метка группы (например, дата съёмки или "условие 2") — для
    # сравнения "все со всеми" по этой метке в результатах, когда одного деления
    # на группы недостаточно
    condition: str = ""

    def __post_init__(self) -> None:
        if self.cols < 1:
            raise ValueError("cols должно быть не меньше 1")
        if self.rows is not None and self.rows < 1:
            raise ValueError("rows должно быть не меньше 1, если задано")


@dataclass
class Shot:
    """Один «кадр» — набор файлов с разной выдержкой одного и того же вида."""

    group: GroupConfig
    shot_key: str                 # имя без части exp<число>, идентифицирует кадр
    exposure_files: dict[int, Path] = field(default_factory=dict)  # exposure_ms -> путь
    chosen_exposure: int | None = None

    @property
    def display_name(self) -> str:
        return f"{self.group.name} / {self.shot_key}"


@dataclass
class SpecimenMask:
    """Маска одного животного/среза на одном кадре."""

    animal_index: int             # индекс животного (столбец), 0-based
    slice_index: int              # индекс среза/повтора (строка), 0-based
    mask: np.ndarray              # bool-массив размера кадра, True = внутри маски
    accepted: bool = False
    # для режима ROSTRAL_CUT: две точки линии отреза в координатах изображения,
    # пока не начали рисовать кистью вручную. None = маска уже "растровая".
    cut_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    # маска целой головы (используется для пересчёта при перетаскивании линии отреза)
    source_blob: np.ndarray | None = None
    # точка, заведомо лежащая на "носовой" стороне исходной маски (задаётся один раз
    # при автоопределении и не пересчитывается) — используется как опорная при
    # перетаскивании линии отреза, чтобы сторона выбора не переворачивалась
    rostral_anchor: tuple[float, float] | None = None
    # редактирование маски точками контура (альтернатива кисти) — список вершин
    # полигона в координатах изображения, по кругу; None = маска пока чисто
    # растровая (обычный режим). Растровая mask остаётся источником истины для
    # площади/яркости — полигон это только способ её РЕДАКТИРОВАТЬ, при любом
    # изменении вершин полигон растеризуется обратно в mask (canvas.py)
    polygon: list[tuple[float, float]] | None = None


@dataclass
class ShotReview:
    """Результат разметки одного кадра: выбранное изображение + список масок."""

    shot: Shot
    image: np.ndarray             # оригинальное изображение (native dtype)
    display_image: np.ndarray     # изображение с растяжкой контраста, uint8, для показа
    masks: list[SpecimenMask] = field(default_factory=list)
    warning: str | None = None    # напр. "найдено 8 пятен вместо 10 — проверьте вручную"


@dataclass
class MeasurementRow:
    group: str
    animal_index: int
    slice_index: int
    area_px: int
    mean_intensity: float
    std_intensity: float
    source_file: str
    # что обводили на этом кадре — нужно, чтобы никогда не сравнивать/усреднять вместе
    # измерения среза мозга и носовой части черепа, даже если пользователь назвал обе
    # группы одинаково (например, "0.2mA" для обеих папок одного и того же животного)
    mode: str = ""
    # выдержка, на которой был снят выбранный кадр (мс) — нужна, чтобы предупредить,
    # если сравниваемые группы сняты на разной выдержке (яркость тогда несравнима),
    # а также чтобы можно было явно сравнивать группы на ОДНОЙ и той же выдержке
    exposure_ms: int | None = None
    # метка роли группы и произвольное условие (см. GroupConfig) — переносятся без
    # изменений, чтобы результаты можно было сравнивать не только "группа vs группа"
    is_control: bool = False
    condition: str = ""
