"""Подсчёт площади и яркости по принятым маскам + экспорт таблицы."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import MeasurementRow, ShotReview


def measure_shot(review: ShotReview) -> list[MeasurementRow]:
    """Считает площадь/среднее/разброс только для масок, которые приняты пользователем."""
    rows: list[MeasurementRow] = []
    for m in review.masks:
        if not m.accepted:
            continue
        area = int(m.mask.sum())
        if area == 0:
            continue
        values = review.image[m.mask].astype(np.float64)
        rows.append(
            MeasurementRow(
                group=review.shot.group.name,
                animal_index=m.animal_index,
                slice_index=m.slice_index,
                area_px=area,
                mean_intensity=float(values.mean()),
                std_intensity=float(values.std()),
                source_file=review.shot.display_name,
                mode=review.shot.group.mode.value,
                exposure_ms=review.shot.chosen_exposure,
            )
        )
    return rows


def rows_to_dataframe(rows: list[MeasurementRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "group", "animal_index", "slice_index", "area_px",
                "mean_intensity", "std_intensity", "source_file",
                "mode", "exposure_ms",
            ]
        )
    return df


def per_animal_average(df: pd.DataFrame) -> pd.DataFrame:
    """Усредняет измерения по животному (все срезы одного животного -> одна строка).

    ОБЯЗАТЕЛЬНЫЙ шаг перед статистическим сравнением групп: срез — не независимое
    наблюдение (срезы одного животного коррелируют между собой), поэтому сравнивать
    группы на "сырых" срезах — псевдоповторность, искусственно раздувающая размер
    выборки и занижающая p-value.

    Животное определяется как (группа, кадр-фото, индекс столбца на этом фото), а
    НЕ просто (группа, индекс столбца) — animal_index это всего лишь "какой по счёту
    слева направо на конкретном фото" и не гарантированно означает одно и то же
    животное на разных фото одной группы. Если объединить по индексу столбца
    вслепую, можно молча усреднить двух РАЗНЫХ животных как одно. Поэтому по
    умолчанию каждое фото в группе даёт свой независимый набор "животных" —
    это самое безопасное допущение при отсутствии явной привязки животного к
    фото в интерфейсе.

    area_px усредняется как среднее по срезам (типичный размер среза у животного).
    mean_intensity и std_intensity объединяются с учётом площади (числа пикселей)
    каждого среза как веса — простое среднее арифметическое средних было бы верно
    только при одинаковой площади всех срезов, а среднее из нескольких SD вообще
    не является корректной оценкой общего разброса (нужно объединять через суммы
    квадратов отклонений, что и делается ниже).
    """
    if df.empty:
        return df
    # mode — в ключ группировки: даже если пользователь случайно назвал две группы
    # одинаково (например, папку "череп и мозг" и папку "срезы" одного животного назвал
    # одним и тем же именем группы), их измерения не должны усредниться в одну строку
    key_cols = ["group", "mode", "source_file", "animal_index"]
    out_rows: list[dict] = []
    for key, g in df.groupby(key_cols, sort=False):
        n = g["area_px"].to_numpy(dtype=np.float64)
        means = g["mean_intensity"].to_numpy(dtype=np.float64)
        stds = g["std_intensity"].to_numpy(dtype=np.float64)
        total_n = n.sum()
        if total_n > 0:
            pooled_mean = float(np.average(means, weights=n))
            pooled_var = float(np.sum(n * stds**2 + n * (means - pooled_mean) ** 2) / total_n)
        else:
            pooled_mean = float(means.mean())
            pooled_var = float((stds**2).mean())
        row = dict(zip(key_cols, key))
        row["area_px"] = float(g["area_px"].mean())
        row["mean_intensity"] = pooled_mean
        row["std_intensity"] = float(np.sqrt(max(pooled_var, 0.0)))
        row["exposure_ms"] = g["exposure_ms"].iloc[0]
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def export_table(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")
