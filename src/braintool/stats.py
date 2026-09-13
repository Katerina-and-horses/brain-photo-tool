"""Сравнение групп: Манн-Уитни для двух групп, Краскел-Уоллис для трёх и более."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без экрана, только сохранение в файл
import matplotlib.pyplot as plt
import pandas as pd

from . import nonparametric


@dataclass
class PairwiseResult:
    group_a: str
    group_b: str
    p_value_raw: float
    p_value: float          # после поправки на множественные сравнения (Холм), если применялась
    effect_size: float       # ранговая бисериальная корреляция, -1..1
    min_possible_p: float    # наименьшее p, в принципе достижимое при таком n1,n2


@dataclass
class ComparisonResult:
    metric: str
    groups: list[str]
    test_name: str
    p_value: float
    pairwise: list[PairwiseResult]
    n_per_group: dict[str, int]
    holm_applied: bool


def compare_groups(df: pd.DataFrame, metric: str) -> ComparisonResult:
    """df должен содержать колонки "group" и `metric`, ОДНА СТРОКА = ОДНО ЖИВОТНОЕ
    (то есть уже усреднённая по животному таблица — см. `measurements.per_animal_average`).
    Сравнение групп на неусреднённой по животному таблице (несколько срезов одного
    животного как отдельные наблюдения) статистически некорректно — это псевдоповторность,
    искусственно завышающая размер выборки и занижающая p-value. Поэтому эта функция
    не принимает "сырую" таблицу по срезам — вызывающий код обязан агрегировать её
    заранее."""
    groups = sorted(df["group"].unique())
    samples = [df.loc[df["group"] == g, metric].dropna().to_numpy() for g in groups]
    n_per_group = {g: len(s) for g, s in zip(groups, samples)}

    if len(groups) < 2:
        raise ValueError("Нужно минимум 2 группы для сравнения")

    if len(groups) == 2:
        n1, n2 = len(samples[0]), len(samples[1])
        u1, overall_p = nonparametric.mannwhitneyu(list(samples[0]), list(samples[1]))
        test_name = "Манн-Уитни"
        pairwise = [PairwiseResult(
            groups[0], groups[1], p_value_raw=overall_p, p_value=overall_p,
            effect_size=nonparametric.mannwhitney_effect_size(u1, n1, n2),
            min_possible_p=nonparametric.mannwhitney_min_possible_p(n1, n2),
        )]
        holm_applied = False
    else:
        _, overall_p = nonparametric.kruskal([list(s) for s in samples])
        test_name = "Краскел-Уоллис"
        pairs = list(combinations(enumerate(groups), 2))
        raw_results = []
        for (i, gi), (j, gj) in pairs:
            ni, nj = len(samples[i]), len(samples[j])
            u1, p = nonparametric.mannwhitneyu(list(samples[i]), list(samples[j]))
            raw_results.append((gi, gj, p, nonparametric.mannwhitney_effect_size(u1, ni, nj),
                                 nonparametric.mannwhitney_min_possible_p(ni, nj)))
        adjusted = nonparametric.holm_bonferroni([r[2] for r in raw_results])
        pairwise = [
            PairwiseResult(gi, gj, p_value_raw=p_raw, p_value=p_adj, effect_size=eff, min_possible_p=min_p)
            for (gi, gj, p_raw, eff, min_p), p_adj in zip(raw_results, adjusted)
        ]
        holm_applied = True

    return ComparisonResult(
        metric=metric, groups=groups, test_name=test_name,
        p_value=overall_p, pairwise=pairwise, n_per_group=n_per_group, holm_applied=holm_applied,
    )


def save_boxplot(df: pd.DataFrame, metric: str, out_path: Path, title: str | None = None) -> None:
    groups = sorted(df["group"].unique())
    data = [df.loc[df["group"] == g, metric].dropna().to_numpy() for g in groups]

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(groups) + 2), 5))
    ax.boxplot(data, tick_labels=groups, showmeans=True)
    for i, d in enumerate(data, start=1):
        ax.scatter([i] * len(d), d, alpha=0.6, s=20, color="#555555")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
