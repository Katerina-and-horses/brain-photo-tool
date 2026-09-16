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

# Категориальная палитра (см. навык dataviz проекта) — фиксированный порядок
# оттенков, провалидированный на различимость при дальтонизме; не переставлять и
# не выбирать цвета "на глаз". Используется и для графика, и (в review_window.py)
# для подсветки строк таблицы результатов по группе — один и тот же цвет группы
# в обоих местах.
CATEGORICAL_PALETTE: list[str] = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]


def color_for_group(index: int) -> str:
    """Цвет для группы по её порядковому номеру (0-based) в отсортированном списке
    групп. Больше 8 групп палитра не покрывает — тогда цвета неизбежно повторяются
    (сознательный компромисс для внутреннего лабораторного инструмента, а не общий
    рецепт: см. dataviz — "9-я серия не генерируется, а уходит в 'Другое'/фасетку")."""
    return CATEGORICAL_PALETTE[index % len(CATEGORICAL_PALETTE)]


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
    empty_groups = [g for g, n in n_per_group.items() if n == 0]
    if empty_groups:
        raise ValueError(
            "В группе(ах) " + ", ".join(f"«{g}»" for g in empty_groups) + " нет ни одного "
            f"животного с посчитанным показателем «{metric}» (например, для него у всех "
            "принятых масок площадь оказалась нулевой) — сравнение невозможно."
        )

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
    colors = [color_for_group(i) for i in range(len(groups))]
    # n прямо в подписи оси — жалоба была не на отсутствие данных (они и так шли в
    # compare_groups), а на то, что размер выборки не виден на самом графике
    tick_labels = [f"{g}\n(n={len(d)})" for g, d in zip(groups, data)]

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(groups) + 2), 5))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    bp = ax.boxplot(data, tick_labels=tick_labels, showmeans=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("#0b0b0b")
    # цвет точек = цвет группы (та же палитра, что и подсветка строк таблицы
    # результатов в review_window.py) — группа узнаётся по цвету в обоих местах
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        ax.scatter([i] * len(d), d, alpha=0.85, s=24, color=color,
                    edgecolor="#0b0b0b", linewidth=0.4, zorder=3)
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.tick_params(colors="#52514e")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
