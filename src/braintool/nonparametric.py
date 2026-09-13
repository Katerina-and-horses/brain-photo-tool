"""Свои реализации критерия Манна-Уитни и Краскела-Уоллиса на чистой математике
(без scipy). Возникло из-за известной несовместимости scipy + PyInstaller + Python 3.12
(https://github.com/pyinstaller/pyinstaller/issues/7992) — при сборке в exe scipy.stats
падает с NameError при импорте. Чтобы не патчить системные файлы Python на компьютере
пользователя, статистику для наших двух тестов считаем сами: это стандартные,
хорошо документированные формулы (ранговые тесты + регуляризованная гамма-функция
для хи-квадрат), без каких-либо приближений "на глаз".
"""
from __future__ import annotations

import math
from collections import Counter


def _rankdata_average(values: list[float]) -> list[float]:
    """Ранги 1..N с усреднением рангов внутри групп равных значений."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # ранги 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _mannwhitney_exact_p(u_stat: float, n1: int, n2: int) -> float:
    """Точное двустороннее p-value критерия Манна-Уитни (без учёта повторов) через
    стандартную рекурсию подсчёта распределения U. Память под кэш выделяется только
    на время одного вызова (а не на весь процесс), чтобы не копилась между сравнениями."""
    memo: dict[tuple[int, int, int], int] = {}

    def count(u: int, a: int, b: int) -> int:
        if u < 0:
            return 0
        if a == 0 or b == 0:
            return 1 if u == 0 else 0
        if u > a * b:
            return 0
        key = (u, a, b)
        cached = memo.get(key)
        if cached is not None:
            return cached
        result = count(u - b, a - 1, b) + count(u, a, b - 1)
        memo[key] = result
        return result

    total = math.comb(n1 + n2, n1)
    u_max = n1 * n2
    u_small = min(u_stat, u_max - u_stat)
    cum = sum(count(u, n1, n2) for u in range(0, int(round(u_small)) + 1))
    return min(1.0, 2 * cum / total)


# Предел на n1*n2, при котором ещё считаем точный тест — по замеру: n1=n2=40
# (n1*n2=1600) считается ~0.5с и ~400 тыс. состояний в памяти, n1=n2=80 (6400) —
# уже ~8с и ~6 млн состояний. Порог взят с запасом, чтобы правка на 1-2 группах
# с сотней измерений не подвешивала интерфейс; для больших выборок используется
# нормальное приближение (это стандартная практика и для scipy).
_EXACT_LIMIT = 2500


def mannwhitneyu(x: list[float], y: list[float]) -> tuple[float, float]:
    """Двусторонний критерий Манна-Уитни. Возвращает (U1, p-value).

    Для выборок без повторов и разумного размера считается точное распределение,
    иначе — нормальное приближение с поправкой на повторы и непрерывность
    (то же самое, что использует scipy при methode='asymptotic')."""
    n1, n2 = len(x), len(y)
    combined = list(x) + list(y)
    ranks = _rankdata_average(combined)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2

    counts = Counter(combined)
    has_ties = any(c > 1 for c in counts.values())

    if not has_ties and n1 * n2 <= _EXACT_LIMIT:
        p = _mannwhitney_exact_p(u1, n1, n2)
    else:
        big_n = n1 + n2
        tie_term = sum(c**3 - c for c in counts.values())
        sigma2 = n1 * n2 / 12 * ((big_n + 1) - tie_term / (big_n * (big_n - 1))) if big_n > 1 else 0.0
        sigma = math.sqrt(max(sigma2, 1e-12))
        mean_u = n1 * n2 / 2
        diff = u1 - mean_u
        continuity = 0.5 if diff > 0 else (-0.5 if diff < 0 else 0.0)
        z = (diff - continuity) / sigma
        p = math.erfc(abs(z) / math.sqrt(2))

    return u1, min(1.0, p)


def mannwhitney_effect_size(u1: float, n1: int, n2: int) -> float:
    """Ранговая бисериальная корреляция — размер эффекта для критерия Манна-Уитни,
    от -1 до 1. При очень малых n (типично для этой лаборатории, ~5 животных на
    группу) p-value крайне неустойчиво и лёгко спутать "незначимо" с "эффекта нет";
    размер эффекта не зависит от объёма выборки и остаётся интерпретируемым."""
    if n1 == 0 or n2 == 0:
        return 0.0
    return 1.0 - 2.0 * u1 / (n1 * n2)


def mannwhitney_min_possible_p(n1: int, n2: int) -> float:
    """Наименьшее двустороннее p-value, в принципе достижимое критерием Манна-Уитни
    при данных n1, n2 (полное разделение групп, U=0). При малых n (например, n1=n2=3
    даёт минимум 0.1) тест физически не может показать значимость при обычном
    α=0.05 — это нужно явно показывать пользователю, а не позволять читать
    "незначимо" как "эффекта нет"."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    return min(1.0, 2.0 / math.comb(n1 + n2, n1))


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Поправка Холма-Бонферрони на множественные сравнения (шаговый метод,
    контролирует вероятность хотя бы одной ложной находки по всей серии сравнений,
    при этом не слабее классического Бонферрони — то есть даёт не меньшую мощность).
    Возвращает скорректированные p-value в том же порядке, что и на входе.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        running_max = max(running_max, min(1.0, p_values[idx] * factor))
        adjusted[idx] = running_max
    return adjusted


# ---------- хи-квадрат через регуляризованную неполную гамма-функцию ----------
# стандартный алгоритм (Numerical Recipes): ряд при x < a+1, непрерывная дробь иначе

def _gamma_series(a: float, x: float, itmax: int = 200, eps: float = 3e-9) -> float:
    if x <= 0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(itmax):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float, itmax: int = 200, eps: float = 3e-9, fpmin: float = 1e-300) -> float:
    b = x + 1 - a
    c = 1 / fpmin
    d = 1 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, df: float) -> float:
    """P(X > x) для хи-квадрат с df степенями свободы."""
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1:
        return 1.0 - _gamma_series(a, xx)
    return _gamma_cf(a, xx)


def kruskal(samples: list[list[float]]) -> tuple[float, float]:
    """Критерий Краскела-Уоллиса. Возвращает (H, p-value)."""
    all_values: list[float] = []
    for s in samples:
        all_values.extend(s)
    ranks = _rankdata_average(all_values)

    big_n = len(all_values)
    h_stat = 0.0
    idx = 0
    for s in samples:
        n_i = len(s)
        r_i = sum(ranks[idx: idx + n_i])
        h_stat += (r_i**2) / n_i
        idx += n_i
    h_stat = 12.0 / (big_n * (big_n + 1)) * h_stat - 3 * (big_n + 1)

    counts = Counter(all_values)
    tie_term = sum(c**3 - c for c in counts.values())
    denom = big_n**3 - big_n
    correction = 1 - tie_term / denom if denom > 0 else 1.0
    if correction > 0:
        h_stat = h_stat / correction

    df = len(samples) - 1
    p = chi2_sf(h_stat, df)
    return h_stat, p
