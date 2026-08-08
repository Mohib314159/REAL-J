"""Uncertainty, clustered by case. Not by rollout.

This is the part that is easy to get wrong in a way that makes a pilot look
far more conclusive than it is.

Twenty stochastic rollouts of ``case_03`` at R3 are twenty samples of one
task. They tell you about the variance of that task. They do not tell you that
anything generalises across tasks, and treating them as twenty independent
observations shrinks the confidence interval by a factor of roughly sqrt(20)
for free. With six held-out cases, the honest n is 6.

So: compute the per-case statistic first, then bootstrap over *cases*. The
cluster bootstrap resamples cases with replacement and recomputes the
statistic from whichever rollouts belong to the drawn cases. When the
per-case counts are unequal it is still the right unit; unequal cluster sizes
widen the interval rather than biasing it.

For a pilot this is the whole analysis. Mixed models with random intercepts
for case become reasonable at paper scale, and deliberately are not used here
-- fitting a random-effects model to six clusters is a way of getting a
p-value, not a way of learning something.

Pure Python and the standard library. No scipy dependency for a bootstrap.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

Row = Mapping[str, object]


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float
    n_clusters: int
    n_rows: int
    level: float = 0.95

    def __str__(self) -> str:
        return (
            f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] "
            f"(n={self.n_clusters} cases, {self.n_rows} rollouts)"
        )

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "level": self.level,
            "n_clusters": self.n_clusters,
            "n_rows": self.n_rows,
        }


def group_by_case(rows: Sequence[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        out[str(row.get("case_id", ""))].append(row)
    return dict(out)


def _numeric(value) -> float | None:
    if value is None or isinstance(value, str):
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def case_means(rows: Sequence[Row], field: str) -> dict[str, float]:
    """Per-case mean of one field, skipping rows where it is missing."""
    out: dict[str, float] = {}
    for case_id, case_rows in group_by_case(rows).items():
        values = [v for v in (_numeric(r.get(field)) for r in case_rows) if v is not None]
        if values:
            out[case_id] = statistics.fmean(values)
    return out


def cluster_bootstrap(
    rows: Sequence[Row],
    statistic: Callable[[Sequence[Row]], float | None],
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate | None:
    """Bootstrap ``statistic`` by resampling cases with replacement."""
    by_case = group_by_case(rows)
    case_ids = sorted(by_case)
    if len(case_ids) < 2:
        return None

    point = statistic(list(rows))
    if point is None:
        return None

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        drawn = [rng.choice(case_ids) for _ in case_ids]
        resampled: list[Row] = []
        for case_id in drawn:
            resampled.extend(by_case[case_id])
        value = statistic(resampled)
        if value is not None:
            draws.append(value)

    if len(draws) < n_boot // 10:
        return None

    draws.sort()
    alpha = (1.0 - level) / 2.0
    lo = draws[max(0, int(alpha * len(draws)) - 1)]
    hi = draws[min(len(draws) - 1, int((1.0 - alpha) * len(draws)))]
    return Estimate(point, lo, hi, len(case_ids), len(rows), level)


def mean_of_case_means(field: str) -> Callable[[Sequence[Row]], float | None]:
    """Statistic: average the per-case means, giving every case equal weight.

    Equal weight per case, not per rollout. A case that happened to run more
    epochs should not count for more.
    """

    def statistic(rows: Sequence[Row]) -> float | None:
        means = case_means(rows, field)
        return statistics.fmean(means.values()) if means else None

    return statistic


def rate(field: str) -> Callable[[Sequence[Row]], float | None]:
    return mean_of_case_means(field)


def by_rung(
    rows: Sequence[Row],
    field: str,
    arm: str | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[int, Estimate]:
    """Per-rung estimate with case-clustered intervals."""
    selected = [r for r in rows if arm is None or r.get("arm") == arm]
    out: dict[int, Estimate] = {}
    for rung in sorted({int(r.get("rung", 0)) for r in selected}):
        subset = [r for r in selected if int(r.get("rung", 0)) == rung]
        estimate = cluster_bootstrap(
            subset, mean_of_case_means(field), n_boot=n_boot, seed=seed
        )
        if estimate:
            out[rung] = estimate
    return out


def paired_difference(
    rows: Sequence[Row],
    field: str,
    rung_a: int,
    rung_b: int,
    arm: str | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Estimate | None:
    """Difference between two rungs, paired within case.

    Pairing within case is the right comparison: the same eight bugs appear at
    every rung, so between-case variance -- which is large and uninteresting --
    can be differenced out.
    """
    selected = [r for r in rows if arm is None or r.get("arm") == arm]

    def statistic(subset: Sequence[Row]) -> float | None:
        a = case_means([r for r in subset if int(r.get("rung", 0)) == rung_a], field)
        b = case_means([r for r in subset if int(r.get("rung", 0)) == rung_b], field)
        shared = sorted(set(a) & set(b))
        if not shared:
            return None
        return statistics.fmean([b[c] - a[c] for c in shared])

    return cluster_bootstrap(selected, statistic, n_boot=n_boot, seed=seed)


def non_monotone_contrast(
    rows: Sequence[Row], field: str, arm: str | None = None, n_boot: int = 10_000, seed: int = 0
) -> Estimate | None:
    """The inverted-U contrast: middle minus the mean of the two ends.

    Positive means the middle rung sits above both ends -- the shape Newman's
    mechanism predicts. Stated as a single preregistered contrast rather than
    an order-restricted model, because with three rungs and six cases a
    contrast is what the data can actually support. A constrained isotonic
    model on n=6 clusters is machinery, not evidence.
    """

    def statistic(subset: Sequence[Row]) -> float | None:
        means = {
            rung: case_means(
                [r for r in subset if int(r.get("rung", 0)) == rung], field
            )
            for rung in (1, 3, 5)
        }
        shared = sorted(set(means[1]) & set(means[3]) & set(means[5]))
        if not shared:
            return None
        return statistics.fmean(
            [means[3][c] - (means[1][c] + means[5][c]) / 2.0 for c in shared]
        )

    selected = [r for r in rows if arm is None or r.get("arm") == arm]
    return cluster_bootstrap(selected, statistic, n_boot=n_boot, seed=seed)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation, for realism win rate against rung index."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return None if den == 0 else num / den


def outcome_table(rows: Sequence[Row]) -> dict[str, dict[str, int]]:
    """Counts of the H/T/V cells, split by arm."""
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        table[str(row.get("arm", "?"))][str(row.get("outcome_cell", "?"))] += 1
    return {arm: dict(cells) for arm, cells in table.items()}
