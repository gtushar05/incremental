"""Uplift evaluation metrics, hand-rolled.

Why not AUC? Uplift has no per-row ground truth — you never observe both
potential outcomes for one person. Evaluation must therefore be done in
AGGREGATE, exploiting randomization: rank everyone by predicted uplift,
walk down the ranking, and at each targeting depth compare treated vs
control outcomes WITHIN the targeted prefix. That comparison is unbiased
only because assignment was random — the whole reason RCT data matters.

The Qini curve plots cumulative incremental outcomes against targeting
depth; its area above the random-targeting diagonal is the Qini
coefficient (Radcliffe). uplift@k is the same idea at a single depth.
"""

from __future__ import annotations

import numpy as np


def _sorted_cumulatives(y: np.ndarray, t: np.ndarray, score: np.ndarray):
    """Cumulative treated/control counts and outcomes, in descending-score order.

    Ties are broken by a fixed shuffle (seeded) so adversarially-ordered
    inputs can't flatter the curve.
    """
    rng = np.random.default_rng(0)
    jitter = rng.uniform(0, 1e-9, size=len(score))
    order = np.argsort(-(score + jitter), kind="stable")
    y, t = np.asarray(y, float)[order], np.asarray(t, float)[order]
    nt = np.cumsum(t)                # treated seen so far
    nc = np.cumsum(1 - t)            # control seen so far
    yt = np.cumsum(y * t)            # treated successes so far
    yc = np.cumsum(y * (1 - t))      # control successes so far
    return nt, nc, yt, yc


def qini_curve(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, n_points: int = 101
) -> tuple[np.ndarray, np.ndarray]:
    """Qini value at each targeting fraction phi in [0, 1].

    Q(phi) = Yt(phi) − Yc(phi) · Nt(phi)/Nc(phi)

    i.e. incremental successes among the targeted prefix, with the control
    group scaled to the treated group's size. Endpoint Q(1) is the total
    incremental effect of treating everyone.
    """
    nt, nc, yt, yc = _sorted_cumulatives(y, t, score)
    n = len(y)
    idx = np.unique(np.linspace(1, n, n_points).astype(int)) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        q = yt[idx] - yc[idx] * np.where(nc[idx] > 0, nt[idx] / nc[idx], 0.0)
    fractions = (idx + 1) / n
    return np.concatenate([[0.0], fractions]), np.concatenate([[0.0], q])


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Area between the model's Qini curve and the random-targeting diagonal,
    normalized per person (so values are comparable across dataset sizes).

    0 = no better than random targeting; higher = better ranking.
    """
    phi, q = qini_curve(y, t, score, n_points=201)
    q_total = q[-1]                      # incremental effect at 100% depth
    random_line = phi * q_total          # random targeting is a straight line
    area = np.trapezoid(q - random_line, phi)
    return float(area / len(y))


def uplift_at_k(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, k: float = 0.2
) -> float:
    """Observed uplift (treated rate − control rate) inside the top-k fraction
    of the score ranking. The single most intuitive uplift metric."""
    nt, nc, yt, yc = _sorted_cumulatives(y, t, score)
    i = max(int(k * len(y)) - 1, 0)
    rate_t = yt[i] / nt[i] if nt[i] > 0 else 0.0
    rate_c = yc[i] / nc[i] if nc[i] > 0 else 0.0
    return float(rate_t - rate_c)
