"""Evaluation machinery on top of metrics.py: bootstrap uncertainty,
uplift calibration, and the persuadables quadrant analysis.

Protocol (pre-registered): fixed 200 bootstrap resamples, stratified by arm
so every resample preserves the randomization ratio; percentile 95% CIs;
results cached by the caller — never re-bootstrapped ad hoc.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

N_BOOT = 200
CI = (2.5, 97.5)


def bootstrap_ci(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_boot: int = N_BOOT,
    seed: int = 7,
) -> dict:
    """Percentile bootstrap CI for any (y, t, score) metric.

    Resamples WITHIN each arm (stratified) so the treated/control ratio —
    part of the experimental design, not sampling noise — stays fixed.
    """
    y, t, score = map(np.asarray, (y, t, score))
    rng = np.random.default_rng(seed)
    idx_t, idx_c = np.where(t == 1)[0], np.where(t == 0)[0]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        bi = np.concatenate([
            rng.choice(idx_t, len(idx_t), replace=True),
            rng.choice(idx_c, len(idx_c), replace=True),
        ])
        stats[b] = metric_fn(y[bi], t[bi], score[bi])
    lo, hi = np.percentile(stats, CI)
    return {
        "point": float(metric_fn(y, t, score)),
        "lo": float(lo),
        "hi": float(hi),
        "boot_std": float(stats.std(ddof=1)),
    }


def calibration_by_decile(
    y: np.ndarray, t: np.ndarray, tau_hat: np.ndarray, n_deciles: int = 10
) -> pd.DataFrame:
    """Predicted uplift vs RCT-observed lift, by predicted-uplift decile.

    Observable only because assignment is random: within any score decile,
    treated vs control rates are an unbiased estimate of that decile's true
    average effect. Points hugging the 45-degree line = scores are honest
    magnitudes, not just a ranking.
    """
    df = pd.DataFrame({"y": y, "t": t, "tau": tau_hat})
    # break exact score ties (common with shallow trees on few features) so
    # bins stay equal-population; jitter is ~1e-10 of the score scale
    rng = np.random.default_rng(0)
    scale = max(float(np.std(tau_hat)), 1e-12)
    jittered = df["tau"] + rng.uniform(0, scale * 1e-9, size=len(df))
    df["decile"] = (
        n_deciles - pd.qcut(jittered, n_deciles, labels=False, duplicates="drop")
    ).astype(int)
    rows = []
    for d, g in df.groupby("decile"):
        gt, gc = g[g.t == 1], g[g.t == 0]
        obs = gt.y.mean() - gc.y.mean()
        # SE of a difference in proportions — honest error bars per decile
        se = np.sqrt(
            gt.y.mean() * (1 - gt.y.mean()) / max(len(gt), 1)
            + gc.y.mean() * (1 - gc.y.mean()) / max(len(gc), 1)
        )
        rows.append(
            {
                "decile": d,
                "n": len(g),
                "predicted_uplift": g.tau.mean(),
                "observed_lift": obs,
                "observed_se": se,
            }
        )
    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)


def quadrant_analysis(
    y: np.ndarray,
    t: np.ndarray,
    tau_hat: np.ndarray,
    p_control_hat: np.ndarray,
) -> pd.DataFrame:
    """The persuadables 2x2, made empirical.

    Segmentation rules (stated, defensible, simple):
      persuadable     tau_hat in the top quartile
      do-not-disturb  tau_hat < 0
      sure thing      baseline P(y|control) above median, tau_hat mid/low
      lost cause      everyone else
    Each segment then shows its RCT-OBSERVED lift — the validation that the
    labels mean what they claim.
    """
    df = pd.DataFrame(
        {"y": y, "t": t, "tau": tau_hat, "p_c": p_control_hat}
    )
    tau_hi = df["tau"].quantile(0.75)
    p_hi = df["p_c"].median()

    def label(r):
        if r.tau < 0:
            return "do-not-disturb"
        if r.tau >= tau_hi:
            return "persuadable"
        if r.p_c >= p_hi:
            return "sure thing"
        return "lost cause"

    df["segment"] = df.apply(label, axis=1)
    rows = []
    for seg, g in df.groupby("segment"):
        gt, gc = g[g.t == 1], g[g.t == 0]
        rows.append(
            {
                "segment": seg,
                "n": len(g),
                "share": len(g) / len(df),
                "mean_predicted_tau": g.tau.mean(),
                "observed_lift": gt.y.mean() - gc.y.mean(),
                "control_rate": gc.y.mean(),
            }
        )
    order = ["persuadable", "sure thing", "lost cause", "do-not-disturb"]
    out = pd.DataFrame(rows)
    out["segment"] = pd.Categorical(out["segment"], order, ordered=True)
    return out.sort_values("segment").reset_index(drop=True)


def bootstrap_delta_ci(
    y: np.ndarray,
    t: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_boot: int = N_BOOT,
    seed: int = 7,
) -> dict:
    """Paired bootstrap CI for metric(a) − metric(b).

    The SAME stratified resample is applied to both rankings, so shared
    sampling noise cancels and the CI reflects the difference itself —
    the statistically honest way to ask "is ranking A better than B?"
    (Overlapping individual CIs cannot answer that.)
    """
    y, t = np.asarray(y), np.asarray(t)
    score_a, score_b = np.asarray(score_a), np.asarray(score_b)
    rng = np.random.default_rng(seed)
    idx_t, idx_c = np.where(t == 1)[0], np.where(t == 0)[0]
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        bi = np.concatenate([
            rng.choice(idx_t, len(idx_t), replace=True),
            rng.choice(idx_c, len(idx_c), replace=True),
        ])
        deltas[b] = metric_fn(y[bi], t[bi], score_a[bi]) - metric_fn(
            y[bi], t[bi], score_b[bi]
        )
    lo, hi = np.percentile(deltas, CI)
    point = float(
        metric_fn(y, t, score_a) - metric_fn(y, t, score_b)
    )
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "separable": bool(lo > 0 or hi < 0),
    }
