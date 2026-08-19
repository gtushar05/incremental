"""Randomization validation for RCT data: sample-ratio mismatch and covariate balance.

Run BEFORE any modeling. If randomization is broken, every causal claim
downstream is invalid — this module is the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# |SMD| below this is conventionally considered balanced in the causal literature
SMD_THRESHOLD = 0.10


def srm_check(assignments: pd.Series, expected: dict[str, float]) -> dict:
    """Sample-ratio mismatch: chi-square goodness-of-fit of observed arm counts
    against the design ratios (Hillstrom's design is an equal 1/3 split).

    A tiny p-value means the split deviates from design — a red flag that
    assignment (or logging) is broken.
    """
    counts = assignments.value_counts()
    arms = list(expected.keys())
    observed = np.array([counts.get(a, 0) for a in arms])
    total = observed.sum()
    expected_counts = np.array([expected[a] * total for a in arms])
    chi2, p = stats.chisquare(observed, expected_counts)
    return {
        "arms": arms,
        "observed": observed.tolist(),
        "expected": expected_counts.round(1).tolist(),
        "chi2": float(chi2),
        "p_value": float(p),
        "pass": bool(p > 0.001),  # generous: SRM alarms are for gross breakage
    }


def standardized_mean_diff(treat: pd.Series, control: pd.Series) -> float:
    """SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2).

    Scale-free, so one threshold works for every covariate; |SMD| < 0.1 ~ balanced.
    """
    mt, mc = treat.mean(), control.mean()
    vt, vc = treat.var(ddof=1), control.var(ddof=1)
    pooled = np.sqrt((vt + vc) / 2.0)
    if pooled == 0:
        return 0.0
    return float((mt - mc) / pooled)


def covariate_balance(
    df: pd.DataFrame,
    arm_col: str,
    control_label: str,
    numeric_covs: list[str],
    categorical_covs: list[str],
) -> pd.DataFrame:
    """SMD of every covariate for each treatment arm vs control.

    Categorical covariates are exploded into one-hot indicator columns first,
    so each level gets its own SMD (a proportion difference, standardized).
    """
    work = df.copy()
    indicator_cols: list[str] = []
    for cov in categorical_covs:
        dummies = pd.get_dummies(work[cov], prefix=cov).astype(float)
        indicator_cols.extend(dummies.columns)
        work = pd.concat([work, dummies], axis=1)

    all_covs = list(numeric_covs) + indicator_cols
    control = work[work[arm_col] == control_label]
    rows = []
    for arm in work[arm_col].unique():
        if arm == control_label:
            continue
        treated = work[work[arm_col] == arm]
        for cov in all_covs:
            smd = standardized_mean_diff(treated[cov], control[cov])
            rows.append(
                {
                    "arm": arm,
                    "covariate": cov,
                    "smd": round(smd, 4),
                    "balanced": abs(smd) < SMD_THRESHOLD,
                }
            )
    return pd.DataFrame(rows)


def rct_outcome_table(
    df: pd.DataFrame, arm_col: str, control_label: str, outcomes: list[str]
) -> pd.DataFrame:
    """Ground-truth per-arm outcome means and lifts vs control.

    On randomized data these differences ARE unbiased average treatment
    effects — the reference every model must be calibrated against.
    """
    grp = df.groupby(arm_col)[outcomes].mean()
    out = grp.copy()
    for o in outcomes:
        out[f"{o}_lift_vs_control"] = grp[o] - grp.loc[control_label, o]
    out["n"] = df.groupby(arm_col).size()
    return out
