"""The policy layer: from uplift scores to contact decisions and money.

Two ideas, kept strictly separate:

1. EXPECTED profit uses the model's tau-hat — it's what you'd plan with:
       E[profit per contact] = tau_hat * margin - cost
   The unconstrained optimum is the MARGINAL-PROFIT RULE: contact exactly
   the people with tau_hat > cost / margin.

2. OBSERVED profit uses only the experiment's arm difference inside the
   selected group — the model is allowed to CHOOSE the group, but the
   money is scored by randomization ground truth, never by the model's
   own beliefs. This is what makes the profit curves honest.

Economics (margin per incremental outcome, cost per contact) are explicit
scenario parameters — stated, adjustable, never hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def observed_policy_value(
    y: np.ndarray,
    t: np.ndarray,
    selected: np.ndarray,
    margin: float,
    cost: float,
) -> dict:
    """Score an arbitrary selection mask with RCT ground truth.

    Incremental outcomes in the selected group = (rate_t - rate_c) * n_selected;
    profit = incremental_outcomes * margin - n_selected * cost.
    """
    y, t, selected = np.asarray(y), np.asarray(t), np.asarray(selected, bool)
    n_sel = int(selected.sum())
    if n_sel == 0:
        return {"n_selected": 0, "uplift_in_group": 0.0,
                "incremental_outcomes": 0.0, "profit": 0.0}
    ys, ts = y[selected], t[selected]
    rate_t = ys[ts == 1].mean() if (ts == 1).any() else 0.0
    rate_c = ys[ts == 0].mean() if (ts == 0).any() else 0.0
    uplift = rate_t - rate_c
    inc = uplift * n_sel
    return {
        "n_selected": n_sel,
        "uplift_in_group": float(uplift),
        "incremental_outcomes": float(inc),
        "profit": float(inc * margin - n_sel * cost),
    }


def profit_curve(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    margin: float,
    cost: float,
    budgets: np.ndarray | None = None,
) -> pd.DataFrame:
    """Observed incremental profit of contacting the top-b fraction by score,
    for each budget b. The decision chart: where does profit peak?"""
    if budgets is None:
        budgets = np.arange(0.02, 0.52, 0.02)
    # Tie-breaking matters: stable argsort on tied blocks inherits the input
    # row order, and if that order is arm-clustered the budget cut inside a
    # tied block selects an arm-skewed, unrepresentative subset — silently
    # biasing uplift_in_group (found via an impossible bootstrap CI). Break
    # ties in RANK space with a seeded permutation: exact, and immune to the
    # float-absorption edge that additive jitter hits on constant scores.
    s = np.asarray(score, dtype=float)
    rng = np.random.default_rng(0)
    order = np.lexsort((rng.permutation(len(s)), -s))
    rows = []
    for b in budgets:
        k = int(round(b * len(y)))
        sel = np.zeros(len(y), bool)
        sel[order[:k]] = True
        v = observed_policy_value(y, t, sel, margin, cost)
        rows.append({"budget": float(b), **v,
                     "profit_per_contact": v["profit"] / max(v["n_selected"], 1)})
    return pd.DataFrame(rows)


def marginal_profit_rule(
    tau_hat: np.ndarray, margin: float, cost: float
) -> dict:
    """The textbook optimum without a budget: contact iff expected value of
    the contact exceeds its cost, i.e. tau_hat > cost / margin."""
    threshold = cost / margin
    selected = np.asarray(tau_hat) > threshold
    expected_profit = float(
        (np.asarray(tau_hat)[selected] * margin - cost).sum()
    )
    return {
        "threshold": float(threshold),
        "n_selected": int(selected.sum()),
        "share_selected": float(selected.mean()),
        "expected_profit_model": expected_profit,
        "selected": selected,
    }
