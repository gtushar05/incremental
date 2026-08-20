import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incremental.policy import (
    marginal_profit_rule,
    observed_policy_value,
    profit_curve,
)


def synthetic(n=40000, seed=13):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.uniform(size=n) < 0.5).astype(int)
    tau = np.where(x > 0, 0.20, 0.0)
    y = (rng.uniform(size=n) < 0.10 + t * tau).astype(int)
    return y, t, tau


def test_full_budget_profit_matches_ate_formula():
    y, t, tau = synthetic()
    margin, cost = 100.0, 1.0
    v = observed_policy_value(y, t, np.ones(len(y), bool), margin, cost)
    ate = y[t == 1].mean() - y[t == 0].mean()
    expected = ate * len(y) * margin - len(y) * cost
    assert abs(v["profit"] - expected) < 1e-6


def test_marginal_rule_selects_above_threshold_only():
    y, t, tau = synthetic()
    res = marginal_profit_rule(tau, margin=100.0, cost=5.0)  # threshold 0.05
    assert res["threshold"] == 0.05
    # exactly the tau=0.20 half should be selected
    assert np.array_equal(res["selected"], tau > 0.05)
    assert 0.45 < res["share_selected"] < 0.55


def test_targeted_beats_treat_all_when_half_have_zero_effect():
    y, t, tau = synthetic()
    # economics chosen so treat-all is barely profitable but targeting wins
    margin, cost = 100.0, 5.0
    treat_all = observed_policy_value(y, t, np.ones(len(y), bool), margin, cost)
    curve = profit_curve(y, t, tau, margin, cost, budgets=np.array([0.5]))
    targeted = curve.iloc[0]["profit"]
    assert targeted > treat_all["profit"]


def test_empty_selection_is_zero():
    y, t, tau = synthetic()
    v = observed_policy_value(y, t, np.zeros(len(y), bool), 100.0, 1.0)
    assert v["profit"] == 0.0 and v["n_selected"] == 0


def test_profit_curve_immune_to_arm_clustered_ties():
    """Regression: all-tied scores + treated-first row order must yield the
    same profit as a fair arm mix — the tie/ordering bias found on Day 8."""
    rng = np.random.default_rng(21)
    n = 40000
    t = np.concatenate([np.ones(n // 2, int), np.zeros(n // 2, int)])  # clustered!
    base, tau_true = 0.10, 0.10
    y = (rng.uniform(size=n) < base + t * tau_true).astype(int)
    tied = np.ones(n)  # every score identical
    curve = profit_curve(y, t, tied, margin=100.0, cost=1.0,
                         budgets=np.array([0.5]))
    got = curve.iloc[0]["uplift_in_group"]
    # a fair tie-break must recover ~the true ATE inside the selection;
    # the buggy version selected treated-only rows and returned garbage
    assert abs(got - tau_true) < 0.02
