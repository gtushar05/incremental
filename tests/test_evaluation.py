import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incremental.evaluation import bootstrap_ci, calibration_by_decile
from incremental.metrics import uplift_at_k


def synthetic(n=30000, seed=9):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.uniform(size=n) < 0.5).astype(int)
    tau = 0.15 / (1 + np.exp(-2 * x))          # smooth effect in (0, 0.15)
    y = (rng.uniform(size=n) < 0.10 + t * tau).astype(int)
    return y, t, tau


def test_bootstrap_ci_brackets_point():
    y, t, tau = synthetic()
    res = bootstrap_ci(y, t, tau, lambda a, b, c: uplift_at_k(a, b, c, 0.2),
                       n_boot=100)
    assert res["lo"] <= res["point"] <= res["hi"]
    assert res["hi"] - res["lo"] < 0.10  # sane width at n=30k


def test_bootstrap_stratification_preserves_arm_ratio_effect():
    # CI of a known-quantity metric (ATE via k=1.0) should cover the true ATE
    y, t, tau = synthetic()
    true_ate = tau.mean()
    res = bootstrap_ci(y, t, np.ones_like(tau), lambda a, b, c: uplift_at_k(a, b, c, 1.0),
                       n_boot=100)
    assert res["lo"] - 0.01 <= true_ate <= res["hi"] + 0.01


def test_calibration_is_monotone_when_scores_are_true_effects():
    y, t, tau = synthetic()
    cal = calibration_by_decile(y, t, tau)
    # top decile's observed lift should exceed bottom decile's clearly
    top = cal.loc[cal.decile == 1, "observed_lift"].iloc[0]
    bot = cal.loc[cal.decile == 10, "observed_lift"].iloc[0]
    assert top > bot + 0.02
    # predicted and observed should correlate strongly across deciles
    r = cal["predicted_uplift"].corr(cal["observed_lift"])
    assert r > 0.7
