"""Verification of the from-scratch meta-learners.

Strategy (causalml does not build on py3.14, so):
1. Synthetic RCT with KNOWN ground-truth tau(x) — the strongest possible test:
   both learners must recover the true effect ranking and the true ATE.
2. T-learner cross-checked against sklift's TwoModels in the Day-3 script
   (same algorithm — score correlation must be ~1.0).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incremental.uplift import TLearner, XLearner
from xgboost import XGBClassifier, XGBRegressor


def make_outcome_model():
    return XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=0
    )


def make_effect_model():
    return XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=0
    )


def synthetic_rct(n=20000, treated_frac=0.5, seed=7):
    """Known heterogeneous effect: tau depends on x0 and x1 only.

    base rate  = sigmoid(0.6*x1 - 1.2)          in ~[0.1, 0.4]
    tau(x)     = 0.15 * sigmoid(2*x0)            in ~[0, 0.15]
    y ~ Bernoulli(base + t * tau)
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=list("abcd"))
    t = (rng.uniform(size=n) < treated_frac).astype(int)
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    base = sigmoid(0.6 * X["b"] - 1.2)
    tau = 0.15 * sigmoid(2.0 * X["a"])
    p = np.clip(base + t * tau, 0, 1)
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y, t, tau.values


def test_tlearner_recovers_ate_and_ranking():
    X, y, t, tau_true = synthetic_rct()
    model = TLearner(make_outcome_model).fit(X, y, t)
    tau_hat = model.predict_uplift(X)
    # ATE recovered within a percentage point
    assert abs(tau_hat.mean() - tau_true.mean()) < 0.01
    # effect RANKING recovered (what targeting actually uses)
    rank_corr = pd.Series(tau_hat).corr(pd.Series(tau_true), method="spearman")
    assert rank_corr > 0.5


def test_xlearner_recovers_ate_and_ranking():
    X, y, t, tau_true = synthetic_rct()
    model = XLearner(make_outcome_model, make_effect_model).fit(X, y, t)
    tau_hat = model.predict_uplift(X)
    assert abs(tau_hat.mean() - tau_true.mean()) < 0.01
    rank_corr = pd.Series(tau_hat).corr(pd.Series(tau_true), method="spearman")
    assert rank_corr > 0.5


def test_xlearner_beats_tlearner_under_imbalance():
    """The X-learner's reason to exist: an 85/15 split like Criteo."""
    X, y, t, tau_true = synthetic_rct(n=30000, treated_frac=0.85, seed=11)
    tau_t = TLearner(make_outcome_model).fit(X, y, t).predict_uplift(X)
    tau_x = (
        XLearner(make_outcome_model, make_effect_model)
        .fit(X, y, t)
        .predict_uplift(X)
    )
    corr_t = pd.Series(tau_t).corr(pd.Series(tau_true), method="spearman")
    corr_x = pd.Series(tau_x).corr(pd.Series(tau_true), method="spearman")
    # X-learner should match or beat the T-learner when arms are imbalanced
    assert corr_x >= corr_t - 0.02


def test_null_effect_gives_near_zero_uplift():
    """No true effect anywhere -> predicted uplift should hug zero."""
    rng = np.random.default_rng(3)
    n = 20000
    X = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=list("abcd"))
    t = (rng.uniform(size=n) < 0.5).astype(int)
    p = 1 / (1 + np.exp(-(0.5 * X["a"] - 1.5)))  # depends on X, NOT on t
    y = (rng.uniform(size=n) < p).astype(int)
    tau_hat = TLearner(make_outcome_model).fit(X, y, t).predict_uplift(X)
    assert abs(tau_hat.mean()) < 0.01
