import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incremental.metrics import qini_coefficient, qini_curve, uplift_at_k


def synthetic(n=40000, seed=5):
    """RCT where the true effect is known per-row: tau high for x>0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.uniform(size=n) < 0.5).astype(int)
    tau = np.where(x > 0, 0.20, 0.0)         # persuadables = x>0
    base = 0.10
    y = (rng.uniform(size=n) < base + t * tau).astype(int)
    return x, y, t, tau


def test_oracle_ranking_beats_random():
    x, y, t, tau = synthetic()
    rng = np.random.default_rng(0)
    q_oracle = qini_coefficient(y, t, tau)          # rank by TRUE effect
    q_random = qini_coefficient(y, t, rng.uniform(size=len(y)))
    assert q_oracle > 0.005
    assert abs(q_random) < 0.003
    assert q_oracle > 5 * abs(q_random)


def test_inverted_ranking_is_negative():
    x, y, t, tau = synthetic()
    assert qini_coefficient(y, t, -tau) < -0.005


def test_uplift_at_k_recovers_known_segment_effect():
    x, y, t, tau = synthetic()
    # top 50% by true tau are exactly the x>0 users with true effect 0.20
    u = uplift_at_k(y, t, tau, k=0.5)
    assert 0.17 < u < 0.23


def test_qini_curve_endpoint_is_total_incremental():
    x, y, t, tau = synthetic()
    phi, q = qini_curve(y, t, tau)
    yt, nt = y[t == 1].sum(), (t == 1).sum()
    yc, nc = y[t == 0].sum(), (t == 0).sum()
    expected_total = yt - yc * (nt / nc)
    assert abs(q[-1] - expected_total) / abs(expected_total) < 0.01
