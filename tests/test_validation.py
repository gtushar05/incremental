import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incremental.validation import (
    srm_check,
    standardized_mean_diff,
    covariate_balance,
)


def test_smd_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(5, 2, 5000))
    assert abs(standardized_mean_diff(x, x.copy())) < 1e-12


def test_smd_detects_shift():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(0, 1, 5000))
    treated = pd.Series(rng.normal(0.5, 1, 5000))  # true SMD = 0.5
    smd = standardized_mean_diff(treated, control)
    assert 0.4 < smd < 0.6


def test_srm_passes_on_designed_split():
    assignments = pd.Series(["A"] * 3340 + ["B"] * 3330 + ["C"] * 3330)
    result = srm_check(assignments, {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    assert result["pass"]


def test_srm_fails_on_broken_split():
    assignments = pd.Series(["A"] * 5000 + ["B"] * 3000 + ["C"] * 2000)
    result = srm_check(assignments, {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    assert not result["pass"]


def test_covariate_balance_shapes():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "arm": rng.choice(["T", "C"], 2000),
            "age": rng.normal(30, 5, 2000),
            "city": rng.choice(["X", "Y"], 2000),
        }
    )
    bal = covariate_balance(df, "arm", "C", ["age"], ["city"])
    # 1 numeric + 2 one-hot levels, single treatment arm
    assert len(bal) == 3
    assert bal["balanced"].all()  # random assignment => balanced
