"""Data loading and the golden-holdout split.

The holdout is created ONCE, hashed, and pre-registered (PREREGISTRATION.md).
It is evaluated exactly once, on Day 9. Nothing else may touch it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

HOLDOUT_FRAC = 0.30
SEED = 42


def load_hillstrom(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "hillstrom.csv"
    df = pd.read_csv(path)
    # normalize column names once, at the boundary
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def make_golden_holdout(
    df: pd.DataFrame, arm_col: str = "segment", strat_outcome: str = "visit"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on (arm x outcome) so both frames preserve the
    randomization ratios and base rates. Deterministic under SEED."""
    holdout_parts = []
    for _, grp in df.groupby([arm_col, strat_outcome]):
        holdout_parts.append(grp.sample(frac=HOLDOUT_FRAC, random_state=SEED))
    holdout = pd.concat(holdout_parts).sort_index()
    train = df.drop(holdout.index).sort_index()
    return train, holdout


def frame_hash(df: pd.DataFrame) -> str:
    """Canonical SHA256: sorted by index, CSV bytes, no float wobble surprises
    (Hillstrom's columns are ints/strings/2dp floats — CSV round-trip is exact).
    """
    canonical = df.sort_index().to_csv(index=True).encode()
    return hashlib.sha256(canonical).hexdigest()
