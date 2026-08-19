"""Feature preparation shared by every model in the project.

One place, one encoding — so the baseline, the meta-learners, and the
serving layer can never drift apart.
"""

from __future__ import annotations

import pandas as pd

NUMERIC = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL = ["history_segment", "zip_code", "channel"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals, pass numerics through.

    Trees don't need scaling; one-hot keeps every model (and later SHAP)
    on an identical, stable column set.
    """
    X = pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=float)
    return X.reindex(sorted(X.columns), axis=1)


def align_features(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Align a scored frame to the training column set (missing levels -> 0)."""
    return X.reindex(columns=columns, fill_value=0.0)
