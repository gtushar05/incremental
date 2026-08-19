"""The propensity (response-model) baseline — the industry default this
project exists to beat.

Pattern it replicates: train P(outcome) on the previously-contacted
population, rank everyone by that score, contact the top deciles.
The misallocation table shows what that actually buys, measurable here
only because the data is a genuine RCT (every decile contains both
treated and control users, so per-decile TRUE lift is observable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from .features import build_features

SEED = 43  # internal split seed — deliberately NOT the golden-holdout seed


def internal_split(
    train_df: pd.DataFrame, val_frac: float = 0.3, arm_col: str = "segment"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dev split inside train.parquet. The golden holdout is untouchable."""
    parts = []
    for _, grp in train_df.groupby([arm_col, "visit"]):
        parts.append(grp.sample(frac=val_frac, random_state=SEED))
    val = pd.concat(parts).sort_index()
    fit = train_df.drop(val.index).sort_index()
    return fit, val


def train_response_model(
    fit_df: pd.DataFrame,
    treated_arm: str = "Mens E-Mail",
    outcome: str = "visit",
    arm_col: str = "segment",
) -> tuple[XGBClassifier, list[str]]:
    """Fit P(outcome | x) on the treated arm only — exactly how response
    models are built in practice (you model last campaign's recipients)."""
    treated = fit_df[fit_df[arm_col] == treated_arm]
    X = build_features(treated)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        eval_metric="auc",
        random_state=SEED,
    )
    model.fit(X, treated[outcome])
    return model, list(X.columns)


def response_auc(
    model: XGBClassifier,
    columns: list[str],
    val_df: pd.DataFrame,
    treated_arm: str = "Mens E-Mail",
    outcome: str = "visit",
    arm_col: str = "segment",
) -> float:
    """The 'impressive' number that answers the wrong question."""
    treated = val_df[val_df[arm_col] == treated_arm]
    X = build_features(treated).reindex(columns=columns, fill_value=0.0)
    return float(roc_auc_score(treated[outcome], model.predict_proba(X)[:, 1]))


def misallocation_table(
    model: XGBClassifier,
    columns: list[str],
    val_df: pd.DataFrame,
    treated_arm: str = "Mens E-Mail",
    control_arm: str = "No E-Mail",
    outcome: str = "visit",
    arm_col: str = "segment",
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Score everyone, cut into response-score deciles (1 = model's favorites),
    then reveal each decile's TRUE lift from the experiment arms inside it."""
    df = val_df[val_df[arm_col].isin([treated_arm, control_arm])].copy()
    X = build_features(df).reindex(columns=columns, fill_value=0.0)
    df["score"] = model.predict_proba(X)[:, 1]
    # decile 1 = highest predicted response
    df["decile"] = (
        n_deciles
        - pd.qcut(df["score"], n_deciles, labels=False, duplicates="drop")
    ).astype(int)

    rows = []
    for d, grp in df.groupby("decile"):
        t = grp[grp[arm_col] == treated_arm]
        c = grp[grp[arm_col] == control_arm]
        uplift = t[outcome].mean() - c[outcome].mean()
        rows.append(
            {
                "decile": d,
                "n": len(grp),
                "mean_score": grp["score"].mean(),
                "treated_rate": t[outcome].mean(),
                "control_rate": c[outcome].mean(),
                "true_uplift": uplift,
                # incremental outcomes if this whole decile were contacted
                "incremental": uplift * len(grp),
            }
        )
    out = pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)
    return out


def misallocation_headline(table: pd.DataFrame, top_k: int = 2) -> dict:
    """The gap statement: incremental outcomes captured by contacting the
    response model's top-k deciles vs the best-possible k deciles (oracle)."""
    total = table["incremental"].sum()
    by_response = table.nsmallest(top_k, "decile")["incremental"].sum()
    oracle = table.nlargest(top_k, "true_uplift")["incremental"].sum()
    return {
        "top_k_deciles": top_k,
        "response_capture": float(by_response),
        "oracle_capture": float(oracle),
        "total_incremental": float(total),
        "response_share": float(by_response / total),
        "oracle_share": float(oracle / total),
        "efficiency_vs_oracle": float(by_response / oracle) if oracle else np.nan,
        "rank_corr_score_vs_uplift": float(
            table["decile"].corr(table["true_uplift"], method="spearman")
        ),
    }
