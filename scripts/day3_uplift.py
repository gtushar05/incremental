"""Day 3: fit T/X-learners on Hillstrom, verify vs sklift, first uplift deciles."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xgboost import XGBClassifier, XGBRegressor
from sklift.models import TwoModels

from incremental.baseline import internal_split
from incremental.features import build_features
from incremental.uplift import TLearner, XLearner

TREATED, CONTROL = "Mens E-Mail", "No E-Mail"
REPORTS = ROOT / "reports"


def make_outcome_model():
    # shallow + regularized: uplift-by-differencing punishes overfit noise twice
    return XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        min_child_weight=25, subsample=0.9, random_state=0,
    )


def make_effect_model():
    return XGBRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        min_child_weight=25, subsample=0.9, random_state=0,
    )


train = pd.read_parquet(ROOT / "data" / "train.parquet")
fit_df, val_df = internal_split(train)

# two-arm subset: Mens E-Mail vs control
fit2 = fit_df[fit_df["segment"].isin([TREATED, CONTROL])]
val2 = val_df[val_df["segment"].isin([TREATED, CONTROL])]
Xf, Xv = build_features(fit2), build_features(val2)
Xv = Xv.reindex(columns=Xf.columns, fill_value=0.0)
yf, tf = fit2["visit"].values, (fit2["segment"] == TREATED).astype(int).values
tv = (val2["segment"] == TREATED).astype(int).values

rct_ate = val2[tv == 1]["visit"].mean() - val2[tv == 0]["visit"].mean()
print(f"fit {len(fit2):,} / val {len(val2):,}   RCT ATE on val = {rct_ate:+.4f}")

# ---- our from-scratch learners ----
t_model = TLearner(make_outcome_model).fit(Xf, yf, tf)
x_model = XLearner(make_outcome_model, make_effect_model).fit(Xf, yf, tf)
tau_t, tau_x = t_model.predict_uplift(Xv), x_model.predict_uplift(Xv)

# ---- verification 1: sklift TwoModels (same algorithm as T-learner) ----
sk = TwoModels(
    estimator_trmnt=make_outcome_model(),
    estimator_ctrl=make_outcome_model(),
    method="vanilla",
).fit(Xf, yf, tf)
tau_sk = sk.predict(Xv)
r_pearson = float(np.corrcoef(tau_t, tau_sk)[0, 1])
max_abs_diff = float(np.abs(tau_t - tau_sk).max())
print(f"\n=== verification vs sklift TwoModels ===")
print(f"  pearson r = {r_pearson:.6f}   max|diff| = {max_abs_diff:.2e}")

# ---- sanity: mean predicted uplift vs RCT ATE ----
print(f"\n=== ATE sanity (val) ===")
print(f"  RCT ground truth : {rct_ate:+.4f}")
print(f"  T-learner mean   : {tau_t.mean():+.4f}")
print(f"  X-learner mean   : {tau_x.mean():+.4f}")

# ---- first uplift-ranked deciles (true lift per predicted-uplift decile) ----
def uplift_deciles(tau: np.ndarray, name: str) -> pd.DataFrame:
    d = val2.copy()
    d["tau"], d["t"] = tau, tv
    d["decile"] = (10 - pd.qcut(d["tau"], 10, labels=False, duplicates="drop")).astype(int)
    rows = []
    for dec, g in d.groupby("decile"):
        lift = g[g.t == 1]["visit"].mean() - g[g.t == 0]["visit"].mean()
        rows.append({"decile": dec, "model": name,
                     "mean_tau": g["tau"].mean(), "true_lift": lift, "n": len(g)})
    return pd.DataFrame(rows)

dec_t, dec_x = uplift_deciles(tau_t, "T"), uplift_deciles(tau_x, "X")
print("\n=== true lift by X-learner predicted-uplift decile (1 = most persuadable) ===")
print(dec_x.round(4).to_string(index=False))

top3 = dec_x.nsmallest(3, "decile")["true_lift"].mean()
bot3 = dec_x.nlargest(3, "decile")["true_lift"].mean()
print(f"\n  top-3 deciles true lift {top3:+.4f} vs bottom-3 {bot3:+.4f} "
      f"(spread = {top3-bot3:.4f})")

REPORTS.mkdir(exist_ok=True)
json.dump(
    {"sklift_pearson_r": r_pearson, "sklift_max_abs_diff": max_abs_diff,
     "rct_ate_val": float(rct_ate),
     "mean_tau_t": float(tau_t.mean()), "mean_tau_x": float(tau_x.mean()),
     "deciles_t": dec_t.round(6).to_dict("records"),
     "deciles_x": dec_x.round(6).to_dict("records")},
    open(REPORTS / "day3_uplift.json", "w"), indent=2,
)
print(f"\nmetrics -> {REPORTS/'day3_uplift.json'}")
