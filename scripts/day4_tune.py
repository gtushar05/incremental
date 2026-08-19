"""Day 4: cross-fitted hyperparameter selection for the uplift learners,
T-vs-X head-to-head, propensity-ranking comparison, and seed stability.

Selection metric: Qini coefficient on pooled out-of-fold predictions.
Standard CV-on-accuracy would tune the OUTCOME models; we need params that
are good at the second-order task (effect estimation), so the scorer must
be an uplift metric.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xgboost import XGBClassifier, XGBRegressor

from incremental.baseline import internal_split, train_response_model
from incremental.features import build_features
from incremental.metrics import qini_coefficient, uplift_at_k
from incremental.uplift import TLearner, XLearner

TREATED, CONTROL = "Mens E-Mail", "No E-Mail"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

train = pd.read_parquet(ROOT / "data" / "train.parquet")
fit_df, val_df = internal_split(train)
fit2 = fit_df[fit_df["segment"].isin([TREATED, CONTROL])].reset_index(drop=True)
val2 = val_df[val_df["segment"].isin([TREATED, CONTROL])].reset_index(drop=True)

Xf = build_features(fit2)
Xv = build_features(val2).reindex(columns=Xf.columns, fill_value=0.0)
yf = fit2["visit"].values
tf = (fit2["segment"] == TREATED).astype(int).values
yv = val2["visit"].values
tv = (val2["segment"] == TREATED).astype(int).values


def make_models(cfg, seed=0):
    clf = lambda: XGBClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=0.05, min_child_weight=cfg["min_child_weight"],
        subsample=0.9, random_state=seed,
    )
    reg = lambda: XGBRegressor(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=0.05, min_child_weight=cfg["min_child_weight"],
        subsample=0.9, random_state=seed,
    )
    return clf, reg


def oof_qini(learner_name: str, cfg: dict, n_folds=3) -> float:
    """Pooled out-of-fold Qini for one config. Stratify folds on t*2+y so
    every fold preserves arm ratios AND base rates."""
    strata = tf * 2 + yf
    tau_oof = np.zeros(len(fit2))
    for tr_idx, te_idx in StratifiedKFold(
        n_folds, shuffle=True, random_state=99
    ).split(Xf, strata):
        clf, reg = make_models(cfg)
        if learner_name == "T":
            m = TLearner(clf).fit(Xf.iloc[tr_idx], yf[tr_idx], tf[tr_idx])
        else:
            m = XLearner(clf, reg).fit(Xf.iloc[tr_idx], yf[tr_idx], tf[tr_idx])
        tau_oof[te_idx] = m.predict_uplift(Xf.iloc[te_idx])
    return qini_coefficient(yf, tf, tau_oof)


GRID = [
    dict(max_depth=d, min_child_weight=w, n_estimators=n)
    for d, w, n in itertools.product([2, 3], [25, 100, 300], [150, 300])
]

results = {"T": [], "X": []}
for name in ["T", "X"]:
    for cfg in GRID:
        q = oof_qini(name, cfg)
        results[name].append({**cfg, "oof_qini": q})
        print(f"  {name}-learner {cfg} -> OOF qini {q:.5f}")

best = {n: max(results[n], key=lambda r: r["oof_qini"]) for n in results}
print(f"\nbest T config: {best['T']}")
print(f"best X config: {best['X']}")

# ---- refit best configs on full fit split, evaluate on val ----
clf_t, _ = make_models(best["T"])
clf_x, reg_x = make_models(best["X"])
tau_t = TLearner(clf_t).fit(Xf, yf, tf).predict_uplift(Xv)
tau_x = XLearner(clf_x, reg_x).fit(Xf, yf, tf).predict_uplift(Xv)

# ---- propensity ranking (Day 2's model) scored with the SAME metric ----
resp_model, resp_cols = train_response_model(fit_df)
Xv_resp = build_features(val2).reindex(columns=resp_cols, fill_value=0.0)
score_resp = resp_model.predict_proba(Xv_resp)[:, 1]

rows = []
for name, s in [("propensity", score_resp), ("T-learner", tau_t), ("X-learner", tau_x)]:
    rows.append({
        "ranking": name,
        "qini": qini_coefficient(yv, tv, s),
        "uplift@10%": uplift_at_k(yv, tv, s, 0.1),
        "uplift@20%": uplift_at_k(yv, tv, s, 0.2),
        "uplift@30%": uplift_at_k(yv, tv, s, 0.3),
    })
comp = pd.DataFrame(rows)
print("\n=== val comparison (point estimates — CIs arrive Day 5) ===")
print(comp.round(5).to_string(index=False))

# ---- seed stability of the best X config ----
taus = []
for seed in [0, 1, 2]:
    clf_s, reg_s = make_models(best["X"], seed=seed)
    taus.append(XLearner(clf_s, reg_s).fit(Xf, yf, tf).predict_uplift(Xv))
cors = [
    pd.Series(a).corr(pd.Series(b), method="spearman")
    for a, b in itertools.combinations(taus, 2)
]
print(f"\nX-learner seed stability: mean pairwise spearman = {np.mean(cors):.4f}")

CONFIGS.mkdir(exist_ok=True)
json.dump(best, open(CONFIGS / "model_params.json", "w"), indent=2)
REPORTS.mkdir(exist_ok=True)
json.dump(
    {"grid_results": results, "best": best,
     "val_comparison": comp.round(6).to_dict("records"),
     "x_seed_stability_spearman": float(np.mean(cors))},
    open(REPORTS / "day4_tuning.json", "w"), indent=2,
)
print(f"\nparams -> {CONFIGS/'model_params.json'}   metrics -> {REPORTS/'day4_tuning.json'}")
