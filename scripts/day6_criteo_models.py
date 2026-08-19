"""Day 6b: first Criteo models — propensity, T, X on the visit label.

Trains on the documented 2M stratified sample; scores a disjoint 600K
validation slice from the train partition. Point estimates only today;
Day 7 adds conversion-label models, bootstrap CIs, and runs the
pre-registered label gate.
"""

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from xgboost import XGBClassifier, XGBRegressor

from incremental.metrics import qini_coefficient, uplift_at_k
from incremental.uplift import TLearner, XLearner

FEATS = [f"f{i}" for i in range(12)]
best = json.load(open(ROOT / "configs" / "model_params.json"))

# ---- data: sample for fit, disjoint stratified 600K slice for val ----
con = duckdb.connect()
fit = con.execute(f"SELECT * FROM read_parquet('{DATA}/criteo_sample_2m.parquet')").fetchdf()
val = con.execute(f"""
SELECT * FROM (
  SELECT t.*, row_number() OVER (
    PARTITION BY t.treatment, t.visit ORDER BY hash(t.rid + 777)
  ) AS rn, count(*) OVER (PARTITION BY t.treatment, t.visit) AS sn
  FROM read_parquet('{DATA}/criteo_train.parquet') t
  ANTI JOIN read_parquet('{DATA}/criteo_sample_2m.parquet') s USING (rid)
) WHERE rn <= sn * {600_000 / 7_786_000:.6f}
""").fetchdf()
print(f"fit: {len(fit):,}   val: {len(val):,} (disjoint, stratified)")

Xf, Xv = fit[FEATS], val[FEATS]
yf, tf = fit["visit"].values, fit["treatment"].values.astype(int)
yv, tv = val["visit"].values, val["treatment"].values.astype(int)
print(f"val ground truth: visit ATE {yv[tv==1].mean()-yv[tv==0].mean():+.5f}")


def make_clf(seed=0):
    cfg = best["X"]
    return XGBClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=0.05, min_child_weight=cfg["min_child_weight"],
        subsample=0.9, random_state=seed, n_jobs=-1, tree_method="hist")


def make_reg(seed=0):
    cfg = best["X"]
    return XGBRegressor(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=0.05, min_child_weight=cfg["min_child_weight"],
        subsample=0.9, random_state=seed, n_jobs=-1, tree_method="hist")


timings = {}

t0 = time.time()
resp = make_clf().fit(Xf[tf == 1], yf[tf == 1])          # response model on treated
score_resp = resp.predict_proba(Xv)[:, 1]
timings["propensity"] = time.time() - t0

t0 = time.time()
t_model = TLearner(make_clf).fit(Xf, yf, tf)
tau_t = t_model.predict_uplift(Xv)
timings["T-learner"] = time.time() - t0

t0 = time.time()
x_model = XLearner(make_clf, make_reg).fit(Xf, yf, tf)
tau_x = x_model.predict_uplift(Xv)
timings["X-learner"] = time.time() - t0

rng = np.random.default_rng(0)
rows = []
for name, s in [("random", rng.uniform(size=len(val))),
                ("propensity", score_resp),
                ("T-learner", tau_t), ("X-learner", tau_x)]:
    rows.append({
        "ranking": name,
        "qini": qini_coefficient(yv, tv, s),
        "uplift@10%": uplift_at_k(yv, tv, s, 0.1),
        "uplift@20%": uplift_at_k(yv, tv, s, 0.2),
        "uplift@30%": uplift_at_k(yv, tv, s, 0.3),
    })
comp = pd.DataFrame(rows)
print("\n=== Criteo (visit label) — val point estimates ===")
print(comp.round(5).to_string(index=False))
print(f"\nmean tau: T {tau_t.mean():+.5f}  X {tau_x.mean():+.5f}  "
      f"(RCT ATE {yv[tv==1].mean()-yv[tv==0].mean():+.5f})")
print("train timings:", {k: f"{v:.0f}s" for k, v in timings.items()})

json.dump(
    {"val_comparison": comp.round(6).to_dict("records"),
     "mean_tau_t": float(tau_t.mean()), "mean_tau_x": float(tau_x.mean()),
     "timings_s": {k: round(v, 1) for k, v in timings.items()}},
    open(ROOT / "reports" / "day6_criteo_models.json", "w"), indent=2)

# persist val scores for Day 7 (gate + CIs reuse them without retraining)
np.savez_compressed(DATA / "criteo_val_scores_visit.npz",
                    rid=val["rid"].values, y=yv, t=tv,
                    resp=score_resp, tau_t=tau_t, tau_x=tau_x)
print("\nsaved -> reports/day6_criteo_models.json, data/criteo_val_scores_visit.npz")
