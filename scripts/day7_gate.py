"""Day 7: conversion-label models + the pre-registered signal-viability gate.

Gate rule (PREREGISTRATION.md, rule #2, committed before any Criteo run):
if conversion-label Qini deltas between models are not separable at
bootstrap 95% CIs, the visit label (4.7%) becomes the primary Criteo
metric and conversion is reported as secondary.

Implementation: PAIRED bootstrap deltas (same resamples on both rankings) —
the strict version of "separable".
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

from incremental.evaluation import bootstrap_ci, bootstrap_delta_ci, calibration_by_decile
from incremental.metrics import qini_coefficient, uplift_at_k
from incremental.uplift import TLearner, XLearner

FEATS = [f"f{i}" for i in range(12)]
best = json.load(open(ROOT / "configs" / "model_params.json"))["X"]


def make_clf(seed=0):
    return XGBClassifier(
        n_estimators=best["n_estimators"], max_depth=best["max_depth"],
        learning_rate=0.05, min_child_weight=best["min_child_weight"],
        subsample=0.9, random_state=seed, n_jobs=-1, tree_method="hist")


def make_reg(seed=0):
    return XGBRegressor(
        n_estimators=best["n_estimators"], max_depth=best["max_depth"],
        learning_rate=0.05, min_child_weight=best["min_child_weight"],
        subsample=0.9, random_state=seed, n_jobs=-1, tree_method="hist")


# ---- load fit sample + the SAME val slice as Day 6 (via saved rids) ----
con = duckdb.connect()
fit = con.execute(f"SELECT * FROM read_parquet('{DATA}/criteo_sample_2m.parquet')").fetchdf()
saved = np.load(DATA / "criteo_val_scores_visit.npz")
val_rids = saved["rid"]
con.register("val_rids", pd.DataFrame({"rid": val_rids}))
val = con.execute(f"""
SELECT t.* FROM read_parquet('{DATA}/criteo_train.parquet') t
SEMI JOIN val_rids USING (rid)
""").fetchdf().set_index("rid").loc[val_rids].reset_index()

Xf, Xv = fit[FEATS], val[FEATS]
tf = fit["treatment"].values.astype(int)
tv = val["treatment"].values.astype(int)
assert np.array_equal(tv, saved["t"]), "val slice mismatch vs Day 6"

# ---- conversion-label models ----
yf_c, yv_c = fit["conversion"].values, val["conversion"].values
print(f"conversions in fit sample: {int(yf_c.sum()):,} "
      f"(control arm: {int(yf_c[tf==0].sum()):,}) — the thin-signal regime")

t0 = time.time()
resp_c = make_clf().fit(Xf[tf == 1], yf_c[tf == 1])
score_resp_c = resp_c.predict_proba(Xv)[:, 1]
tau_t_c = TLearner(make_clf).fit(Xf, yf_c, tf).predict_uplift(Xv)
tau_x_c = XLearner(make_clf, make_reg).fit(Xf, yf_c, tf).predict_uplift(Xv)
print(f"conversion models trained in {time.time()-t0:.0f}s")

# visit-label scores from Day 6 (no retraining)
yv_v = saved["y"]
score_resp_v, tau_t_v, tau_x_v = saved["resp"], saved["tau_t"], saved["tau_x"]

# ---- THE GATE: paired deltas on conversion Qini ----
print("\n=== GATE: conversion-label Qini deltas (paired bootstrap, 200 resamples) ===")
gate_pairs = [
    ("X-learner - propensity", tau_x_c, score_resp_c),
    ("T-learner - propensity", tau_t_c, score_resp_c),
    ("X-learner - T-learner", tau_x_c, tau_t_c),
]
gate_results = {}
for name, a, b in gate_pairs:
    r = bootstrap_delta_ci(yv_c, tv, a, b, qini_coefficient)
    gate_results[name] = r
    print(f"  {name}: delta {r['point']:+.6f}  CI [{r['lo']:+.6f}, {r['hi']:+.6f}]"
          f"  separable={r['separable']}")

any_separable = any(r["separable"] for r in gate_results.values())
primary = "conversion" if any_separable else "visit"
print(f"\nGATE DECISION (pre-registered rule #2): PRIMARY CRITEO LABEL = {primary.upper()}")

# ---- the headline candidate: visit-label deltas, paired ----
print("\n=== visit-label deltas (paired bootstrap) ===")
visit_pairs = [
    ("X uplift@10% - propensity uplift@10%", tau_x_v, score_resp_v,
     lambda a, b, c: uplift_at_k(a, b, c, 0.1)),
    ("T uplift@10% - propensity uplift@10%", tau_t_v, score_resp_v,
     lambda a, b, c: uplift_at_k(a, b, c, 0.1)),
    ("X qini - propensity qini", tau_x_v, score_resp_v, qini_coefficient),
]
visit_results = {}
for name, a, b, fn in visit_pairs:
    r = bootstrap_delta_ci(yv_v, tv, a, b, fn)
    visit_results[name] = r
    print(f"  {name}: delta {r['point']:+.5f}  CI [{r['lo']:+.5f}, {r['hi']:+.5f}]"
          f"  separable={r['separable']}")

# ---- absolute CIs for the primary-label ladder (README table) ----
print("\n=== primary-label (visit) ladder with absolute CIs ===")
ladder = []
rng = np.random.default_rng(0)
for name, s in [("random", rng.uniform(size=len(val))),
                ("propensity", score_resp_v),
                ("T-learner", tau_t_v), ("X-learner", tau_x_v)]:
    q = bootstrap_ci(yv_v, tv, s, qini_coefficient)
    u10 = bootstrap_ci(yv_v, tv, s, lambda a, b, c: uplift_at_k(a, b, c, 0.1))
    ladder.append({"ranking": name,
                   "qini": q["point"], "qini_lo": q["lo"], "qini_hi": q["hi"],
                   "u10": u10["point"], "u10_lo": u10["lo"], "u10_hi": u10["hi"]})
    print(f"  {name:<12} qini {q['point']:.5f} [{q['lo']:.5f},{q['hi']:.5f}]   "
          f"uplift@10 {u10['point']:.5f} [{u10['lo']:.5f},{u10['hi']:.5f}]")

# ---- calibration on the primary label (X-learner) ----
cal = calibration_by_decile(yv_v, tv, tau_x_v)
cal_r = cal["predicted_uplift"].corr(cal["observed_lift"])
print(f"\nX-learner visit calibration corr across deciles: {cal_r:.3f}")
print(cal.round(5).to_string(index=False))

json.dump(
    {"gate": {k: v for k, v in gate_results.items()},
     "gate_decision_primary_label": primary,
     "visit_deltas": {k: v for k, v in visit_results.items()},
     "visit_ladder": ladder,
     "visit_calibration": cal.round(6).to_dict("records"),
     "visit_calibration_corr": float(cal_r),
     "conversion_point_estimates": {
         "propensity_qini": qini_coefficient(yv_c, tv, score_resp_c),
         "T_qini": qini_coefficient(yv_c, tv, tau_t_c),
         "X_qini": qini_coefficient(yv_c, tv, tau_x_c)}},
    open(ROOT / "reports" / "day7_gate.json", "w"), indent=2)

np.savez_compressed(DATA / "criteo_val_scores_conversion.npz",
                    rid=val_rids, y=yv_c, t=tv,
                    resp=score_resp_c, tau_t=tau_t_c, tau_x=tau_x_c)
print(f"\nsaved -> reports/day7_gate.json, data/criteo_val_scores_conversion.npz")
