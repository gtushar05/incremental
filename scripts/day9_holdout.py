"""Day 9: THE single golden-holdout evaluation + the confirmatory
experiment design. Run once. The numbers printed under FROZEN are the
project's headline numbers, permanently.

Protocol (PREREGISTRATION.md):
- Hillstrom: models refit on the full train split (44,800) with the frozen
  configs; holdout (19,200, SHA256 4b1e135a...) evaluated once.
- Criteo: the documented final fit = frozen configs on the documented 2M
  stratified sample (deterministic seeds); full 4.19M holdout evaluated once.
- Metrics: the pre-registered ladder (random -> propensity -> T -> X) with
  Qini and uplift@k, 200-resample stratified bootstrap CIs, paired deltas
  for the headline claims, calibration, and the frozen policy's profit.
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

from incremental.baseline import train_response_model
from incremental.data import load_hillstrom, make_golden_holdout, frame_hash
from incremental.evaluation import (
    bootstrap_ci, bootstrap_delta_ci, calibration_by_decile)
from incremental.features import build_features
from incremental.metrics import qini_coefficient, uplift_at_k
from incremental.policy import observed_policy_value, profit_curve
from incremental.uplift import TLearner, XLearner

best = json.load(open(ROOT / "configs" / "model_params.json"))
OUT = {}
t_start = time.time()


def clf_factory(cfg, seed=0, hist=False):
    kw = dict(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
              learning_rate=0.05, min_child_weight=cfg["min_child_weight"],
              subsample=0.9, random_state=seed, n_jobs=-1)
    if hist:
        kw["tree_method"] = "hist"
    return lambda: XGBClassifier(**kw)


def reg_factory(cfg, seed=0, hist=False):
    kw = dict(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
              learning_rate=0.05, min_child_weight=cfg["min_child_weight"],
              subsample=0.9, random_state=seed, n_jobs=-1)
    if hist:
        kw["tree_method"] = "hist"
    return lambda: XGBRegressor(**kw)


# ================= PART A: confirmatory experiment design =================
print("=" * 70)
print("PART A - confirmatory A/B design (rollout of the frozen 8% policy)")
print("=" * 70)
vis_val = np.load(DATA / "criteo_val_scores_visit.npz")
order = np.lexsort((np.random.default_rng(0).permutation(len(vis_val["tau_x"])),
                    -vis_val["tau_x"]))
top8 = np.zeros(len(order), bool)
top8[order[: int(0.08 * len(order))]] = True
p_c = vis_val["y"][top8 & (vis_val["t"] == 0)].mean()
uplift_val = (vis_val["y"][top8 & (vis_val["t"] == 1)].mean() - p_c)
mde = 0.75 * uplift_val  # power for 75% of the val-estimated effect
z_a, z_b = 1.959964, 0.841621
p_bar = p_c + mde / 2
n_arm = 2 * (z_a + z_b) ** 2 * p_bar * (1 - p_bar) / mde ** 2
print(f"targeted population: top-8% by frozen X score")
print(f"control visit rate in target group (val): {p_c:.4f}")
print(f"val-estimated uplift in target group: {uplift_val:+.4f}")
print(f"design MDE (75% of estimate, conservative): {mde:.4f}")
print(f"required n per arm: {n_arm:,.0f}  (alpha=.05 two-sided, power=.80)")
print("guardrails: SRM chi-square on arm counts daily; full-week runtime;")
print("            uplift measured vs held-out control WITHIN the target group")
OUT["experiment_design"] = {
    "policy": "top 8% by frozen X-learner visit score",
    "control_rate_target_group": float(p_c),
    "val_uplift_target_group": float(uplift_val),
    "mde": float(mde), "n_per_arm": float(n_arm),
    "alpha": 0.05, "power": 0.80,
}

# ================= PART B: Hillstrom holdout (once) =================
print("\n" + "=" * 70)
print("PART B - HILLSTROM golden holdout: first and only evaluation")
print("=" * 70)
df = load_hillstrom()
train, holdout = make_golden_holdout(df)
h = frame_hash(holdout)
expected = "4b1e135a1a59b382d1bc566818c93acceaec1a61411e96b93590ee57a56897fc"
assert h == expected, "HOLDOUT HASH MISMATCH - abort"
print(f"holdout seal verified: {h[:16]}... == PREREGISTRATION ✓")

TREATED, CONTROL = "Mens E-Mail", "No E-Mail"
tr2 = train[train["segment"].isin([TREATED, CONTROL])].reset_index(drop=True)
ho2 = holdout[holdout["segment"].isin([TREATED, CONTROL])].reset_index(drop=True)
Xtr, Xho = build_features(tr2), build_features(ho2)
Xho = Xho.reindex(columns=Xtr.columns, fill_value=0.0)
ytr, ttr = tr2["visit"].values, (tr2["segment"] == TREATED).astype(int).values
yho, tho = ho2["visit"].values, (ho2["segment"] == TREATED).astype(int).values

tau_t_h = TLearner(clf_factory(best["T"])).fit(Xtr, ytr, ttr).predict_uplift(Xho)
tau_x_h = XLearner(clf_factory(best["X"]), reg_factory(best["X"])).fit(
    Xtr, ytr, ttr).predict_uplift(Xho)
resp_m, resp_cols = train_response_model(train)
resp_h = resp_m.predict_proba(
    build_features(ho2).reindex(columns=resp_cols, fill_value=0.0))[:, 1]
rng = np.random.default_rng(0)

hill_ladder = []
for name, s in [("random", rng.uniform(size=len(ho2))), ("propensity", resp_h),
                ("T-learner", tau_t_h), ("X-learner", tau_x_h)]:
    q = bootstrap_ci(yho, tho, s, qini_coefficient)
    u20 = bootstrap_ci(yho, tho, s, lambda a, b, c: uplift_at_k(a, b, c, 0.2))
    hill_ladder.append({"ranking": name, **{f"qini_{k}": v for k, v in q.items()},
                        **{f"u20_{k}": v for k, v in u20.items()}})
    print(f"  {name:<11} qini {q['point']:.5f} [{q['lo']:.5f},{q['hi']:.5f}]  "
          f"uplift@20 {u20['point']:.4f} [{u20['lo']:.4f},{u20['hi']:.4f}]")
OUT["hillstrom_ladder"] = hill_ladder

# ================= PART C: Criteo holdout (once) =================
print("\n" + "=" * 70)
print("PART C - CRITEO golden holdout (4.19M rows): first and only evaluation")
print("=" * 70)
FEATS = [f"f{i}" for i in range(12)]
con = duckdb.connect()
fp_stats = con.execute(f"""
SELECT count(*) n, sum(treatment) st, sum(visit) sv, sum(conversion) sc,
{", ".join(f"round(sum(f{i}), 3) sf{i}" for i in range(12))}
FROM read_parquet('{DATA}/criteo_holdout.parquet')""").fetchdf().to_dict("records")[0]
import hashlib
fp = hashlib.sha256(json.dumps({k: str(v) for k, v in fp_stats.items()},
                               sort_keys=True).encode()).hexdigest()
expected_fp = "eabff873fb5d5467630164e9cac7ddfdd8529d91178d6a0a9fa86817e7b2296b"
assert fp == expected_fp, "CRITEO FINGERPRINT MISMATCH - abort"
print(f"holdout seal verified: {fp[:16]}... == PREREGISTRATION ✓")

fit = con.execute(f"SELECT * FROM read_parquet('{DATA}/criteo_sample_2m.parquet')").fetchdf()
ho = con.execute(f"SELECT * FROM read_parquet('{DATA}/criteo_holdout.parquet')").fetchdf()
Xf, Xh = fit[FEATS], ho[FEATS]
tf = fit["treatment"].values.astype(int)
th = ho["treatment"].values.astype(int)

crit = {}
for label in ["visit", "conversion"]:
    yf, yh = fit[label].values, ho[label].values
    resp_model = clf_factory(best["X"], hist=True)()
    resp_model.fit(Xf[tf == 1], yf[tf == 1])
    s_resp = resp_model.predict_proba(Xh)[:, 1]
    s_t = TLearner(clf_factory(best["X"], hist=True)).fit(Xf, yf, tf).predict_uplift(Xh)
    s_x = XLearner(clf_factory(best["X"], hist=True),
                   reg_factory(best["X"], hist=True)).fit(Xf, yf, tf).predict_uplift(Xh)
    crit[label] = {"resp": s_resp, "t": s_t, "x": s_x, "y": yh}
    print(f"  {label}: models fit + 4.19M holdout scored")

# headline deltas (paired, pre-registered claims)
print("\nheadline paired deltas on the holdout:")
d1 = bootstrap_delta_ci(crit["visit"]["y"], th, crit["visit"]["x"],
                        crit["visit"]["resp"],
                        lambda a, b, c: uplift_at_k(a, b, c, 0.1))
print(f"  VISIT  X-prop uplift@10%: {d1['point']:+.5f} "
      f"[{d1['lo']:+.5f},{d1['hi']:+.5f}] separable={d1['separable']}")
d2 = bootstrap_delta_ci(crit["conversion"]["y"], th, crit["conversion"]["x"],
                        crit["conversion"]["resp"], qini_coefficient)
print(f"  CONV   X-prop qini:       {d2['point']:+.6f} "
      f"[{d2['lo']:+.6f},{d2['hi']:+.6f}] separable={d2['separable']}")
OUT["criteo_headline_deltas"] = {"visit_u10_X_minus_prop": d1,
                                 "conversion_qini_X_minus_prop": d2}

# ladder with CIs (visit = the budget-relevant label)
print("\nCriteo visit ladder (holdout, 200-boot CIs):")
crit_ladder = []
yv = crit["visit"]["y"]
for name, s in [("random", np.random.default_rng(0).uniform(size=len(ho))),
                ("propensity", crit["visit"]["resp"]),
                ("T-learner", crit["visit"]["t"]),
                ("X-learner", crit["visit"]["x"])]:
    q = bootstrap_ci(yv, th, s, qini_coefficient)
    u10 = bootstrap_ci(yv, th, s, lambda a, b, c: uplift_at_k(a, b, c, 0.1))
    crit_ladder.append({"ranking": name, **{f"qini_{k}": v for k, v in q.items()},
                        **{f"u10_{k}": v for k, v in u10.items()}})
    print(f"  {name:<11} qini {q['point']:.5f} [{q['lo']:.5f},{q['hi']:.5f}]  "
          f"uplift@10 {u10['point']:.4f} [{u10['lo']:.4f},{u10['hi']:.4f}]")
OUT["criteo_visit_ladder"] = crit_ladder

# calibration + the frozen policy's money, holdout-confirmed
cal = calibration_by_decile(yv, th, crit["visit"]["x"])
OUT["criteo_visit_calibration_corr"] = float(
    cal["predicted_uplift"].corr(cal["observed_lift"]))
print(f"\nX calibration corr on holdout: {OUT['criteo_visit_calibration_corr']:.3f}")

m, c = 25.0, 0.40
pc = profit_curve(yv, th, crit["visit"]["x"], m, c, budgets=np.array([0.08]))
pp = profit_curve(yv, th, crit["visit"]["resp"], m, c, budgets=np.array([0.08]))
ta = observed_policy_value(yv, th, np.ones(len(yv), bool), m, c)
print(f"frozen 8% policy on holdout: X profit Rs{pc.iloc[0]['profit']:,.0f} | "
      f"propensity Rs{pp.iloc[0]['profit']:,.0f} | treat-all Rs{ta['profit']:,.0f}")
OUT["criteo_policy_at_8pct"] = {
    "X_profit": float(pc.iloc[0]["profit"]),
    "prop_profit": float(pp.iloc[0]["profit"]),
    "treat_all_profit": float(ta["profit"]),
    "X_uplift_in_group": float(pc.iloc[0]["uplift_in_group"]),
    "economics": {"margin": m, "cost": c},
}

OUT["runtime_s"] = round(time.time() - t_start, 1)
json.dump(OUT, open(ROOT / "reports" / "day9_holdout_FROZEN.json", "w"),
          indent=2, default=float)
print(f"\n{'='*70}\nFROZEN. reports/day9_holdout_FROZEN.json written. "
      f"({OUT['runtime_s']}s)\nThese are the project's headline numbers, permanently.")
