"""Day 5: the Tier-1 evaluation table — baseline ladder with bootstrap CIs,
Qini curves, uplift calibration, and the quadrant analysis. All on the
internal val split; the golden holdout stays sealed until Day 9."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xgboost import XGBClassifier, XGBRegressor

from incremental.baseline import internal_split, train_response_model
from incremental.evaluation import (
    bootstrap_ci,
    calibration_by_decile,
    quadrant_analysis,
)
from incremental.features import build_features
from incremental.metrics import qini_coefficient, qini_curve, uplift_at_k
from incremental.uplift import TLearner, XLearner

TREATED, CONTROL = "Mens E-Mail", "No E-Mail"
REPORTS = ROOT / "reports"
best = json.load(open(ROOT / "configs" / "model_params.json"))


def models_from(cfg, seed=0):
    clf = lambda: XGBClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=0.05, min_child_weight=cfg["min_child_weight"],
        subsample=0.9, random_state=seed)
    reg = lambda: XGBRegressor(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=0.05, min_child_weight=cfg["min_child_weight"],
        subsample=0.9, random_state=seed)
    return clf, reg


train = pd.read_parquet(ROOT / "data" / "train.parquet")
fit_df, val_df = internal_split(train)
fit2 = fit_df[fit_df["segment"].isin([TREATED, CONTROL])].reset_index(drop=True)
val2 = val_df[val_df["segment"].isin([TREATED, CONTROL])].reset_index(drop=True)
Xf = build_features(fit2)
Xv = build_features(val2).reindex(columns=Xf.columns, fill_value=0.0)
yf, tf = fit2["visit"].values, (fit2["segment"] == TREATED).astype(int).values
yv, tv = val2["visit"].values, (val2["segment"] == TREATED).astype(int).values

# ---- rankings ----
clf_t, _ = models_from(best["T"])
clf_x, reg_x = models_from(best["X"])
t_model = TLearner(clf_t).fit(Xf, yf, tf)
x_model = XLearner(clf_x, reg_x).fit(Xf, yf, tf)
tau_t, tau_x = t_model.predict_uplift(Xv), x_model.predict_uplift(Xv)

resp_model, resp_cols = train_response_model(fit_df)
score_resp = resp_model.predict_proba(
    build_features(val2).reindex(columns=resp_cols, fill_value=0.0))[:, 1]
rng = np.random.default_rng(0)
score_rand = rng.uniform(size=len(val2))

RANKINGS = {
    "random": score_rand,
    "propensity (response model)": score_resp,
    "T-learner (ours)": tau_t,
    "X-learner (ours)": tau_x,
}

# ---- Tier-1 ladder with bootstrap CIs (protocol: 200 resamples, cached) ----
rows = []
for name, s in RANKINGS.items():
    q = bootstrap_ci(yv, tv, s, qini_coefficient)
    u20 = bootstrap_ci(yv, tv, s, lambda a, b, c: uplift_at_k(a, b, c, 0.2))
    u30 = bootstrap_ci(yv, tv, s, lambda a, b, c: uplift_at_k(a, b, c, 0.3))
    rows.append({
        "ranking": name,
        "qini": q["point"], "qini_lo": q["lo"], "qini_hi": q["hi"],
        "uplift@20": u20["point"], "u20_lo": u20["lo"], "u20_hi": u20["hi"],
        "uplift@30": u30["point"], "u30_lo": u30["lo"], "u30_hi": u30["hi"],
    })
ladder = pd.DataFrame(rows)
print("=== Tier-1 ladder (val, 200-resample bootstrap 95% CIs) ===")
with pd.option_context("display.width", 160):
    print(ladder.round(5).to_string(index=False))

# ---- Qini curves ----
fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
palette = {"random": "#999999", "propensity (response model)": "#B23A48",
           "T-learner (ours)": "#3E6C8E", "X-learner (ours)": "#1F6347"}
for name, s in RANKINGS.items():
    phi, q = qini_curve(yv, tv, s)
    ax.plot(phi, q, label=name, color=palette[name],
            lw=2 if "ours" in name else 1.4,
            ls="--" if name == "random" else "-")
ax.set_xlabel("Fraction of population targeted (by ranking)")
ax.set_ylabel("Cumulative incremental visits (Qini)")
ax.set_title("Qini curves — who finds the persuadables first (val split)")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS / "qini_curves.png")

# ---- calibration by decile (X-learner) ----
cal = calibration_by_decile(yv, tv, tau_x)
print("\n=== X-learner calibration by decile ===")
print(cal.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)
ax.errorbar(cal["predicted_uplift"] * 100, cal["observed_lift"] * 100,
            yerr=cal["observed_se"] * 100 * 1.96, fmt="o", color="#1F6347",
            capsize=3, ms=6)
lim = [
    min(cal["predicted_uplift"].min(), cal["observed_lift"].min()) * 100 - 2,
    max(cal["predicted_uplift"].max(), cal["observed_lift"].max()) * 100 + 2,
]
ax.plot(lim, lim, ls="--", lw=1, color="#999")
ax.set_xlabel("Predicted uplift (pp)")
ax.set_ylabel("RCT-observed lift (pp, 95% CI)")
ax.set_title("Uplift calibration — predicted vs observed by decile (X-learner)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS / "calibration.png")

# calibration correlation (the decision-grade check)
cal_r = cal["predicted_uplift"].corr(cal["observed_lift"])
print(f"\ncalibration correlation (pred vs obs across deciles): {cal_r:.3f}")

# ---- quadrant analysis ----
p_control = t_model.model_control_.predict_proba(Xv)[:, 1]
quads = quadrant_analysis(yv, tv, tau_x, p_control)
print("\n=== persuadables quadrant analysis (X-learner scores) ===")
print(quads.round(4).to_string(index=False))

# ---- persist everything ----
json.dump(
    {"ladder": ladder.round(6).to_dict("records"),
     "calibration": cal.round(6).to_dict("records"),
     "calibration_corr": float(cal_r),
     "quadrants": quads.round(6).to_dict("records"),
     "protocol": {"n_boot": 200, "ci": [2.5, 97.5], "split": "internal val"}},
    open(REPORTS / "day5_evaluation.json", "w"), indent=2)
print(f"\nsaved -> {REPORTS/'day5_evaluation.json'}, qini_curves.png, calibration.png")
