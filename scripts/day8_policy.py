"""Day 8: the policy layer — observed profit curves, the marginal-profit
rule, and the money chart.

Scenario economics (illustrative display-advertising numbers, stated and
adjustable — the shape of the argument survives any reasonable values):
  visit scenario:      cost/contact = Rs 0.40, value/incremental visit = Rs 25
  conversion scenario: cost/contact = Rs 0.40, value/incremental conv  = Rs 4,000

Key property of the visit scenario: treat-all is UNPROFITABLE
(ATE 1.03pp x Rs25 = Rs0.26 < Rs0.40 cost) — targeting isn't an
optimization here, it's the difference between making and losing money.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incremental.evaluation import bootstrap_delta_ci
from incremental.policy import (
    marginal_profit_rule,
    observed_policy_value,
    profit_curve,
)

REPORTS = ROOT / "reports"
ECON = {
    "visit": {"margin": 25.0, "cost": 0.40},
    "conversion": {"margin": 4000.0, "cost": 0.40},
}

vis = np.load(ROOT / "data" / "criteo_val_scores_visit.npz")
con = np.load(ROOT / "data" / "criteo_val_scores_conversion.npz")
rng = np.random.default_rng(0)

results = {}
for label, d in [("visit", vis), ("conversion", con)]:
    m, c = ECON[label]["margin"], ECON[label]["cost"]
    y, t = d["y"], d["t"]
    rankings = {"X-learner": d["tau_x"], "propensity": d["resp"],
                "random": rng.uniform(size=len(y))}

    treat_all = observed_policy_value(y, t, np.ones(len(y), bool), m, c)
    curves = {n: profit_curve(y, t, s, m, c) for n, s in rankings.items()}
    peaks = {n: cv.loc[cv["profit"].idxmax()] for n, cv in curves.items()}

    print(f"\n=== {label.upper()} scenario (margin Rs{m:g}, cost Rs{c}) ===")
    print(f"  treat-all profit: Rs{treat_all['profit']:,.0f} on {len(y):,} users")
    for n, p in peaks.items():
        print(f"  {n:<11} peak profit Rs{p['profit']:,.0f} at budget "
              f"{p['budget']:.0%} ({p['n_selected']:,} contacts)")

    # marginal-profit rule on the X-learner (model plans, RCT scores it)
    rule = marginal_profit_rule(d["tau_x"], m, c)
    sel_value = observed_policy_value(y, t, rule["selected"], m, c)
    print(f"  X marginal rule: tau > {rule['threshold']:.4f} selects "
          f"{rule['share_selected']:.1%}; model expects Rs{rule['expected_profit_model']:,.0f}, "
          f"RCT-observed Rs{sel_value['profit']:,.0f}")

    results[label] = {
        "economics": ECON[label],
        "treat_all": treat_all,
        "curves": {n: cv.round(6).to_dict("records") for n, cv in curves.items()},
        "peaks": {n: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in p.items()} for n, p in peaks.items()},
        "marginal_rule": {k: v for k, v in rule.items() if k != "selected"},
        "marginal_rule_observed": sel_value,
    }

# ---- paired profit delta at the visit peak budget: X vs propensity ----
m, c = ECON["visit"]["margin"], ECON["visit"]["cost"]
peak_b = float(results["visit"]["peaks"]["X-learner"]["budget"])
def profit_at_peak(y_, t_, s_):
    return profit_curve(y_, t_, s_, m, c, budgets=np.array([peak_b])).iloc[0]["profit"]
delta = bootstrap_delta_ci(vis["y"], vis["t"], vis["tau_x"], vis["resp"], profit_at_peak)
per100k = {k: delta[k] * 100_000 / len(vis["y"]) for k in ("point", "lo", "hi")}
print(f"\nX vs propensity, observed profit @ {peak_b:.0%} budget (paired bootstrap):")
print(f"  delta Rs{delta['point']:,.0f}  CI [Rs{delta['lo']:,.0f}, Rs{delta['hi']:,.0f}]"
      f"  separable={delta['separable']}")
print(f"  per 100K users: Rs{per100k['point']:,.0f}  "
      f"CI [Rs{per100k['lo']:,.0f}, Rs{per100k['hi']:,.0f}]")
results["visit_profit_delta_at_peak"] = {**delta, "budget": peak_b,
                                         "per_100k_users": per100k}

# ---- the money chart ----
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=150)
colors = {"X-learner": "#1F6347", "propensity": "#B23A48", "random": "#999999"}
for ax, label in zip(axes, ["visit", "conversion"]):
    for n in ["X-learner", "propensity", "random"]:
        cv = results[label]["curves"][n]
        ax.plot([r["budget"] for r in cv], [r["profit"] for r in cv],
                label=n, color=colors[n], lw=2 if n == "X-learner" else 1.4,
                ls="--" if n == "random" else "-")
    ta = results[label]["treat_all"]["profit"]
    ax.axhline(ta, color="#8A6212", lw=1.1, ls=":",
               label=f"treat everyone (Rs{ta/1000:,.0f}K)")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xlabel("Contact budget (fraction of population)")
    ax.set_ylabel("Observed incremental profit (Rs)")
    ax.set_title(f"{label} economics: margin Rs{ECON[label]['margin']:g}, "
                 f"cost Rs{ECON[label]['cost']}")
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Money on the table: observed profit vs contact budget (600K-user val, RCT-scored)",
             fontsize=12, y=1.00)
fig.tight_layout()
fig.savefig(REPORTS / "profit_curves.png", bbox_inches="tight")

json.dump(results, open(REPORTS / "day8_policy.json", "w"), indent=2, default=float)
print(f"\nsaved -> reports/day8_policy.json, reports/profit_curves.png")
