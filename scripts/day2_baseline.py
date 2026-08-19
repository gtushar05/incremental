"""Day 2: train the propensity baseline and produce the misallocation exhibit."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incremental.baseline import (
    internal_split,
    train_response_model,
    response_auc,
    misallocation_table,
    misallocation_headline,
)

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

train = pd.read_parquet(ROOT / "data" / "train.parquet")
fit, val = internal_split(train)
print(f"internal split: fit {len(fit):,} / val {len(val):,} (golden holdout untouched)")

model, cols = train_response_model(fit)
auc = response_auc(model, cols, val)
print(f"response model AUC on val (treated arm): {auc:.4f}  <- the wrong-question number")

table = misallocation_table(model, cols, val)
head = misallocation_headline(table)

print("\n=== response-score deciles vs TRUE experimental uplift (visit) ===")
show = table.copy()
for c in ["mean_score", "treated_rate", "control_rate", "true_uplift"]:
    show[c] = show[c].round(4)
show["incremental"] = show["incremental"].round(1)
print(show.to_string(index=False))

print("\n=== misallocation at increasing contact budgets ===")
budget_rows = []
for k in range(2, 6):
    h = misallocation_headline(table, top_k=k)
    budget_rows.append(h)
    print(
        f"  budget {k*10}%: response captures {h['response_share']:.1%} of total "
        f"incremental vs oracle {h['oracle_share']:.1%}  "
        f"-> efficiency {h['efficiency_vs_oracle']:.1%}"
    )
head = budget_rows[0]

# ---- the money chart v1 ----
fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=150)
colors = ["#B23A48" if d <= 2 else "#3E6C8E" for d in table["decile"]]
ax.bar(table["decile"], table["true_uplift"] * 100, color=colors, width=0.72)
avg = (table["incremental"].sum() / table["n"].sum()) * 100
ax.axhline(avg, ls="--", lw=1.2, color="#555", label=f"average uplift ({avg:.1f}pp)")
ax.set_xlabel("Response-model decile (1 = model's favorite customers)")
ax.set_ylabel("True experimental uplift in visits (pp)")
eff30 = budget_rows[1]["efficiency_vs_oracle"]
ax.set_title(
    "Response-model ranking ≠ uplift ranking\n"
    "Its mid-rank 'favorites' (deciles 3–6) lift less than deciles it would skip (7–10); "
    f"at a 30% budget it forfeits {1-eff30:.0%} of capturable incremental visits"
)
ax.set_xticks(table["decile"])
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(REPORTS / "misallocation.png")
print(f"\nchart -> {REPORTS/'misallocation.png'}")

with open(REPORTS / "day2_baseline.json", "w") as f:
    json.dump(
        {"auc_treated_val": auc, "headline": head,
         "deciles": table.round(6).to_dict(orient="records")},
        f, indent=2,
    )
print(f"metrics -> {REPORTS/'day2_baseline.json'}")
