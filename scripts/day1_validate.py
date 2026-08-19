"""Day 1: validate randomization, report RCT ground truth, lock the golden holdout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incremental.data import (
    DATA_DIR,
    load_hillstrom,
    make_golden_holdout,
    frame_hash,
)
from incremental.validation import srm_check, covariate_balance, rct_outcome_table

CONTROL = "No E-Mail"
ARM_COL = "segment"

df = load_hillstrom()
print(f"Loaded Hillstrom: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"Columns: {list(df.columns)}\n")

# ---- 1. Sample-ratio mismatch (design: equal thirds) ----
arms = df[ARM_COL].unique()
srm = srm_check(df[ARM_COL], {a: 1 / 3 for a in arms})
print("=== SRM check (design = 1/3 each) ===")
for a, o, e in zip(srm["arms"], srm["observed"], srm["expected"]):
    print(f"  {a:<15} observed {o:>6,}   expected {e:>9,.1f}")
print(f"  chi2 = {srm['chi2']:.3f}   p = {srm['p_value']:.4f}   PASS = {srm['pass']}\n")

# ---- 2. Covariate balance (each arm vs control) ----
numeric = ["recency", "history", "mens", "womens", "newbie"]
categorical = ["history_segment", "zip_code", "channel"]
bal = covariate_balance(df, ARM_COL, CONTROL, numeric, categorical)
worst = bal.reindex(bal["smd"].abs().sort_values(ascending=False).index).head(8)
print("=== Covariate balance: 8 largest |SMD| (threshold 0.10) ===")
print(worst.to_string(index=False))
n_imbalanced = int((~bal["balanced"]).sum())
print(f"\nImbalanced covariate-arm pairs: {n_imbalanced} / {len(bal)}\n")

# ---- 3. RCT ground-truth outcomes ----
truth = rct_outcome_table(df, ARM_COL, CONTROL, ["visit", "conversion", "spend"])
print("=== RCT ground truth (the numbers every model must respect) ===")
print(truth.round(4).to_string(), "\n")

# ---- 4. Lock the golden holdout ----
train, holdout = make_golden_holdout(df)
DATA_DIR.mkdir(exist_ok=True)
train.to_parquet(DATA_DIR / "train.parquet")
holdout.to_parquet(DATA_DIR / "golden_holdout.parquet")
h = frame_hash(holdout)
print("=== Golden holdout locked ===")
print(f"  train:   {len(train):,} rows")
print(f"  holdout: {len(holdout):,} rows ({len(holdout)/len(df):.1%})")
print(f"  SHA256:  {h}")
