"""Day 6a: Criteo-UPLIFT v2 ETL in DuckDB — 13.9M rows on a laptop.

Steps (per the pre-registered protocol):
1. Validate randomization at full scale: treatment ratio vs the 85/15 design,
   covariate balance (SMD) across all 12 features.
2. Ground-truth lifts for both labels (visit 4.7%, conversion 0.29%).
3. Deterministic 70/30 train/holdout split (row_number % 10), holdout
   fingerprinted and recorded in PREREGISTRATION.md BEFORE any model run.
4. Stratified ~2M training sample (by treatment x visit) for model fitting;
   full holdout reserved for Day 9.
"""

import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
con = duckdb.connect()

con.execute(f"""
CREATE VIEW criteo AS
SELECT row_number() OVER () AS rid, *
FROM read_csv_auto('{DATA}/criteo.csv.gz')
""")

# ---- 1. scale + randomization ----
n, = con.execute("SELECT count(*) FROM criteo").fetchone()
ratio, = con.execute("SELECT avg(treatment) FROM criteo").fetchone()
print(f"rows: {n:,}   treated share: {ratio:.4f} (design: 0.85)")

feats = [f"f{i}" for i in range(12)]
stats = con.execute(
    "SELECT treatment, "
    + ", ".join(f"avg({f}) m_{f}, var_samp({f}) v_{f}" for f in feats)
    + " FROM criteo GROUP BY treatment ORDER BY treatment"
).fetchdf()
c, t = stats.iloc[0], stats.iloc[1]
smds = {
    f: float((t[f"m_{f}"] - c[f"m_{f}"]) / np.sqrt((t[f"v_{f}"] + c[f"v_{f}"]) / 2))
    for f in feats
}
worst = max(smds.items(), key=lambda kv: abs(kv[1]))
n_bad = sum(abs(v) >= 0.10 for v in smds.values())
print(f"covariate balance: {n_bad}/12 features |SMD|>=0.10   worst: {worst[0]}={worst[1]:+.4f}")

# ---- 2. ground truth ----
gt = con.execute("""
SELECT treatment, count(*) n, avg(visit) visit_rate, avg(conversion) conv_rate
FROM criteo GROUP BY treatment ORDER BY treatment
""").fetchdf()
print("\nground truth by arm:")
print(gt.round(5).to_string(index=False))
v_lift = gt.visit_rate[1] - gt.visit_rate[0]
c_lift = gt.conv_rate[1] - gt.conv_rate[0]
print(f"ATE: visit {v_lift:+.5f}   conversion {c_lift:+.5f}")

# ---- 3. deterministic split + fingerprint ----
con.execute(f"""
COPY (SELECT * FROM criteo WHERE rid % 10 < 3)
TO '{DATA}/criteo_holdout.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
con.execute(f"""
COPY (SELECT * FROM criteo WHERE rid % 10 >= 3)
TO '{DATA}/criteo_train.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

fp_stats = con.execute(f"""
SELECT count(*) n, sum(treatment) st, sum(visit) sv, sum(conversion) sc,
{", ".join(f"round(sum(f{i}), 3) sf{i}" for i in range(12))}
FROM read_parquet('{DATA}/criteo_holdout.parquet')
""").fetchdf().to_dict("records")[0]
fingerprint = hashlib.sha256(
    json.dumps({k: str(v) for k, v in fp_stats.items()}, sort_keys=True).encode()
).hexdigest()
print(f"\nholdout: {fp_stats['n']:,} rows   fingerprint: {fingerprint}")

# ---- 4. stratified ~2M training sample ----
con.execute(f"""
COPY (
  SELECT * FROM (
    SELECT *, row_number() OVER (
      PARTITION BY treatment, visit ORDER BY hash(rid)
    ) AS strat_rn,
    count(*) OVER (PARTITION BY treatment, visit) AS strat_n
    FROM read_parquet('{DATA}/criteo_train.parquet')
  ) WHERE strat_rn <= strat_n * {2_000_000 / 9_786_000:.6f}
) TO '{DATA}/criteo_sample_2m.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
ns, = con.execute(
    f"SELECT count(*) FROM read_parquet('{DATA}/criteo_sample_2m.parquet')"
).fetchone()
print(f"training sample: {ns:,} rows (stratified by treatment x visit)")

json.dump(
    {"n_total": n, "treated_share": ratio, "smds": smds,
     "visit_ate": v_lift, "conversion_ate": c_lift,
     "holdout_fingerprint": fingerprint, "holdout_n": int(fp_stats["n"]),
     "sample_n": int(ns)},
    open(ROOT / "reports" / "day6_criteo_etl.json", "w"), indent=2)
print(f"\nsaved -> reports/day6_criteo_etl.json")
