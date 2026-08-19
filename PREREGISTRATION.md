# Pre-registration — golden holdout

**Date locked:** 2026-08-19 (Day 1 of build)

## Commitment

The golden holdout (`data/golden_holdout.parquet`, 19,200 rows = 30% of Hillstrom,
stratified by treatment arm × visit outcome, seed 42) will be evaluated **exactly
once**, on Day 9 of the build, to produce the headline metrics reported in the
README. No model selection, hyperparameter tuning, feature choice, or threshold
decision may use it. All development uses `data/train.parquet` with internal CV.

## Holdout fingerprint

```
SHA256(canonical CSV of holdout frame) =
4b1e135a1a59b382d1bc566818c93acceaec1a61411e96b93590ee57a56897fc
```

Verify anytime with:

```python
from incremental.data import load_hillstrom, make_golden_holdout, frame_hash
_, holdout = make_golden_holdout(load_hillstrom())
assert frame_hash(holdout) == "4b1e135a1a59b382d1bc566818c93acceaec1a61411e96b93590ee57a56897fc"
```

## Pre-committed decision rules

1. **Primary metric:** Qini coefficient and incremental conversions @ 20% contact
   budget on the holdout, models ranked against the baseline ladder
   (random → treat-all → propensity → T-learner → X-learner).
2. **Criteo label rule (Day 7 gate):** if conversion-label (0.29% base rate) Qini
   deltas between models are not separable at bootstrap 95% CIs, the **visit label
   (4.7%)** becomes the pre-registered primary Criteo metric and conversion is
   reported as secondary. This rule is being committed BEFORE any Criteo model run.
3. **Sampling protocol:** Criteo models tune on a stratified ≤3M-row slice;
   one documented final fit; evaluation on the full Criteo holdout.
4. No headline multiple is pre-committed; whatever the CI-bounded measured
   result is on Day 9 is what gets published.

## Randomization validation (Day 1, full dataset)

- SRM: χ² = 0.203, p = 0.904 — consistent with the designed 1/3 split
- Covariate balance: 0 of 36 arm×covariate pairs exceed |SMD| ≥ 0.10 (max 0.016)
- RCT ground truth (visit rate): control 10.62%, Mens E-Mail 18.28% (+7.66pp),
  Womens E-Mail 15.14% (+4.52pp)

---

## Amendment 1 — Criteo golden holdout (Day 6, before any Criteo model run)

**Date:** 2026-08-24. Committed BEFORE the first model is trained on Criteo data.

- Split rule: deterministic `row_number % 10 < 3` → holdout = 4,193,879 rows (30%);
  train partition = 9,785,713 rows.
- Holdout fingerprint (SHA256 of per-column aggregate signature):
  `eabff873fb5d5467630164e9cac7ddfdd8529d91178d6a0a9fa86817e7b2296b`
- Training sample: 1,999,943 rows, stratified by treatment × visit, drawn by
  `hash(rid)` ranking within strata — the documented sampling protocol.
- Randomization validation at full scale: treated share 0.8500 (design 0.85);
  0/12 features with |SMD| ≥ 0.10 (worst f3 = −0.049).
- Ground truth ATEs: visit +1.034pp, conversion +0.115pp.
- The Day-7 label gate (rule #2 above) will be evaluated on internal validation
  slices only; the holdout is evaluated once on Day 9.
