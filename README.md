# Incremental — a causal targeting engine

**Who should we actually contact — not who will convert.**

Response models rank users by P(convert) and burn campaign budget on people who
would have converted anyway (and on people who react *badly* to contact). This
project learns individual treatment effects (uplift) from randomized-experiment
data and turns them into a budget-constrained targeting policy with quantified
incremental profit.

> 🚧 Build in progress — Day 1 of a 14-day sprint. Headline metrics land Day 9
> (single pre-registered golden-holdout evaluation — see [PREREGISTRATION.md](PREREGISTRATION.md)).

## Status

| Day | Milestone | Status |
|---|---|---|
| 1 | Randomization validated (SRM p=0.90, 0/36 covariates imbalanced) · golden holdout locked & hashed | ✅ |
| 2 | Propensity baseline (AUC 0.58 on the wrong question) + misallocation exhibit: response targeting forfeits 14-17% of capturable lift at 30-50% budgets | ✅ |
| 3–4 | T/X-learners from scratch, verified (r=1.000 vs sklift + synthetic ground-truth suite); Qini-based CV tuning; seed stability 0.987 | ✅ |
| 5 | Hand-rolled Qini/AUUC + bootstrap CIs | ⬜ |
| 6–7 | Criteo 13.9M: ETL, models, signal-viability gate | ⬜ |
| 8 | Policy layer: incremental-profit curves | ⬜ |
| 9 | **Golden-holdout evaluation — headline numbers** | ⬜ |
| 10–11 | Streamlit decision cockpit (HF Spaces) | ⬜ |
| 12–14 | README, write-up, stretch, mocks | ⬜ |

## Data

- [Hillstrom Email Analytics](http://www.minethatdata.com/blog/2008/03/minethatdata-e-mail-analytics-and-data.html) — 64K customers, randomized 3-arm email experiment (dev dataset)
- [Criteo-UPLIFT v2](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) — 13.9M users, randomized 85/15 (scale + headline dataset)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[model,dev]"
python scripts/day1_validate.py
pytest
```
