"""Incremental — the decision cockpit.

Everything shown here is PRECOMPUTED: the app reads reports/*.json produced
by the day scripts and renders decisions. No model runs at request time, so
the demo is instant and can never break in front of an interviewer.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Incremental — decision cockpit",
                   page_icon="🧲", layout="wide")


@st.cache_data
def load(name):
    p = REPORTS / name
    return json.load(open(p)) if p.exists() else None


day2 = load("day2_baseline.json")
day5 = load("day5_evaluation.json")
day7 = load("day7_gate.json")
day8 = load("day8_policy.json")
day9 = load("day9_holdout_FROZEN.json")

# ---------- header ----------
st.title("Incremental — a causal targeting engine")
st.caption(
    "Who should we actually contact — not who will convert. Uplift models on "
    "13.9M-user randomized-experiment data, turned into budget-constrained "
    "profit decisions. github.com/gtushar05/incremental")

if day9:
    d1 = day9["criteo_headline_deltas"]["visit_u10_X_minus_prop"]
    pol = day9["criteo_policy_at_8pct"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uplift concentrated in top 10% (X-learner)",
              f"{day9['criteo_visit_ladder'][3]['u10_point']*100:.2f}pp",
              help="Observed lift inside the X-learner's top decile on the "
                   "sealed 4.19M-row holdout. Population average: ~1pp.")
    c2.metric("vs propensity targeting (paired 95% CI)",
              f"+{d1['point']*100:.2f}pp",
              f"CI [{d1['lo']*100:+.2f}, {d1['hi']*100:+.2f}]pp",
              delta_color="normal")
    c3.metric("Frozen 8%-budget policy profit (holdout)",
              f"₹{pol['X_profit']:,.0f}",
              f"treat-all: ₹{pol['treat_all_profit']:,.0f}",
              delta_color="off")
    c4.metric("Uplift calibration (predicted vs observed)",
              f"r = {day9['criteo_visit_calibration_corr']:.3f}",
              help="Correlation between predicted uplift and RCT-observed "
                   "lift across score deciles, on the sealed holdout.")
    st.caption("Headline numbers from the single pre-registered golden-holdout "
               "evaluation (Day 9) — sealed data, hashed in the git log, "
               "evaluated exactly once.")
else:
    st.info("Golden-holdout numbers not yet frozen — showing validation-split "
            "results (Days 2–8).")

st.divider()

# ---------- the budget slider ----------
st.header("Slide the budget — watch the money move")
if day8:
    scenario = st.radio(
        "Economics scenario", ["visit", "conversion"], horizontal=True,
        format_func=lambda s: {
            "visit": "Visits — margin ₹25 / contact ₹0.40 (thin margins: targeting is existential)",
            "conversion": "Conversions — margin ₹4,000 / contact ₹0.40 (fat margins: treat-all is near-optimal)",
        }[s])
    sc = day8[scenario]
    budgets = [r["budget"] for r in sc["curves"]["X-learner"]]
    budget = st.slider("Contact budget (share of 600K-user population)",
                       min_value=min(budgets), max_value=max(budgets),
                       value=0.08, step=0.02, format="%.0f%%" if False else "%g")

    def at_budget(name):
        rows = sc["curves"][name]
        i = int(np.argmin([abs(r["budget"] - budget) for r in rows]))
        return rows[i]

    x_row, p_row, r_row = at_budget("X-learner"), at_budget("propensity"), at_budget("random")
    ta = sc["treat_all"]["profit"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("X-learner profit", f"₹{x_row['profit']:,.0f}",
              f"{x_row['n_selected']:,.0f} contacts", delta_color="off")
    m2.metric("Propensity profit", f"₹{p_row['profit']:,.0f}",
              f"₹{x_row['profit']-p_row['profit']:+,.0f} vs X", delta_color="off")
    m3.metric("Random targeting", f"₹{r_row['profit']:,.0f}", delta_color="off")
    m4.metric("Treat everyone", f"₹{ta:,.0f}", delta_color="off")

    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=120)
    for name, color, lw in [("X-learner", "#1F6347", 2.2),
                            ("propensity", "#B23A48", 1.6),
                            ("random", "#999999", 1.3)]:
        rows = sc["curves"][name]
        ax.plot([r["budget"] for r in rows], [r["profit"] for r in rows],
                label=name, color=color, lw=lw,
                ls="--" if name == "random" else "-")
    ax.axhline(ta, color="#8A6212", lw=1.1, ls=":", label="treat everyone")
    ax.axhline(0, color="#444", lw=0.8)
    ax.axvline(budget, color="#276678", lw=1.2, alpha=0.7)
    ax.set_xlabel("contact budget")
    ax.set_ylabel("observed incremental profit (₹)")
    ax.legend(frameon=False, fontsize=9, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)
    st.caption(
        "Profit is OBSERVED, not predicted: the model only chooses who to "
        "contact; the money is scored by treated-vs-control differences inside "
        "the chosen group — unbiased because assignment was randomized.")

st.divider()

# ---------- evidence tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Qini & calibration", "🧩 Segments", "🔬 Method & pre-registration", "📜 Findings"])

with tab1:
    c1, c2 = st.columns(2)
    if (REPORTS / "qini_curves.png").exists():
        c1.image(str(REPORTS / "qini_curves.png"),
                 caption="Qini curves — who finds the persuadables first")
    if (REPORTS / "calibration.png").exists():
        c2.image(str(REPORTS / "calibration.png"),
                 caption="Predicted vs observed uplift by decile")
    if (REPORTS / "misallocation.png").exists():
        st.image(str(REPORTS / "misallocation.png"),
                 caption="Day 2: where a response model spends budget vs where lift lives")

with tab2:
    if day5:
        st.subheader("The persuadables quadrant (Hillstrom, X-learner scores)")
        st.dataframe(pd.DataFrame(day5["quadrants"]).round(4),
                     use_container_width=True, hide_index=True)
        st.caption("Each segment's observed lift comes from treated-vs-control "
                   "differences INSIDE the segment — the labels are validated, "
                   "not asserted.")

with tab3:
    st.markdown("""
**The three data fences.** Models train on a fit split; all comparisons and
tuning happen on a validation split; final numbers come from **golden holdouts
sealed on day 1** (Hillstrom SHA256 `4b1e135a…`, Criteo fingerprint
`eabff873…`) — committed to git **before any model existed**, evaluated
exactly once.

**Pre-registered decision rules.** The primary-label gate, the sampling
protocol, and the bootstrap protocol (200 stratified resamples) were committed
before results existed. The gate then bound us to keep the label where our own
models **lose** (conversions) — that negative finding is reported, not hidden.

**Paired-delta inference.** Model comparisons use paired bootstrap deltas
(same resamples on both rankings) — overlapping individual CIs cannot
adjudicate differences. The machinery caught a real tie-ordering bug in the
profit layer via a mathematically impossible CI; the fix and regression test
are in the git history.
""")

with tab4:
    st.markdown("""
1. **The industry default burns money — measurably.** Response-model targeting
   forfeits 14–17% of capturable lift at 30–50% budgets (oracle-efficiency
   analysis), because ranking by P(outcome) ≠ ranking by effect.
2. **Small data is fog.** On 64K rows, T/X/propensity are statistically
   indistinguishable — stated with CIs, not hidden.
3. **At scale, uplift wins where theory says:** the X-learner's top decile
   concentrates ~6× the average lift, beating propensity by a CI-separable
   margin — and at thin margins (visits), targeting flips the campaign from
   loss to profit at an 8% budget.
4. **And loses where theory says:** at a 0.29% base rate (conversions),
   effect-differencing is noise-dominated and the propensity model wins —
   while fat margins make treat-all near-optimal anyway. **Uplift modeling
   pays on responsive outcomes under tight budgets; response models remain
   right for ultra-rare outcomes; and when margin dwarfs cost, don't target
   at all.** Knowing the regime is the skill.
""")

st.divider()
st.caption("Built in a 14-day sprint · pure NumPy/XGBoost uplift library, "
           "verified vs sklift + synthetic ground truth · all numbers "
           "reproducible from the repo's day scripts.")
