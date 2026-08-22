# Defense pack — CV bullets, demo script, and the gauntlet

The final deliverable of the 14-day sprint: everything needed to present and
defend this project in placement interviews. Every number is frozen
(reports/day9_holdout_FROZEN.json).

## CV bullets (final)

- Built **Incremental**, a causal targeting engine on 13.9M-user
  randomized-experiment data (Criteo-UPLIFT v2): implemented T/X meta-learners
  and Qini/AUUC evaluation from scratch (T-learner bit-identical to sklift,
  r = 1.000000; X-learner verified on synthetic ground truth), with
  hyperparameters selected by out-of-fold Qini.
- On a **pre-registered, sealed 4.19M-row holdout** (hashes committed before
  any model run, evaluated exactly once): X-learner's top decile concentrates
  **6.33pp of incremental visits vs propensity's 5.16pp** — a **+1.17pp paired
  bootstrap advantage, 95% CI [+0.89, +1.48]** — with predicted-vs-observed
  calibration **r = 0.996**.
- Converted uplift scores into a **budget-constrained profit policy** scored
  by RCT ground truth: at an 8% contact budget the policy earns **₹480K where
  treating everyone loses ₹577K** (+47% vs propensity targeting); designed the
  confirmatory rollout A/B (984 users/arm at α=.05, power .80).
- Engineering: 13.9M-row ETL in DuckDB on a laptop; 29-test suite including
  synthetic-RCT ground truth and adversarial validator tests; paired-bootstrap
  inference machinery that **caught a real tie-ordering bug via a
  mathematically impossible CI**; live zero-dependency demo on HF Spaces.

## The 90-second demo script

1. Open gtushar-05-incremental-cockpit.static.hf.space. "Every number here is
   a precomputed artifact from sealed-holdout evaluation — nothing runs live,
   so nothing can break."
2. Point at treat-everyone: "−₹577K. An untargeted campaign *loses* money
   under these economics. Targeting isn't optimization — it's survival."
3. Drag budget slider to 8%: "My policy: ₹480K profit from contacting 8% of
   users — 47% more than the standard response-model approach at the same
   budget."
4. Switch scenario to conversions: "Same system, honest answer: at fat
   margins, treat-everyone is near-optimal and my model adds nothing. Knowing
   *when not to use uplift* is half the project."
5. Close: "Sealed holdouts, pre-registered rules, paired-CI inference —
   check the git log; the seal predates every model."

## The gauntlet — hostile questions, model answers

**"Why not just AUC?"** Uplift has no per-row ground truth — one potential
outcome per person. Evaluation must be aggregate and randomization-powered:
rank, walk down, compare arms within the prefix — that's the Qini curve, which
I hand-rolled and unit-tested (endpoint provably equals total scaled effect).

**"Your Qini is 0.003. That's tiny."** Per-person normalization; the scale is
set by the 1pp ATE. Compare between rankings, never to 1.0. At the top decile
— where budgets operate — the same ranking concentrates 6.33pp, six times the
population average.

**"Propensity's aggregate Qini matches yours. So why bother?"** Because Qini
integrates over all depths including ones no budget reaches. At deployable
depths the paired delta is +1.17pp [+0.89, +1.48]. And in money: +47% profit
at 8% budget. Metric choice is a business decision; I can defend mine.

**"Why did your models LOSE on conversions?"** 0.29% base rate = 548 control
conversions in the fit sample. Effect estimation differences two noisy
probabilities; the noise doubles while propensity coasts on outcome-effect
correlation. Theory predicts it, my pre-registration forced me to headline
it, and it replicated on the sealed holdout. That's the decision boundary:
uplift needs estimable signal.

**"How do I know you didn't tune on the holdout?"** You don't have to trust
me — verify. The holdout hashes are in dated commits that precede every
model; the decision rules (label gate, sampling protocol) are pre-registered
in the same file; the holdout was evaluated once, on Day 9, and the numbers
froze.

**"Depth-2 trees? Really?"** Chosen by out-of-fold Qini, not accuracy.
Effect signals are second-order — differencing doubles the noise — so uplift
wants heavy regularization. The sweep confirmed what theory predicts.

**"What breaks with observational data?"** (The Slice question.) No
randomization → arm differences confound treatment with type. I'd need
propensity adjustment / IPW / doubly-robust estimators, and I'd lose the
clean within-decile validation — claims soften from "measured" to "estimated
under assumptions." At Slice I'd push for holdout experiments precisely to
keep the measured version.

**"Biggest mistake during the build?"** The profit layer's tie-ordering bias:
a 167K-row tied score block met arm-clustered rows and silently skewed
selections. My paired bootstrap caught it — the point estimate landed outside
its own CI, which is mathematically impossible on clean inputs. The fix's
first version (additive jitter) then failed its own regression test on
constant scores (float absorption). Both fixes and tests are in the history.
Point estimates alone would never have caught any of it.

**"What would you build next?"** In ROI order: serving layer with latency
budget (FastAPI/ONNX), formal OPE (IPS/DR) so the system works without fresh
RCTs, S/DR-learners, X5 RetailHero for the messy-feature story, and the full
LLM brief-writer eval set.
