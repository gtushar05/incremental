# Why your response model is burning your campaign budget

Every consumer company runs the same play: train a model to predict who will
convert, rank customers by that probability, contact the top of the list.
It feels rigorous. It is measurably wrong.

The problem is *who* sits at the top of that ranking: your surest customers —
people who would have converted anyway. Paying to contact them buys nothing.
Worse, some customers react badly to contact and are *lost because of it*.
The object that actually matters is not P(convert) but the **individual
treatment effect**: how much does contacting this person *change* what they
do? Estimating that is causal inference, and it needs experiment data —
you never observe both worlds for the same person, so randomization must
stand in for the missing one.

**Incremental** builds that system end to end on two public randomized
experiments (Hillstrom, 64K; Criteo-UPLIFT v2, 13.9M). T- and X-learner
meta-models, written from scratch and verified two ways — bit-identical
scores against an established library, and recovery of known effects on
synthetic experiments. Evaluation is the hard part: uplift has no per-row
ground truth, so the machinery is hand-rolled Qini curves, uplift-at-budget,
bootstrap confidence intervals, and paired-delta tests for every model
comparison. All headline numbers come from a **single evaluation of sealed
holdout data** — hashed into the git history before any model existed, with
decision rules pre-registered so no result could be cherry-picked after the
fact.

Three findings survived that discipline.

**First, the default is quantifiably wasteful.** The response model's ranking
is non-monotonic in true effect: its mid-rank favorites lift less than
customers it skips entirely. Against an oracle that knows each decile's true
effect, propensity targeting forfeits 14–17% of capturable lift at realistic
budgets.

**Second, uplift wins exactly where theory predicts — and the win is money.**
On 4.19M sealed rows, the X-learner's top decile concentrates 6.33pp of lift
(~6× the population average), beating propensity by +1.17pp with a confidence
interval that cleanly excludes zero. Under stated economics where treating
everyone *loses* ₹577K, the frozen 8%-budget policy earns ₹480K — 47% more
than propensity targeting at the same budget. The scores are also honest
magnitudes, not just rankings: predicted-versus-observed calibration of
r = 0.996, which is what licenses profit arithmetic at all.

**Third — the finding I'm proudest of — uplift also loses where theory
predicts, and the pre-registration forced me to headline it.** On conversions
(a 0.29% base rate), effect estimation differences two noisy probabilities
and drowns; the simpler propensity model wins, significantly. With fat
margins, treat-everyone is near-optimal anyway.

So the deliverable is not "my model won." It is a **decision boundary**:
uplift modeling pays on responsive outcomes under tight budgets; response
models remain right for ultra-rare outcomes; when margin dwarfs cost, don't
target at all. Knowing the regime before modeling — and building evaluation
rigorous enough to trust which regime you're in — is the actual skill this
project demonstrates.

*Repo: github.com/gtushar05/incremental · Live demo:
gtushar-05-incremental-cockpit.static.hf.space*
