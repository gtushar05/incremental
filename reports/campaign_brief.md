# Campaign targeting brief — incremental policy (auto-generated)

## Recommendation
Deploy the X-learner targeting policy at an 8% contact budget. Do NOT run an
untargeted campaign: treating everyone is measured at Rs-577,472
(a loss), while the 8% policy earns Rs480,498 — 47%
more than propensity targeting at the same budget (Rs327,287).

## Evidence (sealed 4.19M-row holdout, evaluated once)
The policy's top decile concentrates 6.33pp of incremental
visits vs 5.16pp under propensity ranking — an advantage
of +1.17pp with 95% CI [0.89, 1.48]
(paired bootstrap; excludes zero). Predicted-vs-observed calibration across
deciles: r = 0.996.

## Next step
Confirmatory rollout A/B inside the targeted segment: with a
21.3% control rate and 7.2pp expected
uplift, 984 users per arm suffice (alpha=0.05, power=0.80).
