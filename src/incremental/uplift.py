"""Uplift meta-learners, implemented from scratch.

Both learners estimate the conditional average treatment effect (CATE):

    tau(x) = E[Y | X=x, T=1] - E[Y | X=x, T=0]

T-LEARNER ("two models"): fit one outcome model per arm, subtract predictions.
Simple and unbiased-ish, but each model is tuned to predict OUTCOMES, not
their difference — errors that would cancel in a joint fit instead add up.

X-LEARNER (Kunzel et al., 2019): a two-stage repair designed for imbalanced
arms (Criteo is 85/15). Stage 1 = the T-learner's outcome models. Stage 2
imputes each individual's treatment effect using the OPPOSITE arm's model,
then regresses those imputed effects on X directly. The final estimate blends
the two stage-2 models, weighted by the propensity score, so the model trained
on the LARGER arm gets more say where it should.

Base learners are injected as factories so the same code runs XGBoost today
and anything else tomorrow.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

OutcomeModel = Callable[[], object]  # classifier factory, has fit/predict_proba
EffectModel = Callable[[], object]   # regressor factory, has fit/predict


class TLearner:
    """Two independent outcome models; uplift = difference of predictions."""

    def __init__(self, make_outcome_model: OutcomeModel):
        self.make_outcome_model = make_outcome_model

    def fit(self, X: pd.DataFrame, y: np.ndarray, t: np.ndarray) -> "TLearner":
        t = np.asarray(t).astype(bool)
        self.model_treated_ = self.make_outcome_model()
        self.model_control_ = self.make_outcome_model()
        self.model_treated_.fit(X[t], y[t])
        self.model_control_.fit(X[~t], y[~t])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p_t = self.model_treated_.predict_proba(X)[:, 1]
        p_c = self.model_control_.predict_proba(X)[:, 1]
        return p_t - p_c


class XLearner:
    """Kunzel et al. X-learner with a constant RCT propensity.

    In a randomized experiment the propensity e(x) is a known constant
    (the treated share), so the blending weight needs no propensity model.
    """

    def __init__(
        self,
        make_outcome_model: OutcomeModel,
        make_effect_model: EffectModel,
    ):
        self.make_outcome_model = make_outcome_model
        self.make_effect_model = make_effect_model

    def fit(self, X: pd.DataFrame, y: np.ndarray, t: np.ndarray) -> "XLearner":
        t = np.asarray(t).astype(bool)
        y = np.asarray(y).astype(float)

        # ---- stage 1: outcome models per arm (identical to T-learner) ----
        self.mu_treated_ = self.make_outcome_model()
        self.mu_control_ = self.make_outcome_model()
        self.mu_treated_.fit(X[t], y[t])
        self.mu_control_.fit(X[~t], y[~t])

        # ---- stage 2: impute individual effects with the OPPOSITE model ----
        # treated user: actual outcome minus what control-world predicts for them
        d_treated = y[t] - self.mu_control_.predict_proba(X[t])[:, 1]
        # control user: what treated-world predicts for them minus actual outcome
        d_control = self.mu_treated_.predict_proba(X[~t])[:, 1] - y[~t]

        self.tau_treated_ = self.make_effect_model()
        self.tau_control_ = self.make_effect_model()
        self.tau_treated_.fit(X[t], d_treated)
        self.tau_control_.fit(X[~t], d_control)

        # ---- blending weight: known constant propensity in an RCT ----
        self.propensity_ = float(t.mean())
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        tau_t = self.tau_treated_.predict(X)
        tau_c = self.tau_control_.predict(X)
        e = self.propensity_
        # weight the model built on MORE data higher where it matters:
        # g(x)=e gives tau_c weight e (control model saw the big arm when e high)
        return e * tau_c + (1.0 - e) * tau_t
