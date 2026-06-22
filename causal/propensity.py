"""Propensity scores & inverse-probability weighting (IPW).  ⛏️  EXERCISE — Milestone 6.

LEARNING GOAL
    Estimate a treatment effect from OBSERVATIONAL data, where treatment wasn't
    randomized (e.g. "does fast delivery cause better reviews?" — but fast delivery
    isn't assigned at random; cheaper/closer orders get it). Propensity weighting
    rebalances the groups to mimic a randomized experiment.

KEY CONCEPT — propensity, IPW, and its assumptions
    Propensity e(x) = P(treated | covariates x), estimated with logistic regression.
    IPW reweights each unit by 1/e(x) (treated) or 1/(1-e(x)) (control) so confounders
    are balanced, then compares weighted outcomes → ATE.
    ASSUMPTIONS you must state: (1) unconfoundedness (you measured the confounders),
    (2) positivity/overlap (every unit could plausibly get either treatment —
    0 < e(x) < 1). Check overlap; CLIP extreme weights or you get huge-variance
    nonsense from a single near-0 propensity.

INTERVIEW ANGLE
    Name the estimand (ATE vs ATT), state both assumptions unprompted, and mention
    weight clipping / trimming. That trio signals real causal literacy.

TEST
    tests/test_propensity.py  (synthetic data with a KNOWN confounded effect)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def estimate_propensity(X: pd.DataFrame, treatment: pd.Series) -> np.ndarray:
    """Return P(treated | X) for each row via logistic regression.

    TODO(lillian): fit sklearn LogisticRegression(treatment ~ X), return
    predict_proba(X)[:, 1]. (In practice you'd scale X; keep it simple and document.)
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, treatment)
    return model.predict_proba(X)[:, 1]



def check_overlap(propensity: np.ndarray, treatment: pd.Series) -> dict:
    """Summarize positivity/overlap: min/max propensity within each arm.

    TODO(lillian): return {'treated_min','treated_max','control_min','control_max',
    'has_overlap': bool}. has_overlap is False if either arm has propensities pinned
    near 0 or 1 (define a small epsilon). This is the diagnostic you'd SHOW before
    trusting any IPW estimate.
    """
    treated_propensity = propensity[treatment == 1]
    control_propensity = propensity[treatment == 0]
    treated_min = treated_propensity.min()
    treated_max = treated_propensity.max()
    control_min = control_propensity.min()
    control_max = control_propensity.max()
    epsilon = 0.05
    ####### is this correct way to calculate overlap?  (no arm has propensities pinned near 0 or 1) #######
    has_overlap = (treated_min > epsilon) and (treated_max < 1 - epsilon) and \
                    (control_min > epsilon) and (control_max < 1 - epsilon)
    return {
        'treated_min': treated_min,
        'treated_max': treated_max,
        'control_min': control_min,
        'control_max': control_max,
        'has_overlap': has_overlap
    }


def ipw_ate(
    outcome: pd.Series, treatment: pd.Series, propensity: np.ndarray, clip: float = 0.01
) -> float:
    """Estimate the Average Treatment Effect via inverse-probability weighting.

    TODO(lillian):
      - clip propensity to [clip, 1 - clip]  (variance control / positivity guard)
      - weights: 1/e for treated, 1/(1-e) for control
      - ATE = weighted mean outcome (treated) - weighted mean outcome (control)
    On the synthetic test, the naive difference-in-means is biased by the confounder
    and IPW recovers the true effect within tolerance — that contrast is the lesson.
    """
    # raise NotImplementedError
    e = np.clip(propensity, clip, 1 - clip)
    w_treated = 1 / e[treatment == 1]
    w_control = 1 / (1 - e[treatment == 0])
    y_treated = outcome[treatment == 1]
    y_control = outcome[treatment == 0]
    mean_treated = np.sum(w_treated * y_treated) / np.sum(w_treated)
    mean_control = np.sum(w_control * y_control) / np.sum(w_control)
    return mean_treated - mean_control