"""LightGBM GBDT model.  ⛏️  EXERCISE — Milestone 3.

LEARNING GOAL
    GBDTs (gradient-boosted decision trees) dominate tabular ML competitions and
    production for a reason: they handle mixed types, missing values, and class
    imbalance well out of the box, and beat logistic regression on most real datasets.
    But they need a baseline to be meaningful — "LightGBM is better" only makes sense
    if you quantify BY HOW MUCH over logistic regression.

KEY CONCEPT — class imbalance via scale_pos_weight
    In a dataset where 25% are positive (underperforming), a naive model predicts
    "healthy" for everything and gets 75% accuracy — useless. scale_pos_weight =
    n_negative / n_positive tells LightGBM to penalize missing a positive n× more
    than missing a negative. This shifts the decision boundary toward catching
    more positives (higher recall at the cost of precision). The correct value is
    computed from the TRAINING set only.

INTERVIEW ANGLE
    "How do you handle class imbalance?" — three approaches: (1) scale_pos_weight
    (adjust loss weights), (2) SMOTE/oversampling (risky: can leak), (3) threshold
    tuning after training. scale_pos_weight is the cleanest: no data manipulation,
    works with early stopping, and is theoretically sound (it reweights the gradient).

TEST
    tests/test_models.py::test_compute_scale_pos_weight
    tests/test_models.py::test_gbdt_fits_and_predicts
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Return n_negative / n_positive — the class imbalance ratio.

    TODO(lillian):
      - Count how many 0s and 1s are in y.
      - Return n_neg / n_pos.
      - Why must you compute this from the TRAINING SET ONLY?
    """
    n_pos = (y == 1).sum()  
    n_neg = (y == 0).sum()
    return n_neg / n_pos



def build_gbdt(scale_pos_weight: float = 1.0) -> lgb.LGBMClassifier:
    """Return an unfitted LGBMClassifier with sensible defaults.

    TODO(lillian):
      - Return an LGBMClassifier. Key params to set:
          n_estimators=300, learning_rate=0.05, num_leaves=31,
          min_child_samples=20, scale_pos_weight=scale_pos_weight,
          random_state=42, verbose=-1
      - What does each param control? Why is num_leaves the key complexity
        knob in LightGBM (vs max_depth in sklearn trees)?
    """
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )

    


def fit_gbdt(
    model: lgb.LGBMClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.LGBMClassifier:
    """Fit the GBDT on train, monitoring validation loss.

    TODO(lillian):
      - Call model.fit() with X_train, y_train.
      - Pass eval_set=[(X_val, y_val)] so LightGBM tracks val loss each round.
      - Add callbacks=[lgb.log_evaluation(period=0)] to suppress console spam.
      - Return the fitted model.
    """
    return model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)]
    )

    # raise NotImplementedError("Implement fit_gbdt — see tests/test_models.py")