"""Logistic regression baseline.  ⛏️  EXERCISE — Milestone 3.

LEARNING GOAL
    Every modeling project needs a baseline before a fancy model. "We beat the
    baseline by X" is the only meaningful frame for model complexity. Logistic
    regression IS the baseline for binary classification: it's fast, interpretable,
    and gives calibrated probabilities out of the box (with StandardScaler).

KEY CONCEPT — sklearn Pipeline
    A Pipeline chains preprocessing + model into one object that you can .fit(),
    .predict_proba(), and cross-validate. This makes leakage structurally impossible:
    the scaler and encoder are fitted inside .fit() and applied inside .predict_proba()
    — they can never see the validation set during training. This is the production
    pattern; bare transforms outside a Pipeline are a leakage trap.

INTERVIEW ANGLE
    "What's your baseline?" is almost always the first modeling question. The right
    answer is logistic regression (not the mean, not a tree). Explain: (1) it gives
    probabilities, (2) it's interpretable via coefficients, (3) it benchmarks whether
    your feature engineering has any signal at all before you add model complexity.

TEST
    tests/test_models.py::test_baseline_pipeline_predicts_probabilities
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_baseline(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Return an unfitted sklearn Pipeline: preprocessor + LogisticRegression.

    TODO(lillian):
      - Use ColumnTransformer to apply StandardScaler to numeric columns and
        OneHotEncoder(handle_unknown='ignore') to categorical columns.
      - Wrap preprocessor + LogisticRegression into a Pipeline and return it.
      - Use class_weight='balanced' on LogisticRegression to handle imbalance.
      - Think about why Pipeline prevents leakage structurally.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_preprocess", StandardScaler(), numeric),
            ("cat_preprocess", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )
    
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ]
    )
