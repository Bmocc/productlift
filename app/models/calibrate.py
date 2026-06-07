"""Calibration utilities.  ⛏️  EXERCISE — Milestone 3.

LEARNING GOAL
    A model's predicted probability p=0.7 should mean "70% of these cases are
    actually positive." Most models are NOT calibrated by default: GBDTs tend to
    push probabilities toward 0 and 1 (overconfident), logistic regression is
    usually well-calibrated. Calibration matters for business: if you tell a
    product manager "this product has a 70% chance of underperforming" and you're
    wrong by 30 percentage points, your downstream decisions will be wrong too.

KEY CONCEPT — reliability diagram (calibration curve)
    Bin predictions into intervals (e.g. [0.0, 0.1), [0.1, 0.2), ...). For each bin:
      - x-axis: mean predicted probability in that bin
      - y-axis: observed positive rate in that bin
    A perfectly calibrated model plots on the diagonal y=x. The area between the
    curve and the diagonal is what ECE measures.

INTERVIEW ANGLE
    "How do you calibrate a model?" — two methods:
      1. Platt scaling: fit a logistic regression on top of the model's scores.
      2. Isotonic regression: a non-parametric monotonic fit (better for GBDTs).
    sklearn has CalibratedClassifierCV(method='isotonic') that wraps either.
    Always calibrate on HELD-OUT data (validation set), not the training set.

TEST
    tests/test_models.py::test_reliability_table_on_perfect_calibration
"""

from __future__ import annotations

import numpy as np


def reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[tuple[float, float]]:
    """Return a reliability table for drawing a calibration curve.

    TODO(lillian):
      - Bin [0, 1] into n_bins equal-width intervals using np.linspace.
      - For each bin, find all predictions that fall in [lo, hi).
      - If the bin is empty → append (nan, nan).
      - Otherwise → append (mean(y_prob in bin), mean(y_true in bin)).
      - Return the list of (mean_pred, observed_freq) tuples.
      The test feeds a perfectly-calibrated predictor and checks that
      mean_pred ≈ observed within each bin (within 0.05).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    reliability = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi)
        if in_bin.sum() == 0:
            reliability.append((np.nan, np.nan))
        else:
            mean_pred = y_prob[in_bin].mean()
            observed_freq = y_true[in_bin].mean()
            reliability.append((mean_pred, observed_freq))
    return reliability

