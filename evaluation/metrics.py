"""Metrics. Discrimination metrics are provided; the calibration, operating-point,
and ranking metrics are  ⛏️  EXERCISES — Milestone 4.

WHY THE SPLIT
    roc_auc / pr_auc / rmse are standard library one-liners — provided so the
    harness runs. The ones you implement (Brier, ECE, recall@k, NDCG@k) are the ones
    interviewers probe and the ones the session doc stresses (calibration!). Know
    them cold; an off-by-one in a metric silently corrupts every experiment.

TEST
    tests/test_metrics.py — hand-computed expected values in the comments.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# --------------------------------------------------------------------------- #
# Provided (discrimination / error)
# --------------------------------------------------------------------------- #
def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve. Rank metric: P(score(pos) > score(neg))."""
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision). Preferred under
    class imbalance, where ROC-AUC can look deceptively good."""
    return float(average_precision_score(y_true, y_score))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# --------------------------------------------------------------------------- #
# ⛏️ EXERCISES
# --------------------------------------------------------------------------- #
def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions: mean((y_prob - y_true)^2).

    TODO(lillian): one line. Lower = better-calibrated AND sharper. The test checks
    a perfect predictor (0.0) and a hand example.
    """
    return np.mean(np.square(y_prob - y_true))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE: bin predictions, take the weighted average gap between mean predicted
    probability and observed frequency per bin.

    TODO(lillian):
      - bin y_prob into n_bins equal-width bins on [0, 1]
      - per non-empty bin: |mean(y_prob in bin) - mean(y_true in bin)|
      - weight each bin's gap by its share of points; sum.
    The test feeds a perfectly-calibrated set (ECE≈0) and a badly-calibrated one.
    This is THE metric to show alongside AUC in a model review.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    num_sample = len(y_true)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue   # placeholder to avoid empty-bin divide-by-zero; implement the actual gap computation here
        mean_prob = y_prob[mask].mean()
        mean_true = y_true[mask].mean()
        gap = abs(mean_prob - mean_true)
        ece += gap * mask.sum() / num_sample

    return ece
    

def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float) -> float:
    """Of all true positives, what share is captured in the top `k_fraction` by score?

    TODO(lillian): take the top ceil(k_fraction * n) items by score; return
    (# true positives among them) / (total true positives). This is the operating-
    point metric — "if we can only review the top 10% of flagged products, how many
    real underperformers do we catch?" Translate the model into a business decision.
    """
    k = int (np.ceil(k_fraction * len(y_true)))
    top_k_indices = np.argsort(y_score)[::-1][:k] #make sure to sort in descending order to get the top k scores
    true_positives_in_top_k = np.sum(y_true[top_k_indices])
    total_true_positives = np.sum(y_true)
    recall = true_positives_in_top_k / total_true_positives if total_true_positives > 0 else 0
    return recall


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Normalized Discounted Cumulative Gain — a ranking metric (the doc lists MAP/
    NDCG). Binary relevance: gain = y_true.

    TODO(lillian): order by y_score desc; DCG@k = Σ_{i=1..k} rel_i / log2(i+1);
    IDCG@k = DCG of the ideal ordering; return DCG/IDCG (0 if IDCG==0). Relevant for
    the ranking framing of "which products to surface/fix first".
    """
    # raise NotImplementedError
    ranked_idx = np.argsort(y_score)[::-1][:k]
    ranked_rel = y_true[ranked_idx]

    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(ranked_rel / discounts)

    ideal_rel = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal_rel / discounts)

    return float(dcg / idcg) if idcg > 0 else 0.0