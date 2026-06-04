"""Temporal features.  ⛏️  EXERCISE — Milestone 2.

LEARNING GOAL
    Capture recency, trend, and tenure — when a product sold, not just how much.
    Temporal features are where look-ahead leakage sneaks in, so they're the best
    place to internalize "as-of" thinking.

KEY CONCEPT — everything is relative to the as-of date
    Given `as_of` (the prediction timestamp), per product compute:
      - tenure_days        : as_of - first_order_date   (how established)
      - days_since_last    : as_of - last_order_date     (recency)
      - orders_last_30/90d : count in the trailing window before as_of (momentum)
      - trend              : orders in recent window vs earlier window (accelerating?)
    NONE of these may reference any timestamp after as_of.

INTERVIEW ANGLE
    Recency/tenure features are how you handle the cold-start vs established-product
    distinction, and how you'd detect a product losing momentum before revenue
    fully reflects it.

TEST
    tests/test_features.py::test_temporal_features
"""

from __future__ import annotations

import pandas as pd


def temporal_features(
    feature_events: pd.DataFrame, as_of: pd.Timestamp, time_col: str = "order_purchase_timestamp"
) -> pd.DataFrame:
    """Return one row per product of time-based features measured at `as_of`.

    TODO(lillian): per product compute at least tenure_days, days_since_last,
    orders_last_30d, orders_last_90d, and a simple trend (e.g. last_30d /
    prior_30d). Assert internally that max(time_col) <= as_of and raise if not —
    a cheap guard that catches leakage at the source. The test feeds events
    straddling as_of and checks the windows.
    """

    assert feature_events[time_col].max() <= as_of, "feature_events contains future data relative to as_of"

    t = feature_events[time_col]

    tenure_days = (as_of - feature_events.groupby("product_id")[time_col].min()).dt.days
    days_since_last = (as_of - feature_events.groupby("product_id")[time_col].max()).dt.days
    orders_last_30d = feature_events[t >= as_of - pd.Timedelta(days=30)].groupby("product_id").size()
    orders_last_90d = feature_events[t >= as_of - pd.Timedelta(days=90)].groupby("product_id").size()
    prior_30d = feature_events[(t < as_of - pd.Timedelta(days=30)) & (t >= as_of - pd.Timedelta(days=60))].groupby("product_id").size()
    trend = orders_last_30d / prior_30d.replace(0,1)

    temporal_df = pd.DataFrame({
        'tenure_days': tenure_days,
        'days_since_last': days_since_last,
        'orders_last_30d': orders_last_30d,
        'orders_last_90d': orders_last_90d,
        'trend': trend
    }).fillna(0).reset_index()

    return temporal_df



    raise NotImplementedError("Implement temporal_features — see tests/test_features.py")
