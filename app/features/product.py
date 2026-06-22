"""Product (static) features.  ⛏️  EXERCISE — Milestone 2.

LEARNING GOAL
    Engineer features from a product's intrinsic attributes — the things true of the
    listing itself, independent of time. These are available even for near-cold
    products, which is exactly why they matter for the cold-start problem the
    interview doc calls out.

KEY CONCEPT — content features & sensible transforms
    Raw Olist product columns (photos count, description length, weight, dimensions)
    are weak alone but predictive together. Engineer:
      - listing-quality signals: n photos, description length, name length
      - physical: weight, volume (l*w*h), density
      - price level (avg unit price) and price relative to category median
    Watch for skew — physical sizes and prices are heavy-tailed; a log transform
    is usually the right move (note it for the modeling step).

INTERVIEW ANGLE
    "What features would you engineer?" → lead with content/listing-quality features
    because they solve cold start, then explain the category-relative price as a way
    to compare like with like.

TEST
    tests/test_features.py::test_product_features
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def product_features(feature_events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per product of static/listing features.

    `feature_events` is the order-item table restricted to the FEATURE window.

    TODO(lillian): build at least:
      - product_id (key)
      - n_photos, description_length, name_length  (listing quality)
      - weight_g, volume_cm3 (= length*height*width), and a log version of skewed ones
      - avg_unit_price (mean price) and price_vs_category_median (ratio)
    Return product-level, one row per product_id. The test checks the volume and the
    category-relative price computation on a small fixture.
    """
    features_product = feature_events[["product_id", "product_category_name", "product_weight_g", 
                                       "product_length_cm", "product_height_cm", "product_width_cm", 
                                       "product_description_lenght", "product_name_lenght", "product_photos_qty"]].drop_duplicates("product_id").copy()
    features_product["volume_cm3"] = (features_product["product_length_cm"] * 
                                      features_product["product_height_cm"] * 
                                      features_product["product_width_cm"])
    features_product['log_weight_g'] = features_product['product_weight_g'].apply(lambda x: np.log(x) if x > 0 else 0)
    features_product['log_volume_cm3'] = features_product['volume_cm3'].apply(lambda x: np.log(x) if x > 0 else 0)

    avg_unit_price = feature_events.groupby("product_id")['price'].mean()
    features_product["avg_unit_price"] = features_product["product_id"].map(avg_unit_price)

    category_median_price = features_product.groupby("product_category_name")['avg_unit_price'].median()
    features_product["price_vs_category_median"] = (
        features_product["avg_unit_price"]
        / features_product["product_category_name"].map(category_median_price)
    )

    return features_product

