"""Deterministic feature engineering extracted from Notebook 5.

Keep feature names and transformations unchanged unless the notebook and
saved preprocessor are rebuilt together and parity tests are updated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "is_late"

LEAKAGE_COLUMNS = [
    "order_delivered_customer_date",
    "delivery_delay_days",
    "order_delivered_carrier_date",
    "order_approved_at",
    "order_status",
]

ID_COLUMNS = [
    "order_id",
    "customer_id",
    "customer_unique_id",
]

# ZIP prefix is a location identifier, not a continuous number.
# We already have state + latitude/longitude, so raw ZIP is dropped
# to avoid unnecessary high-cardinality encoding.
DROP_LOCATION_IDENTIFIER_COLUMNS = [
    "customer_zip_code_prefix",
]


# ============================================================
# 4. FEATURE ENGINEERING FUNCTIONS
# ============================================================

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
]


def haversine_distance_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """
    Calculate approximate great-circle distance in kilometers
    between customer and seller coordinates.
    """
    earth_radius_km = 6371.0088

    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_km * c


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build only features that are available at purchase time.
    """
    result = df.copy()

    # --------------------------------------------------------
    # Parse dates
    # --------------------------------------------------------

    for column in DATE_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column],
                format="mixed",
                errors="coerce",
            )

    # --------------------------------------------------------
    # Purchase-time features
    # --------------------------------------------------------

    purchase = result["order_purchase_timestamp"]

    result["purchase_year"] = purchase.dt.year
    result["purchase_month"] = purchase.dt.month
    result["purchase_day"] = purchase.dt.day
    result["purchase_weekday"] = purchase.dt.day_of_week
    result["purchase_hour"] = purchase.dt.hour
    result["purchase_is_weekend"] = (purchase.dt.day_of_week >= 5).astype("int8")

    # --------------------------------------------------------
    # Estimated delivery window
    # --------------------------------------------------------

    result["estimated_delivery_window_days"] = (
        result["order_estimated_delivery_date"] - result["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400.0

    # --------------------------------------------------------
    # Geographic distance
    # --------------------------------------------------------

    required_geo_columns = [
        "customer_lat",
        "customer_lng",
        "avg_seller_lat",
        "avg_seller_lng",
    ]

    if all(column in result.columns for column in required_geo_columns):
        result["customer_seller_distance_km"] = haversine_distance_km(
            result["customer_lat"],
            result["customer_lng"],
            result["avg_seller_lat"],
            result["avg_seller_lng"],
        )

    # --------------------------------------------------------
    # Log features for strongly right-skewed variables
    # --------------------------------------------------------

    skewed_columns = [
        "total_item_price",
        "avg_item_price",
        "max_item_price",
        "total_freight_value",
        "avg_product_weight_g",
        "total_payment_value",
    ]

    for column in skewed_columns:
        if column in result.columns:
            result[f"log1p_{column}"] = np.log1p(result[column].clip(lower=0))

    # --------------------------------------------------------
    # Drop raw dates after extracting useful features
    # --------------------------------------------------------

    result = result.drop(
        columns=[column for column in DATE_COLUMNS if column in result.columns],
        errors="ignore",
    )

    # --------------------------------------------------------
    # Remove leakage, IDs, and raw ZIP identifier
    # --------------------------------------------------------

    result = result.drop(
        columns=[
            *LEAKAGE_COLUMNS,
            *ID_COLUMNS,
            *DROP_LOCATION_IDENTIFIER_COLUMNS,
        ],
        errors="ignore",
    )

    return result
