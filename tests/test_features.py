from __future__ import annotations

import numpy as np
import pandas as pd

from src.olist_inference.features import engineer_features


def test_engineer_features_preserves_notebook_logic() -> None:
    row = {
        "order_purchase_timestamp": "2018-06-09 10:00:00",
        "order_estimated_delivery_date": "2018-06-19 10:00:00",
        "total_item_price": 9.0,
        "avg_item_price": 9.0,
        "max_item_price": 9.0,
        "total_freight_value": 2.0,
        "avg_product_weight_g": 100.0,
        "total_payment_value": 11.0,
        "customer_lat": -23.0,
        "customer_lng": -46.0,
        "avg_seller_lat": -23.0,
        "avg_seller_lng": -46.0,
        "order_delivered_customer_date": "2018-06-18 10:00:00",
        "is_late": 0,
    }
    result = engineer_features(pd.DataFrame([row]))
    assert result.loc[0, "purchase_weekday"] == 5
    assert result.loc[0, "purchase_is_weekend"] == 1
    assert result.loc[0, "estimated_delivery_window_days"] == 10.0
    assert np.isclose(result.loc[0, "log1p_total_item_price"], np.log1p(9.0))
    assert np.isclose(result.loc[0, "customer_seller_distance_km"], 0.0)
    assert "order_delivered_customer_date" not in result.columns
