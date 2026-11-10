"""Business validation regression tests; require no Great Expectations install."""

from __future__ import annotations

import pandas as pd
import pytest

from src.olist_inference.data_quality import assess_dataframe, numeric_bounds_violation


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order_purchase_timestamp": "2018-06-01 12:00:00",
                "order_estimated_delivery_date": "2018-06-12 12:00:00",
                "item_count": 1,
                "total_item_price": 100.0,
                "total_payment_value": 105.0,
                "payment_record_count": 1,
                "payment_type_count": 1,
                "payment_types": "credit_card",
                "max_payment_installments": 1,
                "customer_lat": -23.5,
                "customer_lng": -46.6,
                "avg_seller_lat": -23.0,
                "avg_seller_lng": -46.0,
            }
        ]
    )


def test_valid_data_passes() -> None:
    assert assess_dataframe(valid_frame()).passed


def test_missing_geocodes_are_allowed_and_reported() -> None:
    frame = valid_frame()
    frame.loc[0, "customer_lat"] = float("nan")
    frame.loc[0, "avg_seller_lng"] = float("nan")
    result = assess_dataframe(frame)
    assert result.passed
    assert result.geocode_missing_percent["customer_lat"] == 100.0


def test_bad_item_count_fails() -> None:
    frame = valid_frame()
    frame.loc[0, "item_count"] = 0
    assert not assess_dataframe(frame).passed


def test_negative_values_fail() -> None:
    frame = valid_frame()
    frame.loc[0, "total_item_price"] = -1
    assert not assess_dataframe(frame).passed


def test_impossible_coordinates_fail() -> None:
    frame = valid_frame()
    frame.loc[0, "avg_seller_lat"] = 100
    assert not assess_dataframe(frame).passed


def test_estimated_delivery_before_purchase_fails() -> None:
    frame = valid_frame()
    frame.loc[0, "order_estimated_delivery_date"] = "2018-05-31 12:00:00"
    assert not assess_dataframe(frame).passed


def test_missing_critical_column_fails() -> None:
    assert not assess_dataframe(valid_frame().drop(columns=["item_count"])).passed


def test_invalid_date_fails() -> None:
    frame = valid_frame()
    frame.loc[0, "order_purchase_timestamp"] = "not a date"
    assert not assess_dataframe(frame).passed


def test_empty_dataset_fails() -> None:
    assert not assess_dataframe(valid_frame().iloc[:0]).passed


def test_report_contains_only_aggregates() -> None:
    report = assess_dataframe(valid_frame()).to_report()
    assert "rows" in report
    assert "payment_types" not in str(report)


@pytest.mark.parametrize(
    ("field", "boundary", "invalid"),
    [
        ("item_count", 1, 0.5),
        ("total_item_price", 0, -0.01),
        ("total_payment_value", 0, -0.01),
        ("customer_lat", -90, -90.01),
        ("customer_lat", 90, 90.01),
        ("avg_seller_lat", -90, -90.01),
        ("avg_seller_lat", 90, 90.01),
        ("customer_lng", -180, -180.01),
        ("customer_lng", 180, 180.01),
        ("avg_seller_lng", -180, -180.01),
        ("avg_seller_lng", 180, 180.01),
    ],
)
def test_scalar_and_dataset_numeric_bounds_agree(
    field: str, boundary: float, invalid: float
) -> None:
    frame = valid_frame()
    frame[field] = boundary
    assert numeric_bounds_violation(field, boundary) is None
    assert assess_dataframe(frame).passed
    frame[field] = invalid
    assert field in numeric_bounds_violation(field, invalid)
    assert not assess_dataframe(frame).passed
