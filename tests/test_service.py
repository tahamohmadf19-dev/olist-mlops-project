from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.olist_inference.config import InferenceConfig
from src.olist_inference.features import engineer_features
from src.olist_inference.service import InferenceService, InputValidationError


INVALID_SEMANTIC_VALUES = [
    ("item_count", 0),
    ("item_count", 0.5),
    ("total_item_price", -0.01),
    ("total_payment_value", -0.01),
    ("customer_lat", -90.01),
    ("customer_lat", 90.01),
    ("avg_seller_lat", -90.01),
    ("avg_seller_lat", 90.01),
    ("customer_lng", -180.01),
    ("customer_lng", 180.01),
    ("avg_seller_lng", -180.01),
    ("avg_seller_lng", 180.01),
]
VALID_SEMANTIC_VALUES = [
    ("item_count", 1),
    ("item_count", 1.5),
    ("total_item_price", 0),
    ("total_payment_value", 0),
    ("customer_lat", -90),
    ("customer_lat", 90),
    ("avg_seller_lat", -90),
    ("avg_seller_lat", 90),
    ("customer_lng", -180),
    ("customer_lng", 180),
    ("avg_seller_lng", -180),
    ("avg_seller_lng", 180),
]


def _sample_order(hour: int) -> dict:
    return {
        "order_purchase_timestamp": f"2018-06-09 {hour:02d}:00:00",
        "order_estimated_delivery_date": "2018-06-19 18:00:00",
        "customer_lat": -23.0,
        "customer_lng": -46.0,
        "avg_seller_lat": -23.1,
        "avg_seller_lng": -46.1,
        "total_item_price": 20.0,
        "avg_item_price": 20.0,
        "max_item_price": 20.0,
        "total_freight_value": 4.0,
        "avg_product_weight_g": 100.0,
        "total_payment_value": 24.0,
        "item_count": 1,
        "payment_types": "credit_card",
    }


@pytest.fixture()
def service(tmp_path: Path) -> InferenceService:
    orders = [_sample_order(8), _sample_order(9), _sample_order(10), _sample_order(11)]
    orders[1]["payment_types"] = "voucher"
    orders[2]["total_item_price"] = 30.0
    orders[3]["payment_types"] = "voucher"
    engineered = engineer_features(pd.DataFrame(orders))
    numeric = engineered.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = engineered.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    prep = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="unknown"),
                        ),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    x = prep.fit_transform(engineered)
    model = DummyClassifier(strategy="prior")
    model.fit(x, [0, 1, 0, 1])
    (tmp_path / "artifacts").mkdir()
    joblib.dump(prep, tmp_path / "artifacts" / "prep.joblib")
    joblib.dump(model, tmp_path / "artifacts" / "model.joblib")
    (tmp_path / "artifacts" / "summary.json").write_text(
        json.dumps({"selected_threshold": 0.3})
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "inference.json").write_text(
        json.dumps(
            {
                "preprocessor_path": "artifacts/prep.joblib",
                "model_path": "artifacts/model.joblib",
                "summary_path": "artifacts/summary.json",
                "max_batch_size": 100,
            }
        )
    )
    return InferenceService(
        InferenceConfig.from_json(tmp_path / "config" / "inference.json")
    )


def test_predict_returns_probability_version_and_label(
    service: InferenceService,
) -> None:
    output = service.predict([_sample_order(8)])
    assert output[0]["prediction"] == 1  # 0.5 >= threshold 0.3
    assert output[0]["late_probability"] == pytest.approx(0.5)
    assert output[0]["model_version"].startswith("sha256:")


def test_rejects_future_or_target_data(service: InferenceService) -> None:
    bad = _sample_order(8)
    bad["order_delivered_customer_date"] = "2018-06-20 01:00:00"
    with pytest.raises(InputValidationError, match="Future/target"):
        service.predict([bad])


def test_rejects_bad_numeric_value(service: InferenceService) -> None:
    bad = _sample_order(8)
    bad["item_count"] = "three"
    with pytest.raises(InputValidationError, match="numeric"):
        service.predict([bad])


def test_allows_missing_optional_features(service: InferenceService) -> None:
    order = _sample_order(8)
    order.pop("item_count")
    output = service.predict([order])
    assert len(output) == 1


def test_rejects_bad_dates(service: InferenceService) -> None:
    bad = _sample_order(8)
    bad["order_purchase_timestamp"] = "not a date"
    with pytest.raises(InputValidationError, match="Invalid timestamp"):
        service.predict([bad])


def test_allows_missing_optional_category(service: InferenceService) -> None:
    order = _sample_order(8)
    order.pop("payment_types")
    output = service.predict([order])
    assert len(output) == 1


def test_batch_predictions_and_limit(service: InferenceService) -> None:
    assert len(service.predict([_sample_order(8), _sample_order(9)])) == 2
    with pytest.raises(InputValidationError, match="Batch size"):
        service.predict([])


def test_rejects_future_estimated_date(service: InferenceService) -> None:
    bad = _sample_order(8)
    bad["order_estimated_delivery_date"] = "2018-06-08 12:00:00"
    with pytest.raises(InputValidationError, match="cannot precede"):
        service.predict([bad])


@pytest.mark.parametrize(("field", "value"), INVALID_SEMANTIC_VALUES)
@pytest.mark.parametrize("invalid_position", [None, 0, 1])
def test_rejects_semantic_values_before_any_feature_or_model_work(
    service: InferenceService, field: str, value: float, invalid_position: int | None
) -> None:
    bad = _sample_order(8)
    bad[field] = value
    orders = [bad]
    if invalid_position is not None:
        orders = [_sample_order(9)]
        orders.insert(invalid_position, bad)
    with (
        patch("src.olist_inference.service.engineer_features") as engineer,
        patch.object(service.preprocessor, "transform") as transform,
        patch.object(service.model, "predict_proba") as predict,
    ):
        with pytest.raises(InputValidationError, match=field):
            service.predict(orders)
        engineer.assert_not_called()
        transform.assert_not_called()
        predict.assert_not_called()


@pytest.mark.parametrize(("field", "value"), VALID_SEMANTIC_VALUES)
def test_semantic_boundaries_preserve_valid_predictions(
    service: InferenceService, field: str, value: float
) -> None:
    order = _sample_order(8)
    expected = service.predict([order])
    order[field] = value
    assert service.predict([order]) == expected


@pytest.mark.parametrize(
    "field",
    [
        "item_count",
        "total_item_price",
        "total_payment_value",
        "customer_lat",
        "customer_lng",
        "avg_seller_lat",
        "avg_seller_lng",
    ],
)
@pytest.mark.parametrize("omit", [False, True])
def test_preserves_missing_and_null_numeric_fields(
    service: InferenceService, field: str, omit: bool
) -> None:
    order = _sample_order(8)
    expected = service.predict([order])
    if omit:
        order.pop(field)
    else:
        order[field] = None
    assert service.predict([order]) == expected


def test_preserves_unknown_category(service: InferenceService) -> None:
    order = _sample_order(8)
    expected = service.predict([order])
    order["payment_types"] = "previously_unseen_payment_type"
    assert service.predict([order]) == expected
