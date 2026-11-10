"""HTTP integration tests using stubs and synthetic fitted artifacts only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_service import (
    INVALID_SEMANTIC_VALUES,
    VALID_SEMANTIC_VALUES,
    _sample_order,
    service as service,
)

from app.main import create_app
from src.olist_inference.service import (
    ArtifactValidationError,
    InferenceService,
    InputValidationError,
)


@dataclass
class FakeService:
    version: str = "sha256:test"
    threshold: float = 0.34
    expected_features: tuple[str, ...] = ("purchase_hour", "item_count")
    failure: str | None = None

    def predict(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.failure == "input":
            raise InputValidationError("Bad input fixture")
        if self.failure == "artifact":
            raise ArtifactValidationError("Secret artifact path")
        if len(orders) > 3:
            raise InputValidationError("Batch too large")
        if any("is_late" in order for order in orders):
            raise InputValidationError("Future/target fields prohibited")
        return [
            {
                "prediction": 0,
                "late_probability": 0.1,
                "model_version": self.version,
            }
            for _ in orders
        ]


@pytest.fixture()
def fake_service() -> FakeService:
    return FakeService()


@pytest.fixture()
def client(fake_service: FakeService):
    with TestClient(create_app(service=fake_service)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info(client: TestClient) -> None:
    response = client.get("/model")
    assert response.status_code == 200
    assert response.json() == {
        "model_version": "sha256:test",
        "decision_threshold": 0.34,
        "input_feature_count": 2,
    }


def test_single_prediction(client: TestClient) -> None:
    response = client.post("/predict", json={"order": {"item_count": 1}})
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 0,
        "late_probability": 0.1,
        "model_version": "sha256:test",
    }


def test_batch_prediction(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"orders": [{}, {}]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_bad_json_contract(client: TestClient) -> None:
    assert client.post("/predict", json={"wrong": {}}).status_code == 422
    assert client.post("/predict", json={"order": []}).status_code == 422
    assert client.post("/predict/batch", json={"orders": []}).status_code == 422
    assert client.post("/predict", json={"order": {}, "extra": 1}).status_code == 422


def test_domain_validation_is_422(client: TestClient) -> None:
    response = client.post("/predict", json={"order": {"is_late": 1}})
    assert response.status_code == 422
    assert "Future/target" in response.json()["detail"]


def test_backend_failure_is_generic_503(
    client: TestClient, fake_service: FakeService
) -> None:
    fake_service.failure = "artifact"
    response = client.post("/predict", json={"order": {}})
    assert response.status_code == 503
    assert "Secret artifact path" not in str(response.json())


def test_input_failure_is_422(client: TestClient, fake_service: FakeService) -> None:
    fake_service.failure = "input"
    response = client.post("/predict", json={"order": {}})
    assert response.status_code == 422
    assert response.json()["detail"] == "Bad input fixture"


def test_openapi_has_required_routes(client: TestClient) -> None:
    document = client.get("/openapi.json")
    assert document.status_code == 200
    routes = document.json()["paths"]
    assert {"/health", "/model", "/predict", "/predict/batch"} <= set(routes)


def test_batch_limit(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"orders": [{}] * 4})
    assert response.status_code == 422


@pytest.fixture()
def inference_client(service: InferenceService):
    with TestClient(create_app(service=service)) as test_client:
        yield test_client


@pytest.mark.parametrize(("field", "value"), INVALID_SEMANTIC_VALUES)
def test_semantic_single_request_is_422(
    inference_client: TestClient, field: str, value: float
) -> None:
    order = _sample_order(8)
    order[field] = value
    response = inference_client.post("/predict", json={"order": order})
    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert field in response.json()["detail"]


@pytest.mark.parametrize(("field", "value"), INVALID_SEMANTIC_VALUES)
@pytest.mark.parametrize("invalid_position", [0, 1])
def test_semantic_mixed_batch_is_entirely_rejected(
    inference_client: TestClient, field: str, value: float, invalid_position: int
) -> None:
    bad = _sample_order(8)
    bad[field] = value
    orders = [_sample_order(9)]
    orders.insert(invalid_position, bad)
    response = inference_client.post("/predict/batch", json={"orders": orders})
    assert response.status_code == 422
    assert set(response.json()) == {"detail"}
    assert field in response.json()["detail"]


@pytest.mark.parametrize(("field", "value"), VALID_SEMANTIC_VALUES)
def test_semantic_boundaries_return_valid_single_and_batch_predictions(
    inference_client: TestClient, field: str, value: float
) -> None:
    order = _sample_order(8)
    order[field] = value
    single = inference_client.post("/predict", json={"order": order})
    batch = inference_client.post("/predict/batch", json={"orders": [order, order]})
    assert single.status_code == 200
    assert single.json()["late_probability"] == pytest.approx(0.5)
    assert single.json()["prediction"] == 1
    assert batch.status_code == 200
    assert batch.json() == {"predictions": [single.json(), single.json()]}


def test_optional_null_partial_geo_and_unknown_category_remain_valid(
    inference_client: TestClient,
) -> None:
    order = _sample_order(8)
    order.pop("item_count")
    order.pop("avg_seller_lng")
    order["total_payment_value"] = None
    order["customer_lat"] = None
    order["payment_types"] = "previously_unseen_payment_type"
    single = inference_client.post("/predict", json={"order": order})
    batch = inference_client.post(
        "/predict/batch", json={"orders": [_sample_order(9), order]}
    )
    assert single.status_code == 200
    assert batch.status_code == 200
    assert batch.json()["predictions"] == [single.json(), single.json()]
