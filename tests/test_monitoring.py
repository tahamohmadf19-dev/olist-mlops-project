"""Monitoring contract: bounded labels, no raw identifiers, proper HTTP status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from app.main import create_app
from src.olist_inference.service import ArtifactValidationError, InputValidationError


@dataclass
class FakeService:
    version: str = "sha256:test"
    threshold: float = 0.34
    expected_features: tuple[str, ...] = ("purchase_hour", "item_count")

    def predict(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if any("is_late" in order for order in orders):
            raise InputValidationError("Forbidden input")
        if any("fail_backend" in order for order in orders):
            raise ArtifactValidationError("internal path secret")
        return [
            {
                "prediction": int(i % 2 == 0),
                "late_probability": 0.9 if i % 2 == 0 else 0.1,
                "model_version": self.version,
            }
            for i, _ in enumerate(orders)
        ]


def test_metrics_count_and_distribution() -> None:
    with TestClient(create_app(service=FakeService())) as client:
        assert (
            client.post("/predict", json={"order": {"item_count": 2}}).status_code
            == 200
        )
        assert (
            client.post("/predict/batch", json={"orders": [{}, {}]}).status_code == 200
        )
        assert (
            client.post("/predict", json={"order": {"is_late": 1}}).status_code == 422
        )
        assert (
            client.post("/predict", json={"order": {"fail_backend": True}}).status_code
            == 503
        )
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        body = resp.text
        assert 'olist_predictions_total{label="1"} 2.0' in body
        assert 'olist_predictions_total{label="0"} 1.0' in body
        assert (
            'olist_http_requests_total{method="POST",route="predict",status="422"} 1.0'
            in body
        )
        assert (
            'olist_http_requests_total{method="POST",route="predict",status="503"} 1.0'
            in body
        )
        assert "olist_prediction_probability_bucket" in body
        assert "olist_http_request_duration_seconds_bucket" in body
        assert "item_count" not in body
        assert "internal path secret" not in body
        assert "is_late" not in body
        assert (
            "/predict/batch" not in body
        )  # Only bounded route names, never arbitrary paths


def test_metrics_isolated_between_apps() -> None:
    with TestClient(create_app(service=FakeService())) as first:
        first.post("/predict", json={"order": {}})
        assert 'olist_predictions_total{label="1"} 1.0' in first.get("/metrics").text
    with TestClient(create_app(service=FakeService())) as second:
        assert (
            'olist_predictions_total{label="1"} 1.0' not in second.get("/metrics").text
        )


def test_metrics_not_in_public_openapi() -> None:
    with TestClient(create_app(service=FakeService())) as client:
        assert "/metrics" not in client.get("/openapi.json").json()["paths"]
