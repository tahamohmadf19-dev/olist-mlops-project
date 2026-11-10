"""Privacy and integration tests for per-request aggregate JSON event logging."""

from __future__ import annotations

import json
import io
import logging
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.prediction_audit import AUDIT_LOGGER, make_prediction_event
from src.olist_inference.service import InputValidationError


@dataclass
class FakeService:
    version: str = "sha256:test"
    threshold: float = 0.34
    expected_features: tuple[str, ...] = ("item_count",)

    def predict(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if any("is_late" in o for o in orders):
            raise InputValidationError("Forbidden target")
        return [
            {
                "prediction": int(i % 2 == 0),
                "late_probability": 0.8 if i % 2 == 0 else 0.2,
                "model_version": self.version,
            }
            for i, _ in enumerate(orders)
        ]


def test_summary_contains_only_allowlisted_aggregate_fields() -> None:
    event = make_prediction_event(
        [
            {"prediction": 1, "late_probability": 0.9, "email": "hidden@example.com"},
            {"prediction": 0, "late_probability": 0.1, "order_id": "secret-123"},
        ],
        model_version="sha256:test",
        inference_latency_ms=12.3456,
    )
    assert event["batch_size"] == 2
    assert "positive_predictions" not in event
    assert "mean_late_probability" not in event
    assert event["inference_latency_ms"] == 12.346
    assert set(event) == {
        "event",
        "schema_version",
        "timestamp_utc",
        "model_version",
        "batch_size",
        "inference_latency_ms",
    }
    assert "hidden@example.com" not in json.dumps(event)
    assert "secret-123" not in json.dumps(event)


def test_prediction_endpoints_emit_aggregate_logs_without_raw_orders() -> None:
    buffer = io.StringIO()
    capture = logging.StreamHandler(buffer)
    AUDIT_LOGGER.addHandler(capture)
    try:
        with TestClient(create_app(service=FakeService())) as client:
            resp = client.post(
                "/predict",
                json={"order": {"item_count": 2, "order_id": "SECRET-ORDER-ID"}},
            )
            assert resp.status_code == 200
            resp = client.post(
                "/predict/batch",
                json={"orders": [{"private_email": "private@example.com"}, {}]},
            )
            assert resp.status_code == 200
            bad = client.post("/predict", json={"order": {"is_late": 1}})
            assert bad.status_code == 422

    finally:
        AUDIT_LOGGER.removeHandler(capture)
    messages = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert len(messages) == 2
    assert [json.loads(msg)["batch_size"] for msg in messages] == [1, 2]
    assert all("positive_predictions" not in json.loads(msg) for msg in messages)
    assert "SECRET-ORDER-ID" not in "\n".join(messages)
    assert "private@example.com" not in "\n".join(messages)
    assert "is_late" not in "\n".join(messages)


def test_invalid_audit_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_prediction_event([], model_version="v", inference_latency_ms=0.1)
    with pytest.raises(ValueError):
        make_prediction_event(
            [{"prediction": 1, "late_probability": 1.2}],
            model_version="v",
            inference_latency_ms=1,
        )
    with pytest.raises(ValueError):
        make_prediction_event(
            [{"prediction": 0, "late_probability": 0.4}],
            model_version="v",
            inference_latency_ms=-1,
        )


def test_large_batch_has_low_precision_aggregate() -> None:
    event = make_prediction_event(
        [
            {
                "prediction": int(i % 2 == 0),
                "late_probability": 0.8 if i % 2 == 0 else 0.2,
            }
            for i in range(20)
        ],
        model_version="sha256:test",
        inference_latency_ms=4.0,
    )
    assert event["positive_predictions"] == 10
    assert event["mean_late_probability"] == 0.5
