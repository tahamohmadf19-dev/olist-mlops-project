"""Read-only parity check for the Dockerized inference API on localhost:8001."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from src.olist_inference.config import InferenceConfig
from src.olist_inference.service import InferenceService


def _check_predictions(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if actual["prediction"] != expected["prediction"]:
        raise AssertionError("Predicted class differs from local model")
    if actual["model_version"] != expected["model_version"]:
        raise AssertionError("Model version differs from local model")
    if not math.isclose(
        actual["late_probability"],
        expected["late_probability"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise AssertionError("Predicted probability differs from local model")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    service = InferenceService(
        InferenceConfig.from_json(root / "config" / "inference.json")
    )
    validation_path = root / "artifacts" / "03_validation.csv"
    sample = pd.read_csv(validation_path, nrows=5)
    if len(sample) != 5:
        raise RuntimeError("Expected at least five saved validation orders")

    columns = [c for c in sample.columns if c in service.allowed_fields]
    orders = json.loads(sample[columns].to_json(orient="records", double_precision=15))
    expected = service.predict(orders)

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json() == {"status": "ok"}, "Unexpected health response"
        print("CONTAINER HEALTH: PASS")

        model = client.get("/model")
        model.raise_for_status()
        metadata = model.json()
        assert metadata["model_version"] == service.version
        assert metadata["input_feature_count"] == len(service.expected_features)
        assert math.isclose(
            metadata["decision_threshold"], service.threshold, abs_tol=1e-12
        )
        print("CONTAINER MODEL INFO: PASS")

        single = client.post("/predict", json={"order": orders[0]})
        single.raise_for_status()
        _check_predictions(single.json(), expected[0])
        print("CONTAINER SINGLE PARITY: PASS")

        batch = client.post("/predict/batch", json={"orders": orders})
        batch.raise_for_status()
        results = batch.json()["predictions"]
        assert len(results) == len(expected)
        for actual, reference in zip(results, expected, strict=True):
            _check_predictions(actual, reference)
        print("CONTAINER BATCH PARITY: PASS")

        rejected = client.post("/predict", json={"order": {"is_late": 1}})
        assert rejected.status_code == 422, rejected.text
        print("CONTAINER TARGET LEAKAGE REJECTION: PASS")

    np.testing.assert_array_equal(
        [result["prediction"] for result in results],
        [result["prediction"] for result in expected],
    )
    print("DOCKER INFERENCE INTEGRATION: PASS")
    print("Orders checked:", len(results))
    print("Model version:", service.version)
    print("Training performed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
