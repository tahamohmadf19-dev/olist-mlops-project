"""Register the verified fitted inference bundle with an explicit MLflow signature.

Reads existing artifacts and validation rows. Does not fit models, mutate v1,
change the API, publish, or deploy. Will not create v3+ on repeated execution.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from src.olist_inference.config import InferenceConfig
from src.olist_inference.features import DATE_COLUMNS, ID_COLUMNS
from src.olist_inference.service import InferenceService

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "olist_late_delivery_inference"
EXPECTED_SHA_PREFIX = "a2485e81e9f4"


def assert_parity(
    model: Any, frame: pd.DataFrame, expected: list[dict[str, Any]]
) -> None:
    """Compare three output fields, including the model fingerprint."""
    actual = model.predict(frame)
    if not isinstance(actual, pd.DataFrame) or len(actual) != len(expected):
        raise AssertionError("Unexpected prediction shape")
    np.testing.assert_array_equal(
        actual["prediction"].to_numpy(),
        np.asarray([row["prediction"] for row in expected]),
    )
    np.testing.assert_allclose(
        actual["late_probability"].to_numpy(dtype=float),
        np.asarray([row["late_probability"] for row in expected]),
        rtol=1e-9,
        atol=1e-9,
    )
    assert actual["model_version"].tolist() == [
        row["model_version"] for row in expected
    ], "Model fingerprint mismatch"


def build_signature(service: InferenceService) -> ModelSignature:
    """Represent 30 raw request fields, not the 39 engineered or 821 encoded fields."""
    dates = set(DATE_COLUMNS)
    ids = set(ID_COLUMNS)
    numeric = set(service.raw_features) & service.numeric_features
    categorical = set(service.raw_features) & service.categorical_features
    actual = dates | ids | numeric | categorical
    if actual != service.allowed_fields or len(actual) != 30:
        raise RuntimeError("Unexpected raw input contract; refuse to register")
    return ModelSignature(
        inputs=Schema(
            [ColSpec("string", name, required=True) for name in DATE_COLUMNS]
            + [ColSpec("double", name, required=False) for name in sorted(numeric)]
            + [ColSpec("string", name, required=False) for name in sorted(categorical)]
            + [ColSpec("string", name, required=False) for name in ID_COLUMNS]
        ),
        outputs=Schema(
            [
                ColSpec("long", "prediction"),
                ColSpec("double", "late_probability"),
                ColSpec("string", "model_version"),
            ]
        ),
    )


def main() -> int:
    config = InferenceConfig.from_json(ROOT / "config" / "inference.json")
    service = InferenceService(config)
    model_sha = hashlib.sha256(config.model_path.read_bytes()).hexdigest()
    if model_sha[:12] != EXPECTED_SHA_PREFIX:
        raise RuntimeError("Original model SHA256 changed; registration blocked")
    signature = build_signature(service)

    source = ROOT / "artifacts" / "03_validation.csv"
    frame = pd.read_csv(source, nrows=5, low_memory=False)
    columns = [name for name in frame.columns if name in service.allowed_fields]
    inputs = frame[columns].copy()
    orders = json.loads(
        inputs.to_json(orient="records", double_precision=15, date_format="iso")
    )
    expected = service.predict(orders)

    mlflow.set_tracking_uri(
        f"sqlite:///{(ROOT / 'mlflow_store' / 'tracking.db').as_posix()}"
    )
    client = MlflowClient()
    versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
    if (
        len(versions) != 1
        or str(versions[0].version) != "1"
        or versions[0].status != "READY"
    ):
        raise RuntimeError(
            "Expected only READY Registry v1. Refusing duplicate registration"
        )
    v1 = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/1")
    if v1.metadata.signature is not None:
        raise RuntimeError(
            "Registry v1 already has a signature; inspect before upgrading"
        )
    assert_parity(v1, inputs, expected)
    print("REGISTRY V1 PARITY: PASS")

    for path in (config.model_path, config.preprocessor_path, config.summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    py_model_code = ROOT / "scripts" / "mlflow_pyfunc_model.py"
    if not py_model_code.is_file():
        raise FileNotFoundError(py_model_code)

    requirements = [
        f"{package}=={version(package)}"
        for package in ("mlflow", "scikit-learn", "pandas", "numpy", "scipy", "joblib")
    ]
    mlflow.set_experiment("olist-late-delivery-inference")
    with mlflow.start_run(run_name="task3-pyfunc-signed-inference-bundle") as run:
        mlflow.set_tags(
            {
                "run_type": "signed_inference_bundle",
                "training_performed": "false",
                "model_sha256": model_sha,
                "supersedes_registry_version": "1",
            }
        )
        info = mlflow.pyfunc.log_model(
            name="signed_inference_bundle",
            python_model=str(py_model_code),
            code_paths=[str(ROOT / "src")],
            artifacts={
                "preprocessor": str(config.preprocessor_path),
                "classifier": str(config.model_path),
                "summary": str(config.summary_path),
            },
            signature=signature,
            pip_requirements=requirements,
            metadata={
                "training_performed": "false",
                "model_sha256": model_sha,
                "decision_threshold": float(service.threshold),
                "input_feature_count": len(service.expected_features),
                "signature_contract": "2_required_dates_28_optional_raw_fields",
            },
        )
        loaded = mlflow.pyfunc.load_model(info.model_uri)
        if loaded.metadata.signature is None:
            raise AssertionError("Signed model lost signature")
        assert_parity(loaded, inputs, expected)
        print("SIGNED PACKAGE PARITY: PASS")
        print("Orders checked:", len(expected))
        print("Signature input fields:", len(loaded.metadata.signature.inputs.inputs))
        print("Signature output fields:", len(loaded.metadata.signature.outputs.inputs))

        registered = mlflow.register_model(
            model_uri=info.model_uri,
            name=MODEL_NAME,
            await_registration_for=180,
        )
        if str(registered.version) != "2":
            raise RuntimeError(f"Unexpected registry version {registered.version}")
        client.set_model_version_tag(
            MODEL_NAME, registered.version, "model_sha256", model_sha
        )
        client.set_model_version_tag(
            MODEL_NAME, registered.version, "training_performed", "false"
        )
        v2 = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/2")
        if v2.metadata.signature is None:
            raise AssertionError("Registry v2 signature is missing")
        assert_parity(v2, inputs, expected)
        print("REGISTERED V2 PARITY: PASS")
        print("MODEL REGISTRY V2: PASS")
        print("Run ID:", run.info.run_id)
        print("Training performed: NO")
        print("Deployment performed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
