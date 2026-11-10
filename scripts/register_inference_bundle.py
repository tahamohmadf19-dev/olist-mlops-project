"""Package, round-trip validate, then register the existing Olist inference model.

Local MLflow SQLite registry only; no training, networking, or deployment.
"""

from __future__ import annotations

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.olist_inference.config import InferenceConfig
from src.olist_inference.service import InferenceService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "olist_late_delivery_inference"
MODEL_CODE = PROJECT_ROOT / "scripts" / "mlflow_pyfunc_model.py"
EXPECTED_FINGERPRINT = "a2485e81e9f4"
HISTORICAL_RUN_ID = "9a69faa4fb7c4740af802ce3af2a2e39"


def main() -> int:
    if not MODEL_CODE.is_file():
        raise FileNotFoundError(MODEL_CODE)
    config = InferenceConfig.from_json(PROJECT_ROOT / "config" / "inference.json")
    service = InferenceService(config)
    sha = hashlib.sha256(config.model_path.read_bytes()).hexdigest()
    if (
        sha[:12] != EXPECTED_FINGERPRINT
        or service.version != f"sha256:{EXPECTED_FINGERPRINT}"
    ):
        raise RuntimeError(
            "Original trained model fingerprint differs from verified version"
        )

    source = PROJECT_ROOT / "artifacts" / "03_validation.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    validation = pd.read_csv(source, nrows=5, low_memory=False)
    input_columns = [
        column for column in validation if column in service.allowed_fields
    ]
    input_df = validation[input_columns].copy()
    orders: list[dict[str, Any]] = json.loads(
        input_df.to_json(orient="records", double_precision=15, date_format="iso")
    )
    expected = service.predict(orders)

    for path in (config.model_path, config.preprocessor_path, config.summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    requirements = [
        f"{name}=={version(name)}"
        for name in ("mlflow", "scikit-learn", "pandas", "numpy", "scipy", "joblib")
    ]
    artifacts = {
        "preprocessor": str(config.preprocessor_path),
        "classifier": str(config.model_path),
        "summary": str(config.summary_path),
    }

    mlflow.set_tracking_uri(
        f"sqlite:///{(PROJECT_ROOT / 'mlflow_store' / 'tracking.db').as_posix()}"
    )
    mlflow.set_experiment("olist-late-delivery-inference")
    print("PACKAGING ORIGINAL FITTED INFERENCE BUNDLE")

    with mlflow.start_run(run_name="task3-pyfunc-inference-bundle") as run:
        mlflow.set_tags(
            {
                "run_type": "inference_bundle_packaging",
                "training_performed": "false",
                "model_version": service.version,
                "historical_import_run_id": HISTORICAL_RUN_ID,
            }
        )
        model_info = mlflow.pyfunc.log_model(
            name="inference_bundle",
            python_model=str(MODEL_CODE),
            code_paths=[str(PROJECT_ROOT / "src")],
            artifacts=artifacts,
            pip_requirements=requirements,
            metadata={
                "training_performed": "false",
                "model_sha256": sha,
                "decision_threshold": float(service.threshold),
                "input_feature_count": len(service.expected_features),
            },
        )
        print("MLFLOW PACKAGE: CREATED")
        print("Logged model URI:", model_info.model_uri)

        # Load the packaged model, not the project's live service, to test its
        # copied code and artifact bundle together.
        loaded = mlflow.pyfunc.load_model(model_info.model_uri)
        actual = loaded.predict(input_df)
        if not isinstance(actual, pd.DataFrame) or len(actual) != len(expected):
            raise AssertionError(
                "Packaged model returned an unexpected prediction shape"
            )
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
        if actual["model_version"].tolist() != [
            row["model_version"] for row in expected
        ]:
            raise AssertionError("Packaged model version mismatch")
        if not math.isclose(
            service.threshold, 0.3389565594999209, rel_tol=0, abs_tol=1e-12
        ):
            raise AssertionError(
                "Saved decision threshold differs from verified service"
            )
        print("PACKAGED MODEL PARITY: PASS")
        print("Orders checked:", len(expected))

        # Registration only happens after successful round-trip parity checks.
        registration = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MODEL_NAME,
            await_registration_for=180,
        )
        client = MlflowClient()
        client.set_model_version_tag(
            MODEL_NAME, registration.version, "model_sha256", sha
        )
        client.set_model_version_tag(
            MODEL_NAME, registration.version, "training_performed", "false"
        )
        print("MODEL REGISTRY: PASS")
        print("Registered model:", MODEL_NAME)
        print("Registry version:", registration.version)
        print("Run ID:", run.info.run_id)
        print("Training performed: NO")
        print("Deployment performed: NO")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
