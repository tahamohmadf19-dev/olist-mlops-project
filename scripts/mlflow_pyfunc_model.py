"""MLflow model-from-code adapter for the trained Olist inference service.

The classifier, fitted preprocessor and threshold are supplied as immutable
MLflow artifacts. No training or fitting occurs here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import set_model

from src.olist_inference.config import InferenceConfig
from src.olist_inference.service import InferenceService


class OlistInferenceModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        required = ("preprocessor", "classifier", "summary")
        missing = set(required) - set(context.artifacts)
        if missing:
            raise RuntimeError(f"Missing model bundle artifacts: {sorted(missing)}")

        config = InferenceConfig(
            preprocessor_path=Path(context.artifacts["preprocessor"]),
            model_path=Path(context.artifacts["classifier"]),
            summary_path=Path(context.artifacts["summary"]),
            max_batch_size=100,
        )
        self.service = InferenceService(config)

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Model input must be a pandas DataFrame")
        if model_input.empty:
            raise ValueError("Model input must contain at least one order")

        # Use the same JSON representation exercised by the local REST adapter.
        # pandas serializes missing values as JSON null, which the service imputes.
        orders = json.loads(
            model_input.to_json(
                orient="records", double_precision=15, date_format="iso"
            )
        )
        result = self.service.predict(orders)
        return pd.DataFrame(
            result,
            columns=["prediction", "late_probability", "model_version"],
        )


set_model(OlistInferenceModel())
