from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from .config import InferenceConfig
from .data_quality import numeric_bounds_violation
from .features import (
    DATE_COLUMNS,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET,
    engineer_features,
)

logger = logging.getLogger(__name__)


class InputValidationError(ValueError):
    """Client submitted an invalid order payload."""


class ArtifactValidationError(RuntimeError):
    """The required trained artifacts are missing or incompatible."""


CREATED_FEATURES = frozenset(
    {
        "purchase_year",
        "purchase_month",
        "purchase_day",
        "purchase_weekday",
        "purchase_hour",
        "purchase_is_weekend",
        "estimated_delivery_window_days",
        "customer_seller_distance_km",
        "log1p_total_item_price",
        "log1p_avg_item_price",
        "log1p_max_item_price",
        "log1p_total_freight_value",
        "log1p_avg_product_weight_g",
        "log1p_total_payment_value",
    }
)


class InferenceService:
    """Loads previously fitted artifacts; never trains or refits at inference."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        for file in (config.preprocessor_path, config.model_path, config.summary_path):
            if not file.is_file():
                raise ArtifactValidationError(f"Required artifact is missing: {file}")

        # joblib.load executes deserialization: load only locally trusted artifacts.
        try:
            self.preprocessor = joblib.load(config.preprocessor_path)
            self.model = joblib.load(config.model_path)
            summary = json.loads(config.summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ImportError, AttributeError) as exc:
            raise ArtifactValidationError("Unable to load the model bundle") from exc

        if not isinstance(summary, dict) or "selected_threshold" not in summary:
            raise ArtifactValidationError(
                "Results summary is missing selected_threshold"
            )
        self.threshold = float(summary["selected_threshold"])
        if not math.isfinite(self.threshold) or not (0.0 < self.threshold < 1.0):
            raise ArtifactValidationError(
                "selected_threshold must be between zero and one"
            )

        if not hasattr(self.preprocessor, "feature_names_in_"):
            raise ArtifactValidationError(
                "Preprocessor has no fitted input feature schema"
            )
        self.expected_features = list(self.preprocessor.feature_names_in_)
        if len(set(self.expected_features)) != len(self.expected_features):
            raise ArtifactValidationError(
                "Preprocessor contains duplicate feature names"
            )
        if not CREATED_FEATURES.issubset(set(self.expected_features)):
            raise ArtifactValidationError(
                "Saved preprocessor is missing Notebook 5 engineered fields"
            )
        if not hasattr(self.model, "classes_") or not hasattr(
            self.model, "predict_proba"
        ):
            raise ArtifactValidationError(
                "Saved model is not a fitted probability classifier"
            )
        classes = np.asarray(self.model.classes_)
        positive_indices = np.flatnonzero(classes == 1)
        if len(positive_indices) != 1:
            raise ArtifactValidationError("Saved model has no unique positive class 1")
        self.positive_index = int(positive_indices[0])

        try:
            self.numeric_features = set(self.preprocessor.transformers_[0][2])
            self.categorical_features = set(self.preprocessor.transformers_[1][2])
        except (AttributeError, IndexError, TypeError) as exc:
            raise ArtifactValidationError(
                "Unexpected ColumnTransformer layout"
            ) from exc

        self.raw_features = [
            name for name in self.expected_features if name not in CREATED_FEATURES
        ]
        self.allowed_fields = (
            set(self.raw_features) | set(DATE_COLUMNS) | set(ID_COLUMNS)
        )
        self.version = "sha256:" + _sha256_file(config.model_path)[:12]

        logger.info("Loaded model artifact %s", self.version)

    def _validate_order(self, order: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(order, Mapping):
            raise InputValidationError("Each order must be a JSON object")

        forbidden = set(order) & (set(LEAKAGE_COLUMNS) | {TARGET})
        if forbidden:
            raise InputValidationError(
                f"Future/target fields prohibited: {sorted(forbidden)}"
            )

        unexpected = set(order) - self.allowed_fields
        if unexpected:
            raise InputValidationError(f"Unknown input fields: {sorted(unexpected)}")

        cleaned: dict[str, Any] = {}
        for name in DATE_COLUMNS:
            value = order.get(name)
            if not isinstance(value, str) or not value.strip():
                raise InputValidationError(f"Required timestamp missing: {name}")
            try:
                timestamp = pd.to_datetime(value, format="mixed", errors="raise")
            except (TypeError, ValueError) as exc:
                raise InputValidationError(f"Invalid timestamp in {name}") from exc
            if pd.isna(timestamp) or getattr(timestamp, "tzinfo", None) is not None:
                raise InputValidationError(f"Timestamp must be timezone-naive: {name}")
            cleaned[name] = value

        if pd.Timestamp(cleaned[DATE_COLUMNS[1]]) < pd.Timestamp(
            cleaned[DATE_COLUMNS[0]]
        ):
            raise InputValidationError(
                "Estimated delivery date cannot precede purchase time"
            )

        for name in self.raw_features:
            value = order.get(name)
            if value is None:
                cleaned[name] = np.nan if name in self.numeric_features else None
            elif name in self.numeric_features:
                if isinstance(value, bool) or not isinstance(value, (float, int)):
                    raise InputValidationError(f"Expected a numeric value: {name}")
                if not math.isfinite(value):
                    raise InputValidationError(f"Expected finite numeric value: {name}")
                violation = numeric_bounds_violation(name, value)
                if violation is not None:
                    raise InputValidationError(violation)
                cleaned[name] = value
            elif name in self.categorical_features:
                if not isinstance(value, str):
                    raise InputValidationError(f"Expected a text value: {name}")
                cleaned[name] = value
            else:
                raise ArtifactValidationError(f"Unrecognized feature type: {name}")

        # These IDs are accepted for traceability, but the notebook drops them.
        for name in ID_COLUMNS:
            if name in order:
                value = order[name]
                if not isinstance(value, str):
                    raise InputValidationError(f"Expected a text ID: {name}")
                cleaned[name] = value

        return cleaned

    def prepare(self, orders: Sequence[Mapping[str, Any]]) -> sparse.csr_matrix:
        if not orders or len(orders) > self.config.max_batch_size:
            raise InputValidationError(
                f"Batch size must be between 1 and {self.config.max_batch_size}"
            )
        rows = [self._validate_order(order) for order in orders]
        frame = pd.DataFrame(rows)
        # Match the original notebook: direct columns plus derived columns in the
        # fitted ColumnTransformer's original order.
        engineered = engineer_features(frame).drop(columns=[TARGET], errors="ignore")
        missing = [
            name for name in self.expected_features if name not in engineered.columns
        ]
        if missing:
            raise ArtifactValidationError(
                f"Unable to construct required features: {missing}"
            )
        for name in self.numeric_features:
            if name in engineered:
                engineered[name] = pd.to_numeric(engineered[name], errors="raise")
        transformed = self.preprocessor.transform(engineered[self.expected_features])
        matrix = sparse.csr_matrix(transformed)
        if not np.isfinite(matrix.data).all():
            raise ArtifactValidationError("Transformed input contains nonfinite values")
        if (
            hasattr(self.model, "n_features_in_")
            and matrix.shape[1] != self.model.n_features_in_
        ):
            raise ArtifactValidationError(
                "Preprocessor/model feature dimension mismatch"
            )
        return matrix

    def predict(self, orders: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        features = self.prepare(orders)
        probabilities = np.asarray(self.model.predict_proba(features))
        if probabilities.ndim != 2 or probabilities.shape[0] != len(orders):
            raise ArtifactValidationError("Unexpected predict_proba output shape")
        scores = probabilities[:, self.positive_index]
        if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ArtifactValidationError("Model returned invalid probabilities")
        return [
            {
                "prediction": int(score >= self.threshold),
                "late_probability": float(score),
                "model_version": self.version,
            }
            for score in scores
        ]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()
