"""Aggregate-only structured logging for successful inference calls.

Never log raw orders, identifiers, user data, IP addresses or exact per-order scores.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

AUDIT_LOGGER = logging.getLogger("olist.prediction_audit")
AUDIT_LOGGER.setLevel(logging.INFO)
AUDIT_LOGGER.propagate = False
if not AUDIT_LOGGER.handlers:
    handler = logging.StreamHandler()  # Container stderr, not an app-managed file.
    handler.setFormatter(logging.Formatter("%(message)s"))
    AUDIT_LOGGER.addHandler(handler)


def make_prediction_event(
    results: Sequence[Mapping[str, Any]],
    *,
    model_version: str,
    inference_latency_ms: float,
) -> dict[str, str | int | float]:
    """Summarize one successful request without retaining order-level records."""
    if not results:
        raise ValueError("Cannot log an empty prediction result")
    if not math.isfinite(inference_latency_ms) or inference_latency_ms < 0:
        raise ValueError("Invalid inference latency")

    count = len(results)
    positives = 0
    score_sum = 0.0
    for item in results:
        label = int(item["prediction"])
        score = float(item["late_probability"])
        if label not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid prediction result for audit")
        positives += label
        score_sum += score

    summary: dict[str, str | int | float] = {
        "event": "prediction_summary",
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "model_version": model_version,
        "batch_size": count,
        "inference_latency_ms": round(inference_latency_ms, 3),
    }
    # Small batches are not individually auditable to reduce correlation risk.
    if count >= 20:
        summary["positive_predictions"] = positives
        summary["mean_late_probability"] = round(score_sum / count, 3)
    return summary


def emit_prediction_event(
    results: Sequence[Mapping[str, Any]],
    *,
    model_version: str,
    inference_latency_ms: float,
) -> None:
    """Emit one JSON line per successful API call to the configured logger."""
    try:
        event = make_prediction_event(
            results,
            model_version=model_version,
            inference_latency_ms=inference_latency_ms,
        )
    except (TypeError, ValueError, KeyError):
        # Logging must never turn an otherwise successful inference into an API failure.
        AUDIT_LOGGER.warning("prediction_summary_unavailable")
        return
    AUDIT_LOGGER.info(json.dumps(event, ensure_ascii=True, separators=(",", ":")))
