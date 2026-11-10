"""Aggregate-only prediction-score shift checks against training predictions.

This detects changes in the model's prediction distribution, not feature,
concept, or target drift. No order-level data is persisted by this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# These must match the bins in app/monitoring.py (Prometheus classic histogram).
# check_prediction_drift.py refuses to compare if the scraped bounds differ.
PROBABILITY_BOUNDS: tuple[float, ...] = (
    0.0,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    math.inf,
)
MIN_OBSERVATIONS = 200


class DriftDataError(ValueError):
    """Baseline or Prometheus histogram is unsuitable for comparison."""


def score_bin_counts(scores: Sequence[float] | np.ndarray) -> list[int]:
    """Use right-inclusive buckets, as Prometheus classic histograms do."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise DriftDataError("Expected a nonempty one-dimensional probability array")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise DriftDataError("Probabilities must be finite and between 0 and 1")
    indices = np.searchsorted(PROBABILITY_BOUNDS, values, side="left")
    return np.bincount(indices, minlength=len(PROBABILITY_BOUNDS)).astype(int).tolist()


def observed_bin_counts(cumulative: Mapping[float, float]) -> list[float]:
    """Turn cumulative Prometheus histogram buckets into exclusive counts."""
    expected = set(PROBABILITY_BOUNDS)
    if set(cumulative) != expected:
        raise DriftDataError(
            "Prometheus histogram boundaries differ from training baseline"
        )

    counts: list[float] = []
    previous = 0.0
    for bound in PROBABILITY_BOUNDS:
        current = float(cumulative[bound])
        if not math.isfinite(current) or current < -1e-6:
            raise DriftDataError("Prometheus histogram contains invalid counts")
        if current + 1e-6 < previous:
            raise DriftDataError("Prometheus cumulative buckets are not monotonic")
        counts.append(max(0.0, current - previous))
        previous = current
    return counts


def calculate_psi(
    baseline_counts: Sequence[float],
    observed_counts: Sequence[float],
    *,
    pseudo_count: float = 0.5,
) -> float:
    """PSI on aggregate score bins, with symmetric pseudocount smoothing."""
    reference = np.asarray(baseline_counts, dtype=np.float64)
    current = np.asarray(observed_counts, dtype=np.float64)
    if (
        reference.shape != (len(PROBABILITY_BOUNDS),)
        or current.shape != reference.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(current).all()
        or np.any(reference < 0)
        or np.any(current < 0)
        or reference.sum() <= 0
        or current.sum() <= 0
        or pseudo_count <= 0
    ):
        raise DriftDataError("Invalid or empty histogram counts")
    ref_freq = (reference + pseudo_count) / (
        reference.sum() + pseudo_count * len(reference)
    )
    cur_freq = (current + pseudo_count) / (current.sum() + pseudo_count * len(current))
    return float(np.sum((cur_freq - ref_freq) * np.log(cur_freq / ref_freq)))


def evaluate_drift(
    baseline: Mapping[str, Any],
    observed: Sequence[float] | None,
    *,
    minimum_samples: int = MIN_OBSERVATIONS,
) -> dict[str, Any]:
    """Return a measured status; never infer drift from a tiny sample."""
    if minimum_samples < 1:
        raise DriftDataError("minimum_samples must be positive")
    if baseline.get("schema_version") != 1 or baseline.get("bounds") != bound_labels():
        raise DriftDataError(
            "Unsupported baseline schema or incompatible bucket boundaries"
        )
    baseline_counts = baseline.get("bin_counts")
    if not isinstance(baseline_counts, list) or len(baseline_counts) != len(
        PROBABILITY_BOUNDS
    ):
        raise DriftDataError("Invalid training histogram in baseline")
    if not all(isinstance(v, int) and v >= 0 for v in baseline_counts):
        raise DriftDataError("Invalid baseline bin counts")
    if (
        sum(baseline_counts) != baseline.get("training_rows")
        or sum(baseline_counts) <= 0
    ):
        raise DriftDataError("Training count does not match baseline histogram")
    if observed is None:
        return {
            "status": "INSUFFICIENT_DATA",
            "observations": 0,
            "minimum_samples": minimum_samples,
            "psi": None,
        }
    current = np.asarray(observed, dtype=float)
    if (
        current.shape != (len(PROBABILITY_BOUNDS),)
        or not np.isfinite(current).all()
        or np.any(current < 0)
    ):
        raise DriftDataError("Invalid observed histogram")
    count = float(current.sum())
    result: dict[str, Any] = {
        "observations": round(count, 3),
        "minimum_samples": minimum_samples,
        "psi": None,
    }
    if count < minimum_samples:
        result["status"] = "INSUFFICIENT_DATA"
        return result
    psi = calculate_psi(baseline_counts, current)
    result["psi"] = round(psi, 6)
    # Heuristic review bands, not calibrated statistical significance tests.
    result["status"] = (
        "REVIEW_HIGH" if psi >= 0.25 else "REVIEW" if psi >= 0.1 else "NO_SHIFT_SIGNAL"
    )
    return result


def bound_labels() -> list[str]:
    return ["+Inf" if math.isinf(v) else str(v) for v in PROBABILITY_BOUNDS]
