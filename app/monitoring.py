"""Privacy-preserving, per-process Prometheus monitoring for local inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram


@dataclass(frozen=True)
class Monitoring:
    registry: CollectorRegistry
    requests: Counter
    latency: Histogram
    predictions: Counter
    probabilities: Histogram
    batch_size: Histogram

    def observe_predictions(self, results: list[dict[str, Any]]) -> None:
        self.batch_size.observe(len(results))
        for result in results:
            label = str(int(result["prediction"]))
            self.predictions.labels(label=label).inc()
            self.probabilities.observe(float(result["late_probability"]))


def new_monitoring() -> Monitoring:
    """Use an isolated registry so test apps do not leak counters across tests."""
    registry = CollectorRegistry()
    return Monitoring(
        registry=registry,
        requests=Counter(
            "olist_http_requests_total",
            "HTTP request counts grouped by bounded route name and status",
            ("route", "method", "status"),
            registry=registry,
        ),
        latency=Histogram(
            "olist_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ("route", "method"),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
            registry=registry,
        ),
        predictions=Counter(
            "olist_predictions_total",
            "Prediction counts by model output label, with no identifiers",
            ("label",),
            registry=registry,
        ),
        probabilities=Histogram(
            "olist_prediction_probability",
            "Distribution of late-delivery prediction probabilities",
            buckets=(0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
            registry=registry,
        ),
        batch_size=Histogram(
            "olist_prediction_batch_size",
            "Number of orders per successful prediction call",
            buckets=(1, 2, 5, 10, 25, 50, 100, 500, 1000),
            registry=registry,
        ),
    )
