"""Compare Prometheus 24h prediction-score histogram with a training baseline."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from src.olist_inference.drift import (
    DriftDataError,
    evaluate_drift,
    observed_bin_counts,
)

DEFAULT_BASELINE = Path("artifacts/10_drift/training_score_baseline.json")
DEFAULT_REPORT = Path("artifacts/10_drift/latest_drift_report.json")


def fetch_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=10) as response:  # nosec: localhost default; no auth or secrets
            value = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unable to read monitoring endpoint: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise DriftDataError("Unexpected response from monitoring service")
    return value


def query_prometheus(base_url: str, expression: str) -> list[dict[str, Any]]:
    url = base_url.rstrip("/") + "/api/v1/query?" + urlencode({"query": expression})
    response = fetch_json(url)
    if (
        response.get("status") != "success"
        or response.get("data", {}).get("resultType") != "vector"
    ):
        raise RuntimeError("Prometheus query did not succeed")
    result = response["data"].get("result", [])
    if not isinstance(result, list):
        raise DriftDataError("Unexpected Prometheus vector")
    return result


def read_histogram(query_result: list[dict[str, Any]]) -> list[float] | None:
    if not query_result:
        return None
    cumulative: dict[float, float] = {}
    for item in query_result:
        label = item["metric"]["le"]
        bound = math.inf if label in ("+Inf", "Inf") else float(label)
        value = float(item["value"][1])
        if bound in cumulative:
            raise DriftDataError("Duplicate bucket boundary")
        cumulative[bound] = value
    return observed_bin_counts(cumulative)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Privacy-preserving aggregate score drift precheck"
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prometheus", default="http://127.0.0.1:9090")
    parser.add_argument("--api", default="http://127.0.0.1:8001")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--minimum-samples", type=int, default=200)
    args = parser.parse_args()

    if not 1 <= args.hours <= 168 or args.minimum_samples < 1:
        parser.error("--hours must be 1..168, and --minimum-samples must be positive")
    # Local-only safeguard. Extend only after an explicit network security review.
    for url in (args.prometheus, args.api):
        if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
            parser.error("Only loopback HTTP endpoints are allowed for this local task")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    model_info = fetch_json(args.api.rstrip("/") + "/model")
    if model_info.get("model_version") != baseline.get("model_version"):
        raise RuntimeError(
            "API model version does not match training baseline; refusing comparison"
        )

    up = query_prometheus(args.prometheus, 'up{job="olist-api"}')
    if len(up) != 1 or str(up[0]["value"][1]) != "1":
        raise RuntimeError(
            "Prometheus does not report exactly one healthy olist-api target"
        )
    promql = (
        "sum by (le) (increase("
        f'olist_prediction_probability_bucket{{job="olist-api"}}[{args.hours}h]'
        "))"
    )
    observed = read_histogram(query_prometheus(args.prometheus, promql))
    outcome = evaluate_drift(baseline, observed, minimum_samples=args.minimum_samples)
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "PROMETHEUS_CLASSIC_HISTOGRAM_INCREASE",
        "model_version": baseline["model_version"],
        "training_rows": baseline["training_rows"],
        "window_hours": args.hours,
        "reference": "TRAINING_FEATURES_ONLY",
        **outcome,
        "note": "Prediction-score distribution only; PSI thresholds are review heuristics, not proof of real-world drift.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print("PREDICTION SCORE DRIFT CHECK:", report["status"])
    print("Window hours:", args.hours)
    print("Observations:", report["observations"])
    print("Minimum required:", args.minimum_samples)
    print("PSI:", report["psi"])
    print("Report:", args.report)
    print("Customer identifiers stored: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
