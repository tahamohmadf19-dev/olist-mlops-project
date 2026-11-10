"""Unit checks for score histogram and insufficient-data safety gates."""

import math

import numpy as np
import pytest

from scripts.check_prediction_drift import read_histogram
from src.olist_inference.drift import (
    PROBABILITY_BOUNDS,
    DriftDataError,
    bound_labels,
    calculate_psi,
    evaluate_drift,
    observed_bin_counts,
    score_bin_counts,
)


def reference() -> dict:
    scores = np.concatenate([np.full(300, 0.04), np.full(700, 0.35)])
    return {
        "schema_version": 1,
        "bounds": bound_labels(),
        "training_rows": 1000,
        "bin_counts": score_bin_counts(scores),
    }


def test_score_bins_match_inclusive_prometheus_bounds() -> None:
    counts = score_bin_counts(np.array([0.0, 0.05, 0.051, 1.0]))
    assert counts[0] == 1
    assert counts[1] == 1
    assert counts[2] == 1
    assert counts[-2] == 1
    assert counts[-1] == 0


def test_invalid_probability_rejected() -> None:
    with pytest.raises(DriftDataError):
        score_bin_counts(np.array([0.1, math.nan]))


def test_prometheus_histogram_read() -> None:
    exclusive = [3.0] + [0.0] * (len(PROBABILITY_BOUNDS) - 1)
    cumulative = np.cumsum(exclusive)
    data = [
        {"metric": {"le": bound}, "value": [0, str(count)]}
        for bound, count in zip(bound_labels(), cumulative)
    ]
    assert read_histogram(data) == exclusive


def test_missing_bucket_rejected() -> None:
    with pytest.raises(DriftDataError):
        observed_bin_counts({0.0: 1.0, math.inf: 1.0})


def test_nonmonotonic_histogram_rejected() -> None:
    cumulative = {bound: 10.0 for bound in PROBABILITY_BOUNDS}
    cumulative[0.2] = 9.0
    with pytest.raises(DriftDataError):
        observed_bin_counts(cumulative)


def test_insufficient_when_prometheus_has_no_samples() -> None:
    output = evaluate_drift(reference(), None, minimum_samples=200)
    assert output["status"] == "INSUFFICIENT_DATA"
    assert output["psi"] is None


def test_insufficient_with_six_predictions() -> None:
    current = [0.0] * len(PROBABILITY_BOUNDS)
    current[1] = 6.0
    output = evaluate_drift(reference(), current, minimum_samples=200)
    assert output["status"] == "INSUFFICIENT_DATA"


def test_same_distribution_has_no_shift_signal() -> None:
    baseline = reference()
    output = evaluate_drift(baseline, baseline["bin_counts"])
    assert output["status"] == "NO_SHIFT_SIGNAL"
    assert output["psi"] == 0.0


def test_large_shift_requires_review() -> None:
    current = score_bin_counts(np.full(1000, 0.9))
    output = evaluate_drift(reference(), current)
    assert output["status"] == "REVIEW_HIGH"
    assert output["psi"] > 0.25


def test_invalid_baseline_rejected() -> None:
    baseline = reference()
    baseline["bounds"] = ["invalid"]
    with pytest.raises(DriftDataError):
        evaluate_drift(baseline, baseline["bin_counts"])


def test_psi_rejects_invalid_count() -> None:
    with pytest.raises(DriftDataError):
        calculate_psi([0.0], [0.0])


def test_build_training_baseline_from_trusted_fitted_model(
    tmp_path, monkeypatch, capsys
) -> None:
    import json
    import sys

    import joblib
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression

    from scripts.build_drift_baseline import main

    rng = np.random.default_rng(7)
    features = rng.normal(size=(60, 4))
    labels = np.arange(60) % 2
    model = LogisticRegression().fit(features, labels)
    model_file = tmp_path / "model.joblib"
    features_file = tmp_path / "features.npz"
    output_file = tmp_path / "baseline.json"
    joblib.dump(model, model_file)
    sparse.save_npz(features_file, sparse.csr_matrix(features))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_drift_baseline",
            "--model",
            str(model_file),
            "--train",
            str(features_file),
            "--output",
            str(output_file),
        ],
    )
    assert main() == 0
    baseline = json.loads(output_file.read_text(encoding="utf-8"))
    assert baseline["training_rows"] == 60
    assert sum(baseline["bin_counts"]) == 60
    assert baseline["model_version"].startswith("sha256:")
    assert baseline["source"] == "TRAINING_FEATURES_ONLY"
    assert main() == 0
    assert "ALREADY VERIFIED" in capsys.readouterr().out


def test_prometheus_query_encodes_window(monkeypatch) -> None:
    from scripts import check_prediction_drift

    seen = {}

    def fake_fetch(url):
        seen["url"] = url
        return {"status": "success", "data": {"resultType": "vector", "result": []}}

    monkeypatch.setattr(check_prediction_drift, "fetch_json", fake_fetch)
    expression = 'sum by (le) (increase(olist_prediction_probability_bucket{job="olist-api"}[24h]))'
    assert (
        check_prediction_drift.query_prometheus("http://127.0.0.1:9090", expression)
        == []
    )
    assert "query=" in seen["url"]
    assert "%24h" not in seen["url"]
    assert "%5B24h%5D" in seen["url"]
