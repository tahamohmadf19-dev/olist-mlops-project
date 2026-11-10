from __future__ import annotations

import argparse
import sys

import pandas as pd
from scipy import sparse

from src.olist_inference.config import InferenceConfig
from src.olist_inference.features import TARGET, engineer_features
from src.olist_inference.service import InferenceService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare extracted feature code with Notebook 5 output"
    )
    parser.add_argument("--config", default="config/inference.json")
    parser.add_argument("--split-csv", default="artifacts/03_validation.csv")
    parser.add_argument(
        "--expected-npz",
        default="artifacts/05_feature_engineering/05_validation_features.npz",
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    service = InferenceService(InferenceConfig.from_json(args.config))
    original = pd.read_csv(args.split_csv)
    engineered = engineer_features(original).drop(columns=[TARGET], errors="ignore")
    if list(engineered.columns) != service.expected_features:
        raise AssertionError("Notebook/serialized input feature order differs")
    actual = sparse.csr_matrix(service.preprocessor.transform(engineered))
    expected = sparse.load_npz(args.expected_npz).tocsr()
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
    diff = (actual - expected).tocsr()
    max_diff = float(abs(diff.data).max()) if diff.nnz else 0.0
    print(f"Compared {actual.shape[0]} rows x {actual.shape[1]} transformed features")
    print(f"Maximum absolute difference: {max_diff:.12g}")
    if max_diff > args.tolerance:
        raise AssertionError(f"Feature parity FAILED (tolerance={args.tolerance})")
    print("Feature parity PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
