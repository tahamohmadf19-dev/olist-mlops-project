"""Generate an aggregate training-score baseline from already-fitted artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from src.olist_inference.drift import bound_labels, score_bin_counts

DEFAULT_TRAIN = Path("artifacts/05_feature_engineering/05_train_features.npz")
DEFAULT_MODEL = Path("artifacts/06_model/06_best_model.joblib")
DEFAULT_OUTPUT = Path("artifacts/10_drift/training_score_baseline.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reference score histogram from training features, without refitting"
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    for path in (args.train, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)

    model_sha = sha256(args.model)
    train_sha = sha256(args.train)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            existing.get("model_sha256") != model_sha
            or existing.get("train_features_sha256") != train_sha
            or existing.get("bounds") != bound_labels()
        ):
            raise RuntimeError(
                "Existing baseline does not match current trusted artifacts; refusing overwrite"
            )
        print("TRAINING BASELINE: ALREADY VERIFIED")
        print("Training rows:", existing["training_rows"])
        print("Model version:", existing["model_version"])
        return 0

    # joblib uses pickle deserialization. Only load your own trusted Task 2 artifact.
    model = joblib.load(args.model)
    features = sparse.load_npz(args.train)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1 or features.shape[0] < 1:
        raise RuntimeError(
            "Model must have exactly one positive class and nonempty training features"
        )
    if hasattr(model, "n_features_in_") and features.shape[1] != model.n_features_in_:
        raise RuntimeError("Saved model and training feature dimensions differ")
    scores = np.asarray(model.predict_proba(features))[:, int(positive[0])]
    counts = score_bin_counts(scores)
    baseline = {
        "schema_version": 1,
        "source": "TRAINING_FEATURES_ONLY",
        "model_sha256": model_sha,
        "model_version": "sha256:" + model_sha[:12],
        "train_features_sha256": train_sha,
        "training_rows": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "bounds": bound_labels(),
        "bin_counts": counts,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print("TRAINING BASELINE: CREATED")
    print("Training rows:", baseline["training_rows"])
    print("Model version:", baseline["model_version"])
    print("Aggregate bins:", len(counts))
    print("Baseline:", args.output)
    print("Training performed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
