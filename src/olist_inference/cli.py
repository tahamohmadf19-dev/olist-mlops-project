from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import InferenceConfig
from .service import InferenceService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local Olist inference (no retraining)"
    )
    parser.add_argument("--config", default="config/inference.json")
    parser.add_argument(
        "--input", required=True, help="JSON file containing one order or a list"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        orders = payload if isinstance(payload, list) else [payload]
        predictions = InferenceService(InferenceConfig.from_json(args.config)).predict(
            orders
        )
        print(json.dumps(predictions, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        logging.error("Inference failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
