"""Offline validation of Olist validation CSV using GX Core 1.24.1.

Use: python -m scripts.validate_dataset_gx
The script never modifies original data, never trains and never contacts the API.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.olist_inference.data_quality import (
    NUMERIC_BOUNDS,
    REQUIRED_NON_NULL,
    assess_dataframe,
)

LOGGER = logging.getLogger("olist.data_quality")
DEFAULT_INPUT = Path("artifacts/03_validation.csv")
DEFAULT_OUTPUT = Path("artifacts/07_data_validation/validation_report.json")


def check_with_great_expectations(frame: pd.DataFrame) -> dict[str, Any]:
    """Run an in-memory GX suite against a pandas DataFrame."""
    try:
        import great_expectations as gx
    except ImportError as exc:
        raise RuntimeError(
            "Great Expectations is not installed. Install requirements/data_quality.txt"
        ) from exc

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(name="olist_validation_source")
    data_asset = data_source.add_dataframe_asset(name="validation_dataframe")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        name="entire_validation"
    )
    suite = context.suites.add(
        gx.core.expectation_suite.ExpectationSuite(name="olist_quality_v1")
    )

    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=1))
    for name in REQUIRED_NON_NULL:
        if name in frame:
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToNotBeNull(column=name)
            )

    for name, (minimum, maximum) in NUMERIC_BOUNDS.items():
        if name in frame:
            suite.add_expectation(
                gx.expectations.ExpectColumnValuesToBeBetween(
                    column=name,
                    min_value=minimum,
                    max_value=maximum,
                )
            )

    definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name="olist_validation_quality",
            data=batch_definition,
            suite=suite,
        )
    )
    result = definition.run(batch_parameters={"dataframe": frame})
    failed = []
    for outcome in result.results:
        if not outcome.success:
            config = outcome.expectation_config
            expectation_type = getattr(config, "type", "unknown_expectation")
            column = getattr(config, "kwargs", {}).get("column", "table")
            failed.append(f"{expectation_type}:{column}")

    return {
        "passed": bool(result.success),
        "evaluated_expectations": len(result.results),
        "failed_expectations": failed,
        "gx_version": gx.__version__,
    }


def run(input_path: Path, output_path: Path) -> bool:
    if not input_path.is_file():
        raise FileNotFoundError(f"Data file not found: {input_path}")

    frame = pd.read_csv(input_path, low_memory=False)
    assessment = assess_dataframe(frame)
    if any(name not in frame for name in REQUIRED_NON_NULL):
        LOGGER.warning("Missing required columns; GX will validate present fields only")

    gx_result = check_with_great_expectations(frame)
    report = {
        "dataset": input_path.name,
        "rows": len(frame),
        "business_rules": assessment.to_report(),
        "great_expectations": gx_result,
        "passed": assessment.passed and gx_result["passed"],
        "availability_at_prediction_time_verified": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"DATA QUALITY: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Rows: {len(frame)}")
    print(f"GX expectations: {gx_result['evaluated_expectations']}")
    print(f"GX failures: {len(gx_result['failed_expectations'])}")
    for problem in assessment.problems:
        print(f"BUSINESS RULE FAILURE: {problem}")
    for failure in gx_result["failed_expectations"]:
        print(f"GX FAILURE: {failure}")
    print(f"Report: {output_path}")
    return bool(report["passed"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate existing Olist data, without retraining"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        return 0 if run(args.input, args.output) else 1
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        LOGGER.error("Data validation failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
