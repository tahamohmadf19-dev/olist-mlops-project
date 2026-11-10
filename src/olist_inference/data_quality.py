"""Dataset-level Olist data-quality rules, independent from model inference.

Business checks complement Great Expectations. They intentionally do not make
location coordinates mandatory because verified Olist validation data have
some missing geocodes. No row-level data are included in diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


REQUIRED_NON_NULL = (
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "item_count",
    "total_item_price",
    "total_payment_value",
    "payment_record_count",
    "payment_type_count",
    "payment_types",
    "max_payment_installments",
)

# Optional fields are allowed to be missing; provided values must be sane.
NUMERIC_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "item_count": (1, None),
    "total_item_price": (0, None),
    "total_payment_value": (0, None),
    "customer_lat": (-90, 90),
    "customer_lng": (-180, 180),
    "avg_seller_lat": (-90, 90),
    "avg_seller_lng": (-180, 180),
}
DATE_FIELDS = ("order_purchase_timestamp", "order_estimated_delivery_date")
REQUIRED_COLUMNS = tuple(dict.fromkeys((*REQUIRED_NON_NULL, *NUMERIC_BOUNDS)))


def numeric_bounds_violation(name: str, value: float) -> str | None:
    """Check shared bounds for a supplied, finite number; null policy is separate."""
    minimum, maximum = NUMERIC_BOUNDS.get(name, (None, None))
    if minimum is not None and value < minimum:
        return f"{name} must be >= {minimum:g}"
    if maximum is not None and value > maximum:
        return f"{name} must be <= {maximum:g}"
    return None


@dataclass(frozen=True)
class DataQualityAssessment:
    row_count: int
    problems: tuple[str, ...]
    geocode_missing_percent: dict[str, float]

    @property
    def passed(self) -> bool:
        return not self.problems

    def to_report(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "rows": self.row_count,
            "problems": list(self.problems),
            "geocode_missing_percent": self.geocode_missing_percent,
        }


def assess_dataframe(frame: pd.DataFrame) -> DataQualityAssessment:
    """Check semantic constraints with aggregate-only error messages.

    This covers date ordering, parseability, forbidden negative counts/prices,
    impossible coordinates, and non-null requirements. It does not infer when
    payment/estimated-delivery fields become available in a live system.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    problems: list[str] = []
    if frame.empty:
        problems.append("dataset must have at least one row")

    for name in REQUIRED_COLUMNS:
        if name not in frame.columns:
            problems.append(f"required column absent: {name}")

    for name in REQUIRED_NON_NULL:
        if name in frame:
            count = int(frame[name].isna().sum())
            if count:
                problems.append(f"{name}: {count} missing values")

    for name, (minimum, maximum) in NUMERIC_BOUNDS.items():
        if name not in frame:
            continue
        original = frame[name]
        numeric = pd.to_numeric(original, errors="coerce")
        malformed = int((original.notna() & numeric.isna()).sum())
        if malformed:
            problems.append(f"{name}: {malformed} non-numeric values")
        if minimum is not None:
            count = int((numeric < minimum).sum())
            if count:
                problems.append(f"{name}: {count} values below {minimum:g}")
        if maximum is not None:
            count = int((numeric > maximum).sum())
            if count:
                problems.append(f"{name}: {count} values above {maximum:g}")

    if all(name in frame for name in DATE_FIELDS):
        purchase = pd.to_datetime(
            frame[DATE_FIELDS[0]], format="mixed", errors="coerce"
        )
        estimated = pd.to_datetime(
            frame[DATE_FIELDS[1]], format="mixed", errors="coerce"
        )
        for name, values in zip(DATE_FIELDS, (purchase, estimated)):
            invalid_count = int((frame[name].notna() & values.isna()).sum())
            if invalid_count:
                problems.append(f"{name}: {invalid_count} unparseable timestamps")
        invalid_order = int((estimated < purchase).sum())
        if invalid_order:
            problems.append(
                f"estimated delivery precedes purchase for {invalid_order} rows"
            )

    geo_cols = ("customer_lat", "customer_lng", "avg_seller_lat", "avg_seller_lng")
    geocode_missing = {
        name: round(float(frame[name].isna().mean() * 100), 4)
        for name in geo_cols
        if name in frame and len(frame) > 0
    }
    return DataQualityAssessment(
        row_count=len(frame),
        problems=tuple(problems),
        geocode_missing_percent=geocode_missing,
    )
