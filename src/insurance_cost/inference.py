"""Input validation and prediction helpers used by the Streamlit app.

These functions only check and reshape user input. All feature engineering happens inside
the fitted pipeline loaded from the artifact.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from insurance_cost.explain import explain_prediction

NUMERIC_INPUTS = ("age", "bmi", "children")
INTEGER_INPUTS = ("age", "children")


def validate_inputs(values: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    """Return human-readable problems; an empty list means the input is usable."""
    errors: list[str] = []
    for field in metadata["input_features"]:
        if values.get(field) is None or values.get(field) == "":
            errors.append(f"Please provide a value for {field}.")
    if errors:
        return errors

    ranges = metadata["train_ranges"]
    for field in NUMERIC_INPUTS:
        try:
            number = float(values[field])
        except (TypeError, ValueError):
            errors.append(f"{field.capitalize()} must be a number.")
            continue
        if field in INTEGER_INPUTS and not number.is_integer():
            errors.append(f"{field.capitalize()} must be a whole number.")
        lo, hi = ranges[field]
        if not lo <= number <= hi:
            errors.append(
                f"{field.capitalize()} must be between {lo:g} and {hi:g}, "
                "the range seen in the training data."
            )

    for field, levels in metadata["category_levels"].items():
        if values[field] not in levels:
            errors.append(f"{field.capitalize()} must be one of: {', '.join(levels)}.")
    return errors


def build_input_frame(values: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    """One-row DataFrame in the training schema's column order and dtypes."""
    errors = validate_inputs(values, metadata)
    if errors:
        raise ValueError(" ".join(errors))
    row = {
        "age": int(values["age"]),
        "sex": str(values["sex"]),
        "bmi": float(values["bmi"]),
        "children": int(values["children"]),
        "smoker": str(values["smoker"]),
        "region": str(values["region"]),
    }
    return pd.DataFrame([row])[metadata["input_features"]]


def estimate(
    pipeline: Pipeline, metadata: dict[str, Any], values: dict[str, Any]
) -> dict[str, Any]:
    """Estimate, driver explanation, and the smoker/non-smoker comparison for one person."""
    row = build_input_frame(values, metadata)
    result = explain_prediction(pipeline, row)
    other = "no" if row.loc[0, "smoker"] == "yes" else "yes"
    result["smoker_comparison"] = {
        "smoker": other,
        "estimate": float(pipeline.predict(row.assign(smoker=other))[0]),
    }
    return result
