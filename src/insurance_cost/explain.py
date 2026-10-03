"""Coefficient tables and per-prediction explanations for the linear pipeline.

The reference profile is the person whose transformed feature row is all zeros: age 40,
BMI 25, no children, female, non-smoker, northeast. For a raw-dollar model the intercept is
that person's estimate, and each coefficient × (feature value − reference value) is an exact
additive dollar contribution relative to them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from insurance_cost.features import AGE_CENTER, BMI_CENTER

REFERENCE_PROFILE = {
    "age": int(AGE_CENTER),
    "sex": "female",
    "bmi": BMI_CENTER,
    "children": 0,
    "smoker": "no",
    "region": "northeast",
}

# Plain-language meaning of each transformed column.
FEATURE_MEANINGS: dict[str, str] = {
    "age_c": "per year of age above 40",
    "bmi_c": "per BMI point above 25 (non-smokers; smokers add smoker × BMI)",
    "children": "per child covered",
    "age_c_sq": "curvature: per (years from 40)²",
    "bmi_c_sq": "curvature: per (BMI points from 25)²",
    "obese": "BMI ≥ 30 vs below 30 (non-smokers; smokers add smoker × obese)",
    "smoker_x_age_c": "extra per year of age above 40 for smokers",
    "smoker_x_bmi_c": "extra per BMI point above 25 for smokers",
    "smoker_x_obese": "extra step for smokers with BMI ≥ 30",
    "sex_male": "male vs female (reference)",
    "smoker_yes": "smoker vs non-smoker, at age 40 and BMI 25",
    "region_northwest": "northwest vs northeast (reference)",
    "region_southeast": "southeast vs northeast (reference)",
    "region_southwest": "southwest vs northeast (reference)",
}

# Transformed columns grouped back into the inputs a person enters.
DRIVER_GROUPS: dict[str, tuple[str, ...]] = {
    "Smoking (incl. its interaction with age and BMI)": (
        "smoker_yes",
        "smoker_x_age_c",
        "smoker_x_bmi_c",
        "smoker_x_obese",
    ),
    "Age": ("age_c", "age_c_sq"),
    "BMI": ("bmi_c", "bmi_c_sq", "obese"),
    "Children": ("children",),
    "Sex": ("sex_male",),
    "Region": ("region_northwest", "region_southeast", "region_southwest"),
}


def _parts(pipeline: Pipeline) -> tuple[Pipeline, object, object]:
    preprocess, model = pipeline[:-1], pipeline[-1]
    inner = getattr(model, "regressor_", None)
    if inner is None or not hasattr(inner, "coef_"):
        raise TypeError("Explanations require a fitted linear model inside ChargesRegressor")
    return preprocess, model, inner


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Coefficients with units. For log-target models, also the approximate % difference."""
    preprocess, model, inner = _parts(pipeline)
    names = list(preprocess.get_feature_names_out())
    log_target = bool(getattr(model, "log_target", False))
    rows = [
        {
            "feature": "intercept",
            "meaning": "estimate for the reference profile"
            + (" (log1p scale)" if log_target else " (USD)"),
            "coefficient": float(inner.intercept_),
        }
    ]
    for name, coef in zip(names, inner.coef_, strict=True):
        rows.append(
            {
                "feature": name,
                "meaning": FEATURE_MEANINGS.get(name, name),
                "coefficient": float(coef),
            }
        )
    table = pd.DataFrame(rows)
    table["unit"] = "log1p(USD)" if log_target else "USD"
    if log_target:
        table["approx_pct_difference"] = np.where(
            table["feature"] == "intercept", np.nan, (np.exp(table["coefficient"]) - 1) * 100
        )
    return table


def explain_prediction(pipeline: Pipeline, row: pd.DataFrame) -> dict[str, object]:
    """Explain one prediction as contributions relative to the reference profile.

    Raw-dollar model: contributions are additive USD and sum exactly to the estimate.
    Log-target model: contributions are multiplicative; they are reported as % differences.
    """
    if len(row) != 1:
        raise ValueError("explain_prediction expects exactly one row")
    preprocess, model, inner = _parts(pipeline)
    reference = pd.DataFrame([REFERENCE_PROFILE])[list(row.columns)]
    design = preprocess.transform(row).iloc[0]
    design_ref = preprocess.transform(reference).iloc[0]
    coefs = pd.Series(inner.coef_, index=design.index)
    term = coefs * (design - design_ref)
    log_target = bool(getattr(model, "log_target", False))

    drivers = []
    for label, cols in DRIVER_GROUPS.items():
        present = [c for c in cols if c in term.index]
        if not present:
            continue
        value = float(term[present].sum())
        drivers.append(
            {
                "driver": label,
                "contribution": (np.exp(value) - 1) * 100 if log_target else value,
            }
        )
    drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return {
        "estimate": float(pipeline.predict(row)[0]),
        "reference_estimate": float(pipeline.predict(reference)[0]),
        "unit": "percent" if log_target else "usd",
        "drivers": drivers,
    }
