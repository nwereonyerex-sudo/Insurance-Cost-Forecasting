from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_cost.features import (
    ENHANCED,
    ENHANCED_OBESITY,
    MAIN_EFFECTS,
    FeatureBuilder,
    FeatureSpec,
    build_preprocessor,
)


@pytest.fixture
def two_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [50, 30],
            "sex": ["male", "female"],
            "bmi": [35.0, 25.0],
            "children": [2, 0],
            "smoker": ["yes", "no"],
            "region": ["southeast", "northeast"],
        }
    )


def test_enhanced_exact_values_and_order(two_rows):
    out = build_preprocessor(ENHANCED).fit_transform(two_rows)
    assert list(out.columns) == [
        "age_c",
        "bmi_c",
        "children",
        "age_c_sq",
        "smoker_x_age_c",
        "smoker_x_bmi_c",
        "sex_male",
        "smoker_yes",
        "region_northwest",
        "region_southeast",
        "region_southwest",
    ]
    np.testing.assert_allclose(out.iloc[0].to_numpy(), [10, 10, 2, 100, 10, 10, 1, 1, 0, 1, 0])
    np.testing.assert_allclose(out.iloc[1].to_numpy(), [-10, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0])


def test_main_effects_columns(two_rows):
    out = build_preprocessor(MAIN_EFFECTS).fit_transform(two_rows)
    assert list(out.columns) == [
        "age_c",
        "bmi_c",
        "children",
        "sex_male",
        "smoker_yes",
        "region_northwest",
        "region_southeast",
        "region_southwest",
    ]


def test_interactions_keep_main_effects():
    builder = FeatureBuilder.from_spec(ENHANCED)
    cols = builder.numeric_columns() + builder.categorical_columns()
    assert {"age_c", "bmi_c", "smoker"} <= set(cols)


def test_exclude_sex_and_bmi_sq(two_rows):
    spec = FeatureSpec(bmi_sq=True, include_sex=False)
    out = build_preprocessor(spec).fit_transform(two_rows)
    assert "sex_male" not in out.columns
    assert out["bmi_c_sq"].tolist() == [100.0, 0.0]


def test_order_independent_of_input_column_order(two_rows):
    shuffled = two_rows[list(reversed(two_rows.columns))]
    a = build_preprocessor(ENHANCED).fit_transform(two_rows)
    b = build_preprocessor(ENHANCED).fit_transform(shuffled)
    pd.testing.assert_frame_equal(a, b)


def test_unknown_category_maps_to_reference(two_rows):
    pre = build_preprocessor(ENHANCED).fit(two_rows)
    row = two_rows.iloc[[1]].assign(region="midwest")
    with pytest.warns(UserWarning, match="unknown categories"):
        out = pre.transform(row)
    assert out[["region_northwest", "region_southeast", "region_southwest"]].sum(axis=1).item() == 0


def test_builder_is_stateless(two_rows):
    """Fitting on different data must not change the transform (no learned state)."""
    other = two_rows.assign(age=[64, 18], bmi=[50.0, 16.0])
    a = FeatureBuilder.from_spec(ENHANCED).fit(two_rows).transform(two_rows)
    b = FeatureBuilder.from_spec(ENHANCED).fit(other).transform(two_rows)
    pd.testing.assert_frame_equal(a, b)


def test_missing_column_raises(two_rows):
    with pytest.raises(ValueError, match="bmi"):
        FeatureBuilder().fit(two_rows.drop(columns="bmi"))


def test_obesity_step_values_and_threshold(two_rows):
    rows = pd.concat([two_rows, two_rows.iloc[[0]].assign(bmi=29.99)], ignore_index=True)
    out = build_preprocessor(ENHANCED_OBESITY).fit_transform(rows)
    assert list(out.columns[:8]) == [
        "age_c",
        "bmi_c",
        "children",
        "age_c_sq",
        "obese",
        "smoker_x_age_c",
        "smoker_x_bmi_c",
        "smoker_x_obese",
    ]
    # BMI 35 smoker, BMI 25 non-smoker, BMI 29.99 smoker (just under the threshold).
    assert out["obese"].tolist() == [1.0, 0.0, 0.0]
    assert out["smoker_x_obese"].tolist() == [1.0, 0.0, 0.0]


def test_obesity_interaction_keeps_main_effect():
    cols = FeatureBuilder.from_spec(ENHANCED_OBESITY).numeric_columns()
    assert "obese" in cols and "smoker_x_obese" in cols
