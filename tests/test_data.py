from __future__ import annotations

import pandas as pd
import pytest

from insurance_cost.config import DATA_PATH
from insurance_cost.data import (
    DataValidationError,
    load_dataset,
    load_raw,
    normalise,
    validate,
    validation_summary,
)


def test_valid_rows_pass_and_are_typed(valid_rows):
    out = validate(valid_rows, duplicate_policy="error")
    assert out["age"].dtype == "int64"
    assert out["children"].dtype == "int64"
    assert out["charges"].dtype == "float64"
    assert list(out.columns) == ["age", "sex", "bmi", "children", "smoker", "region", "charges"]


def test_missing_column_raises(valid_rows):
    with pytest.raises(DataValidationError, match="Missing required columns.*smoker"):
        validate(valid_rows.drop(columns="smoker"), duplicate_policy="keep")


def test_renamed_column_raises(valid_rows):
    with pytest.raises(DataValidationError, match="bmi"):
        validate(valid_rows.rename(columns={"bmi": "body_mass"}), duplicate_policy="keep")


def test_unexpected_category_raises(valid_rows):
    valid_rows.loc[0, "region"] = "midwest"
    with pytest.raises(DataValidationError, match="unexpected categories.*midwest"):
        validate(valid_rows, duplicate_policy="keep")


def test_unparseable_number_raises(valid_rows):
    df = valid_rows.astype({"bmi": object})
    df.loc[1, "bmi"] = "thirty"
    with pytest.raises(DataValidationError, match="bmi: 1 value"):
        validate(df, duplicate_policy="keep")


def test_negative_charges_raise(valid_rows):
    valid_rows.loc[0, "charges"] = -5.0
    with pytest.raises(DataValidationError, match="charges.*outside"):
        validate(valid_rows, duplicate_policy="keep")


def test_missing_value_raises(valid_rows):
    valid_rows.loc[2, "age"] = None
    with pytest.raises(DataValidationError, match="age: 1 missing"):
        validate(valid_rows, duplicate_policy="keep")


def test_fractional_children_raise(valid_rows):
    valid_rows["children"] = [0, 1.5, 2]
    with pytest.raises(DataValidationError, match="children: non-integer"):
        validate(valid_rows, duplicate_policy="keep")


def test_duplicate_policies(valid_rows):
    dup = pd.concat([valid_rows, valid_rows.iloc[[0]]], ignore_index=True)
    assert len(validate(dup, duplicate_policy="keep")) == 4
    assert len(validate(dup, duplicate_policy="drop")) == 3
    with pytest.raises(DataValidationError, match="duplicate"):
        validate(dup, duplicate_policy="error")


def test_normalise_strips_names_and_values(valid_rows):
    messy = valid_rows.rename(columns={"sex": " Sex "})
    messy.loc[0, " Sex "] = " Female "
    out = normalise(messy)
    assert "sex" in out.columns
    assert out.loc[0, "sex"] == "female"


def test_features_only_validation(valid_rows):
    features = valid_rows.drop(columns="charges")
    out = validate(features, duplicate_policy="keep", require_target=False)
    assert "charges" not in out.columns


def test_summary_contents(valid_rows):
    summary = validation_summary(validate(valid_rows, duplicate_policy="keep"))
    assert summary["rows"] == 3
    assert summary["duplicate_rows"] == 0
    assert summary["category_levels"]["smoker"] == {"no": 2, "yes": 1}
    assert summary["numeric_ranges"]["age"] == {"min": 19.0, "max": 60.0}


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_raw(tmp_path / "nope.csv")


@pytest.mark.integration
@pytest.mark.skipif(not DATA_PATH.exists(), reason="real dataset not downloaded")
def test_real_dataset_contract():
    df = load_dataset()
    assert len(df) == 1338
    assert int(df.duplicated().sum()) == 1
