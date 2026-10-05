from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from insurance_cost.artifacts import load_artifact
from insurance_cost.config import FEATURES
from insurance_cost.inference import build_input_frame, estimate, validate_inputs
from insurance_cost.models import CANDIDATES_BY_NAME
from insurance_cost.train import train

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
FAST = tuple(
    CANDIDATES_BY_NAME[n] for n in ("baseline_mean", "ols_enhanced", "ols_enhanced_obesity")
)

VALID = {
    "age": 35,
    "sex": "female",
    "bmi": 26.0,
    "children": 0,
    "smoker": "no",
    "region": "northeast",
}


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory) -> Path:
    from tests.conftest import make_synthetic

    tmp = tmp_path_factory.mktemp("artifact")
    csv = tmp / "insurance.csv"
    make_synthetic(n=400, seed=3).to_csv(csv, index=False)
    path = tmp / "model.joblib"
    train(csv, artifact_path=path, metrics_path=None, figures_dir=None, candidates=FAST)
    return path


@pytest.fixture(scope="module")
def model(artifact_path):
    return load_artifact(artifact_path)


def test_input_mapping_matches_training_schema(model):
    _, meta = model
    frame = build_input_frame({**VALID, "age": 35.0, "children": 2.0}, meta)
    assert list(frame.columns) == list(FEATURES) == meta["input_features"]
    assert frame["age"].dtype == "int64" and frame["children"].dtype == "int64"
    assert frame["bmi"].dtype == "float64"
    assert frame.loc[0, "children"] == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"age": 70}, "Age must be between"),
        ({"age": 30.5}, "Age must be a whole number"),
        ({"bmi": "heavy"}, "Bmi must be a number"),
        ({"children": None}, "Please provide a value for children"),
        ({"region": "midwest"}, "Region must be one of"),
        ({"smoker": "sometimes"}, "Smoker must be one of"),
    ],
)
def test_invalid_inputs_give_clear_messages(model, change, message):
    _, meta = model
    errors = validate_inputs({**VALID, **change}, meta)
    assert any(message in e for e in errors), errors
    with pytest.raises(ValueError):
        build_input_frame({**VALID, **change}, meta)


@pytest.mark.parametrize(
    "profile",
    [
        {**VALID, "age": 20, "bmi": 22.0},  # low
        {**VALID, "age": 50, "bmi": 31.0, "children": 2},  # middle
        {**VALID, "age": 60, "bmi": 38.0, "smoker": "yes", "sex": "male"},  # high
    ],
)
def test_low_middle_high_examples_end_to_end(model, profile):
    pipeline, meta = model
    result = estimate(pipeline, meta, profile)
    assert np.isfinite(result["estimate"]) and result["estimate"] >= 0
    total = result["reference_estimate"] + sum(d["contribution"] for d in result["drivers"])
    assert total == pytest.approx(max(total, 0.0))
    assert result["smoker_comparison"]["smoker"] != profile["smoker"]


def test_examples_are_ordered(model):
    pipeline, meta = model
    low = estimate(pipeline, meta, {**VALID, "age": 20, "bmi": 22.0})["estimate"]
    high = estimate(pipeline, meta, {**VALID, "age": 60, "bmi": 38.0, "smoker": "yes"})["estimate"]
    assert high > low


def test_streamlit_app_valid_submission(artifact_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("INSURANCE_COST_ARTIFACT", str(artifact_path))
    import importlib

    import insurance_cost.config

    importlib.reload(insurance_cost.config)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    at.number_input[0].set_value(45)
    at.number_input[1].set_value(31.5)
    at.radio[0].set_value("yes")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert not at.error
    metric = at.metric[0]
    assert metric.label == "Estimated annual medical charges"
    assert metric.value.startswith("$")
    # Dollar amounts in markdown must be escaped, or Streamlit renders them as LaTeX.
    error_text = next(m.value for m in at.markdown if "typically off by" in m.value)
    assert r"about **\$" in error_text
    assert at.info and r"\$" in at.info[0].value


def test_streamlit_app_reports_missing_artifact(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("INSURANCE_COST_ARTIFACT", str(tmp_path / "missing.joblib"))
    import importlib

    import insurance_cost.config

    importlib.reload(insurance_cost.config)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert at.error and "could not be loaded" in at.error[0].value
