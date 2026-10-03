from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from insurance_cost import explain
from insurance_cost.artifacts import ArtifactError, load_artifact, save_artifact
from insurance_cost.config import FEATURES
from insurance_cost.evaluate import regression_metrics, subgroup_table
from insurance_cost.models import CANDIDATES_BY_NAME, ChargesRegressor, comparison_candidates
from insurance_cost.train import select_model, split_data, train

FAST = tuple(
    CANDIDATES_BY_NAME[n] for n in ("baseline_mean", "ols_main", "ols_enhanced", "log_enhanced")
)


@pytest.fixture
def synthetic_csv(synthetic_df, tmp_path):
    path = tmp_path / "insurance.csv"
    synthetic_df.to_csv(path, index=False)
    return path


def test_end_to_end_fit_predict_one_row(synthetic_csv, tmp_path):
    result = train(
        synthetic_csv,
        artifact_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        figures_dir=None,
        candidates=FAST,
    )
    row = pd.DataFrame(
        [
            {
                "age": 45,
                "sex": "male",
                "bmi": 31.0,
                "children": 1,
                "smoker": "yes",
                "region": "southeast",
            }
        ]
    )
    pred = result.pipeline.predict(row)
    assert pred.shape == (1,) and pred[0] > 0
    assert result.test_metrics["mae"] < result.baseline_test_metrics["mae"]
    assert result.test_metrics["rmse"] < result.baseline_test_metrics["rmse"]
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["selected_model"] == result.selected
    assert metrics["data"]["random_state"] == 42


def test_training_is_reproducible(synthetic_csv):
    a = train(synthetic_csv, None, None, None, candidates=FAST)
    b = train(synthetic_csv, None, None, None, candidates=FAST)
    pd.testing.assert_frame_equal(a.cv_table, b.cv_table)
    assert a.test_metrics == pytest.approx(b.test_metrics)


def test_split_is_stratified_and_disjoint(synthetic_df):
    split = split_data(synthetic_df)
    assert len(split.X_test) == pytest.approx(0.2 * len(synthetic_df), abs=1)
    assert not set(split.X_train.index) & set(split.X_test.index)
    train_rate = (split.X_train["smoker"] == "yes").mean()
    test_rate = (split.X_test["smoker"] == "yes").mean()
    assert abs(train_rate - test_rate) < 0.03


def test_preprocessing_does_not_learn_from_test(synthetic_df):
    """Changing test rows must not change what the fitted pipeline learned."""
    split = split_data(synthetic_df)
    ridge = comparison_candidates(CANDIDATES_BY_NAME["ols_enhanced"])[1]
    pipe = ridge.build().fit(split.X_train, split.y_train)
    scaler = pipe.named_steps["preprocess"].named_steps["encode"].named_transformers_["num"]
    built = pipe.named_steps["preprocess"].named_steps["build"].transform(split.X_train)
    np.testing.assert_allclose(scaler.mean_, built[scaler.feature_names_in_].mean().to_numpy())


def test_log_target_predictions_are_nonnegative_dollars(synthetic_df):
    X, y = synthetic_df[list(FEATURES)], synthetic_df["charges"]
    pipe = CANDIDATES_BY_NAME["log_enhanced"].build().fit(X, y)
    pred = pipe.predict(X)
    assert (pred >= 0).all()
    # Dollar scale, not log scale.
    assert pred.mean() > 1000


def test_smearing_raises_predictions():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = np.expm1(5 + X @ [0.3, -0.2] + rng.normal(0, 0.5, 200))
    plain = ChargesRegressor(log_target=True).fit(X, y)
    smeared = ChargesRegressor(log_target=True, smearing=True).fit(X, y)
    assert smeared.smearing_factor_ > 1
    assert (smeared.predict(X) > plain.predict(X)).all()


def test_raw_model_clips_negative_predictions():
    X = np.array([[0.0], [1.0], [2.0]])
    model = ChargesRegressor().fit(X, [-10.0, 0.0, 10.0])
    assert model.predict(np.array([[-5.0]]))[0] == 0.0


def test_selection_rule_prefers_simpler_within_tolerance():
    table = pd.DataFrame(
        [
            {
                "model": "a",
                "selectable": True,
                "complexity": 1,
                "cv_mae_mean": 100.5,
                "cv_rmse_mean": 150,
            },
            {
                "model": "b",
                "selectable": True,
                "complexity": 2,
                "cv_mae_mean": 100.0,
                "cv_rmse_mean": 149,
            },
            {
                "model": "c",
                "selectable": True,
                "complexity": 3,
                "cv_mae_mean": 90.0,
                "cv_rmse_mean": 200,
            },
            {
                "model": "d",
                "selectable": False,
                "complexity": 0,
                "cv_mae_mean": 50.0,
                "cv_rmse_mean": 50,
            },
        ]
    )
    # c is excluded by the RMSE tolerance, d is comparison-only, a is within 1% MAE of b.
    assert select_model(table) == "a"


def test_metrics_are_deterministic():
    m = regression_metrics([100.0, 200.0, 300.0], [110.0, 190.0, 330.0])
    assert m["mae"] == pytest.approx(50 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert m["mean_signed_error"] == pytest.approx(10.0)


def test_subgroup_table_counts(synthetic_df):
    X = synthetic_df[list(FEATURES)]
    table = subgroup_table(X, synthetic_df["charges"], synthetic_df["charges"] + 10)
    smokers = table[(table.group == "smoker") & (table.level == "yes")].iloc[0]
    assert smokers["n"] == (X["smoker"] == "yes").sum()
    assert smokers["signed_mean_error"] == pytest.approx(10.0)


def test_raw_explanation_sums_to_estimate(synthetic_df):
    X, y = synthetic_df[list(FEATURES)], synthetic_df["charges"]
    pipe = CANDIDATES_BY_NAME["ols_enhanced"].build().fit(X, y)
    row = X.iloc[[3]]
    exp = explain.explain_prediction(pipe, row)
    total = exp["reference_estimate"] + sum(d["contribution"] for d in exp["drivers"])
    raw = pipe[-1].predict_link(pipe[:-1].transform(row))[0]
    assert total == pytest.approx(raw)
    assert exp["unit"] == "usd"


def test_coefficient_table_has_units(synthetic_df):
    X, y = synthetic_df[list(FEATURES)], synthetic_df["charges"]
    pipe = CANDIDATES_BY_NAME["log_enhanced"].build().fit(X, y)
    table = explain.coefficient_table(pipe)
    assert table["unit"].eq("log1p(USD)").all()
    assert "approx_pct_difference" in table.columns


def test_artifact_round_trip_and_schema_check(synthetic_df, tmp_path):
    X, y = synthetic_df[list(FEATURES)], synthetic_df["charges"]
    pipe = CANDIDATES_BY_NAME["ols_enhanced"].build().fit(X, y)
    meta = {"schema_version": "1.0", "input_features": list(FEATURES)}
    path = save_artifact(pipe, tmp_path / "m.joblib", meta)
    loaded, loaded_meta = load_artifact(path)
    np.testing.assert_allclose(loaded.predict(X), pipe.predict(X))
    assert loaded_meta == meta

    save_artifact(pipe, path, {**meta, "schema_version": "0.1"})
    with pytest.raises(ArtifactError, match="Incompatible artifact schema"):
        load_artifact(path)


def test_missing_artifact_message(tmp_path):
    with pytest.raises(ArtifactError, match="insurance_cost.train"):
        load_artifact(tmp_path / "missing.joblib")


def test_obesity_explanation_is_relative_to_reference(synthetic_df):
    """With the obesity step, contributions still sum exactly to the estimate."""
    X, y = synthetic_df[list(FEATURES)], synthetic_df["charges"]
    pipe = CANDIDATES_BY_NAME["ols_enhanced_obesity"].build().fit(X, y)
    for i in range(5):
        row = X.iloc[[i]]
        exp = explain.explain_prediction(pipe, row)
        total = exp["reference_estimate"] + sum(d["contribution"] for d in exp["drivers"])
        assert total == pytest.approx(pipe[-1].predict_link(pipe[:-1].transform(row))[0])


def test_comparison_models_are_added(synthetic_csv):
    result = train(synthetic_csv, None, None, None, candidates=FAST)
    names = set(result.cv_table["model"])
    assert {f"{result.selected}_no_sex", f"{result.selected}_ridge"} <= names
    assert not result.cv_table.set_index("model").loc[f"{result.selected}_no_sex", "selectable"]
