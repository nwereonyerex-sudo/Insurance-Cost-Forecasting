"""Train, compare and select a model, evaluate it once on the test set, save the artifact.

Usage::

    python -m insurance_cost.train --data data/raw/insurance.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from insurance_cost import artifacts, evaluate, explain
from insurance_cost.config import (
    ARTIFACT_PATH,
    CV_FOLDS,
    DATA_PATH,
    FEATURES,
    FIGURES_DIR,
    METRICS_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from insurance_cost.data import DEFAULT_DUPLICATE_POLICY, file_fingerprint, load_raw, validate
from insurance_cost.models import (
    CANDIDATES,
    CANDIDATES_BY_NAME,
    Candidate,
    comparison_candidates,
)

logger = logging.getLogger(__name__)

# Selection rule, fixed before looking at the test set:
# 1. shortlist selectable models whose CV RMSE is within RMSE_TOLERANCE of the best CV RMSE;
# 2. choose the lowest CV MAE in the shortlist;
# 3. prefer a simpler model if its CV MAE is within SIMPLICITY_TOLERANCE of the chosen one.
RMSE_TOLERANCE = 0.05
SIMPLICITY_TOLERANCE = 0.01

SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class TrainingResult:
    selected: str
    pipeline: Pipeline
    cv_table: pd.DataFrame
    test_metrics: dict[str, float]
    baseline_test_metrics: dict[str, float]
    split: Split
    extras: dict[str, Any] = field(default_factory=dict)


def split_data(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> Split:
    """80/20 split stratified by smoking status (the dominant driver of charges)."""
    X = df.loc[:, list(FEATURES)]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=X["smoker"]
    )
    return Split(X_tr, X_te, y_tr, y_te)


def cross_validate_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: tuple[Candidate, ...] = CANDIDATES,
    folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Five-fold CV on training data only. All metrics are on the dollar scale."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    rows = []
    for cand in candidates:
        res = cross_validate(cand.build(), X, y, cv=cv, scoring=SCORING)
        rows.append(
            {
                "model": cand.name,
                "description": cand.description,
                "selectable": cand.selectable,
                "complexity": cand.complexity,
                "cv_mae_mean": -res["test_mae"].mean(),
                "cv_mae_std": res["test_mae"].std(),
                "cv_rmse_mean": -res["test_rmse"].mean(),
                "cv_rmse_std": res["test_rmse"].std(),
                "cv_r2_mean": res["test_r2"].mean(),
                "cv_r2_std": res["test_r2"].std(),
            }
        )
        logger.info(
            "CV %-24s MAE %.0f RMSE %.0f",
            cand.name,
            rows[-1]["cv_mae_mean"],
            rows[-1]["cv_rmse_mean"],
        )
    return pd.DataFrame(rows)


def select_model(cv_table: pd.DataFrame) -> str:
    """Apply the predefined selection rule to the CV table."""
    pool = cv_table[cv_table["selectable"]]
    best_rmse = pool["cv_rmse_mean"].min()
    shortlist = pool[pool["cv_rmse_mean"] <= best_rmse * (1 + RMSE_TOLERANCE)]
    best = shortlist.loc[shortlist["cv_mae_mean"].idxmin()]
    simpler = shortlist[
        (shortlist["complexity"] < best["complexity"])
        & (shortlist["cv_mae_mean"] <= best["cv_mae_mean"] * (1 + SIMPLICITY_TOLERANCE))
    ]
    if not simpler.empty:
        best = simpler.loc[simpler["complexity"].idxmin()]
    return str(best["model"])


def train(
    data_path: Path = DATA_PATH,
    artifact_path: Path | None = ARTIFACT_PATH,
    metrics_path: Path | None = METRICS_PATH,
    figures_dir: Path | None = FIGURES_DIR,
    candidates: tuple[Candidate, ...] = CANDIDATES,
) -> TrainingResult:
    """Run the full workflow. Paths set to ``None`` skip writing that output."""
    df = validate(load_raw(data_path), duplicate_policy=DEFAULT_DUPLICATE_POLICY)
    split = split_data(df)
    logger.info("Train rows: %d, test rows: %d", len(split.X_train), len(split.X_test))

    by_name = {c.name: c for c in candidates}
    cv_table = cross_validate_candidates(split.X_train, split.y_train, candidates)
    selected = select_model(cv_table)
    logger.info("Selected model: %s", selected)
    comparisons = cross_validate_candidates(
        split.X_train, split.y_train, comparison_candidates(by_name[selected])
    )
    cv_table = pd.concat([cv_table, comparisons], ignore_index=True)

    # The artifact is exactly the model evaluated on the test set: fitted on the training split.
    pipeline = by_name[selected].build().fit(split.X_train, split.y_train)
    baseline = CANDIDATES_BY_NAME["baseline_mean"].build().fit(split.X_train, split.y_train)

    test_pred = pipeline.predict(split.X_test)
    test_metrics = evaluate.regression_metrics(split.y_test, test_pred)
    baseline_test_metrics = evaluate.regression_metrics(
        split.y_test, baseline.predict(split.X_test)
    )

    subgroups = evaluate.subgroup_table(split.X_test, split.y_test, test_pred)
    largest = evaluate.largest_errors(split.X_test, split.y_test, test_pred)
    diagnostics = evaluate.residual_diagnostics(
        pipeline, split.X_train, split.y_train, split.X_test, split.y_test
    )
    train_ranges = {
        col: [float(split.X_train[col].min()), float(split.X_train[col].max())]
        for col in ("age", "bmi", "children")
    }

    result = TrainingResult(
        selected=selected,
        pipeline=pipeline,
        cv_table=cv_table,
        test_metrics=test_metrics,
        baseline_test_metrics=baseline_test_metrics,
        split=split,
        extras={
            "subgroups": subgroups,
            "largest_errors": largest,
            "diagnostics": diagnostics,
            "train_ranges": train_ranges,
            "coefficients": explain.coefficient_table(pipeline),
        },
    )

    if figures_dir is not None:
        evaluate.save_model_figures(result, figures_dir)
    if metrics_path is not None:
        write_metrics(result, metrics_path, data_path)
    if artifact_path is not None:
        artifacts.save_artifact(
            pipeline,
            artifact_path,
            artifacts.build_metadata(
                model_name=selected,
                candidate=by_name[selected],
                data_fingerprint=file_fingerprint(data_path),
                test_metrics=test_metrics,
                baseline_test_metrics=baseline_test_metrics,
                train_ranges=train_ranges,
                n_train=len(split.X_train),
                n_test=len(split.X_test),
            ),
        )
    return result


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def write_metrics(result: TrainingResult, path: Path, data_path: Path) -> None:
    payload = {
        "selected_model": result.selected,
        "selection_rule": {
            "rmse_tolerance": RMSE_TOLERANCE,
            "simplicity_tolerance": SIMPLICITY_TOLERANCE,
        },
        "data": {
            "path": str(Path(data_path).name),
            "sha256": file_fingerprint(data_path),
            "n_train": len(result.split.X_train),
            "n_test": len(result.split.X_test),
            "test_size": TEST_SIZE,
            "stratify": "smoker",
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
        },
        "cv": _records(result.cv_table),
        "test": result.test_metrics,
        "baseline_test": result.baseline_test_metrics,
        "subgroups_test": _records(result.extras["subgroups"]),
        "largest_test_errors": _records(result.extras["largest_errors"]),
        "diagnostics": result.extras["diagnostics"],
        "coefficients": _records(result.extras["coefficients"]),
        "train_ranges": result.extras["train_ranges"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("Wrote metrics to %s", path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = train(args.data, args.artifact, args.metrics, args.figures)
    cols = ["model", "cv_mae_mean", "cv_rmse_mean", "cv_r2_mean"]
    print("\nCross-validation (training data, USD):")
    print(result.cv_table[cols].round(3).to_string(index=False))
    print(f"\nSelected: {result.selected}")
    print("Test set (evaluated once):")
    for name, metrics in (
        ("model", result.test_metrics),
        ("baseline", result.baseline_test_metrics),
    ):
        print(
            f"  {name:8s} MAE ${metrics['mae']:,.0f}  RMSE ${metrics['rmse']:,.0f}  "
            f"R² {metrics['r2']:.3f}"
        )
    print(f"\nArtifact: {args.artifact}")


if __name__ == "__main__":
    main()
