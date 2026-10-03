"""Save and load the fitted pipeline together with its metadata."""

from __future__ import annotations

import logging
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from insurance_cost import __version__
from insurance_cost.config import (
    CATEGORY_LEVELS,
    FEATURES,
    RANDOM_STATE,
    SCHEMA_VERSION,
    TARGET,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

PACKAGES = ("scikit-learn", "pandas", "numpy", "joblib", "statsmodels")


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, malformed, or incompatible."""


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def build_metadata(
    *,
    model_name: str,
    candidate: Any,
    data_fingerprint: str,
    test_metrics: dict[str, float],
    baseline_test_metrics: dict[str, float],
    train_ranges: dict[str, list[float]],
    n_train: int,
    n_test: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": f"{__version__}+{model_name}",
        "model_name": model_name,
        "model_description": candidate.description,
        "feature_spec": candidate.spec.to_dict(),
        "log_target": candidate.log_target,
        "smearing": candidate.smearing,
        "input_features": list(FEATURES),
        "target": TARGET,
        "target_unit": "USD per year",
        "category_levels": {k: list(v) for k, v in CATEGORY_LEVELS.items()},
        "train_ranges": train_ranges,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "n_train": n_train,
        "n_test": n_test,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "data_sha256": data_fingerprint,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_versions": _package_versions(),
    }


def save_artifact(pipeline: Pipeline, path: Path, metadata: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, path)
    logger.info("Saved artifact to %s", path)
    return path


def load_artifact(path: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load and check the artifact before it is used for inference."""
    path = Path(path)
    if not path.exists():
        raise ArtifactError(
            f"Model artifact not found at {path}. "
            "Run `python -m insurance_cost.train --data data/raw/insurance.csv` first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or {"pipeline", "metadata"} - bundle.keys():
        raise ArtifactError("Artifact is malformed: expected 'pipeline' and 'metadata'")
    metadata = bundle["metadata"]
    found = metadata.get("schema_version")
    if found != SCHEMA_VERSION:
        raise ArtifactError(
            f"Incompatible artifact schema {found!r}; this code expects {SCHEMA_VERSION!r}. "
            "Retrain the model."
        )
    if metadata.get("input_features") != list(FEATURES):
        raise ArtifactError("Artifact input features do not match the current schema")
    return bundle["pipeline"], metadata


__all__ = ["ArtifactError", "build_metadata", "load_artifact", "save_artifact"]
