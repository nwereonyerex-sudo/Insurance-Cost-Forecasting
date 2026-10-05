"""Project-wide constants and default paths."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("INSURANCE_COST_ROOT", Path(__file__).resolve().parents[2]))

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "insurance.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACT_PATH = Path(
    os.environ.get("INSURANCE_COST_ARTIFACT", MODELS_DIR / "insurance_cost_model.joblib")
)
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Bump when the input columns, their meaning, or the artifact layout change.
SCHEMA_VERSION = "1.0"

TARGET = "charges"
NUMERIC_FEATURES: tuple[str, ...] = ("age", "bmi", "children")
CATEGORICAL_FEATURES: tuple[str, ...] = ("sex", "smoker", "region")
FEATURES: tuple[str, ...] = ("age", "sex", "bmi", "children", "smoker", "region")

# The first level of each category is the reference level for the linear model.
CATEGORY_LEVELS: dict[str, tuple[str, ...]] = {
    "sex": ("female", "male"),
    "smoker": ("no", "yes"),
    "region": ("northeast", "northwest", "southeast", "southwest"),
}

# Sanity bounds for validation. These are deliberately wider than the observed
# training range; the observed range is recorded in the artifact and enforced by the app.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (18, 120),
    "bmi": (10.0, 80.0),
    "children": (0, 20),
    "charges": (0.0, 1_000_000.0),
}
