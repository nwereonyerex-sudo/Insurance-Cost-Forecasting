"""Loading and validating the raw Medical Cost Personal Dataset."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from insurance_cost.config import (
    CATEGORICAL_FEATURES,
    CATEGORY_LEVELS,
    DATA_PATH,
    FEATURES,
    NUMERIC_BOUNDS,
    NUMERIC_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)

DuplicatePolicy = Literal["keep", "drop", "error"]

# The dataset contains one exact duplicate row. With no identifier column we cannot tell
# a data-entry repeat from two people with identical attributes, so it is kept and reported.
DEFAULT_DUPLICATE_POLICY: DuplicatePolicy = "keep"

REQUIRED_COLUMNS: tuple[str, ...] = (*FEATURES, TARGET)
INTEGER_COLUMNS: tuple[str, ...] = ("age", "children")


class DataValidationError(ValueError):
    """Raised when the input data does not match the expected contract."""


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV without modifying the file, normalising names and string whitespace."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download insurance.csv from Kaggle "
            "(mirichoi0218/insurance) into data/raw/."
        )
    df = pd.read_csv(path)
    return normalise(df)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip column names; strip and lower-case string values."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: v.strip().lower() if isinstance(v, str) else v)
    return out


def validate(
    df: pd.DataFrame,
    duplicate_policy: DuplicatePolicy,
    require_target: bool = True,
) -> pd.DataFrame:
    """Check the data contract and return a typed copy.

    ``duplicate_policy`` must be chosen explicitly: ``keep`` logs duplicates, ``drop`` removes
    them, ``error`` raises.
    """
    required = REQUIRED_COLUMNS if require_target else FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    out = df.loc[:, list(required)].copy()
    problems: list[str] = []

    numeric_cols = [*NUMERIC_FEATURES, TARGET] if require_target else list(NUMERIC_FEATURES)
    for col in numeric_cols:
        parsed = pd.to_numeric(out[col], errors="coerce")
        bad = parsed.isna() & out[col].notna()
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} value(s) could not be parsed as numbers")
        out[col] = parsed

    nulls = out.isna().sum()
    for col, n in nulls[nulls > 0].items():
        problems.append(f"{col}: {int(n)} missing value(s)")

    for col in numeric_cols:
        lo, hi = NUMERIC_BOUNDS[col]
        values = out[col].dropna()
        outside = (values < lo) | (values > hi)
        if outside.any():
            problems.append(f"{col}: {int(outside.sum())} value(s) outside [{lo}, {hi}]")
        if col in INTEGER_COLUMNS and not (values % 1 == 0).all():
            problems.append(f"{col}: non-integer values found")

    for col in CATEGORICAL_FEATURES:
        allowed = set(CATEGORY_LEVELS[col])
        seen = set(out[col].dropna().astype(str).unique())
        unexpected = sorted(seen - allowed)
        if unexpected:
            problems.append(f"{col}: unexpected categories {unexpected}; allowed {sorted(allowed)}")

    if problems:
        raise DataValidationError("Data validation failed:\n- " + "\n- ".join(problems))

    for col in INTEGER_COLUMNS:
        out[col] = out[col].astype("int64")
    out["bmi"] = out["bmi"].astype("float64")
    if require_target:
        out[TARGET] = out[TARGET].astype("float64")

    n_dup = int(out.duplicated().sum())
    if n_dup:
        if duplicate_policy == "error":
            raise DataValidationError(f"{n_dup} duplicate row(s) found")
        if duplicate_policy == "drop":
            logger.info("Dropping %d duplicate row(s)", n_dup)
            out = out.drop_duplicates().reset_index(drop=True)
        else:
            logger.info("Keeping %d duplicate row(s) (policy=keep)", n_dup)
    return out


def validation_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Row count, dtypes, missing values, duplicates, numeric ranges and category levels."""
    numeric = df.select_dtypes("number")
    return {
        "rows": int(len(df)),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(n) for c, n in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_ranges": {
            c: {"min": float(numeric[c].min()), "max": float(numeric[c].max())}
            for c in numeric.columns
        },
        "category_levels": {
            c: {str(k): int(v) for k, v in df[c].value_counts().sort_index().items()}
            for c in CATEGORICAL_FEATURES
            if c in df.columns
        },
    }


def file_fingerprint(path: Path | str) -> str:
    """SHA-256 of the raw file, recorded in the model artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(
    path: Path | str = DATA_PATH, duplicate_policy: DuplicatePolicy = DEFAULT_DUPLICATE_POLICY
) -> pd.DataFrame:
    """Load and validate in one step."""
    return validate(load_raw(path), duplicate_policy=duplicate_policy)
