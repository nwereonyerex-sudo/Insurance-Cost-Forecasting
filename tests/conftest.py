from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows with the dataset's schema and a smoker x BMI interaction."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, n)
    bmi = rng.uniform(16, 50, n).round(2)
    children = rng.integers(0, 6, n)
    sex = rng.choice(["female", "male"], n)
    smoker = rng.choice(["no", "yes"], n, p=[0.8, 0.2])
    region = rng.choice(["northeast", "northwest", "southeast", "southwest"], n)
    is_smoker = smoker == "yes"
    charges = (
        2000
        + 250 * age
        + 50 * bmi
        + 400 * children
        + is_smoker * (12000 + 1400 * np.maximum(bmi - 30, 0) + 100 * bmi)
        + rng.normal(0, 1500, n)
    )
    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": np.maximum(charges, 500.0),
        }
    )


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture
def valid_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [19, 45, 60],
            "sex": ["female", "male", "female"],
            "bmi": [27.9, 33.0, 22.5],
            "children": [0, 2, 1],
            "smoker": ["yes", "no", "no"],
            "region": ["southwest", "northeast", "southeast"],
            "charges": [16884.92, 7000.0, 13000.0],
        }
    )
