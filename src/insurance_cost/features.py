"""Deterministic feature engineering and the preprocessing pipeline.

Age and BMI are centred on fixed constants (not learned from data) so that main-effect
coefficients describe a meaningful reference person instead of someone aged 0 with BMI 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from insurance_cost.config import CATEGORY_LEVELS, FEATURES

AGE_CENTER = 40.0
BMI_CENTER = 30.0


@dataclass(frozen=True)
class FeatureSpec:
    """Which engineered terms to include. Main effects are always kept."""

    age_sq: bool = False
    bmi_sq: bool = False
    smoker_interactions: bool = False
    include_sex: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


MAIN_EFFECTS = FeatureSpec()
ENHANCED = FeatureSpec(age_sq=True, smoker_interactions=True)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Add centred numeric terms, squares and smoker interactions to the raw columns.

    Stateless: ``fit`` only records the output column names, so there is nothing to leak.
    """

    def __init__(
        self,
        age_sq: bool = False,
        bmi_sq: bool = False,
        smoker_interactions: bool = False,
        include_sex: bool = True,
    ) -> None:
        self.age_sq = age_sq
        self.bmi_sq = bmi_sq
        self.smoker_interactions = smoker_interactions
        self.include_sex = include_sex

    @classmethod
    def from_spec(cls, spec: FeatureSpec) -> FeatureBuilder:
        return cls(**spec.to_dict())

    def numeric_columns(self) -> list[str]:
        cols = ["age_c", "bmi_c", "children"]
        if self.age_sq:
            cols.append("age_c_sq")
        if self.bmi_sq:
            cols.append("bmi_c_sq")
        if self.smoker_interactions:
            cols += ["smoker_x_age_c", "smoker_x_bmi_c"]
        return cols

    def categorical_columns(self) -> list[str]:
        return ["sex", "smoker", "region"] if self.include_sex else ["smoker", "region"]

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureBuilder:
        missing = [c for c in FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"FeatureBuilder missing input columns: {missing}")
        self.feature_names_out_ = np.array(self.numeric_columns() + self.categorical_columns())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        age_c = X["age"].astype(float) - AGE_CENTER
        bmi_c = X["bmi"].astype(float) - BMI_CENTER
        out = pd.DataFrame(
            {"age_c": age_c, "bmi_c": bmi_c, "children": X["children"].astype(float)},
            index=X.index,
        )
        if self.age_sq:
            out["age_c_sq"] = age_c**2
        if self.bmi_sq:
            out["bmi_c_sq"] = bmi_c**2
        if self.smoker_interactions:
            is_smoker = (X["smoker"] == "yes").astype(float)
            out["smoker_x_age_c"] = is_smoker * age_c
            out["smoker_x_bmi_c"] = is_smoker * bmi_c
        for col in self.categorical_columns():
            out[col] = X[col].astype(object)
        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.array(self.numeric_columns() + self.categorical_columns())


def build_preprocessor(spec: FeatureSpec, scale: bool = False) -> Pipeline:
    """Feature builder followed by one-hot encoding (first level is the reference)."""
    builder = FeatureBuilder.from_spec(spec)
    cat_cols = builder.categorical_columns()
    encoder = OneHotEncoder(
        categories=[list(CATEGORY_LEVELS[c]) for c in cat_cols],
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )
    numeric = StandardScaler() if scale else "passthrough"
    encode = ColumnTransformer(
        [("num", numeric, builder.numeric_columns()), ("cat", encoder, cat_cols)],
        verbose_feature_names_out=False,
    )
    return Pipeline([("build", builder), ("encode", encode)]).set_output(transform="pandas")
