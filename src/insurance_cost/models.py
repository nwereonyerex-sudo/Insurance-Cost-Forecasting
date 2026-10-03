"""Estimators and the candidate model specifications."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from insurance_cost.features import (
    ENHANCED,
    ENHANCED_OBESITY,
    MAIN_EFFECTS,
    FeatureSpec,
    build_preprocessor,
)


class ChargesRegressor(BaseEstimator, RegressorMixin):
    """Wrap a regressor so predictions are always non-negative dollars.

    With ``log_target`` the inner model is fitted on ``log1p(charges)`` and predictions are
    converted back with ``expm1``. ``expm1`` of the mean log prediction estimates the
    *median*, not the mean, so it underpredicts on average (retransformation bias).
    ``smearing=True`` applies Duan's smearing estimate, computed from training residuals,
    as a correction.
    """

    def __init__(
        self,
        regressor: BaseEstimator | None = None,
        log_target: bool = False,
        smearing: bool = False,
        clip_min: float = 0.0,
    ) -> None:
        self.regressor = regressor
        self.log_target = log_target
        self.smearing = smearing
        self.clip_min = clip_min

    def fit(self, X: object, y: object) -> ChargesRegressor:
        y = np.asarray(y, dtype=float)
        target = np.log1p(y) if self.log_target else y
        base = self.regressor if self.regressor is not None else LinearRegression()
        self.regressor_ = clone(base).fit(X, target)
        self.smearing_factor_ = 1.0
        if self.log_target and self.smearing:
            residuals = target - self.regressor_.predict(X)
            self.smearing_factor_ = float(np.mean(np.exp(residuals)))
        return self

    def predict_link(self, X: object) -> np.ndarray:
        """Prediction on the model's fitting scale (dollars, or log1p dollars)."""
        return np.asarray(self.regressor_.predict(X), dtype=float)

    def predict(self, X: object) -> np.ndarray:
        link = self.predict_link(X)
        if self.log_target:
            dollars = np.exp(link) * self.smearing_factor_ - 1.0
        else:
            dollars = link
        return np.maximum(dollars, self.clip_min)


@dataclass(frozen=True)
class Candidate:
    """A named model specification evaluated in cross-validation."""

    name: str
    description: str
    spec: FeatureSpec
    kind: str  # "baseline" | "ols" | "ridge"
    log_target: bool = False
    smearing: bool = False
    selectable: bool = True  # False for comparison-only models
    complexity: int = 0  # tie-breaker: lower is simpler

    def build(self) -> Pipeline:
        if self.kind == "baseline":
            inner: BaseEstimator = DummyRegressor(strategy="mean")
        elif self.kind == "ridge":
            inner = Ridge(alpha=1.0)
        else:
            inner = LinearRegression()
        model = ChargesRegressor(inner, log_target=self.log_target, smearing=self.smearing)
        scale = self.kind == "ridge"
        return Pipeline(
            [("preprocess", build_preprocessor(self.spec, scale=scale)), ("model", model)]
        )


ENHANCED_BMI_SQ = FeatureSpec(age_sq=True, bmi_sq=True, smoker_interactions=True)

CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "baseline_mean", "Predict the training mean", MAIN_EFFECTS, "baseline", selectable=False
    ),
    Candidate("ols_main", "OLS, main effects only", MAIN_EFFECTS, "ols", complexity=1),
    Candidate("ols_enhanced", "OLS + age², smoker×age, smoker×BMI", ENHANCED, "ols", complexity=2),
    Candidate(
        "log_enhanced",
        "Enhanced OLS on log1p(charges), no bias correction",
        ENHANCED,
        "ols",
        log_target=True,
        complexity=3,
    ),
    Candidate(
        "log_enhanced_smearing",
        "Enhanced OLS on log1p(charges), smearing correction",
        ENHANCED,
        "ols",
        log_target=True,
        smearing=True,
        complexity=4,
    ),
    Candidate("ols_enhanced_bmi_sq", "Enhanced OLS + BMI²", ENHANCED_BMI_SQ, "ols", complexity=5),
    Candidate(
        "ols_enhanced_obesity",
        "Enhanced OLS + obese (BMI ≥ 30) and smoker × obese",
        ENHANCED_OBESITY,
        "ols",
        complexity=6,
    ),
    Candidate(
        "log_enhanced_obesity",
        "Enhanced + obesity step on log1p(charges), no bias correction",
        ENHANCED_OBESITY,
        "ols",
        log_target=True,
        complexity=7,
    ),
)

CANDIDATES_BY_NAME: dict[str, Candidate] = {c.name: c for c in CANDIDATES}


def comparison_candidates(selected: Candidate) -> tuple[Candidate, ...]:
    """Comparison-only variants of the selected model: without sex, and a Ridge check."""
    no_sex = replace(selected.spec, include_sex=False)
    return (
        replace(
            selected,
            name=f"{selected.name}_no_sex",
            description=f"{selected.description}, without sex (fairness comparison)",
            spec=no_sex,
            selectable=False,
        ),
        replace(
            selected,
            name=f"{selected.name}_ridge",
            description=f"Ridge (alpha=1, standardised) on {selected.name} features",
            kind="ridge",
            selectable=False,
        ),
    )
