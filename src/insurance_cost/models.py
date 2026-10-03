"""Estimators and the candidate model specifications."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from insurance_cost.features import ENHANCED, MAIN_EFFECTS, FeatureSpec, build_preprocessor


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
ENHANCED_NO_SEX = FeatureSpec(age_sq=True, smoker_interactions=True, include_sex=False)

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
        "ridge_enhanced",
        "Ridge (alpha=1, standardised) on enhanced features — stability check",
        ENHANCED,
        "ridge",
        selectable=False,
    ),
    Candidate(
        "ols_enhanced_no_sex",
        "Enhanced OLS without sex — fairness comparison",
        ENHANCED_NO_SEX,
        "ols",
        selectable=False,
    ),
)

CANDIDATES_BY_NAME: dict[str, Candidate] = {c.name: c for c in CANDIDATES}
