# Model card: Insurance Cost Forecasting

| | |
|---|---|
| **Model version** | `1.0.0+ols_enhanced_obesity` (artifact schema 1.0) |
| **Model type** | Ordinary least-squares linear regression in a scikit-learn `Pipeline` |
| **Target** | `charges`: annual medical charges in USD (**not** an insurance premium) |
| **Training command** | `python -m insurance_cost.train --data data/raw/insurance.csv` |
| **Data fingerprint** | SHA-256 `388eff67…21fd47` (recorded in full in the artifact and `reports/metrics.json`) |
| **Seed / split** | `random_state=42`; 80/20 train/test split stratified by smoking status (1,070 / 268 rows) |
| **Selection** | 5-fold cross-validation on the training split only |
| **Environment** | Python 3.14.6, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.3, statsmodels 0.15.0 |

## Intended use

A learning and portfolio demonstrator of explainable multiple regression. It shows how
demographic and lifestyle attributes were **associated** with annual medical charges in one
small historical dataset.

**Not intended for:** quoting or underwriting insurance, medical or financial decisions,
eligibility or coverage decisions, or any population, country, year or product not represented
in the data. Outputs are estimates of historical charges, not quotes, diagnoses or guarantees.

## Data

[Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance):
1,338 rows. Features are age, sex, BMI, number of children, smoker and US region; the target is charges.

- No missing values; all categories within the expected levels.
- One exact duplicate row, **kept**: with no ID column it cannot be shown to be an error.
- Observed training ranges: age 18–64, BMI 15.96–53.13 kg/m², children 0–5. The app refuses
  inputs outside these ranges.
- Smokers are 20% of rows (274). Rows with 4–5 children are rare (25 and 18).
- The source, collection method and year of the data are not documented. Treat it as
  unrepresentative of any specific real population.

## Features

All feature engineering is deterministic and lives inside the fitted pipeline, so the app
reuses it rather than reimplementing it.

| Term | Definition |
|---|---|
| `age_c`, `age_c_sq` | age − 40, and its square (curvature) |
| `bmi_c` | BMI − 25 |
| `children` | number of children covered |
| `obese` | 1 if BMI ≥ 30 |
| `smoker_x_age_c`, `smoker_x_bmi_c`, `smoker_x_obese` | smoker indicator × each term above |
| `sex_male`, `smoker_yes`, `region_*` | one-hot; reference levels **female, non-smoker, northeast** |

Main effects are kept for every interaction. The `obese` / `smoker × obese` terms were added in
spec v1.1 after EDA and residual diagnostics showed that, for smokers, charges step up at
BMI 30 rather than rising in a straight line (see *Model history*).

## Model selection (training data, 5-fold CV, USD)

Selection rule, written before the test set was scored: shortlist models within 5% of the best CV
RMSE, pick the lowest CV MAE, and prefer a simpler model if its MAE is within 1%.

| Model | CV MAE | CV RMSE | CV R² |
|---|---|---|---|
| `baseline_mean` | $9,062 ± 497 | $12,086 ± 583 | −0.002 ± 0.003 |
| `ols_main` (main effects) | $4,257 ± 210 | $6,202 ± 415 | 0.736 ± 0.016 |
| `ols_enhanced` (+ age², smoker×age, smoker×BMI) | $2,953 ± 206 | $4,870 ± 595 | 0.837 ± 0.027 |
| `log_enhanced` (log1p target) | $2,923 ± 244 | $5,639 ± 567 | 0.781 ± 0.031 |
| `log_enhanced_smearing` | $3,751 ± 219 | $6,175 ± 548 | 0.737 ± 0.037 |
| `ols_enhanced_bmi_sq` | $2,952 ± 196 | $4,863 ± 580 | 0.838 ± 0.026 |
| **`ols_enhanced_obesity` (selected)** | **$2,405 ± 219** | **$4,428 ± 627** | **0.865 ± 0.026** |
| `log_enhanced_obesity` | $2,541 ± 244 | $4,975 ± 603 | 0.830 ± 0.026 |
| `ols_enhanced_obesity_no_sex` *(comparison)* | $2,407 ± 226 | $4,437 ± 623 | 0.865 ± 0.026 |
| `ols_enhanced_obesity_ridge` *(stability check)* | $2,405 ± 222 | $4,428 ± 628 | 0.865 ± 0.026 |

± is the standard deviation across folds.

**Log target.** Fitting on `log1p(charges)` gives a competitive MAE but a clearly worse RMSE: in dollars it
under-predicts the expensive cases. Converting back with `expm1` estimates a *median*, not a mean
(retransformation bias). Duan's smearing correction raises every prediction by the same factor, which
over-corrects for the many low-cost people because the residuals are so skewed, and it made both metrics
worse. The raw-dollar model was preferred. Log-target predictions, when used, are converted back to
dollars and clipped at $0 inside the pipeline.

**Ridge** matches OLS to the dollar, so the OLS coefficients are stable and not over-fitted.

## Test-set performance (n = 268, USD)

| | MAE | RMSE | R² | Mean signed error |
|---|---|---|---|---|
| **Selected model** | **$2,227** | **$4,109** | **0.886** | +$180 |
| Mean baseline | $9,240 | $12,147 | −0.000 | +$73 |

The model beats the baseline on both MAE and RMSE, as the success criteria require. The gap between MAE and RMSE
reflects a small number of very large errors (see *Residual diagnostics*).

## Interpretation

**Reference profile:** a 40-year-old female non-smoker, BMI 25, no children, in the northeast.
The model's estimate for this person is **$8,341**. The coefficients below are differences relative
to that person, in USD per year, and describe associations in this dataset only.

| Term | Coefficient | Plain-language reading |
|---|---|---|
| Age | +$267 / year at 40, curvature +$3.5 / year² | estimates rise with age, a little faster at older ages |
| Children | +$661 per child | |
| BMI (non-smokers) | −$12 / point, `obese` +$38 | essentially no BMI association for non-smokers |
| Smoker (at 40, BMI 25) | +$12,880 | |
| Smoker × BMI | +$532 per BMI point above 25 | for smokers only |
| Smoker × obese | +$15,034 | additional step for smokers at BMI ≥ 30 |
| Smoker × age | −$12.5 / year | smokers' age slope is slightly flatter |
| Male (vs female) | −$740 | after accounting for the other inputs; see *Fairness* |
| Region (vs northeast) | NW −$469, SE −$949, SW −$1,394 | small; not evidence of regional risk |

**Interactions only make sense together.** Never read `smoker × obese` on its own. Conditional
model estimates for a 40-year-old female in the northeast with no children:

| BMI | Non-smoker | Smoker | Difference |
|---|---|---|---|
| 25 | $8,341 | $21,221 | +$12,880 |
| 29.9 | $8,282 | $23,769 | +$15,487 |
| 30 | $8,319 | $38,893 | +$30,574 |
| 35 | $8,258 | $41,493 | +$33,235 |

The model learned that among smokers, crossing BMI 30 is associated with roughly $15k higher
annual charges. Among non-smokers, BMI is associated with almost no difference. By age (BMI 25):
25-year-olds $5,123 vs $18,192, 60-year-olds $15,087 vs $27,717 (non-smoker vs smoker).
See `reports/figures/model_interactions.png`.

**Per-person explanations.** Because the model is linear in dollars, each estimate splits exactly
into the reference estimate plus one contribution per input. Smoking is grouped with its interactions.
Example: a 52-year-old male smoker with BMI 34 and 2 children in the southeast gets an estimate of $44,166: reference $8,341,
smoking +$32,553, age +$3,711, children +$1,322, region −$949, sex −$740, BMI −$71.

## Residual diagnostics (test set)

Figures: `diag_residuals_overview.png`, `diag_residuals_by_feature.png`.

- **Skew and influential errors.** Residuals are strongly right-skewed (skewness 3.5). 91% of
  test rows are predicted within $2k, but 7.5% are under-predicted by more than $5k (up to about $20k),
  and 85% of those are **non-smokers**. The largest test error is −$20,104 (row 806: 40-year-old female non-smoker, BMI 41).
  The EDA showed the same "middle band" of high-cost non-smokers: something not in the data
  (e.g. a health condition) drives these costs, and no function of these six inputs can predict it.
- **Systematic offset.** Because OLS fits the mean and those outliers pull it up, the typical
  person is *over*-predicted (median residual −$1,306) while the outliers are badly under-predicted.
- **Variance.** Breusch–Pagan p = 0.65 on the test residuals: no strong evidence of
  residual variance changing with the features once the obesity step is included.
- **Nonlinearity.** After the obesity step, residuals show no remaining pattern against
  age or BMI. Before it, smokers' residuals had a clear sawtooth around BMI 30.
- **Influence.** 72 training points exceed the 4/n Cook's distance rule of thumb (max 0.055,
  far below 1). No single point dominates the fit. No rows were removed.
- **Normality.** The Q–Q plot is far from normal in the upper tail. That matters for classical
  confidence intervals and p-values, which this project does not rely on. It does not invalidate the
  point predictions, which are judged on held-out error.

## Subgroup performance (test set)

Signed error = mean(predicted − actual); positive means the model over-predicts.

| Group | n | MAE | Signed error |
|---|---|---|---|
| Non-smoker | 213 | $2,412 | +$202 |
| Smoker | 55 | $1,510 | +$95 |
| Female | 136 | $2,076 | +$655 |
| Male | 132 | $2,382 | −$309 |
| Northeast | 62 | $2,398 | +$652 |
| Northwest | 65 | $2,388 | +$242 |
| Southeast | 73 | $2,357 | +$29 |
| Southwest | 68 | $1,777 | −$147 |

Smokers are now predicted *more* accurately than non-smokers in absolute dollars. Non-smoker
error is dominated by the unexplained high-cost group. Women are over-predicted on average and men
slightly under-predicted. Region subgroups have only 62–73 test rows each, so gaps of a few hundred
dollars are within noise.

## Fairness and responsible use

- **Sex.** Removing `sex` changes CV MAE by $2 (± $226 across folds): it adds essentially no
  predictive value. The −$740 male coefficient is an association conditional on the other inputs, not
  a property of men. Dropping `sex` would cost nothing measurable in accuracy and would avoid
  estimating charges from a protected attribute. **Owner decision (3 Oct 2026): keep `sex`** in the
  model to stay aligned with the dataset's full feature set. The trade-off is documented here, and
  the no-sex variant is retrained and reported on every training run (`*_no_sex` in `metrics.json`).
- **Region** differences are small and may partly stand in for BMI and smoking rates (the
  southeast has the highest average BMI and smoking rate). They should not be read as regional risk.
- **No causal claims.** Coefficients describe what the model learned from this dataset. They do not
  show that smoking, BMI, age or anything else *causes* a particular person's costs, or that changing
  an input would change that person's bill.
- **Privacy.** The app does not store, log or transmit what users enter.

## Limitations

- Small (1,338 rows), undocumented, historical dataset of unknown provenance and year.
- Six coarse inputs. Important drivers (health conditions, plan type, location within region,
  prices) are absent, which causes the large under-predictions for some non-smokers.
- Typical error is about $2.2k (MAE), but individual errors can exceed $20k. There is no validated
  prediction interval yet (optional extension).
- Inputs outside the training ranges are rejected rather than extrapolated.
- The BMI-30 step is a hard threshold. People just either side of 30 get very different smoker
  estimates, which reflects the data's pattern but is a simplification.

## Model history

| Version | Change | Evidence |
|---|---|---|
| spec 1.0 | Selected `ols_enhanced` (age², smoker×age, smoker×BMI). Test MAE $2,836, RMSE $4,567, R² 0.859 | CV MAE $2,953 |
| spec 1.1 | Added `obese` + `smoker × obese` with owner approval; selected `ols_enhanced_obesity`. Test MAE $2,227, RMSE $4,109, R² 0.886 | CV MAE $2,405; the smoker residual sawtooth was removed |

**Test-set use:** the held-out test set has been scored twice, once per spec version. The
feature change was motivated by EDA and residual patterns, and the choice between models was
made on training-data cross-validation only. Even so, the second test score is no longer a fully
untouched estimate and may be slightly optimistic.
