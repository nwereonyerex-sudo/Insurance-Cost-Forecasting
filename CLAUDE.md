# CLAUDE.md

## Project context

This repository is an explainable regression portfolio project that predicts **annual medical insurance charges** from the Medical Cost Personal Dataset. It does not predict an insurance premium and must never be presented as an insurance quote or medical advice.

Read these files before changing code:

1. `PROJECT_SPEC.md` — product requirements, pipeline, success criteria, and definition of done.
2. `RULE.md` — mandatory data, modelling, testing, interpretation, and safety rules.
3. `README.md` — actual setup and commands once it exists.

If code and documentation disagree, do not silently choose one. Identify the mismatch and align the implementation with the project owner's current request and `PROJECT_SPEC.md`.

## Intended stack

- Python 3.11+
- pandas and NumPy
- scikit-learn
- matplotlib and seaborn
- statsmodels for selected diagnostics, when useful
- joblib
- Streamlit
- pytest
- Ruff for linting/format checks

Do not introduce a new framework or model-serving layer unless it solves a demonstrated requirement.

## Expected layout

```text
data/raw/insurance.csv
notebooks/01_eda.ipynb
src/insurance_cost/{config,data,features,train,evaluate,explain,artifacts}.py
app/streamlit_app.py
tests/
reports/figures/
reports/model_card.md
models/
```

Create missing paths only when required by the active task. Do not scaffold unused files.

## Working method

1. Read the relevant requirement and nearby code/tests.
2. Check the working tree and preserve unrelated changes.
3. Make the smallest coherent change that completes the requirement.
4. Keep data preparation and prediction in one scikit-learn pipeline.
5. Add or update focused tests.
6. Run the relevant checks.
7. Update user-facing documentation when behaviour changes.
8. Summarize changed files, verification, and any remaining limitations.

## Planned commands

Use the commands declared by the repository when `pyproject.toml` exists. The intended interface is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
python -m insurance_cost.train --data data/raw/insurance.csv
streamlit run app/streamlit_app.py
```

Do not report a command as working until it has actually been run. If the repository later adopts `uv`, Poetry, Make, or another documented interface, use that interface consistently and update this file.

## Non-negotiable implementation constraints

- The raw target column is `charges` and business-facing evaluation is on the USD scale.
- Use a held-out test set and training-only cross-validation.
- Use `random_state=42` by default.
- Fit encoders, scalers, imputers, and feature-generation choices without test leakage.
- Preserve main effects when using `smoker × BMI` or `smoker × age` interactions.
- The Streamlit app loads the fitted pipeline and must not carry a second copy of preprocessing rules.
- Validate the artifact schema/version before inference.
- App copy must use “Estimated annual medical charges.”
- Explanation text must be associational, not causal.
- Do not store submitted app values.

## Model-selection guidance

Start with the mean baseline and ordinary least squares. Compare it with the specified enhanced linear and log-target models using five-fold cross-validation. Prefer the model that balances dollar-scale MAE/RMSE, residual behaviour, subgroup behaviour, and interpretability. Do not pick a model on R² alone.

When a log-target model is used:

- keep the transformation within the estimator pipeline where possible;
- use `log1p` and `expm1` safely;
- report metrics after inverse transformation;
- document retransformation bias and any correction;
- ensure final predictions cannot be displayed as negative charges.

## Required verification by change type

| Change | Minimum verification |
|---|---|
| Data validation | validator unit tests with valid and invalid fixtures |
| Feature engineering | exact-value and column-order tests |
| Model training | small end-to-end fit/predict test plus reproducibility check |
| Artifact format | save/load prediction-equivalence test |
| Metrics or diagnostics | deterministic metric test and visual smoke generation |
| Streamlit app | input-mapping unit test and valid-form smoke test |
| Documentation only | inspect links, paths, terminology, and internal consistency |

For a full delivery, run the complete test and lint suites. If data is unavailable, run tests that use fixtures and clearly report that real-data training was not verified.

## Completion response

At the end of a task, state:

- the outcome first;
- the files changed;
- checks actually run and their results;
- any unverified assumption or next required input.

Do not claim completion if tests fail, the real model artifact is missing for an app task, or the implementation contradicts `PROJECT_SPEC.md`.

