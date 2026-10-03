# Project Rules

These rules apply to every human or coding agent working on the Insurance Cost Forecasting project. `PROJECT_SPEC.md` defines the product; this file defines how changes must be made.

## 1. Instruction priority

1. Follow the project owner's current request.
2. Follow `PROJECT_SPEC.md` for intended behaviour and scope.
3. Follow this file for engineering rules.
4. Follow `CLAUDE.md` for agent workflow and commands.

If instructions conflict or a choice would materially change the model or product, stop and document the conflict instead of guessing.

## 2. Core principles

- Keep the work reproducible, explainable, and honest.
- Prefer the simplest model that meets the stated objective.
- Treat `charges` as annual medical insurance charges, not an insurance premium.
- Never describe associations as causal effects.
- Never present the app output as an insurance quote, medical advice, or a guarantee.
- Do not collect, transmit, or persist user form inputs.

## 3. Data rules

- Never edit the raw CSV in place.
- Do not commit restricted data, local downloads, credentials, or personal information.
- Validate required columns, dtypes, nulls, duplicates, numeric ranges, and category values before training.
- Split the data before fitting encoders, scalers, imputers, feature selectors, or models.
- Fit every learned transformation inside a scikit-learn pipeline using training data only.
- Do not remove outliers because they hurt metrics. Remove rows only for a documented data-quality reason.
- Do not change target meaning or units without updating the spec, app copy, metrics, and tests.

## 4. Modelling rules

- Establish the mean-prediction baseline first.
- Use `random_state=42` unless a documented experiment requires otherwise.
- Select models with training-set cross-validation; use the test set only for final evaluation.
- Keep main effects when adding their interaction terms.
- Create feature engineering in reusable code, not only in a notebook.
- Evaluate MAE, RMSE, and R² on the original dollar scale.
- Record the exact feature set and preprocessing with each saved artifact.
- Do not claim that a more complex model is better based on training score alone.
- Do not tune against the held-out test set.
- If the selected model changes, update the model card and provide evidence for the change.

## 5. Interpretation and fairness rules

- State coefficient units and reference categories.
- Explain interactions with conditional examples or plots.
- Use “associated with” or “the model learned,” not “causes.”
- Report subgroup sample size and error alongside subgroup metrics.
- Compare performance by smoking status and sex; inspect region where sample size supports it.
- Compare a model excluding `sex`, and document the performance/ethical trade-off.
- Do not expose row-level personal identifiers; the current dataset should not contain any.

## 6. Code rules

- Target Python 3.11 or later unless the repository declares another supported version.
- Put reusable code in `src/insurance_cost/`; keep notebooks thin.
- Use type hints on public functions and short docstrings where behaviour is not obvious.
- Keep functions focused and avoid hidden global state.
- Use `pathlib` for paths and configuration for environment-specific locations.
- Use `logging` in reusable code; reserve `print` for deliberate CLI output.
- Avoid hard-coded absolute paths, test metrics, category mappings, or model versions.
- Keep preprocessing in the serialized model pipeline; do not reimplement it in Streamlit.
- Pin or constrain direct dependencies in `pyproject.toml`.
- Never add secrets, tokens, `.env` files, raw data, model artifacts, or generated caches to source control unless explicitly intended and safe.

## 7. Testing rules

- Add or update tests for every behaviour change and bug fix.
- Include unit tests for validation and feature engineering.
- Include an artifact round-trip test.
- Include at least one end-to-end prediction test using a one-row DataFrame.
- Include a Streamlit smoke test where practical.
- Tests must use small fixtures or synthetic rows, not depend on a developer's local CSV unless explicitly marked as integration tests.
- Never weaken, skip, or delete a failing test merely to make the suite pass.

## 8. Documentation rules

- Keep `README.md` focused on setup, commands, results, and screenshots.
- Keep design requirements in `PROJECT_SPEC.md`.
- Keep model limitations, training context, and metrics in `reports/model_card.md`.
- Update documentation in the same change as the code it describes.
- Label exploratory observations separately from final test results.
- Include units, dataset split details, random seed, and model version in reported results.

## 9. Change discipline

Before editing:

- inspect the relevant files and current tests;
- preserve unrelated user changes;
- state assumptions that can affect scope or model meaning.

Before declaring work complete:

- run the most relevant tests and quality checks;
- inspect changed files for leaked paths, data, credentials, and stale claims;
- report what changed, what was verified, and what remains unverified.

## 10. Prohibited actions

- No fabricated metrics, charts, citations, predictions, or test results.
- No data leakage.
- No unreviewed network upload of data or user inputs.
- No destructive commands or broad file deletion without explicit authorization.
- No silent schema, target, model, or user-facing terminology changes.
- No medical, underwriting, or financial decision recommendation based on this demonstrator.

