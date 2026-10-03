# Insurance Cost Forecasting

An explainable multiple-regression project that estimates **annual medical charges** from age, sex,
BMI, number of children, smoking status and US region, delivered as a Streamlit estimator.
It is a learning/portfolio demonstrator: **not an insurance quote and not medical advice.**

See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for requirements and [`RULE.md`](RULE.md) for project rules.

## Setup

Requires Python 3.11+ (developed on 3.14).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

> **macOS + Python 3.13/3.14 note:** if `import insurance_cost` fails after the editable install,
> the `.pth` file may have the macOS `hidden` flag, which recent Python versions skip. Fix with
> `chflags nohidden .venv/lib/python3.*/site-packages/*.pth`.

## Data

The [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
(1,338 rows) is not committed. Download `insurance.csv` from Kaggle and place it at:

```text
data/raw/insurance.csv
```

The raw file is never modified. Loading validates columns, types, missing values, ranges and
categories. The dataset contains one exact duplicate row; with no ID column it cannot be shown to be
an error, so it is kept and reported (`DEFAULT_DUPLICATE_POLICY = "keep"` in `insurance_cost.data`).

## Checks

```bash
pytest          # unit tests use synthetic fixtures; one integration test uses the real CSV if present
ruff check .
```
