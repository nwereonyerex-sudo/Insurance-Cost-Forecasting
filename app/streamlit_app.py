"""Streamlit estimator for annual medical charges.

Run with: streamlit run app/streamlit_app.py

Submitted values are used for a single in-memory prediction and are never written to disk,
logs, analytics or any remote service.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from insurance_cost.artifacts import ArtifactError, load_artifact
from insurance_cost.config import ARTIFACT_PATH
from insurance_cost.inference import estimate, validate_inputs

INCREASE = "#eb6834"
DECREASE = "#2a78d6"

st.set_page_config(page_title="Medical Charges Estimator", page_icon="🩺", layout="centered")


@st.cache_resource
def get_model(path: str) -> tuple[object, dict]:
    return load_artifact(Path(path))


def usd(value: float) -> str:
    return f"{'−' if value < 0 else ''}${abs(value):,.0f}"


def md_usd(value: float) -> str:
    """Dollar amount for markdown text, where a bare $ would start a LaTeX formula."""
    return usd(value).replace("$", r"\$")


st.title("Medical charges estimator")
st.markdown(
    "Estimates **annual medical charges** in US dollars from six details, using a linear "
    "regression trained on a small public dataset (1,338 people). "
    "**This is not an insurance quote and not medical advice.**"
)

try:
    pipeline, meta = get_model(str(ARTIFACT_PATH))
except ArtifactError as exc:
    st.error(f"The model could not be loaded. {exc}")
    st.stop()

ranges = meta["train_ranges"]
levels = meta["category_levels"]
test = meta["test_metrics"]

with st.form("profile"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=int(ranges["age"][0]),
            max_value=int(ranges["age"][1]),
            value=35,
            step=1,
        )
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=float(ranges["bmi"][0]),
            max_value=float(ranges["bmi"][1]),
            value=26.0,
            step=0.1,
            format="%.1f",
            help="Body mass index = weight in kg ÷ (height in m)². "
            "Example: 70 kg and 1.75 m gives 22.9.",
        )
        children = st.number_input(
            "Children covered",
            min_value=int(ranges["children"][0]),
            max_value=int(ranges["children"][1]),
            value=0,
            step=1,
        )
    with col2:
        smoker = st.radio(
            "Smoker?",
            options=levels["smoker"],
            format_func={"no": "No", "yes": "Yes"}.get,
            horizontal=True,
        )
        sex = st.selectbox("Sex (as recorded in the dataset)", options=levels["sex"])
        region = st.selectbox("US region", options=levels["region"])
    submitted = st.form_submit_button("Estimate charges", type="primary")

if submitted:
    values = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }
    errors = validate_inputs(values, meta)
    if errors:
        for message in errors:
            st.error(message)
        st.stop()

    result = estimate(pipeline, meta, values)
    st.metric("Estimated annual medical charges", usd(result["estimate"]))
    st.markdown(
        f"On held-out test data, predictions were typically off by about "
        f"**{md_usd(test['mae'])}** (mean absolute error), and for some people by more "
        "than \\$15,000. This estimate reflects patterns in a small historical dataset and is "
        "not an insurance quote or medical advice."
    )

    st.subheader("What moved this estimate")
    st.caption(
        f"Starting point: {md_usd(result['reference_estimate'])} for the model's reference "
        "person (40-year-old female non-smoker, BMI 25, no children, northeast). "
        "Each bar shows how much the model's estimate differs because of that input. "
        "These are patterns the model learned, not causes."
    )
    drivers = pd.DataFrame(result["drivers"])
    drivers["direction"] = drivers["contribution"].map(
        lambda v: "Raises estimate" if v >= 0 else "Lowers estimate"
    )
    drivers["label"] = drivers["contribution"].map(lambda v: f"+{usd(v)}" if v >= 0 else usd(v))
    base = alt.Chart(drivers).encode(
        y=alt.Y(
            "driver:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelOverlap=False)
        ),
        x=alt.X("contribution:Q", title="Difference from reference estimate (USD)"),
        tooltip=[
            alt.Tooltip("driver:N", title="Input"),
            alt.Tooltip("label:N", title="Difference"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(
                domain=["Raises estimate", "Lowers estimate"], range=[INCREASE, DECREASE]
            ),
            legend=alt.Legend(title=None, orient="bottom"),
        )
    )
    text = base.mark_text(align="left", dx=4, color="gray").encode(text="label:N")
    chart = (bars + text).properties(height=44 * len(drivers))
    st.altair_chart(chart, use_container_width=True)

    comparison = result["smoker_comparison"]
    other = "a smoker" if comparison["smoker"] == "yes" else "a non-smoker"
    st.info(
        f"**Comparison:** for someone with the same details who is {other}, the model's "
        f"estimate is **{md_usd(comparison['estimate'])}**. This is a difference the model "
        "learned between groups in the data. It does not mean that starting or stopping "
        "smoking would change a particular person's charges by this amount."
    )

with st.expander("Model performance"):
    baseline = meta["baseline_test_metrics"]
    st.markdown(
        f"""
| Held-out test set ({meta["n_test"]} people) | This model | Predict-the-average baseline |
|---|---|---|
| Mean absolute error | {md_usd(test["mae"])} | {md_usd(baseline["mae"])} |
| Root mean squared error | {md_usd(test["rmse"])} | {md_usd(baseline["rmse"])} |
| R² | {test["r2"]:.3f} | {baseline["r2"]:.3f} |

Model `{meta["model_version"]}`: {meta["model_description"]}.
Trained on {meta["n_train"]} people (80% split, seed {meta["random_state"]}) at
{meta["trained_at"]}. Errors are larger for a group of high-cost non-smokers whose costs
the six inputs cannot explain. See the model card in the repository.
"""
    )

range_text = ", ".join(
    f"{name} {ranges[key][0]:g}–{ranges[key][1]:g}"
    for key, name in (("age", "age"), ("bmi", "BMI"), ("children", "children"))
)
with st.expander("Limitations and disclaimer", expanded=not submitted):
    st.markdown(
        f"""
- **Not a quote, diagnosis or guarantee.** The output estimates historical annual medical
  charges, not the price of any insurance policy.
- **Small, historical, undocumented dataset** (1,338 rows) that may not represent you,
  your country, the current year or any insurance product.
- **Associations, not causes.** The model describes patterns in the data; it cannot say
  what would happen if any of your details changed.
- **Large individual errors are possible.** Important factors such as health conditions
  and plan type are not inputs.
- Inputs are limited to the ranges seen in training: {range_text}.
- **Privacy:** what you enter is used for this one calculation and is not stored,
  logged or sent anywhere.
- Do not use this for medical, financial, eligibility or coverage decisions.
"""
    )
