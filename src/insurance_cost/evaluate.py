"""Exploratory figures, metrics, residual diagnostics and subgroup analysis."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: E402
from statsmodels.stats.outliers_influence import OLSInfluence  # noqa: E402

from insurance_cost.config import FEATURES, TARGET  # noqa: E402

if TYPE_CHECKING:
    from insurance_cost.train import TrainingResult

logger = logging.getLogger(__name__)

# Validated categorical palette (slot 1 blue, slot 2 orange); text stays in neutral ink.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SMOKER_PALETTE = {"no": BLUE, "yes": ORANGE}
SMOKER_LABELS = {"no": "Non-smoker", "yes": "Smoker"}

usd = FuncFormatter(lambda x, _: f"{'−' if x < 0 else ''}${abs(x):,.0f}")


def apply_style() -> None:
    """Quiet axes, recessive grid, readable type."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def _smoker_scatter(ax: plt.Axes, df: pd.DataFrame, x: str) -> None:
    for level in ("no", "yes"):
        sub = df[df["smoker"] == level]
        ax.scatter(
            sub[x],
            sub[TARGET],
            s=14,
            alpha=0.6,
            color=SMOKER_PALETTE[level],
            edgecolor="white",
            linewidth=0.4,
            label=f"{SMOKER_LABELS[level]} (n={len(sub)})",
        )
    ax.yaxis.set_major_formatter(usd)
    ax.legend(loc="upper left")


# --- Exploratory figures -------------------------------------------------------------


def fig_target_distribution(df: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df[TARGET], bins=40, color=BLUE, edgecolor="white")
    axes[0].set(
        title="Annual charges are right-skewed",
        xlabel="Annual medical charges (USD)",
        ylabel="Number of people",
    )
    axes[0].xaxis.set_major_formatter(usd)
    axes[1].hist(np.log1p(df[TARGET]), bins=40, color=BLUE, edgecolor="white")
    axes[1].set(
        title="Log scale is closer to symmetric",
        xlabel="log(1 + annual charges)",
        ylabel="Number of people",
    )
    return fig


def fig_numeric_relationships(df: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    _smoker_scatter(axes[0], df, "age")
    axes[0].set(
        title="Charges vs age",
        xlabel="Age (years)",
        ylabel="Annual medical charges (USD)",
    )
    _smoker_scatter(axes[1], df, "bmi")
    axes[1].set(title="Charges vs BMI", xlabel="BMI (kg/m²)")
    axes[1].axvline(30, color=INK_SECONDARY, linestyle="--", linewidth=1)
    axes[1].annotate("BMI 30", (30.4, 2_000), color=INK_SECONDARY)
    sns.boxplot(
        data=df,
        x="children",
        y=TARGET,
        ax=axes[2],
        color=BLUE,
        fill=False,
        linewidth=1.2,
        flierprops={"markersize": 3},
    )
    axes[2].set(title="Charges by number of children", xlabel="Children covered", ylabel="")
    return fig


def fig_categorical_groups(df: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    titles = {"smoker": "smoking status", "sex": "sex", "region": "US region"}
    for ax, col in zip(axes, ("smoker", "sex", "region"), strict=True):
        order = sorted(df[col].unique())
        sns.boxplot(
            data=df,
            x=col,
            y=TARGET,
            order=order,
            ax=ax,
            color=BLUE,
            fill=False,
            linewidth=1.2,
            flierprops={"markersize": 3},
        )
        counts = df[col].value_counts()
        ax.set_xticks(range(len(order)), [f"{lvl}\n(n={counts[lvl]})" for lvl in order])
        ax.set(
            title=f"Charges by {titles[col]}",
            xlabel=titles[col][0].upper() + titles[col][1:],
            ylabel="",
        )
    axes[0].set_ylabel("Annual medical charges (USD)")
    axes[0].yaxis.set_major_formatter(usd)
    return fig


def fig_bmi_by_smoker(df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    _smoker_scatter(ax, df, "bmi")
    for level in ("no", "yes"):
        sub = df[df["smoker"] == level]
        slope, intercept = np.polyfit(sub["bmi"], sub[TARGET], 1)
        xs = np.linspace(sub["bmi"].min(), sub["bmi"].max(), 50)
        ax.plot(xs, intercept + slope * xs, color=SMOKER_PALETTE[level], linewidth=2)
        ax.annotate(
            f"{SMOKER_LABELS[level]}: +${slope:,.0f} per BMI point",
            (xs[-1], intercept + slope * xs[-1]),
            xytext=(-5, -18 if level == "yes" else 8),
            textcoords="offset points",
            ha="right",
            color=INK,
        )
    ax.set(
        title="BMI is associated with much higher charges for smokers only",
        xlabel="BMI (kg/m²)",
        ylabel="Annual medical charges (USD)",
    )
    return fig


def fig_correlations(df: pd.DataFrame) -> Figure:
    cols = ["age", "bmi", "children", TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        square=True,
        linewidths=2,
        linecolor="white",
        cbar_kws={"label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Correlations among numeric variables")
    ax.grid(False)
    return fig


def fig_group_counts(df: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, col in zip(axes, ["smoker", "sex", "region", "children"], strict=True):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.to_numpy(), color=BLUE, width=0.7)
        for i, v in enumerate(counts.to_numpy()):
            ax.annotate(str(v), (i, v), xytext=(0, 3), textcoords="offset points", ha="center")
        ax.set(
            title=f"Rows by {col}", xlabel=col, ylabel="Number of people" if col == "smoker" else ""
        )
        ax.tick_params(axis="x", rotation=30 if col == "region" else 0)
    return fig


EDA_FIGURES: dict[str, Callable[[pd.DataFrame], Figure]] = {
    "eda_target_distribution": fig_target_distribution,
    "eda_numeric_relationships": fig_numeric_relationships,
    "eda_categorical_groups": fig_categorical_groups,
    "eda_bmi_by_smoker": fig_bmi_by_smoker,
    "eda_correlations": fig_correlations,
    "eda_group_counts": fig_group_counts,
}


def save_figure(fig: Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_eda_figures(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Render every exploratory figure to ``out_dir`` and return the paths."""
    apply_style()
    paths = [save_figure(make(df), out_dir, name) for name, make in EDA_FIGURES.items()]
    logger.info("Saved %d EDA figures to %s", len(paths), out_dir)
    return paths


# --- Model evaluation ----------------------------------------------------------------


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """MAE, RMSE, R² and mean signed error (prediction − actual), all in USD."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_signed_error": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def subgroup_table(
    X: pd.DataFrame,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    columns: tuple[str, ...] = ("smoker", "sex", "region"),
) -> pd.DataFrame:
    """Per-group n, MAE and signed mean error (positive = model overpredicts)."""
    frame = X.loc[:, list(columns)].copy()
    frame["actual"] = np.asarray(y_true, dtype=float)
    frame["predicted"] = np.asarray(y_pred, dtype=float)
    frame["error"] = frame["predicted"] - frame["actual"]
    rows = []
    for col in columns:
        for level, grp in frame.groupby(col, sort=True):
            rows.append(
                {
                    "group": col,
                    "level": str(level),
                    "n": int(len(grp)),
                    "mae": float(grp["error"].abs().mean()),
                    "signed_mean_error": float(grp["error"].mean()),
                    "mean_actual": float(grp["actual"].mean()),
                }
            )
    return pd.DataFrame(rows)


def largest_errors(
    X: pd.DataFrame, y_true: ArrayLike, y_pred: ArrayLike, k: int = 10
) -> pd.DataFrame:
    """The ``k`` largest absolute errors, identified only by row index and feature values."""
    out = X.copy()
    out.insert(0, "row_index", X.index)
    out["actual"] = np.asarray(y_true, dtype=float)
    out["predicted"] = np.asarray(y_pred, dtype=float)
    out["error"] = out["predicted"] - out["actual"]
    return (
        out.reindex(out["error"].abs().sort_values(ascending=False).index)
        .head(k)
        .reset_index(drop=True)
    )


def residual_diagnostics(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Numeric checks for skew, unequal variance and influential training points."""
    test_resid = np.asarray(y_test, dtype=float) - pipeline.predict(X_test)
    design_test = sm.add_constant(pipeline[:-1].transform(X_test).to_numpy(), has_constant="add")
    _, bp_pvalue, _, _ = het_breuschpagan(test_resid, design_test)

    design_train = sm.add_constant(pipeline[:-1].transform(X_train).to_numpy(), has_constant="add")
    model = pipeline[-1]
    target = np.log1p(y_train) if model.log_target else np.asarray(y_train, dtype=float)
    cooks = OLSInfluence(sm.OLS(target, design_train).fit()).cooks_distance[0]
    threshold = 4 / len(X_train)
    return {
        "test_residual_skew": float(stats.skew(test_resid)),
        "test_residual_mean": float(np.mean(test_resid)),
        "breusch_pagan_pvalue": float(bp_pvalue),
        "cooks_threshold": float(threshold),
        "cooks_n_above_threshold": int((cooks > threshold).sum()),
        "cooks_max": float(cooks.max()),
        "smearing_factor": float(getattr(model, "smearing_factor_", 1.0)),
    }


# --- Model figures -------------------------------------------------------------------


def fig_cv_comparison(cv_table: pd.DataFrame) -> Figure:
    order = cv_table.sort_values("cv_mae_mean", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, metric, label in zip(axes, ("mae", "rmse"), ("MAE", "RMSE"), strict=True):
        ax.barh(
            order["model"],
            order[f"cv_{metric}_mean"],
            xerr=order[f"cv_{metric}_std"],
            color=BLUE,
            height=0.6,
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 3},
        )
        ends = order[f"cv_{metric}_mean"] + order[f"cv_{metric}_std"]
        for i, (v, end) in enumerate(zip(order[f"cv_{metric}_mean"], ends, strict=True)):
            ax.annotate(
                f"${v:,.0f}", (end, i), xytext=(6, 0), textcoords="offset points", va="center"
            )
        ax.set_xlim(0, ends.max() * 1.18)
        ax.set(title=f"5-fold CV {label} (training data)", xlabel=f"{label} (USD, lower is better)")
        ax.xaxis.set_major_formatter(usd)
        ax.grid(axis="y", visible=False)
    return fig


def fig_residuals_overview(y_true: ArrayLike, y_pred: ArrayLike) -> Figure:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.scatter(y_pred, resid, s=14, alpha=0.6, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axhline(0, color=INK_SECONDARY, linewidth=1)
    ax.set(
        title="Residuals vs fitted (test set)",
        xlabel="Predicted charges (USD)",
        ylabel="Residual: actual − predicted (USD)",
    )
    ax.xaxis.set_major_formatter(usd)
    ax.yaxis.set_major_formatter(usd)

    ax = axes[0, 1]
    ax.scatter(y_true, y_pred, s=14, alpha=0.6, color=BLUE, edgecolor="white", linewidth=0.4)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, color=INK_SECONDARY, linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set(
        title="Actual vs predicted (test set)",
        xlabel="Actual charges (USD)",
        ylabel="Predicted charges (USD)",
        xlim=lim,
        ylim=lim,
    )
    ax.xaxis.set_major_formatter(usd)
    ax.yaxis.set_major_formatter(usd)
    ax.legend(loc="upper left")

    ax = axes[1, 0]
    ax.hist(resid, bins=40, color=BLUE, edgecolor="white")
    ax.set(title="Residual distribution", xlabel="Residual (USD)", ylabel="Number of people")
    ax.xaxis.set_major_formatter(usd)

    ax = axes[1, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, s=12, color=BLUE)
    ax.plot(osm, slope * osm + intercept, color=INK_SECONDARY, linewidth=1)
    ax.set(
        title="Normal Q–Q plot of residuals",
        xlabel="Theoretical normal quantiles",
        ylabel="Residual quantiles (USD)",
    )
    ax.yaxis.set_major_formatter(usd)
    fig.tight_layout()
    return fig


def fig_residuals_by_feature(X: pd.DataFrame, y_true: ArrayLike, y_pred: ArrayLike) -> Figure:
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    frame = X.assign(resid=resid)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, col, label in zip(
        axes[:2], ("age", "bmi"), ("Age (years)", "BMI (kg/m²)"), strict=True
    ):
        for level in ("no", "yes"):
            sub = frame[frame["smoker"] == level]
            ax.scatter(
                sub[col],
                sub["resid"],
                s=14,
                alpha=0.6,
                color=SMOKER_PALETTE[level],
                edgecolor="white",
                linewidth=0.4,
                label=SMOKER_LABELS[level],
            )
        ax.axhline(0, color=INK_SECONDARY, linewidth=1)
        ax.set(title=f"Residuals vs {col}", xlabel=label)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Residual: actual − predicted (USD)")
    axes[0].yaxis.set_major_formatter(usd)
    axes[1].axvline(30, color=INK_SECONDARY, linestyle="--", linewidth=1)
    sns.boxplot(
        data=frame,
        x="smoker",
        y="resid",
        order=["no", "yes"],
        ax=axes[2],
        hue="smoker",
        palette=SMOKER_PALETTE,
        fill=False,
        linewidth=1.2,
        flierprops={"markersize": 3},
        legend=False,
    )
    counts = frame["smoker"].value_counts()
    axes[2].set_xticks(
        [0, 1], [f"{SMOKER_LABELS[s]}\n(n={counts.get(s, 0)})" for s in ("no", "yes")]
    )
    axes[2].axhline(0, color=INK_SECONDARY, linewidth=1)
    axes[2].set(title="Residuals by smoking status", xlabel="", ylabel="")
    return fig


def fig_interactions(pipeline: Pipeline, train_ranges: dict[str, list[float]]) -> Figure:
    """Model predictions across BMI and age for smokers vs non-smokers, others held fixed."""
    reference = {"age": 40, "sex": "female", "bmi": 30.0, "children": 0, "region": "northeast"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, col, label in zip(axes, ("bmi", "age"), ("BMI (kg/m²)", "Age (years)"), strict=True):
        lo, hi = train_ranges[col]
        grid = np.linspace(lo, hi, 60)
        for level in ("no", "yes"):
            rows = pd.DataFrame([{**reference, "smoker": level, col: v} for v in grid])
            rows["age"] = rows["age"].round().astype(int)
            pred = pipeline.predict(rows[list(FEATURES)])
            ax.plot(
                grid, pred, color=SMOKER_PALETTE[level], linewidth=2, label=SMOKER_LABELS[level]
            )
            ax.annotate(
                f"{SMOKER_LABELS[level]}",
                (grid[-1], pred[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                color=INK,
            )
        held = "age 40" if col == "bmi" else "BMI 30"
        ax.set(
            title=f"Model estimate vs {col.upper() if col == 'bmi' else col} ({held})",
            xlabel=label,
        )
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Estimated annual charges (USD)")
    axes[0].yaxis.set_major_formatter(usd)
    fig.text(
        0.01,
        -0.03,
        "Other inputs fixed at the reference profile: female, 0 children, northeast. "
        "Lines show what the model learned, not causal effects.",
        color=INK_SECONDARY,
        fontsize=9,
    )
    return fig


def fig_subgroups(subgroups: pd.DataFrame) -> Figure:
    labels = [
        f"{g}={lvl} (n={n})"
        for g, lvl, n in subgroups[["group", "level", "n"]].itertuples(index=False)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    y = np.arange(len(subgroups))
    axes[0].barh(y, subgroups["mae"], color=BLUE, height=0.6)
    axes[0].set(title="Test MAE by subgroup", xlabel="MAE (USD)")
    axes[1].barh(y, subgroups["signed_mean_error"], color=BLUE, height=0.6)
    axes[1].axvline(0, color=INK_SECONDARY, linewidth=1)
    axes[1].set(
        title="Signed mean error by subgroup", xlabel="Mean of predicted − actual (USD; + = over)"
    )
    for ax in axes:
        ax.xaxis.set_major_formatter(usd)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    return fig


def save_model_figures(result: TrainingResult, out_dir: Path) -> list[Path]:
    """Render CV comparison, residual diagnostics, interaction and subgroup figures."""
    apply_style()
    X_te, y_te = result.split.X_test, result.split.y_test
    pred = result.pipeline.predict(X_te)
    figures = {
        "model_cv_comparison": fig_cv_comparison(result.cv_table),
        "diag_residuals_overview": fig_residuals_overview(y_te, pred),
        "diag_residuals_by_feature": fig_residuals_by_feature(X_te, y_te, pred),
        "model_interactions": fig_interactions(result.pipeline, result.extras["train_ranges"]),
        "model_subgroups": fig_subgroups(result.extras["subgroups"]),
    }
    paths = [save_figure(fig, out_dir, name) for name, fig in figures.items()]
    logger.info("Saved %d model figures to %s", len(paths), out_dir)
    return paths
