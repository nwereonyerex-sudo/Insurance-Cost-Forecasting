"""Exploratory figures, metrics, residual diagnostics and subgroup analysis."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from insurance_cost.config import TARGET  # noqa: E402

logger = logging.getLogger(__name__)

# Validated categorical palette (slot 1 blue, slot 2 orange); text stays in neutral ink.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SMOKER_PALETTE = {"no": BLUE, "yes": ORANGE}
SMOKER_LABELS = {"no": "Non-smoker", "yes": "Smoker"}

usd = FuncFormatter(lambda x, _: f"${x:,.0f}")


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
