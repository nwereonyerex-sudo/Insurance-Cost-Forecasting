from __future__ import annotations

from insurance_cost.evaluate import EDA_FIGURES, save_eda_figures


def test_eda_figures_render(synthetic_df, tmp_path):
    paths = save_eda_figures(synthetic_df, tmp_path)
    assert [p.stem for p in paths] == list(EDA_FIGURES)
    assert all(p.stat().st_size > 10_000 for p in paths)
