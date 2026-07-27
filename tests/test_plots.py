"""Figures.  Rendered head-less; we assert on structure, not on pixels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peltwtcn import (BreakResult, WaveletConfig, plot_all_forecasts,
                      plot_breakpoints, plot_correlation_drivers,
                      plot_denoising, plot_dm_heatmap, plot_feature_importance,
                      plot_forecast, plot_model_comparison, plot_price_history,
                      plot_residual_density, plot_residuals_over_time,
                      plot_training_time, plot_wavelet_decomposition,
                      regimes_from_breakpoints, set_journal_style,
                      wavelet_decompose, wavelet_denoise)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module", autouse=True)
def _style():
    set_journal_style()


@pytest.fixture
def preds():
    r = np.random.RandomState(0)
    idx = pd.bdate_range("2022-01-01", periods=200)
    actual = 50.0 + np.cumsum(r.normal(0, 0.5, 200))
    return idx, actual, {
        "PELT-WT-TCN": actual + r.normal(0, 0.4, 200),
        "PELT-WT-GRU": actual + r.normal(0, 0.7, 200),
    }


# --------------------------------------------------------------------------
def test_set_journal_style_is_idempotent():
    set_journal_style()
    set_journal_style(font_scale=1.2, serif=False)
    assert plt.rcParams["figure.dpi"] > 0


def test_price_history_draws_the_series(toy_panel):
    fig = plot_price_history(toy_panel["Carbon_Price"])
    ax = fig.axes[0]
    assert len(ax.lines) >= 1
    assert ax.get_ylabel()


def test_price_history_marks_events(toy_panel):
    # events are (date, sign, description), as in datasets.POLICY_EVENTS
    events = [(str(toy_panel.index[100].date()), +1, "Event A"),
              (str(toy_panel.index[300].date()), -1, "Event B")]
    fig = plot_price_history(toy_panel["Carbon_Price"], events=events)
    assert len(fig.axes[0].lines) >= 3          # price plus two event markers


def test_correlation_plot_drops_the_targets_self_correlation(toy_panel):
    fig = plot_correlation_drivers(toy_panel, "Carbon_Price")
    assert len(fig.axes[0].patches) == len(toy_panel.columns) - 1


def test_feature_importance_has_one_bar_per_feature(toy_panel):
    fig = plot_feature_importance(toy_panel, "Carbon_Price")
    n_features = len(toy_panel.columns) - 1
    assert len(fig.axes[0].patches) == n_features


def test_breakpoints_plot_draws_a_line_per_break(toy_panel):
    n = len(toy_panel)
    br = BreakResult([100, 250], n, "PELT", regimes_from_breakpoints(n, [100, 250]), {})
    fig = plot_breakpoints(toy_panel["Carbon_Price"], br)
    # two dashed break lines plus the price line
    assert len(fig.axes[0].lines) >= 3


def test_breakpoints_plot_rejects_a_length_mismatch(toy_panel):
    br = BreakResult([10], 42, "PELT", regimes_from_breakpoints(42, [10]), {})
    with pytest.raises(ValueError, match="estimated on 42"):
        plot_breakpoints(toy_panel["Carbon_Price"], br)


def test_breakpoints_plot_accepts_a_per_feature_dict(toy_panel):
    n = len(toy_panel)
    groups = {c: BreakResult([120], n, c, regimes_from_breakpoints(n, [120]), {})
              for c in toy_panel.columns}
    fig = plot_breakpoints(toy_panel["Carbon_Price"], groups)
    assert fig.axes[0].get_legend() is not None


def test_denoising_plot_shows_raw_and_filtered(toy_panel):
    price = toy_panel["Carbon_Price"]
    wt = wavelet_denoise(price.to_numpy(float), WaveletConfig(level=1))
    fig = plot_denoising(price, wt)
    assert len(fig.axes[0].lines) >= 2


def test_denoising_plot_accepts_a_zoom_window(toy_panel):
    price = toy_panel["Carbon_Price"]
    wt = wavelet_denoise(price.to_numpy(float), WaveletConfig(level=1))
    lo, hi = str(price.index[50].date()), str(price.index[150].date())
    fig = plot_denoising(price, wt, zoom=(lo, hi))
    assert len(fig.axes) >= 1


def test_wavelet_decomposition_plot_has_a_panel_per_band(toy_panel):
    price = toy_panel["Carbon_Price"]
    dec = wavelet_decompose(price.to_numpy(float), WaveletConfig(level=3))
    fig = plot_wavelet_decomposition(price, dec)
    assert len(fig.axes) >= 4          # approximation + three details


def test_forecast_plot_draws_actual_and_predicted(preds):
    idx, actual, p = preds
    fig = plot_forecast(actual, p["PELT-WT-TCN"], idx, "PELT-WT-TCN")
    assert len(fig.axes[0].lines) >= 2
    assert fig.axes[0].get_legend() is not None


def test_all_forecasts_plot_draws_every_series(preds):
    idx, actual, p = preds
    fig = plot_all_forecasts(actual, p, idx)
    assert len(fig.axes[0].lines) >= len(p) + 1


def test_model_comparison_plot_groups_the_metrics():
    t = pd.DataFrame({"MAE": [1.2, 2.4], "RMSE": [1.6, 2.8], "R2": [0.99, 0.96]},
                     index=["TCN", "LSTM"])
    fig = plot_model_comparison(t)
    assert len(fig.axes[0].patches) >= 4


def test_training_time_plot_has_one_bar_per_model():
    fig = plot_training_time({"TCN": 52.5, "GRU": 14.3, "LSTM": 26.1})
    assert len(fig.axes[0].patches) == 3


def test_residuals_over_time_draws_a_line_per_model(preds):
    idx, actual, p = preds
    resid = pd.DataFrame({k: actual - v for k, v in p.items()}, index=idx)
    fig = plot_residuals_over_time(resid)
    assert len(fig.axes[0].lines) >= len(p)


def test_residual_density_draws_a_curve_per_model(preds):
    idx, actual, p = preds
    resid = pd.DataFrame({k: actual - v for k, v in p.items()}, index=idx)
    fig = plot_residual_density(resid)
    assert len(fig.axes[0].lines) + len(fig.axes[0].collections) >= len(p)


def test_dm_heatmap_renders_a_square_matrix():
    names = ["A", "B", "C"]
    dm = pd.DataFrame(np.full((3, 3), 0.2), index=names, columns=names)
    np.fill_diagonal(dm.values, np.nan)
    fig = plot_dm_heatmap(dm)
    assert len(fig.axes) >= 1


def test_figures_can_be_saved_to_disk(toy_panel, tmp_path):
    fig = plot_price_history(toy_panel["Carbon_Price"])
    out = tmp_path / "fig.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 1000


def test_plots_accept_an_existing_axis(toy_panel):
    fig, ax = plt.subplots()
    returned = plot_price_history(toy_panel["Carbon_Price"], ax=ax)
    assert returned is fig
