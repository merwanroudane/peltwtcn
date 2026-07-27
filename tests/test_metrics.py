"""Accuracy metrics and the two formal tests the paper omits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peltwtcn import (diebold_mariano, evaluate, evaluate_many, mae,
                      model_confidence_set, mape, mse, naive_random_walk, r2,
                      rmse, smape, theil_u)


# --------------------------------------------------------------------------
# point metrics
# --------------------------------------------------------------------------
def test_perfect_forecast_gives_zero_error_and_unit_r2():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert r2(y, y) == 1.0


def test_metrics_have_the_textbook_values():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 5.0])          # errors -1, 0, -2
    assert mae(y, p) == pytest.approx(1.0)
    assert mse(y, p) == pytest.approx(5.0 / 3.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(5.0 / 3.0))


def test_rmse_is_at_least_mae():
    r = np.random.RandomState(0)
    y, p = r.normal(size=200), r.normal(size=200)
    assert rmse(y, p) >= mae(y, p) - 1e-12


def test_mape_and_smape_are_percentages():
    y = np.full(50, 100.0)
    p = np.full(50, 110.0)
    assert mape(y, p) == pytest.approx(10.0)
    assert 0.0 < smape(y, p) < 100.0


def test_r2_of_the_mean_forecast_is_zero():
    r = np.random.RandomState(1)
    y = r.normal(10.0, 2.0, 500)
    assert r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-12)


def test_theil_u_of_the_random_walk_is_one():
    r = np.random.RandomState(2)
    y = np.cumsum(r.normal(size=300)) + 50.0
    assert theil_u(y, naive_random_walk(y)) == pytest.approx(1.0, rel=1e-9)


def test_theil_u_below_one_means_better_than_a_random_walk():
    r = np.random.RandomState(3)
    y = np.cumsum(r.normal(size=400)) + 50.0
    good = y + 0.01 * r.normal(size=400)
    assert theil_u(y, good) < 1.0


def test_naive_random_walk_is_a_one_step_lag():
    y = np.array([1.0, 2.0, 4.0, 8.0])
    rw = naive_random_walk(y)
    assert rw.shape == y.shape
    assert np.allclose(rw[1:], y[:-1])


def test_nans_are_dropped_pairwise():
    y = np.array([1.0, np.nan, 3.0])
    p = np.array([1.0, 2.0, 3.0])
    assert mae(y, p) == 0.0


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(Exception):
        mae(np.arange(5.0), np.arange(6.0))


# --------------------------------------------------------------------------
# tables of metrics
# --------------------------------------------------------------------------
def test_evaluate_returns_all_headline_keys():
    d = evaluate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "perfect")
    assert {"Model", "MAE", "RMSE", "MAPE (%)", "R2", "Theil U"} == set(d)
    assert d["Model"] == "perfect"


def test_evaluate_many_is_sorted_by_rmse_ascending():
    r = np.random.RandomState(4)
    y = r.normal(size=300)
    preds = {"bad": y + r.normal(0, 2, 300),
             "good": y + r.normal(0, 0.1, 300),
             "ok": y + r.normal(0, 0.7, 300)}
    t = evaluate_many(y, preds)
    assert list(t.index) == ["good", "ok", "bad"]
    assert t["RMSE"].is_monotonic_increasing


def test_evaluate_many_adds_training_times():
    y = np.arange(50.0)
    t = evaluate_many(y, {"m": y}, training_times={"m": 12.5})
    assert t.loc["m", "Train (s)"] == 12.5


# --------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------
def test_dm_detects_a_clearly_better_forecast():
    r = np.random.RandomState(0)
    y = r.randn(500)
    good, bad = y + 0.1 * r.randn(500), y + 1.0 * r.randn(500)
    res = diebold_mariano(y, good, bad)
    assert res["better"] == "A"
    assert res["DM"] < 0
    assert res["p_value"] < 0.01


def test_dm_is_antisymmetric_in_its_arguments():
    r = np.random.RandomState(1)
    y = r.randn(300)
    a, b = y + 0.2 * r.randn(300), y + 0.9 * r.randn(300)
    ab = diebold_mariano(y, a, b)
    ba = diebold_mariano(y, b, a)
    assert ab["DM"] == pytest.approx(-ba["DM"], rel=1e-9)
    assert ab["p_value"] == pytest.approx(ba["p_value"], rel=1e-9)


def test_dm_finds_no_difference_between_identical_forecasts():
    r = np.random.RandomState(2)
    y = r.randn(200)
    p = y + 0.3 * r.randn(200)
    res = diebold_mariano(y, p, p.copy())
    assert res["p_value"] > 0.99 or np.isnan(res["DM"])


def test_dm_rejects_a_tiny_sample():
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0], [1.0, 2.0], [2.0, 1.0])


@pytest.mark.parametrize("loss", ["mse", "mae"])
def test_dm_supports_both_loss_functions(loss):
    r = np.random.RandomState(3)
    y = r.randn(300)
    res = diebold_mariano(y, y + 0.1 * r.randn(300), y + r.randn(300),
                          loss=loss)
    assert res["p_value"] < 0.05


# --------------------------------------------------------------------------
# Model Confidence Set
# --------------------------------------------------------------------------
def test_mcs_keeps_the_best_model_and_drops_a_hopeless_one():
    r = np.random.RandomState(5)
    y = r.randn(400)
    preds = {"good": y + 0.05 * r.randn(400),
             "hopeless": y + 5.0 * r.randn(400)}
    out = model_confidence_set(y, preds, alpha=0.10, n_boot=300)
    assert out.loc["good", "in_MCS"]
    assert not out.loc["hopeless", "in_MCS"]


def test_mcs_output_shape_and_columns():
    r = np.random.RandomState(6)
    y = r.randn(200)
    preds = {f"m{i}": y + (0.1 * (i + 1)) * r.randn(200) for i in range(4)}
    out = model_confidence_set(y, preds, alpha=0.10, n_boot=200)
    assert len(out) == 4
    assert {"avg_loss", "p_MCS", "in_MCS"} <= set(out.columns)
    assert out["avg_loss"].is_monotonic_increasing


def test_mcs_never_returns_an_empty_set():
    r = np.random.RandomState(7)
    y = r.randn(200)
    preds = {f"m{i}": y + r.randn(200) for i in range(3)}
    out = model_confidence_set(y, preds, alpha=0.50, n_boot=200)
    assert out["in_MCS"].sum() >= 1
