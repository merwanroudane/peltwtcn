"""Design matrix, windowing and scaling (the z_t of Section 3.3.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peltwtcn import (BreakResult, SupervisedData, WindowScaler,
                      build_design_matrix, build_regime_matrix, make_windows,
                      regimes_from_breakpoints, train_test_split_index)


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------
def test_split_reproduces_the_papers_80_20():
    assert train_test_split_index(6113, 0.8) == (4890, 1223)


def test_split_parts_sum_to_n():
    for n in (100, 501, 6113):
        a, b = train_test_split_index(n, 0.8)
        assert a + b == n


# --------------------------------------------------------------------------
# design matrix
# --------------------------------------------------------------------------
def test_design_matrix_orders_price_then_exog_then_regimes():
    n = 20
    exog = pd.DataFrame({"u1": np.ones(n), "u2": np.zeros(n)})
    br = BreakResult([10], n, "t", regimes_from_breakpoints(n, [10]), {})
    reg = build_regime_matrix(br, n)
    Z = build_design_matrix(np.arange(float(n)), exog, reg)
    assert list(Z.columns) == ["Carbon_Price_WT", "u1", "u2",
                               "regime_r0", "regime_r1"]
    assert len(Z) == n


def test_design_matrix_univariate_is_price_only():
    Z = build_design_matrix(np.arange(10.0))
    assert Z.shape == (10, 1)


def test_design_matrix_rejects_a_mismatched_index():
    with pytest.raises(ValueError):
        build_design_matrix(np.arange(10.0), index=pd.RangeIndex(9))


def test_regime_matrix_rejects_a_length_mismatch():
    br = BreakResult([5], 10, "t", regimes_from_breakpoints(10, [5]), {})
    with pytest.raises(ValueError):
        build_regime_matrix(br, 11)


def test_regime_matrix_onehot_rows_sum_to_one():
    n = 30
    br = BreakResult([10, 20], n, "t", regimes_from_breakpoints(n, [10, 20]), {})
    reg = build_regime_matrix(br, n)
    assert reg.shape == (n, 3)
    assert np.allclose(reg.values.sum(axis=1), 1.0)


def test_regime_matrix_ordinal_encoding_is_one_column():
    n = 30
    br = BreakResult([10, 20], n, "t", regimes_from_breakpoints(n, [10, 20]), {})
    reg = build_regime_matrix(br, n, encoding="ordinal")
    assert reg.shape == (n, 1)


def test_regime_matrix_from_a_dict_concatenates_blocks():
    n = 20
    a = BreakResult([10], n, "a", regimes_from_breakpoints(n, [10]), {})
    b = BreakResult([5, 15], n, "b", regimes_from_breakpoints(n, [5, 15]), {})
    reg = build_regime_matrix({"a": a, "b": b}, n)
    assert reg.shape == (n, 2 + 3)


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------
def test_window_shapes_and_counts():
    Z = pd.DataFrame({"a": np.arange(200.0), "b": np.arange(200.0) * 2})
    d = make_windows(Z, np.arange(200.0), window=10, train_size=0.8)
    assert isinstance(d, SupervisedData)
    # 200 rows, a 10-step window and a 1-step horizon leave 190 usable windows
    assert d.X_train.shape[1:] == (10, 2)
    assert d.X_train.shape[0] + d.X_test.shape[0] == 190
    assert d.window == 10 and d.n_features == 2


def test_windows_are_chronological_with_no_overlap_across_the_split():
    n = 300
    Z = pd.DataFrame({"a": np.arange(float(n))})
    d = make_windows(Z, np.arange(float(n)), window=30, train_size=0.8)
    assert len(d.index_test) == d.X_test.shape[0]
    assert d.index_train[-1] < d.index_test[0]


def test_target_alignment_is_one_step_ahead():
    """y for window ending at t must be the value at t + horizon."""
    n = 100
    y = np.arange(float(n))
    Z = pd.DataFrame({"a": y})
    d = make_windows(Z, y, window=5, horizon=1, train_size=1.0, scale="none")
    # the first window covers indices 0..4 and must predict y[5]
    assert d.X_train[0, -1, 0] == pytest.approx(4.0)
    assert d.y_train[0] == pytest.approx(5.0)


def test_inverse_y_roundtrips():
    n = 200
    y = 10.0 + np.arange(float(n))
    Z = pd.DataFrame({"a": y})
    d = make_windows(Z, y, window=10, train_size=0.8, scale="minmax")
    assert np.allclose(d.inverse_y(d.y_test), d.y_test_raw, atol=1e-8)


def test_scaling_on_train_does_not_use_test_rows():
    """A scaler fitted on train only must leave test values outside [0, 1]."""
    n = 300
    y = np.arange(float(n))          # strictly increasing: test is out of range
    Z = pd.DataFrame({"a": y})
    d = make_windows(Z, y, window=10, train_size=0.8, scale="minmax",
                     scale_on="train")
    assert d.X_test.max() > 1.0


def test_scaling_on_all_leaks_and_stays_within_range():
    n = 300
    y = np.arange(float(n))
    Z = pd.DataFrame({"a": y})
    d = make_windows(Z, y, window=10, train_size=0.8, scale="minmax",
                     scale_on="all")
    assert d.X_test.max() <= 1.0 + 1e-9


def test_stride_reduces_the_number_of_windows():
    Z = pd.DataFrame({"a": np.arange(400.0)})
    one = make_windows(Z, np.arange(400.0), window=20, stride=1)
    two = make_windows(Z, np.arange(400.0), window=20, stride=4)
    assert two.X_train.shape[0] < one.X_train.shape[0]


# --------------------------------------------------------------------------
# scaler
# --------------------------------------------------------------------------
def test_minmax_scaler_maps_to_unit_interval():
    X = np.random.RandomState(0).normal(5.0, 3.0, (100, 4))
    s = WindowScaler("minmax").fit(X)
    Xs = s.transform(X)
    assert Xs.min() >= -1e-9 and Xs.max() <= 1 + 1e-9


def test_standard_scaler_centres_and_scales():
    X = np.random.RandomState(0).normal(5.0, 3.0, (500, 3))
    Xs = WindowScaler("standard").fit_transform(X)
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-6)


@pytest.mark.parametrize("method", ["minmax", "standard", "none"])
def test_scaler_inverse_transform_roundtrips(method):
    X = np.random.RandomState(1).normal(0.0, 2.0, (50, 3))
    s = WindowScaler(method).fit(X)
    assert np.allclose(s.inverse_transform(s.transform(X)), X, atol=1e-8)


def test_scaler_handles_a_constant_column():
    X = np.column_stack([np.ones(50), np.arange(50.0)])
    Xs = WindowScaler("minmax").fit_transform(X)
    assert np.all(np.isfinite(Xs))
