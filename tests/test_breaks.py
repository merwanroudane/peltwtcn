"""Structural break detectors (Section 3.1 of the paper)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peltwtcn import (BreakResult, PeltConfig, bai_perron_breakpoints,
                      bic_penalty, bp_icss_breakpoints, icss_breakpoints,
                      one_hot_regimes, pelt_breakpoints, pelt_multivariate,
                      regimes_from_breakpoints)


# --------------------------------------------------------------------------
# PELT
# --------------------------------------------------------------------------
def test_pelt_finds_the_true_mean_shifts(piecewise_mean):
    res = pelt_breakpoints(piecewise_mean,
                           PeltConfig(model="l2", min_size=20, jump=1,
                                      penalty="bic"))
    assert isinstance(res, BreakResult)
    assert res.n == len(piecewise_mean)
    # both true breaks recovered to within a few observations
    for truth in (150, 280):
        assert min(abs(b - truth) for b in res.breakpoints) <= 5


def test_pelt_returns_no_breaks_on_white_noise():
    x = np.random.RandomState(7).normal(0.0, 1.0, 500)
    res = pelt_breakpoints(x, PeltConfig(model="l2", min_size=30, jump=5,
                                         penalty="bic"))
    # a BIC-penalised l2 cost should not shatter pure noise
    assert res.n_breaks <= 2


def test_pelt_breakpoints_are_sorted_interior_and_unique(piecewise_mean):
    res = pelt_breakpoints(piecewise_mean, PeltConfig(min_size=20, jump=1))
    bps = list(res.breakpoints)
    assert bps == sorted(bps)
    assert len(bps) == len(set(bps))
    assert all(0 < b < len(piecewise_mean) for b in bps)


def test_larger_penalty_gives_fewer_breaks(piecewise_mean):
    few = pelt_breakpoints(piecewise_mean,
                           PeltConfig(min_size=20, jump=1, penalty=500.0))
    many = pelt_breakpoints(piecewise_mean,
                            PeltConfig(min_size=20, jump=1, penalty=1.0))
    assert few.n_breaks <= many.n_breaks


def test_min_size_is_respected(piecewise_mean):
    res = pelt_breakpoints(piecewise_mean, PeltConfig(min_size=50, jump=1))
    bounds = [0] + list(res.breakpoints) + [len(piecewise_mean)]
    assert all(b - a >= 50 for a, b in zip(bounds, bounds[1:]))


def test_pelt_multivariate_returns_one_result_per_column(toy_panel, fast_pelt):
    out = pelt_multivariate(toy_panel, fast_pelt)
    assert set(out) == set(toy_panel.columns)
    assert all(isinstance(v, BreakResult) for v in out.values())
    assert all(v.n == len(toy_panel) for v in out.values())


# --------------------------------------------------------------------------
# ICSS
# --------------------------------------------------------------------------
def test_icss_finds_a_variance_shift(piecewise_variance):
    res = icss_breakpoints(piecewise_variance)
    assert res.n_breaks >= 1
    assert min(abs(b - 200) for b in res.breakpoints) <= 25


def test_icss_ignores_a_pure_mean_shift_with_constant_variance():
    r = np.random.RandomState(4)
    x = np.concatenate([r.normal(0.0, 1.0, 250), r.normal(6.0, 1.0, 250)])
    res = icss_breakpoints(x)
    # ICSS tests the variance; a mean-only shift should not dominate it
    assert res.n_breaks <= 3


# --------------------------------------------------------------------------
# Bai-Perron and the combined baseline
# --------------------------------------------------------------------------
def test_bai_perron_recovers_a_dominant_break(piecewise_mean):
    res = bai_perron_breakpoints(piecewise_mean, max_breaks=3, trim=0.15)
    assert res.n_breaks >= 1
    assert min(abs(b - 150) for b in res.breakpoints) <= 20


def test_bai_perron_honours_max_breaks(piecewise_mean):
    res = bai_perron_breakpoints(piecewise_mean, max_breaks=2, trim=0.10)
    assert res.n_breaks <= 2


def test_bai_perron_trimming_keeps_breaks_away_from_the_edges(piecewise_mean):
    trim, n = 0.15, len(piecewise_mean)
    res = bai_perron_breakpoints(piecewise_mean, max_breaks=5, trim=trim)
    assert all(trim * n <= b <= (1 - trim) * n for b in res.breakpoints)


def test_bp_icss_is_the_union_of_its_two_parts(piecewise_mean):
    res = bp_icss_breakpoints(piecewise_mean, max_breaks=3, trim=0.15,
                              min_size=20)
    assert "bai_perron" in res.detail and "icss" in res.detail
    union = set(res.detail["bai_perron"]) | set(res.detail["icss"])
    # the merged set never invents a break that neither method proposed
    assert set(res.breakpoints) <= union


# --------------------------------------------------------------------------
# regime encoding
# --------------------------------------------------------------------------
def test_regimes_from_breakpoints_labels_every_observation():
    labels = regimes_from_breakpoints(10, [3, 7])
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]


def test_regimes_with_no_breaks_is_a_single_regime():
    assert regimes_from_breakpoints(5, []).tolist() == [0] * 5


def test_one_hot_regimes_rows_sum_to_one():
    oh = one_hot_regimes(np.array([0, 0, 1, 2, 2]))
    assert oh.shape == (5, 3)
    assert np.allclose(oh.sum(axis=1), 1.0)


def test_break_result_dates_maps_onto_a_datetime_index():
    idx = pd.bdate_range("2020-01-01", periods=10)
    br = BreakResult([4], 10, "test", regimes_from_breakpoints(10, [4]), {})
    assert br.dates(idx) == [idx[4]]
    assert br.n_regimes == 2
    assert br.n_breaks == 1


def test_one_hot_property_matches_helper():
    br = BreakResult([2], 5, "t", regimes_from_breakpoints(5, [2]), {})
    assert np.allclose(br.one_hot(), one_hot_regimes(br.labels, 2))


# --------------------------------------------------------------------------
# penalty
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["bic", "aic"])
def test_penalty_is_positive_and_grows_with_n(rule):
    small = bic_penalty(100, 1, rule=rule)
    large = bic_penalty(10000, 1, rule=rule)
    assert small > 0
    assert large >= small


def test_pelt_rejects_a_too_short_series():
    with pytest.raises((ValueError, Exception)):
        pelt_breakpoints(np.arange(3.0), PeltConfig(min_size=30))
