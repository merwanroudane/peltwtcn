"""Wavelet denoising (Section 3.2 of the paper)."""

from __future__ import annotations

import numpy as np
import pytest

from peltwtcn import (WaveletConfig, causal_wavelet_denoise, max_useful_level,
                      universal_threshold, wavelet_decompose, wavelet_denoise)


# --------------------------------------------------------------------------
# decomposition
# --------------------------------------------------------------------------
def test_decompose_returns_approximation_and_details(noisy_trend):
    _, noisy = noisy_trend
    out = wavelet_decompose(noisy, WaveletConfig(wavelet="db4", level=3))
    assert {"cA", "cD", "A", "D"} <= set(out)
    assert len(out["cD"]) == 3


def test_reconstruction_identity_holds(noisy_trend):
    r"""Equation (17): f = A_J f + sum_j D_j f, to numerical precision."""
    _, noisy = noisy_trend
    out = wavelet_decompose(noisy, WaveletConfig(wavelet="db4", level=3))
    total = np.asarray(out["A"]) + np.sum(out["D"], axis=0)
    assert np.allclose(total, noisy, atol=1e-8)


def test_level_one_is_the_papers_choice(noisy_trend):
    _, noisy = noisy_trend
    out = wavelet_decompose(noisy, WaveletConfig(level=1))
    assert len(out["cD"]) == 1


def test_max_useful_level_is_sane():
    assert max_useful_level(6113, "db4") >= 8
    assert max_useful_level(16, "db4") >= 0


# --------------------------------------------------------------------------
# denoising
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["paper", "threshold", "causal",
                                  "causal_threshold"])
def test_every_mode_preserves_length_and_is_finite(noisy_trend, mode):
    _, noisy = noisy_trend
    out = wavelet_denoise(noisy, WaveletConfig(denoise_mode=mode,
                                               causal_window=128))
    assert out.shape == noisy.shape
    assert np.all(np.isfinite(out))


def test_denoising_reduces_error_against_the_clean_signal(noisy_trend):
    clean, noisy = noisy_trend
    out = wavelet_denoise(noisy, WaveletConfig(wavelet="db4", level=1,
                                               denoise_mode="paper"))
    before = np.sqrt(np.mean((noisy - clean) ** 2))
    after = np.sqrt(np.mean((out - clean) ** 2))
    assert after < before


def test_denoising_is_smoother_than_the_input(noisy_trend):
    _, noisy = noisy_trend
    out = wavelet_denoise(noisy, WaveletConfig(level=1, denoise_mode="paper"))
    rough_in = np.mean(np.abs(np.diff(noisy)))
    rough_out = np.mean(np.abs(np.diff(out)))
    assert rough_out < rough_in


def test_denoise_mode_none_is_a_passthrough(noisy_trend):
    _, noisy = noisy_trend
    out = wavelet_denoise(noisy, WaveletConfig(denoise_mode="none"))
    assert np.allclose(out, noisy)


def test_paper_mode_leaks_the_future_and_causal_mode_does_not():
    """The heart of the replication caveat.

    A db4 level-1 filter has finite support, so the leak is *local* rather
    than global: perturbing x from t0 onwards changes the denoised value at a
    handful of points before t0.  That is still fatal for a one-step-ahead
    forecast, because the input at t already encodes x at t+1.  The causal
    variant must show no leak at all.
    """
    r = np.random.RandomState(11)
    x = np.cumsum(r.normal(0, 1, 400)) + 50.0
    t0 = 300

    x2 = x.copy()
    x2[t0:] += 25.0

    paper_a = wavelet_denoise(x, WaveletConfig(denoise_mode="paper"))
    paper_b = wavelet_denoise(x2, WaveletConfig(denoise_mode="paper"))
    leaked = np.nonzero(np.abs(paper_a - paper_b) > 1e-10)[0]
    look_ahead = t0 - int(leaked.min())
    assert look_ahead > 0, "the paper's filter should leak future information"
    assert look_ahead < 20, "the leak should be local to the filter support"

    causal_a = causal_wavelet_denoise(x, WaveletConfig(denoise_mode="causal",
                                                       causal_window=128))
    causal_b = causal_wavelet_denoise(x2, WaveletConfig(denoise_mode="causal",
                                                        causal_window=128))
    assert np.allclose(causal_a[:t0], causal_b[:t0], atol=1e-10), (
        "the causal filter must not see the future")


def test_paper_mode_leak_reaches_the_very_next_observation():
    """The minimal statement of the problem: denoised_t depends on x_{t+1}."""
    r = np.random.RandomState(12)
    x = np.cumsum(r.normal(0, 1, 256)) + 40.0
    cfg = WaveletConfig(wavelet="db4", level=1, denoise_mode="paper")

    base = wavelet_denoise(x, cfg)
    for t in (100, 150, 200):
        bumped = x.copy()
        bumped[t + 1] += 10.0
        assert not np.isclose(wavelet_denoise(bumped, cfg)[t], base[t]), (
            f"denoised value at t={t} should already contain x[{t + 1}]")


def test_causal_denoise_matches_a_manual_rolling_recomputation():
    """Recompute one point by hand and check the rolling window agrees."""
    r = np.random.RandomState(5)
    x = np.cumsum(r.normal(0, 1, 300)) + 30.0
    cfg = WaveletConfig(wavelet="db4", level=1, denoise_mode="causal",
                        causal_window=64)
    out = causal_wavelet_denoise(x, cfg)

    t = 200
    window = x[t - 64 + 1: t + 1]
    ref = wavelet_denoise(window, WaveletConfig(wavelet="db4", level=1,
                                                denoise_mode="paper"))
    assert np.isclose(out[t], ref[-1], atol=1e-8)


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------
def test_universal_threshold_scales_with_the_noise_level():
    r = np.random.RandomState(6)
    quiet = universal_threshold(r.normal(0, 1, 1000))
    loud = universal_threshold(r.normal(0, 8, 1000))
    assert loud > quiet > 0


def test_universal_threshold_of_an_empty_array_is_zero():
    assert universal_threshold(np.array([])) == 0.0


def test_universal_threshold_recovers_sigma_root_2_log_n():
    r = np.random.RandomState(8)
    d = r.normal(0.0, 3.0, 5000)
    expected = 3.0 * np.sqrt(2.0 * np.log(5000))
    assert universal_threshold(d) == pytest.approx(expected, rel=0.10)


@pytest.mark.parametrize("rule", ["universal", "sqtwolog", "minimax", "sure"])
def test_all_threshold_rules_run(noisy_trend, rule):
    _, noisy = noisy_trend
    out = wavelet_denoise(noisy, WaveletConfig(denoise_mode="threshold",
                                               threshold_rule=rule))
    assert out.shape == noisy.shape and np.all(np.isfinite(out))


def test_unknown_wavelet_raises():
    with pytest.raises(Exception):
        wavelet_denoise(np.arange(100.0), WaveletConfig(wavelet="not_a_wavelet"))


def test_config_roundtrips_through_as_dict():
    cfg = WaveletConfig(wavelet="sym8", level=2, denoise_mode="causal")
    assert WaveletConfig(**cfg.as_dict()) == cfg
