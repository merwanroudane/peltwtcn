"""
Wavelet denoising for carbon price series.

Implements Section 3.2 of Ren et al. (2025), plus a leakage-free (causal)
alternative that is required for honest out-of-sample evaluation.

Two denoising regimes are provided
---------------------------------
``mode="paper"``
    Exactly what the paper describes: a single-level DWT is applied to the
    *whole* series, the detail (high-frequency) coefficients are discarded,
    and the signal is reconstructed from the approximation coefficients only.
    This is a two-sided (non-causal) filter: the denoised value at time t
    depends on y_{t+1}, y_{t+2}, ...  It therefore leaks future information
    into the model inputs.  Reproduced here for exact replication.

``mode="causal"``
    A rolling-window variant.  For every t the DWT is computed on the window
    y_{t-L+1..t} only and the *last* reconstructed point is kept.  No future
    observation ever touches the denoised value at t, so the resulting series
    is safe to use as a model input.

``mode="threshold"``
    Classical wavelet shrinkage (Donoho-Johnstone): detail coefficients are
    soft/hard thresholded rather than deleted.  Can be combined with
    ``causal=True``.

Author: Dr Merwan Roudane
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pywt

__all__ = [
    "WaveletConfig",
    "wavelet_denoise",
    "causal_wavelet_denoise",
    "universal_threshold",
    "wavelet_decompose",
    "max_useful_level",
]

DenoiseMode = Literal["paper", "threshold", "causal", "causal_threshold", "none"]
ThreshRule = Literal["universal", "sqtwolog", "minimax", "sure"]


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass
class WaveletConfig:
    """Container for every wavelet setting used by the pipeline.

    Parameters
    ----------
    wavelet : str
        Any wavelet name accepted by :func:`pywt.Wavelet`, e.g. ``"db4"``,
        ``"haar"``, ``"sym8"``, ``"coif3"``.  The paper does not report which
        family it used; ``"db4"`` is the de-facto standard in the carbon price
        literature and is the default here.
    level : int
        Number of decomposition levels.  The paper uses ``1`` ("a single level
        decomposition is chosen because excessive decomposition may lead to
        information loss").
    mode : str
        Signal-extension mode passed to PyWavelets (``"symmetric"``,
        ``"periodization"``, ``"zero"``, ``"reflect"``, ...).
    denoise_mode : DenoiseMode
        See module docstring.  ``"none"`` disables denoising entirely and is
        the ablation baseline.
    threshold_rule : ThreshRule
        Only used when ``denoise_mode`` involves thresholding.
    threshold_mode : {"soft", "hard"}
        Shrinkage type for :func:`pywt.threshold`.
    causal_window : int
        Length L of the rolling window used by the causal variants.  Must be
        at least ``2 ** level * (filter_length - 1)`` for the transform to be
        well defined; 256 is a safe default for db4/level 1.
    """

    wavelet: str = "db4"
    level: int = 1
    mode: str = "symmetric"
    denoise_mode: DenoiseMode = "paper"
    threshold_rule: ThreshRule = "universal"
    threshold_mode: Literal["soft", "hard"] = "soft"
    causal_window: int = 256

    def as_dict(self) -> dict:
        return dict(
            wavelet=self.wavelet,
            level=self.level,
            mode=self.mode,
            denoise_mode=self.denoise_mode,
            threshold_rule=self.threshold_rule,
            threshold_mode=self.threshold_mode,
            causal_window=self.causal_window,
        )


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def max_useful_level(n: int, wavelet: str = "db4") -> int:
    """Largest decomposition level that is still meaningful for ``n`` points."""
    return int(pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len))


def universal_threshold(detail: np.ndarray, n: int | None = None) -> float:
    r"""Donoho-Johnstone universal threshold.

    .. math::  \lambda = \hat\sigma \sqrt{2 \log n},
        \qquad \hat\sigma = \operatorname{median}(|d|) / 0.6745

    ``detail`` should be the finest-scale detail coefficients, from which the
    noise level is estimated by the (robust) median absolute deviation.
    """
    detail = np.asarray(detail, dtype=float)
    if detail.size == 0:
        return 0.0
    n = int(n if n is not None else detail.size)
    sigma = np.median(np.abs(detail)) / 0.6745
    return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))


def _threshold_value(detail: np.ndarray, n: int, rule: ThreshRule) -> float:
    detail = np.asarray(detail, dtype=float)
    sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0
    if rule in ("universal", "sqtwolog"):
        return float(sigma * np.sqrt(2.0 * np.log(max(n, 2))))
    if rule == "minimax":
        # Donoho-Johnstone minimax constant, tabulated approximation
        if n <= 32:
            lam = 0.0
        else:
            lam = 0.3936 + 0.1829 * (np.log(n) / np.log(2.0))
        return float(sigma * lam)
    if rule == "sure":
        # Stein unbiased risk estimate, applied to the normalised coefficients
        if detail.size == 0 or sigma == 0:
            return 0.0
        d = np.sort(np.abs(detail) / sigma) ** 2
        m = d.size
        risk = (m - 2.0 * np.arange(1, m + 1) + np.cumsum(d)
                + (m - np.arange(1, m + 1)) * d) / m
        return float(sigma * np.sqrt(d[int(np.argmin(risk))]))
    raise ValueError(f"unknown threshold rule: {rule!r}")


def _match_length(rec: np.ndarray, n: int) -> np.ndarray:
    """``waverec`` returns an even-length signal; trim/pad back to ``n``."""
    rec = np.asarray(rec, dtype=float)
    if rec.size == n:
        return rec
    if rec.size > n:
        return rec[:n]
    return np.concatenate([rec, np.full(n - rec.size, rec[-1])])


# ----------------------------------------------------------------------------
# decomposition
# ----------------------------------------------------------------------------
def wavelet_decompose(
    signal: Sequence[float] | np.ndarray,
    cfg: WaveletConfig | None = None,
    **kwargs,
) -> dict:
    r"""Multilevel DWT returning the approximation and every detail band.

    Implements equation (17) of the paper in its *finite* form

    .. math::  f = A_J f + \sum_{j=1}^{J} D_j f

    (the paper prints :math:`\sum_{j=1}^{\infty}`, which double counts the
    coarse scales already inside :math:`A_J`).

    Returns
    -------
    dict
        ``{"cA": approximation coefficients,
           "cD": [finest ... coarsest detail coefficient arrays],
           "A":  reconstructed approximation signal (length n),
           "D":  [reconstructed detail signals (length n)]}``
        so that ``A + sum(D) == signal`` up to numerical error.
    """
    cfg = _resolve(cfg, kwargs)
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    coeffs = pywt.wavedec(x, wavelet=cfg.wavelet, level=cfg.level, mode=cfg.mode)

    # reconstructed approximation: zero every detail band
    only_a = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    A = _match_length(pywt.waverec(only_a, wavelet=cfg.wavelet, mode=cfg.mode), n)

    D = []
    for j in range(1, len(coeffs)):
        band = [np.zeros_like(coeffs[0])] + [
            (coeffs[k] if k == j else np.zeros_like(coeffs[k]))
            for k in range(1, len(coeffs))
        ]
        D.append(_match_length(pywt.waverec(band, wavelet=cfg.wavelet, mode=cfg.mode), n))

    # pywt orders details coarsest-first; flip so D[0] is the finest scale D1
    D = D[::-1]
    return {"cA": coeffs[0], "cD": coeffs[1:][::-1], "A": A, "D": D}


# ----------------------------------------------------------------------------
# denoising
# ----------------------------------------------------------------------------
def wavelet_denoise(
    signal: Sequence[float] | np.ndarray,
    cfg: WaveletConfig | None = None,
    **kwargs,
) -> np.ndarray:
    """Denoise a 1-D series according to ``cfg.denoise_mode``.

    Examples
    --------
    >>> import numpy as np
    >>> from peltwtcn.wavelet import wavelet_denoise, WaveletConfig
    >>> y = np.cumsum(np.random.RandomState(0).randn(512)) + 50
    >>> yhat = wavelet_denoise(y, WaveletConfig(denoise_mode="paper"))
    >>> yhat.shape
    (512,)

    Notes
    -----
    ``"paper"`` and ``"threshold"`` are **non-causal**.  Use them only to
    replicate the published numbers, never to claim out-of-sample accuracy.
    ``"none"`` returns the input untouched, which is the "-RAW-" ablation used
    to measure what the wavelet stage actually contributes.
    """
    cfg = _resolve(cfg, kwargs)
    x = np.asarray(signal, dtype=float).ravel()

    if cfg.denoise_mode == "none":
        return x.copy()

    if cfg.denoise_mode in ("causal", "causal_threshold"):
        return causal_wavelet_denoise(x, cfg)

    n = x.size
    coeffs = pywt.wavedec(x, wavelet=cfg.wavelet, level=cfg.level, mode=cfg.mode)

    if cfg.denoise_mode == "paper":
        # "We retain only the approximation component as input to the model"
        new = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    elif cfg.denoise_mode == "threshold":
        lam = _threshold_value(coeffs[-1], n, cfg.threshold_rule)
        new = [coeffs[0]] + [
            pywt.threshold(c, lam, mode=cfg.threshold_mode) for c in coeffs[1:]
        ]
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(f"unknown denoise_mode: {cfg.denoise_mode!r}")

    return _match_length(pywt.waverec(new, wavelet=cfg.wavelet, mode=cfg.mode), n)


def causal_wavelet_denoise(
    signal: Sequence[float] | np.ndarray,
    cfg: WaveletConfig | None = None,
    **kwargs,
) -> np.ndarray:
    r"""Leakage-free rolling-window wavelet denoising.

    For each t the transform is applied to ``y[t-L+1 : t+1]`` and only the last
    reconstructed sample is retained, so

    .. math::  \tilde y_t = g\bigl(y_{t-L+1}, \dots, y_t\bigr)

    depends on the past and the present only.  The first ``L-1`` points are
    filled by expanding windows, which keeps the output length equal to the
    input length at the cost of a slightly noisier burn-in period.

    This is the estimator to use when the denoised series feeds a forecasting
    model, and it is what ``mode="causal"`` selects in
    :class:`~peltwtcn.pipeline.PELTWTPipeline`.
    """
    cfg = _resolve(cfg, kwargs)
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    L = int(cfg.causal_window)
    thresholding = cfg.denoise_mode == "causal_threshold"

    # minimum window for the transform to be defined at this level
    min_len = max(2 ** cfg.level, pywt.Wavelet(cfg.wavelet).dec_len)
    out = np.empty(n, dtype=float)

    for t in range(n):
        lo = max(0, t - L + 1)
        win = x[lo : t + 1]
        if win.size < min_len:
            out[t] = x[t]
            continue
        level = min(cfg.level, max_useful_level(win.size, cfg.wavelet))
        if level < 1:
            out[t] = x[t]
            continue
        coeffs = pywt.wavedec(win, wavelet=cfg.wavelet, level=level, mode=cfg.mode)
        if thresholding:
            lam = _threshold_value(coeffs[-1], win.size, cfg.threshold_rule)
            new = [coeffs[0]] + [
                pywt.threshold(c, lam, mode=cfg.threshold_mode) for c in coeffs[1:]
            ]
        else:
            new = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec = _match_length(
            pywt.waverec(new, wavelet=cfg.wavelet, mode=cfg.mode), win.size
        )
        out[t] = rec[-1]

    return out


# ----------------------------------------------------------------------------
def _resolve(cfg: WaveletConfig | None, kwargs: dict) -> WaveletConfig:
    if cfg is None:
        cfg = WaveletConfig()
    if kwargs:
        cfg = WaveletConfig(**{**cfg.as_dict(), **kwargs})
    return cfg
