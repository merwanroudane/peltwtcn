"""
Feature construction: the unified input vector z_t and the sliding windows.

Implements the input design of Section 3.3.1 of Ren et al. (2025),

.. math::  z_t = \\bigl[\\tilde y_t,\\; u_t,\\; e_t\\bigr]^\\top \\in \\mathbb{R}^d

where :math:`\\tilde y_t` is the wavelet-denoised carbon price, :math:`u_t` the
exogenous block and :math:`e_t` the one-hot regime indicator produced by the
break detector.

Scaling discipline
------------------
:class:`WindowScaler` is fitted on the **training window only** and then
applied to the test window.  The paper never states how it scaled, and fitting
a MinMax scaler on the full sample is a silent leak of the test-period maximum
into training - which for a series that triples in the last 20% of the sample
is not a small effect.

Author: Dr Merwan Roudane
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .breaks import BreakResult, one_hot_regimes

__all__ = [
    "build_design_matrix",
    "build_regime_matrix",
    "make_windows",
    "train_test_split_index",
    "WindowScaler",
    "SupervisedData",
]


# ---------------------------------------------------------------------------
@dataclass
class SupervisedData:
    """Windowed tensors ready for Keras, plus everything needed to invert them."""

    X_train: np.ndarray            # (n_train, window, n_features)
    y_train: np.ndarray            # (n_train,)
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str
    scaler_X: "WindowScaler"
    scaler_y: "WindowScaler"
    index_train: pd.Index
    index_test: pd.Index
    y_train_raw: np.ndarray        # unscaled targets, aligned with index_train
    y_test_raw: np.ndarray

    @property
    def window(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[2])

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        """Map scaled predictions back to price units."""
        return self.scaler_y.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SupervisedData(train={self.X_train.shape}, test={self.X_test.shape}, "
            f"window={self.window}, n_features={self.n_features}, "
            f"target={self.target_name!r})"
        )


# ---------------------------------------------------------------------------
class WindowScaler:
    """Min-max or standard scaler that is *only ever* fitted on training rows.

    Parameters
    ----------
    method : {"minmax", "standard", "none"}
    feature_range : tuple
        Output range for ``"minmax"``.

    Examples
    --------
    >>> import numpy as np
    >>> s = WindowScaler("minmax")
    >>> X = np.arange(10, dtype=float).reshape(-1, 1)
    >>> _ = s.fit(X[:5])
    >>> np.round(s.transform(X[:5]).ravel(), 3)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    >>> np.allclose(s.inverse_transform(s.transform(X)), X)
    True
    """

    def __init__(self, method: Literal["minmax", "standard", "none"] = "minmax",
                 feature_range: tuple[float, float] = (0.0, 1.0)):
        self.method = method
        self.feature_range = feature_range
        self.a_: np.ndarray | None = None
        self.b_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "WindowScaler":
        X = np.asarray(X, dtype=float)
        X = X.reshape(-1, X.shape[-1]) if X.ndim > 1 else X.reshape(-1, 1)
        if self.method == "minmax":
            lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
            rng = np.where((hi - lo) == 0, 1.0, hi - lo)
            f0, f1 = self.feature_range
            self.a_ = (f1 - f0) / rng
            self.b_ = f0 - lo * self.a_
        elif self.method == "standard":
            mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
            sd = np.where(sd == 0, 1.0, sd)
            self.a_, self.b_ = 1.0 / sd, -mu / sd
        else:
            self.a_ = np.ones(X.shape[1])
            self.b_ = np.zeros(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.a_ is None:
            raise RuntimeError("WindowScaler must be fitted first")
        X = np.asarray(X, dtype=float)
        return X * self.a_ + self.b_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.a_ is None:
            raise RuntimeError("WindowScaler must be fitted first")
        return (np.asarray(X, dtype=float) - self.b_) / self.a_


# ---------------------------------------------------------------------------
def build_regime_matrix(
    breaks: BreakResult | dict[str, BreakResult],
    n: int,
    encoding: Literal["onehot", "ordinal"] = "onehot",
    prefix: str = "regime",
) -> pd.DataFrame:
    r"""Turn break results into the model input :math:`e_t`.

    Accepts either a single :class:`~peltwtcn.breaks.BreakResult` (one
    segmentation of the carbon price) or the per-column dictionary returned by
    :func:`~peltwtcn.breaks.pelt_multivariate`, in which case one block of
    regime columns is produced per feature and the blocks are concatenated -
    this is what the paper means by "breakpoints were measured for each column
    of data".

    Examples
    --------
    >>> from peltwtcn.breaks import BreakResult, regimes_from_breakpoints
    >>> br = BreakResult([2], 5, "test", regimes_from_breakpoints(5, [2]), {})
    >>> build_regime_matrix(br, 5).values.tolist()
    [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    """
    if isinstance(breaks, BreakResult):
        breaks = {prefix: breaks}

    blocks: list[pd.DataFrame] = []
    for name, res in breaks.items():
        if res.n != n:
            raise ValueError(f"{name}: break result has n={res.n}, expected {n}")
        if encoding == "ordinal":
            blocks.append(pd.DataFrame({f"{name}_regime": res.labels.astype(float)}))
        else:
            oh = one_hot_regimes(res.labels, res.n_regimes)
            cols = [f"{name}_r{i}" for i in range(oh.shape[1])]
            blocks.append(pd.DataFrame(oh, columns=cols))
    return pd.concat(blocks, axis=1)


def build_design_matrix(
    denoised_price: Sequence[float] | np.ndarray | pd.Series,
    exog: pd.DataFrame | None = None,
    regimes: pd.DataFrame | None = None,
    index: pd.Index | None = None,
    price_name: str = "Carbon_Price_WT",
) -> pd.DataFrame:
    r"""Assemble :math:`z_t = [\tilde y_t,\, u_t,\, e_t]`.

    Parameters
    ----------
    denoised_price : array-like, shape (n,)
        Output of :func:`peltwtcn.wavelet.wavelet_denoise`.
    exog : DataFrame, optional
        Exogenous block :math:`u_t`.  Pass ``None`` for the univariate model.
    regimes : DataFrame, optional
        Output of :func:`build_regime_matrix`.  Pass ``None`` to drop
        :math:`e_t` (ablation).

    Returns
    -------
    DataFrame of shape (n, d), columns in the order price / exog / regimes.
    """
    y = np.asarray(denoised_price, dtype=float).ravel()
    n = y.size
    idx = pd.Index(index) if index is not None else pd.RangeIndex(n)
    if len(idx) != n:
        raise ValueError(f"index length {len(idx)} != series length {n}")

    Z = pd.DataFrame({price_name: y}, index=idx)
    if exog is not None and len(exog.columns):
        e = exog.copy()
        e.index = idx
        Z = pd.concat([Z, e], axis=1)
    if regimes is not None and len(regimes.columns):
        r = regimes.copy()
        r.index = idx
        Z = pd.concat([Z, r], axis=1)
    return Z


# ---------------------------------------------------------------------------
def train_test_split_index(n: int, train_size: float = 0.8) -> tuple[int, int]:
    """Chronological split point (the paper's 80/20).

    Examples
    --------
    >>> train_test_split_index(6113, 0.8)
    (4890, 1223)
    """
    n_train = int(np.floor(n * float(train_size)))
    return n_train, n - n_train


def make_windows(
    Z: pd.DataFrame,
    target: Sequence[float] | np.ndarray | pd.Series,
    window: int = 30,
    horizon: int = 1,
    stride: int = 1,
    train_size: float = 0.8,
    scale: Literal["minmax", "standard", "none"] = "minmax",
    scale_on: Literal["train", "all"] = "train",
    target_name: str = "Carbon_Price",
) -> SupervisedData:
    r"""Build the sliding-window tensors used by every model in the paper.

    For each t the model sees :math:`X_t = [z_{t-T+1}, \dots, z_t]` and must
    predict :math:`y_{t+h}` with :math:`T` = ``window`` (30 in the paper),
    :math:`h` = ``horizon`` (1) and step ``stride`` (1).

    Parameters
    ----------
    Z : DataFrame, shape (n, d)
        Design matrix from :func:`build_design_matrix`.
    target : array-like, shape (n,)
        Series to predict.  Pass the **raw** carbon price for an honest
        evaluation, or the denoised price to reproduce the paper.
    scale_on : {"train", "all"}
        ``"train"`` (default, correct) fits the scaler on training rows only.
        ``"all"`` reproduces the common - and leaky - practice of scaling the
        whole sample first; provided so the effect can be measured.

    Returns
    -------
    SupervisedData

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> Z = pd.DataFrame({"a": np.arange(200.0), "b": np.arange(200.0) * 2})
    >>> d = make_windows(Z, np.arange(200.0), window=10, train_size=0.8)
    >>> d.X_train.shape[1:], d.X_train.shape[0] + d.X_test.shape[0]
    ((10, 2), 190)
    """
    Zv = np.asarray(Z, dtype=float)
    yv = np.asarray(target, dtype=float).ravel()
    n = Zv.shape[0]
    if yv.size != n:
        raise ValueError(f"target length {yv.size} != design length {n}")
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be >= 1")
    if n <= window + horizon:
        raise ValueError("series too short for this window/horizon")

    n_train, _ = train_test_split_index(n, train_size)

    # scalers -----------------------------------------------------------------
    fit_slice = slice(0, n_train) if scale_on == "train" else slice(None)
    sx = WindowScaler(scale).fit(Zv[fit_slice])
    sy = WindowScaler(scale).fit(yv[fit_slice].reshape(-1, 1))
    Zs = sx.transform(Zv)
    ys = sy.transform(yv.reshape(-1, 1)).ravel()

    # windows -----------------------------------------------------------------
    starts = np.arange(0, n - window - horizon + 1, int(stride))
    X = np.stack([Zs[s: s + window] for s in starts])
    tgt_pos = starts + window + horizon - 1
    y = ys[tgt_pos]
    y_raw = yv[tgt_pos]
    idx = Z.index[tgt_pos] if isinstance(Z, pd.DataFrame) else pd.Index(tgt_pos)

    # A window is a training example only when its *target* falls before the
    # split, so no training row ever contains a test-period observation.
    is_train = tgt_pos < n_train
    is_test = ~is_train

    return SupervisedData(
        X_train=X[is_train], y_train=y[is_train],
        X_test=X[is_test], y_test=y[is_test],
        feature_names=list(map(str, Z.columns)) if isinstance(Z, pd.DataFrame)
        else [f"f{i}" for i in range(Zv.shape[1])],
        target_name=target_name,
        scaler_X=sx, scaler_y=sy,
        index_train=idx[is_train], index_test=idx[is_test],
        y_train_raw=y_raw[is_train], y_test_raw=y_raw[is_test],
    )
