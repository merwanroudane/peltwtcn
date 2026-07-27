"""
Structural break detection: PELT, ICSS and Bai-Perron.

Implements Section 3.1 of Ren et al. (2025).  All three detectors return the
same object -- a sorted list of break indices -- which is converted into
regime labels r_t and then into the one-hot vector e_t that enters the deep
learning input z_t = [y_tilde_t, u_t, e_t].

Correctness notes (the paper's equations are not all right)
-----------------------------------------------------------
* Equation (9) of the paper is the *Optimal Partitioning* recursion, which is
  O(n^2).  PELT is that recursion **plus** the pruning rule

      F(s) + C(y_{s+1:t}) + K >= F(t)  =>  s can never be optimal later,

  which the paper never states.  ``ruptures`` implements the pruning; the
  O(n) claim in the paper is the *expected* cost when the number of change
  points grows linearly with n, not a worst-case bound.

* Equation (5) of the paper is not the Inclan-Tiao statistic.  The correct
  ICSS construction, implemented in :func:`icss_breakpoints`, is

      C_k = sum_{i<=k} a_i^2,  D_k = C_k / C_T - k/T,
      IT  = max_k sqrt(T/2) |D_k|,   critical value 1.358 at 5%,

  where a_t are the mean-centred observations.

Author: Dr Merwan Roudane
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import ruptures as rpt

__all__ = [
    "PeltConfig",
    "pelt_breakpoints",
    "pelt_multivariate",
    "icss_breakpoints",
    "bai_perron_breakpoints",
    "bp_icss_breakpoints",
    "regimes_from_breakpoints",
    "one_hot_regimes",
    "bic_penalty",
    "BreakResult",
]

CostModel = Literal["l1", "l2", "rbf", "normal", "linear", "ar", "mahalanobis"]

# Inclan & Tiao (1994) asymptotic critical values for sup |D_k| * sqrt(T/2)
ICSS_CRITICAL = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}

# ---------------------------------------------------------------------------
# Detection is the slow stage: exact PELT with jump=1 on 6,113 points takes
# about 90 s, and run_experiment() would otherwise repeat the identical
# segmentation once per model.  Results are memoised on the content of the
# signal plus the settings, so a full five-model experiment pays for it once.
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def clear_cache() -> None:
    """Empty the break-detection memo cache."""
    _CACHE.clear()


def _signature(x: np.ndarray, *params) -> tuple:
    a = np.ascontiguousarray(x, dtype=float)
    return (a.shape, hash(a.tobytes()), params)


# ----------------------------------------------------------------------------
@dataclass
class BreakResult:
    """Outcome of a break-detection run."""

    breakpoints: list[int]          # interior break indices (0-based, exclusive end)
    n: int                          # length of the series
    method: str
    labels: np.ndarray              # regime label per observation, 0..n_regimes-1
    detail: dict                    # method-specific extras

    @property
    def n_regimes(self) -> int:
        return int(self.labels.max()) + 1 if self.labels.size else 0

    @property
    def n_breaks(self) -> int:
        return len(self.breakpoints)

    def one_hot(self) -> np.ndarray:
        """(n, n_regimes) one-hot matrix e_t."""
        return one_hot_regimes(self.labels, self.n_regimes)

    def to_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        idx = index if index is not None else pd.RangeIndex(self.n)
        return pd.DataFrame({"regime": self.labels}, index=idx)

    def dates(self, index: pd.Index) -> list:
        """Break locations expressed on a user index (e.g. dates)."""
        return [index[b] for b in self.breakpoints if 0 <= b < len(index)]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BreakResult(method={self.method!r}, n={self.n}, "
            f"n_breaks={self.n_breaks}, n_regimes={self.n_regimes})"
        )


@dataclass
class PeltConfig:
    """PELT settings.

    The paper reports none of these, so the defaults below are stated
    explicitly and documented rather than hidden.

    Parameters
    ----------
    model : str
        Segment cost.  ``"l2"`` = change in mean (piecewise constant),
        ``"normal"`` = change in mean and/or variance (Gaussian likelihood,
        this is the "negative log-likelihood" the paper alludes to),
        ``"rbf"`` = kernel cost for distributional change.
    min_size : int
        Minimum segment length.  Guards against the over-segmentation that is
        clearly visible in Figure 7 of the paper.
    jump : int
        Grid subsampling.  ``1`` = consider every index (exact, slower).
    penalty : float | str
        Either a numeric beta, or ``"bic"`` / ``"mbic"`` / ``"aic"`` to derive
        it from the data via :func:`bic_penalty`.
    """

    model: CostModel = "l2"
    min_size: int = 30
    jump: int = 1
    penalty: float | Literal["bic", "mbic", "aic"] = "bic"

    def as_dict(self) -> dict:
        return dict(model=self.model, min_size=self.min_size,
                    jump=self.jump, penalty=self.penalty)


# ----------------------------------------------------------------------------
# penalties
# ----------------------------------------------------------------------------
def bic_penalty(n: int, n_params: int = 1, rule: str = "bic",
                sigma2: float = 1.0) -> float:
    r"""Information-criterion penalty :math:`\beta` for the PELT objective.

    ``bic``   : :math:`\beta = p \sigma^2 \log n`
    ``mbic``  : modified BIC of Zhang & Siegmund (2007), :math:`3 p \sigma^2 \log n`
    ``aic``   : :math:`\beta = 2 p \sigma^2`
    """
    n = max(int(n), 2)
    if rule == "bic":
        return float(n_params * sigma2 * np.log(n))
    if rule == "mbic":
        return float(3.0 * n_params * sigma2 * np.log(n))
    if rule == "aic":
        return float(2.0 * n_params * sigma2)
    raise ValueError(f"unknown penalty rule: {rule!r}")


# ----------------------------------------------------------------------------
# PELT
# ----------------------------------------------------------------------------
def pelt_breakpoints(
    signal: Sequence[float] | np.ndarray | pd.Series,
    cfg: PeltConfig | None = None,
    **kwargs,
) -> BreakResult:
    """Detect multiple change points with PELT (Killick et al., 2012).

    Parameters
    ----------
    signal : array-like, shape (n,) or (n, d)
        Univariate or multivariate signal.
    cfg : PeltConfig
        Cost model, minimum segment size, jump and penalty.

    Returns
    -------
    BreakResult

    Examples
    --------
    >>> import numpy as np
    >>> from peltwtcn.breaks import pelt_breakpoints, PeltConfig
    >>> rng = np.random.RandomState(0)
    >>> x = np.r_[rng.randn(200), rng.randn(200) + 6.0]
    >>> res = pelt_breakpoints(x, PeltConfig(model="l2", min_size=30, penalty=50))
    >>> res.n_breaks
    1
    >>> abs(res.breakpoints[0] - 200) <= 3
    True
    """
    cfg = _resolve_pelt(cfg, kwargs)
    x = _as_2d(signal)
    n = x.shape[0]

    key = ("pelt", _signature(x, cfg.model, cfg.min_size, cfg.jump, cfg.penalty))
    if key in _CACHE:
        return _CACHE[key]

    pen = cfg.penalty
    if isinstance(pen, str):
        # scale the penalty by the residual variance so that it is unit-free
        sigma2 = float(np.mean(np.var(x, axis=0))) or 1.0
        pen = bic_penalty(n, n_params=x.shape[1], rule=pen, sigma2=sigma2)

    algo = rpt.Pelt(model=cfg.model, min_size=cfg.min_size, jump=cfg.jump).fit(x)
    bkps = algo.predict(pen=float(pen))
    interior = [int(b) for b in bkps if 0 < b < n]

    res = BreakResult(
        breakpoints=interior,
        n=n,
        method=f"PELT[{cfg.model}]",
        labels=regimes_from_breakpoints(n, interior),
        detail={**cfg.as_dict(), "penalty_rule": cfg.penalty,
                "penalty": float(pen)},
    )
    _CACHE[key] = res
    return res


def pelt_multivariate(
    frame: pd.DataFrame,
    cfg: PeltConfig | None = None,
    columns: Iterable[str] | None = None,
    **kwargs,
) -> dict[str, BreakResult]:
    """Run PELT column by column.

    The paper states "PELT was chosen because breakpoints were measured for
    each column of data", i.e. every feature gets its own segmentation.  This
    helper reproduces that and returns one :class:`BreakResult` per column,
    which :func:`peltwtcn.features.build_regime_matrix` can then stack.
    """
    cfg = _resolve_pelt(cfg, kwargs)
    cols = list(columns) if columns is not None else list(frame.columns)
    return {c: pelt_breakpoints(frame[c].to_numpy(float), cfg) for c in cols}


# ----------------------------------------------------------------------------
# ICSS  (Inclan & Tiao, 1994)
# ----------------------------------------------------------------------------
def icss_breakpoints(
    signal: Sequence[float] | np.ndarray | pd.Series,
    alpha: float = 0.05,
    min_size: int = 30,
    demean: bool = True,
    max_iter: int = 100,
) -> BreakResult:
    r"""Iterated Cumulative Sums of Squares detection of *variance* breaks.

    The full three-step Inclan-Tiao algorithm is implemented: an initial
    exploratory pass, a refinement pass on each candidate, and a final
    convergence check on the ordered set of change points.

    Parameters
    ----------
    signal : array-like
        Series in which to look for variance shifts.  Because ICSS assumes a
        constant mean, this should normally be a *return* or residual series,
        not a price level; ``demean=True`` at least removes the sample mean.
    alpha : {0.10, 0.05, 0.01}
        Significance level; sets the critical value of ``sqrt(T/2) sup|D_k|``.
    min_size : int
        Minimum spacing between accepted breaks.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(1)
    >>> x = np.r_[rng.randn(300), 5.0 * rng.randn(300)]
    >>> res = icss_breakpoints(x, min_size=50)
    >>> res.n_breaks >= 1
    True
    """
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    crit = ICSS_CRITICAL.get(round(float(alpha), 2))
    if crit is None:
        raise ValueError(f"alpha must be one of {sorted(ICSS_CRITICAL)}")

    key = ("icss", _signature(x, alpha, min_size, demean, max_iter))
    if key in _CACHE:
        return _CACHE[key]

    a = x - x.mean() if demean else x

    def _dk(seg: np.ndarray) -> tuple[np.ndarray, float, int]:
        """D_k series, the IT statistic and the argmax for one segment."""
        T = seg.size
        if T < 4:
            return np.zeros(T), 0.0, 0
        C = np.cumsum(seg ** 2)
        CT = C[-1]
        if CT <= 0:
            return np.zeros(T), 0.0, 0
        k = np.arange(1, T + 1)
        D = C / CT - k / T
        stat = np.sqrt(T / 2.0) * np.abs(D)
        j = int(np.argmax(stat))
        return D, float(stat[j]), j

    # ---- step 1: exploratory recursive search -----------------------------
    candidates: list[int] = []
    stack: list[tuple[int, int]] = [(0, n)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2 * min_size:
            continue
        _, stat, j = _dk(a[lo:hi])
        if stat > crit:
            k = lo + j
            if k - lo >= min_size and hi - k >= min_size:
                candidates.append(k)
                stack.append((lo, k))
                stack.append((k, hi))
    candidates = sorted(set(candidates))

    # ---- step 2/3: refinement until the set stops changing ----------------
    for _ in range(max_iter):
        if not candidates:
            break
        bounds = [0] + candidates + [n]
        refined: list[int] = []
        for i in range(1, len(bounds) - 1):
            lo, hi = bounds[i - 1], bounds[i + 1]
            _, stat, j = _dk(a[lo:hi])
            if stat > crit:
                k = lo + j
                if k - lo >= min_size and hi - k >= min_size:
                    refined.append(k)
        refined = sorted(set(refined))
        if refined == candidates:
            break
        candidates = refined

    _, stat_full, _ = _dk(a)
    res = BreakResult(
        breakpoints=candidates,
        n=n,
        method="ICSS",
        labels=regimes_from_breakpoints(n, candidates),
        detail={"alpha": alpha, "critical_value": crit,
                "IT_statistic_full_sample": stat_full, "min_size": min_size},
    )
    _CACHE[key] = res
    return res


# ----------------------------------------------------------------------------
# Bai-Perron
# ----------------------------------------------------------------------------
def bai_perron_breakpoints(
    y: Sequence[float] | np.ndarray | pd.Series,
    X: np.ndarray | pd.DataFrame | None = None,
    max_breaks: int = 5,
    trim: float = 0.15,
    criterion: Literal["bic", "lwz", "fixed"] = "bic",
    n_breaks: int | None = None,
) -> BreakResult:
    r"""Global minimiser of the multiple-break least-squares problem.

    Solves, by dynamic programming exactly as in Bai & Perron (2003),

    .. math::
        S_T = \min_{\tau_1,\dots,\tau_m} \sum_{j=1}^{m+1}
              \sum_{t=\tau_{j-1}+1}^{\tau_j}\bigl(y_t - x_t'\beta_j\bigr)^2

    jointly over the break dates *and* the regime coefficients (the paper's
    equation (2) minimises over the dates only, with ``beta_hat`` already
    plugged in).  The number of breaks is chosen by BIC or by the LWZ
    criterion unless ``n_breaks`` is given.

    Parameters
    ----------
    y : array-like, shape (n,)
        Dependent variable.
    X : array-like, shape (n, q), optional
        Regressors whose coefficients are allowed to break.  Defaults to an
        intercept only, i.e. a pure structural change in mean.
    max_breaks : int
        Upper bound m on the number of breaks.
    trim : float
        Trimming fraction eta; minimum regime length is ``ceil(trim * n)``.
    criterion : {"bic", "lwz", "fixed"}
        Model-selection rule for m.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(2)
    >>> y = np.r_[rng.randn(150), rng.randn(150) + 4.0]
    >>> res = bai_perron_breakpoints(y, max_breaks=3, trim=0.15)
    >>> res.n_breaks
    1
    """
    yv = np.asarray(y, dtype=float).ravel()
    n = yv.size
    Xm = np.ones((n, 1)) if X is None else np.asarray(X, dtype=float).reshape(n, -1)
    q = Xm.shape[1]
    h = max(int(np.ceil(trim * n)), q + 1)

    if 2 * h > n:
        raise ValueError("trim too large for this sample size")

    key = ("bp", _signature(yv, q, max_breaks, trim, criterion, n_breaks),
           _signature(Xm))
    if key in _CACHE:
        return _CACHE[key]

    # ---- segment SSR ------------------------------------------------------
    # Pure mean shift (the usual case, and the one the baseline uses) has a
    # closed form from cumulative sums, so the O(n^2) table never has to be
    # materialised.  For general regressors we fall back on least squares.
    mean_shift = X is None or (q == 1 and np.allclose(Xm, 1.0))
    if mean_shift:
        S1 = np.concatenate([[0.0], np.cumsum(yv)])
        S2 = np.concatenate([[0.0], np.cumsum(yv ** 2)])

        def ssr_vec(s: np.ndarray, t: int) -> np.ndarray:
            """SSR of every segment [s, t) for a vector of start points."""
            length = t - s
            sm = S1[t] - S1[s]
            sq = S2[t] - S2[s]
            return sq - sm * sm / np.maximum(length, 1)
    else:
        cache: dict[tuple[int, int], float] = {}

        def _ssr(i: int, j: int) -> float:
            key = (i, j)
            if key not in cache:
                Xs, ys = Xm[i:j], yv[i:j]
                beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
                r = ys - Xs @ beta
                cache[key] = float(r @ r)
            return cache[key]

        def ssr_vec(s: np.ndarray, t: int) -> np.ndarray:
            return np.array([_ssr(int(a), t) for a in s], dtype=float)

    # ---- dynamic program over the number of breaks ------------------------
    max_m = min(int(max_breaks), (n // h) - 1)
    max_m = max(max_m, 0)
    best: dict[int, tuple[float, list[int]]] = {0: (float(ssr_vec(np.array([0]), n)[0]), [])}

    # prev_cost[t] / prev_path[t]: optimum over observations 0..t with m-1 breaks
    prev_cost = np.full(n + 1, np.inf)
    prev_path: dict[int, list[int]] = {}
    for t in range(h, n + 1):
        prev_cost[t] = ssr_vec(np.array([0]), t)[0]
        prev_path[t] = []

    for m in range(1, max_m + 1):
        cur_cost = np.full(n + 1, np.inf)
        cur_path: dict[int, list[int]] = {}
        for t in range((m + 1) * h, n + 1):
            s = np.arange(m * h, t - h + 1)
            if s.size == 0:
                continue
            total = prev_cost[s] + ssr_vec(s, t)
            k = int(np.argmin(total))
            if np.isfinite(total[k]):
                cur_cost[t] = total[k]
                cur_path[t] = prev_path[int(s[k])] + [int(s[k])]
        if not np.isfinite(cur_cost[n]):
            break
        best[m] = (float(cur_cost[n]), cur_path[n])
        prev_cost, prev_path = cur_cost, cur_path

    # ---- select m ----------------------------------------------------------
    if n_breaks is not None:
        m_star = int(n_breaks)
        if m_star not in best:
            raise ValueError(f"cannot fit {n_breaks} breaks with trim={trim}")
    else:
        scores = {}
        for m, (sse, _) in best.items():
            k = (m + 1) * q + m           # coefficients + break dates
            sigma2 = max(sse / n, 1e-300)
            if criterion == "lwz":
                scores[m] = np.log(sigma2) + 0.299 * k * (np.log(n) ** 2.1) / n
            else:  # bic
                scores[m] = np.log(sigma2) + k * np.log(n) / n
        m_star = int(min(scores, key=scores.get))

    sse, path = best[m_star]
    res = BreakResult(
        breakpoints=[int(b) for b in path],
        n=n,
        method="Bai-Perron",
        labels=regimes_from_breakpoints(n, path),
        detail={"ssr": float(sse), "max_breaks": max_breaks, "trim": trim,
                "criterion": criterion, "n_regressors": q,
                "ssr_by_m": {m: float(v[0]) for m, v in best.items()}},
    )
    _CACHE[key] = res
    return res


def bp_icss_breakpoints(
    signal: Sequence[float] | np.ndarray | pd.Series,
    max_breaks: int = 5,
    trim: float = 0.15,
    alpha: float = 0.05,
    min_size: int = 30,
    on_returns_for_icss: bool = True,
) -> BreakResult:
    """The baseline detector of Lin & Zhang (2022): Bai-Perron **and** ICSS.

    Mean breaks from Bai-Perron and variance breaks from ICSS are merged into
    a single ordered set, exactly as the paper's "Unified Output" paragraph
    describes.  This is what powers the ``BP&ICSS-WT-LSTM`` baseline row.
    """
    x = np.asarray(signal, dtype=float).ravel()
    bp = bai_perron_breakpoints(x, max_breaks=max_breaks, trim=trim)

    if on_returns_for_icss:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(np.log(np.clip(x, 1e-8, None)), prepend=np.log(max(x[0], 1e-8)))
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        r = x
    ic = icss_breakpoints(r, alpha=alpha, min_size=min_size)

    merged = sorted(set(bp.breakpoints) | set(ic.breakpoints))
    # enforce minimum spacing after the merge
    kept: list[int] = []
    for b in merged:
        if not kept or b - kept[-1] >= min_size:
            kept.append(int(b))

    return BreakResult(
        breakpoints=kept,
        n=x.size,
        method="BP&ICSS",
        labels=regimes_from_breakpoints(x.size, kept),
        detail={"bai_perron": bp.breakpoints, "icss": ic.breakpoints,
                "bp_detail": bp.detail, "icss_detail": ic.detail},
    )


# ----------------------------------------------------------------------------
# regime encoding
# ----------------------------------------------------------------------------
def regimes_from_breakpoints(n: int, breakpoints: Iterable[int]) -> np.ndarray:
    r"""Map break locations to integer regime labels :math:`r_t`.

    With breaks at 500 and 1200 the labels are

    .. math::
        r_t = \begin{cases}
            0 & t \le 500\\
            1 & 500 < t \le 1200\\
            2 & t > 1200
        \end{cases}

    Examples
    --------
    >>> regimes_from_breakpoints(6, [2, 4])
    array([0, 0, 1, 1, 2, 2])
    """
    labels = np.zeros(int(n), dtype=int)
    for i, b in enumerate(sorted(int(x) for x in breakpoints), start=1):
        if 0 < b < n:
            labels[b:] = i
    return labels


def one_hot_regimes(labels: np.ndarray, n_regimes: int | None = None) -> np.ndarray:
    r"""One-hot encode regime labels into :math:`e_t \in \mathbb{R}^{m+1}`.

    Examples
    --------
    >>> one_hot_regimes(np.array([0, 1, 2]))
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    labels = np.asarray(labels, dtype=int).ravel()
    k = int(n_regimes if n_regimes is not None else (labels.max() + 1 if labels.size else 0))
    out = np.zeros((labels.size, k), dtype=float)
    if k:
        valid = (labels >= 0) & (labels < k)
        out[np.arange(labels.size)[valid], labels[valid]] = 1.0
    return out


# ----------------------------------------------------------------------------
def _as_2d(signal) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _resolve_pelt(cfg: PeltConfig | None, kwargs: dict) -> PeltConfig:
    if cfg is None:
        cfg = PeltConfig()
    if kwargs:
        cfg = PeltConfig(**{**cfg.as_dict(), **kwargs})
    return cfg
