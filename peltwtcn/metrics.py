"""
Forecast evaluation metrics and formal accuracy tests.

The four metrics of Table 1 of Ren et al. (2025) are implemented exactly as
defined there, and three tools the paper is missing are added:

* :func:`diebold_mariano` - is the accuracy difference between two models
  statistically significant, or seed noise?
* :func:`model_confidence_set` - Hansen, Lunde & Nason (2011) MCS, which
  models survive as "best" at a given confidence level.
* :func:`naive_random_walk` - the benchmark every daily price forecast must
  beat before any claim is made.

Author: Dr Merwan Roudane
"""

from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "mae", "rmse", "mse", "mape", "smape", "r2", "theil_u",
    "evaluate", "evaluate_many",
    "diebold_mariano", "model_confidence_set", "naive_random_walk",
]


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    r"""Mean absolute error, :math:`\frac1n\sum|y_t-\hat y_t|`.

    >>> mae([1, 2, 3], [1, 2, 4])
    0.3333333333333333
    """
    a, b = _clean(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def mse(y_true, y_pred) -> float:
    """Mean squared error."""
    a, b = _clean(y_true, y_pred)
    return float(np.mean((a - b) ** 2))


def rmse(y_true, y_pred) -> float:
    r"""Root mean squared error, :math:`\sqrt{\frac1n\sum(y_t-\hat y_t)^2}`.

    >>> round(rmse([1, 2, 3], [1, 2, 5]), 4)
    1.1547
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    r"""Mean absolute percentage error in **percent**.

    Undefined when the actual value is zero.  EUA traded at EUR 0.01 in 2007,
    so MAPE is only meaningful on a test window that excludes the Phase I
    collapse - which the paper's chronological 80/20 split happens to do.

    >>> round(mape([100, 200], [110, 180]), 4)
    10.0
    """
    a, b = _clean(y_true, y_pred)
    denom = np.where(np.abs(a) < eps, np.nan, a)
    return float(np.nanmean(np.abs((a - b) / denom)) * 100.0)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent; finite even when actuals hit zero."""
    a, b = _clean(y_true, y_pred)
    denom = (np.abs(a) + np.abs(b)) / 2.0
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(np.abs(a - b) / denom) * 100.0)


def r2(y_true, y_pred) -> float:
    r"""Coefficient of determination :math:`1 - SS_{res}/SS_{tot}`.

    >>> round(r2([1, 2, 3, 4], [1.1, 2.0, 2.9, 4.1]), 4)
    0.9924
    """
    a, b = _clean(y_true, y_pred)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def theil_u(y_true, y_pred) -> float:
    """Theil's U2: RMSE of the model divided by RMSE of a random walk.

    Values below 1 mean the model beats "tomorrow equals today".
    """
    a, b = _clean(y_true, y_pred)
    if a.size < 2:
        return np.nan
    num = np.sqrt(np.mean((a[1:] - b[1:]) ** 2))
    den = np.sqrt(np.mean((a[1:] - a[:-1]) ** 2))
    return float(num / den) if den > 0 else np.nan


# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, name: str = "model") -> dict:
    """All headline metrics for one model, as a dict.

    >>> d = evaluate([1, 2, 3], [1, 2, 3], "perfect")
    >>> d["MAE"], d["R2"]
    (0.0, 1.0)
    """
    return {
        "Model": name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "Theil U": theil_u(y_true, y_pred),
    }


def evaluate_many(
    y_true,
    predictions: Mapping[str, Sequence[float]],
    training_times: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Table 1 of the paper: one row per model, sorted by RMSE.

    Parameters
    ----------
    y_true : array-like
        Realised values on the test window.
    predictions : mapping
        ``{model_name: predicted_series}``.
    training_times : mapping, optional
        ``{model_name: seconds}`` to add a "Train (s)" column.
    """
    rows = [evaluate(y_true, p, name) for name, p in predictions.items()]
    df = pd.DataFrame(rows).set_index("Model")
    if training_times:
        df["Train (s)"] = [training_times.get(m, np.nan) for m in df.index]
    return df.sort_values("RMSE")


# ---------------------------------------------------------------------------
# formal tests
# ---------------------------------------------------------------------------
def diebold_mariano(
    y_true,
    pred_a,
    pred_b,
    horizon: int = 1,
    loss: Literal["mse", "mae"] = "mse",
    harvey_correction: bool = True,
) -> dict:
    r"""Diebold-Mariano test of equal predictive accuracy.

    :math:`H_0`: the two forecasts have the same expected loss.  A negative
    statistic favours model **A**.  The small-sample correction of Harvey,
    Leybourne & Newbold (1997) is applied by default and the p-value is taken
    from a t distribution with ``n-1`` degrees of freedom.

    Returns
    -------
    dict
        ``{"DM", "p_value", "mean_loss_diff", "better", "n"}``

    Examples
    --------
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(500)
    >>> good = y + 0.10 * rng.randn(500)
    >>> bad = y + 1.00 * rng.randn(500)
    >>> res = diebold_mariano(y, good, bad)
    >>> res["better"], res["p_value"] < 0.01
    ('A', True)
    """
    a, pa = _clean(y_true, pred_a)
    _, pb = _clean(y_true, pred_b)
    n = a.size
    if n < 3:
        raise ValueError("need at least 3 observations")

    ea, eb = a - pa, a - pb
    d = (ea ** 2 - eb ** 2) if loss == "mse" else (np.abs(ea) - np.abs(eb))
    dbar = float(np.mean(d))

    # Newey-West long-run variance with h-1 lags
    h = int(horizon)
    gamma0 = float(np.mean((d - dbar) ** 2))
    lrv = gamma0
    for k in range(1, h):
        g = float(np.mean((d[k:] - dbar) * (d[:-k] - dbar)))
        lrv += 2.0 * (1.0 - k / h) * g
    lrv = max(lrv, 1e-300)

    dm = dbar / np.sqrt(lrv / n)
    if harvey_correction and h > 1:
        adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm *= adj
    p = float(2.0 * (1.0 - stats.t.cdf(abs(dm), df=n - 1)))

    return {
        "DM": float(dm),
        "p_value": p,
        "mean_loss_diff": dbar,
        "better": "A" if dbar < 0 else ("B" if dbar > 0 else "tie"),
        "n": int(n),
    }


def model_confidence_set(
    y_true,
    predictions: Mapping[str, Sequence[float]],
    alpha: float = 0.10,
    n_boot: int = 1000,
    block: int = 10,
    loss: Literal["mse", "mae"] = "mse",
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Hansen-Lunde-Nason Model Confidence Set (range statistic, T_R).

    Iteratively eliminates the worst model until the null of equal predictive
    ability can no longer be rejected.  The survivors are the set of models
    that cannot be statistically separated from the best one.

    Returns
    -------
    DataFrame
        Columns ``avg_loss``, ``p_MCS``, ``in_MCS``, ordered by average loss.
    """
    rng = np.random.RandomState(random_state)
    names = list(predictions)
    a = np.asarray(y_true, dtype=float).ravel()
    L = np.column_stack([
        (a - np.asarray(predictions[k], dtype=float).ravel()) ** 2 if loss == "mse"
        else np.abs(a - np.asarray(predictions[k], dtype=float).ravel())
        for k in names
    ])
    n = L.shape[0]

    # stationary block bootstrap indices, reused across elimination rounds
    n_blocks = int(np.ceil(n / block))
    boot_idx = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        starts = rng.randint(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        boot_idx[b] = idx

    alive = list(range(len(names)))
    pvals: dict[str, float] = {}
    while len(alive) > 1:
        sub = L[:, alive]
        dbar = sub.mean(axis=0)
        dij = dbar[:, None] - dbar[None, :]
        boot = np.stack([sub[boot_idx[b]].mean(axis=0) for b in range(n_boot)])
        bij = boot[:, :, None] - boot[:, None, :]
        var = bij.var(axis=0) + 1e-300
        t_obs = np.abs(dij) / np.sqrt(var)
        t_boot = np.abs(bij - dij[None]) / np.sqrt(var)[None]

        iu = np.triu_indices(len(alive), 1)
        TR = t_obs[iu].max() if iu[0].size else 0.0
        TR_boot = t_boot[:, iu[0], iu[1]].max(axis=1) if iu[0].size else np.zeros(n_boot)
        p = float(np.mean(TR_boot >= TR))

        worst = int(np.argmax((dij / np.sqrt(var)).max(axis=1)))
        pvals[names[alive[worst]]] = p
        if p > alpha:
            break
        alive.pop(worst)
    for i in alive:
        pvals.setdefault(names[i], 1.0)

    survivors = {names[i] for i in alive}
    out = pd.DataFrame({
        "avg_loss": L.mean(axis=0),
        "p_MCS": [pvals.get(k, np.nan) for k in names],
        "in_MCS": [k in survivors for k in names],
    }, index=names)
    return out.sort_values("avg_loss")


def naive_random_walk(y_true) -> np.ndarray:
    r"""The no-change benchmark :math:`\hat y_t = y_{t-1}`.

    Any daily price model that cannot beat this has not demonstrated anything.

    >>> naive_random_walk([1.0, 2.0, 3.0]).tolist()
    [1.0, 1.0, 2.0]
    """
    a = np.asarray(y_true, dtype=float).ravel()
    out = np.empty_like(a)
    out[0] = a[0]
    out[1:] = a[:-1]
    return out
