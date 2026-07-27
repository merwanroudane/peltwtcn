"""
Publication-quality figures.

Reproduces every figure of Ren et al. (2025) -- Figures 5 to 18 -- in a style
suitable for submission to an Elsevier / Wiley / Springer journal: serif type,
no chart junk, colour-blind-safe palette, hairline spines, 600 dpi vector or
raster export.

All functions follow the same contract: they take data, return a
``matplotlib.figure.Figure``, never call ``plt.show()``, and accept ``ax=`` so
they can be composed into multi-panel layouts.

Author: Dr Merwan Roudane
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "set_journal_style",
    "PALETTE",
    "plot_price_history",
    "plot_correlation_drivers",
    "plot_feature_importance",
    "plot_breakpoints",
    "plot_denoising",
    "plot_wavelet_decomposition",
    "plot_forecast",
    "plot_all_forecasts",
    "plot_model_comparison",
    "plot_training_time",
    "plot_residuals_over_time",
    "plot_residual_density",
    "plot_dm_heatmap",
    "save_all_figures",
]

#: Colour-blind-safe qualitative palette (Okabe-Ito), used everywhere.
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

_GREY = "#4D4D4D"
_LIGHT = "#BFBFBF"


def set_journal_style(font_scale: float = 1.0, serif: bool = True,
                      dpi: int = 150) -> None:
    """Install the journal rcParams globally.

    Call once at the top of a script or notebook.  Uses a serif family when
    available (falls back silently if the font is missing), hairline axes,
    outward ticks and no top/right spines.
    """
    families = (["Times New Roman", "DejaVu Serif", "serif"] if serif
                else ["Arial", "DejaVu Sans", "sans-serif"])
    mpl.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif" if serif else "sans-serif",
        ("font.serif" if serif else "font.sans-serif"): families,
        "font.size": 9 * font_scale,
        "axes.titlesize": 10 * font_scale,
        "axes.labelsize": 9.5 * font_scale,
        "axes.titleweight": "normal",
        "axes.linewidth": 0.7,
        "axes.edgecolor": _GREY,
        "axes.labelcolor": "black",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "grid.color": _LIGHT,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.55,
        "xtick.labelsize": 8.5 * font_scale,
        "ytick.labelsize": 8.5 * font_scale,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.color": _GREY,
        "ytick.color": _GREY,
        "legend.frameon": False,
        "legend.fontsize": 8.5 * font_scale,
        "legend.handlelength": 1.6,
        "lines.linewidth": 1.2,
        "lines.solid_capstyle": "round",
        "figure.constrained_layout.use": True,
        "mathtext.fontset": "cm",
    })


def _fig_ax(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _panel_label(ax, text: str) -> None:
    if text:
        ax.text(-0.06, 1.04, text, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------------------
# data description
# ---------------------------------------------------------------------------
def plot_price_history(price: pd.Series, events: Sequence[tuple] | None = None,
                       ax=None, figsize=(7.0, 3.0), title: str | None = None,
                       label: str = ""):
    """Carbon price over the full sample, optionally annotated with events."""
    fig, ax = _fig_ax(ax, figsize)
    ax.plot(price.index, price.values, color=PALETTE[0], lw=0.9)
    ax.fill_between(price.index, 0, price.values, color=PALETTE[0], alpha=0.08)
    if events:
        for date, sign, desc in events:
            d = pd.Timestamp(date)
            if price.index[0] <= d <= price.index[-1]:
                ax.axvline(d, color=PALETTE[1] if sign > 0 else PALETTE[3],
                           lw=0.55, ls=(0, (3, 2)), alpha=0.75)
    ax.set_xlabel("Date")
    ax.set_ylabel("EUA price (EUR / tCO$_2$)")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    if title:
        ax.set_title(title)
    _panel_label(ax, label)
    return fig


def plot_correlation_drivers(df: pd.DataFrame, target: str = "Carbon_Price",
                             method: str = "pearson", ax=None,
                             figsize=(5.6, 3.6), label: str = ""):
    """Figure 5: Pearson correlation of each feature with the carbon price.

    The paper's own version includes the target's correlation with itself
    (1.0); it is dropped here.
    """
    fig, ax = _fig_ax(ax, figsize)
    corr = (df.corr(numeric_only=True, method=method)[target]
              .drop(labels=[target], errors="ignore")
              .sort_values())
    colors = [PALETTE[1] if v < 0 else PALETTE[0] for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, height=0.72,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=_GREY, lw=0.7)
    ax.set_xlabel(f"{method.capitalize()} correlation with {target.replace('_', ' ').lower()}")
    ax.set_ylabel("")
    ax.set_xlim(-1, 1)
    for y, v in enumerate(corr.values):
        ax.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7.5,
                color=_GREY)
    ax.grid(axis="y", visible=False)
    _panel_label(ax, label)
    return fig


def plot_feature_importance(df: pd.DataFrame, target: str = "Carbon_Price",
                            n_estimators: int = 300, random_state: int = 0,
                            ax=None, figsize=(5.6, 3.6), label: str = ""):
    """Figure 6: Extra-Trees feature importance (Geurts et al., 2006)."""
    from sklearn.ensemble import ExtraTreesRegressor

    fig, ax = _fig_ax(ax, figsize)
    X = df.drop(columns=[target]).select_dtypes("number").dropna()
    y = df.loc[X.index, target]
    model = ExtraTreesRegressor(n_estimators=n_estimators,
                                random_state=random_state, n_jobs=-1).fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    ax.barh(imp.index, imp.values, color=PALETTE[2], height=0.72,
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Extra-Trees importance score")
    for y_, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.015, y_, f"{v:.3f}", va="center",
                fontsize=7.5, color=_GREY)
    ax.set_xlim(0, imp.max() * 1.16)
    ax.grid(axis="y", visible=False)
    _panel_label(ax, label)
    return fig


def plot_breakpoints(price: pd.Series, breaks, ax=None, figsize=(7.2, 3.2),
                     max_lines: int = 60, show_regimes: bool = True,
                     label: str = "", title: str | None = None):
    """Figure 7: carbon price with detected structural breaks.

    ``breaks`` may be a single :class:`~peltwtcn.breaks.BreakResult` or the
    per-feature dictionary from :func:`~peltwtcn.breaks.pelt_multivariate`.
    Regimes are shaded alternately so over-segmentation is visible at a glance
    rather than hidden behind a forest of dashed lines.
    """
    from .breaks import BreakResult

    fig, ax = _fig_ax(ax, figsize)
    idx = price.index

    if isinstance(breaks, BreakResult):
        groups = {breaks.method: breaks}
    else:
        groups = dict(breaks)

    # A break index only means something on the series it was estimated from.
    # Refuse to silently draw dates from a different sample.
    for name, res in groups.items():
        if getattr(res, "n", len(idx)) != len(idx):
            raise ValueError(
                f"the {name!r} break result was estimated on {res.n} "
                f"observations but 'price' has {len(idx)}. Re-run the detector "
                "on the same series you are plotting."
            )

    if show_regimes and len(groups) == 1:
        res = next(iter(groups.values()))
        bounds = [0] + [b for b in res.breakpoints if 0 < b < len(idx)] + [len(idx)]
        for i in range(len(bounds) - 1):
            if i % 2:
                ax.axvspan(idx[bounds[i]], idx[min(bounds[i + 1], len(idx) - 1)],
                           color=PALETTE[5], alpha=0.10, lw=0)

    for k, (name, res) in enumerate(groups.items()):
        bps = list(res.breakpoints)[:max_lines]
        col = PALETTE[(k + 1) % len(PALETTE)]
        for j, b in enumerate(bps):
            if 0 <= b < len(idx):
                ax.axvline(idx[b], color=col, lw=0.5, ls=(0, (2.5, 2)),
                           alpha=0.7,
                           label=f"{name} ({res.n_breaks})" if j == 0 else None)

    ax.plot(idx, price.values, color="black", lw=0.85, zorder=5,
            label="Carbon price")
    ax.set_xlabel("Date")
    ax.set_ylabel("EUA price (EUR / tCO$_2$)")
    ax.margins(x=0.01)
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, ncol=min(3, len(handles)), loc="upper left")
    if title:
        ax.set_title(title)
    _panel_label(ax, label)
    return fig


def plot_denoising(raw: pd.Series, denoised: np.ndarray, ax=None,
                   figsize=(7.2, 3.2), zoom: tuple | None = None,
                   label: str = ""):
    """Figure 8: raw versus wavelet-denoised carbon price.

    Unlike the paper's Figure 8, the denoised line is drawn *under* the raw
    line with transparency, so the two are actually distinguishable, and an
    optional inset zooms into a window where the smoothing is visible.
    """
    fig, ax = _fig_ax(ax, figsize)
    ax.plot(raw.index, raw.values, color=_LIGHT, lw=0.8, label="Raw price")
    ax.plot(raw.index, np.asarray(denoised, float), color=PALETTE[1], lw=0.9,
            label="Wavelet denoised")
    ax.set_xlabel("Date")
    ax.set_ylabel("EUA price (EUR / tCO$_2$)")
    ax.margins(x=0.01)
    ax.legend(loc="upper left")

    if zoom is not None:
        lo, hi = pd.Timestamp(zoom[0]), pd.Timestamp(zoom[1])
        axins = ax.inset_axes([0.56, 0.08, 0.42, 0.42])
        m = (raw.index >= lo) & (raw.index <= hi)
        axins.plot(raw.index[m], raw.values[m], color=_LIGHT, lw=0.9)
        axins.plot(raw.index[m], np.asarray(denoised, float)[m],
                   color=PALETTE[1], lw=1.0)
        axins.set_xticks([])
        axins.tick_params(labelsize=6.5)
        axins.grid(False)
        for s in axins.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
            s.set_edgecolor(_GREY)
    _panel_label(ax, label)
    return fig


def plot_wavelet_decomposition(raw: pd.Series, decomposition: Mapping,
                               figsize=(7.2, 5.6)):
    """Multi-panel view of the approximation and every detail band."""
    D = list(decomposition["D"])
    n_rows = 2 + len(D)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    axes[0].plot(raw.index, raw.values, color="black", lw=0.7)
    axes[0].set_ylabel("$f$", rotation=0, ha="right", va="center")
    axes[1].plot(raw.index, decomposition["A"], color=PALETTE[0], lw=0.7)
    axes[1].set_ylabel(f"$A_{{{len(D)}}}$", rotation=0, ha="right", va="center")
    for i, d in enumerate(D, start=1):
        axes[i + 1].plot(raw.index, d, color=PALETTE[1], lw=0.5)
        axes[i + 1].set_ylabel(f"$D_{i}$", rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Date")
    for a in axes:
        a.margins(x=0.01)
    return fig


# ---------------------------------------------------------------------------
# forecast diagnostics
# ---------------------------------------------------------------------------
def plot_forecast(actual, predicted, index=None, name: str = "Model", ax=None,
                  figsize=(6.4, 2.8), label: str = "", show_metrics: bool = True):
    """Figures 9-13: actual versus predicted carbon prices for one model."""
    from .metrics import mae, rmse, r2

    fig, ax = _fig_ax(ax, figsize)
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    x = index if index is not None else np.arange(a.size)
    ax.plot(x, a, color="black", lw=0.9, label="Actual")
    ax.plot(x, p, color=PALETTE[1], lw=0.9, alpha=0.9, label=name)
    ax.set_ylabel("EUA price (EUR / tCO$_2$)")
    ax.set_xlabel("Date" if index is not None else "Test-set index")
    ax.margins(x=0.01)
    ax.legend(loc="upper left", ncol=2)
    if show_metrics:
        ax.text(0.985, 0.06,
                f"MAE {mae(a, p):.3f}   RMSE {rmse(a, p):.3f}   $R^2$ {r2(a, p):.4f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                color=_GREY)
    _panel_label(ax, label)
    return fig


def plot_all_forecasts(actual, predictions: Mapping[str, Sequence[float]],
                       index=None, figsize=(7.2, 3.4), label: str = ""):
    """Figure 14: every model against the realised series on one axis."""
    fig, ax = plt.subplots(figsize=figsize)
    a = np.asarray(actual, float)
    x = index if index is not None else np.arange(a.size)
    ax.plot(x, a, color="black", lw=1.3, label="Actual", zorder=10)
    for i, (k, v) in enumerate(predictions.items()):
        ax.plot(x, np.asarray(v, float), lw=0.8, alpha=0.85,
                color=PALETTE[i % len(PALETTE)], ls=["-", "--", "-.", ":"][i % 4],
                label=k)
    ax.set_ylabel("EUA price (EUR / tCO$_2$)")
    ax.set_xlabel("Date" if index is not None else "Test-set index")
    ax.margins(x=0.01)
    ax.legend(ncol=3, loc="upper left")
    _panel_label(ax, label)
    return fig


def plot_model_comparison(table: pd.DataFrame,
                          metrics: Sequence[str] = ("MAE", "RMSE", "MAPE (%)", "R2"),
                          figsize=(7.2, 2.6), label: str = ""):
    """Figure 15: grouped bars of the headline metrics.

    Unlike the paper's Figure 15, R^2 is drawn on its own right-hand axis:
    plotting a 0-1 quantity on the same scale as a 0-6 error metric makes the
    R^2 bars meaningless.
    """
    metrics = [m for m in metrics if m in table.columns]
    err = [m for m in metrics if m != "R2"]
    fig, ax = plt.subplots(figsize=figsize)
    models = list(table.index)
    n = len(err)
    width = 0.8 / max(n + (1 if "R2" in metrics else 0), 1)
    xs = np.arange(len(models))

    for i, m in enumerate(err):
        ax.bar(xs + i * width, table[m].values, width=width * 0.92,
               color=PALETTE[i], label=m, edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Error metric")
    ax.set_xticks(xs + width * (n - (0 if "R2" not in metrics else -0.5)) / 2)
    ax.set_xticklabels(models, rotation=18, ha="right")
    ax.grid(axis="x", visible=False)

    if "R2" in metrics:
        ax2 = ax.twinx()
        ax2.bar(xs + n * width, table["R2"].values, width=width * 0.92,
                color=PALETTE[3], label="$R^2$", edgecolor="white", linewidth=0.4)
        ax2.set_ylabel("$R^2$")
        ax2.set_ylim(0, 1.05)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, 1.18))
    else:
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    _panel_label(ax, label)
    return fig


def plot_training_time(times: Mapping[str, float], ax=None, figsize=(5.6, 2.6),
                       label: str = ""):
    """Figure 16: wall-clock training time per model."""
    fig, ax = _fig_ax(ax, figsize)
    items = [(k, v) for k, v in times.items() if np.isfinite(v) and v > 0]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    bars = ax.bar(names, vals, color=PALETTE[0], width=0.62,
                  edgecolor="white", linewidth=0.4)
    slowest = int(np.argmax(vals)) if vals else -1
    if slowest >= 0:
        bars[slowest].set_color(PALETTE[1])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.1f} s",
                ha="center", va="bottom", fontsize=7.5, color=_GREY)
    ax.set_ylabel("Training time (s)")
    ax.set_ylim(0, max(vals) * 1.16 if vals else 1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.grid(axis="x", visible=False)
    _panel_label(ax, label)
    return fig


def plot_residuals_over_time(residuals: pd.DataFrame, ax=None,
                             figsize=(7.2, 3.0), label: str = ""):
    """Figure 17: residual paths of every model on the test window."""
    fig, ax = _fig_ax(ax, figsize)
    for i, c in enumerate(residuals.columns):
        ax.plot(residuals.index, residuals[c].values, lw=0.6, alpha=0.85,
                color=PALETTE[i % len(PALETTE)], label=c)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Residual (EUR / tCO$_2$)")
    ax.set_xlabel("Date")
    ax.margins(x=0.01)
    ax.legend(ncol=3, loc="upper left")
    _panel_label(ax, label)
    return fig


def plot_residual_density(residuals: pd.DataFrame, ax=None, figsize=(5.8, 3.0),
                          bandwidth: float | None = None, label: str = ""):
    """Figure 18: kernel density of the residual distributions."""
    from scipy import stats as _st

    fig, ax = _fig_ax(ax, figsize)
    lo = float(np.nanpercentile(residuals.values, 0.2))
    hi = float(np.nanpercentile(residuals.values, 99.8))
    grid = np.linspace(lo, hi, 512)
    for i, c in enumerate(residuals.columns):
        v = residuals[c].dropna().values
        if v.size < 5 or np.allclose(v, v[0]):
            continue
        kde = _st.gaussian_kde(v, bw_method=bandwidth)
        ax.plot(grid, kde(grid), lw=1.1, color=PALETTE[i % len(PALETTE)], label=c)
        ax.fill_between(grid, kde(grid), color=PALETTE[i % len(PALETTE)], alpha=0.07)
    ax.axvline(0, color="black", lw=0.7, ls=(0, (3, 2)))
    ax.set_xlabel("Residual (EUR / tCO$_2$)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    _panel_label(ax, label)
    return fig


def plot_dm_heatmap(dm: pd.DataFrame, ax=None, figsize=(5.2, 4.2),
                    label: str = ""):
    """Diebold-Mariano p-value matrix (not in the paper, but it should be).

    Cell (i, j) is the p-value of the test that row model i and column model j
    have equal predictive accuracy.  Dark = the difference is significant.
    """
    fig, ax = _fig_ax(ax, figsize)
    data = dm.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=0.2, aspect="auto")
    ax.set_xticks(range(len(dm.columns)))
    ax.set_xticklabels(dm.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(dm.index)))
    ax.set_yticklabels(dm.index)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color="black" if v > 0.05 else "white")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Diebold-Mariano $p$-value")
    cb.outline.set_linewidth(0.5)
    _panel_label(ax, label)
    return fig


# ---------------------------------------------------------------------------
def save_all_figures(result, df: pd.DataFrame, outdir: str | Path = "figures",
                     price_col: str = "Carbon_Price", fmt: Sequence[str] = ("png", "pdf"),
                     denoised: np.ndarray | None = None, breaks=None,
                     verbose: bool = True) -> list[Path]:
    """Render and save the whole figure set from an :class:`ExperimentResult`.

    Returns
    -------
    list of Path
        Every file written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _save(fig, stem: str):
        for ext in fmt:
            fp = outdir / f"{stem}.{ext}"
            fig.savefig(fp)
            written.append(fp)
        plt.close(fig)
        if verbose:
            print(f"  wrote {outdir / stem}.{'/'.join(fmt)}")

    price = df[price_col]
    _save(plot_price_history(price, events=_events()), "fig01_price_history")
    _save(plot_correlation_drivers(df, price_col), "fig05_correlation_drivers")
    _save(plot_feature_importance(df, price_col), "fig06_feature_importance")
    if breaks is not None:
        _save(plot_breakpoints(price, breaks), "fig07_breakpoints")
    if denoised is not None:
        _save(plot_denoising(price, denoised, zoom=("2022-01-01", "2022-12-31")),
              "fig08_denoising")

    # Figures 9-13 of the paper are the five per-model forecast panels.  Any
    # extra series (the random-walk benchmark, which the paper omits) is
    # numbered after them so the stems never collide.
    n = 9
    for name, pred in result.predictions.items():
        safe = (name.replace(" ", "_").replace("&", "and")
                    .replace("(", "").replace(")", ""))
        _save(plot_forecast(result.actual, pred, result.index, name),
              f"fig{n:02d}_forecast_{safe}")
        n += 1

    for stem, fig in (
        ("all_forecasts", plot_all_forecasts(result.actual, result.predictions,
                                             result.index)),
        ("model_comparison", plot_model_comparison(result.table)),
        ("training_time", plot_training_time(result.training_times)),
        ("residuals_time", plot_residuals_over_time(result.residuals())),
        ("residual_density", plot_residual_density(result.residuals())),
    ):
        _save(fig, f"fig{n:02d}_{stem}")
        n += 1

    from .tables import dm_matrix
    try:
        _save(plot_dm_heatmap(dm_matrix(result)), f"fig{n:02d}_dm_pvalues")
    except Exception:
        pass
    return written


def _events():
    from .datasets import POLICY_EVENTS
    return POLICY_EVENTS
