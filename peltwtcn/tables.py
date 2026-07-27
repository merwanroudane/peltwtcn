"""
Publication-quality tables.

Turns pipeline output into the tables a referee expects: Table 1 of the paper,
descriptive statistics, a break inventory, and the Diebold-Mariano matrix the
paper omits.  Every table can be exported to Markdown, LaTeX (booktabs),
HTML, CSV or Excel.

Author: Dr Merwan Roudane
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "results_table",
    "comparison_table",
    "summary_statistics",
    "describe_breaks",
    "dm_matrix",
    "mcs_table",
    "improvement_table",
    "export_table",
    "PAPER_TABLE1",
    "PAPER_TRAIN_TIMES",
    "paper_table1",
    "compare_with_paper",
]

Fmt = Literal["plain", "markdown", "latex", "html"]

_DEFAULT_FLOATS = {
    "MAE": 4, "RMSE": 4, "MAPE (%)": 4, "R2": 4, "Theil U": 4,
    "Train (s)": 1, "avg_loss": 5, "p_MCS": 3,
}


# ---------------------------------------------------------------------------
def results_table(
    table: pd.DataFrame,
    fmt: Fmt = "plain",
    decimals: Mapping[str, int] | None = None,
    bold_best: bool = True,
    caption: str = "Performance comparison for carbon price prediction",
    label: str = "tab:performance",
) -> str:
    """Render Table 1 of the paper.

    Parameters
    ----------
    table : DataFrame
        Output of :func:`peltwtcn.metrics.evaluate_many`.
    fmt : {"plain", "markdown", "latex", "html"}
    bold_best : bool
        Emphasise the best cell of every column (lowest error, highest R^2).

    Examples
    --------
    >>> import pandas as pd
    >>> t = pd.DataFrame({"MAE": [1.2, 2.4], "RMSE": [1.6, 2.8], "R2": [0.99, 0.96]},
    ...                  index=["TCN", "LSTM"])
    >>> print(results_table(t, fmt="markdown"))     # doctest: +ELLIPSIS
    | Model   |   MAE |...
    """
    dec = {**_DEFAULT_FLOATS, **(decimals or {})}
    df = table.copy()
    df.index.name = df.index.name or "Model"

    higher_is_better = {"R2"}
    best: dict[str, object] = {}
    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        best[c] = df[c].idxmax() if c in higher_is_better else df[c].idxmin()

    out = df.copy()
    for c in out.columns:
        if np.issubdtype(out[c].dtype, np.number):
            out[c] = out[c].map(lambda v, d=dec.get(c, 4): f"{v:.{d}f}")

    if bold_best:
        mark = {"markdown": "**{}**", "latex": r"\textbf{{{}}}",
                "html": "<b>{}</b>", "plain": "{}*"}[fmt]
        for c, row in best.items():
            if c in out.columns and row in out.index:
                out.loc[row, c] = mark.format(out.loc[row, c])

    if fmt == "markdown":
        return out.to_markdown()
    if fmt == "html":
        return out.to_html(escape=False)
    if fmt == "latex":
        body = out.to_latex(escape=False, column_format="l" + "r" * out.shape[1],
                            caption=caption, label=label, position="htbp")
        # booktabs: replace the default rules
        body = (body.replace(r"\toprule", r"\toprule")
                    .replace(r"\midrule", r"\midrule")
                    .replace(r"\bottomrule", r"\bottomrule"))
        return "\\begin{table}[htbp]\n\\centering\n" + body.split("\\begin{table}")[-1] \
            if "\\begin{table}" in body else body
    return out.to_string()


def comparison_table(result, baseline: str | None = None,
                     fmt: Fmt = "plain") -> str:
    """Table 1 plus percentage improvement over a chosen baseline row."""
    return results_table(improvement_table(result.table, baseline), fmt=fmt)


def improvement_table(table: pd.DataFrame, baseline: str | None = None) -> pd.DataFrame:
    r"""Add ``\Delta MAE (%)`` and ``\Delta RMSE (%)`` versus a baseline row.

    The paper quotes "22.35% in RMSE and 18.63% in MAE" against its baseline,
    but those numbers cannot be reproduced from its own Table 1: the reductions
    implied by the published figures are 70.55% / 74.42% against
    ``BP&ICSS-WT-LSTM`` and 6.60% / 10.92% against ``PELT-WT-GRU``.  This helper
    computes them explicitly so the claim is always checkable.

    Examples
    --------
    >>> t = pd.DataFrame({"MAE": [1.1855, 4.6345], "RMSE": [1.5866, 5.3878]},
    ...                  index=["PELT-WT-TCN", "BP&ICSS-WT-LSTM"])
    >>> imp = improvement_table(t, baseline="BP&ICSS-WT-LSTM")
    >>> round(float(imp.loc["PELT-WT-TCN", "dRMSE (%)"]), 2)
    70.55
    """
    df = table.copy()
    if baseline is None:
        candidates = [i for i in df.index if str(i).lower() != "random walk"]
        baseline = df.loc[candidates, "RMSE"].idxmax() if candidates else df.index[0]
    for m, col in (("MAE", "dMAE (%)"), ("RMSE", "dRMSE (%)")):
        if m in df.columns:
            b = float(df.loc[baseline, m])
            df[col] = (1.0 - df[m] / b) * 100.0 if b else np.nan
    df.attrs["baseline"] = baseline
    return df


# ---------------------------------------------------------------------------
def summary_statistics(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Descriptive statistics with skewness, kurtosis, Jarque-Bera and ADF.

    The paper asserts that "the carbon price series exhibits positive skewness
    and high kurtosis"; this table makes that testable.
    """
    from scipy import stats

    rows = []
    for c in df.select_dtypes("number").columns:
        x = df[c].dropna().to_numpy(float)
        if x.size < 8:
            continue
        jb, jb_p = stats.jarque_bera(x)
        row = {
            "Variable": c, "N": x.size, "Mean": x.mean(), "Std": x.std(ddof=1),
            "Min": x.min(), "Median": float(np.median(x)), "Max": x.max(),
            "Skewness": float(stats.skew(x)), "Kurtosis": float(stats.kurtosis(x, fisher=False)),
            "Jarque-Bera": float(jb), "JB p": float(jb_p),
        }
        try:
            from statsmodels.tsa.stattools import adfuller
            adf = adfuller(x, autolag="AIC")
            row["ADF"] = float(adf[0])
            row["ADF p"] = float(adf[1])
        except Exception:
            row["ADF"], row["ADF p"] = np.nan, np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Variable").round(decimals)


def describe_breaks(breaks, index: pd.Index, price: pd.Series | None = None,
                    events: Sequence[tuple] | None = None,
                    tolerance_days: int = 45) -> pd.DataFrame:
    """Inventory of detected breaks: date, regime length, level shift, event.

    Optionally matches each break to the nearest policy event within
    ``tolerance_days``, which is what turns a break table into an
    interpretable one.
    """
    from .breaks import BreakResult

    if not isinstance(breaks, BreakResult):
        frames = [describe_breaks(v, index, price, events, tolerance_days)
                     .assign(Series=k) for k, v in breaks.items()]
        return pd.concat(frames).reset_index(drop=True)

    idx = pd.DatetimeIndex(index)
    bps = list(breaks.breakpoints)
    bounds = [0] + bps + [len(idx)]
    rows = []
    for i, b in enumerate(bps):
        seg_prev = slice(bounds[i], b)
        seg_next = slice(b, bounds[i + 2] if i + 2 < len(bounds) else len(idx))
        row = {
            "Break": i + 1,
            "Index": int(b),
            "Date": idx[b] if b < len(idx) else pd.NaT,
            "Regime length (before)": b - bounds[i],
            "Regime length (after)": bounds[i + 2] - b if i + 2 < len(bounds) else len(idx) - b,
        }
        if price is not None:
            p = price.to_numpy(float)
            row["Mean before"] = float(np.mean(p[seg_prev])) if b > bounds[i] else np.nan
            row["Mean after"] = float(np.mean(p[seg_next]))
            row["Shift"] = row["Mean after"] - row["Mean before"]
            row["Std before"] = float(np.std(p[seg_prev], ddof=1)) if b - bounds[i] > 1 else np.nan
            row["Std after"] = float(np.std(p[seg_next], ddof=1)) if seg_next.stop - b > 1 else np.nan
        if events:
            d = row["Date"]
            best, best_gap = None, None
            for date, _sign, desc in events:
                gap = abs((pd.Timestamp(date) - d).days)
                if best_gap is None or gap < best_gap:
                    best, best_gap = desc, gap
            row["Nearest event"] = best if best_gap is not None and best_gap <= tolerance_days else ""
            row["Gap (days)"] = best_gap if best_gap is not None and best_gap <= tolerance_days else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def dm_matrix(result, loss: str = "mse", horizon: int = 1) -> pd.DataFrame:
    """Pairwise Diebold-Mariano p-values for every model in an experiment.

    Cell (i, j) tests H0: models i and j are equally accurate.
    """
    from .metrics import diebold_mariano

    names = list(result.predictions)
    out = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            try:
                out.loc[a, b] = diebold_mariano(
                    result.actual, result.predictions[a], result.predictions[b],
                    horizon=horizon, loss=loss)["p_value"]
            except Exception:
                pass
    return out


def mcs_table(result, alpha: float = 0.10, n_boot: int = 1000,
              block: int = 10) -> pd.DataFrame:
    """Model Confidence Set membership at level ``alpha``."""
    from .metrics import model_confidence_set
    return model_confidence_set(result.actual, result.predictions, alpha=alpha,
                                n_boot=n_boot, block=block)


# ---------------------------------------------------------------------------
# The published numbers, so a replication can always be diffed against them
# ---------------------------------------------------------------------------
#: Table 1 of Ren et al. (2025), p. 22, transcribed verbatim.
PAPER_TABLE1: pd.DataFrame = pd.DataFrame(
    {
        "MAE":      [4.6345, 2.3627, 1.8192, 1.3308, 1.1855],
        "RMSE":     [5.3878, 2.7488, 2.2967, 1.6987, 1.5866],
        "MAPE (%)": [5.8731, 3.0582, 2.3267, 1.7401, 1.6451],
        "R2":       [0.8712, 0.9664, 0.9765, 0.9872, 0.9888],
    },
    index=pd.Index(
        ["BP&ICSS-WT-LSTM", "PELT-WT-LSTM (uni)", "PELT-WT-LSTM (multi)",
         "PELT-WT-GRU", "PELT-WT-TCN"],
        name="Model",
    ),
)

#: Training times in seconds.  The paper reports these twice and the two
#: statements disagree, so both are kept.  Figure 16 (the bar chart) is used as
#: the primary source because it is the one the text refers to for scalability.
PAPER_TRAIN_TIMES: dict[str, dict[str, float]] = {
    "figure_16": {
        "BP&ICSS-WT-LSTM": 26.1, "PELT-WT-LSTM (uni)": 26.4,
        "PELT-WT-LSTM (multi)": 26.2, "PELT-WT-GRU": 14.3,
        "PELT-WT-TCN": 52.5,
    },
    "section_4.3_text": {
        "BP&ICSS-WT-LSTM": 24.5, "PELT-WT-LSTM (uni)": 24.7,
        "PELT-WT-LSTM (multi)": 24.9, "PELT-WT-GRU": 19.6,
        "PELT-WT-TCN": 48.7,
    },
}


def paper_table1(fmt: Fmt | None = None):
    """Return (or render) Table 1 of the paper exactly as published.

    Examples
    --------
    >>> paper_table1().loc["PELT-WT-TCN", "RMSE"]
    1.5866
    """
    return PAPER_TABLE1.copy() if fmt is None else results_table(PAPER_TABLE1, fmt=fmt)


def compare_with_paper(table: pd.DataFrame,
                       metrics: Sequence[str] = ("MAE", "RMSE", "MAPE (%)", "R2"),
                       ) -> pd.DataFrame:
    """Put a replication side by side with the published Table 1.

    For every model present in both, report the published value, the replicated
    value, and the signed difference ``replication - paper``.  Rows the paper
    does not contain (e.g. ``Random walk``) are kept with ``NaN`` in the paper
    columns, because dropping them would hide the most informative benchmark.

    Parameters
    ----------
    table : DataFrame
        Output of :func:`peltwtcn.metrics.evaluate_many`, i.e.
        ``ExperimentResult.table``.
    metrics : sequence of str
        Which columns to compare.

    Returns
    -------
    DataFrame
        MultiIndex columns ``(metric, {"Paper", "Replication", "Diff"})``.

    Notes
    -----
    Differences are expected and are not evidence of a coding error.  The paper
    does not publish a random seed, its exogenous feature set includes three
    subscription-only series this package cannot download (see
    :data:`peltwtcn.datasets.UNAVAILABLE_FEATURES`), and its wavelet filter is
    two-sided, so its reported errors benefit from look-ahead information.  See
    ``docs/REPLICATION_NOTES.md``.

    Examples
    --------
    >>> import pandas as pd
    >>> t = pd.DataFrame({"MAE": [1.20], "RMSE": [1.60]}, index=["PELT-WT-TCN"])
    >>> cmp = compare_with_paper(t, metrics=("MAE",))
    >>> round(float(cmp.loc["PELT-WT-TCN", ("MAE", "Diff")]), 4)
    0.0145
    """
    rows = list(dict.fromkeys(list(PAPER_TABLE1.index) + list(table.index)))
    frames: dict[tuple, pd.Series] = {}
    for m in metrics:
        paper = PAPER_TABLE1[m].reindex(rows) if m in PAPER_TABLE1.columns else pd.Series(np.nan, index=rows)
        repl = table[m].reindex(rows) if m in table.columns else pd.Series(np.nan, index=rows)
        frames[(m, "Paper")] = paper
        frames[(m, "Replication")] = repl
        frames[(m, "Diff")] = repl - paper
    out = pd.DataFrame(frames)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index.name = "Model"
    return out


# ---------------------------------------------------------------------------
def export_table(table: pd.DataFrame, path: str | Path,
                 caption: str = "", label: str = "", index: bool = True) -> Path:
    """Write a table to ``.md``, ``.tex``, ``.csv``, ``.xlsx`` or ``.html``.

    The extension of ``path`` decides the format.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()

    if ext == ".csv":
        table.to_csv(p, index=index)
    elif ext in (".xlsx", ".xls"):
        table.to_excel(p, index=index)
    elif ext == ".md":
        p.write_text(
            (f"**{caption}**\n\n" if caption else "") + table.to_markdown(index=index),
            encoding="utf-8")
    elif ext == ".tex":
        p.write_text(
            table.to_latex(index=index, escape=False, caption=caption or None,
                           label=label or None, position="htbp",
                           column_format="l" + "r" * table.shape[1]),
            encoding="utf-8")
    elif ext in (".html", ".htm"):
        p.write_text(table.to_html(index=index), encoding="utf-8")
    else:
        raise ValueError(f"unsupported extension {ext!r}")
    return p
