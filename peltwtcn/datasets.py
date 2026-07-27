"""
Real-data loaders for the EU ETS carbon price study.

Everything in this module downloads **live market and index data**.  Nothing is
simulated.  Three public providers are used:

======================  =======================================================
Provider                Series
======================  =======================================================
investing.com           EUA carbon price (the exact series used by the paper,
                        instrument id 8848, EUR/tCO2, daily since 2007-09-10)
Yahoo Finance           Coal API2 Rotterdam, TTF gas, Henry Hub gas, Brent,
                        Euro Stoxx 50, DAX, VIX, EUR/USD
policyuncertainty.com   US and UK daily Economic Policy Uncertainty indices
matteoiacoviello.com    Daily Geopolitical Risk index (Caldara & Iacoviello)
ECB Data Portal         Euro-area 10-year government benchmark bond yield
======================  =======================================================

Reproducing the paper's sample exactly
--------------------------------------
The paper reports "6,113 samples" between 2007-09-10 and 2024-06-04.  There are
only 4,346 *trading* days in that window; 6,113 is the number of **calendar**
days.  ``load_eua(frequency="calendar")`` therefore reindexes to calendar-daily
and forward-fills, which reproduces N = 6113 to the observation.  Use
``frequency="trading"`` for the econometrically cleaner series.

Two features named in Figures 5-6 of the paper are commercial products with no
free feed (Epex Spot Germany day-ahead, Citi Economic Surprise Index Eurozone)
and are listed in :data:`UNAVAILABLE_FEATURES`.  The paper's dominant feature,
"Policy", is never defined anywhere in the text; :func:`build_policy_features`
reconstructs it from the event chronology the paper itself gives in Section 4.1
and is documented as a reconstruction.

Author: Dr Merwan Roudane
"""

from __future__ import annotations

import io
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import requests

__all__ = [
    "CARBON_ID",
    "YAHOO_FEATURES",
    "UNAVAILABLE_FEATURES",
    "POLICY_EVENTS",
    "ETS_PHASES",
    "PAPER_START",
    "PAPER_END",
    "fetch_investing",
    "fetch_yahoo",
    "fetch_epu",
    "fetch_gpr",
    "fetch_ecb",
    "load_eua",
    "load_exogenous",
    "build_policy_features",
    "load_paper_dataset",
    "cache_dir",
]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
PAPER_START = "2007-09-10"
PAPER_END = "2024-06-04"

#: investing.com instrument id for "Carbon Emissions Futures" (EUR/tCO2).
CARBON_ID = 8848

#: Paper feature name -> Yahoo Finance ticker.
YAHOO_FEATURES: dict[str, str] = {
    "Europe_Coal": "MTF=F",        # ICE Rotterdam Coal (API2) futures
    "TTF_Natural_Gas": "TTF=F",    # Dutch TTF natural gas futures (from 2017)
    "Henry_Hub_Gas": "NG=F",       # long-history gas fallback
    "Brent_Crude": "BZ=F",
    "Euro_Stoxx_50": "^STOXX50E",
    "DAX": "^GDAXI",
    "VIX": "^VIX",
    "EURUSD": "EURUSD=X",
}

#: Named in the paper but with no free public feed.
UNAVAILABLE_FEATURES: dict[str, str] = {
    "Epex_Spot_Germany": "EPEX SPOT day-ahead DE - commercial licence required",
    "Citi_CESI_Eurozone": "Citi Economic Surprise Index - Bloomberg/Citi only",
    "Euribor_1W": "1-week Euribor - EMMI redistribution licence required",
}

#: EU ETS trading phases (objective, from EU legislation).
ETS_PHASES: list[tuple[str, str, int]] = [
    ("2005-01-01", "2007-12-31", 1),
    ("2008-01-01", "2012-12-31", 2),
    ("2013-01-01", "2020-12-31", 3),
    ("2021-01-01", "2030-12-31", 4),
]

#: Policy / market events with the price impact reported in Section 4.1 of the
#: paper.  ``sign`` is +1 for bullish and -1 for bearish carbon price news.
POLICY_EVENTS: list[tuple[str, int, str]] = [
    ("2007-12-31", -1, "End of EU ETS Phase I: surplus allowances, no banking, price to ~EUR 0"),
    ("2008-01-01", +1, "Phase II starts: tighter caps and partial CDM credits, price back to ~EUR 20"),
    ("2008-09-15", -1, "Global financial crisis: industrial output collapses, EUR 20 -> EUR 8"),
    ("2009-12-18", -1, "COP15 Copenhagen fails to reach a binding agreement, EUR 15 -> EUR 12"),
    ("2012-06-01", -1, "Eurozone debt crisis and persistent oversupply, price EUR 6.76"),
    ("2013-04-16", -1, "European Parliament rejects backloading, EUR 7.10 -> EUR 2.75"),
    ("2015-07-01", +1, "EU adopts the Market Stability Reserve proposal"),
    ("2018-01-01", +1, "Post-2020 reform: LRF raised to 2.2%, free allocation tightened, EUR 7 -> EUR 25"),
    ("2021-07-14", +1, "Fit for 55 package: 2030 target raised from 40% to 62%, EUR 33 -> EUR 60"),
    ("2022-02-24", +1, "Russia-Ukraine war: gas spike, coal switching, peak EUR 97 in Aug 2022"),
    ("2023-02-21", +1, "EUA closes above EUR 100 for the first time (EUR 101)"),
    ("2024-01-01", -1, "Mild winter, weak power demand, ample renewables, EUR 84 -> EUR 52"),
]


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------
def _user_cache_dir() -> Path:
    """Per-user cache directory, following each platform's convention."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "peltwtcn" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "peltwtcn"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "peltwtcn"


def cache_dir(path: str | os.PathLike | None = None) -> Path:
    """Directory used to cache downloaded series.

    Resolution order:

    1. ``path``, if given.
    2. The ``PELTWTCN_CACHE`` environment variable.
    3. ``./data`` **if it already exists** - so running from a clone of the
       repository keeps using the CSVs shipped with it, and nothing has to be
       re-downloaded.
    4. Otherwise a per-user cache directory, e.g.
       ``%LOCALAPPDATA%\\peltwtcn\\Cache`` on Windows,
       ``~/Library/Caches/peltwtcn`` on macOS,
       ``~/.cache/peltwtcn`` elsewhere.

    Step 3 is deliberately conditional on the directory *already* existing.  An
    installed copy of the package must not scatter a ``data/`` folder into
    whatever directory the user happens to be working in.

    Examples
    --------
    >>> cache_dir().is_dir()
    True
    """
    if path is not None:
        root = Path(path)
    elif os.environ.get("PELTWTCN_CACHE"):
        root = Path(os.environ["PELTWTCN_CACHE"])
    elif Path("data").is_dir():
        root = Path("data")
    else:
        root = _user_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cached(name: str, loader, use_cache: bool = True) -> pd.DataFrame:
    fp = cache_dir() / f"{name}.csv"
    if use_cache and fp.exists():
        return pd.read_csv(fp, index_col=0, parse_dates=True)
    df = loader()
    if use_cache and df is not None and len(df):
        df.to_csv(fp)
    return df


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/csv, */*",
}


def fetch_investing(
    pair_id: int = CARBON_ID,
    start: str = PAPER_START,
    end: str = PAPER_END,
    timeout: int = 60,
) -> pd.DataFrame:
    """Download a daily OHLCV history from investing.com.

    Parameters
    ----------
    pair_id : int
        Instrument id.  ``8848`` is EUA "Carbon Emissions Futures", the series
        the paper used.
    start, end : str
        ``YYYY-MM-DD`` bounds, inclusive.

    Returns
    -------
    DataFrame
        Columns ``Open, High, Low, Close, Volume`` indexed by ``Date``
        (trading days only).

    Examples
    --------
    >>> df = fetch_investing(8848, "2024-05-01", "2024-06-04")   # doctest: +SKIP
    >>> float(df["Close"].iloc[-1])                              # doctest: +SKIP
    72.58
    """
    url = (
        f"https://api.investing.com/api/financialdata/historical/{int(pair_id)}"
        f"?start-date={start}&end-date={end}"
        f"&time-frame=Daily&add-missing-rows=false"
    )
    r = requests.get(url, headers={**_UA, "domain-id": "www"}, timeout=timeout)
    r.raise_for_status()
    rows = r.json().get("data", [])
    if not rows:
        raise RuntimeError(f"investing.com returned no rows for pair {pair_id}")

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["rowDateTimestamp"]).dt.tz_localize(None).dt.normalize()
    def _num(col: str) -> np.ndarray:
        # .to_numpy() is essential: passing Series into DataFrame(index=dates)
        # would reindex them by their integer position and yield all-NaN.
        if col not in df:
            return np.full(len(df), np.nan)
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "Open": _num("last_openRaw"),
            "High": _num("last_maxRaw"),
            "Low": _num("last_minRaw"),
            "Close": _num("last_closeRaw"),
            "Volume": _num("volumeRaw"),
        },
        index=pd.DatetimeIndex(df["Date"]),
    ).sort_index()
    out.index.name = "Date"
    return out[~out.index.duplicated(keep="last")]


def fetch_yahoo(
    ticker: str,
    start: str = PAPER_START,
    end: str = PAPER_END,
    column: str = "Close",
) -> pd.Series:
    """Download one daily price series from Yahoo Finance.

    Returns an empty Series (with a warning) rather than raising, so that one
    dead ticker never breaks a whole feature build.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install yfinance") from exc

    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(ticker, start=start, end=end_excl, progress=False,
                         auto_adjust=True, threads=False)
    if df is None or len(df) == 0:
        warnings.warn(f"Yahoo returned no data for {ticker!r}", RuntimeWarning)
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = pd.to_numeric(df[column], errors="coerce")
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.name = ticker
    return s.sort_index()


def fetch_epu(country: Literal["US", "UK"] = "US", timeout: int = 60) -> pd.Series:
    """Daily Economic Policy Uncertainty index (Baker, Bloom & Davis)."""
    urls = {
        "US": "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv",
        "UK": "https://www.policyuncertainty.com/media/UK_Daily_Policy_Data.csv",
    }
    r = requests.get(urls[country], headers=_UA, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.dropna(subset=["year", "month", "day"])
    idx = pd.to_datetime(
        dict(year=df["year"].astype(int), month=df["month"].astype(int),
             day=df["day"].astype(int)), errors="coerce"
    )
    s = pd.Series(pd.to_numeric(df["daily_policy_index"], errors="coerce").values,
                  index=idx, name=f"EPU_{country}")
    return s[~s.index.isna()].sort_index()


def fetch_gpr(timeout: int = 90) -> pd.Series:
    """Daily Geopolitical Risk index (Caldara & Iacoviello)."""
    base = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent"
    df, last = None, None
    for suffix, engine in ((".xlsx", "openpyxl"), (".xls", "xlrd"), (".xls", None)):
        try:
            r = requests.get(base + suffix, headers=_UA, timeout=timeout)
            r.raise_for_status()
            df = pd.read_excel(io.BytesIO(r.content), engine=engine)
            break
        except Exception as exc:  # missing file, or missing/old excel engine
            last = exc
    if df is None:
        raise RuntimeError(f"could not read the GPR workbook: {last}")
    dcol = next((c for c in df.columns if str(c).lower().startswith("date")), df.columns[0])
    vcol = next((c for c in df.columns if str(c).upper() in ("GPRD", "GPR")), None)
    if vcol is None:
        vcol = df.select_dtypes("number").columns[0]
    s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values,
                  index=pd.to_datetime(df[dcol], errors="coerce"), name="GPR")
    return s[~s.index.isna()].sort_index()


def fetch_ecb(series_key: str = "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
              start: str = PAPER_START, timeout: int = 90) -> pd.Series:
    """Any daily series from the ECB Data Portal.

    The default key is the euro-area 10-year government benchmark bond yield,
    i.e. the paper's "EU: 10-Year Government Bond Yield".
    """
    flow, _, key = series_key.partition(".")
    url = (f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
           f"?format=csvdata&startPeriod={start}")
    r = requests.get(url, headers=_UA, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
                  index=pd.to_datetime(df["TIME_PERIOD"], errors="coerce"),
                  name=series_key.split(".")[-1])
    return s[~s.index.isna()].sort_index()


# ---------------------------------------------------------------------------
# high-level loaders
# ---------------------------------------------------------------------------
def load_eua(
    start: str = PAPER_START,
    end: str = PAPER_END,
    frequency: Literal["calendar", "trading"] = "calendar",
    source: Literal["investing", "csv"] = "investing",
    csv_path: str | os.PathLike | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load the EUA carbon price series.

    Parameters
    ----------
    frequency : {"calendar", "trading"}
        ``"calendar"`` reindexes to every calendar day and forward-fills,
        reproducing the paper's N = 6113.  ``"trading"`` keeps the 4,346
        genuine trading days.
    source : {"investing", "csv"}
        ``"csv"`` reads your own export (e.g. the investing.com download);
        it must contain a date column and a close/price column.

    Returns
    -------
    DataFrame
        Column ``Carbon_Price`` (plus OHLCV when available), indexed by date.

    Examples
    --------
    >>> eua = load_eua()                                  # doctest: +SKIP
    >>> len(eua)                                          # doctest: +SKIP
    6113
    """
    if source == "csv":
        if csv_path is None:
            raise ValueError("csv_path is required when source='csv'")
        raw = pd.read_csv(csv_path)
        dcol = next(c for c in raw.columns if "date" in str(c).lower())
        pcol = next(c for c in raw.columns
                    if str(c).lower() in ("close", "price", "last", "carbon_price"))
        raw[dcol] = pd.to_datetime(raw[dcol], errors="coerce", dayfirst=False)
        px = (raw.dropna(subset=[dcol])
                 .set_index(dcol)[pcol]
                 .pipe(pd.to_numeric, errors="coerce")
                 .sort_index())
        df = px.to_frame("Close")
    else:
        df = _cached(
            f"eua_investing_{start}_{end}",
            lambda: fetch_investing(CARBON_ID, start, end),
            use_cache,
        )

    df = df.loc[str(start): str(end)].copy()
    if frequency == "calendar":
        idx = pd.date_range(start, end, freq="D")
        df = df.reindex(idx).ffill().bfill()
        df.index.name = "Date"
    df.insert(0, "Carbon_Price", df["Close"].astype(float))
    return df


def load_exogenous(
    start: str = PAPER_START,
    end: str = PAPER_END,
    frequency: Literal["calendar", "trading"] = "calendar",
    features: Iterable[str] | None = None,
    include_uncertainty: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Assemble the exogenous feature block u_t from live sources.

    Parameters
    ----------
    features : iterable of str, optional
        Subset of :data:`YAHOO_FEATURES` keys.  ``None`` = all of them.
    include_uncertainty : bool
        Also fetch the US/UK EPU indices, the GPR index and the euro-area
        10-year yield.

    Returns
    -------
    DataFrame
        One column per feature, forward-filled onto the requested calendar.
    """
    keys = list(features) if features is not None else list(YAHOO_FEATURES)
    cols: dict[str, pd.Series] = {}

    for name in keys:
        ticker = YAHOO_FEATURES[name]
        s = _cached(f"yf_{ticker.replace('=','_').replace('^','idx_')}_{start}_{end}",
                    lambda t=ticker: fetch_yahoo(t, start, end).to_frame("value"),
                    use_cache)
        if isinstance(s, pd.DataFrame):
            s = s["value"] if "value" in s else s.iloc[:, 0]
        if len(s):
            cols[name] = s

    if include_uncertainty:
        for name, fn in (("EPU_US", lambda: fetch_epu("US").to_frame("value")),
                         ("EPU_UK", lambda: fetch_epu("UK").to_frame("value")),
                         ("GPR", lambda: fetch_gpr().to_frame("value")),
                         ("EU_10Y_Yield", lambda: fetch_ecb().to_frame("value"))):
            try:
                d = _cached(f"{name.lower()}", fn, use_cache)
                s = d["value"] if isinstance(d, pd.DataFrame) else d
                if len(s):
                    cols[name] = s
            except Exception as exc:  # network / provider hiccup
                warnings.warn(f"{name} unavailable: {type(exc).__name__}: {exc}",
                              RuntimeWarning)

    if not cols:
        raise RuntimeError("no exogenous series could be downloaded")

    idx = (pd.date_range(start, end, freq="D") if frequency == "calendar"
           else pd.bdate_range(start, end))
    out = pd.DataFrame(index=idx)
    out.index.name = "Date"
    for k, s in cols.items():
        s = s[~s.index.duplicated(keep="last")]
        out[k] = s.reindex(out.index.union(s.index)).ffill().reindex(out.index)
    return out.ffill().bfill()


def build_policy_features(index: pd.DatetimeIndex, halflife: float = 30.0) -> pd.DataFrame:
    """Reconstruct the paper's undefined "Policy" feature.

    The paper reports "Policy" as by far the most important predictor
    (Extra-Trees importance > 0.5) but never says what it is.  This function
    builds a transparent, fully documented stand-in from three ingredients:

    ``Policy_Phase``
        EU ETS trading phase (1-4) from EU legislation.  Purely objective.
    ``Policy_Event``
        Signed impulse at each of the twelve events the paper itself lists and
        dates in Section 4.1 (see :data:`POLICY_EVENTS`).
    ``Policy_Shock``
        Exponentially decaying version of ``Policy_Event`` with the given
        half-life in days, so that a policy announcement keeps influencing the
        model for several weeks.
    ``Policy``
        ``Policy_Shock`` plus the phase level; the single column used as the
        paper's "Policy" input.

    Because the impulses are dated from a document written *after* the sample,
    this feature is mildly forward-looking by construction.  Set
    ``PELTWTPipeline(use_policy=False)`` to exclude it from an honest
    out-of-sample exercise.

    Examples
    --------
    >>> idx = pd.date_range("2021-07-01", "2021-08-01", freq="D")
    >>> pf = build_policy_features(idx)
    >>> float(pf.loc["2021-07-14", "Policy_Event"])
    1.0
    """
    index = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=index)
    out.index.name = "Date"

    phase = pd.Series(0, index=index, dtype=float)
    for lo, hi, p in ETS_PHASES:
        phase.loc[(index >= lo) & (index <= hi)] = float(p)
    out["Policy_Phase"] = phase.replace(0.0, np.nan).ffill().bfill()

    ev = pd.Series(0.0, index=index)
    for date, sign, _desc in POLICY_EVENTS:
        d = pd.Timestamp(date)
        pos = index.searchsorted(d)
        if 0 <= pos < len(index):
            ev.iloc[pos] += float(sign)
    out["Policy_Event"] = ev

    decay = np.exp(-np.log(2.0) / float(halflife))
    shock = np.zeros(len(index))
    acc = 0.0
    for i, v in enumerate(ev.to_numpy()):
        acc = acc * decay + v
        shock[i] = acc
    out["Policy_Shock"] = shock
    out["Policy"] = out["Policy_Phase"] + out["Policy_Shock"]
    return out


def load_paper_dataset(
    start: str = PAPER_START,
    end: str = PAPER_END,
    frequency: Literal["calendar", "trading"] = "calendar",
    include_policy: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """One call that builds the complete modelling frame used by the paper.

    Returns
    -------
    DataFrame
        ``Carbon_Price`` first, then every exogenous feature, indexed by date.
        With the defaults this has **6,113 rows**, matching the paper.

    Examples
    --------
    >>> df = load_paper_dataset()          # doctest: +SKIP
    >>> df.shape                           # doctest: +SKIP
    (6113, 13)
    """
    eua = load_eua(start, end, frequency=frequency, use_cache=use_cache)
    exo = load_exogenous(start, end, frequency=frequency, use_cache=use_cache)
    df = eua[["Carbon_Price"]].join(exo, how="left")
    if include_policy:
        df = df.join(build_policy_features(df.index)[["Policy"]], how="left")
    return df.ffill().bfill()
