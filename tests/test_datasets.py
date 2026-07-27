"""Data loading.  Network tests are marked and skipped by default."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peltwtcn import (CARBON_ID, PAPER_END, PAPER_START, POLICY_EVENTS,
                      UNAVAILABLE_FEATURES, YAHOO_FEATURES,
                      build_policy_features, cache_dir, load_paper_dataset)
from peltwtcn.datasets import ETS_PHASES


# --------------------------------------------------------------------------
# constants, checked against the paper
# --------------------------------------------------------------------------
def test_sample_period_matches_the_paper():
    assert PAPER_START == "2007-09-10"
    assert PAPER_END == "2024-06-04"


def test_carbon_instrument_id_is_set():
    assert isinstance(CARBON_ID, int) and CARBON_ID > 0


def test_yahoo_feature_map_covers_the_energy_and_financial_drivers():
    assert {"Europe_Coal", "TTF_Natural_Gas", "Brent_Crude", "Euro_Stoxx_50",
            "VIX", "EURUSD"} <= set(YAHOO_FEATURES)
    assert all(isinstance(v, str) and v for v in YAHOO_FEATURES.values())


def test_unavailable_features_are_documented_with_a_reason():
    assert set(UNAVAILABLE_FEATURES) == {"Epex_Spot_Germany",
                                         "Citi_CESI_Eurozone", "Euribor_1W"}
    assert all(len(v) > 20 for v in UNAVAILABLE_FEATURES.values())


def test_policy_events_are_well_formed_and_chronological():
    dates = [pd.Timestamp(d) for d, _, _ in POLICY_EVENTS]
    assert dates == sorted(dates)
    assert all(s in (-1, +1) for _, s, _ in POLICY_EVENTS)
    assert all(len(desc) > 10 for _, _, desc in POLICY_EVENTS)


def test_policy_events_lie_inside_the_sample():
    lo, hi = pd.Timestamp(PAPER_START), pd.Timestamp(PAPER_END)
    for d, _, _ in POLICY_EVENTS:
        assert lo <= pd.Timestamp(d) <= hi + pd.Timedelta(days=1)


def test_ets_phases_are_contiguous_and_numbered():
    assert [p for _, _, p in ETS_PHASES] == [1, 2, 3, 4]
    for (_, end, _), (start, _, _) in zip(ETS_PHASES, ETS_PHASES[1:]):
        assert pd.Timestamp(start) == pd.Timestamp(end) + pd.Timedelta(days=1)


# --------------------------------------------------------------------------
# policy features
# --------------------------------------------------------------------------
def test_policy_features_are_built_on_the_given_index():
    idx = pd.bdate_range(PAPER_START, PAPER_END)
    pf = build_policy_features(idx)
    assert len(pf) == len(idx)
    assert list(pf.index) == list(idx)
    assert np.all(np.isfinite(pf.to_numpy(float)))


def test_policy_features_expose_the_four_documented_columns():
    idx = pd.bdate_range("2021-06-01", "2021-09-01")
    pf = build_policy_features(idx)
    assert list(pf.columns) == ["Policy_Phase", "Policy_Event",
                               "Policy_Shock", "Policy"]


def test_policy_event_is_an_impulse_on_the_event_date():
    idx = pd.date_range("2021-07-01", "2021-08-01", freq="D")
    pf = build_policy_features(idx)
    # Fit for 55, 14 July 2021, a bullish (+1) event
    assert float(pf.loc["2021-07-14", "Policy_Event"]) == 1.0
    assert float(pf.loc["2021-07-15", "Policy_Event"]) == 0.0


def test_policy_shock_jumps_at_the_event_date():
    idx = pd.date_range("2021-07-01", "2021-08-01", freq="D")
    pf = build_policy_features(idx, halflife=30.0)
    before = float(pf.loc["2021-07-13", "Policy_Shock"])
    after = float(pf.loc["2021-07-14", "Policy_Shock"])
    assert after - before == pytest.approx(1.0, abs=0.05)


def test_policy_is_the_phase_level_plus_the_shock():
    idx = pd.bdate_range("2021-06-01", "2021-09-01")
    pf = build_policy_features(idx)
    assert np.allclose(pf["Policy"], pf["Policy_Phase"] + pf["Policy_Shock"])


def test_policy_phase_is_four_after_2021():
    idx = pd.bdate_range("2021-06-01", "2021-09-01")
    pf = build_policy_features(idx)
    assert (pf["Policy_Phase"] == 4.0).all()


def test_policy_shock_decays_faster_with_a_smaller_halflife():
    idx = pd.date_range("2021-07-14", "2021-12-31", freq="D")
    fast = build_policy_features(idx, halflife=5.0)
    slow = build_policy_features(idx, halflife=90.0)
    tail = slice("2021-11-01", None)
    assert (fast["Policy_Shock"].loc[tail].abs().mean()
            < slow["Policy_Shock"].loc[tail].abs().mean())


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------
def test_cache_dir_is_created(tmp_path, monkeypatch):
    monkeypatch.setenv("PELTWTCN_CACHE", str(tmp_path / "cache"))
    d = cache_dir()
    assert d.exists() and d.is_dir()


def test_explicit_path_wins_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PELTWTCN_CACHE", str(tmp_path / "from_env"))
    assert cache_dir(tmp_path / "explicit").name == "explicit"


def test_cache_dir_uses_a_local_data_folder_when_one_exists(tmp_path, monkeypatch):
    """Running from a clone must reuse the CSVs shipped with the repository."""
    monkeypatch.delenv("PELTWTCN_CACHE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    assert cache_dir() == Path("data")


def test_cache_dir_falls_back_to_a_user_directory_when_installed(tmp_path,
                                                                monkeypatch):
    """An installed copy must not scatter a data/ folder into the cwd."""
    monkeypatch.delenv("PELTWTCN_CACHE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "data").exists()

    d = cache_dir()
    assert d.is_absolute()
    assert d.name.lower() in {"cache", "peltwtcn"}
    assert "peltwtcn" in str(d).lower()
    # the crucial part: nothing was created in the working directory
    assert not (tmp_path / "data").exists()


# --------------------------------------------------------------------------
# the bundled dataset
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundled():
    """The dataset assembled from the CSVs shipped in ``data/``."""
    try:
        return load_paper_dataset(frequency="calendar")
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"bundled data unavailable: {exc}")


def test_bundled_dataset_has_the_papers_sample_size(bundled):
    """The paper reports 6,113 daily observations."""
    assert len(bundled) == 6113


def test_bundled_dataset_spans_the_paper_period(bundled):
    assert bundled.index[0] == pd.Timestamp(PAPER_START)
    assert bundled.index[-1] == pd.Timestamp(PAPER_END)


def test_bundled_dataset_has_a_carbon_price_and_drivers(bundled):
    assert "Carbon_Price" in bundled.columns
    assert "Policy" in bundled.columns
    assert bundled.shape[1] >= 10


def test_bundled_dataset_is_complete_and_finite(bundled):
    assert int(bundled.isna().sum().sum()) == 0
    assert np.all(np.isfinite(bundled.to_numpy(float)))


def test_bundled_index_is_sorted_and_unique(bundled):
    assert bundled.index.is_monotonic_increasing
    assert bundled.index.is_unique


def test_carbon_price_is_positive_and_peaks_around_100_euro(bundled):
    p = bundled["Carbon_Price"]
    assert p.min() > 0
    # the paper notes an all-time high just above EUR 100 in Feb 2023
    assert 95.0 <= p.max() <= 110.0


def test_carbon_price_is_right_skewed_as_the_paper_states(bundled):
    """Section 4.1: "positive skewness and high kurtosis"."""
    assert bundled["Carbon_Price"].skew() > 0


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------
@pytest.mark.network
def test_live_download_matches_the_bundled_shape():
    fresh = load_paper_dataset(start=PAPER_START, end=PAPER_END,
                               frequency="calendar", use_cache=False)
    assert len(fresh) > 5000
    assert "Carbon_Price" in fresh.columns
