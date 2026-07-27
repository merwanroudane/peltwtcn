"""Table rendering, export, and the comparison against the published Table 1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peltwtcn import (PAPER_TABLE1, PAPER_TRAIN_TIMES, BreakResult,
                      compare_with_paper, describe_breaks, export_table,
                      improvement_table, paper_table1,
                      regimes_from_breakpoints, results_table,
                      summary_statistics)


@pytest.fixture
def table():
    return pd.DataFrame(
        {"MAE": [1.1855, 1.3308, 4.6345],
         "RMSE": [1.5866, 1.6987, 5.3878],
         "MAPE (%)": [1.6451, 1.7401, 5.8731],
         "R2": [0.9888, 0.9872, 0.8712]},
        index=pd.Index(["PELT-WT-TCN", "PELT-WT-GRU", "BP&ICSS-WT-LSTM"],
                       name="Model"),
    )


# --------------------------------------------------------------------------
# the published reference
# --------------------------------------------------------------------------
def test_paper_table1_matches_the_published_values():
    t = paper_table1()
    assert t.shape == (5, 4)
    assert t.loc["PELT-WT-TCN", "MAE"] == 1.1855
    assert t.loc["PELT-WT-TCN", "RMSE"] == 1.5866
    assert t.loc["PELT-WT-TCN", "MAPE (%)"] == 1.6451
    assert t.loc["PELT-WT-TCN", "R2"] == 0.9888
    assert t.loc["BP&ICSS-WT-LSTM", "RMSE"] == 5.3878


def test_paper_table1_ranks_tcn_best_on_every_metric():
    t = PAPER_TABLE1
    for m in ("MAE", "RMSE", "MAPE (%)"):
        assert t[m].idxmin() == "PELT-WT-TCN"
    assert t["R2"].idxmax() == "PELT-WT-TCN"


def test_paper_table1_is_not_mutated_by_callers():
    paper_table1().loc["PELT-WT-TCN", "MAE"] = 99.0
    assert PAPER_TABLE1.loc["PELT-WT-TCN", "MAE"] == 1.1855


def test_both_reported_training_time_sets_are_kept():
    assert set(PAPER_TRAIN_TIMES) == {"figure_16", "section_4.3_text"}
    for d in PAPER_TRAIN_TIMES.values():
        assert set(d) == set(PAPER_TABLE1.index)
        assert max(d, key=d.get) == "PELT-WT-TCN"      # TCN is slowest in both


def test_compare_with_paper_reports_the_signed_difference():
    t = pd.DataFrame({"MAE": [1.20], "RMSE": [1.60]}, index=["PELT-WT-TCN"])
    cmp = compare_with_paper(t, metrics=("MAE", "RMSE"))
    assert cmp.loc["PELT-WT-TCN", ("MAE", "Paper")] == 1.1855
    assert cmp.loc["PELT-WT-TCN", ("MAE", "Replication")] == 1.20
    assert cmp.loc["PELT-WT-TCN", ("MAE", "Diff")] == pytest.approx(0.0145)


def test_compare_with_paper_keeps_rows_the_paper_lacks():
    t = pd.DataFrame({"MAE": [1.2, 0.8]}, index=["PELT-WT-TCN", "Random walk"])
    cmp = compare_with_paper(t, metrics=("MAE",))
    assert "Random walk" in cmp.index
    assert np.isnan(cmp.loc["Random walk", ("MAE", "Paper")])
    assert cmp.loc["Random walk", ("MAE", "Replication")] == 0.8


def test_compare_with_paper_lists_every_published_model():
    cmp = compare_with_paper(pd.DataFrame({"MAE": [1.0]}, index=["PELT-WT-TCN"]),
                             metrics=("MAE",))
    assert set(PAPER_TABLE1.index) <= set(cmp.index)


# --------------------------------------------------------------------------
# improvement
# --------------------------------------------------------------------------
def test_improvement_reproduces_the_70_55_percent_figure(table):
    imp = improvement_table(table, baseline="BP&ICSS-WT-LSTM")
    assert imp.loc["PELT-WT-TCN", "dRMSE (%)"] == pytest.approx(70.55, abs=0.01)
    assert imp.loc["PELT-WT-TCN", "dMAE (%)"] == pytest.approx(74.42, abs=0.01)


def test_improvement_against_the_strongest_baseline_is_not_the_papers_claim(table):
    """The abstract's 22.35% / 18.63% cannot be recovered from Table 1."""
    imp = improvement_table(table, baseline="PELT-WT-GRU")
    assert imp.loc["PELT-WT-TCN", "dRMSE (%)"] == pytest.approx(6.60, abs=0.01)
    assert imp.loc["PELT-WT-TCN", "dMAE (%)"] == pytest.approx(10.92, abs=0.01)


def test_improvement_baseline_row_is_zero(table):
    imp = improvement_table(table, baseline="BP&ICSS-WT-LSTM")
    assert imp.loc["BP&ICSS-WT-LSTM", "dRMSE (%)"] == pytest.approx(0.0)


def test_improvement_picks_the_worst_model_when_no_baseline_given(table):
    imp = improvement_table(table)
    assert imp.attrs["baseline"] == "BP&ICSS-WT-LSTM"


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ["plain", "markdown", "latex", "html"])
def test_every_render_format_produces_text(table, fmt):
    out = results_table(table, fmt=fmt)
    assert isinstance(out, str) and out.strip()
    assert "PELT-WT-TCN" in out


def test_markdown_bolds_the_best_cell(table):
    out = results_table(table, fmt="markdown", bold_best=True)
    assert "**1.1855**" in out


def test_latex_carries_the_caption_and_label(table):
    out = results_table(table, fmt="latex", caption="My caption",
                        label="tab:mine")
    assert "My caption" in out and "tab:mine" in out


def test_bold_best_can_be_switched_off(table):
    assert "**" not in results_table(table, fmt="markdown", bold_best=False)


# --------------------------------------------------------------------------
# descriptive statistics
# --------------------------------------------------------------------------
def test_summary_statistics_reports_the_expected_columns(toy_panel):
    st = summary_statistics(toy_panel)
    assert {"N", "Mean", "Std", "Min", "Median", "Max", "Skewness",
            "Kurtosis", "Jarque-Bera", "JB p", "ADF", "ADF p"} <= set(st.columns)
    assert list(st.index) == list(toy_panel.columns)


def test_summary_statistics_counts_are_right(toy_panel):
    st = summary_statistics(toy_panel)
    assert (st["N"] == len(toy_panel)).all()


def test_summary_statistics_mean_matches_pandas(toy_panel):
    st = summary_statistics(toy_panel)
    assert st.loc["Carbon_Price", "Mean"] == pytest.approx(
        toy_panel["Carbon_Price"].mean(), abs=1e-4)


# --------------------------------------------------------------------------
# break inventory
# --------------------------------------------------------------------------
def test_describe_breaks_lists_one_row_per_break(toy_panel):
    n = len(toy_panel)
    br = BreakResult([100, 300], n, "t", regimes_from_breakpoints(n, [100, 300]), {})
    tbl = describe_breaks(br, toy_panel.index, toy_panel["Carbon_Price"])
    assert len(tbl) == 2
    assert {"Break", "Date", "Mean before", "Mean after", "Shift"} <= set(tbl.columns)
    assert tbl["Date"].iloc[0] == toy_panel.index[100]


def test_describe_breaks_shift_is_the_difference_in_means(toy_panel):
    n = len(toy_panel)
    br = BreakResult([250], n, "t", regimes_from_breakpoints(n, [250]), {})
    tbl = describe_breaks(br, toy_panel.index, toy_panel["Carbon_Price"])
    row = tbl.iloc[0]
    assert row["Shift"] == pytest.approx(row["Mean after"] - row["Mean before"],
                                         abs=1e-8)


def test_describe_breaks_matches_the_nearest_policy_event(toy_panel):
    n = len(toy_panel)
    br = BreakResult([200], n, "t", regimes_from_breakpoints(n, [200]), {})
    events = [(str(toy_panel.index[205].date()), +1, "A nearby event")]
    tbl = describe_breaks(br, toy_panel.index, toy_panel["Carbon_Price"],
                          events=events)
    assert tbl["Nearest event"].iloc[0] == "A nearby event"
    assert abs(int(tbl["Gap (days)"].iloc[0])) <= 10


def test_describe_breaks_with_no_breaks_is_empty(toy_panel):
    n = len(toy_panel)
    br = BreakResult([], n, "t", regimes_from_breakpoints(n, []), {})
    assert len(describe_breaks(br, toy_panel.index,
                              toy_panel["Carbon_Price"])) == 0


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ext", [".csv", ".md", ".tex", ".html"])
def test_export_writes_a_non_empty_file(table, tmp_path, ext):
    p = export_table(table, tmp_path / f"t{ext}")
    assert p.exists() and p.stat().st_size > 0


def test_exported_csv_round_trips(table, tmp_path):
    p = export_table(table, tmp_path / "t.csv")
    back = pd.read_csv(p, index_col=0)
    assert np.allclose(back["RMSE"].values, table["RMSE"].values)


def test_export_creates_missing_parent_directories(table, tmp_path):
    p = export_table(table, tmp_path / "deep" / "deeper" / "t.csv")
    assert p.exists()


def test_export_rejects_an_unknown_extension(table, tmp_path):
    with pytest.raises(ValueError):
        export_table(table, tmp_path / "t.parquet")
