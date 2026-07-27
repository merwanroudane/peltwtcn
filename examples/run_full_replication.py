"""
Full replication of Ren et al. (2025) on live EU ETS data.

Run it with

    python examples/run_full_replication.py

It assembles the real EUA carbon price and its exogenous drivers, then runs the
five specifications of the paper's Table 1 under three protocols:

  A  "paper"      the level model exactly as the paper specifies it
  B  "stationary" the same thing, but modelling the first difference
  C  "causal"     stationary and with every look-ahead removed

Protocol A is the faithful replication.  It does not work out of sample, and
the point of running all three is to show precisely why.  See
docs/REPLICATION_NOTES.md.

Everything is written to  results/  and  assets/ .

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

# Make the repository copy of peltwtcn importable when the script is run
# straight from a git clone, i.e. without `pip install -e .` first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import peltwtcn as pw

warnings.filterwarnings("ignore")

RESULTS = Path("results")
FIGURES = Path("assets")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

pw.set_journal_style()
pw.set_seed(42)

PELT = pw.PeltConfig(model="l2", min_size=30, jump=1, penalty="bic")


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


# ===========================================================================
# STEP 1 -- real data
# ===========================================================================
banner("STEP 1  Assembling the real EU ETS dataset")
t0 = time.perf_counter()
df = pw.load_paper_dataset(start=pw.PAPER_START, end=pw.PAPER_END,
                           frequency="calendar")
print(f"shape            : {df.shape}   (the paper reports 6,113 samples)")
print(f"period           : {df.index[0].date()} -> {df.index[-1].date()}")
print(f"carbon price     : min EUR {df.Carbon_Price.min():.2f}, "
      f"max EUR {df.Carbon_Price.max():.2f}, "
      f"last EUR {df.Carbon_Price.iloc[-1]:.2f}")
print(f"features         : {', '.join(df.columns)}")
print(f"missing values   : {int(df.isna().sum().sum())}")
print(f"load time        : {time.perf_counter() - t0:.1f}s")

n_train, n_test = pw.train_test_split_index(len(df), 0.80)
print(f"\n80/20 split      : {n_train} train / {n_test} test")
print(f"train price range: EUR {df.Carbon_Price[:n_train].min():.2f} - "
      f"{df.Carbon_Price[:n_train].max():.2f}")
print(f"test  price range: EUR {df.Carbon_Price[n_train:].min():.2f} - "
      f"{df.Carbon_Price[n_train:].max():.2f}")
print("The test window peaks 2.8x above the training maximum. Remember this;")
print("it is the single fact that decides whether the deep models can work.")

stats = pw.summary_statistics(df)
pw.export_table(stats, RESULTS / "table_A1_descriptive_statistics.csv")
pw.export_table(stats, RESULTS / "table_A1_descriptive_statistics.tex",
                caption="Descriptive statistics of the carbon price and its drivers",
                label="tab:descriptives")
print("\nDescriptive statistics (head):")
print(stats[["N", "Mean", "Std", "Min", "Max", "Skewness", "Kurtosis"]].head(4).to_string())


# ===========================================================================
# STEP 2 -- structural breaks
# ===========================================================================
banner("STEP 2  Structural break detection (Section 3.1)")
price = df.Carbon_Price
t0 = time.perf_counter()
pelt = pw.pelt_breakpoints(price.to_numpy(float), PELT)
print(f"PELT     : {pelt.n_breaks:3d} breaks, {pelt.n_regimes} regimes "
      f"({time.perf_counter() - t0:.1f}s)")

bpicss = pw.bp_icss_breakpoints(price.to_numpy(float), max_breaks=5, trim=0.15,
                                min_size=30)
print(f"BP&ICSS  : {bpicss.n_breaks:3d} breaks "
      f"(Bai-Perron {len(bpicss.detail['bai_perron'])}, "
      f"ICSS {len(bpicss.detail['icss'])})")

train_labels = sorted(set(pelt.labels[:n_train].tolist()))
test_labels = sorted(set(pelt.labels[n_train:].tolist()))
print(f"\nregimes seen in training : {train_labels}")
print(f"regimes seen in testing  : {test_labels}")
unseen = sorted(set(test_labels) - set(train_labels))
print(f"regimes that occur ONLY in the test window: {unseen}")
print("Those one-hot columns are identically zero for every training row, so")
print("the network cannot have learned a weight for them.  This is why the")
print("paper's e_t encoding cannot generalise past the last training break.")

breaks_tbl = pw.describe_breaks(pelt, df.index, price, events=pw.POLICY_EVENTS)
pw.export_table(breaks_tbl, RESULTS / "table_A2_pelt_breaks.csv", index=False)
print("\nDetected breaks matched to the policy chronology of Section 4.1:")
cols = ["Break", "Date", "Mean before", "Mean after", "Shift", "Nearest event"]
print(breaks_tbl[cols].to_string(index=False, max_colwidth=52))


# ===========================================================================
# STEP 3 -- wavelet denoising
# ===========================================================================
banner("STEP 3  Wavelet denoising (Section 3.2)")
raw = price.to_numpy(float)
wt_paper = pw.wavelet_denoise(raw, pw.WaveletConfig(wavelet="db4", level=1,
                                                    denoise_mode="paper"))
wt_causal = pw.wavelet_denoise(raw, pw.WaveletConfig(wavelet="db4", level=1,
                                                     denoise_mode="causal",
                                                     causal_window=256))
print(f"paper  (two-sided): RMSE vs raw {np.sqrt(np.mean((wt_paper - raw) ** 2)):.4f}, "
      f"corr {np.corrcoef(wt_paper, raw)[0, 1]:.6f}")
print(f"causal (rolling)  : RMSE vs raw {np.sqrt(np.mean((wt_causal - raw) ** 2)):.4f}, "
      f"corr {np.corrcoef(wt_causal, raw)[0, 1]:.6f}")

# How far into the future does the paper's filter reach?
bumped = raw.copy()
bumped[3000:] += 25.0
diff = np.abs(pw.wavelet_denoise(bumped, pw.WaveletConfig(level=1)) - wt_paper)
first = int(np.nonzero(diff > 1e-10)[0].min())
print(f"\nPerturbing the price from t=3000 onwards changes the paper's denoised")
print(f"series as early as t={first}, i.e. a look-ahead of {3000 - first} observations.")
print("For a one-step-ahead forecast, one observation would already be too many.")


# ===========================================================================
# STEP 4-6 -- the five specifications under three protocols
# ===========================================================================
PROTOCOLS = [
    ("A_paper", dict(mode="paper", stationary=False),
     "Protocol A: the level model, exactly as the paper specifies it"),
    ("B_stationary", dict(mode="paper", stationary=True),
     "Protocol B: the same models, applied to the first difference"),
    ("C_causal", dict(mode="causal", stationary=True),
     "Protocol C: stationary and leakage-free"),
]

runs: dict[str, pw.ExperimentResult] = {}
for tag, kwargs, title in PROTOCOLS:
    banner(f"{title}   {kwargs}")
    t0 = time.perf_counter()
    res = pw.run_experiment(df, verbose=True, pelt=PELT, **kwargs)
    runs[tag] = res
    print(f"\ntotal fitting time: {time.perf_counter() - t0:.1f}s\n")
    print(pw.results_table(res.table, fmt="plain"))
    pw.export_table(res.table.round(4), RESULTS / f"table_{tag}.csv")
    pw.export_table(res.table.round(4), RESULTS / f"table_{tag}.tex",
                    caption=title, label=f"tab:{tag}")


# ===========================================================================
# STEP 7 -- side by side, and against the published table
# ===========================================================================
banner("STEP 7  Comparison against Table 1 of the paper")
print("Table 1 as published in Ren et al. (2025), p. 22:")
print(pw.paper_table1().to_string())

for tag, _, title in PROTOCOLS:
    cmp = pw.compare_with_paper(runs[tag].table)
    pw.export_table(cmp.round(4), RESULTS / f"table_vs_paper_{tag}.csv")
    print(f"\n--- {title} ---")
    print(cmp[[("RMSE", "Paper"), ("RMSE", "Replication"), ("RMSE", "Diff"),
               ("R2", "Replication")]].round(3).to_string())

side = pd.DataFrame({
    "Paper": pw.PAPER_TABLE1["RMSE"],
    "A level": runs["A_paper"].table["RMSE"],
    "B stationary": runs["B_stationary"].table["RMSE"],
    "C causal": runs["C_causal"].table["RMSE"],
})
pw.export_table(side.round(4), RESULTS / "table_protocol_comparison.csv")
print("\nRMSE under every protocol (lower is better):")
print(side.round(4).to_string())

imp = pw.improvement_table(runs["B_stationary"].table,
                           baseline="BP&ICSS-WT-LSTM")
pw.export_table(imp.round(4), RESULTS / "table_improvement.csv")
print("\nImprovement over the BP&ICSS-WT-LSTM baseline, protocol B (%):")
print(imp[["MAE", "RMSE", "dMAE (%)", "dRMSE (%)"]].round(3).to_string())


# ===========================================================================
# STEP 8 -- tests the paper does not run
# ===========================================================================
banner("STEP 8  Diebold-Mariano and Model Confidence Set (protocol C)")
best_res = runs["C_causal"]
dm = pw.dm_matrix(best_res)
pw.export_table(dm.round(4), RESULTS / "table_diebold_mariano.csv")
print("Diebold-Mariano p-values:")
print(dm.round(3).to_string())

best = best_res.best("RMSE")
print(f"\nBest by RMSE: {best}")
for other in best_res.predictions:
    if other == best:
        continue
    t = pw.diebold_mariano(best_res.actual, best_res.predictions[best],
                           best_res.predictions[other])
    verdict = "significant" if t["p_value"] < 0.05 else "NOT significant"
    print(f"  {best} vs {other:<24s} DM={t['DM']:+7.3f}  p={t['p_value']:.4f}  {verdict}")

try:
    mcs = pw.mcs_table(best_res, alpha=0.10, n_boot=500)
    pw.export_table(mcs.round(4), RESULTS / "table_model_confidence_set.csv")
    print("\nModel Confidence Set (alpha = 0.10):")
    print(mcs.round(4).to_string())
except Exception as exc:
    print(f"MCS failed: {exc}")


# ===========================================================================
# STEP 9 -- figures
# ===========================================================================
banner("STEP 9  Rendering figures")
files = pw.save_all_figures(runs["B_stationary"], df, outdir=FIGURES,
                            denoised=wt_paper, breaks=pelt, fmt=("png", "pdf"))
print(f"{len(files)} files written to {FIGURES}/")

with open(RESULTS / "run_config.json", "w", encoding="utf-8") as fh:
    json.dump({tag: runs[tag].config for tag, _, _ in PROTOCOLS}
              | {"n_obs": int(len(df)), "features": list(df.columns),
                 "n_train": int(n_train), "n_test": int(n_test),
                 "pelt_breaks": [str(d.date()) for d in pelt.dates(df.index)],
                 "regimes_only_in_test": unseen},
              fh, indent=2, default=str)

banner("DONE")
print(f"tables  -> {RESULTS.resolve()}")
print(f"figures -> {FIGURES.resolve()}")
