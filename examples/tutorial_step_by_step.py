"""
peltwtcn tutorial: build the whole analysis yourself, one step at a time.
========================================================================

This is the companion script to ``docs/STEP_BY_STEP_GUIDE.md``.  It is written
to be *read top to bottom* and run as-is:

    python examples/tutorial_step_by_step.py

Thirteen numbered steps take you from an empty script to a finished set of
tables and figures.  Every step prints what it produced, so you can check your
own version against ``docs/TUTORIAL_OUTPUT.md``, which is the verbatim output of
this file.

It is deliberately *fast* -- roughly 5 minutes on a CPU -- because a tutorial
you cannot run is not a tutorial.  It uses 8 epochs instead of the paper's 50
and a coarser PELT grid.  For the real thing, with the paper's own settings and
all three protocols, run ``examples/run_full_replication.py`` instead.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")            # render without a display; drop this in Jupyter

import numpy as np
import pandas as pd

# Make the repository copy importable when run from a clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import peltwtcn as pw

OUT = Path("tutorial_output")
OUT.mkdir(exist_ok=True)

STEP = 0


def step(title: str) -> None:
    """Print a numbered banner so the output maps onto the guide."""
    global STEP
    STEP += 1
    print(f"\n{'=' * 72}\nSTEP {STEP}  {title}\n{'=' * 72}", flush=True)


# Tutorial-speed settings.  Comment these two lines out to use the paper's.
TRAIN = pw.TrainConfig(epochs=8, verbose=0)
PELT = pw.PeltConfig(model="l2", min_size=30, jump=5, penalty="bic")


# ===========================================================================
step("Set up: seed everything and pick the journal style")
# ===========================================================================
# Always seed before you build anything, or two runs will not agree.
pw.set_seed(42)

# Call this once.  It sets serif type, thin spines, a colour-blind safe
# palette and 300 dpi on save.  Use serif=False for slides.
pw.set_journal_style()

print(f"peltwtcn version : {pw.__version__}")
print(f"public names     : {len(pw.__all__)}")
print(f"tutorial settings: {TRAIN.epochs} epochs, PELT jump={PELT.jump}")
print(f"writing to       : {OUT.resolve()}")


# ===========================================================================
step("Load the real data")
# ===========================================================================
# One call assembles the EUA carbon price and every driver, aligns them on a
# common calendar, forward-fills gaps and appends the policy feature.
# The CSVs in data/ act as a cache, so this is offline and instant.
t0 = time.perf_counter()
df = pw.load_paper_dataset(start=pw.PAPER_START, end=pw.PAPER_END,
                           frequency="calendar")
print(f"shape      : {df.shape}      <- the paper reports 6,113 samples")
print(f"period     : {df.index[0].date()} -> {df.index[-1].date()}")
print(f"columns    : {', '.join(df.columns)}")
print(f"missing    : {int(df.isna().sum().sum())}")
print(f"load time  : {time.perf_counter() - t0:.2f}s")

print("\nFirst three rows, first five columns:")
print(df.iloc[:3, :5].to_string())

# Three features named in the paper have no free feed.  They are listed
# explicitly rather than silently omitted.
print("\nNamed in the paper but unavailable:")
for name, why in pw.UNAVAILABLE_FEATURES.items():
    print(f"  {name:22s} {why}")


# ===========================================================================
step("Look at the train/test split BEFORE modelling anything")
# ===========================================================================
# This is the single most important diagnostic in the whole exercise.  Do it
# first, every time, on any dataset.
n_train, n_test = pw.train_test_split_index(len(df), train_size=0.80)
price = df["Carbon_Price"]

print(f"split          : {n_train} train / {n_test} test  (80/20, chronological)")
print(f"train range    : EUR {price[:n_train].min():6.2f} - {price[:n_train].max():6.2f}"
      f"   ({price.index[0].date()} -> {price.index[n_train - 1].date()})")
print(f"test  range    : EUR {price[n_train:].min():6.2f} - {price[n_train:].max():6.2f}"
      f"   ({price.index[n_train].date()} -> {price.index[-1].date()})")
ratio = price[n_train:].max() / price[:n_train].max()
print(f"\ntest max / train max = {ratio:.2f}x")
print("The test window peaks far above anything seen in training.  A tanh-based")
print("LSTM or GRU saturates past its training range, so a LEVEL model cannot")
print("extrapolate there.  Remember this at step 10.")


# ===========================================================================
step("Describe the data, and test the paper's claims about it")
# ===========================================================================
stats = pw.summary_statistics(df)
print("Skewness, kurtosis and a unit-root test for the first four columns:")
print(stats[["N", "Mean", "Std", "Min", "Max", "Skewness", "Kurtosis",
             "ADF p"]].head(4).to_string())

sk = float(stats.loc["Carbon_Price", "Skewness"])
adfp = float(stats.loc["Carbon_Price", "ADF p"])
print(f"\nThe paper says the price shows 'positive skewness and high kurtosis'.")
print(f"  skewness    = {sk:+.4f}   -> {'positive, as claimed' if sk > 0 else 'NOT positive'}")
print(f"  ADF p-value = {adfp:.4f}   -> {'non-stationary' if adfp > 0.05 else 'stationary'}")

pw.export_table(stats, OUT / "table_descriptives.csv")
pw.export_table(stats, OUT / "table_descriptives.tex",
                caption="Descriptive statistics", label="tab:desc")
print(f"\nwrote {OUT / 'table_descriptives.csv'} and .tex")


# ===========================================================================
step("Detect the structural breaks (Section 3.1 of the paper)")
# ===========================================================================
y = price.to_numpy(float)

# PELT -- the paper's choice.  O(n), so it scales.
t0 = time.perf_counter()
pelt = pw.pelt_breakpoints(y, PELT)
print(f"PELT      : {pelt.n_breaks:2d} breaks, {pelt.n_regimes} regimes "
      f"({time.perf_counter() - t0:.1f}s)")

# ICSS -- variance shifts, not mean shifts.
icss = pw.icss_breakpoints(y)
print(f"ICSS      : {icss.n_breaks:2d} breaks (variance)")

# Bai-Perron -- mean shifts by least squares.
bp = pw.bai_perron_breakpoints(y, max_breaks=5, trim=0.15)
print(f"Bai-Perron: {bp.n_breaks:2d} breaks (mean)")

# The union of the two -- the paper's BP&ICSS baseline (Lin & Zhang, 2022).
bpicss = pw.bp_icss_breakpoints(y, max_breaks=5, trim=0.15, min_size=30)
print(f"BP&ICSS   : {bpicss.n_breaks:2d} breaks = "
      f"{len(bpicss.detail['bai_perron'])} BP + {len(bpicss.detail['icss'])} ICSS")

# Turn breaks into something a referee can read: match them to the policy
# chronology the paper gives in Section 4.1.
breaks_tbl = pw.describe_breaks(pelt, df.index, price, events=pw.POLICY_EVENTS)
print("\nBreaks, matched to the policy events of Section 4.1:")
print(breaks_tbl[["Break", "Date", "Mean before", "Mean after", "Shift",
                  "Nearest event"]].to_string(index=False, max_colwidth=40))
pw.export_table(breaks_tbl, OUT / "table_breaks.csv", index=False)


# ===========================================================================
step("The catch with one-hot regimes -- check it yourself")
# ===========================================================================
train_labels = sorted(set(pelt.labels[:n_train].tolist()))
test_labels = sorted(set(pelt.labels[n_train:].tolist()))
unseen = sorted(set(test_labels) - set(train_labels))

print(f"regimes in training : {train_labels}")
print(f"regimes in testing  : {test_labels}")
print(f"ONLY in testing     : {unseen}")
print(f"\n{len(unseen)} of {pelt.n_regimes} one-hot columns are identically zero for")
print("every training row, so no network can learn a weight for them.  When the")
print("test period arrives those columns switch on and the model has never seen")
print("the pattern.  This is structural to the paper's e_t encoding: the last")
print("regime always begins after the last training observation.")


# ===========================================================================
step("Denoise with a wavelet (Section 3.2)")
# ===========================================================================
# The paper decomposes one level and keeps the approximation component.
wt = pw.wavelet_denoise(y, pw.WaveletConfig(wavelet="db4", level=1,
                                            denoise_mode="paper"))
# The leakage-free alternative: the same filter in a trailing window.
wt_causal = pw.wavelet_denoise(y, pw.WaveletConfig(wavelet="db4", level=1,
                                                   denoise_mode="causal",
                                                   causal_window=256))
print(f"paper  (two-sided): RMSE vs raw {np.sqrt(np.mean((wt - y) ** 2)):.4f}, "
      f"corr {np.corrcoef(wt, y)[0, 1]:.6f}")
print(f"causal (rolling)  : RMSE vs raw {np.sqrt(np.mean((wt_causal - y) ** 2)):.4f}, "
      f"corr {np.corrcoef(wt_causal, y)[0, 1]:.6f}")

# Prove the paper's filter looks ahead: change only the future, then see how
# far back the output moves.
bumped = y.copy()
bumped[3000:] += 25.0
moved = np.abs(pw.wavelet_denoise(bumped, pw.WaveletConfig(level=1)) - wt)
first = int(np.nonzero(moved > 1e-10)[0].min())
print(f"\nPerturbing the price from t=3000 changes the denoised series from "
      f"t={first}.")
print(f"That is a look-ahead of {3000 - first} observations.  For a one-step-ahead")
print("forecast, one observation would already be too many.")

# The reconstruction identity of equation (17): f = A_J f + sum_j D_j f
dec = pw.wavelet_decompose(y, pw.WaveletConfig(level=3))
recon = np.asarray(dec["A"]) + np.sum(dec["D"], axis=0)
print(f"\nreconstruction identity |A + sum(D) - f|max = "
      f"{np.abs(recon - y).max():.2e}   (should be ~0)")


# ===========================================================================
step("Assemble the model input z_t = [y_t, u_t, e_t]")
# ===========================================================================
regimes = pw.build_regime_matrix(pelt, len(df))              # e_t, one-hot
exog = df.drop(columns=["Carbon_Price"])                     # u_t
Z = pw.build_design_matrix(wt, exog, regimes, index=df.index)

print("denoised price : 1 column")
print(f"exogenous u_t  : {exog.shape[1]} columns")
print(f"regimes e_t    : {regimes.shape[1]} columns")
print(f"z_t total      : {Z.shape[1]} columns, {Z.shape[0]} rows")
print(f"\ncolumn order is always price, then exogenous, then regimes:")
print(f"  {Z.columns.tolist()[:3]} ... {Z.columns.tolist()[-2:]}")

# The univariate variant is just the price on its own.
Z_uni = pw.build_design_matrix(wt)
print(f"\nunivariate variant: {Z_uni.shape}")


# ===========================================================================
step("Cut it into sliding windows")
# ===========================================================================
data = pw.make_windows(Z, wt, window=30, horizon=1, stride=1,
                       train_size=0.80, scale="minmax", scale_on="all")
print(f"X_train {data.X_train.shape}   y_train {data.y_train.shape}")
print(f"X_test  {data.X_test.shape}   y_test  {data.y_test.shape}")
print(f"window={data.window}, n_features={data.n_features}")
print(f"\ntest dates: {data.index_test[0].date()} -> {data.index_test[-1].date()}")

# scale_on is the argument that decides whether the paper's numbers are
# reachable at all:
#   "all"   fits the scaler on the whole sample.  Leaks, but reproduces
#           the paper, and is the default in mode="paper".
#   "train" fits on training rows only.  Correct; forced by mode="causal".
strict = pw.make_windows(Z, wt, window=30, train_size=0.80, scale="minmax",
                         scale_on="train")
print(f"\nscale_on='all'  : X_test in [{data.X_test.min():.2f}, {data.X_test.max():.2f}]")
print(f"scale_on='train': X_test in [{strict.X_test.min():.2f}, {strict.X_test.max():.2f}]"
      "   <- far outside [0,1]")
print("A saturating network cannot do anything sensible with the second one.")


# ===========================================================================
step("Fit a model -- and watch the level version fail")
# ===========================================================================
# First, exactly what the paper specifies: predict the LEVEL.
print("Fitting PELT-WT-GRU on the level (the paper's specification) ...")
pw.set_seed(42)
level = pw.PELTWTPipeline(model="gru", mode="paper", stationary=False,
                          train=TRAIN, pelt=PELT).fit(df)
print(f"  RMSE {level.metrics_['RMSE']:8.4f}   R2 {level.metrics_['R2']:8.4f}")
print(f"  predictions span EUR {level.y_pred_.min():.1f} - {level.y_pred_.max():.1f}")
print(f"  truth       spans EUR {level.y_true_.min():.1f} - {level.y_true_.max():.1f}")
print("  -> the forecast is stuck in a flat band.  This is the saturation")
print("     predicted at step 3, not a bug.")

# Now the same model on the first difference.  Levels are rebuilt as
# (last observed value + predicted change), so the metrics stay in EUR.
print("\nFitting the same model with stationary=True ...")
pw.set_seed(42)
diff = pw.PELTWTPipeline(model="gru", mode="paper", stationary=True,
                         train=TRAIN, pelt=PELT).fit(df)
print(f"  RMSE {diff.metrics_['RMSE']:8.4f}   R2 {diff.metrics_['R2']:8.4f}")
print(f"  predictions span EUR {diff.y_pred_.min():.1f} - {diff.y_pred_.max():.1f}")
print(f"  -> now it tracks.  RMSE improved by a factor of "
      f"{level.metrics_['RMSE'] / diff.metrics_['RMSE']:.0f}.")

print("\nFull report for the working model:")
print(diff.summary())


# ===========================================================================
step("Score it honestly: always include a random walk")
# ===========================================================================
rw = pw.naive_random_walk(diff.y_true_)
print("model vs the no-change benchmark 'tomorrow equals today':")
print(f"  PELT-WT-GRU : {pw.evaluate(diff.y_true_, diff.y_pred_, 'GRU')}")
print(f"  Random walk : {pw.evaluate(diff.y_true_, rw, 'RW')}")
print(f"\nTheil's U = {diff.metrics_['Theil U']:.4f}")
print("  below 1 means you beat the random walk; at or above 1 means you did not.")
print("On daily carbon prices in LEVELS a high R2 proves very little, because")
print("the level is almost entirely explained by its own last value.  The paper")
print("never runs this comparison; its best reported RMSE is 1.5866.")


# ===========================================================================
step("Run all five specifications of Table 1, then compare with the paper")
# ===========================================================================
print("Fitting the five models of Table 1 (stationary protocol) ...")
t0 = time.perf_counter()
res = pw.run_experiment(df, mode="paper", stationary=True, verbose=True,
                        train=TRAIN, pelt=PELT)
print(f"done in {time.perf_counter() - t0:.0f}s\n")
print(pw.results_table(res.table, fmt="plain"))

print("\nAs published in the paper (Table 1, p. 22):")
print(pw.paper_table1().to_string())

print("\nSide by side (Diff = replication - paper):")
cmp = pw.compare_with_paper(res.table)
print(cmp[[("RMSE", "Paper"), ("RMSE", "Replication"), ("RMSE", "Diff")]]
      .round(3).to_string())

# Check the abstract's headline claim for yourself.
imp = pw.improvement_table(res.table, baseline="BP&ICSS-WT-LSTM")
print("\nImprovement over the BP&ICSS-WT-LSTM baseline:")
print(imp[["MAE", "RMSE", "dMAE (%)", "dRMSE (%)"]].round(3).to_string())
print("\nAgainst the paper's OWN Table 1 the reduction is 70.55% RMSE /")
print("74.42% MAE, not the 22.35% / 18.63% the abstract claims:")
paper_imp = pw.improvement_table(pw.PAPER_TABLE1, baseline="BP&ICSS-WT-LSTM")
print(paper_imp[["dMAE (%)", "dRMSE (%)"]].round(2).to_string())

# The extension decides the format, so one loop covers all three.
for name in ("table1.csv", "table1.tex", "table1.md"):
    pw.export_table(res.table.round(4), OUT / name,
                    caption="Performance comparison for carbon price prediction",
                    label="tab:performance")
print(f"\nwrote table1.csv / .tex / .md to {OUT}")


# ===========================================================================
step("Test whether the differences are significant, then draw everything")
# ===========================================================================
# The paper declares a winner without testing it.  Two tests fix that.
dm = pw.dm_matrix(res)
print("Diebold-Mariano p-values (H0: equal predictive accuracy):")
print(dm.round(3).to_string())

best = res.best("RMSE")
print(f"\nbest by RMSE: {best}")
for other in res.predictions:
    if other == best:
        continue
    t = pw.diebold_mariano(res.actual, res.predictions[best],
                           res.predictions[other])
    verdict = "significant" if t["p_value"] < 0.05 else "NOT significant"
    print(f"  {best} vs {other:<22s} DM={t['DM']:+7.3f} p={t['p_value']:.4f} {verdict}")

try:
    mcs = pw.mcs_table(res, alpha=0.10, n_boot=300)
    print("\nModel Confidence Set (alpha = 0.10):")
    print(mcs.round(4).to_string())
except Exception as exc:
    print(f"MCS unavailable: {exc}")

# Every figure of the paper, at 300 dpi.
files = pw.save_all_figures(res, df, outdir=OUT / "figures", denoised=wt,
                            breaks=pelt, fmt=("png",), verbose=False)
print(f"\n{len(files)} figures written to {OUT / 'figures'}")
for f in sorted({Path(x).name for x in files}):
    print(f"  {f}")

print(f"\n{'=' * 72}\nFINISHED.  Everything is in {OUT.resolve()}\n{'=' * 72}")
print("Next: examples/run_full_replication.py runs all three protocols with")
print("the paper's own 50-epoch settings.  See docs/STEP_BY_STEP_GUIDE.md for")
print("the narrative version of these thirteen steps, and docs/SYNTAX.md for the")
print("full API reference.")
