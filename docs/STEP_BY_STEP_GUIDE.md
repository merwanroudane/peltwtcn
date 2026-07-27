# Step-by-step guide

How to write the code yourself, one stage at a time. Every block runs on its
own — paste it into a script or a notebook and it works.

If you would rather run a finished script first and read afterwards, the same
material is packaged as a single runnable tutorial:

```bash
python examples/tutorial_step_by_step.py
```

It takes about five minutes, prints what every stage produced, and its verbatim
output is saved in [`TUTORIAL_OUTPUT.md`](TUTORIAL_OUTPUT.md) so you can check
your own run against it. This document is the narrative version, with the
reasoning behind each choice.

If you only want the finished result, skip to [step 10](#step-10-all-of-it-in-one-call).

**Contents**

1. [Install](#step-1-install)
2. [Get the data](#step-2-get-the-data)
3. [Look at the data first](#step-3-look-at-the-data-first)
4. [Detect the structural breaks](#step-4-detect-the-structural-breaks)
5. [Denoise with a wavelet](#step-5-denoise-with-a-wavelet)
6. [Build the model input `z_t`](#step-6-build-the-model-input-z_t)
7. [Cut it into sliding windows](#step-7-cut-it-into-sliding-windows)
8. [Fit the network](#step-8-fit-the-network)
9. [Score it honestly](#step-9-score-it-honestly)
10. [All of it in one call](#step-10-all-of-it-in-one-call)
11. [The five models of Table 1](#step-11-the-five-models-of-table-1)
12. [Publication tables](#step-12-publication-tables)
13. [Publication figures](#step-13-publication-figures)
14. [Formal tests](#step-14-formal-tests-the-paper-omits)
15. [Your own data](#step-15-your-own-data)

---

## Step 1: Install

For the library alone:

```bash
pip install "peltwtcn[all]"
```

To follow this guide with the real dataset and run the example scripts, install
from a clone instead — the price CSVs and the examples are not shipped in the
PyPI distribution:

```bash
git clone https://github.com/merwanroudane/peltwtcn.git
cd peltwtcn
pip install -e ".[all]"
```

`[all]` pulls in TensorFlow (needed for the three networks), yfinance (for the
live loaders) and seaborn. Without TensorFlow everything except the models still
works: the detectors, the wavelet layer, the metrics, the tables and the plots.

Check it:

```python
import peltwtcn as pw
print(pw.__version__)
```

---

## Step 2: Get the data

One call assembles the carbon price and every driver, aligns them on a common
calendar, forward-fills the gaps and appends the reconstructed policy feature.

```python
import peltwtcn as pw

df = pw.load_paper_dataset(start=pw.PAPER_START,   # "2007-09-10"
                           end=pw.PAPER_END,       # "2024-06-04"
                           frequency="calendar")

print(df.shape)          # (6113, 14) — the paper reports 6,113 samples
print(df.columns.tolist())
print(df.head())
```

The CSVs in `data/` are used as a cache, so the first call is offline and
instant. To force a fresh download pass `use_cache=False`.

Individual loaders, if you want one series at a time:

```python
eua   = pw.load_eua()                        # the EUA spot price
exog  = pw.load_exogenous()                  # every driver
brent = pw.fetch_yahoo("BZ=F")               # any Yahoo ticker
epu   = pw.fetch_epu("US")                   # Economic Policy Uncertainty
gpr   = pw.fetch_gpr()                       # Geopolitical Risk
yld   = pw.fetch_ecb()                       # euro-area 10-year yield
```

Three features named in the paper have no free feed. They are listed so the gap
is explicit:

```python
for name, why in pw.UNAVAILABLE_FEATURES.items():
    print(f"{name:22s} {why}")
```

### Before you go on: look at the split

This single fact determines whether the deep models can work at all.

```python
n_train, n_test = pw.train_test_split_index(len(df), 0.80)
p = df.Carbon_Price
print(f"train  EUR {p[:n_train].min():.2f} - {p[:n_train].max():.2f}")
print(f"test   EUR {p[n_train:].min():.2f} - {p[n_train:].max():.2f}")
```

```
train  EUR 0.01 - 35.14
test   EUR 33.08 - 98.01
```

The test window peaks 2.8× above the training maximum. Come back to this at
[step 8](#step-8-fit-the-network).

---

## Step 3: Look at the data first

Descriptive statistics, with skewness, kurtosis, Jarque–Bera and an augmented
Dickey–Fuller test:

```python
stats = pw.summary_statistics(df)
print(stats[["N", "Mean", "Std", "Min", "Max", "Skewness", "Kurtosis", "ADF p"]])
```

The paper claims the price "exhibits positive skewness and high kurtosis"; this
is where you check it. The ADF column tells you the series is not stationary,
which matters later.

Reproduce the paper's Figures 5 and 6:

```python
pw.set_journal_style()                       # call once, before any plotting

fig = pw.plot_correlation_drivers(df)        # Figure 5
fig.savefig("fig05.png")

fig = pw.plot_feature_importance(df)         # Figure 6, Extra-Trees
fig.savefig("fig06.png")
```

---

## Step 4: Detect the structural breaks

Section 3.1 of the paper. Three detectors, one interface.

```python
price = df.Carbon_Price.to_numpy(float)

# PELT — the paper's choice, O(n)
pelt = pw.pelt_breakpoints(price, pw.PeltConfig(model="l2", min_size=30,
                                                jump=1, penalty="bic"))
print(pelt)                                  # 11 breaks, 12 regimes
print(pelt.dates(df.index))                  # as calendar dates
```

`jump=1` is exact but slower; `jump=5` searches every fifth point and is roughly
five times faster, which is useful while you are still experimenting.

```python
# ICSS — variance shifts (Inclán & Tiao, 1994)
icss = pw.icss_breakpoints(price)

# Bai–Perron — mean shifts by least squares (Bai & Perron, 2003)
bp = pw.bai_perron_breakpoints(price, max_breaks=5, trim=0.15)

# The union of the two, which is the paper's BP&ICSS baseline
bpicss = pw.bp_icss_breakpoints(price, max_breaks=5, trim=0.15, min_size=30)
print(bpicss.detail["bai_perron"], bpicss.detail["icss"])
```

Turn the breaks into an interpretable table, matched to the policy chronology
the paper gives in Section 4.1:

```python
tbl = pw.describe_breaks(pelt, df.index, df.Carbon_Price,
                         events=pw.POLICY_EVENTS)
print(tbl[["Break", "Date", "Mean before", "Mean after", "Shift",
           "Nearest event"]].to_string(index=False))
```

And Figure 7:

```python
fig = pw.plot_breakpoints(df.Carbon_Price, pelt)
fig.savefig("fig07.png")
```

The paper says breaks were "measured for each column of data". That is
`pelt_multivariate`:

```python
per_column = pw.pelt_multivariate(df, pw.PeltConfig(min_size=30, jump=5))
print({k: v.n_breaks for k, v in per_column.items()})

fig = pw.plot_breakpoints(df.Carbon_Price, per_column)   # one colour per feature
```

### The catch you need to know about

```python
train_labels = sorted(set(pelt.labels[:n_train].tolist()))
test_labels  = sorted(set(pelt.labels[n_train:].tolist()))
print("train:", train_labels)                # [0, 1, 2, 3, 4, 5, 6]
print("test :", test_labels)                 # [6, 7, 8, 9, 10, 11]
print("test only:", sorted(set(test_labels) - set(train_labels)))
```

Five of the twelve regimes occur **only** in the test window. Their one-hot
columns are identically zero for every training row, so no network can learn a
weight for them. See
[REPLICATION_NOTES §3.2](REPLICATION_NOTES.md#32-one-hot-regime-dummies-cannot-describe-a-future-regime).

---

## Step 5: Denoise with a wavelet

Section 3.2. The paper decomposes one level and keeps the approximation.

```python
wt = pw.wavelet_denoise(price, pw.WaveletConfig(wavelet="db4", level=1,
                                                denoise_mode="paper"))
fig = pw.plot_denoising(df.Carbon_Price, wt,
                        zoom=("2022-01-01", "2022-12-31"))   # Figure 8
```

The four modes:

| `denoise_mode` | What it does | Causal? |
|---|---|---|
| `"paper"` | zero every detail band, keep the approximation | no |
| `"threshold"` | soft/hard shrinkage of the details | no |
| `"causal"` | `"paper"` recomputed in a trailing window | yes |
| `"causal_threshold"` | `"threshold"` in a trailing window | yes |
| `"none"` | passthrough, the ablation baseline | yes |

Convince yourself the paper's filter looks ahead:

```python
import numpy as np
bumped = price.copy()
bumped[3000:] += 25.0                        # change only the future
diff = np.abs(pw.wavelet_denoise(bumped, pw.WaveletConfig(level=1)) - wt)
first = int(np.nonzero(diff > 1e-10)[0].min())
print(f"look-ahead: {3000 - first} observations")     # 6
```

Six observations of the future are already inside today's "denoised" value. For
a one-step-ahead forecast, use the causal variant:

```python
wt_causal = pw.wavelet_denoise(price, pw.WaveletConfig(
    wavelet="db4", level=1, denoise_mode="causal", causal_window=256))
```

To see the bands themselves:

```python
dec = pw.wavelet_decompose(price, pw.WaveletConfig(level=3))
print(dec.keys())                            # cA, cD, A, D
fig = pw.plot_wavelet_decomposition(df.Carbon_Price, dec)
```

---

## Step 6: Build the model input `z_t`

The paper's unified input vector is `z_t = [ỹ_t, u_t, e_t]`: denoised price,
exogenous block, one-hot regime label.

```python
regimes = pw.build_regime_matrix(pelt, len(df))      # e_t, one-hot
exog    = df.drop(columns=["Carbon_Price"])          # u_t

Z = pw.build_design_matrix(wt, exog, regimes, index=df.index)
print(Z.shape)                                       # (6113, 26)
print(Z.columns.tolist()[:6])
```

The column order is always price, then exogenous, then regimes. For the
univariate model just leave the other two out:

```python
Z_uni = pw.build_design_matrix(wt)                   # (6113, 1)
```

---

## Step 7: Cut it into sliding windows

30 steps in, one step ahead, stride 1 — the paper's setup.

```python
data = pw.make_windows(Z, wt,
                       window=30, horizon=1, stride=1,
                       train_size=0.80,
                       scale="minmax", scale_on="all")
print(data)
print(data.X_train.shape, data.X_test.shape)         # (4860, 30, 26) (1223, 30, 26)
```

`scale_on` is the important argument:

- `"all"` fits the scaler on the whole sample before splitting. This leaks — the
  training rows are normalised using the test period's min and max — but it is
  what makes the paper's numbers attainable, and it is the default in
  `mode="paper"`.
- `"train"` fits on the training rows only. Correct, and what `mode="causal"`
  forces.

A window is a training example only when its **target** falls before the split,
so no training row ever contains a test-period observation.

Everything needed to undo the scaling travels with the object:

```python
data.inverse_y(data.y_test)      # back to EUR; equals data.y_test_raw
```

---

## Step 8: Fit the network

```python
model = pw.build_tcn(data.X_train.shape[1:])
model.summary()
```

Or train in one call:

```python
cfg = pw.TrainConfig()           # every default is the paper's
out = pw.fit_model("tcn", data.X_train, data.y_train, data.X_test, cfg)

print(out["n_params"], out["epochs_run"], f"{out['train_time']:.1f}s")
y_pred = data.inverse_y(out["y_pred"])
```

The three builders are `build_lstm`, `build_gru`, `build_tcn`, and
`build_model("tcn", ...)` dispatches by name. `TrainConfig` holds every
hyper-parameter reported in Section 4.2:

```python
pw.TrainConfig(units=128, n_layers=2, dropout=0.2,          # LSTM / GRU
               tcn_filters=64, tcn_blocks=4, tcn_kernel_size=3,
               learning_rate=1e-3, beta_1=0.9, beta_2=0.999,
               batch_size=64, epochs=50, patience=10,
               validation_split=0.10, seed=42, verbose=0)
```

### Why the level model fails, and what to do

Fit the above on levels and you get RMSE ≈ 55 with R² ≈ −12, the forecast stuck
in a flat band around EUR 20. That is not a bug. An LSTM or GRU squashes its
state through `tanh`; once the inputs leave the range seen in training the state
saturates and the forecast flattens. The test window peaks 2.8× above the
training maximum ([step 2](#before-you-go-on-look-at-the-split)), so it leaves
that range immediately.

The fix is to model the change rather than the level:

```python
pipe = pw.PELTWTPipeline(model="gru", stationary=True).fit(df)
print(pipe.summary())
```

`stationary=True` differences the continuous inputs, predicts the one-step
change, and rebuilds the level as *last observed value + predicted change*. The
metrics stay in EUR, so they remain directly comparable with the paper's
Table 1. Measured on the real data:

| | RMSE (level) | RMSE (stationary) |
|---|---:|---:|
| PELT-WT-GRU | 55.18 | 1.236 |
| PELT-WT-LSTM (multi) | 33.66 | 1.262 |
| PELT-WT-TCN | 33.86 | 2.101 |

---

## Step 9: Score it honestly

```python
print(pw.evaluate(pipe.y_true_, pipe.y_pred_, "PELT-WT-GRU"))
```

Every metric individually:

```python
pw.mae(y, p); pw.rmse(y, p); pw.mape(y, p); pw.smape(y, p)
pw.r2(y, p);  pw.theil_u(y, p)
```

**Always look at Theil's U.** It is the model's RMSE divided by a random walk's.
Below 1 means you beat "tomorrow equals today"; at or above 1 means you did not.
On daily carbon prices in levels, a high R² proves very little — the level is
almost entirely explained by its own last value:

```python
rw = pw.naive_random_walk(pipe.y_true_)
print(pw.evaluate(pipe.y_true_, rw, "Random walk"))
# MAE 0.846  RMSE 1.223  MAPE 1.157 %  R2 0.9934
```

The paper's best reported model has RMSE 1.5866, which is worse than this. The
paper never runs the comparison. `evaluate_many` therefore adds the random-walk
row for you by default.

---

## Step 10: All of it in one call

Steps 4 to 9, in one object:

```python
pipe = pw.PELTWTPipeline(model="tcn", stationary=True).fit(df)
print(pipe.summary())
```

Everything is kept for inspection:

```python
pipe.breaks_        # BreakResult, or a dict of them
pipe.denoised_      # the filtered series
pipe.data_          # the SupervisedData object from step 7
pipe.y_pred_        # predictions, in EUR
pipe.y_true_        # realised values, in EUR
pipe.metrics_       # the metric dict
pipe.history_       # the Keras training history
pipe.train_time_    # seconds
pipe.n_params_      # parameter count
pipe.predictions    # a dated DataFrame: actual / predicted / residual
```

Override any setting as a keyword:

```python
pw.PELTWTPipeline(model="gru", detector="bp_icss", window=60,
                  use_regimes=False, stationary=True,
                  train=pw.TrainConfig(epochs=10)).fit(df)
```

Or build the config explicitly:

```python
cfg = pw.PipelineConfig(mode="causal", model="tcn", stationary=True)
pw.PELTWTPipeline(cfg).fit(df)
```

### The three protocols

| Protocol | Call | Use it for |
|---|---|---|
| A, level | `mode="paper", stationary=False` | reproducing the paper literally |
| B, stationary | `mode="paper", stationary=True` | models that actually work |
| C, causal | `mode="causal", stationary=True` | an honest out-of-sample claim |

`mode="causal"` forces a causal wavelet, a training-only scaler and a raw
target; there is no way to leave a leak in by accident.

---

## Step 11: The five models of Table 1

```python
res = pw.run_experiment(df, mode="paper", stationary=True, verbose=True)
print(pw.results_table(res.table, fmt="plain"))
```

The five specifications are exactly the paper's:

```python
print(pw.PAPER_MODELS)
# BP&ICSS-WT-LSTM, PELT-WT-LSTM (uni), PELT-WT-LSTM (multi),
# PELT-WT-GRU, PELT-WT-TCN
```

`ExperimentResult` gives you:

```python
res.table            # the metric table, sorted by RMSE
res.predictions      # {name: array}
res.actual           # realised values
res.index            # test dates
res.pipelines        # {name: fitted PELTWTPipeline}
res.residuals()      # DataFrame of residuals
res.to_frame()       # actual and every prediction, side by side
res.best("RMSE")     # name of the winner
```

Your own set of specifications:

```python
res = pw.run_experiment(df, models={
    "GRU no regimes": dict(model="gru", use_regimes=False),
    "GRU + regimes":  dict(model="gru", use_regimes=True),
    "TCN raw":        dict(model="tcn", wavelet=pw.WaveletConfig(denoise_mode="none")),
}, stationary=True)
```

---

## Step 12: Publication tables

```python
print(pw.results_table(res.table, fmt="markdown"))   # or plain / latex / html
```

LaTeX with booktabs, a caption and a label, ready to `\input`:

```python
pw.export_table(res.table.round(4), "results/table1.tex",
                caption="Performance comparison for carbon price prediction",
                label="tab:performance")
```

The extension decides the format — `.csv`, `.tex`, `.md`, `.html`, `.xlsx`.

Against the published numbers:

```python
print(pw.paper_table1())                       # Table 1, transcribed verbatim
print(pw.compare_with_paper(res.table))        # Paper / Replication / Diff
```

Percentage improvement over any baseline:

```python
imp = pw.improvement_table(res.table, baseline="BP&ICSS-WT-LSTM")
print(imp[["MAE", "RMSE", "dMAE (%)", "dRMSE (%)"]])
```

This is how you check the abstract's headline claim. Against the paper's own
Table 1 the reduction is 70.55 % in RMSE and 74.42 % in MAE, not the 22.35 % and
18.63 % claimed — see
[REPLICATION_NOTES §6](REPLICATION_NOTES.md#6-the-2235--1863--claim-does-not-reconcile).

---

## Step 13: Publication figures

One call for the whole set:

```python
pw.set_journal_style()
files = pw.save_all_figures(res, df, outdir="assets",
                            denoised=wt, breaks=pelt, fmt=("png", "pdf"))
```

Or individually:

```python
pw.plot_price_history(df.Carbon_Price, events=pw.POLICY_EVENTS)
pw.plot_correlation_drivers(df)                       # Figure 5
pw.plot_feature_importance(df)                        # Figure 6
pw.plot_breakpoints(df.Carbon_Price, pelt)            # Figure 7
pw.plot_denoising(df.Carbon_Price, wt)                # Figure 8
pw.plot_forecast(res.actual, res.predictions["PELT-WT-TCN"],
                 res.index, "PELT-WT-TCN")            # Figures 9-13
pw.plot_all_forecasts(res.actual, res.predictions, res.index)   # Figure 14
pw.plot_model_comparison(res.table)                   # Figure 15
pw.plot_training_time(res.training_times)             # Figure 16
pw.plot_residuals_over_time(res.residuals())          # Figure 17
pw.plot_residual_density(res.residuals())             # Figure 18
pw.plot_dm_heatmap(pw.dm_matrix(res))                 # extra
```

Every function takes `ax=` so you can compose multi-panel layouts, and returns
the figure:

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(7, 6))
pw.plot_breakpoints(df.Carbon_Price, pelt, ax=axes[0], label="(a)")
pw.plot_denoising(df.Carbon_Price, wt, ax=axes[1], label="(b)")
fig.tight_layout()
```

`set_journal_style()` sets serif type, 300 dpi, thin spines and a colour-blind
safe palette. `set_journal_style(serif=False, font_scale=1.2)` for slides.

---

## Step 14: Formal tests the paper omits

The paper reports five models and declares a winner without testing whether any
difference is significant. Two tests fix that.

Diebold–Mariano, pairwise:

```python
t = pw.diebold_mariano(res.actual,
                       res.predictions["PELT-WT-TCN"],
                       res.predictions["PELT-WT-GRU"])
print(t["DM"], t["p_value"], t["better"])
```

A negative statistic favours the first argument. The Harvey–Leybourne–Newbold
small-sample correction is applied by default. The full matrix:

```python
print(pw.dm_matrix(res).round(3))
fig = pw.plot_dm_heatmap(pw.dm_matrix(res))
```

Model Confidence Set (Hansen, Lunde & Nason) — which models survive at the 10 %
level:

```python
mcs = pw.mcs_table(res, alpha=0.10, n_boot=1000)
print(mcs)                       # avg_loss, p_MCS, in_MCS
```

---

## Step 15: Your own data

Nothing is specific to carbon. Any dated `DataFrame` with a target column works:

```python
import pandas as pd

my = pd.read_csv("my_series.csv", index_col=0, parse_dates=True)
# columns: Price, Driver1, Driver2, ...

pipe = pw.PELTWTPipeline(model="tcn", stationary=True).fit(my, price_col="Price")
print(pipe.summary())

res = pw.run_experiment(my, price_col="Price", stationary=True)
print(pw.results_table(res.table, fmt="markdown"))
```

Practical advice, learned the hard way in
[REPLICATION_NOTES §3](REPLICATION_NOTES.md#3-the-arithmetic-that-decides-everything):

- **Compare the train and test ranges before anything else.** If the target
  trends out of its training range, use `stationary=True`.
- **Use `mode="causal"`** for any number you intend to publish as
  out-of-sample.
- **Read Theil's U before R².** On a near-unit-root series in levels, R² is
  close to 1 for any model that merely tracks the level.
- **Keep the random-walk row.** It is the benchmark that matters, and it is free.
- **Be wary of one-hot regime dummies.** If a regime begins after your last
  training observation, the network cannot use it.

---

## Where to go next

- [`SYNTAX.md`](SYNTAX.md) — every function, argument and return value
- [`REPLICATION_NOTES.md`](REPLICATION_NOTES.md) — what matches the paper and what does not
- [`examples/run_full_replication.py`](../examples/run_full_replication.py) — all three protocols end to end
