# Syntax reference

Every public name in `peltwtcn`, with its full signature, arguments and return
value. 90 names, grouped by stage.

```python
import peltwtcn as pw
```

**Contents**

- [Constants](#constants)
- [1. Data](#1-data)
- [2. Structural breaks](#2-structural-breaks)
- [3. Wavelet](#3-wavelet)
- [4. Features](#4-features)
- [5. Models](#5-models)
- [6. Pipeline](#6-pipeline)
- [7. Metrics](#7-metrics)
- [8. Tables](#8-tables)
- [9. Figures](#9-figures)

---

## Constants

| Name | Type | Value / meaning |
|---|---|---|
| `pw.__version__` | `str` | `"1.0.0"` |
| `PAPER_START` | `str` | `"2007-09-10"` |
| `PAPER_END` | `str` | `"2024-06-04"` |
| `CARBON_ID` | `int` | `8848`, the investing.com id for Carbon Emissions Futures |
| `YAHOO_FEATURES` | `dict[str, str]` | paper feature name → Yahoo ticker |
| `UNAVAILABLE_FEATURES` | `dict[str, str]` | features named in the paper with no free feed, and why |
| `POLICY_EVENTS` | `list[tuple[str, int, str]]` | the 12 dated events of Section 4.1, as `(date, sign, description)` |
| `ETS_PHASES` | `list[tuple[str, str, int]]` | EU ETS trading phases, as `(start, end, phase)` |
| `PAPER_MODELS` | `dict[str, dict]` | the 5 specifications of Table 1 |
| `PAPER_TABLE1` | `DataFrame` | Table 1, transcribed verbatim |
| `PAPER_TRAIN_TIMES` | `dict[str, dict[str, float]]` | training times, both mutually inconsistent versions |

---

## 1. Data

### `load_paper_dataset`

```python
load_paper_dataset(start="2007-09-10", end="2024-06-04",
                   frequency="calendar", include_policy=True,
                   use_cache=True) -> pd.DataFrame
```

The one call that assembles everything: carbon price, every driver, aligned and
forward-filled, plus the reconstructed policy feature.

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `start`, `end` | `str` | paper period | ISO dates |
| `frequency` | `"calendar"` \| `"trading"` | `"calendar"` | every day, or business days only. `"calendar"` gives the paper's 6,113 |
| `include_policy` | `bool` | `True` | append `build_policy_features` |
| `use_cache` | `bool` | `True` | read/write the CSVs in `data/` |

**Returns** a `DataFrame` indexed by date, with `Carbon_Price` first.

### `load_eua`

```python
load_eua(start=PAPER_START, end=PAPER_END, frequency="calendar",
         source="investing", csv_path=None, use_cache=True) -> pd.DataFrame
```

The EUA price alone. `source="csv"` reads `csv_path` instead of downloading.

### `load_exogenous`

```python
load_exogenous(start=PAPER_START, end=PAPER_END, frequency="calendar",
               features=None, include_uncertainty=True,
               use_cache=True) -> pd.DataFrame
```

Every driver. `features` selects a subset of `YAHOO_FEATURES` keys;
`include_uncertainty=False` drops EPU, GPR and the bond yield.

### `build_policy_features`

```python
build_policy_features(index, halflife=30.0) -> pd.DataFrame
```

Reconstructs the paper's undefined `Policy` input. Returns four columns:

| Column | Meaning |
|---|---|
| `Policy_Phase` | EU ETS trading phase 1–4, objective |
| `Policy_Event` | signed impulse, `+1` bullish / `-1` bearish, on each event date |
| `Policy_Shock` | `Policy_Event` decayed exponentially with `halflife` in days |
| `Policy` | `Policy_Phase + Policy_Shock`, the single column used as input |

### Single-series fetchers

```python
fetch_investing(pair_id=8848, start=PAPER_START, end=PAPER_END, timeout=60) -> pd.DataFrame
fetch_yahoo(ticker, start=PAPER_START, end=PAPER_END, column="Close") -> pd.Series
fetch_epu(country="US", timeout=60) -> pd.Series          # "US" or "UK"
fetch_gpr(timeout=90) -> pd.Series                        # Geopolitical Risk
fetch_ecb(series_key="YC.B.U2...SR_10Y", start=PAPER_START, timeout=90) -> pd.Series
```

### `cache_dir`

```python
cache_dir(path=None) -> Path
```

Where downloads are cached: `./data` by default, overridden by the
`PELTWTCN_CACHE` environment variable. Creates the directory.

---

## 2. Structural breaks

### `PeltConfig`

```python
PeltConfig(model="l2", min_size=30, jump=1, penalty="bic")
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `model` | `"l1"` \| `"l2"` \| `"rbf"` \| `"normal"` | `"l2"` | segment cost |
| `min_size` | `int` | `30` | shortest admissible regime |
| `jump` | `int` | `1` | grid step. `1` is exact, `5` is ~5× faster |
| `penalty` | `float` \| `"bic"` \| `"mbic"` \| `"aic"` | `"bic"` | β in the PELT objective |

### `pelt_breakpoints`

```python
pelt_breakpoints(signal, cfg=None, **kwargs) -> BreakResult
```

Pruned Exact Linear Time detection (Killick et al., 2012). `**kwargs` override
individual `PeltConfig` fields, so `pelt_breakpoints(x, min_size=50)` works.

### `pelt_multivariate`

```python
pelt_multivariate(frame, cfg=None, columns=None, **kwargs) -> dict[str, BreakResult]
```

PELT on every column independently — the paper's "breakpoints were measured for
each column of data".

### `icss_breakpoints`

```python
icss_breakpoints(signal, alpha=0.05, min_size=30, demean=True,
                 max_iter=100) -> BreakResult
```

Iterative Cumulative Sum of Squares (Inclán & Tiao, 1994). Detects **variance**
shifts, not mean shifts.

### `bai_perron_breakpoints`

```python
bai_perron_breakpoints(y, X=None, max_breaks=5, trim=0.15,
                       criterion="bic", n_breaks=None) -> BreakResult
```

Multiple structural change by least squares (Bai & Perron, 2003).

| Argument | Meaning |
|---|---|
| `X` | regressors; `None` fits a mean-shift model |
| `max_breaks` | upper bound on m |
| `trim` | minimum fraction of the sample at each end |
| `criterion` | `"bic"`, `"lwz"`, or `"fixed"` to force `n_breaks` |

### `bp_icss_breakpoints`

```python
bp_icss_breakpoints(signal, max_breaks=5, trim=0.15, alpha=0.05,
                    min_size=30, on_returns_for_icss=True) -> BreakResult
```

The union of the two, reproducing the BP&ICSS baseline of Lin & Zhang (2022).
`result.detail["bai_perron"]` and `result.detail["icss"]` hold the two
contributions separately.

### `BreakResult`

```python
BreakResult(breakpoints, n, method, labels, detail)
```

| Attribute / method | Type | Meaning |
|---|---|---|
| `.breakpoints` | `list[int]` | interior break indices, 0-based |
| `.n` | `int` | length of the series it was estimated on |
| `.method` | `str` | detector label |
| `.labels` | `ndarray` | regime index per observation |
| `.detail` | `dict` | method-specific extras |
| `.n_breaks` | `int` | `len(breakpoints)` |
| `.n_regimes` | `int` | `n_breaks + 1` |
| `.one_hot()` | `ndarray` | `(n, n_regimes)` indicator matrix |
| `.dates(index)` | `list` | break locations on a date index |
| `.to_frame(index=None)` | `DataFrame` | regime label per row |

### Regime helpers

```python
regimes_from_breakpoints(n, breakpoints) -> ndarray     # (n,) integer labels
one_hot_regimes(labels, n_regimes=None) -> ndarray      # (n, n_regimes)
bic_penalty(n, n_params=1, rule="bic", sigma2=1.0) -> float
clear_cache() -> None                                   # clears memoised detections
```

---

## 3. Wavelet

### `WaveletConfig`

```python
WaveletConfig(wavelet="db4", level=1, mode="symmetric",
              denoise_mode="paper", threshold_rule="universal",
              threshold_mode="soft", causal_window=256)
```

| Argument | Values | Meaning |
|---|---|---|
| `wavelet` | any `pywt` name | `"db4"`, `"haar"`, `"sym8"`, `"coif3"`, … |
| `level` | `int` | decomposition levels. The paper uses `1` |
| `mode` | `str` | signal extension: `"symmetric"`, `"periodization"`, … |
| `denoise_mode` | see below | which filter to apply |
| `threshold_rule` | `"universal"` \| `"sqtwolog"` \| `"minimax"` \| `"sure"` | shrinkage rule |
| `threshold_mode` | `"soft"` \| `"hard"` | shrinkage type |
| `causal_window` | `int` | trailing window length L for the causal variants |

`denoise_mode`:

| Value | Behaviour | Causal? |
|---|---|---|
| `"paper"` | zero every detail band, keep the approximation | **no** |
| `"threshold"` | shrink the detail bands | **no** |
| `"causal"` | `"paper"`, recomputed in a trailing window | yes |
| `"causal_threshold"` | `"threshold"` in a trailing window | yes |
| `"none"` | return the input unchanged (ablation) | yes |

### `wavelet_denoise`

```python
wavelet_denoise(signal, cfg=None, **kwargs) -> ndarray
```

Returns an array the same length as `signal`. Dispatches to
`causal_wavelet_denoise` for the causal modes.

### `causal_wavelet_denoise`

```python
causal_wavelet_denoise(signal, cfg=None, **kwargs) -> ndarray
```

Recomputes the transform in a trailing window so the value at `t` uses only data
up to `t`. Costs O(n·L) instead of O(n).

### `wavelet_decompose`

```python
wavelet_decompose(signal, cfg=None, **kwargs) -> dict
```

**Returns** `{"cA", "cD", "A", "D"}` — approximation coefficients, list of
detail coefficients, and the reconstructed bands. `A + sum(D)` equals the input
to numerical precision (the paper's equation 17).

### Threshold helpers

```python
universal_threshold(detail, n=None) -> float   # sigma * sqrt(2 log n), MAD-based
max_useful_level(n, wavelet="db4") -> int
```

---

## 4. Features

### `build_regime_matrix`

```python
build_regime_matrix(breaks, n, encoding="onehot", prefix="regime") -> pd.DataFrame
```

Builds `e_t`. Accepts a single `BreakResult` or the dict from
`pelt_multivariate`, in which case one block of columns is produced per feature.
`encoding="ordinal"` gives a single integer column instead of one-hot.

> One-hot columns for regimes that begin after the last training observation are
> identically zero in training. See
> [REPLICATION_NOTES §3.2](REPLICATION_NOTES.md#32-one-hot-regime-dummies-cannot-describe-a-future-regime).

### `build_design_matrix`

```python
build_design_matrix(denoised_price, exog=None, regimes=None,
                    index=None, price_name="Carbon_Price_WT") -> pd.DataFrame
```

Assembles `z_t = [ỹ_t, u_t, e_t]`. Column order is always price, exogenous,
regimes. Pass `exog=None` for the univariate model, `regimes=None` to ablate.

### `make_windows`

```python
make_windows(Z, target, window=30, horizon=1, stride=1, train_size=0.8,
             scale="minmax", scale_on="train",
             target_name="Carbon_Price") -> SupervisedData
```

| Argument | Default | Meaning |
|---|---|---|
| `window` | `30` | steps per input, T in the paper |
| `horizon` | `1` | steps ahead |
| `stride` | `1` | window step |
| `train_size` | `0.8` | chronological split fraction |
| `scale` | `"minmax"` | `"minmax"`, `"standard"` or `"none"` |
| `scale_on` | `"train"` | `"train"` fits the scaler on training rows only (correct); `"all"` fits on everything (leaks, but reproduces the paper) |

A window becomes a training example only when its **target** falls before the
split, so no training row contains a test observation.

### `SupervisedData`

| Attribute | Meaning |
|---|---|
| `.X_train`, `.X_test` | `(n, window, n_features)` |
| `.y_train`, `.y_test` | scaled targets |
| `.y_train_raw`, `.y_test_raw` | targets in original units |
| `.index_train`, `.index_test` | dates of the targets |
| `.feature_names`, `.target_name` | labels |
| `.scaler_X`, `.scaler_y` | the fitted `WindowScaler`s |
| `.window`, `.n_features` | shape shortcuts |
| `.inverse_y(y)` | scaled predictions → original units |

### `WindowScaler`

```python
WindowScaler(method="minmax", feature_range=(0.0, 1.0))
```

`.fit(X)`, `.transform(X)`, `.fit_transform(X)`, `.inverse_transform(X)`.
Constant columns are handled without dividing by zero.

### `train_test_split_index`

```python
train_test_split_index(n, train_size=0.8) -> tuple[int, int]
```

`train_test_split_index(6113, 0.8)` → `(4890, 1223)`.

---

## 5. Models

### `TrainConfig`

Every hyper-parameter in Section 4.2 of the paper. All defaults are the paper's.

```python
TrainConfig(units=128, n_layers=2, dropout=0.2,
            tcn_filters=64, tcn_blocks=4, tcn_kernel_size=3,
            learning_rate=1e-3, beta_1=0.9, beta_2=0.999,
            batch_size=64, epochs=50, validation_split=0.10,
            patience=10, loss="mse", seed=42, verbose=0)
```

| Argument | Paper value | Applies to |
|---|---|---|
| `units`, `n_layers` | 128, 2 | LSTM, GRU |
| `dropout` | 0.2 | all |
| `tcn_filters`, `tcn_blocks` | 64, 4 | TCN |
| `tcn_kernel_size` | 3 | TCN |
| `learning_rate`, `beta_1`, `beta_2` | 1e-3, 0.9, 0.999 | Adam |
| `batch_size`, `epochs` | 64, 50 | all |
| `validation_split`, `patience` | 0.10, 10 | early stopping on val MSE |

`.as_dict()` round-trips through the constructor.

### Builders

```python
build_lstm(input_shape, cfg=None)                    # 2 x 128 LSTM
build_gru(input_shape, cfg=None)                     # 2 x 128 GRU
build_tcn(input_shape, cfg=None, dilations=None)     # 4 blocks x 64 ch
build_model(kind, input_shape, cfg=None)             # kind: "lstm"|"gru"|"tcn"
```

`input_shape` is `(window, n_features)`, i.e. `data.X_train.shape[1:]`. All
return a compiled-ready Keras model with a scalar output.

`build_tcn` dilations default to `[1, 2, 4, 8]`, giving a receptive field of
`1 + 2(k-1)·Σd = 61` steps for `k = 3`, comfortably covering the 30-step window.

> The dilated causal convolution is written as an explicit sum over its taps,
> not `tf.nn.conv1d(dilations=d)`, because TensorFlow's CPU backend cannot
> backpropagate a dilated convolution. It is verified for exactness and strict
> causality in `tests/test_models.py`.

### `fit_model`

```python
fit_model(kind, X_train, y_train, X_test, cfg=None) -> dict
```

Compiles, trains with early stopping, and predicts. **Returns**

| Key | Meaning |
|---|---|
| `"model"` | the fitted Keras model |
| `"history"` | `dict` of loss curves |
| `"y_pred"` | test predictions, still **scaled** |
| `"train_time"` | seconds |
| `"epochs_run"` | epochs before early stopping |
| `"n_params"` | parameter count |

`shuffle=False` always — a time series is never shuffled across the split.

### `set_seed`

```python
set_seed(seed=42) -> None
```

Seeds Python, NumPy and TensorFlow. Call before building anything.

---

## 6. Pipeline

### `PipelineConfig`

```python
PipelineConfig(mode="paper", train_size=0.8, window=30, horizon=1, stride=1,
               scale="minmax", scale_on="all",
               detector="pelt", pelt=PeltConfig(), per_column_breaks=False,
               max_breaks_bp=5, trim_bp=0.15, use_regimes=True,
               wavelet=WaveletConfig(),
               model="tcn", multivariate=True, target="denoised",
               train=TrainConfig(), stationary=False)
```

| Argument | Default | Meaning |
|---|---|---|
| `mode` | `"paper"` | `"paper"` or `"causal"`. `"causal"` forces a causal wavelet, `scale_on="train"` and `target="raw"` |
| `detector` | `"pelt"` | `"pelt"`, `"bp_icss"` or `"none"` |
| `per_column_breaks` | `False` | run the detector on every feature |
| `use_regimes` | `True` | include `e_t` |
| `model` | `"tcn"` | `"lstm"`, `"gru"` or `"tcn"` |
| `multivariate` | `True` | include the exogenous block |
| `target` | `"denoised"` | predict the filtered or the raw price |
| `scale_on` | `"all"` | see below |
| `stationary` | `False` | model the first difference and rebuild the level |

Two arguments decide whether the thing works at all:

- **`scale_on="all"`** fits the scaler on the whole sample. This leaks, but it is
  what makes the paper's numbers attainable, so it is the default in
  `mode="paper"`. `mode="causal"` forces `"train"`.
- **`stationary=True`** differences the continuous inputs, predicts the change,
  and reconstructs the level as *last observed value + predicted change*.
  Metrics stay in price units. Without it, the deep models cannot extrapolate
  past the training range and R² goes negative —
  [REPLICATION_NOTES §3](REPLICATION_NOTES.md#3-the-arithmetic-that-decides-everything).

`.describe()` returns a flat JSON-friendly dict of every setting.

### `PELTWTPipeline`

```python
PELTWTPipeline(cfg=None, **overrides)
```

Any `PipelineConfig` field can be passed directly:
`PELTWTPipeline(model="gru", window=60, stationary=True)`.

**Methods**

```python
.fit(df, price_col="Carbon_Price", exog_cols=None) -> self
.detect_breaks(df, price_col, exog_cols)           # stage 1, standalone
.denoise(price)                                    # stage 2, standalone
.summary() -> str                                  # one-screen report
```

**Attributes after `.fit()`**

| Attribute | Meaning |
|---|---|
| `.breaks_` | `BreakResult` or `dict` |
| `.denoised_` | filtered series |
| `.data_` | the `SupervisedData` |
| `.y_pred_`, `.y_true_` | predictions and realised values, in price units |
| `.metrics_` | metric dict |
| `.history_` | Keras history |
| `.train_time_`, `.n_params_`, `.epochs_run_` | bookkeeping |
| `.name` | label in the paper's notation, e.g. `"PELT-WT-TCN"` |
| `.predictions` | dated `DataFrame`: `actual`, `predicted`, `residual` |

### `run_experiment`

```python
run_experiment(df, price_col="Carbon_Price", models=None, mode="paper",
               include_random_walk=True, base=None, verbose=True,
               **overrides) -> ExperimentResult
```

Fits several specifications on identical windows and returns one comparable
table. `models` defaults to `PAPER_MODELS`; `**overrides` apply to all of them,
so `run_experiment(df, stationary=True)` switches the whole set.

Raises if two specifications produce different numbers of test points, so a
table can never silently compare unlike things.

### `ExperimentResult`

| Attribute / method | Meaning |
|---|---|
| `.table` | metric `DataFrame`, sorted by RMSE |
| `.predictions` | `{name: ndarray}` |
| `.actual`, `.index` | realised values and their dates |
| `.pipelines` | `{name: fitted PELTWTPipeline}` |
| `.training_times` | `{name: seconds}` |
| `.config` | the shared configuration |
| `.residuals()` | `DataFrame` of residuals |
| `.to_frame()` | actual plus every prediction |
| `.best(by="RMSE")` | name of the winner |

---

## 7. Metrics

```python
mae(y_true, y_pred) -> float
mse(y_true, y_pred) -> float
rmse(y_true, y_pred) -> float
mape(y_true, y_pred, eps=1e-8) -> float          # percent
smape(y_true, y_pred) -> float                   # percent
r2(y_true, y_pred) -> float
theil_u(y_true, y_pred) -> float                 # < 1 beats a random walk
naive_random_walk(y_true) -> ndarray             # the no-change benchmark
```

`NaN`s are dropped pairwise. Mismatched lengths raise.

### `evaluate` / `evaluate_many`

```python
evaluate(y_true, y_pred, name="model") -> dict
```

Keys: `Model`, `MAE`, `RMSE`, `MAPE (%)`, `R2`, `Theil U`.

```python
evaluate_many(y_true, predictions, training_times=None) -> pd.DataFrame
```

Table 1 of the paper: one row per model, sorted by RMSE, plus a `Train (s)`
column when times are supplied.

### `diebold_mariano`

```python
diebold_mariano(y_true, pred_a, pred_b, horizon=1, loss="mse",
                harvey_correction=True) -> dict
```

H₀: equal expected loss. A **negative** statistic favours model A. Newey–West
long-run variance with `horizon - 1` lags; the Harvey–Leybourne–Newbold
small-sample correction is on by default and the p-value comes from a
t-distribution with n−1 degrees of freedom.

**Returns** `{"DM", "p_value", "mean_loss_diff", "better", "n"}`.

### `model_confidence_set`

```python
model_confidence_set(y_true, predictions, alpha=0.10, n_boot=1000,
                     block=10, loss="mse", random_state=0) -> pd.DataFrame
```

Hansen–Lunde–Nason MCS by stationary block bootstrap. **Returns** columns
`avg_loss`, `p_MCS`, `in_MCS`, sorted by loss. Never returns an empty set.

---

## 8. Tables

### `results_table`

```python
results_table(table, fmt="plain", decimals=None, bold_best=True,
              caption="...", label="tab:performance") -> str
```

`fmt` is `"plain"`, `"markdown"`, `"latex"` or `"html"`. `bold_best` emphasises
the best cell per column — lowest error, highest R².

### `export_table`

```python
export_table(table, path, caption="", label="", index=True) -> Path
```

The extension of `path` decides the format: `.csv`, `.tex` (booktabs), `.md`,
`.html`, `.xlsx`. Parent directories are created.

### Comparison against the paper

```python
paper_table1(fmt=None) -> DataFrame | str
compare_with_paper(table, metrics=("MAE","RMSE","MAPE (%)","R2")) -> DataFrame
improvement_table(table, baseline=None) -> DataFrame
comparison_table(result, baseline=None, fmt="plain") -> str
```

`compare_with_paper` returns MultiIndex columns
`(metric, {"Paper","Replication","Diff"})`, with `Diff = replication - paper`.
Rows the paper lacks (e.g. `Random walk`) are kept with `NaN` in the paper
columns.

`improvement_table` adds `dMAE (%)` and `dRMSE (%)` against `baseline`, which
defaults to the worst non-random-walk row. This is how the abstract's headline
claim is checked.

### Other tables

```python
summary_statistics(df, decimals=4) -> DataFrame
```
`N`, `Mean`, `Std`, `Min`, `Median`, `Max`, `Skewness`, `Kurtosis`,
`Jarque-Bera`, `JB p`, `ADF`, `ADF p`.

```python
describe_breaks(breaks, index, price=None, events=None,
                tolerance_days=45) -> DataFrame
```
One row per break: date, regime lengths, means and standard deviations either
side, the level shift, and the nearest event within `tolerance_days`. Pass
`events=pw.POLICY_EVENTS`.

```python
dm_matrix(result, loss="mse", horizon=1) -> DataFrame     # pairwise DM p-values
mcs_table(result, alpha=0.10, n_boot=1000, block=10) -> DataFrame
```

---

## 9. Figures

### `set_journal_style`

```python
set_journal_style(font_scale=1.0, serif=True, dpi=150) -> None
```

Call once before plotting. Serif type, thin spines, colour-blind safe palette,
300 dpi on save. `serif=False, font_scale=1.2` suits slides.

### The figure set

Every function returns a `matplotlib` `Figure` and accepts `ax=` for
composition, plus `label=` for a panel tag like `"(a)"`.

| Function | Paper figure |
|---|---|
| `plot_price_history(price, events=None, ax=None, figsize=(7,3), title=None, label="")` | — |
| `plot_correlation_drivers(df, target="Carbon_Price", method="pearson", ...)` | 5 |
| `plot_feature_importance(df, target="Carbon_Price", n_estimators=300, ...)` | 6 |
| `plot_breakpoints(price, breaks, max_lines=60, show_regimes=True, ...)` | 7 |
| `plot_denoising(raw, denoised, zoom=None, ...)` | 8 |
| `plot_wavelet_decomposition(raw, decomposition, figsize=(7.2,5.6))` | 2 |
| `plot_forecast(actual, predicted, index=None, name="Model", show_metrics=True, ...)` | 9–13 |
| `plot_all_forecasts(actual, predictions, index=None, ...)` | 14 |
| `plot_model_comparison(table, metrics=("MAE","RMSE","MAPE (%)","R2"), ...)` | 15 |
| `plot_training_time(times, ...)` | 16 |
| `plot_residuals_over_time(residuals, ...)` | 17 |
| `plot_residual_density(residuals, bandwidth=None, ...)` | 18 |
| `plot_dm_heatmap(dm, ...)` | extra |

`plot_breakpoints` accepts a single `BreakResult` or the dict from
`pelt_multivariate`. It raises `ValueError` if the break result was estimated on
a different number of observations than `price` — a break index is meaningless
on another sample.

### `save_all_figures`

```python
save_all_figures(result, df, outdir="figures", price_col="Carbon_Price",
                 fmt=("png", "pdf"), denoised=None, breaks=None,
                 verbose=True) -> list[Path]
```

Renders and saves the whole set. **Returns** every path written.

```python
files = pw.save_all_figures(res, df, outdir="assets",
                            denoised=wt, breaks=pelt, fmt=("png", "pdf"))
```

---

## See also

- [`STEP_BY_STEP_GUIDE.md`](STEP_BY_STEP_GUIDE.md) — the same API as a tutorial
- [`REPLICATION_NOTES.md`](REPLICATION_NOTES.md) — fidelity to the paper, and its errata
