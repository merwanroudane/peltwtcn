# peltwtcn

**A faithful Python implementation of the PELT → wavelet → deep-learning carbon
price forecasting framework, with the replication audit the paper needs.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-196%20passing-brightgreen.svg)](tests/)

Implements

> Ren, R., Li, J., Li, Y., Huang, S., Shen, J., Li, W., Le, J. and Wang, S.
> (2025) "A Hybrid Deep Learning based Carbon Price Forecasting Framework with
> Structural Breakpoints Detection and Signal Denoising",
> [arXiv:2511.04988](https://arxiv.org/abs/2511.04988).

**Author of this package:** Dr Merwan Roudane —
<merwanroudane920@gmail.com> —
[github.com/merwanroudane](https://github.com/merwanroudane)

Everything is built on the **real** EU ETS dataset — 6,113 daily EUA
observations from 2007-09-10 to 2024-06-04, matching the paper exactly. Nothing
here is simulated.

---

## Table of contents

1. [What this package does](#1-what-this-package-does)
2. [Installation](#2-installation)
3. [Sixty-second quick start](#3-sixty-second-quick-start)
4. [The three protocols](#4-the-three-protocols)
5. [Results](#5-results)
6. [What the replication found](#6-what-the-replication-found)
7. [The real dataset](#7-the-real-dataset)
8. [Module map](#8-module-map)
9. [Documentation](#9-documentation)
10. [Testing](#10-testing)
11. [Citation](#11-citation)
12. [Author](#12-author)

---

## 1. What this package does

The paper's pipeline, end to end, in one call — plus the diagnostics it omits.

- **Three structural break detectors** — PELT (Killick et al., 2012), ICSS
  (Inclán & Tiao, 1994), Bai–Perron (2003), and the combined BP&ICSS baseline of
  Lin & Zhang (2022).
- **Wavelet denoising** in the paper's two-sided form *and* a leakage-free
  rolling form.
- **LSTM, GRU and TCN** in Keras, with the paper's exact hyper-parameters. The
  TCN's dilated causal convolution is written from scratch so it runs on a CPU
  and is verified for strict causality.
- **Live data loaders** for the EUA price and its drivers, with a bundled cache.
- **Publication-quality tables and figures** — LaTeX/booktabs, Markdown, HTML,
  CSV, Excel; every figure of the paper at 300 dpi.
- **Diebold–Mariano and Model Confidence Set tests**, plus a random-walk
  benchmark and Theil's U. The paper reports none of these.
- **The published Table 1 as a checkable constant**, so any run can be diffed
  against it.

---

## 2. Installation

```bash
git clone https://github.com/merwanroudane/peltwtcn.git
cd peltwtcn
pip install -e ".[all]"
```

Extras: `[deep]` TensorFlow · `[data]` yfinance + openpyxl · `[plots]` seaborn ·
`[test]` pytest. Without TensorFlow everything except the three networks still
works.

Requires Python ≥ 3.9.

---

## 3. Sixty-second quick start

```python
import peltwtcn as pw

# 1. the real EUA carbon price plus 13 exogenous drivers
df = pw.load_paper_dataset()
print(df.shape)                      # (6113, 14)

# 2. the paper's best model, in a configuration that works out of sample
pipe = pw.PELTWTPipeline(model="tcn", stationary=True).fit(df)
print(pipe.summary())

# 3. all five specifications of Table 1
res = pw.run_experiment(df, stationary=True)
print(pw.results_table(res.table, fmt="markdown"))

# 4. how does it compare with the published numbers?
print(pw.compare_with_paper(res.table))

# 5. every figure, at 300 dpi
pw.set_journal_style()
pw.save_all_figures(res, df, outdir="assets")
```

The full replication, all three protocols, tables and figures:

```bash
python examples/run_full_replication.py
```

---

## 4. The three protocols

The paper's specification does not survive an honest chronological split. Rather
than quietly fix it, this package implements all three variants and reports them
side by side.

| Protocol | Call | What it is |
|---|---|---|
| **A** level | `mode="paper", stationary=False` | The faithful replication — the paper's specification, literally. Does not work out of sample. |
| **B** stationary | `mode="paper", stationary=True` | Same models, same hyper-parameters, applied to the first difference. Levels rebuilt as *last value + predicted change*, so metrics stay in EUR. |
| **C** causal | `mode="causal", stationary=True` | Adds a causal wavelet, a training-only scaler and a raw target. The only protocol whose numbers are an honest out-of-sample claim. |

`mode="causal"` forces the leak-free settings, so you cannot leave one in by
accident.

---

## 5. Results

Protocol B on the real dataset, 25 epochs:

| Model | MAE | RMSE | MAPE (%) | R² | Theil U |
|---|---:|---:|---:|---:|---:|
| PELT-WT-GRU | 0.848 | 1.236 | 1.161 | 0.9933 | 1.011 |
| PELT-WT-LSTM (multi) | 0.893 | 1.262 | 1.229 | 0.9930 | 1.032 |
| PELT-WT-TCN | 1.675 | 2.101 | 2.298 | 0.9806 | 1.718 |
| *Random walk* | *0.846* | *1.223* | *1.157* | *0.9934* | *1.000* |

As published in the paper (Table 1, p. 22):

| Model | MAE | RMSE | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| BP&ICSS-WT-LSTM | 4.6345 | 5.3878 | 5.8731 | 0.8712 |
| PELT-WT-LSTM (uni) | 2.3627 | 2.7488 | 3.0582 | 0.9664 |
| PELT-WT-LSTM (multi) | 1.8192 | 2.2967 | 2.3267 | 0.9765 |
| PELT-WT-GRU | 1.3308 | 1.6987 | 1.7401 | 0.9872 |
| **PELT-WT-TCN** | **1.1855** | **1.5866** | **1.6451** | **0.9888** |

Available in code as `pw.PAPER_TABLE1`.

Under protocol A — the paper's specification applied literally — RMSE is **34 to
55** with a **negative R²**, and the forecast collapses to a flat band around
EUR 20 against a truth of EUR 33–98. That is not a bug in this implementation;
[§6](#6-what-the-replication-found) explains exactly why, and the reasoning is
reproducible.

Full numbers land in `results/` after running the replication script.

---

## 6. What the replication found

Four findings, all reproducible. Details and derivations in
[`docs/REPLICATION_NOTES.md`](docs/REPLICATION_NOTES.md).

### The models cannot extrapolate past the training range

The EUA price trends hard: the 80 % training window tops out at **EUR 35.14**
while the test window reaches **EUR 98.01** — 2.79× higher. An LSTM or GRU
squashes its state through `tanh`, so once inputs leave the range seen in
training the state saturates and the forecast flattens. Measured, 25 epochs:

| Configuration | RMSE | R² | Prediction range |
|---|---:|---:|---|
| one-hot regimes + exogenous (the paper's spec) | 55.18 | −12.37 | EUR 15.4 – 25.9 |
| exogenous only | 24.01 | −1.53 | EUR 32.2 – 69.2 |
| denoised price only | 4.42 | 0.914 | EUR 34.3 – 90.5 |

The *more* faithful the configuration, the *worse* it does. `stationary=True`
fixes it by modelling the change instead of the level.

### One-hot regime dummies cannot describe a future regime

PELT finds 11 breaks, so 12 regimes. After an 80/20 split, regimes **7–11 occur
only in the test window**: five of the twelve one-hot columns are identically
zero for every training row, so no network can learn a weight for them. This is
structural to the paper's `e_t` encoding — the last regime always begins after
the last training observation.

### The wavelet filter looks ahead

`wavedec`/`waverec` over the whole series is two-sided. Perturbing the price from
`t = 3000` onwards changes the denoised value as early as `t = 2994` — a
**six-observation look-ahead**. For a one-step-ahead forecast, one would already
be too many. Use `denoise_mode="causal"`.

### The paper never benchmarks against a random walk

On this data the no-change forecast gives **RMSE 1.223, R² 0.9934** — better
than every model in the paper's Table 1, including the winner (1.5866). Plain
OLS on the same 30-step windows reaches **RMSE 0.8189**. A high R² on a
near-unit-root series in levels is not evidence of skill.

Neither Diebold–Mariano nor a Model Confidence Set appears in the paper, so none
of its reported differences is shown to be significant. Both are provided here.

### Two errata

- The abstract's headline **"22.35 % RMSE / 18.63 % MAE"** improvement cannot be
  recovered from the paper's own Table 1. The true figures are 70.55 % / 74.42 %
  against `BP&ICSS-WT-LSTM` and 6.60 % / 10.92 % against `PELT-WT-GRU`; no pair
  of rows gives 22.35 % / 18.63 %. Check it with `pw.improvement_table`.
- **Training times are reported twice and disagree** (Figure 16 vs the Section
  4.3 text). Both are kept in `pw.PAPER_TRAIN_TIMES`.

---

## 7. The real dataset

`load_paper_dataset()` returns 6,113 rows × 14 columns, exactly the paper's
sample size, cached in `data/`.

| Block | Columns |
|---|---|
| Target | `Carbon_Price` (EUA spot, EUR/tCO₂) |
| Energy | `Europe_Coal`, `TTF_Natural_Gas`, `Henry_Hub_Gas`, `Brent_Crude` |
| Equity / FX | `Euro_Stoxx_50`, `DAX`, `VIX`, `EURUSD` |
| Uncertainty | `EPU_US`, `EPU_UK`, `GPR`, `EU_10Y_Yield` |
| Policy | `Policy` (reconstructed — see below) |

**Three features named in the paper have no free feed** and are listed in
`pw.UNAVAILABLE_FEATURES`: Epex Spot Germany (commercial licence), the Citi
Economic Surprise Index (Bloomberg/Citi), and 1-week Euribor (EMMI licence).
Epex is the paper's second-ranked driver, so its absence is material.

**The `Policy` feature is never defined in the paper**, despite being its most
important predictor (Extra-Trees importance > 0.5). `build_policy_features`
reconstructs it transparently from the twelve dated events the paper itself
lists in Section 4.1, as a signed exponentially-decaying impulse plus the EU ETS
phase number. It is documented as a reconstruction, not the original series.

Your own data works too — any dated frame with a target column:

```python
pipe = pw.PELTWTPipeline(stationary=True).fit(my_df, price_col="Price")
```

---

## 8. Module map

| Module | Contents |
|---|---|
| `peltwtcn/datasets.py` | live loaders, caching, the policy reconstruction, `POLICY_EVENTS` |
| `peltwtcn/breaks.py` | PELT, ICSS, Bai–Perron, BP&ICSS, regime encoding |
| `peltwtcn/wavelet.py` | decomposition, denoising, causal denoising, thresholds |
| `peltwtcn/features.py` | `z_t` design matrix, sliding windows, scalers |
| `peltwtcn/models.py` | LSTM, GRU, TCN, weight-normalised dilated causal conv |
| `peltwtcn/pipeline.py` | `PipelineConfig`, `PELTWTPipeline`, `run_experiment` |
| `peltwtcn/metrics.py` | MAE/RMSE/MAPE/R²/Theil U, Diebold–Mariano, MCS |
| `peltwtcn/tables.py` | Table 1, descriptives, break inventory, paper comparison |
| `peltwtcn/plots.py` | every figure of the paper, journal styling |

---

## 9. Documentation

| Document | What it covers |
|---|---|
| [`docs/STEP_BY_STEP_GUIDE.md`](docs/STEP_BY_STEP_GUIDE.md) | **How to write the code, stage by stage** — 15 steps, every block runnable |
| [`docs/SYNTAX.md`](docs/SYNTAX.md) | Complete API reference: every function, argument and return value |
| [`docs/REPLICATION_NOTES.md`](docs/REPLICATION_NOTES.md) | What matches the paper, what cannot, and the errata |
| [`examples/run_full_replication.py`](examples/run_full_replication.py) | All three protocols end to end on real data |

---

## 10. Testing

```bash
pytest                          # everything
pytest -m "not slow"            # skip the network fits
pytest -m "not network"         # skip the live downloads
```

196 fast tests. Highlights of what is actually verified, rather than merely
asserted:

- the TCN's dilated convolution is **strictly causal** — perturbing `x_t` for
  `t ≥ 20` leaves every output before `t = 20` bit-identical — and matches a
  hand-rolled reference convolution to float32 precision;
- the wavelet reconstruction identity `A + ΣD = f`;
- the paper's filter **does** leak and the causal one **does not**;
- PELT recovers known break locations to within 5 observations, and ICSS finds a
  variance shift while ignoring a mean shift;
- Theil's U of a random walk is exactly 1;
- the bundled dataset is 6,113 rows spanning the paper's exact dates;
- the 70.55 % and 6.60 % improvement figures, pinned as regression tests.

---

## 11. Citation

If you use this software, please cite both the implementation and the original
paper. A `CITATION.cff` is included.

```bibtex
@software{roudane2026peltwtcn,
  author  = {Roudane, Merwan},
  title   = {peltwtcn: Hybrid deep-learning carbon price forecasting with
             PELT structural-break detection and wavelet denoising},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/merwanroudane/peltwtcn},
  license = {MIT}
}

@article{ren2025hybrid,
  author  = {Ren, Runsheng and Li, Jing and Li, Yanxiu and Huang, Shixun and
             Shen, Jun and Li, Wanqing and Le, John and Wang, Sheng},
  title   = {A Hybrid Deep Learning based Carbon Price Forecasting Framework
             with Structural Breakpoints Detection and Signal Denoising},
  journal = {arXiv preprint arXiv:2511.04988},
  year    = {2025},
  url     = {https://arxiv.org/abs/2511.04988}
}
```

### Key methodological references

| Reference | DOI |
|---|---|
| Bai & Perron (2003), *J. Applied Econometrics* | [10.1002/jae.659](https://doi.org/10.1002/jae.659) |
| Inclán & Tiao (1994), *JASA* | [10.2307/2290916](https://doi.org/10.2307/2290916) |
| Killick, Fearnhead & Eckley (2012), *JRSS-B* | [10.1111/j.1467-9868.2011.01004.x](https://doi.org/10.1111/j.1467-9868.2011.01004.x) |
| Mallat (1989), *IEEE TPAMI* | [10.1109/34.192463](https://doi.org/10.1109/34.192463) |
| Lin & Zhang (2022), *Process Safety and Env. Protection* | [10.1016/j.psep.2022.08.011](https://doi.org/10.1016/j.psep.2022.08.011) |
| Liu et al. (2012), *Energy Economics* | [10.1016/j.eneco.2011.09.002](https://doi.org/10.1016/j.eneco.2011.09.002) |
| Geurts, Ernst & Wehenkel (2006), *Machine Learning* | [10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1) |
| Cho et al. (2014), *EMNLP* | [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179) |
| Bai, Kolter & Koltun (2018) | [arXiv:1803.01271](https://arxiv.org/abs/1803.01271) |
| Hochreiter & Schmidhuber (1997), *Neural Computation* | [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) |

---

## 12. Author

**Dr Merwan Roudane**

- Email: <merwanroudane920@gmail.com>
- GitHub: [@merwanroudane](https://github.com/merwanroudane)

Other packages: `QuantileOnQuantile`, `mqqr`, `qqkrls`, `mqqcause` (CRAN).

---

## Licence

MIT — see [LICENSE](LICENSE).

The implementation is original work. The methodology is due to Ren et al.
(2025); please cite them. Redistribution of the underlying price series is
subject to each provider's own terms.
