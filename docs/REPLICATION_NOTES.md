# Replication notes

What this package reproduces, what it cannot, and what it found along the way.

Reference paper:

> Ren, R., Li, J., Li, Y., Huang, S., Shen, J., Li, W., Le, J. and Wang, S.
> (2025) "A Hybrid Deep Learning based Carbon Price Forecasting Framework with
> Structural Breakpoints Detection and Signal Denoising", arXiv:2511.04988v1.

Everything below is checkable by running

```bash
python examples/run_full_replication.py
```

---

## 1. What matches the paper exactly

| Item | Paper | This package |
|---|---|---|
| Sample | EUA spot, 2007-09-10 to 2024-06-04, 6,113 obs | 6,113 obs, identical dates |
| Split | 80 / 20, chronological | `train_size=0.80` → 4,890 / 1,223 |
| Input window | 30 steps, stride 1 | `window=30, stride=1` |
| Horizon | 1 step ahead | `horizon=1` |
| Wavelet | single-level, approximation retained | `WaveletConfig(level=1, denoise_mode="paper")` |
| LSTM / GRU | 2 layers × 128 units, dropout 0.2 | `TrainConfig(n_layers=2, units=128, dropout=0.2)` |
| TCN | 4 residual blocks × 64 channels | `TrainConfig(tcn_blocks=4, tcn_filters=64)` |
| Optimiser | Adam, lr 1e-3, β₁ 0.9, β₂ 0.999 | same |
| Batch / epochs | 64 / max 50 | same |
| Early stopping | val MSE, patience 10, 10 % validation | same |
| Break detectors | Bai–Perron, ICSS, PELT | all three, plus the BP&ICSS union |
| Metrics | MAE, RMSE, MAPE, R², training time | same, plus Theil's U |

The unified input vector is built exactly as the paper's equation for
`z_t = [ỹ_t, u_t, e_t]`: denoised price, exogenous block, one-hot regime label.

---

## 2. Three features cannot be obtained

The paper's Figures 5 and 6 list fifteen inputs. Three have no free public
feed and are therefore absent from `load_paper_dataset`:

| Feature | Why |
|---|---|
| Epex Spot Germany | EPEX SPOT day-ahead DE, commercial licence required |
| Citi Economic Surprise Index (Eurozone) | Bloomberg / Citi only |
| Euribor 1-week | EMMI redistribution licence required |

They are listed in `peltwtcn.UNAVAILABLE_FEATURES` so the gap is explicit
rather than silent. Two substitutes with long free histories are added instead
(`Henry_Hub_Gas`, `DAX`). `Epex_Spot_Germany` is the second-ranked driver in
the paper's Figure 6, so its absence is material.

The paper's most important feature, **`Policy`** (Extra-Trees importance > 0.5,
higher than everything else combined), is never defined anywhere in the text.
`build_policy_features` reconstructs it from the twelve dated events the paper
itself lists in Section 4.1, as a signed exponentially-decaying impulse plus the
EU ETS phase number. This is a documented reconstruction, not the original
series. Because the events are dated from a document written after the sample,
the feature is mildly forward-looking by construction.

---

## 3. The arithmetic that decides everything

The EUA price trends hard across the sample:

```
train window (4,890 obs, 2007-09-10 → 2021-01-28)   EUR  0.01 –  35.14
test  window (1,223 obs, 2021-01-29 → 2024-06-04)   EUR 33.08 –  98.01
```

The test window peaks **2.79× above the training maximum**. Two consequences
follow, and between them they explain why the paper's protocol cannot work out
of sample.

### 3.1 Min–max scaling plus a saturating recurrent net cannot extrapolate

An LSTM or GRU pushes its state through `tanh`. Once the inputs leave the range
seen during training the state saturates and the forecast flattens. Measured on
the real dataset, 25 epochs, everything else at the paper's settings:

| Configuration | RMSE | R² | Prediction range |
|---|---:|---:|---|
| one-hot regimes + exogenous (the paper's spec) | 55.18 | −12.37 | EUR 15.4 – 25.9 |
| exogenous only, no regimes | 24.01 | −1.53 | EUR 32.2 – 69.2 |
| denoised price only | 4.42 | 0.914 | EUR 34.3 – 90.5 |

Every additional block of level-valued features makes it worse, because each one
also trends out of its training range (gas, coal and power all spiked in
2021–22). The true test values run EUR 33 → 98.

Note the direction: the *more* faithful the configuration is to the paper, the
*worse* it does. That is the finding, not a bug.

### 3.2 One-hot regime dummies cannot describe a future regime

PELT with a BIC penalty finds 11 breaks, so 12 regimes. Splitting 80/20:

```
regimes present in training : [0, 1, 2, 3, 4, 5, 6]
regimes present in testing  : [6, 7, 8, 9, 10, 11]
regimes ONLY in testing     : [7, 8, 9, 10, 11]
```

Five of the twelve one-hot columns are **identically zero for every training
row**. The network cannot have learned a weight for them. When the test period
arrives, those columns switch on and the model receives an input pattern it has
never seen; the other regime columns are all zero, so the network falls back to
roughly the training mean. Hence the collapse to a near-constant EUR 15–26
forecast in the first row of the table above.

This is structural, not a tuning problem. Any one-hot encoding of full-sample
break locations has it: the last regime always begins after the last training
observation. It is the central methodological weakness of the paper's `e_t`.

### 3.3 The wavelet filter is two-sided

`pywt.wavedec` / `waverec` over the whole series is a non-causal filter. On this
data, perturbing the price from `t = 3000` onwards changes the denoised value as
early as `t = 2994`: a **look-ahead of 6 observations**. For a one-step-ahead
forecast one observation would already be too many, because the input at `t`
already encodes the price at `t+1`.

`causal_wavelet_denoise` recomputes the transform in a trailing window so the
value at `t` uses only data up to `t`. `tests/test_wavelet.py` verifies both the
leak and its absence.

---

## 4. The three protocols

`run_experiment` therefore offers three settings, and the replication script
runs all of them:

| Protocol | Settings | What it is for |
|---|---|---|
| **A** level | `mode="paper", stationary=False` | The faithful replication. Reproduces the paper's specification literally. Does not work out of sample, for the reasons in §3. |
| **B** stationary | `mode="paper", stationary=True` | Same models and hyper-parameters, applied to the first difference. Levels are rebuilt as *last observed value + predicted change*, so the metrics stay in EUR and remain comparable with Table 1. |
| **C** causal | `mode="causal", stationary=True` | Adds a causal wavelet, a training-only scaler and a raw target. The only protocol whose numbers are an honest out-of-sample claim. |

### Measured results, all three protocols

Real data, the paper's own settings (50 epochs, early stopping, seed 42). RMSE:

| Model | Paper | A level | B stationary | C causal |
|---|---:|---:|---:|---:|
| BP&ICSS-WT-LSTM | 5.3878 | 48.8677 | 1.2847 | 1.7253 |
| PELT-WT-LSTM (uni) | 2.7488 | 43.3046 | 1.3459 | 1.7221 |
| PELT-WT-LSTM (multi) | 2.2967 | 38.2507 | 1.2615 | 1.7215 |
| PELT-WT-GRU | 1.6987 | 55.1775 | 1.2361 | 1.7269 |
| PELT-WT-TCN | 1.5866 | 37.2711 | 2.1012 | 2.2677 |
| *Random walk* | — | *1.2230* | *1.2230* | *1.7217* |

Protocol B in full:

| Model | MAE | RMSE | MAPE (%) | R² | Theil U | Train (s) |
|---|---:|---:|---:|---:|---:|---:|
| *Random walk* | *0.8457* | *1.2230* | *1.1571* | *0.9934* | *1.0000* | *0.0* |
| PELT-WT-GRU | 0.8484 | 1.2361 | 1.1611 | 0.9933 | 1.0107 | 79.1 |
| PELT-WT-LSTM (multi) | 0.8930 | 1.2615 | 1.2292 | 0.9930 | 1.0315 | 64.9 |
| BP&ICSS-WT-LSTM | 0.9210 | 1.2847 | 1.2583 | 0.9928 | 1.0505 | 76.7 |
| PELT-WT-LSTM (uni) | 1.0078 | 1.3459 | 1.4313 | 0.9920 | 1.1005 | 271.1 |
| PELT-WT-TCN | 1.6752 | 2.1012 | 2.2983 | 0.9806 | 1.7180 | 46.8 |

Protocol C in full — the only honest out-of-sample numbers:

| Model | MAE | RMSE | MAPE (%) | R² | Theil U | Train (s) |
|---|---:|---:|---:|---:|---:|---:|
| PELT-WT-LSTM (multi) | 1.0528 | **1.7215** | 1.4475 | 0.9870 | **0.9997** | 93.6 |
| *Random walk* | *1.0416* | *1.7217* | *1.4300* | *0.9870* | *1.0000* | *0.0* |
| PELT-WT-LSTM (uni) | 1.0433 | 1.7221 | 1.4338 | 0.9870 | 1.0001 | 148.9 |
| BP&ICSS-WT-LSTM | 1.0611 | 1.7253 | 1.4584 | 0.9870 | 1.0020 | 181.0 |
| PELT-WT-GRU | 1.0649 | 1.7269 | 1.4637 | 0.9869 | 1.0029 | 50.7 |
| PELT-WT-TCN | 1.6246 | 2.2677 | 2.2641 | 0.9775 | 1.3171 | 187.9 |

Four things to notice.

1. **Protocol A fails completely.** Every model lands between RMSE 37 and 55
   with R² between −5.1 and −12.4, against a random walk at 1.22. That is the
   paper's own specification, run literally.
2. **The paper's ranking does not survive.** It puts PELT-WT-TCN first. Here the
   TCN is **last** under both working protocols, by a wide margin, and it is the
   only model the Model Confidence Set rejects.
3. **Protocol B beats the published figures** for four of the five models — but
   only because the published figures are worse than a random walk.
4. **Under protocol C every model collapses onto the random walk.** The four
   recurrent models sit within 0.3 % of each other and of the benchmark, and
   Theil's U is 1.000 to three decimals. The best model beats "tomorrow equals
   today" by 0.03 %.

The random-walk RMSE differs between protocols (1.2230 vs 1.7217) because
`mode="causal"` forces `target="raw"`: protocols A and B predict the *denoised*
price, which is smoother and therefore easier, while protocol C predicts the
actual price.

---

## 5. The paper never benchmarks against a random walk

On this dataset the no-change forecast "tomorrow equals today" gives

```
MAE 0.846    RMSE 1.223    MAPE 1.157 %    R² 0.9934
```

The paper's best reported model, PELT-WT-TCN, has RMSE 1.5866 — **worse than a
random walk**, and its weakest baseline (5.3878) is worse by a factor of four.
Its R² of 0.9888 is also below the random walk's 0.9934.

A high R² on a near-unit-root series in levels is not evidence of skill; the
level is almost entirely explained by its own last value. This is why
`evaluate_many` adds the random-walk row by default and why `Theil U` is in
every table. An ordinary least squares regression on the same 30-step windows
reaches RMSE 0.8189 (R² 0.9971), better than every deep model in the paper.

Neither the Diebold–Mariano test nor the Model Confidence Set appears in the
paper, so no reported difference between its five models is shown to be
statistically significant. Both are provided here (`dm_matrix`, `mcs_table`),
and under protocol C they settle the question.

Diebold–Mariano against the best model by RMSE, `PELT-WT-LSTM (multi)`:

| Comparison | DM | p | Verdict |
|---|---:|---:|---|
| vs BP&ICSS-WT-LSTM | −1.736 | 0.0827 | not significant |
| vs PELT-WT-LSTM (uni) | −0.324 | 0.7462 | not significant |
| vs PELT-WT-GRU | −1.731 | 0.0837 | not significant |
| vs PELT-WT-TCN | −6.147 | 0.0000 | **significant** |
| **vs Random walk** | **−0.122** | **0.9027** | **not significant** |

Model Confidence Set, α = 0.10:

| Model | avg_loss | p_MCS | in MCS |
|---|---:|---:|:--:|
| PELT-WT-LSTM (multi) | 2.9635 | 1.000 | yes |
| Random walk | 2.9643 | 1.000 | yes |
| PELT-WT-LSTM (uni) | 2.9655 | 1.000 | yes |
| BP&ICSS-WT-LSTM | 2.9767 | 1.000 | yes |
| PELT-WT-GRU | 2.9822 | 0.228 | yes |
| PELT-WT-TCN | 5.1424 | 0.000 | **no** |

**Five of the six models survive, including the random walk.** The only
specification the data rejects is the paper's own preferred one, the TCN — and
it is rejected for being significantly *worse*. Once the look-ahead is removed,
there is no evidence that any of the paper's five architectures forecasts the
EUA price better than assuming tomorrow's price equals today's.

---

## 6. The 22.35 % / 18.63 % claim does not reconcile

The abstract and the third contribution bullet both claim a 22.35 % RMSE and
18.63 % MAE reduction against the baseline. Those figures cannot be recovered
from the paper's own Table 1. Working from the published numbers:

| Comparison | ΔRMSE | ΔMAE |
|---|---:|---:|
| TCN vs BP&ICSS-WT-LSTM (weakest) | 70.55 % | 74.42 % |
| TCN vs PELT-WT-GRU (strongest) | 6.60 % | 10.92 % |
| TCN vs PELT-WT-LSTM (multi) | 30.92 % | 34.84 % |
| TCN vs PELT-WT-LSTM (uni) | 42.28 % | 49.82 % |

The 70.55 % / 74.42 % pair is exactly the comparison against
`BP&ICSS-WT-LSTM`, which the abstract instead attributes to "the original LSTM
without decomposition". No pair of rows yields 22.35 % / 18.63 %.

`improvement_table` computes these explicitly so the claim is always
recomputable. `tests/test_tables.py` pins the 70.55 % and 6.60 % figures.

## 7. Other minor inconsistencies

- **Training times are reported twice and disagree.** Figure 16 gives 26.1 /
  26.4 / 26.2 / 14.3 / 52.5 seconds; the Section 4.3 text gives 24.5 / 24.7 /
  24.9 / 19.6 / 48.7. Both are kept in `PAPER_TRAIN_TIMES`.
- **The wavelet family is never named.** `db4` is the default here, being the
  de-facto standard in the carbon-price literature.
- **The PELT penalty is never given.** BIC is the default here.
- **`SupF` critical values are not reported**, so the Bai–Perron break count is
  not exactly pinned down; `max_breaks` and `trim` are exposed instead.
- **Figure 7's legend lists `Rotterdam_coal` and `DAX`**, which do not appear in
  the Figure 5 or 6 feature lists.

---

## 8. Reproducing these numbers

The replication must be run from a clone. `pip install peltwtcn` gives you the
library but not the bundled price CSVs or the example scripts.

```bash
git clone https://github.com/merwanroudane/peltwtcn.git
cd peltwtcn
pip install -e ".[all]"
python examples/run_full_replication.py
```

Runtime is roughly 25–35 minutes on a CPU: fifteen networks over 6,113
observations. `pw.set_seed(42)` is called first, but exact bit-for-bit equality
across machines is not guaranteed — TensorFlow's CPU kernels are not
deterministic by default.

Note that TensorFlow's CPU backend cannot backpropagate a dilated convolution
at all ("Current CPU implementations do not yet support dilation rates larger
than 1"), so the TCN's dilated causal convolution is written out as an explicit
sum over its taps in `peltwtcn/models.py`. It is verified against a hand-rolled
reference implementation and for strict causality in `tests/test_models.py`.
