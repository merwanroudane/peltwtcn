# Tutorial output

The verbatim output of

```bash
python examples/tutorial_step_by_step.py
```

Run it yourself and compare. This is a real run on the real EU ETS dataset
(6,113 daily EUA observations, 2007-09-10 to 2024-06-04), at tutorial speed:
8 epochs instead of the paper's 50, and a coarser PELT grid. So the numbers here
are *deliberately* not the paper-settings numbers — for those see
[`REPLICATION_NOTES.md`](REPLICATION_NOTES.md) or run
`examples/run_full_replication.py`.

TensorFlow's console warnings have been stripped; nothing else is edited.

The narrative walkthrough of these thirteen steps, with the reasoning behind
each choice, is [`STEP_BY_STEP_GUIDE.md`](STEP_BY_STEP_GUIDE.md).

---

## Contents

1. [Set up: seed everything and pick the journal style](#step-1-set-up-seed-everything-and-pick-the-journal-style)
2. [Load the real data](#step-2-load-the-real-data)
3. [Look at the train/test split BEFORE modelling anything](#step-3-look-at-the-train-test-split-before-modelling-anything)
4. [Describe the data, and test the paper's claims about it](#step-4-describe-the-data-and-test-the-paper-s-claims-about-it)
5. [Detect the structural breaks (Section 3.1 of the paper)](#step-5-detect-the-structural-breaks-section-3-1-of-the-paper)
6. [The catch with one-hot regimes -- check it yourself](#step-6-the-catch-with-one-hot-regimes-check-it-yourself)
7. [Denoise with a wavelet (Section 3.2)](#step-7-denoise-with-a-wavelet-section-3-2)
8. [Assemble the model input z_t = [y_t, u_t, e_t]](#step-8-assemble-the-model-input-z-t-y-t-u-t-e-t)
9. [Cut it into sliding windows](#step-9-cut-it-into-sliding-windows)
10. [Fit a model -- and watch the level version fail](#step-10-fit-a-model-and-watch-the-level-version-fail)
11. [Score it honestly: always include a random walk](#step-11-score-it-honestly-always-include-a-random-walk)
12. [Run all five specifications of Table 1, then compare with the paper](#step-12-run-all-five-specifications-of-table-1-then-compare-with-the-paper)
13. [Test whether the differences are significant, then draw everything](#step-13-test-whether-the-differences-are-significant-then-draw-everything)

---

## Step 1: Set up: seed everything and pick the journal style

```text

peltwtcn version : 1.0.0
public names     : 94
tutorial settings: 8 epochs, PELT jump=5
writing to       : C:\Users\HP\Documents\xtpmg\peltcnwt\tutorial_output
```

## Step 2: Load the real data

```text

shape      : (6113, 14)      <- the paper reports 6,113 samples
period     : 2007-09-10 -> 2024-06-04
columns    : Carbon_Price, Europe_Coal, TTF_Natural_Gas, Henry_Hub_Gas, Brent_Crude, Euro_Stoxx_50, DAX, VIX, EURUSD, EPU_US, EPU_UK, GPR, EU_10Y_Yield, Policy
missing    : 0
load time  : 0.10s

First three rows, first five columns:
            Carbon_Price  Europe_Coal  TTF_Natural_Gas  Henry_Hub_Gas  Brent_Crude
Date
2007-09-10          0.09        122.5            18.09          5.891    75.480003
2007-09-11          0.08        122.5            18.09          5.934    76.379997
2007-09-12          0.09        122.5            18.09          6.438    77.680000

Named in the paper but unavailable:
  Epex_Spot_Germany      EPEX SPOT day-ahead DE - commercial licence required
  Citi_CESI_Eurozone     Citi Economic Surprise Index - Bloomberg/Citi only
  Euribor_1W             1-week Euribor - EMMI redistribution licence required
```

## Step 3: Look at the train/test split BEFORE modelling anything

```text

split          : 4890 train / 1223 test  (80/20, chronological)
train range    : EUR   0.01 -  35.14   (2007-09-10 -> 2021-01-28)
test  range    : EUR  33.08 -  98.01   (2021-01-29 -> 2024-06-04)

test max / train max = 2.79x
The test window peaks far above anything seen in training.  A tanh-based
LSTM or GRU saturates past its training range, so a LEVEL model cannot
extrapolate there.  Remember this at step 10.
```

## Step 4: Describe the data, and test the paper's claims about it

```text

Skewness, kurtosis and a unit-root test for the first four columns:
                    N      Mean      Std     Min      Max  Skewness  Kurtosis   ADF p
Variable
Carbon_Price     6113   24.9257  26.0249   0.010   98.010    1.4448    3.7275  0.9167
Europe_Coal      6113  105.4336  58.0420  38.600  438.350    2.6729   11.8204  0.2051
TTF_Natural_Gas  6113   27.8546  31.7508   3.510  339.196    4.2034   24.1721  0.0520
Henry_Hub_Gas    6113    3.8402   1.9047   1.482   13.577    2.0480    8.0082  0.0427

The paper says the price shows 'positive skewness and high kurtosis'.
  skewness    = +1.4448   -> positive, as claimed
  ADF p-value = 0.9167   -> non-stationary

wrote tutorial_output\table_descriptives.csv and .tex
```

## Step 5: Detect the structural breaks (Section 3.1 of the paper)

```text

PELT      : 11 breaks, 12 regimes (3.5s)
ICSS      :  7 breaks (variance)
Bai-Perron:  4 breaks (mean)
BP&ICSS   :  5 breaks = 4 BP + 1 ICSS

Breaks, matched to the policy events of Section 4.1:
 Break       Date  Mean before  Mean after      Shift                            Nearest event
     1 2007-12-19     0.067900   23.676484  23.608584 End of EU ETS Phase I: surplus allowa...
     2 2008-10-24    23.676484   14.013273  -9.663211 Global financial crisis: industrial o...
     3 2011-10-29    14.013273    6.338349  -7.674924
     4 2018-03-06     6.338349   15.995636   9.657287
     5 2018-08-18    15.995636   25.304058   9.308422
     6 2020-12-25    25.304058   39.119760  13.815702
     7 2021-04-29    39.119760   56.867805  17.748045
     8 2021-11-20    56.867805   82.970966  26.103161
     9 2022-09-06    82.970966   71.482000 -11.488966
    10 2022-11-25    71.482000   86.051167  14.569167
    11 2023-11-20    86.051167   66.655353 -19.395814 Mild winter, weak power demand, ample...
```

## Step 6: The catch with one-hot regimes -- check it yourself

```text

regimes in training : [0, 1, 2, 3, 4, 5, 6]
regimes in testing  : [6, 7, 8, 9, 10, 11]
ONLY in testing     : [7, 8, 9, 10, 11]

5 of 12 one-hot columns are identically zero for
every training row, so no network can learn a weight for them.  When the
test period arrives those columns switch on and the model has never seen
the pattern.  This is structural to the paper's e_t encoding: the last
regime always begins after the last training observation.
```

## Step 7: Denoise with a wavelet (Section 3.2)

```text

paper  (two-sided): RMSE vs raw 0.3763, corr 0.999895
causal (rolling)  : RMSE vs raw 0.1553, corr 0.999982

Perturbing the price from t=3000 changes the denoised series from t=2994.
That is a look-ahead of 6 observations.  For a one-step-ahead
forecast, one observation would already be too many.

reconstruction identity |A + sum(D) - f|max = 7.11e-14   (should be ~0)
```

## Step 8: Assemble the model input z_t = [y_t, u_t, e_t]

```text

denoised price : 1 column
exogenous u_t  : 13 columns
regimes e_t    : 12 columns
z_t total      : 26 columns, 6113 rows

column order is always price, then exogenous, then regimes:
  ['Carbon_Price_WT', 'Europe_Coal', 'TTF_Natural_Gas'] ... ['regime_r10', 'regime_r11']

univariate variant: (6113, 1)
```

## Step 9: Cut it into sliding windows

```text

X_train (4860, 30, 26)   y_train (4860,)
X_test  (1223, 30, 26)   y_test  (1223,)
window=30, n_features=26

test dates: 2021-01-29 -> 2024-06-04

scale_on='all'  : X_test in [0.00, 1.00]
scale_on='train': X_test in [-0.14, 13.04]   <- far outside [0,1]
A saturating network cannot do anything sensible with the second one.
```

## Step 10: Fit a model -- and watch the level version fail

```text

Fitting PELT-WT-GRU on the level (the paper's specification) ...
  RMSE  54.7385   R2 -12.1606
  predictions span EUR 15.8 - 26.4
  truth       spans EUR 33.1 - 98.1
  -> the forecast is stuck in a flat band.  This is the saturation
     predicted at step 3, not a bug.

Fitting the same model with stationary=True ...
  RMSE   1.2481   R2   0.9932
  predictions span EUR 33.5 - 98.6
  -> now it tracks.  RMSE improved by a factor of 44.

Full report for the working model:
==============================================================
 PELT-WT-GRU   [mode=paper, target=denoised]
==============================================================
 breaks detected : 11
 wavelet         : db4, level 1, paper
 input shape     : (30, 26) (26 features)
 train / test    : 4860 / 1223
 parameters      : 159,105   epochs: 8
 training time   : 30.5 s
--------------------------------------------------------------
 MAE       0.8649     RMSE      1.2481
 MAPE      1.1957 %   R2        0.9932
 Theil U   1.0204  (<1 beats a random walk)
==============================================================
```

## Step 11: Score it honestly: always include a random walk

```text

model vs the no-change benchmark 'tomorrow equals today':
  PELT-WT-GRU : {'Model': 'GRU', 'MAE': 0.8649031098269903, 'RMSE': 1.2480657563879998, 'MAPE (%)': 1.195702425183536, 'R2': 0.9931582907661451, 'Theil U': 1.0204044230314002}
  Random walk : {'Model': 'RW', 'MAE': 0.8457187541956575, 'RMSE': 1.2229747535425977, 'MAPE (%)': 1.1570860865810337, 'R2': 0.9934306157980112, 'Theil U': 1.0}

Theil's U = 1.0204
  below 1 means you beat the random walk; at or above 1 means you did not.
On daily carbon prices in LEVELS a high R2 proves very little, because
the level is almost entirely explained by its own last value.  The paper
never runs this comparison; its best reported RMSE is 1.5866.
```

## Step 12: Run all five specifications of Table 1, then compare with the paper

```text

Fitting the five models of Table 1 (stationary protocol) ...
  fitting BP&ICSS-WT-LSTM ...
  fitting PELT-WT-LSTM (uni) ...
  fitting PELT-WT-LSTM (multi) ...
  fitting PELT-WT-GRU ...
  fitting PELT-WT-TCN ...

done in 135s

                          MAE     RMSE MAPE (%)       R2  Theil U Train (s)
Model
Random walk           0.8457*  1.2230*  1.1571*  0.9934*  1.0000*      0.0*
PELT-WT-GRU            0.8649   1.2481   1.1957   0.9932   1.0204      27.5
PELT-WT-LSTM (multi)   0.8915   1.2586   1.2277   0.9930   1.0292      29.6
BP&ICSS-WT-LSTM        1.0234   1.3727   1.3990   0.9917   1.1224      28.0
PELT-WT-LSTM (uni)     1.4453   1.7557   2.1819   0.9865   1.4351      27.8
PELT-WT-TCN            1.9308   2.3472   2.6069   0.9758   1.9193      16.6

As published in the paper (Table 1, p. 22):
                         MAE    RMSE  MAPE (%)      R2
Model
BP&ICSS-WT-LSTM       4.6345  5.3878    5.8731  0.8712
PELT-WT-LSTM (uni)    2.3627  2.7488    3.0582  0.9664
PELT-WT-LSTM (multi)  1.8192  2.2967    2.3267  0.9765
PELT-WT-GRU           1.3308  1.6987    1.7401  0.9872
PELT-WT-TCN           1.1855  1.5866    1.6451  0.9888

Side by side (Diff = replication - paper):
                       RMSE
                      Paper Replication   Diff
Model
BP&ICSS-WT-LSTM       5.388       1.373 -4.015
PELT-WT-LSTM (uni)    2.749       1.756 -0.993
PELT-WT-LSTM (multi)  2.297       1.259 -1.038
PELT-WT-GRU           1.699       1.248 -0.451
PELT-WT-TCN           1.587       2.347  0.761
Random walk             NaN       1.223    NaN

Improvement over the BP&ICSS-WT-LSTM baseline:
                        MAE   RMSE  dMAE (%)  dRMSE (%)
Model
Random walk           0.846  1.223    17.364     10.907
PELT-WT-GRU           0.865  1.248    15.490      9.079
PELT-WT-LSTM (multi)  0.891  1.259    12.892      8.309
BP&ICSS-WT-LSTM       1.023  1.373     0.000      0.000
PELT-WT-LSTM (uni)    1.445  1.756   -41.223    -27.901
PELT-WT-TCN           1.931  2.347   -88.658    -70.993

Against the paper's OWN Table 1 the reduction is 70.55% RMSE /
74.42% MAE, not the 22.35% / 18.63% the abstract claims:
                      dMAE (%)  dRMSE (%)
Model
BP&ICSS-WT-LSTM           0.00       0.00
PELT-WT-LSTM (uni)       49.02      48.98
PELT-WT-LSTM (multi)     60.75      57.37
PELT-WT-GRU              71.28      68.47
PELT-WT-TCN              74.42      70.55

wrote table1.csv / .tex / .md to tutorial_output
```

## Step 13: Test whether the differences are significant, then draw everything

```text

Diebold-Mariano p-values (H0: equal predictive accuracy):
                      BP&ICSS-WT-LSTM  PELT-WT-LSTM (uni)  PELT-WT-LSTM (multi)  PELT-WT-GRU  PELT-WT-TCN  Random walk
BP&ICSS-WT-LSTM                   NaN                 0.0                 0.000        0.000          0.0        0.000
PELT-WT-LSTM (uni)                0.0                 NaN                 0.000        0.000          0.0        0.000
PELT-WT-LSTM (multi)              0.0                 0.0                   NaN        0.435          0.0        0.000
PELT-WT-GRU                       0.0                 0.0                 0.435          NaN          0.0        0.006
PELT-WT-TCN                       0.0                 0.0                 0.000        0.000          NaN        0.000
Random walk                       0.0                 0.0                 0.000        0.006          0.0          NaN

best by RMSE: Random walk
  Random walk vs BP&ICSS-WT-LSTM        DM= -8.616 p=0.0000 significant
  Random walk vs PELT-WT-LSTM (uni)     DM=-18.908 p=0.0000 significant
  Random walk vs PELT-WT-LSTM (multi)   DM= -4.717 p=0.0000 significant
  Random walk vs PELT-WT-GRU            DM= -2.747 p=0.0061 significant
  Random walk vs PELT-WT-TCN            DM=-20.641 p=0.0000 significant

Model Confidence Set (alpha = 0.10):
                      avg_loss   p_MCS  in_MCS
Random walk             1.4957  1.0000    True
PELT-WT-GRU             1.5577  0.0333   False
PELT-WT-LSTM (multi)    1.5842  0.0000   False
BP&ICSS-WT-LSTM         1.8843  0.0000   False
PELT-WT-LSTM (uni)      3.0825  0.0000   False
PELT-WT-TCN             5.5094  0.0000   False

17 figures written to tutorial_output\figures
  fig01_price_history.png
  fig05_correlation_drivers.png
  fig06_feature_importance.png
  fig07_breakpoints.png
  fig08_denoising.png
  fig09_forecast_BPandICSS-WT-LSTM.png
  fig10_forecast_PELT-WT-LSTM_uni.png
  fig11_forecast_PELT-WT-LSTM_multi.png
  fig12_forecast_PELT-WT-GRU.png
  fig13_forecast_PELT-WT-TCN.png
  fig14_forecast_Random_walk.png
  fig15_all_forecasts.png
  fig16_model_comparison.png
  fig17_training_time.png
  fig18_residuals_time.png
  fig19_residual_density.png
  fig20_dm_pvalues.png

========================================================================
FINISHED.  Everything is in C:\Users\HP\Documents\xtpmg\peltcnwt\tutorial_output
========================================================================
Next: examples/run_full_replication.py runs all three protocols with
the paper's own 50-epoch settings.  See docs/STEP_BY_STEP_GUIDE.md for
the narrative version of these thirteen steps, and docs/SYNTAX.md for the
full API reference.
```
