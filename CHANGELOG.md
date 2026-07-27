# Changelog

All notable changes to `peltwtcn`. Follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[semantic versioning](https://semver.org/).

## [1.0.1] — 2026-07-27

Documentation only. No change to any function, argument or result.

### Fixed

- **The documentation was invisible on the PyPI page.** PyPI renders the README
  as the project description, but the README linked to the guides with relative
  paths (`docs/STEP_BY_STEP_GUIDE.md`), which PyPI resolves against `pypi.org`
  rather than GitHub. All eight documentation links 404'd and the three figures
  did not render at all. Every link and image in the README is now an absolute
  URL, so the guide, the tutorial, the syntax reference and the replication
  notes are reachable from the PyPI page.
- The step-by-step guide and the runnable tutorial are now signposted at the
  top of the README instead of only in section 9.

### Added

- Direct links to the guide, the tutorial, the syntax reference, the replication
  notes and this changelog in `[project.urls]`, so they appear in PyPI's
  "Project links" sidebar.
- This changelog.

## [1.0.0] — 2026-07-27

First release. A faithful implementation of

> Ren, R., Li, J., Li, Y., Huang, S., Shen, J., Li, W., Le, J. and Wang, S.
> (2025) "A Hybrid Deep Learning based Carbon Price Forecasting Framework with
> Structural Breakpoints Detection and Signal Denoising", arXiv:2511.04988.

built on the real EU ETS dataset — 6,113 daily EUA observations, 2007-09-10 to
2024-06-04, matching the paper's sample exactly.

### Added

- **Break detectors** — PELT, ICSS, Bai–Perron, and the combined BP&ICSS
  baseline of Lin & Zhang (2022), with one-hot and ordinal regime encoding.
- **Wavelet layer** — multilevel decomposition and denoising in the paper's
  two-sided form, plus a leakage-free rolling form and a passthrough ablation.
- **Models** — LSTM, GRU and TCN at the paper's exact hyper-parameters. The
  TCN's dilated causal convolution is written as an explicit sum over its taps
  rather than `tf.nn.conv1d(dilations=d)`, because TensorFlow's CPU backend
  cannot backpropagate a dilated convolution; it is verified against a
  hand-rolled reference and for strict causality.
- **Pipeline** — `PipelineConfig`, `PELTWTPipeline`, `run_experiment`, with
  three protocols: `paper` (the level model as specified), `stationary=True`
  (first difference, levels rebuilt), and `mode="causal"` (every look-ahead
  removed).
- **Metrics** — MAE, RMSE, MAPE, sMAPE, R², Theil's U, a random-walk benchmark,
  Diebold–Mariano and the Model Confidence Set. The paper reports none of the
  last four.
- **Tables and figures** — every figure of the paper at 300 dpi; tables to
  LaTeX/booktabs, Markdown, HTML, CSV and Excel. `PAPER_TABLE1` holds the
  published numbers so any run can be diffed against them.
- **Data loaders** for the EUA price and its drivers, cached per user.
- 216 tests, a 13-step runnable tutorial, a 15-stage written guide, a full
  syntax reference, and replication notes.

### Findings

Recorded here because they are properties of the method, not of this code. All
are reproducible with `examples/run_full_replication.py`.

- The paper's specification does not survive its own chronological split. The
  training window tops out at EUR 35.14 while the test window reaches EUR 98.01,
  2.79× higher, and a `tanh` state saturates beyond it. Applied literally, all
  five models land at RMSE 37–55 with R² between −5.1 and −12.4.
- One-hot regime dummies cannot describe a future regime. PELT finds 11 breaks;
  after an 80/20 split, regimes 7–11 occur only in the test window, so five of
  twelve columns are identically zero in training.
- The wavelet filter is two-sided, with a measured six-observation look-ahead.
- With every leak removed, all four recurrent models land within 0.3 % of a
  random walk. Diebold–Mariano against the random walk gives p = 0.9027, and the
  Model Confidence Set at α = 0.10 retains five of six models *including* the
  random walk. The only rejected specification is the paper's own preferred
  PELT-WT-TCN — rejected for being significantly worse.
- The abstract's 22.35 % / 18.63 % improvement claim cannot be recovered from
  the paper's own Table 1. The true figures are 70.55 % / 74.42 % against
  BP&ICSS-WT-LSTM and 6.60 % / 10.92 % against PELT-WT-GRU.

[1.0.1]: https://github.com/merwanroudane/peltwtcn/releases/tag/v1.0.1
[1.0.0]: https://github.com/merwanroudane/peltwtcn/releases/tag/v1.0.0
