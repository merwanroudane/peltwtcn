"""
peltwtcn
========

A faithful, fully documented Python implementation of

    Ren, R., Li, J., Li, Y., Huang, S., Shen, J., Li, W., Le, J. and Wang, S.
    (2025) "A Hybrid Deep Learning based Carbon Price Forecasting Framework
    with Structural Breakpoints Detection and Signal Denoising",
    arXiv:2511.04988v1.

The package implements the complete PELT -> Wavelet -> Deep-model pipeline for
EU ETS carbon price forecasting, with

* three structural break detectors (PELT, ICSS, Bai-Perron, and the combined
  BP&ICSS baseline of Lin & Zhang, 2022),
* wavelet denoising in both the paper's non-causal form and a leakage-free
  rolling-window form,
* LSTM, GRU and TCN sequence models in Keras with the paper's exact
  hyper-parameters,
* live loaders for the real EUA carbon price and its exogenous drivers,
* publication-quality tables and figures,
* Diebold-Mariano and Model Confidence Set tests, which the paper omits.

Quick start
-----------
>>> from peltwtcn import load_paper_dataset, run_experiment, results_table
>>> df = load_paper_dataset()                       # doctest: +SKIP
>>> res = run_experiment(df, mode="paper")          # doctest: +SKIP
>>> print(results_table(res.table, fmt="markdown")) # doctest: +SKIP

Author
------
Dr Merwan Roudane <merwanroudane920@gmail.com>
https://github.com/merwanroudane
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/merwanroudane/peltwtcn"

from .breaks import (BreakResult, PeltConfig, bai_perron_breakpoints,
                     bic_penalty, bp_icss_breakpoints, clear_cache,
                     icss_breakpoints, one_hot_regimes, pelt_breakpoints,
                     pelt_multivariate, regimes_from_breakpoints)
from .datasets import (CARBON_ID, PAPER_END, PAPER_START, POLICY_EVENTS,
                       UNAVAILABLE_FEATURES, YAHOO_FEATURES,
                       build_policy_features, cache_dir, fetch_ecb, fetch_epu,
                       fetch_gpr, fetch_investing, fetch_yahoo, load_eua,
                       load_exogenous, load_paper_dataset)
from .features import (SupervisedData, WindowScaler, build_design_matrix,
                       build_regime_matrix, make_windows,
                       train_test_split_index)
from .metrics import (diebold_mariano, evaluate, evaluate_many, mae,
                      model_confidence_set, mape, mse, naive_random_walk, r2,
                      rmse, smape, theil_u)
from .models import (TrainConfig, build_gru, build_lstm, build_model,
                     build_tcn, fit_model, set_seed)
from .pipeline import (PAPER_MODELS, ExperimentResult, PELTWTPipeline,
                       PipelineConfig, run_experiment)
from .plots import (plot_all_forecasts, plot_breakpoints,
                    plot_correlation_drivers, plot_denoising, plot_dm_heatmap,
                    plot_feature_importance, plot_forecast,
                    plot_model_comparison, plot_price_history,
                    plot_residual_density, plot_residuals_over_time,
                    plot_training_time, plot_wavelet_decomposition,
                    save_all_figures, set_journal_style)
from .tables import (PAPER_TABLE1, PAPER_TRAIN_TIMES, comparison_table,
                     compare_with_paper, describe_breaks, dm_matrix,
                     export_table, improvement_table, mcs_table, paper_table1,
                     results_table, summary_statistics)
from .wavelet import (WaveletConfig, causal_wavelet_denoise, max_useful_level,
                      universal_threshold, wavelet_decompose, wavelet_denoise)

__all__ = [
    "__version__", "__author__", "__email__", "__url__",
    # breaks
    "BreakResult", "PeltConfig", "pelt_breakpoints", "pelt_multivariate",
    "icss_breakpoints", "bai_perron_breakpoints", "bp_icss_breakpoints",
    "regimes_from_breakpoints", "one_hot_regimes", "bic_penalty",
    "clear_cache",
    # wavelet
    "WaveletConfig", "wavelet_denoise", "causal_wavelet_denoise",
    "wavelet_decompose", "universal_threshold", "max_useful_level",
    # data
    "load_paper_dataset", "load_eua", "load_exogenous", "build_policy_features",
    "fetch_investing", "fetch_yahoo", "fetch_epu", "fetch_gpr", "fetch_ecb",
    "cache_dir",
    "CARBON_ID", "PAPER_START", "PAPER_END", "YAHOO_FEATURES",
    "UNAVAILABLE_FEATURES", "POLICY_EVENTS",
    # features
    "build_design_matrix", "build_regime_matrix", "make_windows",
    "train_test_split_index", "WindowScaler", "SupervisedData",
    # models
    "TrainConfig", "build_lstm", "build_gru", "build_tcn", "build_model",
    "fit_model", "set_seed",
    # pipeline
    "PipelineConfig", "PELTWTPipeline", "run_experiment", "ExperimentResult",
    "PAPER_MODELS",
    # metrics
    "mae", "mse", "rmse", "mape", "smape", "r2", "theil_u", "evaluate",
    "evaluate_many", "diebold_mariano", "model_confidence_set",
    "naive_random_walk",
    # tables
    "results_table", "comparison_table", "summary_statistics", "describe_breaks",
    "dm_matrix", "mcs_table", "improvement_table", "export_table",
    "PAPER_TABLE1", "PAPER_TRAIN_TIMES", "paper_table1", "compare_with_paper",
    # plots
    "set_journal_style", "plot_price_history", "plot_correlation_drivers",
    "plot_feature_importance", "plot_breakpoints", "plot_denoising",
    "plot_wavelet_decomposition", "plot_forecast", "plot_all_forecasts",
    "plot_model_comparison", "plot_training_time", "plot_residuals_over_time",
    "plot_residual_density", "plot_dm_heatmap", "save_all_figures",
]
