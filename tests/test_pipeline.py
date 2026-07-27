"""The end-to-end PELT -> WT -> deep model pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from peltwtcn import (PAPER_MODELS, ExperimentResult, PELTWTPipeline,
                      PipelineConfig, WaveletConfig, run_experiment)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def test_default_config_is_the_papers_configuration():
    cfg = PipelineConfig()
    assert cfg.mode == "paper"
    assert cfg.train_size == 0.80
    assert cfg.window == 30 and cfg.horizon == 1 and cfg.stride == 1
    assert cfg.detector == "pelt"
    assert cfg.target == "denoised"
    assert cfg.wavelet.level == 1
    assert cfg.use_regimes is True


def test_causal_mode_forces_a_causal_filter_and_a_raw_target():
    cfg = PipelineConfig(mode="causal")
    assert cfg.wavelet.denoise_mode == "causal"
    assert cfg.target == "raw"
    assert cfg.scale_on == "train"


def test_paper_mode_scales_on_the_whole_sample():
    """Leaky, but it is what makes the published numbers attainable."""
    assert PipelineConfig(mode="paper").scale_on == "all"


def test_causal_mode_overrides_an_explicit_scale_on():
    """You cannot leave a leak in by accident in causal mode."""
    assert PipelineConfig(mode="causal", scale_on="all").scale_on == "train"


def test_stationary_is_off_by_default():
    """The paper models the level, so the faithful default must too."""
    assert PipelineConfig().stationary is False


def test_stationary_appears_in_describe():
    assert PipelineConfig(stationary=True).describe()["stationary"] is True


def test_causal_mode_keeps_other_wavelet_settings():
    cfg = PipelineConfig(mode="causal",
                         wavelet=WaveletConfig(wavelet="sym8", level=2))
    assert cfg.wavelet.wavelet == "sym8" and cfg.wavelet.level == 2
    assert cfg.wavelet.denoise_mode == "causal"


def test_overrides_reach_the_config():
    pipe = PELTWTPipeline(model="gru", window=15, train_size=0.7)
    assert pipe.cfg.model == "gru"
    assert pipe.cfg.window == 15
    assert pipe.cfg.train_size == 0.7


def test_describe_is_json_friendly():
    d = PipelineConfig().describe()
    assert d["model"] == "tcn" and d["detector"] == "pelt"
    assert any(k.startswith("wavelet_") for k in d)
    assert any(k.startswith("train_") for k in d)


def test_paper_models_are_the_five_rows_of_table_1():
    assert len(PAPER_MODELS) == 5
    assert set(PAPER_MODELS) == {
        "BP&ICSS-WT-LSTM", "PELT-WT-LSTM (uni)", "PELT-WT-LSTM (multi)",
        "PELT-WT-GRU", "PELT-WT-TCN"}
    assert PAPER_MODELS["PELT-WT-LSTM (uni)"]["multivariate"] is False
    assert PAPER_MODELS["PELT-WT-LSTM (multi)"]["multivariate"] is True


# --------------------------------------------------------------------------
# stages in isolation
# --------------------------------------------------------------------------
def test_detect_breaks_returns_a_result_per_column_when_asked(toy_panel,
                                                              fast_pelt):
    pipe = PELTWTPipeline(pelt=fast_pelt, per_column_breaks=True)
    out = pipe.detect_breaks(toy_panel, "Carbon_Price",
                             [c for c in toy_panel.columns if c != "Carbon_Price"])
    assert isinstance(out, dict)
    assert set(out) == set(toy_panel.columns)


def test_detector_none_yields_a_single_regime(toy_panel):
    pipe = PELTWTPipeline(detector="none")
    out = pipe.detect_breaks(toy_panel, "Carbon_Price", [])
    assert out.n_breaks == 0
    assert out.n == len(toy_panel)


def test_denoise_preserves_length(toy_panel):
    pipe = PELTWTPipeline()
    out = pipe.denoise(toy_panel["Carbon_Price"].to_numpy(float))
    assert out.shape == (len(toy_panel),)


def test_fit_rejects_a_missing_price_column(toy_panel):
    with pytest.raises(KeyError):
        PELTWTPipeline().fit(toy_panel, price_col="Not_A_Column")


# --------------------------------------------------------------------------
# full fit
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_fit_populates_every_documented_attribute(toy_panel, fast_train,
                                                  fast_pelt):
    pytest.importorskip("tensorflow")
    pipe = PELTWTPipeline(model="gru", train=fast_train,
                          pelt=fast_pelt).fit(toy_panel)

    assert pipe.breaks_ is not None
    assert pipe.denoised_.shape == (len(toy_panel),)
    assert pipe.data_ is not None
    assert pipe.y_pred_.shape == pipe.y_true_.shape
    assert pipe.n_params_ > 0
    assert pipe.train_time_ > 0
    assert set(pipe.metrics_) >= {"MAE", "RMSE", "R2"}
    assert np.all(np.isfinite(pipe.y_pred_))


@pytest.mark.slow
def test_predictions_frame_is_dated_and_consistent(toy_panel, fast_train,
                                                   fast_pelt):
    pytest.importorskip("tensorflow")
    pipe = PELTWTPipeline(model="gru", train=fast_train,
                          pelt=fast_pelt).fit(toy_panel)
    pred = pipe.predictions
    assert list(pred.columns) == ["actual", "predicted", "residual"]
    assert isinstance(pred.index, pd.DatetimeIndex)
    assert np.allclose(pred["residual"], pred["actual"] - pred["predicted"])


@pytest.mark.slow
def test_summary_is_printable(toy_panel, fast_train, fast_pelt):
    pytest.importorskip("tensorflow")
    pipe = PELTWTPipeline(model="gru", train=fast_train,
                          pelt=fast_pelt).fit(toy_panel)
    text = pipe.summary()
    assert "RMSE" in text and pipe.name in text


def test_summary_before_fit_raises():
    with pytest.raises(RuntimeError):
        PELTWTPipeline().summary()


def test_predictions_before_fit_raises():
    with pytest.raises(RuntimeError):
        _ = PELTWTPipeline().predictions


# --------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs,expected", [
    (dict(detector="pelt", model="tcn"), "PELT-WT-TCN"),
    (dict(detector="pelt", model="gru"), "PELT-WT-GRU"),
    (dict(detector="bp_icss", model="lstm", multivariate=True),
     "BP&ICSS-WT-LSTM (multi)"),
    (dict(detector="pelt", model="lstm", multivariate=False),
     "PELT-WT-LSTM (uni)"),
    (dict(detector="none", model="tcn"), "NoBreak-WT-TCN"),
])
def test_model_names_follow_the_papers_notation(kwargs, expected):
    assert PELTWTPipeline(**kwargs).name == expected


def test_name_reports_raw_when_denoising_is_off():
    pipe = PELTWTPipeline(model="tcn", wavelet=WaveletConfig(denoise_mode="none"))
    assert pipe.name == "PELT-RAW-TCN"


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_run_experiment_compares_models_on_the_same_test_window(
        toy_panel, fast_train, fast_pelt):
    pytest.importorskip("tensorflow")
    models = {"gru": dict(detector="pelt", model="gru"),
              "tcn": dict(detector="pelt", model="tcn")}
    res = run_experiment(toy_panel, models=models, verbose=False,
                         train=fast_train, pelt=fast_pelt)

    assert isinstance(res, ExperimentResult)
    assert set(res.predictions) == {"gru", "tcn", "Random walk"}
    lengths = {len(v) for v in res.predictions.values()}
    assert len(lengths) == 1                       # like for like
    assert len(res.actual) == lengths.pop()
    assert res.table["RMSE"].is_monotonic_increasing
    assert res.best("RMSE") == res.table["RMSE"].idxmin()


@pytest.mark.slow
def test_run_experiment_can_omit_the_random_walk(toy_panel, fast_train,
                                                 fast_pelt):
    pytest.importorskip("tensorflow")
    res = run_experiment(toy_panel, models={"gru": dict(model="gru")},
                         include_random_walk=False, verbose=False,
                         train=fast_train, pelt=fast_pelt)
    assert "Random walk" not in res.predictions


@pytest.mark.slow
def test_stationary_mode_still_reports_metrics_in_price_units(toy_panel,
                                                             fast_train,
                                                             fast_pelt):
    """Levels are rebuilt, so the numbers stay comparable with Table 1."""
    pytest.importorskip("tensorflow")
    pipe = PELTWTPipeline(model="gru", stationary=True, train=fast_train,
                          pelt=fast_pelt).fit(toy_panel)
    price = toy_panel["Carbon_Price"]
    # y_true_ must be actual price levels, not differences
    assert pipe.y_true_.min() >= price.min() - 1e-6
    assert pipe.y_true_.max() <= price.max() + 1e-6
    # and the predictions must live on the same scale
    assert abs(pipe.y_pred_.mean() - pipe.y_true_.mean()) < 0.5 * price.std()


@pytest.mark.slow
def test_stationary_mode_reconstructs_from_the_last_observed_level(toy_panel,
                                                                  fast_train,
                                                                  fast_pelt):
    """y_pred = previous level + predicted change, so it tracks the level."""
    pytest.importorskip("tensorflow")
    pipe = PELTWTPipeline(model="gru", stationary=True, train=fast_train,
                          pelt=fast_pelt).fit(toy_panel)
    # a level-tracking forecast correlates strongly with the truth
    assert np.corrcoef(pipe.y_true_, pipe.y_pred_)[0, 1] > 0.9


@pytest.mark.slow
def test_stationary_mode_beats_level_mode_on_a_trending_series(fast_train,
                                                              fast_pelt):
    """The headline finding, as a regression test."""
    pytest.importorskip("tensorflow")
    # a series whose test window sits well above anything seen in training
    n = 700
    r = np.random.RandomState(9)
    trend = np.concatenate([np.full(int(n * 0.8), 0.02), np.full(n - int(n * 0.8), 0.35)])
    price = 20.0 + np.cumsum(trend + r.normal(0, 0.25, n))
    df = pd.DataFrame({"Carbon_Price": price},
                      index=pd.bdate_range("2015-01-01", periods=n))

    level = PELTWTPipeline(model="gru", stationary=False, multivariate=False,
                           train=fast_train, pelt=fast_pelt).fit(df)
    diff = PELTWTPipeline(model="gru", stationary=True, multivariate=False,
                          train=fast_train, pelt=fast_pelt).fit(df)
    assert diff.metrics_["RMSE"] < level.metrics_["RMSE"]


@pytest.mark.slow
def test_residuals_and_to_frame_line_up(toy_panel, fast_train, fast_pelt):
    pytest.importorskip("tensorflow")
    res = run_experiment(toy_panel, models={"gru": dict(model="gru")},
                         verbose=False, train=fast_train, pelt=fast_pelt)
    resid = res.residuals()
    frame = res.to_frame()
    assert list(resid.index) == list(res.index)
    assert "Actual" in frame.columns
    assert np.allclose(resid["gru"], res.actual - res.predictions["gru"])
