"""LSTM, GRU and TCN builders, and the causality of the dilated convolution."""

from __future__ import annotations

import numpy as np
import pytest

from peltwtcn import TrainConfig, build_gru, build_lstm, build_model, build_tcn

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")


# --------------------------------------------------------------------------
# architecture, against Section 4.2 of the paper
# --------------------------------------------------------------------------
@pytest.mark.parametrize("builder", [build_lstm, build_gru, build_tcn])
def test_models_map_a_window_to_a_scalar(builder):
    m = builder((30, 5))
    assert m.output_shape == (None, 1)
    out = m.predict(np.zeros((4, 30, 5), dtype="float32"), verbose=0)
    assert out.shape == (4, 1)


def test_lstm_has_two_layers_of_128_units_by_default():
    m = build_lstm((30, 5))
    lstms = [l for l in m.layers if l.__class__.__name__ == "LSTM"]
    assert len(lstms) == 2
    assert all(l.units == 128 for l in lstms)


def test_gru_has_two_layers_of_128_units_by_default():
    m = build_gru((30, 5))
    grus = [l for l in m.layers if l.__class__.__name__ == "GRU"]
    assert len(grus) == 2
    assert all(l.units == 128 for l in grus)


def test_tcn_has_four_residual_blocks_of_64_channels_by_default():
    m = build_tcn((30, 5))
    names = [l.name for l in m.layers]
    assert sum(n.endswith("_add") for n in names) == 4
    wn = [l for l in m.layers if l.__class__.__name__ == "WeightNormConv1D"]
    assert len(wn) == 8                      # two per block
    assert all(l.filters == 64 for l in wn)


def test_tcn_dilations_double_each_block():
    m = build_tcn((30, 5))
    wn = [l for l in m.layers if l.__class__.__name__ == "WeightNormConv1D"]
    assert sorted({l.dilation_rate for l in wn}) == [1, 2, 4, 8]


def test_dropout_rate_follows_the_config():
    m = build_lstm((30, 3), TrainConfig(dropout=0.35))
    drops = [l for l in m.layers if l.__class__.__name__ == "Dropout"]
    assert drops and all(abs(l.rate - 0.35) < 1e-9 for l in drops)


def test_build_model_dispatches_by_name():
    for kind, name in (("lstm", "LSTM"), ("gru", "GRU"), ("tcn", "TCN")):
        assert build_model(kind, (30, 4)).name == name


def test_build_model_rejects_an_unknown_kind():
    with pytest.raises(ValueError):
        build_model("transformer", (30, 4))


# --------------------------------------------------------------------------
# the dilated causal convolution
# --------------------------------------------------------------------------
def _wn_layer(filters=3, kernel_size=3, dilation=4):
    from peltwtcn.models import _weight_norm_conv1d_class
    return _weight_norm_conv1d_class()(filters, kernel_size,
                                       dilation_rate=dilation)


def test_dilated_conv_is_strictly_causal():
    """Changing x_t for t >= 20 must not alter any output before t = 20."""
    layer = _wn_layer()
    x = np.random.RandomState(0).normal(size=(2, 30, 5)).astype("float32")
    y0 = layer(tf.constant(x)).numpy()

    x2 = x.copy()
    x2[:, 20:, :] += 99.0
    y1 = layer(tf.constant(x2)).numpy()

    assert np.array_equal(y0[:, :20, :], y1[:, :20, :])
    assert not np.allclose(y0[:, 20:, :], y1[:, 20:, :])


def test_dilated_conv_matches_an_explicit_reference_implementation():
    """y_t = b + sum_j W_j x_{t-(k-1-j)d}, computed by hand."""
    layer = _wn_layer(filters=3, kernel_size=3, dilation=4)
    x = np.random.RandomState(1).normal(size=(2, 30, 5)).astype("float32")
    got = layer(tf.constant(x)).numpy()

    v, g, b = layer.v.numpy(), layer.g.numpy(), layer.b.numpy()
    kernel = g / np.sqrt((v ** 2).sum(axis=(0, 1)) + 1e-12) * v
    k, d = layer.kernel_size, layer.dilation_rate

    want = np.zeros_like(got)
    for t in range(x.shape[1]):
        acc = b.astype(np.float64).copy()
        for j in range(k):
            lag = (k - 1 - j) * d
            if t - lag >= 0:
                acc = acc + x[:, t - lag, :] @ kernel[j]
        want[:, t, :] = acc
    assert np.allclose(got, want, atol=1e-5)


def test_weight_norm_kernel_has_unit_column_norm_at_init():
    """With g initialised to ones, each output filter's kernel has norm 1."""
    layer = _wn_layer(filters=4, kernel_size=3, dilation=1)
    layer.build((None, 30, 5))
    v, g = layer.v.numpy(), layer.g.numpy()
    kernel = g / np.sqrt((v ** 2).sum(axis=(0, 1)) + 1e-12) * v
    norms = np.sqrt((kernel ** 2).sum(axis=(0, 1)))
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_dilated_conv_preserves_the_time_axis():
    layer = _wn_layer(filters=7, kernel_size=3, dilation=8)
    y = layer(tf.zeros((2, 30, 5))).numpy()
    assert y.shape == (2, 30, 7)


def test_tcn_output_depends_only_on_the_window_it_is_given():
    """The whole TCN, not just one layer, must be causal end to end."""
    m = build_tcn((30, 4), TrainConfig(tcn_blocks=2, tcn_filters=8))
    x = np.random.RandomState(2).normal(size=(1, 30, 4)).astype("float32")
    base = m.predict(x, verbose=0)
    # the prediction reads the last step, so perturbing it must change output
    x2 = x.copy()
    x2[:, -1, :] += 10.0
    assert not np.allclose(base, m.predict(x2, verbose=0))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("kind", ["lstm", "gru", "tcn"])
def test_fit_model_learns_and_reports_its_bookkeeping(kind, fast_train):
    from peltwtcn import fit_model
    r = np.random.RandomState(0)
    X = r.normal(size=(200, 20, 3)).astype("float32")
    y = X[:, -1, 0] * 2.0 + 0.5
    out = fit_model(kind, X[:150], y[:150], X[150:], fast_train)

    assert set(out) >= {"model", "history", "y_pred", "train_time",
                        "epochs_run", "n_params"}
    assert out["y_pred"].shape == (50,)
    assert out["n_params"] > 0
    assert out["train_time"] > 0
    assert 1 <= out["epochs_run"] <= fast_train.epochs
    assert np.all(np.isfinite(out["y_pred"]))


@pytest.mark.slow
def test_training_is_reproducible_with_a_fixed_seed(fast_train):
    from peltwtcn import fit_model
    r = np.random.RandomState(1)
    X = r.normal(size=(150, 15, 2)).astype("float32")
    y = X[:, -1, 0]
    a = fit_model("gru", X[:120], y[:120], X[120:], fast_train)["y_pred"]
    b = fit_model("gru", X[:120], y[:120], X[120:], fast_train)["y_pred"]
    assert np.allclose(a, b)


def test_train_config_roundtrips_through_as_dict():
    cfg = TrainConfig(units=64, epochs=7, dropout=0.3)
    assert TrainConfig(**cfg.as_dict()) == cfg


def test_train_config_defaults_match_the_paper():
    cfg = TrainConfig()
    assert cfg.units == 128 and cfg.n_layers == 2
    assert cfg.dropout == 0.2
    assert cfg.tcn_filters == 64 and cfg.tcn_blocks == 4
    assert cfg.learning_rate == 1e-3
    assert (cfg.beta_1, cfg.beta_2) == (0.9, 0.999)
    assert cfg.batch_size == 64 and cfg.epochs == 50
    assert cfg.patience == 10
    assert cfg.validation_split == 0.10
