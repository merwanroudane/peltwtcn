"""
Sequence forecasting models: LSTM, GRU and TCN.

Implements Section 3.3 of Ren et al. (2025) in TensorFlow/Keras with the
training configuration of Section 4.2:

    Adam, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, batch = 64, <= 50 epochs,
    10% validation split, early stopping on val MSE with patience 10,
    LSTM/GRU = 2 hidden layers x 128 units with dropout 0.2 between them,
    TCN = 4 residual blocks x 64 channels.

A correctness note on the paper's equations
-------------------------------------------
Equation (21) of the paper writes the LSTM cell-state update as

    C_t = sigma(f_t * C_{t-1} + i_t * C~_t)

with an outer sigmoid.  That is wrong: squashing the cell state into (0, 1)
destroys the constant error carousel, which is the whole point of the
architecture.  The correct update, used by Keras and by this package, is

    C_t = f_t * C_{t-1} + i_t * C~_t.

Author: Dr Merwan Roudane
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

__all__ = [
    "TrainConfig",
    "set_seed",
    "build_lstm",
    "build_gru",
    "build_tcn",
    "build_model",
    "fit_model",
    "WeightNormConv1D",
    "residual_block",
]

ModelName = Literal["lstm", "gru", "tcn"]


# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Every training hyper-parameter reported in Section 4.2 of the paper."""

    units: int = 128                 # LSTM / GRU hidden units per layer
    n_layers: int = 2                # LSTM / GRU stacked layers
    dropout: float = 0.2
    tcn_filters: int = 64            # channels per residual block
    tcn_blocks: int = 4              # number of residual blocks
    tcn_kernel_size: int = 3
    learning_rate: float = 1e-3
    beta_1: float = 0.9
    beta_2: float = 0.999
    batch_size: int = 64
    epochs: int = 50
    validation_split: float = 0.10
    patience: int = 10
    loss: str = "mse"
    seed: int = 42
    verbose: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def set_seed(seed: int = 42) -> None:
    """Make a run reproducible across Python, NumPy and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover - tensorflow optional at import time
        pass


def _keras():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for the deep learning models. "
            "Install it with:  pip install tensorflow"
        ) from exc
    return tf, tf.keras


# ---------------------------------------------------------------------------
# TCN building blocks
# ---------------------------------------------------------------------------
def _weight_norm_conv1d_class():
    tf, keras = _keras()

    class WeightNormConv1D(keras.layers.Layer):
        """Dilated causal Conv1D with weight normalisation.

        Implements the "WeightNorm" box of Figure 4 of the paper: the kernel is
        reparameterised as ``W = g * V / ||V||``, which stabilises the
        activation scale and speeds up convergence (Salimans & Kingma, 2016).
        Keras has no built-in weight norm since ``tensorflow_addons`` was
        retired, so it is implemented here directly.

        The convolution itself is written out as an explicit sum over the
        ``kernel_size`` dilated taps

        .. math::  y_t = b + \\sum_{j=0}^{k-1} W_j \\, x_{t-(k-1-j)d}

        rather than delegating to ``tf.nn.conv1d(..., dilations=d)``.  Two
        reasons:

        1. **It runs on a CPU.**  TensorFlow's CPU kernel for the gradient of
           a dilated convolution is not implemented ("Current CPU
           implementations do not yet support dilation rates larger than 1"),
           so a ``dilations>1`` Conv1D raises during ``fit`` on every machine
           without a GPU -- which is most machines running this package.
        2. **It is exactly causal.**  Left-padding by :math:`(k-1)d` and
           reading the taps at lags :math:`0, d, \\ldots, (k-1)d` uses
           :math:`x_t` and no later value.  A ``padding="SAME"`` convolution
           that is shifted afterwards instead loses the current step, which
           silently shortens the receptive field.

        With ``k = 3`` the loop unrolls to three slices, so the cost is the
        same order as the fused kernel.
        """

        def __init__(self, filters, kernel_size, dilation_rate=1,
                     activation=None, **kwargs):
            super().__init__(**kwargs)
            self.filters = int(filters)
            self.kernel_size = int(kernel_size)
            self.dilation_rate = int(dilation_rate)
            self.activation = keras.activations.get(activation)

        def build(self, input_shape):
            in_ch = int(input_shape[-1])
            self.v = self.add_weight(
                name="v", shape=(self.kernel_size, in_ch, self.filters),
                initializer=keras.initializers.HeNormal(), trainable=True)
            self.g = self.add_weight(
                name="g", shape=(self.filters,),
                initializer=keras.initializers.Ones(), trainable=True)
            self.b = self.add_weight(
                name="b", shape=(self.filters,),
                initializer=keras.initializers.Zeros(), trainable=True)
            super().build(input_shape)

        def call(self, x):
            # W = g * V / ||V||, the norm taken over (kernel, in_channels)
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.v), axis=[0, 1]) + 1e-12)
            kernel = self.g / norm * self.v          # (k, in_ch, filters)

            k, d = self.kernel_size, self.dilation_rate
            pad = (k - 1) * d

            length = x.shape[1]
            if length is None:                       # dynamic time axis
                length = tf.shape(x)[1]

            xp = tf.pad(x, [[0, 0], [pad, 0], [0, 0]]) if pad else x

            # taps[j] is x shifted by (k-1-j)*d, so taps[k-1] is x itself and
            # the kernel's last slice multiplies the current time step.
            taps = [xp[:, j * d: j * d + length, :] for j in range(k)]
            stacked = tf.stack(taps, axis=2)         # (batch, time, k, in_ch)

            y = tf.einsum("btkc,kcf->btf", stacked, kernel)
            y = tf.nn.bias_add(y, self.b)
            return self.activation(y) if self.activation is not None else y

        def compute_output_shape(self, input_shape):
            return tuple(input_shape[:-1]) + (self.filters,)

        def get_config(self):
            cfg = super().get_config()
            cfg.update(filters=self.filters, kernel_size=self.kernel_size,
                       dilation_rate=self.dilation_rate,
                       activation=keras.activations.serialize(self.activation))
            return cfg

    return WeightNormConv1D


class _Lazy:
    """Expose ``WeightNormConv1D`` without importing TensorFlow at module load."""

    def __call__(self, *a, **k):
        return _weight_norm_conv1d_class()(*a, **k)

    def __getattr__(self, item):
        return getattr(_weight_norm_conv1d_class(), item)


WeightNormConv1D = _Lazy()


def residual_block(x, filters: int, kernel_size: int, dilation: int,
                   dropout: float, name: str):
    """One TCN residual block, exactly as drawn in Figure 4 of the paper.

    Dilated causal conv -> weight norm -> crop -> ReLU -> dropout, twice, then
    a 1x1 convolution on the shortcut to match channel counts, then addition.
    """
    tf, keras = _keras()
    WN = _weight_norm_conv1d_class()

    y = x
    for i in (1, 2):
        y = WN(filters, kernel_size, dilation_rate=dilation,
               name=f"{name}_wnconv{i}")(y)
        y = keras.layers.Activation("relu", name=f"{name}_relu{i}")(y)
        y = keras.layers.Dropout(dropout, name=f"{name}_drop{i}")(y)

    if int(x.shape[-1]) != filters:
        x = keras.layers.Conv1D(filters, 1, padding="same",
                                name=f"{name}_shortcut")(x)
    out = keras.layers.Add(name=f"{name}_add")([x, y])
    return keras.layers.Activation("relu", name=f"{name}_out")(out)


# ---------------------------------------------------------------------------
# model builders
# ---------------------------------------------------------------------------
def build_lstm(input_shape: tuple[int, int], cfg: TrainConfig | None = None):
    """Stacked LSTM: 2 layers x 128 units, dropout 0.2, linear output.

    Examples
    --------
    >>> m = build_lstm((30, 5))                     # doctest: +SKIP
    >>> m.output_shape                              # doctest: +SKIP
    (None, 1)
    """
    cfg = cfg or TrainConfig()
    _, keras = _keras()
    m = keras.Sequential(name="LSTM")
    m.add(keras.layers.Input(shape=input_shape))
    for i in range(cfg.n_layers):
        m.add(keras.layers.LSTM(cfg.units, return_sequences=(i < cfg.n_layers - 1),
                                name=f"lstm_{i + 1}"))
        m.add(keras.layers.Dropout(cfg.dropout, name=f"drop_{i + 1}"))
    m.add(keras.layers.Dense(1, name="output"))
    return m


def build_gru(input_shape: tuple[int, int], cfg: TrainConfig | None = None):
    """Stacked GRU: 2 layers x 128 units, dropout 0.2, linear output."""
    cfg = cfg or TrainConfig()
    _, keras = _keras()
    m = keras.Sequential(name="GRU")
    m.add(keras.layers.Input(shape=input_shape))
    for i in range(cfg.n_layers):
        m.add(keras.layers.GRU(cfg.units, return_sequences=(i < cfg.n_layers - 1),
                               name=f"gru_{i + 1}"))
        m.add(keras.layers.Dropout(cfg.dropout, name=f"drop_{i + 1}"))
    m.add(keras.layers.Dense(1, name="output"))
    return m


def build_tcn(input_shape: tuple[int, int], cfg: TrainConfig | None = None,
              dilations: Sequence[int] | None = None):
    """Temporal Convolutional Network: 4 residual blocks x 64 channels.

    Dilations default to ``[1, 2, 4, 8]``, giving a receptive field of
    ``1 + 2*(k-1)*sum(dilations)`` = 61 steps for k = 3, comfortably covering
    the 30-step input window.
    """
    cfg = cfg or TrainConfig()
    _, keras = _keras()
    dil = list(dilations) if dilations is not None else [2 ** i for i in range(cfg.tcn_blocks)]

    inp = keras.layers.Input(shape=input_shape, name="input")
    x = inp
    for i, d in enumerate(dil, start=1):
        x = residual_block(x, cfg.tcn_filters, cfg.tcn_kernel_size, d,
                           cfg.dropout, name=f"block{i}")
    x = keras.layers.Lambda(lambda t: t[:, -1, :], name="last_step")(x)
    out = keras.layers.Dense(1, name="output")(x)
    return keras.Model(inp, out, name="TCN")


def build_model(kind: ModelName, input_shape: tuple[int, int],
                cfg: TrainConfig | None = None):
    """Dispatch to :func:`build_lstm`, :func:`build_gru` or :func:`build_tcn`."""
    builders = {"lstm": build_lstm, "gru": build_gru, "tcn": build_tcn}
    key = str(kind).lower()
    if key not in builders:
        raise ValueError(f"unknown model {kind!r}; choose from {sorted(builders)}")
    return builders[key](input_shape, cfg)


# ---------------------------------------------------------------------------
def fit_model(
    kind: ModelName,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: TrainConfig | None = None,
):
    """Compile, train and predict in one call.

    Returns
    -------
    dict
        ``{"model", "history", "y_pred", "train_time", "epochs_run",
        "n_params"}`` where ``y_pred`` is the (scaled) test prediction.
    """
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    _, keras = _keras()

    model = build_model(kind, X_train.shape[1:], cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate,
                                        beta_1=cfg.beta_1, beta_2=cfg.beta_2),
        loss=cfg.loss,
        metrics=["mae"],
    )
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.patience,
        restore_best_weights=True, verbose=cfg.verbose,
    )

    t0 = time.perf_counter()
    hist = model.fit(
        X_train, y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        callbacks=[stopper],
        shuffle=False,          # never shuffle a time series across the split
        verbose=cfg.verbose,
    )
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test, batch_size=cfg.batch_size,
                           verbose=cfg.verbose).ravel()
    return {
        "model": model,
        "history": hist.history,
        "y_pred": y_pred,
        "train_time": float(train_time),
        "epochs_run": int(len(hist.history.get("loss", []))),
        "n_params": int(model.count_params()),
    }
