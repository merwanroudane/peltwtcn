"""
End-to-end pipeline: breaks -> wavelet -> windows -> deep model -> metrics.

This module wires Sections 3.1, 3.2 and 3.3 of Ren et al. (2025) into a single
scikit-learn-style object, :class:`PELTWTPipeline`, and provides
:func:`run_experiment` to reproduce the five rows of Table 1 in one call.

The ``mode`` switch
-------------------
``mode="paper"``
    Reproduces the published protocol byte for byte: the wavelet transform and
    the break detector both see the whole sample, and the model is trained to
    predict the *denoised* price.  Use it to replicate, never to claim
    out-of-sample accuracy.

``mode="causal"``
    The same architecture with every look-ahead removed: rolling-window
    wavelet denoising, break detection fitted on the training window only with
    test-period regimes assigned recursively, scaler fitted on training rows
    only, and the *raw* price as the target.  Use it to find out what the
    method is actually worth.

Author: Dr Merwan Roudane
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .breaks import (BreakResult, PeltConfig, bp_icss_breakpoints,
                     pelt_breakpoints, pelt_multivariate,
                     regimes_from_breakpoints)
from .features import (SupervisedData, build_design_matrix, build_regime_matrix,
                       make_windows, train_test_split_index)
from .metrics import evaluate_many, naive_random_walk
from .models import TrainConfig, fit_model
from .wavelet import WaveletConfig, wavelet_denoise

__all__ = [
    "PipelineConfig",
    "PELTWTPipeline",
    "ExperimentResult",
    "run_experiment",
    "PAPER_MODELS",
]

Mode = Literal["paper", "causal"]
Detector = Literal["pelt", "bp_icss", "none"]


# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """All settings of one PELT-WT-<model> run.

    The defaults are the paper's, so ``PipelineConfig()`` is the published
    configuration.
    """

    # --- experiment mode ---------------------------------------------------
    mode: Mode = "paper"

    # --- data ---------------------------------------------------------------
    train_size: float = 0.80
    window: int = 30
    horizon: int = 1
    stride: int = 1
    scale: Literal["minmax", "standard", "none"] = "minmax"

    # Where the scaler is fitted.  This single switch decides whether the
    # paper's Table 1 is reachable at all, so it is worth spelling out.
    #
    # The EUA price trends hard: the 80% training window tops out at EUR 35.14
    # while the test window reaches EUR 98.01, 2.79 times higher.  Fitting a
    # min-max scaler on the training rows alone therefore pushes every test
    # input far outside [0, 1], where the saturating gates of an LSTM or GRU
    # cannot extrapolate; predictions pin near the training maximum and the
    # RMSE lands around 34 instead of 1.6, with a negative R^2.
    #
    # "all" fits the scaler on the whole sample before splitting.  That is a
    # genuine information leak - the training set is normalised using the test
    # period's minimum and maximum - but it is what makes the published numbers
    # attainable, and it is standard practice in this literature.  It is the
    # default so that mode="paper" reproduces the paper.  mode="causal" forces
    # "train" (see __post_init__).
    scale_on: Literal["train", "all"] = "all"

    # --- structural breaks --------------------------------------------------
    detector: Detector = "pelt"
    pelt: PeltConfig = field(default_factory=PeltConfig)
    per_column_breaks: bool = False       # PELT on every feature, as the paper says
    max_breaks_bp: int = 5
    trim_bp: float = 0.15
    use_regimes: bool = True

    # --- wavelet ------------------------------------------------------------
    wavelet: WaveletConfig = field(default_factory=WaveletConfig)

    # --- model / target -----------------------------------------------------
    model: Literal["lstm", "gru", "tcn"] = "tcn"
    multivariate: bool = True
    target: Literal["raw", "denoised"] = "denoised"
    train: TrainConfig = field(default_factory=TrainConfig)

    # Model the first difference of the series instead of its level.
    #
    # Off by default, because the paper models the level.  Turn it on when you
    # want the deep models to actually work out of sample.  The reason is
    # arithmetic, not taste: the EUA price roughly triples between the training
    # and test windows, and an LSTM or GRU squashes its state through tanh, so
    # once the inputs leave the range seen in training the hidden state
    # saturates and the forecast flattens out.  Measured on the real dataset
    # with 25 epochs and everything else at the paper's settings:
    #
    #     one-hot regimes + exogenous   RMSE 55.2   R2 -12.37   (predictions
    #                                                            stuck in
    #                                                            EUR 15-26)
    #     exogenous only                RMSE 24.0   R2  -1.53
    #     denoised price only           RMSE  4.42  R2   0.914
    #
    # Differencing puts every input back in a range the network has seen, so
    # the level is reconstructed as (last observed level + predicted change).
    stationary: bool = False

    def __post_init__(self):
        if self.mode == "causal":
            # enforce the leakage-free protocol
            if self.wavelet.denoise_mode in ("paper", "threshold"):
                self.wavelet = WaveletConfig(
                    **{**self.wavelet.as_dict(), "denoise_mode": "causal"}
                )
            self.target = "raw"
            self.scale_on = "train"

    def describe(self) -> dict:
        return {
            "mode": self.mode, "detector": self.detector, "model": self.model,
            "multivariate": self.multivariate, "target": self.target,
            "window": self.window, "horizon": self.horizon,
            "train_size": self.train_size, "scale": self.scale,
            "scale_on": self.scale_on,
            "use_regimes": self.use_regimes,
            "per_column_breaks": self.per_column_breaks,
            "stationary": self.stationary,
            **{f"wavelet_{k}": v for k, v in self.wavelet.as_dict().items()},
            **{f"pelt_{k}": v for k, v in self.pelt.as_dict().items()},
            **{f"train_{k}": v for k, v in self.train.as_dict().items()},
        }


# ---------------------------------------------------------------------------
class PELTWTPipeline:
    """PELT + wavelet + deep sequence model, fitted in one call.

    Parameters
    ----------
    cfg : PipelineConfig, optional
        Defaults to the paper's configuration.
    **overrides
        Any field of :class:`PipelineConfig`, e.g. ``model="gru"``.

    Attributes set after :meth:`fit`
    --------------------------------
    breaks_ : BreakResult | dict[str, BreakResult]
    denoised_ : np.ndarray
    data_ : SupervisedData
    y_pred_ : np.ndarray          predictions in price units
    y_true_ : np.ndarray          realised values in price units
    metrics_ : dict
    train_time_ : float

    Examples
    --------
    >>> from peltwtcn import PELTWTPipeline, load_paper_dataset   # doctest: +SKIP
    >>> df = load_paper_dataset()                                 # doctest: +SKIP
    >>> pipe = PELTWTPipeline(model="tcn").fit(df)                # doctest: +SKIP
    >>> pipe.metrics_["RMSE"]                                     # doctest: +SKIP
    """

    def __init__(self, cfg: PipelineConfig | None = None, **overrides):
        cfg = cfg or PipelineConfig()
        if overrides:
            base = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}
            cfg = PipelineConfig(**{**base, **overrides})
        self.cfg = cfg
        self.breaks_ = None
        self.denoised_ = None
        self.data_: SupervisedData | None = None
        self.y_pred_ = None
        self.y_true_ = None
        self.metrics_: dict = {}
        self.history_: dict = {}
        self.train_time_: float = float("nan")
        self.n_params_: int = 0

    # -- individual stages, exposed so they can be inspected or reused -------
    def detect_breaks(self, df: pd.DataFrame, price_col: str,
                      exog_cols: Sequence[str]) -> BreakResult | dict:
        """Stage 1: structural break detection (Section 3.1)."""
        cfg = self.cfg
        n = len(df)
        price = df[price_col].to_numpy(float)

        if cfg.detector == "none":
            return BreakResult([], n, "none", np.zeros(n, dtype=int), {})

        # In causal mode the detector may only see the training window; the
        # test period is then assigned to the last known regime, which is the
        # only assignment available in real time.
        fit_n = train_test_split_index(n, cfg.train_size)[0] if cfg.mode == "causal" else n

        if cfg.detector == "bp_icss":
            res = bp_icss_breakpoints(price[:fit_n], max_breaks=cfg.max_breaks_bp,
                                      trim=cfg.trim_bp,
                                      min_size=cfg.pelt.min_size)
        elif cfg.per_column_breaks:
            sub = df[[price_col, *exog_cols]].iloc[:fit_n]
            per = pelt_multivariate(sub, cfg.pelt)
            if fit_n != n:
                per = {k: BreakResult(v.breakpoints, n, v.method,
                                      regimes_from_breakpoints(n, v.breakpoints),
                                      v.detail)
                       for k, v in per.items()}
            return per
        else:
            res = pelt_breakpoints(price[:fit_n], cfg.pelt)

        if fit_n != n:
            res = BreakResult(res.breakpoints, n, res.method,
                              regimes_from_breakpoints(n, res.breakpoints),
                              res.detail)
        return res

    def denoise(self, price: np.ndarray) -> np.ndarray:
        """Stage 2: wavelet denoising (Section 3.2)."""
        return wavelet_denoise(price, self.cfg.wavelet)

    # -- main entry point ----------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        price_col: str = "Carbon_Price",
        exog_cols: Iterable[str] | None = None,
    ) -> "PELTWTPipeline":
        """Run the whole pipeline on a dated DataFrame.

        Parameters
        ----------
        df : DataFrame
            Must contain ``price_col``; any other column may be used as
            exogenous input.
        exog_cols : iterable of str, optional
            Defaults to every column except the price when
            ``cfg.multivariate`` is True, and to nothing otherwise.
        """
        cfg = self.cfg
        if price_col not in df.columns:
            raise KeyError(f"{price_col!r} not in the DataFrame")

        if exog_cols is None:
            exog_cols = [c for c in df.columns if c != price_col] if cfg.multivariate else []
        exog_cols = list(exog_cols)

        price = df[price_col].to_numpy(float)
        n = price.size

        # 1 -- breaks ---------------------------------------------------------
        self.breaks_ = self.detect_breaks(df, price_col, exog_cols)
        regimes = (build_regime_matrix(self.breaks_, n)
                   if cfg.use_regimes and cfg.detector != "none" else None)

        # 2 -- wavelet --------------------------------------------------------
        self.denoised_ = self.denoise(price)

        # 3 -- design matrix and windows --------------------------------------
        exog = df[exog_cols] if exog_cols else None
        Z = build_design_matrix(self.denoised_, exog, regimes, index=df.index)
        level = self.denoised_ if cfg.target == "denoised" else price

        if cfg.stationary:
            # Difference the continuous block and leave the regime dummies as
            # they are: they are already indicators, not levels.
            reg_cols = set(regimes.columns) if regimes is not None else set()
            Zd = Z.copy()
            for c in Zd.columns:
                if c not in reg_cols:
                    Zd[c] = Zd[c].diff().fillna(0.0)
            Z = Zd
            target = np.diff(level, prepend=level[0])
        else:
            target = level

        self.data_ = make_windows(
            Z, target,
            window=cfg.window, horizon=cfg.horizon, stride=cfg.stride,
            train_size=cfg.train_size, scale=cfg.scale, scale_on=cfg.scale_on,
            target_name=price_col,
        )

        # 4 -- model ----------------------------------------------------------
        out = fit_model(cfg.model, self.data_.X_train, self.data_.y_train,
                        self.data_.X_test, cfg.train)
        self.history_ = out["history"]
        self.train_time_ = out["train_time"]
        self.n_params_ = out["n_params"]
        self.epochs_run_ = out["epochs_run"]

        pred = self.data_.inverse_y(out["y_pred"])
        if cfg.stationary:
            # Rebuild the level: last observed value plus the predicted change.
            # Metrics stay in price units, so they remain directly comparable
            # with the paper's Table 1 and with the level-mode runs.
            pos = df.index.get_indexer(self.data_.index_test)
            base = level[pos - cfg.horizon]
            self.y_pred_ = base + pred
            self.y_true_ = level[pos]
        else:
            self.y_pred_ = pred
            self.y_true_ = self.data_.y_test_raw

        from .metrics import evaluate
        self.metrics_ = evaluate(self.y_true_, self.y_pred_, name=self.name)
        return self

    # -- convenience ---------------------------------------------------------
    @property
    def name(self) -> str:
        """Model label in the paper's notation, e.g. ``PELT-WT-TCN``."""
        det = {"pelt": "PELT", "bp_icss": "BP&ICSS", "none": "NoBreak"}[self.cfg.detector]
        wt = "WT" if self.cfg.wavelet.denoise_mode != "none" else "RAW"
        mdl = self.cfg.model.upper()
        suffix = "" if self.cfg.model != "lstm" else (
            " (multi)" if self.cfg.multivariate else " (uni)")
        return f"{det}-{wt}-{mdl}{suffix}"

    @property
    def predictions(self) -> pd.DataFrame:
        """Test-window actual vs predicted, indexed by date."""
        if self.y_pred_ is None:
            raise RuntimeError("call fit() first")
        return pd.DataFrame(
            {"actual": self.y_true_, "predicted": self.y_pred_,
             "residual": self.y_true_ - self.y_pred_},
            index=self.data_.index_test,
        )

    def summary(self) -> str:
        """Human-readable one-screen report."""
        if not self.metrics_:
            raise RuntimeError("call fit() first")
        m = self.metrics_
        b = self.breaks_
        nb = (sum(v.n_breaks for v in b.values()) if isinstance(b, dict)
              else getattr(b, "n_breaks", 0))
        lines = [
            f"{'=' * 62}",
            f" {self.name}   [mode={self.cfg.mode}, target={self.cfg.target}]",
            f"{'=' * 62}",
            f" breaks detected : {nb}",
            f" wavelet         : {self.cfg.wavelet.wavelet}, level "
            f"{self.cfg.wavelet.level}, {self.cfg.wavelet.denoise_mode}",
            f" input shape     : {self.data_.X_train.shape[1:]} "
            f"({self.data_.n_features} features)",
            f" train / test    : {len(self.data_.y_train)} / {len(self.data_.y_test)}",
            f" parameters      : {self.n_params_:,}   epochs: {self.epochs_run_}",
            f" training time   : {self.train_time_:.1f} s",
            f"{'-' * 62}",
            f" MAE   {m['MAE']:>10.4f}     RMSE  {m['RMSE']:>10.4f}",
            f" MAPE  {m['MAPE (%)']:>10.4f} %   R2    {m['R2']:>10.4f}",
            f" Theil U {m['Theil U']:>8.4f}  (<1 beats a random walk)",
            f"{'=' * 62}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#: The five specifications compared in Table 1 of the paper.
PAPER_MODELS: dict[str, dict] = {
    "BP&ICSS-WT-LSTM":      dict(detector="bp_icss", model="lstm", multivariate=True),
    "PELT-WT-LSTM (uni)":   dict(detector="pelt",    model="lstm", multivariate=False),
    "PELT-WT-LSTM (multi)": dict(detector="pelt",    model="lstm", multivariate=True),
    "PELT-WT-GRU":          dict(detector="pelt",    model="gru",  multivariate=True),
    "PELT-WT-TCN":          dict(detector="pelt",    model="tcn",  multivariate=True),
}


@dataclass
class ExperimentResult:
    """Everything produced by :func:`run_experiment`."""

    table: pd.DataFrame
    predictions: dict[str, np.ndarray]
    actual: np.ndarray
    index: pd.Index
    pipelines: dict[str, PELTWTPipeline]
    training_times: dict[str, float]
    config: dict

    def residuals(self) -> pd.DataFrame:
        return pd.DataFrame(
            {k: self.actual - v for k, v in self.predictions.items()},
            index=self.index,
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"Actual": self.actual, **self.predictions},
                            index=self.index)

    def best(self, by: str = "RMSE") -> str:
        return str(self.table[by].idxmin())


def run_experiment(
    df: pd.DataFrame,
    price_col: str = "Carbon_Price",
    models: Mapping[str, dict] | None = None,
    mode: Mode = "paper",
    include_random_walk: bool = True,
    base: PipelineConfig | None = None,
    verbose: bool = True,
    **overrides,
) -> ExperimentResult:
    """Fit every specification of Table 1 and return a comparable table.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`peltwtcn.datasets.load_paper_dataset`.
    models : mapping, optional
        ``{label: pipeline_kwargs}``.  Defaults to :data:`PAPER_MODELS`.
    mode : {"paper", "causal"}
        See the module docstring.
    include_random_walk : bool
        Append the no-change benchmark to the table.  Strongly recommended.

    Returns
    -------
    ExperimentResult

    Examples
    --------
    >>> res = run_experiment(df)                        # doctest: +SKIP
    >>> res.table.round(4)                              # doctest: +SKIP
    """
    models = dict(models or PAPER_MODELS)
    preds: dict[str, np.ndarray] = {}
    pipes: dict[str, PELTWTPipeline] = {}
    times: dict[str, float] = {}
    actual = None
    index = None

    for label, kw in models.items():
        if verbose:
            print(f"  fitting {label} ...", flush=True)
        cfg_kwargs = {**kw, **overrides, "mode": mode}
        pipe = PELTWTPipeline(base, **cfg_kwargs).fit(df, price_col=price_col)
        pipes[label] = pipe
        preds[label] = pipe.y_pred_
        times[label] = pipe.train_time_
        if actual is None:
            actual, index = pipe.y_true_, pipe.data_.index_test
        elif len(pipe.y_true_) != len(actual):
            raise RuntimeError(
                f"{label} produced {len(pipe.y_true_)} test points but the "
                f"first model produced {len(actual)}; the comparison would "
                "not be like for like"
            )

    if include_random_walk:
        preds["Random walk"] = naive_random_walk(actual)
        times["Random walk"] = 0.0

    table = evaluate_many(actual, preds, training_times=times)
    cfg_desc = pipes[next(iter(pipes))].cfg.describe() if pipes else {}
    return ExperimentResult(table=table, predictions=preds, actual=actual,
                            index=index, pipelines=pipes, training_times=times,
                            config={**cfg_desc, "mode": mode})
