"""Shared fixtures for the peltwtcn test suite."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.RandomState(12345)


@pytest.fixture(scope="session")
def piecewise_mean():
    """400 points, mean jumps 0 -> 10 -> 3 at t = 150 and t = 280."""
    r = np.random.RandomState(0)
    x = np.concatenate([
        r.normal(0.0, 0.5, 150),
        r.normal(10.0, 0.5, 130),
        r.normal(3.0, 0.5, 120),
    ])
    return x


@pytest.fixture(scope="session")
def piecewise_variance():
    """400 points, constant mean, sigma jumps 0.5 -> 4.0 at t = 200."""
    r = np.random.RandomState(1)
    return np.concatenate([r.normal(0.0, 0.5, 200), r.normal(0.0, 4.0, 200)])


@pytest.fixture(scope="session")
def noisy_trend():
    """A smooth trend plus high-frequency noise, for the wavelet tests."""
    t = np.linspace(0, 8 * np.pi, 512)
    clean = 20.0 + 5.0 * np.sin(t) + 0.02 * np.arange(512)
    noise = np.random.RandomState(2).normal(0.0, 1.0, 512)
    return clean, clean + noise


@pytest.fixture(scope="session")
def toy_panel():
    """A small dated DataFrame shaped like the real dataset."""
    n = 500
    r = np.random.RandomState(3)
    idx = pd.bdate_range("2015-01-01", periods=n)
    price = 20.0 + np.cumsum(r.normal(0.0, 0.4, n))
    price = np.maximum(price, 1.0)
    return pd.DataFrame(
        {
            "Carbon_Price": price,
            "Brent_Crude": 60.0 + np.cumsum(r.normal(0.0, 0.5, n)),
            "TTF_Natural_Gas": 25.0 + np.cumsum(r.normal(0.0, 0.3, n)),
            "Policy": (np.arange(n) > 250).astype(float),
        },
        index=idx,
    )


@pytest.fixture(scope="session")
def fast_train():
    """A TrainConfig that finishes in seconds instead of minutes."""
    from peltwtcn import TrainConfig
    return TrainConfig(epochs=2, batch_size=64, units=16, tcn_filters=8,
                       tcn_blocks=2, patience=2, verbose=0)


@pytest.fixture(scope="session")
def fast_pelt():
    """PELT settings that keep the O(n) search cheap in tests."""
    from peltwtcn import PeltConfig
    return PeltConfig(model="l2", min_size=30, jump=10, penalty="bic")
