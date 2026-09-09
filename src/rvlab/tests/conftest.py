"""Shared fixtures. Every test runs on synthetic data so none of them need the
gitignored market files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session", autouse=True)
def _synthetic_mode():
    """Force the synthetic path so the suite is reproducible anywhere."""
    os.environ["RVLAB_FORCE_SYNTHETIC"] = "1"
    yield
    os.environ.pop("RVLAB_FORCE_SYNTHETIC", None)


@pytest.fixture(scope="session")
def panel():
    from rvlab.data import load_smile_panel
    return load_smile_panel()


@pytest.fixture(scope="session")
def intraday():
    """A continuous 1-minute panel with 3 expiries per bar and no overnight gaps."""
    import numpy as np
    import pandas as pd

    ts = pd.date_range("2026-01-05 09:30", periods=600, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ts": np.repeat(ts, 3),
        "expiry": np.tile(["a", "b", "c"], len(ts)),
        "x": rng.standard_normal(3 * len(ts)),
    })
