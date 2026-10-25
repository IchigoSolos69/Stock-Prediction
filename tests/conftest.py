import numpy as np
import pandas as pd
import pytest
import requests
import yfinance as yf
from market_time import IST, schedule


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests may not use live network providers.")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(yf, "Ticker", blocked)


@pytest.fixture
def candles():
    sessions = schedule("2024-01-01", "2026-06-30")
    index = pd.DatetimeIndex(sessions.index[:360]).tz_localize(IST)
    x = np.arange(len(index))
    close = 100 * np.exp(0.0005 * x + 0.012 * np.sin(x / 8))
    opening = close * (1 + 0.002 * np.cos(x / 4))
    frame = pd.DataFrame(
        {
            "Open": opening,
            "High": np.maximum(opening, close) * 1.01,
            "Low": np.minimum(opening, close) * 0.99,
            "Close": close,
            "Volume": 1000 + x * 3,
        },
        index=index,
    )
    frame.attrs.update(adjusted=True, source="Synthetic fixture")
    return frame


@pytest.fixture
def intraday_candles():
    days = schedule("2025-01-06", "2025-01-10")
    stamps = []
    for row in days.itertuples():
        stamps.extend(pd.date_range(row.market_open, row.market_close - pd.Timedelta(minutes=5), freq="5min"))
    index = pd.DatetimeIndex(stamps).tz_convert(IST)
    x = np.arange(len(index))
    close = 100 + 0.01 * x + np.sin(x / 9)
    frame = pd.DataFrame(
        {"Open": close - 0.1, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1000 + x}, index=index
    )
    frame.attrs.update(adjusted=False, source="Synthetic fixture", interval_minutes=5)
    return frame
