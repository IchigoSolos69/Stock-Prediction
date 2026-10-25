"""Provider access and causal OHLCV validation; no fabricated candles."""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from angel_bridge import angel
from market_time import IST, completed_daily, completed_intraday, now_ist
from symbols import canonical_symbol, is_indian

log = logging.getLogger(__name__)
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
INTERVALS = {
    "1m": (1, "ONE_MINUTE"),
    "5m": (5, "FIVE_MINUTE"),
    "15m": (15, "FIFTEEN_MINUTE"),
    "30m": (30, "THIRTY_MINUTE"),
    "60m": (60, "ONE_HOUR"),
}


def clean_ohlcv(data, timezone=IST):
    if data is None or data.empty:
        return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz=timezone, name="Date"))
    frame = data.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if len(frame.columns.get_level_values(-1).unique()) != 1:
            raise ValueError("Expected one instrument's candles.")
        frame.columns = frame.columns.get_level_values(0)
    if not set(OHLCV).issubset(frame.columns):
        raise ValueError("Provider did not return complete OHLCV columns.")
    index = pd.to_datetime(frame.index, errors="coerce")
    index = (
        index.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
        if index.tz is None
        else index.tz_convert(timezone)
    )
    frame.index = index
    frame = frame.loc[frame.index.notna(), OHLCV]
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(subset=OHLCV)
    valid = (
        (frame[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        & (frame.Volume >= 0)
        & (frame.High >= frame[["Open", "Close", "Low"]].max(axis=1))
        & (frame.Low <= frame[["Open", "Close", "High"]].min(axis=1))
    )
    frame = frame.loc[valid].copy()
    frame.index.name = "Date"
    frame.attrs.update(data.attrs)
    return frame


def fetch_data(ticker="^NSEI", start_date="2010-01-01", now=None):
    ticker = canonical_symbol(ticker)
    try:
        # Always adjusted history: raw broker bars cannot silently replace this basis.
        raw = yf.Ticker(ticker).history(start=start_date, auto_adjust=True, timeout=10, raise_errors=True)
        raw.attrs.update(source="Yahoo Finance", adjusted=True)
        frame = clean_ohlcv(raw, IST if is_indian(ticker) else "UTC")
        if is_indian(ticker):
            frame = completed_daily(frame, now, "BSE" if ticker.endswith(".BO") else "NSE")
        elif not frame.empty:
            current = now_ist(now).tz_convert("UTC")
            frame = frame.loc[frame.index.normalize() + pd.Timedelta(days=1) <= current]
        return frame
    except Exception as exc:
        log.warning("Daily history unavailable for %s (%s).", ticker, type(exc).__name__)
        return clean_ohlcv(pd.DataFrame())


def fetch_intraday_data(ticker, period="1mo", interval="5m", now=None):
    ticker = canonical_symbol(ticker)
    if interval not in INTERVALS:
        raise ValueError("Unsupported candle interval.")
    minutes, broker_interval = INTERVALS[interval]
    raw = pd.DataFrame()
    if angel.session is not None and ticker.endswith((".NS", ".BO")) and period == "1mo":
        try:
            raw = angel.fetch_historical_data(ticker, interval=broker_interval, days=30)
        except Exception:
            log.warning("Broker intraday history unavailable for %s.", ticker)
    if raw.empty:
        try:
            raw = yf.Ticker(ticker).history(
                period=period, interval=interval, auto_adjust=False, timeout=10, raise_errors=True
            )
            raw.attrs.update(source="Yahoo Finance", adjusted=False)
        except Exception as exc:
            log.warning("Intraday history unavailable for %s (%s).", ticker, type(exc).__name__)
            return clean_ohlcv(pd.DataFrame())
    frame = clean_ohlcv(raw, IST if is_indian(ticker) else "UTC")
    if is_indian(ticker):
        frame = completed_intraday(frame, minutes, now, "BSE" if ticker.endswith(".BO") else "NSE")
    frame.attrs.update(interval_minutes=minutes)
    return frame
