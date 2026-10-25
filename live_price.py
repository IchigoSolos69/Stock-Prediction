"""Quotes retain their source timestamp and explicit freshness state."""

from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from market_time import IST, now_ist, session_bounds
from symbols import canonical_symbol, is_indian

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Quote:
    price: float | None
    timestamp: pd.Timestamp | None
    source: str
    status: str


def get_quote(ticker, now=None):
    ticker = canonical_symbol(ticker)
    current = now_ist(now)
    try:
        frame = yf.Ticker(ticker).history(period="5d", interval="1m", auto_adjust=False, timeout=10, raise_errors=True)
        if frame.empty:
            return Quote(None, None, "Yahoo Finance", "unavailable")
        frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
        prices = pd.to_numeric(frame.Close, errors="coerce")
        valid = prices.notna() & np.isfinite(prices) & (prices > 0)
        if not valid.any():
            return Quote(None, None, "Yahoo Finance", "unavailable")
        stamp = pd.Timestamp(frame.index[valid][-1])
        if stamp.tzinfo is None:
            return Quote(None, None, "Yahoo Finance", "unverified timestamp")
        stamp = stamp.tz_convert(IST)
        if stamp > current:
            return Quote(None, stamp, "Yahoo Finance", "invalid future timestamp")
        bounds = (
            session_bounds(current.date(), "BSE" if ticker.endswith(".BO") else "NSE") if is_indian(ticker) else None
        )
        if not is_indian(ticker):
            status = "venue freshness unverified"
        elif bounds is None or not bounds[0] <= current <= bounds[1]:
            status = "market closed"
        elif stamp.date() != current.date() or current - stamp > pd.Timedelta(minutes=5):
            status = "stale"
        else:
            status = "live minute quote"
        return Quote(float(prices.loc[valid].iloc[-1]), stamp, "Yahoo Finance", status)
    except Exception as exc:
        log.warning("Quote unavailable for %s (%s).", ticker, type(exc).__name__)
        return Quote(None, None, "Yahoo Finance", "unavailable")


def get_realtime_price(ticker):
    quote = get_quote(ticker)
    return quote.price if quote.status == "live minute quote" else None
