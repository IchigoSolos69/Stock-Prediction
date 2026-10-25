"""Options summaries; insufficient activity is not a directional signal."""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from symbols import canonical_symbol

log = logging.getLogger(__name__)


def summarize_options(calls, puts, expiry):
    frames = []
    for source in (calls, puts):
        if not {"strike", "openInterest", "volume"}.issubset(source.columns):
            raise ValueError("Option chain schema unavailable.")
        frame = source[["strike", "openInterest", "volume"]].apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.loc[frame.strike.gt(0) & frame.openInterest.ge(0)].copy()
        frame["volume"] = frame.volume.where(frame.volume.ge(0))
        frames.append(frame)
    calls, puts = frames
    if calls.empty or puts.empty or calls.openInterest.sum() <= 0 or puts.openInterest.sum() <= 0:
        raise ValueError("No sufficient open interest on both sides of the chain.")
    call_vol, put_vol = calls.volume.sum(min_count=1), puts.volume.sum(min_count=1)
    pcr_vol = float(put_vol / call_vol) if np.isfinite(call_vol) and call_vol > 0 and np.isfinite(put_vol) else None
    pcr_oi = float(puts.openInterest.sum() / calls.openInterest.sum())
    strikes = sorted(set(calls.strike) | set(puts.strike))
    losses = [
        (np.maximum(price - calls.strike, 0) * calls.openInterest).sum()
        + (np.maximum(puts.strike - price, 0) * puts.openInterest).sum()
        for price in strikes
    ]
    return {
        "expiry": expiry,
        "pcr_vol": pcr_vol,
        "pcr_oi": pcr_oi,
        "max_pain": float(strikes[int(np.argmin(losses))]),
        "resistance_oi": float(calls.loc[calls.openInterest.idxmax(), "strike"]),
        "support_oi": float(puts.loc[puts.openInterest.idxmax(), "strike"]),
        "calls_vol": float(call_vol) if np.isfinite(call_vol) else None,
        "puts_vol": float(put_vol) if np.isfinite(put_vol) else None,
    }


def get_derivatives_data(ticker_symbol):
    symbol = canonical_symbol(ticker_symbol)
    try:
        ticker = yf.Ticker(symbol)
        exps = ticker.options
        if not exps:
            return None
        today = pd.Timestamp.now(tz="UTC").date()
        valid = sorted(day for day in exps if pd.Timestamp(day).date() >= today)
        if not valid:
            return None
        chain = ticker.option_chain(valid[0])
        result = summarize_options(chain.calls, chain.puts, valid[0])
        result.update(source="Yahoo Finance", retrieved_at=pd.Timestamp.now(tz="UTC"))
        return result
    except Exception as exc:
        log.warning("Option chain unavailable for %s (%s).", symbol, type(exc).__name__)
        return None


def analyze_pcr(pcr):
    if pcr is None or not np.isfinite(pcr) or pcr <= 0:
        return "Unavailable"
    # Descriptive ratios, not an unvalidated directional trading rule.
    return "More put open interest" if pcr > 1 else "More call open interest" if pcr < 1 else "Balanced open interest"
