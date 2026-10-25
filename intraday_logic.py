"""Opening-range and volume signals from completed, current-session bars."""

import numpy as np
import pandas as pd
from market_time import completed_intraday, now_ist, session_bounds


def calculate_intraday_signals(intraday_df, now=None, interval_minutes=5, exchange="NSE"):
    if interval_minutes not in (5, 15):
        raise ValueError("Opening range supports start-stamped 5m or 15m candles.")
    current = now_ist(now)
    bounds = session_bounds(current.date(), exchange)
    if intraday_df.empty or bounds is None or current < bounds[0] + pd.Timedelta(minutes=15):
        return None
    if (
        intraday_df.index.tz is None
        or intraday_df.index.has_duplicates
        or not intraday_df.index.is_monotonic_increasing
    ):
        raise ValueError("Scanner requires sorted unique timezone-aware timestamps.")
    data = completed_intraday(intraday_df, interval_minutes, current, exchange)
    today = data.loc[data.index.date == current.date()].copy()
    if today.empty:
        return None
    opening, closing = bounds
    expected = pd.date_range(
        opening, opening + pd.Timedelta(minutes=15 - interval_minutes), freq=f"{interval_minutes}min"
    )
    if not expected.isin(today.index).all():
        return None
    # A missing or delayed latest candle must not be interpreted as live.
    reference = min(current, closing)
    if reference - (today.index[-1] + pd.Timedelta(minutes=interval_minutes)) > pd.Timedelta(minutes=10):
        return None
    opening_bars = today.loc[(today.index >= opening) & (today.index < opening + pd.Timedelta(minutes=15))]
    orb_high, orb_low = float(opening_bars.High.max()), float(opening_bars.Low.min())
    price = float(today.Close.iloc[-1])
    orb_status = "BULLISH BREAKOUT" if price > orb_high else "BEARISH BREAKOUT" if price < orb_low else "Inside Range"
    volume = today.Volume.cumsum()
    vwap = (((today.High + today.Low + today.Close) / 3) * today.Volume).cumsum() / volume.replace(0, np.nan)
    value = float(vwap.iloc[-1]) if volume.iloc[-1] > 0 else None
    distance = (price / value - 1) * 100 if value is not None and value > 0 else None
    signal = (
        "Volume unavailable"
        if distance is None
        else "Overbought (above VWAP)"
        if distance > 2
        else "Oversold (below VWAP)"
        if distance < -2
        else "Neutral"
    )
    return {
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_status": orb_status,
        "current_vwap": value,
        "vwap_dist": distance,
        "vwap_signal": signal,
        "vwap_series": vwap,
        "valid_rows": len(today),
        "timestamp": today.index[-1],
        "session_status": "closed" if current >= closing else "open",
    }
