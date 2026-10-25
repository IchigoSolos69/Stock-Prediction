"""Deterministic causal features; fetching is a separate, explicit operation."""

import numpy as np
import pandas as pd
from market_time import IST, schedule

FEATURE_VERSION = 2
MACRO_TICKERS = {
    "US_Market": "^GSPC",
    "NASDAQ": "^IXIC",
    "USD_INR": "INR=X",
    "Gold": "GC=F",
    "Crude_Oil": "CL=F",
    "India_VIX": "^INDIAVIX",
    "US_Treasury_10Y": "^TNX",
    "Dollar_Index": "DX-Y.NYB",
}
TECHNICAL_FEATURES = [
    "DayOfWeek",
    "Month",
    "IsMonthEnd",
    "Log_Return",
    "Dist_SMA50",
    "Dist_EMA20",
    "RSI",
    "ADX",
    "ATR_Pct",
    "Dist_VWAP",
    "RVol",
    "Volume_Available",
    "Squeeze_On",
    "BB_Pct",
    "Quarter",
    "Is_Month_Start",
    "Doji",
    "Hammer",
]
LAG_COLUMNS = ["Log_Return", "RSI", "Dist_VWAP", "RVol"]
BASE_FEATURES = TECHNICAL_FEATURES + [f"{name}_Lag_{lag}" for name in LAG_COLUMNS for lag in (1, 2, 3, 5)]


def fetch_enhanced_macros(start_date):
    """Each source day becomes available only at next local midnight, conservatively."""
    import yfinance as yf

    sources = []
    for name, ticker in MACRO_TICKERS.items():
        raw = yf.Ticker(ticker).history(start=start_date, auto_adjust=True, timeout=10, raise_errors=True)
        if raw.empty:
            raise ValueError(f"Required macro series {name} is unavailable.")
        close = pd.to_numeric(raw["Close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        index = pd.DatetimeIndex(raw.index)
        if index.tz is None:
            index = index.tz_localize("UTC")
        available = (index.normalize() + pd.Timedelta(days=1)).tz_convert("UTC")
        item = pd.DataFrame(
            {
                name: close.to_numpy(),
                f"{name}_Change": close.pct_change(fill_method=None).to_numpy(),
                f"{name}_MA5": close.rolling(5).mean().to_numpy(),
                f"{name}_MA20": close.rolling(20).mean().to_numpy(),
            },
            index=available,
        )
        item = item.loc[~item.index.duplicated(keep="last")].sort_index()
        sources.append(item)
    # Sources align on availability, rather than mismatched source-midnight labels.
    frame = pd.concat(sources, axis=1).sort_index().ffill()
    frame.index.name = "AvailableAt"
    return frame


def merge_available_features(frame, external, intraday=False):
    if external.empty or external.index.tz is None or external.index.has_duplicates:
        raise ValueError("External features require unique timezone-aware availability timestamps.")
    collisions = set(external.columns) & set(frame.columns)
    if collisions:
        raise ValueError("External feature names collide with existing columns.")
    times = frame.index if intraday else frame.index.normalize() + pd.Timedelta(hours=15, minutes=30)
    left = pd.DataFrame({"AvailableAt": times.tz_convert("UTC"), "_row": range(len(frame))})
    right = external.copy().sort_index().reset_index()
    right.rename(columns={right.columns[0]: "AvailableAt"}, inplace=True)
    right["AvailableAt"] = pd.to_datetime(right["AvailableAt"], utc=True)
    merged = pd.merge_asof(
        left.sort_values("AvailableAt"), right, on="AvailableAt", direction="backward", tolerance=pd.Timedelta(days=7)
    ).sort_values("_row")
    out = frame.copy()
    for column in external.columns:
        out[column] = merged[column].to_numpy()
    return out


def add_features(data, macros=None, fii_data=None, intraday=False):
    frame = data.copy()
    if frame.empty:
        return frame
    if (
        not isinstance(frame.index, pd.DatetimeIndex)
        or frame.index.tz is None
        or frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError("Features require sorted, unique, timezone-aware candle timestamps.")
    frame.index = frame.index.tz_convert(IST)
    close, high, low, volume = frame.Close, frame.High, frame.Low, frame.Volume
    if not np.isfinite(frame[["Open", "High", "Low", "Close", "Volume"]]).all().all() or (close <= 0).any():
        raise ValueError("Invalid candle values.")
    frame["DayOfWeek"] = frame.index.dayofweek
    frame["Month"] = frame.index.month
    frame["IsMonthEnd"] = frame.index.is_month_end.astype(int)
    frame["Log_Return"] = np.log(close / close.shift(1))
    frame["SMA_50"] = close.rolling(50).mean()
    frame["Dist_SMA50"] = close / frame.SMA_50 - 1
    frame["EMA_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    frame["Dist_EMA20"] = close / frame.EMA_20 - 1
    changes = close.diff()
    gain = changes.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-changes.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    frame.loc[loss.eq(0) & gain.gt(0), "RSI"] = 100
    frame.loc[loss.eq(0) & gain.eq(0), "RSI"] = 50
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    frame["ATR"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["ATR_Pct"] = frame.ATR / close
    up, down = high.diff(), -low.diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=frame.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=frame.index)
    di_plus = 100 * plus.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / frame.ATR.replace(0, np.nan)
    di_minus = 100 * minus.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / frame.ATR.replace(0, np.nan)
    denominator = di_plus + di_minus
    dx = 100 * (di_plus - di_minus).abs() / denominator.replace(0, np.nan)
    dx = dx.mask(denominator.eq(0) | frame.ATR.eq(0), 0)
    frame["ADX"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    typical = (high + low + close) / 3
    if intraday:
        group = pd.Series(frame.index.date, index=frame.index)
        numerator = (typical * volume).groupby(group).cumsum()
        vol_sum = volume.groupby(group).cumsum()
    else:
        numerator = (typical * volume).rolling(20).sum()
        vol_sum = volume.rolling(20).sum()
    frame["VWAP"] = numerator / vol_sum.replace(0, np.nan)
    # Explicit missing-volume indicator accompanies neutral volume-derived inputs.
    frame["Volume_Available"] = vol_sum.gt(0).astype(int)
    frame["Dist_VWAP"] = (close / frame.VWAP - 1).mask(vol_sum.eq(0), 0)
    frame["Vol_SMA20"] = volume.rolling(20).mean()
    frame["RVol"] = (volume / frame.Vol_SMA20.replace(0, np.nan)).mask(frame.Vol_SMA20.eq(0), 0)
    middle, deviation = close.rolling(20).mean(), close.rolling(20).std(ddof=0)
    lower, upper = middle - 2 * deviation, middle + 2 * deviation
    frame["Squeeze_On"] = ((lower > frame.EMA_20 - 2 * frame.ATR) & (upper < frame.EMA_20 + 2 * frame.ATR)).astype(int)
    frame["BB_Pct"] = ((close - lower) / (upper - lower).replace(0, np.nan)).mask((upper - lower).eq(0), 0.5)
    frame = add_advanced_features(frame)
    for column in LAG_COLUMNS:
        for lag in (1, 2, 3, 5):
            frame[f"{column}_Lag_{lag}"] = frame[column].shift(lag)
    if macros is not None:
        frame = merge_available_features(frame, macros, intraday)
    if fii_data is not None:
        frame = merge_available_features(frame, fii_data, intraday)
    # Label is deliberately excluded by the explicit BASE_FEATURES predictor list.
    frame["Target_Return"] = np.log(close.shift(-1) / close)
    future = frame.index.to_series().shift(-1)
    if intraday:
        frame["Target_Time"] = future
    else:
        sessions = schedule(frame.index[0].date(), frame.index[-1].date() + pd.Timedelta(days=14))
        dates = list(sessions.index.date)
        next_dates = dict(zip(dates[:-1], dates[1:], strict=True))
        expected = pd.Series([next_dates.get(stamp.date()) for stamp in frame.index], index=frame.index)
        correct_session = future.dt.date == expected
        frame["Target_Return"] = frame.Target_Return.where(correct_session)
        frame["Target_Time"] = (future.dt.normalize() + pd.Timedelta(hours=15, minutes=30)).where(correct_session)
    return frame.replace([np.inf, -np.inf], np.nan)


def add_advanced_features(frame):
    frame = frame.copy()
    frame["Quarter"] = frame.index.quarter
    frame["Is_Month_Start"] = frame.index.is_month_start.astype(int)
    span = (frame.High - frame.Low).replace(0, np.nan)
    body = (frame.Close - frame.Open).abs()
    lower_wick = frame[["Open", "Close"]].min(axis=1) - frame.Low
    upper_wick = frame.High - frame[["Open", "Close"]].max(axis=1)
    frame["Doji"] = (body / span < 0.1).astype(int)
    frame["Hammer"] = ((lower_wick >= 2 * body) & (upper_wick <= body) & (body > 0)).astype(int)
    return frame


def predictor_columns(macros=None):
    return BASE_FEATURES + ([] if macros is None else list(macros.columns))
