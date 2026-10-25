"""Exchange sessions and completed-candle checks in exchange time."""

from functools import lru_cache
import pandas as pd

IST = "Asia/Kolkata"


def now_ist(now=None):
    value = pd.Timestamp.now(tz=IST) if now is None else pd.Timestamp(now)
    return value.tz_localize(IST) if value.tzinfo is None else value.tz_convert(IST)


@lru_cache(maxsize=64)
def _schedule(start, end, exchange):
    import pandas_market_calendars as mcal

    # The package's NSE/BSE alias stops listing holidays after 2024. XBOM's
    # maintained exchange-calendars adapter includes 2025/2026 holidays and
    # raises outside its supported range instead of inventing weekday sessions.
    if exchange not in {"NSE", "BSE"}:
        raise ValueError("Unsupported exchange calendar.")
    return mcal.get_calendar("XBOM").schedule(start_date=start, end_date=end)


def schedule(start, end, exchange="NSE"):
    return _schedule(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()), exchange).copy()


def session_bounds(day, exchange="NSE"):
    sessions = schedule(day, day, exchange)
    if sessions.empty:
        return None
    row = sessions.iloc[0]
    return row["market_open"].tz_convert(IST), row["market_close"].tz_convert(IST)


def next_trading_day(day, exchange="NSE"):
    start = pd.Timestamp(day).date() + pd.Timedelta(days=1)
    sessions = schedule(start, start + pd.Timedelta(days=31), exchange)
    if sessions.empty:
        raise ValueError("No upcoming exchange session in the calendar.")
    return sessions.index[0].date()


def completed_daily(data, now=None, exchange="NSE"):
    if data.empty:
        return data.copy()
    current = now_ist(now)
    if data.index.min().date() > current.date():
        return data.iloc[:0].copy()
    sessions = schedule(data.index.min().date(), current.date(), exchange)
    closes = {day.date(): close for day, close in sessions["market_close"].items()}
    mask = [day.date() in closes and closes[day.date()] <= current for day in data.index]
    return data.loc[mask].copy()


def completed_intraday(data, interval_minutes=5, now=None, exchange="NSE"):
    if data.empty:
        return data.copy()
    current = now_ist(now)
    if data.index.min().date() > current.date():
        return data.iloc[:0].copy()
    sessions = schedule(data.index.min().date(), min(data.index.max().date(), current.date()), exchange)
    bounds = {day.date(): (row.market_open, row.market_close) for day, row in sessions.iterrows()}
    mask = []
    for stamp in data.index:
        opening, closing = bounds.get(stamp.date(), (None, None))
        end = stamp + pd.Timedelta(minutes=interval_minutes)
        mask.append(opening is not None and stamp >= opening and end <= closing and end <= current)
    return data.loc[mask].copy()
