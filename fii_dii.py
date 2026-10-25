"""Institutional flows with source dates and explicit unavailable values."""

import logging
import numpy as np
import pandas as pd
import requests
from market_time import now_ist, schedule

log = logging.getLogger(__name__)
TIMEOUT = (3.05, 8)
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/reports/fii-dii"}


def parse_fii_dii(data, now=None):
    if not isinstance(data, list) or not data:
        raise ValueError("Institutional response is empty.")
    values, dates = {}, []
    for row in data:
        category = str(row.get("category", "")).strip().upper()
        key = "fii_net" if category.startswith("FII/FPI") else "dii_net" if category.startswith("DII") else None
        if key:
            value = float(str(row["netValue"]).replace(",", ""))
            day = pd.to_datetime(row["date"], dayfirst=True, errors="raise").date()
            if not np.isfinite(value) or key in values:
                raise ValueError("Invalid/duplicate institutional category.")
            values[key] = value
            dates.append(day)
    current = now_ist(now)
    if set(values) != {"fii_net", "dii_net"} or len(set(dates)) != 1 or dates[0] > current.date():
        raise ValueError("Institutional categories or publication dates are invalid.")
    sessions = schedule(current.date() - pd.Timedelta(days=14), current.date())
    closed = sessions.loc[sessions.market_close <= current]
    latest_closed = closed.index[-1].date() if len(closed) else current.date()
    signal = "Bearish" if values["fii_net"] < -500 else "Bullish" if values["fii_net"] > 500 else "Neutral"
    return {
        "date": dates[0],
        **values,
        "signal": signal,
        "status": "available" if dates[0] >= latest_closed else "stale",
        "source": "NSE",
    }


def fetch_fii_dii_data(now=None):
    try:
        with requests.Session() as session:
            home = session.get("https://www.nseindia.com", headers=HEADERS, timeout=TIMEOUT)
            home.raise_for_status()
            response = session.get("https://www.nseindia.com/api/fiidii", headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            return parse_fii_dii(response.json(), now)
    except Exception as exc:
        log.warning("Institutional feed unavailable (%s).", type(exc).__name__)
        return {
            "date": None,
            "fii_net": None,
            "dii_net": None,
            "signal": "Data Unavailable",
            "status": "unavailable",
            "source": "NSE",
        }
