"""Bounded NSE universe fetching with a last-known-good local cache."""

from io import StringIO
import json
import logging
import os
import tempfile

import pandas as pd
import requests
from symbols import ROOT, canonical_symbol

log = logging.getLogger(__name__)
CACHE = ROOT / ".cache" / "nse_universe.json"


def cached_stock_universe(cache_path=CACHE):
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        stocks = payload["stocks"]
        if not isinstance(stocks, dict) or not stocks or any(canonical_symbol(s) != s for s in stocks.values()):
            raise ValueError("Invalid universe cache.")
        return {"stocks": stocks, "status": "cached", "timestamp": payload["timestamp"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {"stocks": {}, "status": "unavailable", "timestamp": None}


def fetch_stock_universe(cache_path=CACHE):
    try:
        response = requests.get(
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=(3.05, 10),
        )
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        frame.columns = frame.columns.str.strip()
        if not {"SYMBOL", "NAME OF COMPANY"}.issubset(frame):
            raise ValueError("NSE universe schema changed.")
        stocks = {
            f"{row['NAME OF COMPANY']} ({row['SYMBOL']})": canonical_symbol(f"{row['SYMBOL']}.NS")
            for _, row in frame.iterrows()
        }
        if not stocks:
            raise ValueError("Empty NSE universe.")
        payload = {"stocks": stocks, "timestamp": pd.Timestamp.now(tz="UTC").isoformat()}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_path.parent, suffix=".tmp", delete=False
        ) as out:
            name = out.name
            json.dump(payload, out)
        try:
            os.replace(name, cache_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return {**payload, "status": "available"}
    except Exception as exc:
        log.warning("NSE universe unavailable (%s); using last known list if present.", type(exc).__name__)
        result = cached_stock_universe(cache_path)
        result["status"] = "degraded" if result["stocks"] else "unavailable"
        return result


def get_all_stocks():
    return fetch_stock_universe()["stocks"]
