"""Optional read-only broker adapter. Importing never authenticates."""

import importlib.util
import logging
import os
from threading import RLock
from unittest.mock import patch

import pandas as pd
import requests
from market_time import IST, now_ist
from symbols import canonical_symbol

SMART_API_AVAILABLE = importlib.util.find_spec("SmartApi") is not None and importlib.util.find_spec("pyotp") is not None
log = logging.getLogger(__name__)
_sdk_import_lock = RLock()


def _load_smart_connect():
    # SmartApi performs an IP lookup in its class body, before its client's
    # timeout exists. Preserve requests.get semantics while bounding that call.
    with _sdk_import_lock:
        original_get = requests.get

        def bounded_get(*args, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = (3.05, 8)
            return original_get(*args, **kwargs)

        with patch.object(requests, "get", bounded_get):
            return importlib.import_module("SmartApi").SmartConnect


class AngelBridge:
    def __init__(self):
        self.session = None
        self._tokens = {}

    def connect(self, api_key=None, client_code=None, pin=None, totp_secret=None):
        self.session = None
        self._tokens.clear()
        if not SMART_API_AVAILABLE:
            return False
        values = (
            api_key or os.getenv("ANGEL_API_KEY"),
            client_code or os.getenv("ANGEL_CLIENT_ID"),
            pin or os.getenv("ANGEL_PIN"),
            totp_secret or os.getenv("ANGEL_TOTP_SECRET"),
        )
        if not all(values):
            return False
        try:
            # The SDK logs auth payloads on errors. Disable its logger before use.
            import logzero

            logzero.logger.disabled = True
            SmartConnect = _load_smart_connect()
            import pyotp

            logging.getLogger("SmartApi.smartConnect").disabled = True
            session = SmartConnect(api_key=values[0], timeout=10)
            logzero.logfile(None)
            response = session.generateSession(values[1], values[2], pyotp.TOTP(values[3]).now())
            if (
                not isinstance(response, dict)
                or not response.get("status")
                or not (response.get("data") or {}).get("jwtToken")
            ):
                return False
            self.session = session
            return True
        except Exception:
            log.warning("Broker login unavailable; using public data provider.")
            return False

    def _instrument(self, symbol):
        symbol = canonical_symbol(symbol)
        if not symbol.endswith((".NS", ".BO")):
            raise ValueError("Broker adapter supports NSE/BSE equities only.")
        if symbol not in self._tokens:
            exchange = "NSE" if symbol.endswith(".NS") else "BSE"
            name = symbol.rsplit(".", 1)[0]
            response = self.session.searchScrip(exchange, name)
            matches = [
                row
                for row in response.get("data", [])
                if row.get("tradingsymbol") == (f"{name}-EQ" if exchange == "NSE" else name)
            ]
            if len(matches) != 1 or not response.get("status"):
                raise ValueError("Broker instrument could not be resolved unambiguously.")
            self._tokens[symbol] = (exchange, matches[0]["tradingsymbol"], matches[0]["symboltoken"])
        return self._tokens[symbol]

    def fetch_historical_data(self, symbol, interval="ONE_DAY", days=30, start_date=None):
        if self.session is None:
            return pd.DataFrame()
        exchange, _, token = self._instrument(symbol)
        end = now_ist()
        start = now_ist(start_date) if start_date is not None else end - pd.Timedelta(days=days)
        windows = []
        step = pd.Timedelta(days=30 if interval == "ONE_DAY" else 5)
        while start < end:
            stop = min(start + step, end)
            response = self.session.getCandleData(
                {
                    "exchange": exchange,
                    "symboltoken": token,
                    "interval": interval,
                    "fromdate": start.strftime("%Y-%m-%d %H:%M"),
                    "todate": stop.strftime("%Y-%m-%d %H:%M"),
                }
            )
            if not response.get("status"):
                raise ValueError("Broker historical data unavailable.")
            windows.extend(response.get("data") or [])
            start = stop
        if not windows:
            return pd.DataFrame()
        frame = pd.DataFrame(windows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        frame.index = pd.to_datetime(frame.pop("Date"), utc=True).dt.tz_convert(IST)
        frame.attrs.update(source="Angel One", adjusted=False)
        return frame

    def get_ltp(self, symbol):
        if self.session is None:
            return None
        exchange, name, token = self._instrument(symbol)
        response = self.session.ltpData(exchange, name, token)
        return (response.get("data") or {}).get("ltp") if response.get("status") else None


angel = AngelBridge()
