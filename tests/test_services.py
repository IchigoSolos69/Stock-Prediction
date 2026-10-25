from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest
import requests

import angel_bridge
import fii_dii
import live_price
import sentiment
import stock_list
from derivatives import analyze_pcr, summarize_options
from market_time import IST
from sentiment import FinancialSentimentAnalyzer, parse_yahoo_news
from strategy import backtest_predictions, signal_for_return
from volatility import train_garch_model


def test_optional_broker_has_a_disconnected_interface():
    client = angel_bridge.AngelBridge()
    assert client.session is None
    assert client.fetch_historical_data("TCS.NS").empty
    assert client.get_ltp("TCS.NS") is None
    assert hasattr(client, "connect")


def test_broker_auth_logging_is_suppressed(monkeypatch):
    import sys

    sdk_logger = SimpleNamespace(disabled=False)
    fake_logzero = SimpleNamespace(logger=sdk_logger, logfile=MagicMock())
    session = MagicMock()

    def login(*args):
        assert sdk_logger.disabled
        return {"status": True, "data": {"jwtToken": "fake-test-token"}}

    session.generateSession.side_effect = login
    monkeypatch.setattr(angel_bridge, "SMART_API_AVAILABLE", True)
    monkeypatch.setitem(sys.modules, "logzero", fake_logzero)
    monkeypatch.setitem(sys.modules, "SmartApi", SimpleNamespace(SmartConnect=lambda **kwargs: session))
    monkeypatch.setitem(sys.modules, "pyotp", SimpleNamespace(TOTP=lambda seed: SimpleNamespace(now=lambda: "123456")))
    client = angel_bridge.AngelBridge()
    assert client.connect("fake-api", "fake-client", "1000", "fake-seed")
    assert client.session is session
    fake_logzero.logfile.assert_called_once_with(None)
    session.generateSession.return_value = {"status": False}
    session.generateSession.side_effect = None
    assert not client.connect("fake-api", "fake-client", "1000", "fake-seed")
    assert client.session is None


def test_broker_sdk_import_request_has_a_timeout(monkeypatch):
    original_get = MagicMock(return_value=SimpleNamespace(text="192.0.2.1"))
    monkeypatch.setattr(angel_bridge.requests, "get", original_get)
    client_class = object()

    def import_sdk(name):
        assert name == "SmartApi"
        angel_bridge.requests.get("https://api.ipify.org")
        angel_bridge.requests.get("https://api.ipify.org", timeout=2)
        return SimpleNamespace(SmartConnect=client_class)

    monkeypatch.setattr(angel_bridge.importlib, "import_module", import_sdk)
    assert angel_bridge._load_smart_connect() is client_class
    assert original_get.call_args_list[0].kwargs["timeout"] == (3.05, 8)
    assert original_get.call_args_list[1].kwargs["timeout"] == 2
    assert angel_bridge.requests.get is original_get

    def failed_import(name):
        raise RuntimeError("Failed SDK import")

    monkeypatch.setattr(angel_bridge.importlib, "import_module", failed_import)
    with pytest.raises(RuntimeError):
        angel_bridge._load_smart_connect()
    assert angel_bridge.requests.get is original_get


def test_institutional_source_dates_and_unavailable_states(monkeypatch):
    data = [
        {"category": "FII/FPI *", "netValue": "-1,250.50", "date": "07-Jan-2025"},
        {"category": "DII **", "netValue": "500", "date": "07-Jan-2025"},
    ]
    result = fii_dii.parse_fii_dii(data, now="2025-01-08 11:00")
    assert result["date"].isoformat() == "2025-01-07"
    assert result["status"] == "available"
    assert result["fii_net"] == -1250.5
    assert fii_dii.parse_fii_dii(data, now="2025-02-03 11:00")["status"] == "stale"
    with pytest.raises(ValueError):
        fii_dii.parse_fii_dii([], now="2025-01-08")
    result = fii_dii.fetch_fii_dii_data()
    assert result["fii_net"] is None and result["date"] is None
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value.json.return_value = data
    monkeypatch.setattr(fii_dii.requests, "Session", lambda: session)
    assert fii_dii.fetch_fii_dii_data("2025-01-08 11:00")["status"] == "available"
    assert all(call.kwargs["timeout"] == fii_dii.TIMEOUT for call in session.get.call_args_list)
    assert session.get.return_value.raise_for_status.call_count == 2


def test_yahoo_news_parses_both_formats_without_invented_dates():
    items = [
        {"content": {"title": "Growth", "pubDate": "2025-01-07T10:00:00Z", "summary": "<b>Profit</b>"}},
        {"title": "Recovery", "providerPublishTime": 1736244000},
        {"content": {"title": "", "pubDate": "2025-01-07T10:00:00Z"}},
        {"title": "Undated"},
    ]
    parsed = parse_yahoo_news(items)
    assert len(parsed) == 2
    assert parsed[0]["PublishedAt"].year == 2025
    assert "<b>" not in parsed[0]["Text"]


def test_google_query_preserves_ampersand(monkeypatch):
    captured = {}

    def get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return SimpleNamespace(content=b"<rss><channel/></rss>", raise_for_status=lambda: None)

    monkeypatch.setattr(sentiment.requests, "get", get)
    analyzer = FinancialSentimentAnalyzer(vader=SimpleNamespace())
    assert analyzer._fetch_google_news_rss("M&M.NS") == []
    prepared = requests.Request("GET", captured["url"], params=captured["params"]).prepare()
    assert parse_qs(urlparse(prepared.url).query)["q"][0].startswith("M&M stock")
    assert captured["timeout"] == (3.05, 8)


def test_news_filters_future_old_and_duplicate_articles(monkeypatch):
    analyzer = FinancialSentimentAnalyzer(vader=SimpleNamespace(polarity_scores=lambda text: {"compound": 0.5}))
    records = [
        {"PublishedAt": pd.Timestamp(stamp), "Title": title, "Text": title, "Source": "fixture"}
        for stamp, title in [
            ("2025-01-08T09:00Z", "Good"),
            ("2025-01-08T08:00Z", "Good"),
            ("2025-01-10T09:00Z", "Future"),
            ("2024-12-01T09:00Z", "Old"),
        ]
    ]
    monkeypatch.setattr(analyzer, "_fetch_google_news_rss", lambda ticker: records)
    result = analyzer.analyze_ticker("TCS.NS", now="2025-01-08T10:00Z")
    assert result["headline_count"] == 1
    assert result["top_headline"] == "Good"


def test_missing_lexicon_does_not_download(monkeypatch):
    def absent():
        raise LookupError("missing")

    monkeypatch.setattr(sentiment, "FinancialSentimentAnalyzer", absent)
    monkeypatch.setattr(sentiment.nltk, "download", lambda *args, **kwargs: pytest.fail("Unexpected download"))
    assert sentiment.get_news_sentiment("TCS.NS") is None


def test_universe_failure_preserves_last_good_cache(tmp_path):
    cache = tmp_path / "stocks.json"
    cache.write_text('{"stocks":{"TCS":"TCS.NS"},"timestamp":"2025-01-01T00:00:00Z"}')
    result = stock_list.fetch_stock_universe(cache)
    assert result["status"] == "degraded"
    assert result["stocks"] == {"TCS": "TCS.NS"}
    assert stock_list.fetch_stock_universe(tmp_path / "missing.json")["stocks"] == {}


def test_option_activity_must_be_real():
    zero = pd.DataFrame({"strike": [100, 110], "openInterest": [0, 0], "volume": [0, 0]})
    with pytest.raises(ValueError):
        summarize_options(zero, zero, "2025-01-30")
    calls = zero.assign(openInterest=[100, 50])
    puts = zero.assign(openInterest=[50, 100])
    result = summarize_options(calls, puts, "2025-01-30")
    assert result["pcr_oi"] == 1
    assert result["pcr_vol"] is None
    assert analyze_pcr(None) == "Unavailable"
    assert analyze_pcr(0) == "Unavailable"


def test_quotes_return_timestamp_and_staleness(monkeypatch):
    frame = pd.DataFrame({"Close": [100.0]}, index=pd.DatetimeIndex(["2024-01-08 09:30"], tz=IST))
    monkeypatch.setattr(live_price.yf, "Ticker", lambda ticker: SimpleNamespace(history=lambda **kwargs: frame))
    quote = live_price.get_quote("TCS.NS", now="2025-01-08 10:00")
    assert quote.price == 100
    assert quote.status == "stale"
    frame.index = pd.DatetimeIndex(["2025-01-08 09:59"], tz=IST)
    assert live_price.get_quote("TCS.NS", now="2025-01-08 10:00").status == "live minute quote"
    assert live_price.get_quote("TCS.NS", now="2025-01-08 08:00").price is None  # future quote
    assert live_price.get_quote("TCS.NS", now="2025-01-11 10:00").status == "market closed"


def test_strategy_threshold_news_and_compounding():
    assert signal_for_return(np.log(1.02)) == "BUY"
    assert signal_for_return(np.log(1.005)) == "NEUTRAL"
    assert signal_for_return(np.log(1.02), -0.3) == "AVOID (News Risk)"
    index = pd.date_range("2025-01-06", periods=3, freq="D", tz=IST)
    frame = pd.DataFrame({"Open": [100, 100, 100], "Close": [100, 110, 110]}, index=index)
    predictions = pd.Series(np.log(1.10), index=index)
    result, metrics = backtest_predictions(frame, predictions, "2025-01-05T15:30:00+05:30", cost_bps=0)
    assert metrics["observations"] == 2
    assert metrics["trade_count"] == 2
    assert metrics["win_rate"] == 1
    assert metrics["strategy_return"] == pytest.approx(0.21)
    assert metrics["market_return"] == pytest.approx(0.10)  # holding through overnight price gap
    assert result.Outcome_Time.notna().all()
    _, cost_metrics = backtest_predictions(frame, predictions, "2025-01-05T15:30:00+05:30", cost_bps=10)
    assert cost_metrics["strategy_return"] < metrics["strategy_return"]
    with pytest.raises(ValueError, match="sentiment"):
        backtest_predictions(frame, predictions, "2025-01-05T15:30:00+05:30", use_sentiment=True)


def test_backtest_excludes_training_outcomes_and_zero_trades():
    index = pd.date_range("2025-01-06", periods=4, tz=IST)
    data = pd.DataFrame({"Open": 100, "Close": 101}, index=index)
    predictions = pd.Series(0.0, index=index)
    result, metrics = backtest_predictions(data, predictions, index[1], cost_bps=0)
    assert list(result.index) == [index[2]]
    assert metrics["trade_count"] == 0 and metrics["win_rate"] is None


def test_volatility_handles_nonfinite_and_small_samples():
    assert train_garch_model([np.nan, np.inf]) is None
    assert train_garch_model([0.1, 0.2, np.nan]) == pytest.approx(pd.Series([0.1, 0.2]).std())
    assert train_garch_model([0.0] * 150) == 0
