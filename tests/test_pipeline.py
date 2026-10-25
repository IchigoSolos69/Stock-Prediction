import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from data_loader import clean_ohlcv
from features import BASE_FEATURES, add_features, merge_available_features
from market_time import completed_daily, completed_intraday, next_trading_day, IST
from intraday_logic import calculate_intraday_signals
from model_trainer import intraday_targets
from symbols import canonical_symbol, intraday_symbol, model_path


@pytest.mark.parametrize("symbol", ["../BAD", "..\\BAD", "C:\\EVIL", "/tmp/BAD", "", "A..B", "A B"])
def test_symbol_paths_rejected(symbol):
    with pytest.raises(ValueError):
        model_path(symbol)


def test_instrument_identity_is_preserved(tmp_path):
    assert model_path("TCS.NS", directory=tmp_path) != model_path("TCS.BO", directory=tmp_path)
    assert model_path("M&M.NS", directory=tmp_path) != model_path("M.M.NS", directory=tmp_path)
    assert intraday_symbol("^NSEI") == "^NSEI"
    assert canonical_symbol(" hindcopper.ns ") == "HINDCOPPER.NS"


def test_bad_rows_are_dropped_not_invented(candles):
    data = candles.iloc[:3].copy().astype(str)
    data.iloc[1, data.columns.get_loc("Close")] = "bad"
    cleaned = clean_ohlcv(data)
    assert len(cleaned) == 2
    assert candles.index[1] not in cleaned.index
    prefix = clean_ohlcv(data.iloc[:2])
    assert_frame_equal(cleaned.loc[: data.index[0]], prefix)


def test_duplicate_and_invalid_timestamps_removed(candles):
    frame = pd.concat([candles.iloc[:2], candles.iloc[:1]])
    invalid = candles.iloc[:1].copy()
    invalid.index = pd.DatetimeIndex([pd.NaT], tz=IST)
    cleaned = clean_ohlcv(pd.concat([frame, invalid]))
    assert len(cleaned) == 2
    assert not cleaned.index.has_duplicates
    assert not cleaned.index.isna().any()


def test_predictors_do_not_change_when_future_is_appended(candles):
    first = add_features(candles.iloc[:200])
    full = add_features(candles)
    assert_frame_equal(first[BASE_FEATURES], full.loc[first.index, BASE_FEATURES])
    assert "Target_Return" not in BASE_FEATURES
    assert first.Target_Return.iloc[-1] != first.Target_Return.iloc[-1]


def test_zero_volume_is_explicit_and_does_not_remove_training(candles):
    candles["Volume"] = 0
    features = add_features(candles)
    mature = features.iloc[60:]
    assert np.isfinite(mature[BASE_FEATURES]).all().all()
    assert mature.Volume_Available.eq(0).all()
    assert mature.Dist_VWAP.eq(0).all()
    assert mature.VWAP.isna().all()


def test_external_features_use_availability_not_date(candles):
    frame = candles.iloc[:3]
    available = frame.index[1] + pd.Timedelta(hours=20)
    external = pd.DataFrame({"Macro": [7.0]}, index=pd.DatetimeIndex([available]).tz_convert("UTC"))
    merged = merge_available_features(frame, external)
    assert merged.Macro.iloc[:2].isna().all()
    assert merged.Macro.iloc[2] == 7


def test_partial_daily_bar_removed(candles):
    last = candles.iloc[-2:]
    result = completed_daily(last, now=last.index[-1] + pd.Timedelta(hours=12))
    assert result.index[-1] == last.index[-2]
    assert next_trading_day("2025-01-24").isoformat() == "2025-01-27"
    assert next_trading_day("2025-03-13").isoformat() == "2025-03-17"  # Holi + weekend


def test_missing_daily_session_has_no_next_session_label(candles):
    data = candles.drop(candles.index[100])
    features = add_features(data)
    assert pd.isna(features.loc[candles.index[99], "Target_Return"])


def test_intraday_target_does_not_cross_sessions_or_gaps(intraday_candles):
    target, times = intraday_targets(intraday_candles, 1)
    days = intraday_candles.index.to_series().dt.date
    assert target.loc[days != days.shift(-1)].isna().all()
    frame = intraday_candles.drop(intraday_candles.index[10])
    result, _ = intraday_targets(frame, 1)
    assert pd.isna(result.loc[intraday_candles.index[9]])
    assert times.iloc[0] == intraday_candles.index[1] + pd.Timedelta(minutes=5)


def test_orb_excludes_0930_and_waits_for_completed_range(intraday_candles):
    data = intraday_candles.iloc[:5].copy()
    data[["High", "Low", "Open", "Close"]] = [101, 99, 100, 100]
    data.iloc[3, data.columns.get_loc("High")] = 110
    data.iloc[4, data.columns.get_loc("High")] = 105
    data.iloc[4, data.columns.get_loc("Close")] = 104
    assert calculate_intraday_signals(data, now="2025-01-06 09:20") is None
    result = calculate_intraday_signals(data, now="2025-01-06 09:40")
    assert result["orb_high"] == 101
    assert result["orb_status"] == "BULLISH BREAKOUT"
    assert calculate_intraday_signals(data, now="2025-01-07 09:40") is None
    assert calculate_intraday_signals(data, now="2025-01-06 11:40") is None


def test_zero_volume_scanner_withholds_volume_signal(intraday_candles):
    data = intraday_candles.iloc[:4].copy()
    data["Volume"] = 0
    result = calculate_intraday_signals(data, now="2025-01-06 09:35")
    assert result["current_vwap"] is None
    assert result["vwap_dist"] is None
    assert result["vwap_signal"] == "Volume unavailable"


def test_incomplete_intraday_bar_removed(intraday_candles):
    result = completed_intraday(intraday_candles.iloc[:4], now="2025-01-06 09:32")
    assert len(result) == 3


def test_macro_sources_align_on_available_instants(monkeypatch):
    from types import SimpleNamespace
    import features
    import yfinance as yf

    index_us = pd.date_range("2025-01-02", periods=25, freq="D", tz="America/New_York")
    index_in = pd.date_range("2025-01-02", periods=25, freq="D", tz=IST)

    def provider(ticker):
        index = index_in if ticker in {"^INDIAVIX", "INR=X"} else index_us
        return SimpleNamespace(history=lambda **kwargs: pd.DataFrame({"Close": np.arange(25) + 100.0}, index=index))

    monkeypatch.setattr(yf, "Ticker", provider)
    external = features.fetch_enhanced_macros("2025-01-01")
    assert external.index.tz is not None
    assert external.US_Market.notna().sum() >= 25
    assert external.India_VIX.notna().sum() >= 25
    assert external.index.min() == index_in[0].normalize() + pd.Timedelta(days=1)
