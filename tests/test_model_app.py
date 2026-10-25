import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from features import add_features, BASE_FEATURES
from forecast import daily_forecast, run_backtest
from model_trainer import load_model, predict_frame, train_model, train_intraday_model
from symbols import model_path


def test_real_training_native_artifact_prediction_and_holdout(candles, tmp_path):
    path = train_model("TCS.NS", data=candles, directory=tmp_path)
    packet = load_model("TCS.NS", directory=tmp_path)
    assert packet["artifact_version"] == 2
    assert packet["features"] == BASE_FEATURES
    assert pd.Timestamp(packet["training_label_end"]) < pd.Timestamp(packet["holdout_start"])
    assert json.loads(Path(path).read_text(encoding="utf-8"))["symbol"] == "TCS.NS"
    frame = add_features(candles)
    latest = frame.iloc[-1:]
    estimate = predict_frame(packet, latest)
    assert np.isfinite(estimate).all()
    with pytest.raises(ValueError, match="unavailable"):
        predict_frame(packet, latest.drop(columns="RSI"))
    broken = latest.copy()
    broken["RSI"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        predict_frame(packet, broken)
    rows, metrics = run_backtest("TCS.NS", packet, data=candles)
    assert rows.index.min() > pd.Timestamp(packet["training_label_end"])
    assert rows.Outcome_Time.notna().all()
    assert metrics["observations"] > 0
    with pytest.raises(ValueError, match="Historical news"):
        run_backtest("TCS.NS", packet, use_sentiment=True, data=candles)
    now = candles.index[-1] + pd.Timedelta(hours=16)
    output = daily_forecast("TCS.NS", packet, data=candles, now=now)
    assert output["forecast_date"] > candles.index[-1].date()
    with pytest.raises(ValueError, match="stale"):
        daily_forecast("TCS.NS", packet, data=candles.iloc[:-5], now=now)
    packet["feature_version"] = 1
    model_path("TCS.NS", directory=tmp_path).write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="version"):
        load_model("TCS.NS", directory=tmp_path)


def test_intraday_models_have_no_future_inputs(intraday_candles, tmp_path):
    train_intraday_model("TCS.NS", data=intraday_candles, directory=tmp_path)
    packet = load_model("TCS.NS", "intraday", tmp_path)
    assert set(packet["models"]) == {"5m", "15m", "30m", "60m"}
    assert "Target_Return" not in packet["features"]
    for metadata in packet["horizon_metadata"].values():
        assert pd.Timestamp(metadata["training_label_end"]) < pd.Timestamp(metadata["holdout_start"])
    frame = add_features(intraday_candles, intraday=True)
    assert np.isfinite(predict_frame(packet, frame.iloc[-1:], "60m")).all()


def test_legacy_pickle_is_never_loaded(tmp_path):
    (tmp_path / "TCS_model.pkl").write_bytes(b"not a model")
    with pytest.raises(FileNotFoundError, match="legacy"):
        load_model("TCS.NS", directory=tmp_path)


def test_app_starts_offline_and_views_are_lazy(model_storage):
    app = AppTest.from_file("app.py", default_timeout=20).run()
    assert not app.exception
    assert any("Legacy" in element.value for element in app.info)
    app.radio[0].set_value("Backtest Lab").run()
    assert not app.exception
    assert any("Historical news" in element.value for element in app.info)
    app.radio[0].set_value("Derivatives (F&O)").run()
    assert not app.exception
    assert any(button.label == "Scan option chain" for button in app.button)


def test_invalid_symbol_does_not_create_paths(model_storage):
    app = AppTest.from_file("app.py", default_timeout=20).run()
    app.text_input[0].set_value("../INVALID").run()
    assert not app.exception
    assert any("valid market symbol" in element.value for element in app.error)


@pytest.fixture
def model_storage(monkeypatch, tmp_path):
    import model_trainer

    monkeypatch.setattr(
        model_trainer,
        "model_path",
        lambda ticker, kind="daily", directory=None: model_path(ticker, kind, directory or tmp_path),
    )
    return tmp_path


def test_full_ui_training_forecast_backtest_and_persistence(candles, model_storage, monkeypatch):
    import data_loader
    import forecast
    import live_price
    import streamlit as st

    st.cache_data.clear()
    monkeypatch.setattr(data_loader, "fetch_data", lambda ticker: candles.copy())
    frozen = candles.index[-1] + pd.Timedelta(hours=16)
    monkeypatch.setattr(forecast, "now_ist", lambda now=None: frozen)
    monkeypatch.setattr(live_price, "get_quote", lambda ticker: live_price.Quote(None, None, "Fixture", "unavailable"))
    app = AppTest.from_file("app.py", default_timeout=20).run()
    app.toggle[0].set_value(False).run()
    next(button for button in app.button if button.label == "Train / retrain daily model").click().run()
    assert not app.exception and not app.error
    assert app.success
    next(button for button in app.button if button.label == "Refresh forecast").click().run()
    assert not app.exception and not app.error
    assert any(metric.label == "Next-session close estimate" for metric in app.metric)
    app.run()
    assert any(metric.label == "Next-session close estimate" for metric in app.metric)
    app.radio[0].set_value("Backtest Lab").run()
    next(button for button in app.button if button.label.startswith("Run backtest")).click().run()
    assert not app.exception and not app.error
    assert any(metric.label == "Strategy return after costs" for metric in app.metric)


def test_options_ui_handles_missing_quote(model_storage, monkeypatch):
    import derivatives
    import live_price
    import streamlit as st

    st.cache_data.clear()
    monkeypatch.setattr(
        derivatives,
        "get_derivatives_data",
        lambda ticker: {
            "pcr_oi": 1.2,
            "max_pain": 100,
            "expiry": "2026-10-29",
            "retrieved_at": pd.Timestamp("2026-10-08", tz="UTC"),
            "source": "Fixture",
            "support_oi": 90,
            "resistance_oi": 110,
        },
    )
    monkeypatch.setattr(live_price, "get_quote", lambda ticker: live_price.Quote(None, None, "Fixture", "unavailable"))
    app = AppTest.from_file("app.py", default_timeout=20).run()
    app.radio[0].set_value("Derivatives (F&O)").run()
    next(button for button in app.button if button.label == "Scan option chain").click().run()
    assert not app.exception and not app.error
    assert next(metric for metric in app.metric if metric.label == "Distance from quote").value == "Unavailable"
