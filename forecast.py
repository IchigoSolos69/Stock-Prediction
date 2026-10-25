"""Forecast and evaluation services shared by UI and tests."""

import numpy as np
import pandas as pd
from market_time import now_ist, next_trading_day, schedule
from model_trainer import prepare_features, predict_frame
from strategy import backtest_predictions, signal_for_return


def daily_forecast(ticker, payload, data=None, news=None, use_sentiment=False, now=None):
    frame, columns = prepare_features(ticker, payload["feature_config"], data)
    if columns != payload["features"]:
        raise ValueError("Current predictor schema does not match this model. Retrain the model.")
    current = now_ist(now)
    exchange = "BSE" if ticker.endswith(".BO") else "NSE"
    sessions = schedule(current.date() - pd.Timedelta(days=14), current.date(), exchange)
    completed = sessions.loc[sessions.market_close <= current]
    if completed.empty or frame.index[-1].date() != completed.index[-1].date():
        raise ValueError("Daily history is stale or incomplete; forecast withheld.")
    latest = frame.iloc[-1:]
    prediction = float(predict_frame(payload, latest).iloc[0])
    base = float(latest.Close.iloc[0])
    score = news["score"] if use_sentiment and news is not None else None
    signal = signal_for_return(prediction, score)
    if use_sentiment and news is None and signal == "BUY":
        signal = "WAIT (News Unavailable)"
    band = float(payload["residual_band_90"])
    return {
        "frame": frame,
        "prediction": prediction,
        "base": base,
        "target": base * np.exp(prediction),
        "low": base * np.exp(prediction - band),
        "high": base * np.exp(prediction + band),
        "signal": signal,
        "forecast_date": next_trading_day(latest.index[0], exchange),
        "as_of": latest.index[0],
    }


def run_backtest(ticker, payload, cost_bps=5, use_sentiment=False, data=None):
    if use_sentiment:
        raise ValueError(
            "Historical news is not provisioned. Disable the news veto to evaluate the technical strategy."
        )
    frame, columns = prepare_features(ticker, payload["feature_config"], data)
    if columns != payload["features"]:
        raise ValueError("Model predictor schema changed. Retrain it.")
    cutoff = pd.Timestamp(payload["training_label_end"])
    # Choose evaluated rows after computing their next-session outcome on the full history.
    candidates = frame.loc[(frame.index > cutoff) & frame.Target_Time.notna()].dropna(subset=columns).tail(100)
    if candidates.empty:
        raise ValueError("No completed outcomes available after the training cutoff.")
    predictions = predict_frame(payload, candidates)
    return backtest_predictions(frame, predictions, cutoff, cost_bps)
