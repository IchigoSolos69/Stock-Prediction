"""Shared signal rules and executable next-open/next-close backtesting."""

import numpy as np
import pandas as pd

BUY_THRESHOLD = 0.01
NEWS_VETO = -0.2


def signal_for_return(log_return, sentiment_score=None):
    if not np.isfinite(log_return):
        raise ValueError("Prediction must be finite.")
    predicted = float(np.expm1(log_return))
    if predicted > BUY_THRESHOLD:
        if sentiment_score is not None and (not np.isfinite(sentiment_score) or sentiment_score < NEWS_VETO):
            return "AVOID (News Risk)"
        return "BUY"
    if predicted < -BUY_THRESHOLD:
        return "SELL"
    return "NEUTRAL"


def backtest_predictions(
    frame, predictions, training_label_end, cost_bps=5.0, sentiment_scores=None, use_sentiment=False
):
    """Signals at completed close; enter next open, exit next close; long-only cash overnight."""
    if not 0 <= cost_bps <= 1000:
        raise ValueError("One-way trading costs must be between 0 and 1000 bps.")
    if use_sentiment and sentiment_scores is None:
        raise ValueError("Historical point-in-time sentiment is required for a news-filtered backtest.")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Backtest prices must be chronologically unique.")
    result = pd.DataFrame(index=frame.index)
    result["Prediction"] = pd.Series(predictions).reindex(frame.index)
    result["Entry"] = frame.Open.shift(-1)
    result["Exit"] = frame.Close.shift(-1)
    result["Outcome_Time"] = frame.index.to_series().shift(-1)
    cutoff = pd.Timestamp(training_label_end)
    if cutoff.tzinfo is None:
        raise ValueError("Training cutoff must be timezone-aware.")
    result = result.loc[(result.index > cutoff) & result[["Prediction", "Entry", "Exit"]].notna().all(axis=1)].copy()
    if result.empty:
        raise ValueError("No fully labeled observations after the model's training cutoff.")
    if (
        not np.isfinite(result[["Prediction", "Entry", "Exit"]]).all().all()
        or (result[["Entry", "Exit"]] <= 0).any().any()
    ):
        raise ValueError("Invalid prediction or execution prices.")
    if use_sentiment:
        scores = pd.Series(sentiment_scores).reindex(result.index)
        if scores.isna().any():
            raise ValueError("Historical sentiment is missing for some predictions.")
    else:
        scores = [None] * len(result)
    result["Signal"] = [signal_for_return(p, s) == "BUY" for p, s in zip(result.Prediction, scores, strict=True)]
    result["Gross_Return"] = result.Exit / result.Entry - 1
    rate = cost_bps / 10000
    result["Strategy_Return"] = np.where(result.Signal, (result.Exit / result.Entry) * (1 - rate) / (1 + rate) - 1, 0)
    result["Cum_Strategy"] = (1 + result.Strategy_Return).cumprod() - 1
    result["Cum_Market"] = (result.Exit / result.Entry.iloc[0]) * (1 - rate) / (1 + rate) - 1
    trades = result.loc[result.Signal, "Strategy_Return"]
    metrics = {
        "market_return": float(result.Cum_Market.iloc[-1]),
        "strategy_return": float(result.Cum_Strategy.iloc[-1]),
        "win_rate": float((trades > 0).mean()) if len(trades) else None,
        "trade_count": len(trades),
        "observations": len(result),
        "cost_bps": cost_bps,
    }
    return result, metrics
