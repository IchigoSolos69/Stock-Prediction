"""Chronological training and versioned native-XGBoost JSON artifacts."""

import importlib.metadata
import json
import os
import tempfile

import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix, XGBRegressor

from data_loader import clean_ohlcv, fetch_data, fetch_intraday_data
from market_time import completed_daily, completed_intraday
from features import FEATURE_VERSION, add_features, fetch_enhanced_macros, predictor_columns
from symbols import canonical_symbol, is_indian, model_path
from volatility import train_garch_model

ARTIFACT_VERSION = 2
PARAMS = dict(
    n_estimators=180,
    max_depth=3,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=2,
)


def _fit_chronological(frame, columns, target="Target_Return", target_time="Target_Time", gap=1):
    if any(column.startswith("Target_") for column in columns):
        raise ValueError("Future labels cannot be predictor columns.")
    valid = frame.dropna(subset=columns + [target, target_time])
    if not np.isfinite(valid[columns + [target]]).all().all():
        raise ValueError("Training data contains nonfinite inputs.")
    if len(valid) < 180:
        raise ValueError("At least 180 fully labeled feature rows are required.")
    holdout_count = min(100, max(30, len(valid) // 5))
    split = len(valid) - holdout_count
    training, holdout = valid.iloc[: split - gap], valid.iloc[split:]
    if training.empty or training[target_time].max() >= holdout.index[0]:
        raise ValueError("Training outcomes overlap the evaluation period.")
    model = XGBRegressor(**PARAMS)
    model.fit(training[columns], training[target])
    estimates = model.predict(holdout[columns])
    errors = holdout[target].to_numpy() - estimates
    actual = holdout[target].to_numpy()
    metrics = {
        "mae": float(np.mean(np.abs(errors))),
        "zero_return_baseline_mae": float(np.mean(np.abs(actual))),
        "directional_accuracy": float(np.mean(np.sign(estimates) == np.sign(actual))),
        "holdout_rows": len(holdout),
        "training_rows": len(training),
    }
    metadata = {
        "training_start": training.index[0].isoformat(),
        "training_end": training.index[-1].isoformat(),
        "training_label_end": training[target_time].max().isoformat(),
        "holdout_start": holdout.index[0].isoformat(),
        "holdout_end": holdout.index[-1].isoformat(),
        "metrics": metrics,
        "residual_band_90": float(np.quantile(np.abs(errors), 0.90)),
    }
    return model, metadata


def _native_model(model):
    return json.loads(model.get_booster().save_raw(raw_format="json"))


def _save(ticker, payload, kind="daily", directory=None):
    path = model_path(ticker, kind, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.update(
        artifact_version=ARTIFACT_VERSION,
        feature_version=FEATURE_VERSION,
        symbol=canonical_symbol(ticker),
        kind=kind,
        created_at=pd.Timestamp.now(tz="UTC").isoformat(),
        dependencies={
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "xgboost", "scikit-learn")
        },
    )
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as out:
            name = out.name
            json.dump(payload, out, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
    return str(path)


def prepare_features(ticker, config, data=None):
    ticker = canonical_symbol(ticker)
    if not is_indian(ticker):
        raise ValueError(
            "Forecast models currently support NSE/BSE equities and Indian indices. Use a complete Yahoo symbol."
        )
    intraday = bool(config.get("intraday", False))
    raw = data if data is not None else (fetch_intraday_data(ticker) if intraday else fetch_data(ticker))
    if raw.empty:
        raise ValueError("No completed, valid price candles are available.")
    if not intraday and raw.attrs.get("adjusted") is False:
        raise ValueError("Daily models require adjusted history, not raw broker candles.")
    raw = clean_ohlcv(raw)
    exchange = "BSE" if ticker.endswith(".BO") else "NSE"
    raw = completed_intraday(raw, 5, exchange=exchange) if intraday else completed_daily(raw, exchange=exchange)
    if raw.empty:
        raise ValueError("No completed exchange-session candles are available.")
    macros = (
        fetch_enhanced_macros((raw.index[0] - pd.Timedelta(days=60)).strftime("%Y-%m-%d"))
        if config.get("use_macros")
        else None
    )
    features = add_features(raw, macros=macros, intraday=intraday)
    return features, predictor_columns(macros)


def train_model(ticker, use_sentiment=False, use_macros=False, data=None, directory=None):
    # News is a live strategy filter, never a substitute for point-in-time labels.
    config = {"use_macros": bool(use_macros), "intraday": False}
    frame, columns = prepare_features(ticker, config, data)
    model, metadata = _fit_chronological(frame, columns)
    payload = {
        "model": _native_model(model),
        "features": columns,
        "feature_config": config,
        "price_basis": "adjusted",
        "target": "next_session_close_to_close_log_return",
        "garch_sigma": train_garch_model(frame.loc[: metadata["training_end"], "Log_Return"]),
        **metadata,
    }
    return _save(ticker, payload, directory=directory)


def intraday_targets(frame, bars, interval_minutes=5):
    future = frame.Close.shift(-bars)
    times = frame.index.to_series().shift(-bars)
    same_session = times.dt.date == frame.index.date
    correct_gap = times - frame.index.to_series() == pd.Timedelta(minutes=bars * interval_minutes)
    target = np.log(future / frame.Close).where(same_session & correct_gap)
    return target, (times + pd.Timedelta(minutes=interval_minutes)).where(same_session & correct_gap)


def train_intraday_model(ticker, data=None, directory=None):
    config = {"use_macros": False, "intraday": True}
    frame, columns = prepare_features(ticker, config, data)
    models, metadata = {}, {}
    for horizon, bars in {"5m": 1, "15m": 3, "30m": 6, "60m": 12}.items():
        target, times = intraday_targets(frame, bars)
        labeled = frame.assign(Horizon_Return=target, Horizon_Time=times)
        model, info = _fit_chronological(labeled, columns, "Horizon_Return", "Horizon_Time", gap=bars + 1)
        models[horizon] = _native_model(model)
        metadata[horizon] = info
    return _save(
        ticker,
        {
            "models": models,
            "features": columns,
            "feature_config": config,
            "horizon_metadata": metadata,
            "price_basis": "raw",
            "target": "same_session_future_log_return",
        },
        "intraday",
        directory,
    )


def load_model(ticker, kind="daily", directory=None):
    path = model_path(ticker, kind, directory)
    if not path.exists():
        raise FileNotFoundError("No compatible model. Train a new model; legacy pickle artifacts are not loaded.")
    if path.stat().st_size > 50_000_000:
        raise ValueError("Model artifact exceeds the supported size.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_version") != ARTIFACT_VERSION or payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model pipeline version changed. Retrain this asset.")
    if payload.get("symbol") != canonical_symbol(ticker) or payload.get("kind") != kind:
        raise ValueError("Model identity does not match this asset.")
    columns = payload.get("features")
    config = payload.get("feature_config")
    if (
        not isinstance(columns, list)
        or not columns
        or len(set(columns)) != len(columns)
        or any(not isinstance(c, str) or c.startswith("Target_") for c in columns)
    ):
        raise ValueError("Invalid model feature schema.")
    if not isinstance(config, dict) or bool(config.get("intraday")) != (kind == "intraday"):
        raise ValueError("Invalid model feature configuration.")
    if kind == "daily":
        for key in ("training_label_end", "holdout_start", "holdout_end", "residual_band_90", "model"):
            if key not in payload:
                raise ValueError("Model is missing evaluation provenance. Retrain it.")
    return payload


def predict_frame(payload, frame, horizon=None):
    columns = payload["features"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("Model inputs unavailable: " + ", ".join(missing))
    values = frame[columns]
    if values.empty or not np.isfinite(values).all().all():
        raise ValueError("Model inputs contain missing/nonfinite values; prediction withheld.")
    native = payload["models"][horizon] if horizon is not None else payload["model"]
    booster = Booster()
    booster.load_model(bytearray(json.dumps(native).encode()))
    output = booster.predict(DMatrix(values))
    if not np.isfinite(output).all():
        raise ValueError("Model returned a nonfinite prediction.")
    return pd.Series(output, index=frame.index, name="Prediction")
