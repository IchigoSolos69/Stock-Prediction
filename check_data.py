"""Explicit diagnostic CLI; exits nonzero on invalid/incomplete inputs."""

import argparse
import numpy as np
from model_trainer import load_model, prepare_features, predict_frame


def verify_data(ticker="HINDCOPPER.NS", use_macros=False):
    frame, columns = prepare_features(ticker, {"use_macros": use_macros, "intraday": False})
    latest = frame.iloc[-1:]
    if latest.empty or not np.isfinite(latest[columns]).all().all():
        raise ValueError("Latest predictor row is incomplete or nonfinite.")
    valid = frame.dropna(subset=columns + ["Target_Return"])
    if len(valid) < 180:
        raise ValueError("Insufficient fully labeled training history.")
    if not latest.Target_Return.isna().all():
        raise ValueError("Latest outcome must be unknown.")
    print(f"Validated {len(valid)} training rows, {len(columns)} predictors; latest bar {frame.index[-1]}.")
    try:
        packet = load_model(ticker)
    except FileNotFoundError:
        print("No compatible model yet; train one before forecasting.")
    else:
        if columns != packet["features"]:
            raise ValueError("Diagnostic configuration differs from the saved model.")
        predict_frame(packet, latest)
        print("Saved model accepts the current predictor schema.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default="HINDCOPPER.NS")
    parser.add_argument("--macros", action="store_true")
    args = parser.parse_args()
    try:
        verify_data(args.symbol, args.macros)
    except Exception as exc:
        parser.exit(1, f"Data validation failed: {exc}\n")
