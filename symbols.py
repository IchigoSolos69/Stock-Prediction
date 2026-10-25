"""Canonical instruments and confined artifact paths."""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


def canonical_symbol(symbol):
    value = str(symbol).strip().upper()
    if not re.fullmatch(r"\^?[A-Z0-9][A-Z0-9.&=\-]{0,39}", value) or ".." in value or value.endswith("."):
        raise ValueError("Enter a valid market symbol, without paths or spaces.")
    return value


def intraday_symbol(symbol):
    return canonical_symbol(symbol)


def is_indian(symbol):
    symbol = canonical_symbol(symbol)
    return symbol.endswith((".NS", ".BO")) or symbol in {"^NSEI", "^NSEBANK", "^BSESN", "^INDIAVIX"}


def model_path(symbol, kind="daily", directory=None):
    symbol = canonical_symbol(symbol)
    if kind not in {"daily", "intraday"}:
        raise ValueError("Unsupported model type.")
    directory = Path(directory or MODEL_DIR).resolve()
    label = re.sub(r"[^A-Z0-9_-]", "_", symbol).strip("_")
    digest = hashlib.sha256(symbol.encode()).hexdigest()[:12]
    path = (directory / f"{label}-{digest}_{kind}.json").resolve()
    if path.parent != directory:
        raise ValueError("Model path must remain inside model storage.")
    return path
