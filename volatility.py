"""Optional GARCH with an honest standard-deviation fallback."""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def train_garch_model(returns):
    values = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(values) < 2:
        return None
    fallback = float(values.std())
    if fallback == 0 or len(values) < 100:
        return fallback
    try:
        from arch import arch_model

        model = arch_model(values * 100, vol="Garch", p=1, q=1, mean="Constant", dist="Normal")
        fit = model.fit(disp="off", show_warning=False)
        if fit.convergence_flag != 0:
            return fallback
        sigma = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0]) / 100)
        return sigma if np.isfinite(sigma) and sigma >= 0 else fallback
    except ImportError:
        return fallback
    except Exception:
        log.warning("Volatility fit unavailable; using sample standard deviation.")
        return fallback
