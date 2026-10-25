# Stock Forecast Terminal

A Python and Streamlit application for researching Indian stocks and indices. It combines next-session price estimates, completed-candle opening-range scanning, options summaries, news sentiment, institutional flows, and chronological strategy backtesting.

The application runs locally for one user. Public data is the default; an optional Angel One adapter provides broker intraday candles after explicit connection. The adapter exposes no order-placement methods.

Project version: **0.2.0**. Documentation reviewed against the source on **25 October 2026**. The earlier repository audit and implementation record are in [AUDIT_REPORT.md](AUDIT_REPORT.md) and [FIX_STATUS.md](FIX_STATUS.md).

## Contents

- [Features and supported assets](#features-and-supported-assets)
- [Installation and launch](#installation-and-launch)
- [Using the application](#using-the-application)
- [Architecture and repository map](#architecture-and-repository-map)
- [Data sources and validation](#data-sources-and-validation)
- [Features, training, and evaluation](#features-training-and-evaluation)
- [Forecast and strategy rules](#forecast-and-strategy-rules)
- [Model artifacts and local storage](#model-artifacts-and-local-storage)
- [Python service usage](#python-service-usage)
- [Optional integrations and configuration](#optional-integrations-and-configuration)
- [Development and testing](#development-and-testing)
- [Troubleshooting](#troubleshooting)
- [Known limitations and improvement areas](#known-limitations-and-improvement-areas)

## Features and supported assets

| Capability | What is implemented |
| --- | --- |
| Daily Forecast | Train an XGBoost model, inspect holdout metrics, and estimate the next session's adjusted close with an empirical error band. |
| Intraday Scanner | Five-minute candles, the first 15 minutes' opening range, session VWAP, freshness checks, and optional refresh every minute. |
| Derivatives (F&O) | Nearest nonexpired option chain, put/call open-interest ratio, minimum payout strike, and largest put/call OI strikes where the provider has data. |
| Backtest Lab | Long-only technical strategy over up to 100 eligible observations, with execution timing, fees/slippage, compounded returns, and a buy-and-hold comparison. |
| News veto | Recent Google News RSS headlines, Yahoo news fallback, local VADER scoring, and a configurable BUY veto. |
| Institutional flows | NSE FII/DII net flows in ₹ crore with source dates and stale/unavailable states. |
| Asset selection | Built-in Indian indices, an on-demand NSE equity list, and a custom full Yahoo symbol. |
| Intraday model services | Python functions to train and predict 5-, 15-, 30-, and 60-minute horizons; these models are separate from the UI scanner. |

Daily models and the opening-range scanner accept NSE equities (`.NS`), BSE equities (`.BO`), and the recognized Indian indices `^NSEI`, `^NSEBANK`, `^BSESN`, and `^INDIAVIX`. Examples include `HINDCOPPER.NS`, `TCS.NS`, and `M&M.NS`. Use a valid Yahoo identifier for the selected exchange. `.NS` and `.BO` identities remain distinct throughout storage and prediction.

Quote and options adapters can query other Yahoo symbols, but the UI's freshness checks are designed for Indian venues, and its price labels use rupees. Non-Indian assets are not a complete supported forecasting workflow. Yahoo options coverage for Indian assets is limited; an F&O listing does not guarantee a usable public option chain.

## Installation and launch

### Requirements

- Python **3.12** is the tested version. Project metadata permits Python `>=3.12,<3.14`; Python 3.13 has not been verified with the repaired dependency set.
- Internet access is needed for installation, the initial VADER download, and live provider requests. Importing the app does not download NLTK corpora.
- Run commands from the project directory. No database, GPU, broker account, or paid API key is required for the default public-data workflow.

### Existing repaired local installation

This workspace already has a repaired Python 3.12 environment named `.venv-fixed`:

```powershell
.\.venv-fixed\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit, normally `http://127.0.0.1:8501`. Stop the server with `Ctrl+C`.

The old `.venv` in this workspace was broken and has been preserved. Use `.venv-fixed` rather than the old environment. Environments, downloaded corpora, and trained models are ignored generated files, so a fresh checkout needs its own setup and training.

### Fresh Windows setup

Use a standard Python 3.12 installation:

```powershell
py -3.12 -m venv .venv-fixed
.\.venv-fixed\Scripts\python.exe -m pip install -r requirements.txt -c requirements.lock.txt
.\.venv-fixed\Scripts\python.exe -m nltk.downloader -d nltk_data vader_lexicon
.\.venv-fixed\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Activation is optional because these commands address the environment's interpreter directly. If the Windows `py` launcher is unavailable, use `python -m venv .venv-fixed` after confirming that `python --version` reports Python 3.12.

### macOS/Linux command equivalents

```bash
python3.12 -m venv .venv-fixed
.venv-fixed/bin/python -m pip install -r requirements.txt -c requirements.lock.txt
.venv-fixed/bin/python -m nltk.downloader -d nltk_data vader_lexicon
.venv-fixed/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

The resolved constraints were generated on Windows/Python 3.12. These are equivalent launch commands, not a claim that a clean macOS/Linux installation has been validated. The Linux CI configuration is described below.

### Dependency files

| File | Purpose |
| --- | --- |
| `requirements.txt` | Direct runtime dependencies with exact versions. |
| `requirements.lock.txt` | Resolved runtime/development dependency constraints from 8 October 2026; use with `-c`. |
| `requirements-dev.txt` | Runtime dependencies plus pytest and Ruff. |
| `requirements-broker.txt` | Runtime dependencies plus SmartAPI, TOTP, and its supporting packages. |
| `requirements-volatility.txt` | Runtime dependencies plus optional `arch` GARCH support. |

The runtime stack includes Streamlit 1.52.2, pandas 2.3.3, NumPy 2.2.6, Plotly 6.5.0, yfinance 1.0, XGBoost 3.1.2, scikit-learn 1.8.0, pandas-market-calendars 5.2.2, NLTK 3.9.2, Beautiful Soup 4.14.3, and requests 2.32.5. TensorFlow, LightGBM, Optuna, and pandas-ta are not required.

Using the lock file as constraints installs only dependencies required by the selected requirements file. Optional packages have direct pins in their own files; their additional transitive dependencies are not all included in the runtime/development lock. Refresh constraints and verify compatibility when upgrading Python, dependencies, or platforms.

## Using the application

1. Select an index or enter a full Yahoo symbol in the sidebar. The default custom symbol is `HINDCOPPER.NS`. **Load / refresh NSE asset list** retrieves the full NSE list when requested; a custom symbol works without it.
2. Choose **Daily Forecast** and optionally enable **Use macro features when training**. Select **Train / retrain daily model** for an asset without a compatible artifact. Training replaces that asset's previous daily JSON artifact.
3. Select **Refresh forecast**. The app fetches completed history, checks its freshness and feature schema, then displays the estimate, signal, completed-close date, available quote, and error band. Retraining and forecast refresh are separate actions.
4. **Use news veto** is enabled by default. It filters BUY recommendations using recent news; it does not add today's news to model training. When news cannot be scored, a technical BUY becomes `WAIT (News Unavailable)`.
5. Choose **Intraday Scanner** to inspect the current session's opening range and VWAP. The scanner fetches on entry to that view. Use **Refresh scanner** or enable **Refresh every minute** for updates while the view is active.
6. Choose **Derivatives (F&O)** and select **Scan option chain**. Missing chains or quotes are shown explicitly instead of inventing values.
7. For **Backtest Lab**, train a daily model, disable the news veto, set one-way fees/slippage in basis points, and select **Run backtest on last 100 eligible sessions**. Historical news is not provisioned, so a news-filtered replay is disabled.
8. Expand **Institutional flows** and select **Load / refresh institutional flows** to request dated NSE figures.

Only the selected view runs its work. Daily forecasts, backtests, option summaries, and flows persist across Streamlit reruns; asset/model/settings keys prevent unrelated results from being displayed together. Daily forecasts and backtests require an explicit refresh to update their saved results. The macro checkbox changes the configuration of the next training run; prediction follows the configuration stored with the model.

## Architecture and repository map

The main daily workflow is:

```text
Yahoo adjusted OHLCV -> validate and filter completed sessions
                    -> causal technical features (+ optional available macros)
                    -> chronological training / native JSON artifact
                    -> checked prediction -> next-session estimate and shared signal rule

Saved artifact + eligible later history -> next-open/next-close backtest -> metrics and charts
Public or optional broker intraday bars -> completed candles -> opening range and session VWAP
```

| Path | Responsibility |
| --- | --- |
| `app.py` | Streamlit views, controls, charts, session state, and UI caching. |
| `data_loader.py` | Public/broker history access, OHLCV validation, adjusted/raw price basis, completed-candle filtering. |
| `market_time.py` | IST normalization, exchange schedules, session boundaries, next trading day. |
| `symbols.py` | Canonical symbols, supported-venue checks, and confined hashed artifact paths. |
| `features.py` | Deterministic technical features, macro fetching, availability-time joins, and explicit predictor list. |
| `model_trainer.py` | Daily/intraday XGBoost fitting, chronological evaluation, artifact persistence, loading, and prediction. |
| `forecast.py` | Daily forecast freshness checks, estimate/error-band generation, and backtest orchestration. |
| `strategy.py` | Shared signal thresholds, execution timing, costs, compounding, and backtest metrics. |
| `intraday_logic.py` | Opening-range completeness, freshness, breakout state, and session VWAP. |
| `live_price.py` | Timestamped minute quotes with source and freshness status. |
| `sentiment.py` | Google/Yahoo news parsing, date filtering, deduplication, and optional local VADER scoring. |
| `fii_dii.py` | Dated NSE institutional-flow retrieval and validation. |
| `derivatives.py` | Option-chain validation, OI/volume ratios, minimum payout, and OI concentration strikes. |
| `stock_list.py` | Bounded NSE equity-universe requests and last-known-good JSON cache. |
| `angel_bridge.py` | Optional disconnected-by-default read-only broker adapter. |
| `volatility.py` | Optional GARCH(1,1) estimate with standard-deviation fallback. |
| `check_data.py` | Explicit command-line data/model-schema diagnostic. |
| `tests/` | Provider fixtures, pipeline/model/service regressions, and Streamlit AppTest coverage. |
| `.github/workflows/tests.yml` | Python 3.12 installation, dependency check, Ruff, and pytest on pushes/PRs. |
| `pyproject.toml` | Project metadata, Python range, pytest paths, and Ruff configuration. |
| `.env.example`, `.gitignore` | Optional configuration names and exclusions for secrets/generated data. |
| `AUDIT_REPORT.md`, `FIX_STATUS.md` | Historical findings, implemented repairs, verification evidence, and remaining actions. |
| `file.csv` | Legacy NSE-format CSV header; unused by the current application and not a bundled training dataset. |

## Data sources and validation

| Source | Usage and behavior |
| --- | --- |
| Yahoo Finance via yfinance | Adjusted daily OHLCV from 2010 where available; raw intraday candles; minute quotes; optional macro series; news fallback; options where covered. |
| Google News RSS | Primary recent-news source, with encoded query parameters and dated records. |
| NSE | Equity-universe CSV and FII/DII flows. Feed outages retain a previous universe or produce unavailable flow values. |
| Angel One, optional | Raw intraday history for supported NSE/BSE equities when connected; public history remains the fallback. Daily models always use adjusted Yahoo history. |
| Local VADER lexicon | Sentiment scoring; provisioned explicitly during setup. |
| XBOM calendar adapter | Regular Indian exchange sessions and holidays through the installed calendar's supported range. |

OHLCV validation requires a single instrument and numeric `Open`, `High`, `Low`, `Close`, and `Volume` columns. It normalizes timezone-aware timestamps, removes invalid timestamps and duplicate rows, sorts history, drops nonfinite/invalid candles, requires positive prices and nonnegative volume, and checks high/low bounds. It does not interpolate, backfill, or manufacture candles.

Indian times use `Asia/Kolkata`. Daily candles are eligible after their exchange close; intraday candles are start-stamped and eligible only after their full interval finishes within a session. Missing regular sessions invalidate the affected daily next-session label. Quotes retain their provider timestamp and can be marked live, stale, market closed, unavailable, or unverified; a historical quote is not presented as a current tick.

Regular NSE/BSE sessions use the installed XBOM adapter rather than the package's outdated NSE/BSE alias. Its verified support ends in **2026**. NSE and BSE regular calendars are treated as shared; exchange-specific exceptions and Muhurat sessions are not modeled. Unsupported dates fail explicitly. Calendar dependencies and exchange schedules must be revalidated before extending support.

### Request and refresh behavior

History/quote requests use a 10-second timeout. Direct news/NSE requests use separate connect/read timeouts. Optional broker client calls use a 10-second timeout, and its SDK import-time IP lookup is also bounded. Some yfinance helper APIs, such as news/options access, do not have an explicit timeout argument configured here. Refresh remains synchronous and can depend on provider behavior.

| UI cache | Lifetime |
| --- | --- |
| Daily history | 600 seconds |
| Intraday history | 30 seconds |
| Quote | 60 seconds |
| News sentiment | 300 seconds |
| Institutional flows | 600 seconds |
| Options summary | 300 seconds |

Refresh actions clear the relevant caches before fetching. Failed daily/intraday/news/flow/options requests raise rather than becoming successful cached results. Quotes carry an explicit status, including unavailable. The NSE universe is stored separately on disk and retained until explicitly refreshed; a failed refresh can return that list with degraded status.

## Features, training, and evaluation

### Predictor contract

The base model uses **34 explicitly named predictors**. Labels and target timestamps are excluded from the predictor list.

| Group | Predictors |
| --- | --- |
| Calendar | `DayOfWeek`, `Month`, `IsMonthEnd`, `Quarter`, `Is_Month_Start` |
| Return/trend | `Log_Return`, `Dist_SMA50`, `Dist_EMA20`, `RSI`, `ADX`, `ATR_Pct` |
| Volume/price position | `Dist_VWAP`, `RVol`, `Volume_Available`, `Squeeze_On`, `BB_Pct` |
| Candle patterns | `Doji`, `Hammer` |
| Lags | `Log_Return`, `RSI`, `Dist_VWAP`, and `RVol`, each lagged by 1, 2, 3, and 5 bars |

Daily volume-weighted price uses a trailing 20-bar window; intraday VWAP resets each session. Zero volume produces an explicit availability feature and neutral volume-derived model inputs. The scanner shows unavailable VWAP when it cannot be computed. Feature calculation itself makes no external requests.

Optional macros add each source's level, percentage change, 5-day mean, and 20-day mean: **32 additional predictors** for a total of **66**.

| Macro | Yahoo symbol |
| --- | --- |
| S&P 500 | `^GSPC` |
| NASDAQ Composite | `^IXIC` |
| USD/INR | `INR=X` |
| Gold futures | `GC=F` |
| Crude oil futures | `CL=F` |
| India VIX | `^INDIAVIX` |
| US 10-year Treasury yield | `^TNX` |
| US Dollar Index | `DX-Y.NYB` |

Macro data is fetched separately and aligned by availability timestamp. Each source day's observation is treated as available at the next midnight in its source timezone, converted to UTC. Backward availability joins have a seven-day tolerance; source alignment carries values forward between observations. This is a conservative daily-data convention, not a complete archive of historical publication times or revisions. Missing required macro inputs stop prediction instead of becoming zero. FII/DII flows are informational and are not fabricated as historical training features.

### Targets and fitting

- **Daily target:** `log(next_session_adjusted_close / current_adjusted_close)`.
- **Intraday targets:** same-session log returns over 1, 3, 6, and 12 five-minute bars. Exact elapsed time is required, and labels cannot cross a session boundary or a missing-bar gap.
- At least **180 fully labeled feature rows** are required after feature warm-up and validation. This requires more than 180 raw candles.
- The final chronological holdout contains `min(100, max(30, labeled_rows // 5))` rows. The daily split leaves a one-row embargo; intraday horizons use `horizon_bars + 1`. Training outcome timestamps must precede the first holdout observation.
- The saved estimator remains fitted to the training window and is not refitted on the holdout. Metrics include log-return MAE, a zero-return baseline MAE, directional accuracy, and row counts.
- XGBoost settings are fixed: 180 trees, depth 3, learning rate 0.04, subsample 0.8, column subsample 0.9, L2 regularization 5.0, seed 42, and two worker threads. There is no automatic hyperparameter search or ensemble selection.

### Recorded model result

The **8 October 2026** verification fetched 4,120 completed HINDCOPPER daily candles through **7 October 2026**, trained the daily model, generated a forecast, and completed a held-out replay:

| Metric | Recorded value |
| --- | --- |
| Training rows | 3,962 |
| Holdout rows | 100 |
| Model MAE, log-return units | 0.01736 |
| Zero-return baseline MAE, same units | 0.01703 |
| Directional accuracy | 54% |
| Technical backtest trades at the fixed BUY threshold | 0 |

These are historical verification results, not current forecasts. The model did **not** beat the zero-return baseline on MAE. Working evaluation does not establish predictive or trading advantage.

## Forecast and strategy rules

The estimated next-session close is `latest_completed_adjusted_close * exp(predicted_log_return)`. The forecast service rejects stale daily history and advances to the next exchange trading session rather than adding one calendar day. A current unadjusted minute quote is shown separately and does not replace the adjusted model input.

The chart's lower/upper estimates apply the holdout's 90th percentile absolute log-return error around the prediction. This is an **empirical residual band**, not a calibrated 90% probability interval. Its future coverage is unverified. Optional GARCH volatility is stored separately and does not determine that chart band.

| Rule | Result |
| --- | --- |
| Predicted simple return strictly greater than +1% | `BUY`, subject to the news veto |
| Predicted simple return strictly below −1% | `SELL` |
| Return between those thresholds, including endpoints | `NEUTRAL` |
| BUY with enabled news score below −0.2 | `AVOID (News Risk)` |
| BUY with enabled news unavailable | `WAIT (News Unavailable)` |

Simple predicted return is `expm1(predicted_log_return)`. News uses dated articles from the last seven days, excludes future/undated articles, deduplicates titles, and weights sentiment by `1 / (age_in_days + 1)`. Its displayed Positive/Negative labels use ±0.15 thresholds; the BUY veto has its separate −0.2 threshold. Sentiment is a headline heuristic, not a trained financial-language model.

### Backtest execution

Signals are formed after a completed daily close. A BUY enters at the next session's open and exits at its close; SELL/NEUTRAL remain in cash. The simulation is long-only, has no overnight strategy position, and does not model short selling, leverage, partial fills, or market impact.

One-way fees/slippage default to **5 basis points** and may be set between 0 and 1,000. A basis point is 0.01%. For one-way rate `c = cost_bps / 10000`, a trade's net return is `(exit / entry) * (1 - c) / (1 + c) - 1`. Equity compounds these returns. Buy-and-hold includes overnight exposure and its entry/exit costs, so its exposure differs from the strategy.

Only known outcomes after the saved training-label cutoff are eligible. The service reports up to the latest 100 eligible observations, strategy/buy-and-hold returns, trade count, observation count, and net win rate. A zero-trade result has no win rate. Later refreshed data can extend evaluation beyond the artifact's original holdout; the window is not permanently fixed. Historical point-in-time sentiment is required for a news-filtered replay and is not bundled.

### Intraday scanner rules

The opening range covers start-stamped candles in **[09:15, 09:30) IST**. All required opening bars must be present and complete; the 09:30 bar is excluded from the range. The scanner compares the latest completed close with that range to show an upside breakout, downside breakout, or inside-range state. VWAP uses current-session typical price and volume; deviations above +2% or below −2% receive descriptive labels.

It requires today's exchange session and withholds results when the latest completed candle ends more than ten minutes behind the current/session-close reference. On holidays or before the range completes, it waits. These are descriptive scanner states, not orders or intraday model predictions.

## Model artifacts and local storage

Daily and intraday models are saved as native-XGBoost JSON inside `models/`, using a readable canonical label plus a 12-character SHA-256 symbol suffix:

```text
models/HINDCOPPER_NS-3c218d94e142_daily.json
models/HINDCOPPER_NS-3c218d94e142_intraday.json
```

The intraday filename illustrates the separate artifact produced by programmatic training; it is not supplied by default. Filenames preserve exchange identity and resist collisions between differently punctuated symbols. Path validation confines writes to the selected model directory.

Artifacts contain the canonical symbol, kind, creation timestamp, feature/artifact versions, exact predictor order, feature configuration, price basis, target definition, native booster data, and dependency versions. Daily artifacts include training/holdout dates, the actual training-outcome cutoff, metrics, residual band, and optional volatility estimate. Intraday artifacts contain four boosters with per-horizon evaluation metadata.

Writes use a temporary file and atomic replacement. Loading checks identity, version, schema, configuration, and required provenance, with a 50 MB size limit. Prediction requires every stored predictor to be finite. A changed feature contract requires retraining. Legacy `.pkl` and `.keras` files are preserved locally but never loaded by the current pipeline. Keep generated artifacts local and load only trusted artifacts.

| Local path | Contents |
| --- | --- |
| `models/` | Generated JSON models and preserved legacy artifacts. |
| `.cache/nse_universe.json` | Last-known-good NSE list and retrieval timestamp. |
| `nltk_data/` | Explicitly downloaded VADER data. |
| `.cache/pytest-*`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/` | Generated development caches. |
| `logs/` | Preserved sanitized historical logs; not a required training input. |
| `.venv-fixed/` | Repaired local Python environment. |

These generated paths are ignored by Git. A clean setup should train its own artifacts rather than expect a bundled model or dataset. No database is required. Streamlit result state is in memory and is lost when its session/server restarts.

## Python service usage

Run Python examples with the repaired environment from the project root. These examples make live provider requests and training writes local artifacts.

```python
from model_trainer import load_model, train_model
from forecast import daily_forecast, run_backtest

symbol = "HINDCOPPER.NS"
artifact_path = train_model(symbol, use_macros=False)
payload = load_model(symbol)
forecast = daily_forecast(symbol, payload, use_sentiment=False)
print(artifact_path, forecast["forecast_date"], forecast["target"], forecast["signal"])

rows, metrics = run_backtest(symbol, payload, cost_bps=5, use_sentiment=False)
print(metrics)
```

Programmatic intraday training/prediction is separate from the scanner:

```python
from model_trainer import load_model, predict_frame, prepare_features, train_intraday_model

symbol = "HINDCOPPER.NS"
train_intraday_model(symbol)
payload = load_model(symbol, kind="intraday")
frame, columns = prepare_features(symbol, payload["feature_config"])
prediction = predict_frame(payload, frame.iloc[-1:], horizon="15m")
print(prediction)
```

Both training functions also accept a validated OHLCV `data` frame and a `directory` for alternate artifact storage. `load_model` accepts that same directory. Supplying daily data explicitly marked raw is rejected. The UI does not provide an intraday-model training/prediction screen or a public HTTP API.

The explicit diagnostic command fetches public history and exits nonzero on invalid/incomplete inputs:

```powershell
.\.venv-fixed\Scripts\python.exe check_data.py HINDCOPPER.NS
# Match the saved feature configuration for a macro-enabled model:
.\.venv-fixed\Scripts\python.exe check_data.py HINDCOPPER.NS --macros
```

It checks the latest predictors, minimum labeled history, unknown latest outcome, and a saved model's schema if present. It does not replace evaluation or verify every external feed.

## Optional integrations and configuration

### Angel One broker

```powershell
.\.venv-fixed\Scripts\python.exe -m pip install -r requirements-broker.txt -c requirements.lock.txt
```

Provide these variables in the process environment before starting Streamlit:

| Variable | Meaning |
| --- | --- |
| `ANGEL_API_KEY` | SmartAPI application key |
| `ANGEL_CLIENT_ID` | Broker client code |
| `ANGEL_PIN` | Broker login PIN |
| `ANGEL_TOTP_SECRET` | TOTP seed used to generate the login code |

`.env.example` lists names only. `.env` files and Streamlit secrets are not automatically loaded by this adapter. Use your local secret-management method to populate process variables; do not commit credentials or include them in issue reports. Restart Streamlit after changing its inherited environment or installing optional dependencies, then select **Connect optional broker**.

Importing the project never logs in. Successful connection enables read-only symbol resolution, history, and LTP methods. Broker history is used for supported intraday requests; adjusted daily training remains on Yahoo data. Client calls and the SDK import-time IP lookup are bounded, and SDK authentication logging is suppressed because payloads may contain credentials. Live account authentication has not been tested.

Historical credentials were removed from source and the old log was sanitized. **Rotate the previously exposed API credentials, login secret, and TOTP seed in the broker account.** Source cleanup does not invalidate exposed values. Git metadata was unavailable during the repair, so published copies/history could not be cleaned here.

### Optional GARCH

```powershell
.\.venv-fixed\Scripts\python.exe -m pip install -r requirements-volatility.txt -c requirements.lock.txt
```

With at least 100 finite, nonconstant returns, the helper attempts a GARCH(1,1) fit. Missing `arch`, constructor/fit errors, nonconvergence, or invalid output fall back to sample standard deviation. Short samples use standard deviation directly; fewer than two usable values return unavailable. Its output is in return units and is not the UI's empirical forecast error band.

## Development and testing

```powershell
.\.venv-fixed\Scripts\python.exe -m pip install -r requirements-dev.txt -c requirements.lock.txt
.\.venv-fixed\Scripts\python.exe -m pip check
.\.venv-fixed\Scripts\ruff.exe check .
.\.venv-fixed\Scripts\python.exe -m pytest -q --basetemp=.cache/pytest-local
```

The regression suite blocks external provider requests and uses deterministic fixtures, while fitting real XGBoost models. Coverage includes symbol/path safety, candle cleaning, causal feature invariance, macro availability, completed sessions, missing-bar labels, intraday horizon embargoes, native model round trips, broker timeout/logging behavior, news parsing, flows, quotes/options failures, backtest costs/compounding, and Streamlit startup/training/forecast/backtest/options interactions.

The last recorded full verification on **8 October 2026** passed **41 tests**, Ruff, and `pip check`. Live public HINDCOPPER training/forecast/backtesting also succeeded then. This documentation update does not establish current feed availability or rerun those live checks.

In a restrictive Windows sandbox, use an existing writable project temp folder:

```powershell
New-Item -ItemType Directory -Force .cache\temp | Out-Null
$env:TEMP = Join-Path (Get-Location) '.cache\temp'
$env:TMP = $env:TEMP
.\.venv-fixed\Scripts\python.exe -m pytest -q --basetemp=.cache/pytest-local
```

Pytest clears its selected base-temp directory between runs; use a dedicated generated test directory. `.github/workflows/tests.yml` configures Ubuntu/Python 3.12 checks for pushes and pull requests with read-only repository permissions. The workflow has not been verified remotely.

## Troubleshooting

| Symptom | Action or explanation |
| --- | --- |
| Python/Streamlit is missing or the old environment fails | Use `.venv-fixed` explicitly, or create it with Python 3.12 and install the pinned runtime requirements. |
| No compatible model / pipeline version changed | Train the selected full symbol again. Legacy pickle/Keras artifacts cannot be reused. |
| Insufficient fully labeled rows | The model needs 180 valid labeled feature rows after rolling-window warm-up. Thin/new listings and missing macro history may not qualify. |
| Daily history is stale or incomplete | Check the latest completed trading session and provider response, then refresh. An incomplete current-day candle is intentionally excluded. |
| Model inputs are missing/nonfinite | Check candle validity and required macro feeds. Retrain after schema/configuration changes; do not fill missing predictors with zero. |
| Forecast is WAIT because news is unavailable | Provision `vader_lexicon` locally and check feed access, or deliberately disable the news veto for a technical-only result. |
| Backtest button is disabled | Disable **Use news veto**; historical point-in-time news is not provisioned. |
| Backtest shows no trades | Predictions did not exceed the fixed +1% BUY threshold. No trades is a valid result, with no win rate. |
| Scanner waits or VWAP is unavailable | It needs a current session, complete opening bars, recent completed candles, and usable volume for VWAP. Holidays/before 09:30 are expected waiting states. |
| NSE list/flows or options are unavailable | Retry later or use a custom symbol. Provider access/schema/coverage can fail; cached lists and explicit unavailable states are intentional. |
| Optional broker cannot connect | Install optional dependencies, restart the server, check environment variables and account/TOTP setup. No credentials are needed for the public-data workflow. |
| Calendar rejects a date | Installed calendar support ends in 2026; update and validate the calendar rather than replacing holidays with generic weekdays. |
| Windows tests fail creating temporary files | Set `TEMP`/`TMP` to the writable project folder shown above and use a dedicated pytest base-temp directory. |

## Known limitations and improvement areas

- **Predictive quality:** the recorded daily model underperformed its zero-return MAE baseline. Feature/parameter experiments need independent future evaluation; the current holdout must not become a repeatedly tuned benchmark.
- **Calendar support:** validate future years, exchange-specific differences, and exceptional sessions before extending beyond the verified regular-session calendar.
- **Historical availability:** archived point-in-time news, publication timestamps/revisions for macros, and historical institutional data would be needed for richer causal research. Live headlines are not historical training labels.
- **Data coverage and reliability:** other tickers and optional macro feeds have not been exhaustively tested. Public provider delays, throttling, schema changes, and option-chain gaps remain possible. Synchronous refresh can wait on provider APIs.
- **Execution realism:** backtesting uses adjusted OHLC prices and simplified one-way costs, without detailed spread/liquidity, impact, financing, taxes by trade, or fill simulation. A next-close target and next-open strategy also measure different return intervals.
- **Broker verification:** credential rotation and live account checks remain external actions. No broker login or trading was performed during the repair.
- **Deployment:** this is a local single-user app without application authentication, per-user isolation, durable job scheduling, or a multi-user persistence layer. Keep the documented loopback binding for local use. Public deployment needs additional design and testing.
- **Portability and CI:** the repaired installation was verified on Windows/Python 3.12. Other platforms, remote CI, and Python 3.13 require verification. Models are not automatically scheduled for retraining.
- **Licensing:** no project license file is present. Dependency and provider terms apply independently; no project redistribution license is declared here.

For the original defects, repair decisions, and verification boundaries, see [the audit](AUDIT_REPORT.md) and [the fix status](FIX_STATUS.md).
