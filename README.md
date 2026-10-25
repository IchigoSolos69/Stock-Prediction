# Stock Forecast Terminal

A local Streamlit terminal for Indian daily forecasts, completed-candle opening-range scanning, options summaries where the public provider has coverage, and chronological strategy evaluation. The October 2026 repairs are mapped to the original audit in [FIX_STATUS.md](FIX_STATUS.md).

## Run the repaired local installation

A working Python 3.12 environment has been created as `.venv-fixed`. From this project directory in PowerShell:

```powershell
.\.venv-fixed\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

The old `.venv` is broken and is preserved rather than used. The app is intended for one local user. It has no account-management or public-deployment authentication layer.

The default HINDCOPPER.NS daily model was rebuilt from public adjusted history through 7 October 2026. Other assets need **Train / retrain daily model**. Use **Refresh forecast** to fetch current completed candles. The news toggle controls a live BUY veto, not the training features. If news is enabled but unavailable, BUY is withheld. Backtesting with the news veto requires historical point-in-time news, which is not provisioned; disable the veto for the technical backtest.

## Clean setup

Use Python 3.12, the version exercised by the regression suite. On a machine with a standard Python installation:

```powershell
python -m venv .venv-fixed
.\.venv-fixed\Scripts\python.exe -m pip install -r requirements.txt -c requirements.lock.txt
.\.venv-fixed\Scripts\python.exe -m nltk.downloader -d nltk_data vader_lexicon
.\.venv-fixed\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

`requirements.txt` pins direct runtime dependencies. `requirements.lock.txt` constrains the complete installed runtime/development dependency graph; using it with `-c` installs only dependencies needed by the selected requirements file. The lock was resolved on Windows/Python 3.12. Refresh and test it when changing supported platforms or Python versions. VADER is now provisioned locally in `nltk_data`; importing/running the app never downloads corpora.

Development checks:

```powershell
.\.venv-fixed\Scripts\python.exe -m pip install -r requirements-dev.txt -c requirements.lock.txt
.\.venv-fixed\Scripts\python.exe -m pip check
.\.venv-fixed\Scripts\ruff.exe check .
.\.venv-fixed\Scripts\python.exe -m pytest -q --basetemp=.cache/pytest-local
```

The tests block external providers and use deterministic fixtures. They also train real XGBoost models, round-trip native artifacts, and exercise Streamlit training, forecast, options, and backtest flows. In restrictive Windows sandboxes, set `TEMP` and `TMP` to an existing project `.cache/temp` folder before testing. The pytest base-temp directory is generated test storage and is cleared by pytest between runs.

An offline diagnostic command is import-safe; running the actual diagnostic fetches public history and exits nonzero on invalid inputs:

```powershell
.\.venv-fixed\Scripts\python.exe check_data.py HINDCOPPER.NS
```

## Data and model behavior

- Daily forecasting and ORB currently support `.NS`, `.BO`, and Indian index symbols such as `^NSEI`. Supply the complete Yahoo symbol; symbols are preserved across views, including exchange identity. Other assets can be queried through quote/options adapters, with explicit venue-freshness limitations.
- Daily history always uses adjusted Yahoo OHLCV. Raw broker candles never silently replace this training basis. Invalid candles are dropped without interpolation, backfill, or invented prices. Daily labels skip missing exchange sessions.
- Only completed exchange-session bars are used. Regular sessions use the maintained XBOM adapter, including 2025/2026 holidays, rather than the package's outdated NSE/BSE alias. NSE/BSE regular calendars are treated as shared; unusual exchange-specific or Muhurat sessions are not currently modeled. The installed XBOM calendar supports dates through 2026; dates outside its supported range fail explicitly. Update calendar dependencies and validate exchange circulars before extending it.
- ORB uses start-stamped bars in `[09:15, 09:30)`, waits for all opening candles to complete, requires current-session data, and withholds stale signals. VWAP resets each intraday session. Missing volume produces an unavailable scanner metric; the model uses explicit volume-availability indicators alongside neutral volume inputs.
- Feature engineering is deterministic and fetches no external data. Optional macro features are fetched separately; each source day's values become available conservatively at the next local midnight. Joins use UTC availability timestamps. Missing required macro inputs stop prediction instead of becoming zero. Historical FII/DII is not fabricated or used as constant model inputs.
- Training uses fixed XGBoost parameters and a final chronological holdout of up to 100 observations, with an embargo to prevent training outcomes reaching the holdout. The saved estimator stays fitted to the training window; it is not refitted on the holdout. The incomplete stacked/voting/tuning prototypes were replaced by this executable baseline.
- Artifacts are atomic native-XGBoost JSON files with schema version, canonical asset identity, feature configuration, training outcome cutoff, evaluation dates, error metrics, and dependency versions. Legacy pickle/Keras artifacts remain on disk but are not loaded. Retraining is required after a feature-version change. Each stored predictor must be present and finite; there is no zero-filling of missing model columns.
- The forecast targets the next session's adjusted close relative to the latest completed adjusted close. Current minute quotes are displayed separately with provider/time/status. The residual band is an empirical held-out error band, not a calibrated probability or a GARCH confidence cone.
- The shared technical rule buys when predicted simple return exceeds 1%; SELL/NEUTRAL remain in cash in the long-only simulation. After a completed-close signal, the backtest buys at the next open and exits at the next close. Fees/slippage are charged on each side; buy-and-hold also includes its entry/exit costs. Equity uses compounded simple returns. Unknown outcomes and dates at/before the training-label cutoff are excluded. It reports up to the latest 100 eligible observations, trade count, and a missing win rate when there are no trades.
- Feed failures are explicit. NSE flows retain their source date; missing values are not zero/Neutral. Stock-list outages preserve the last-known-good cache. News supports flat/nested Yahoo records, encoded symbol queries, UTC dates, recent-article filtering and deduplication. Options with insufficient open interest are unavailable; PCR is descriptive rather than a validated directional trading rule. Yahoo options coverage for NSE/BSE is limited and this app does not promise a working chain for every Indian asset.

## Optional broker and GARCH setup

The public-data app runs without broker credentials, TensorFlow, LightGBM, Optuna, pandas-ta or arch. Install the optional adapter only when needed:

```powershell
.\.venv-fixed\Scripts\python.exe -m pip install -r requirements-broker.txt -c requirements.lock.txt
```

Set `ANGEL_API_KEY`, `ANGEL_CLIENT_ID`, `ANGEL_PIN` and `ANGEL_TOTP_SECRET` as process environment variables using your newly rotated credentials, then choose **Connect optional broker**. `.env.example` lists the variable names; `.env` files are not automatically loaded. Login occurs only on explicit connection. The adapter exposes symbol resolution, historical candles and LTP; it exposes no order-placement methods. SDK request logging is suppressed because it can include authentication payloads. Client calls and the SDK's import-time IP lookup have timeouts. Login/history interfaces follow the [official SmartAPI SDK](https://github.com/angel-one/smartapi-python). Live account authentication was not tested.

GARCH is optional:

```powershell
.\.venv-fixed\Scripts\python.exe -m pip install -r requirements-volatility.txt -c requirements.lock.txt
```

Its helper validates finite returns, handles construction/fit failures and falls back to sample standard deviation. The UI's displayed error band is evaluated separately from this helper.

## Validation and remaining work

The rebuilt HINDCOPPER model's 100-row holdout MAE is **0.01736**, versus **0.01703** for predicting zero return; directional accuracy is **54%**. Its technical backtest generated **zero trades** at the fixed 1% BUY threshold. These results establish working evaluation, not an improved predictive edge. More data/feature experimentation needs independent future validation.

Broker secrets were removed from source and the historical authentication log was sanitized. **Credential and TOTP rotation still must be completed in the broker account.** No Git metadata was present, so previously published files or repository history could not be cleaned here. Provider coverage/access, live broker authentication, unusual exchange sessions, historical news data and public multi-user deployment remain outside the verified local setup. The CI workflow is added but has not run remotely.
