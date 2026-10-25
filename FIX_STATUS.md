# Audit fix status — 8 October 2026

All 20 findings from the original [audit](AUDIT_REPORT.md) have implementation changes. External account actions and verification limits are recorded below. Source files were refactored, so the original audit's line references describe the pre-fix snapshot.

| Finding | Implementation |
| --- | --- |
| 1. Exposed credentials | Environment-based configuration, sanitized historical log, removed old broker bytecode, ignored secrets/logs. Broker-side rotation and any published history cleanup remain required. |
| 2. Broken broker imports/interface | One disconnected read-only adapter, optional SDK, explicit login, symbol lookup, bounded SDK calls, suppressed auth logging. Live broker authentication remains unverified. |
| 3. Missing/broken trainers | Shared daily/intraday training, feature contract and artifact API; incomplete experimental ensemble/tuning paths removed. |
| 4. Broken environment/dependencies | Working `.venv-fixed`, Python requirements, full resolved constraints, separated optional dependencies, README and CI. |
| 5. Feature mismatch | Versioned native JSON; required columns/finite values checked; legacy artifacts refused; real HINDCOPPER daily model rebuilt. |
| 6. Future label as input | Explicit predictor allowlist excludes labels; regression verifies prefix-invariant predictors. |
| 7. Incorrect backtest | Shared threshold, chronological cutoff, next-open/close execution, costs, correct compounding, unknown outcomes excluded; historical news is required to enable a news-filtered replay. |
| 8. Future interpolation | Invalid candles dropped; no interpolation/backfill; daily labels skip missing sessions. |
| 9. Unsafe model paths | Canonical validation, preserved exchange, confined hashed filenames and native JSON loading. |
| 10. Unbounded NSE requests | Connect/read timeouts, explicit errors, short successful-result caching and on-demand view work. |
| 11. Wrong opening range | Exclusive endpoint, completed opening bars and current-session/freshness gates. |
| 12. Stale live views | Quote timestamp/source/status, incomplete-bar removal and explicit/minute scanner refresh. |
| 13. Bad intraday timestamps | Timezone normalization, duplicate/NaT removal and candle validation. |
| 14. Zero-volume faults | Unavailable scanner VWAP; explicit model availability feature; no infinity or dropped entire zero-volume dataset. |
| 15. Overnight minute labels | Same-session targets, exact elapsed-time checks and horizon embargo. |
| 16. Macro/CV leakage | Availability-time macro joins; pure features; chronological held-out baseline instead of ordinary-CV prototypes. |
| 17. Invented FII/DII values/dates | Source dates retained, categories validated, stale/unavailable states, no manufactured neutral flows. |
| 18. News parsing/controls | Encoded queries, both Yahoo formats, real timestamps, deduplication, no runtime corpus download, toggle-controlled veto. |
| 19. Invalid options/quote failures | Activity/schema checks, undefined ratios unavailable, descriptive PCR and missing-quote guard. Public provider coverage remains limited. |
| 20. Risk/units/calendar faults | Removed unused false confidence helper; optional GARCH fallback fixed; empirical residual band, trading-day forecast dates and correctly labeled percentage metrics. Calendar adapter support must be renewed beyond 2026. |

Additional changes: presentation/services are separated; successful UI results survive reruns and are keyed by asset/model/settings; model writes are atomic; caches preserve a last-known-good universe; logging uses safe messages; generated files are ignored; diagnostics fail on invalid inputs; and deterministic regression coverage plus GitHub Actions configuration were added.

Validation: **41 local regression tests passed**, including real daily/intraday XGBoost fitting, native-model round trips, causal-feature/session/label tests, provider fixtures, bounded broker SDK import requests, offline app startup and full Streamlit training/forecast/backtest/options interactions. Ruff and `pip check` passed. VADER was provisioned in the project. Live public Yahoo history returned 4,120 completed HINDCOPPER candles through 7 October 2026; model training, forecast generation and a 100-observation held-out replay succeeded.

The rebuilt model does not beat the zero-return baseline on holdout MAE (0.01736 versus 0.01703), and its fixed-threshold backtest produced zero trades. No claims of improved predictive accuracy or profitability are made. Broker-side credential rotation and live authentication remain user/account actions; no broker login or trading was performed. Git history was unavailable. Other tickers, optional macro live feeds, NSE availability and remote CI have not been exhaustively validated.
