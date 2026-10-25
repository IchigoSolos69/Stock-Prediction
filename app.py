"""Streamlit presentation; fetching, features and strategy live in shared services."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from angel_bridge import angel
from data_loader import fetch_data, fetch_intraday_data
from derivatives import analyze_pcr, get_derivatives_data
from fii_dii import fetch_fii_dii_data
from forecast import daily_forecast, run_backtest
from intraday_logic import calculate_intraday_signals
from live_price import get_quote
from market_time import now_ist
from model_trainer import load_model, train_model
from sentiment import get_news_sentiment
from stock_list import cached_stock_universe, fetch_stock_universe
from symbols import canonical_symbol, is_indian

st.set_page_config(page_title="Stock Forecast Terminal", page_icon="⚡", layout="wide")
st.markdown(
    """<style>
.stApp {background-color:#0e1117;color:#e0e0e0;}
section[data-testid="stSidebar"] {background-color:#161b22;}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600, show_spinner=False)
def cached_daily(ticker):
    data = fetch_data(ticker)
    if data.empty:
        raise ValueError("Daily data unavailable. Check the symbol or retry later.")
    return data


@st.cache_data(ttl=30, show_spinner=False)
def cached_intraday(ticker):
    data = fetch_intraday_data(ticker)
    if data.empty:
        raise ValueError("Completed intraday bars unavailable.")
    return data


@st.cache_data(ttl=60, show_spinner=False)
def cached_quote(ticker):
    return get_quote(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_news(ticker):
    result = get_news_sentiment(ticker)
    if result is None:
        raise ValueError("News sentiment unavailable.")
    return result


@st.cache_data(ttl=600, show_spinner=False)
def cached_flows():
    result = fetch_fii_dii_data()
    if result["status"] == "unavailable":
        raise ValueError("Institutional feed unavailable; no values were substituted.")
    return result


@st.cache_data(ttl=300, show_spinner=False)
def cached_options(ticker):
    result = get_derivatives_data(ticker)
    if result is None:
        raise ValueError("No usable option chain from this provider for this asset.")
    return result


def chart_layout(figure, title):
    figure.update_layout(
        title=title, template="plotly_dark", height=420, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117"
    )
    return figure


def current_model(ticker):
    try:
        return load_model(ticker)
    except FileNotFoundError:
        return None
    except (ValueError, KeyError, TypeError) as exc:
        st.error(str(exc))
        return None


st.sidebar.title("⚡ Forecast Terminal")
st.sidebar.caption("Public data provider" if angel.session is None else "Broker adapter connected")
if angel.session is None and st.sidebar.button("Connect optional broker"):
    if angel.connect():
        cached_intraday.clear()
        st.sidebar.success("Broker connected for intraday candles.")
    else:
        st.sidebar.info(
            "Broker unavailable. Check optional setup and environment configuration; public data remains available."
        )
if "universe" not in st.session_state:
    st.session_state.universe = cached_stock_universe()
if st.sidebar.button("Load / refresh NSE asset list"):
    with st.spinner("Loading asset list..."):
        st.session_state.universe = fetch_stock_universe()
universe = st.session_state.universe
if universe["status"] in {"degraded", "unavailable"}:
    st.sidebar.caption("Full NSE list unavailable; use a custom symbol or retry.")
assets = {"Nifty 50": "^NSEI", "Nifty Bank": "^NSEBANK", "Sensex": "^BSESN", **universe["stocks"]}
choice = st.sidebar.selectbox("Asset", ["Custom symbol"] + list(assets))
symbol_input = (
    st.sidebar.text_input(
        "Yahoo symbol",
        "HINDCOPPER.NS",
        help="Use .NS or .BO for Indian equities; preserve index symbols such as ^NSEI.",
    )
    if choice == "Custom symbol"
    else assets[choice]
)
try:
    ticker = canonical_symbol(symbol_input)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
use_sentiment = st.sidebar.toggle(
    "Use news veto", value=True, help="Filters BUY signals using recent news; does not train on today's news."
)
use_macros = st.sidebar.checkbox("Use macro features when training", value=False)
view = st.sidebar.radio("View", ["Daily Forecast", "Derivatives (F&O)", "Intraday Scanner", "Backtest Lab"])

st.title(ticker)

if view == "Daily Forecast":
    st.subheader("Daily forecast")
    if not is_indian(ticker):
        st.info("Daily models currently support NSE/BSE equities and Indian indices.")
    else:
        if st.button("Train / retrain daily model", type="primary"):
            try:
                with st.spinner("Training with a chronological holdout..."):
                    cached_daily.clear()
                    train_model(ticker, use_macros=use_macros, data=cached_daily(ticker))
                st.success("Model trained. Evaluation data was kept out of fitting.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
        payload = current_model(ticker)
        if payload is None:
            st.info("Train a model for this asset. Legacy model files require retraining.")
        else:
            metrics = payload["metrics"]
            st.caption(
                f"Trained through {pd.Timestamp(payload['training_label_end']).date()} · "
                f"Holdout: {metrics['holdout_rows']} rows · MAE {metrics['mae']:.5f} · "
                f"Zero-return baseline MAE {metrics['zero_return_baseline_mae']:.5f}"
            )
            key = (ticker, payload["created_at"], use_sentiment)
            if st.button("Refresh forecast"):
                try:
                    with st.spinner("Fetching completed candles and checking inputs..."):
                        cached_daily.clear()
                        cached_quote.clear()
                        cached_news.clear()
                        news = None
                        if use_sentiment:
                            try:
                                news = cached_news(ticker)
                            except ValueError:
                                st.warning("News filter unavailable; BUY signals will be withheld.")
                        result = daily_forecast(ticker, payload, cached_daily(ticker), news, use_sentiment)
                        st.session_state["forecast_result"] = (key, result, cached_quote(ticker), news)
                except Exception as exc:
                    st.session_state.pop("forecast_result", None)
                    st.error(f"Forecast unavailable: {exc}")
            saved = st.session_state.get("forecast_result")
            if saved is not None and saved[0] == key:
                _, result, quote, news = saved
                a, b, c = st.columns(3)
                a.metric("Latest completed close", f"₹{result['base']:,.2f}")
                b.metric(
                    "Next-session close estimate",
                    f"₹{result['target']:,.2f}",
                    f"{np.expm1(result['prediction']) * 100:+.2f}%",
                )
                c.metric("Signal", result["signal"])
                st.caption(f"Forecast session: {result['forecast_date']} · Completed close: {result['as_of'].date()}")
                if quote.price is not None:
                    st.caption(f"Quote ₹{quote.price:,.2f} · {quote.status} · {quote.timestamp} · {quote.source}")
                else:
                    st.caption("Current quote unavailable; estimate is based on the completed close.")
                if news is not None:
                    st.caption(f"News: {news['label']} ({news['score']:.2f}) · {news['timestamp']}")
                    st.write(news["top_headline"])
                history = result["frame"].tail(100)
                fig = go.Figure(go.Scatter(x=history.index, y=history.Close, name="Completed closes"))
                fig.add_trace(
                    go.Scatter(
                        x=[result["forecast_date"]],
                        y=[result["target"]],
                        mode="markers",
                        name="Estimate",
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=[result["high"] - result["target"]],
                            arrayminus=[result["target"] - result["low"]],
                        ),
                    )
                )
                st.plotly_chart(chart_layout(fig, "Next-session estimate"), width="stretch")
                st.caption(
                    "Band uses the 90th percentile of held-out absolute errors; future coverage is not established."
                )

elif view == "Derivatives (F&O)":
    st.subheader("Options open interest")
    if st.button("Scan option chain", type="primary"):
        try:
            cached_options.clear()
            st.session_state["options_result"] = (ticker, cached_options(ticker), get_quote(ticker))
        except Exception as exc:
            st.session_state.pop("options_result", None)
            st.info(str(exc))
    saved = st.session_state.get("options_result")
    if saved is not None and saved[0] == ticker:
        _, options, quote = saved
        a, b, c = st.columns(3)
        a.metric("Put/call open-interest ratio", f"{options['pcr_oi']:.2f}")
        a.caption(analyze_pcr(options["pcr_oi"]))
        b.metric("Minimum payout strike", f"₹{options['max_pain']:,.2f}")
        c.metric(
            "Distance from quote",
            f"{(options['max_pain'] / quote.price - 1) * 100:+.2f}%" if quote.price is not None else "Unavailable",
        )
        st.caption(f"Expiry {options['expiry']} · Retrieved {options['retrieved_at']} · {options['source']}")
        if quote.timestamp is not None:
            st.caption(f"Reference quote: {quote.status} · {quote.timestamp}")
        st.write(
            f"Largest put OI strike: {options['support_oi']:,.2f} · Largest call OI strike: {options['resistance_oi']:,.2f}"
        )

elif view == "Intraday Scanner":
    st.subheader("Completed five-minute candles")
    if not is_indian(ticker):
        st.info("Opening-range scanning currently supports Indian exchange sessions.")
    else:
        automatic = st.checkbox("Refresh every minute", value=False)

        @st.fragment(run_every=60 if automatic else None)
        def scanner():
            if st.button("Refresh scanner"):
                cached_intraday.clear()
            try:
                data = cached_intraday(ticker)
                signals = calculate_intraday_signals(data, exchange="BSE" if ticker.endswith(".BO") else "NSE")
                st.caption(
                    f"Fetched {now_ist()} · Latest completed candle {data.index[-1]} · {data.attrs.get('source', 'Provider')}"
                )
                if signals is None:
                    st.info("Waiting for a complete opening range and recent candles from today's trading session.")
                    return
                a, b, c = st.columns(3)
                a.metric("Opening range", signals["orb_status"])
                a.caption(f"{signals['orb_low']:.2f} – {signals['orb_high']:.2f}")
                b.metric(
                    "VWAP deviation", "Unavailable" if signals["vwap_dist"] is None else f"{signals['vwap_dist']:.2f}%"
                )
                c.metric(
                    "Session VWAP",
                    "Unavailable" if signals["current_vwap"] is None else f"₹{signals['current_vwap']:.2f}",
                )
                st.caption(f"{signals['vwap_signal']} · Session {signals['session_status']}")
                today = data.loc[data.index.date == signals["timestamp"].date()]
                figure = go.Figure(
                    go.Candlestick(
                        x=today.index, open=today.Open, high=today.High, low=today.Low, close=today.Close, name="Price"
                    )
                )
                figure.add_hline(y=signals["orb_high"], line_color="#f85149")
                figure.add_hline(y=signals["orb_low"], line_color="#3fb950")
                figure.add_trace(
                    go.Scatter(x=signals["vwap_series"].index, y=signals["vwap_series"], name="Session VWAP")
                )
                figure.update_layout(xaxis_rangeslider_visible=False)
                st.plotly_chart(chart_layout(figure, "Opening range and VWAP"), width="stretch")
            except Exception as exc:
                st.warning(f"Scanner unavailable: {exc}")

        scanner()

elif view == "Backtest Lab":
    st.subheader("Held-out technical strategy")
    st.caption(
        "Signals at completed close; buy at next open and exit at next close. SELL/NEUTRAL stay in cash. Both strategies include costs; buy-and-hold includes overnight exposure."
    )
    costs = st.number_input(
        "One-way fees and slippage (basis points)", min_value=0.0, max_value=1000.0, value=5.0, step=1.0
    )
    if use_sentiment:
        st.info("Historical news is unavailable. Disable the news veto to evaluate the technical strategy.")
    payload = current_model(ticker)
    if payload is None:
        st.info("Train the daily model first.")
    else:
        key = (ticker, payload["created_at"], costs, use_sentiment)
        if st.button("Run backtest on last 100 eligible sessions", disabled=use_sentiment, type="primary"):
            try:
                cached_daily.clear()
                result, metrics = run_backtest(ticker, payload, costs, use_sentiment, cached_daily(ticker))
                st.session_state["backtest_result"] = (key, result, metrics)
            except Exception as exc:
                st.session_state.pop("backtest_result", None)
                st.error(f"Backtest unavailable: {exc}")
        saved = st.session_state.get("backtest_result")
        if saved is not None and saved[0] == key:
            _, result, metrics = saved
            a, b, c = st.columns(3)
            a.metric("Buy-and-hold after costs", f"{metrics['market_return'] * 100:.2f}%")
            b.metric("Strategy return after costs", f"{metrics['strategy_return'] * 100:.2f}%")
            c.metric(
                "Net trade win rate",
                "No trades" if metrics["win_rate"] is None else f"{metrics['win_rate'] * 100:.1f}%",
            )
            st.caption(
                f"{metrics['trade_count']} trades · {metrics['observations']} observations · Training outcomes end {payload['training_label_end']}"
            )
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=result.Outcome_Time, y=result.Cum_Market * 100, name="Buy and hold"))
            figure.add_trace(
                go.Scatter(x=result.Outcome_Time, y=result.Cum_Strategy * 100, name="Strategy after costs")
            )
            figure.update_yaxes(title="Compounded return (%)")
            st.plotly_chart(chart_layout(figure, "Out-of-sample equity curves"), width="stretch")

with st.expander("Institutional flows"):
    if st.button("Load / refresh institutional flows"):
        try:
            cached_flows.clear()
            st.session_state["flow_result"] = cached_flows()
        except ValueError as exc:
            st.session_state.pop("flow_result", None)
            st.info(str(exc))
    flows = st.session_state.get("flow_result")
    if flows is not None:
        a, b, c = st.columns(3)
        a.metric("FII net flow (₹ crore)", f"{flows['fii_net']:,.2f}")
        b.metric("DII net flow (₹ crore)", f"{flows['dii_net']:,.2f}")
        c.metric("Flow classification", flows["signal"])
        st.caption(f"Source date {flows['date']} · {flows['status']} · {flows['source']}")
