"""Timestamped news parsing and optional locally provisioned VADER scoring."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
import xml.etree.ElementTree as ET

import nltk
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from symbols import ROOT, canonical_symbol

log = logging.getLogger(__name__)
if str(ROOT / "nltk_data") not in nltk.data.path:
    nltk.data.path.append(str(ROOT / "nltk_data"))


def _timestamp(value):
    if isinstance(value, (int, float)):
        return pd.Timestamp(datetime.fromtimestamp(value, tz=timezone.utc))
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC") if stamp.tzinfo is not None else None


def parse_yahoo_news(items):
    records = []
    for item in items:
        content = item.get("content") or item
        title = str(content.get("title") or "").strip()
        value = content.get("pubDate") or content.get("providerPublishTime")
        try:
            stamp = _timestamp(value) if value is not None else None
        except (ValueError, TypeError, OverflowError):
            stamp = None
        if title and stamp is not None and not pd.isna(stamp):
            summary = BeautifulSoup(str(content.get("summary") or ""), "html.parser").get_text(" ", strip=True)
            records.append(
                {"PublishedAt": stamp, "Title": title, "Text": f"{title}. {summary}", "Source": "Yahoo Finance"}
            )
    return records


class FinancialSentimentAnalyzer:
    def __init__(self, vader=None):
        # Never download data at import or during an app interaction.
        self.vader = vader if vader is not None else SentimentIntensityAnalyzer()

    def _fetch_google_news_rss(self, ticker):
        name = {"^NSEI": "Nifty 50", "^NSEBANK": "Nifty Bank", "^BSESN": "Sensex"}.get(ticker, ticker.rsplit(".", 1)[0])
        try:
            response = requests.get(
                "https://news.google.com/rss/search",
                params={
                    "q": f"{name} stock share price financial news when:7d",
                    "hl": "en-IN",
                    "gl": "IN",
                    "ceid": "IN:en",
                },
                timeout=(3.05, 8),
            )
            response.raise_for_status()
            records = []
            for item in ET.fromstring(response.content).findall("./channel/item")[:15]:
                title = (item.findtext("title") or "").strip()
                try:
                    stamp = _timestamp(parsedate_to_datetime(item.findtext("pubDate") or ""))
                except (ValueError, TypeError):
                    continue
                if not title or stamp is None:
                    continue
                description = BeautifulSoup(item.findtext("description") or "", "html.parser").get_text(" ", strip=True)
                records.append(
                    {"PublishedAt": stamp, "Title": title, "Text": f"{title}. {description}", "Source": "Google News"}
                )
            return records
        except Exception as exc:
            log.warning("Google news unavailable (%s).", type(exc).__name__)
            return []

    def _fetch_yfinance_news(self, ticker):
        try:
            return parse_yahoo_news(yf.Ticker(ticker).get_news(count=10))
        except Exception as exc:
            log.warning("Yahoo news unavailable (%s).", type(exc).__name__)
            return []

    def analyze_ticker(self, ticker, now=None):
        ticker = canonical_symbol(ticker)
        records = self._fetch_google_news_rss(ticker) or self._fetch_yfinance_news(ticker)
        if not records:
            return None
        current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
        frame = pd.DataFrame(records).sort_values("PublishedAt", ascending=False)
        frame = frame.loc[(frame.PublishedAt <= current) & (frame.PublishedAt >= current - pd.Timedelta(days=7))]
        frame = frame.drop_duplicates("Title")
        if frame.empty:
            return None
        frame["Sentiment"] = frame.Text.map(lambda text: self.vader.polarity_scores(text)["compound"])
        age_days = (current - frame.PublishedAt).dt.total_seconds() / 86400
        weights = 1 / (age_days + 1)
        score = float((frame.Sentiment * weights).sum() / weights.sum())
        label = "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral"
        return {
            "score": score,
            "label": label,
            "headline_count": len(frame),
            "top_headline": frame.Title.iloc[0],
            "timestamp": frame.PublishedAt.iloc[0],
            "raw_df": frame[["PublishedAt", "Title", "Sentiment", "Source"]],
        }


def get_news_sentiment(ticker, now=None):
    try:
        return FinancialSentimentAnalyzer().analyze_ticker(ticker, now)
    except LookupError:
        log.warning("News scoring unavailable: provision the local VADER lexicon using the README setup command.")
        return None
