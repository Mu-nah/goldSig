import os
import time
import requests
import pandas as pd
import numpy as np
import feedparser
import torch
import asyncio
from telegram import Bot
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

# ──────────────────────────────
# CONFIG
# ──────────────────────────────
load_dotenv()

SYMBOL = "XAU/USD"
API_KEYS = os.getenv("TD_API_KEYS", "").split(",")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STDDEV = 2
MIN_PIP_DISTANCE = 1.0
SLEEP_SECS = 900  # 15 minutes

bot = Bot(token=TELEGRAM_TOKEN)

# ──────────────────────────────
# FINBERT SENTIMENT SETUP
# ──────────────────────────────
os.environ["HF_HOME"] = "/tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache"

labels = ["Positive", "Negative", "Neutral"]
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# ──────────────────────────────
# DATA FETCH
# ──────────────────────────────
def fetch_data(interval, limit=100):
    base_url = "https://api.twelvedata.com/time_series"
    for key in API_KEYS:
        url = f"{base_url}?symbol={SYMBOL}&interval={interval}&outputsize={limit}&apikey={key.strip()}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if "values" in data:
                    df = pd.DataFrame(data["values"])
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.sort_values("datetime")
                    df = df.astype({"open": float, "high": float, "low": float, "close": float})
                    return df
        except Exception as e:
            continue
    return None

# ──────────────────────────────
# INDICATORS
# ──────────────────────────────
def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bb(series, period=BB_PERIOD, stddev=BB_STDDEV):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + (stddev * std)
    lower = ma - (stddev * std)
    return ma, upper, lower

def generate_signal(df_1h, df_1d):
    df_1h["rsi"] = compute_rsi(df_1h["close"])
    ma, upper, lower = compute_bb(df_1h["close"])
    df_1h["bb_upper"] = upper
    df_1h["bb_lower"] = lower

    latest = df_1h.iloc[-1]
    prev = df_1h.iloc[-2]

    if prev["close"] < prev["bb_lower"] and latest["close"] > latest["bb_lower"] and latest["rsi"] < 30:
        return "BUY", latest
    elif prev["close"] > prev["bb_upper"] and latest["close"] < latest["bb_upper"] and latest["rsi"] > 70:
        return "SELL", latest
    return None, None

# ──────────────────────────────
# NEWS SENTIMENT
# ──────────────────────────────
def fetch_news(query, num_articles=10):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries[:num_articles]]

def finbert_sentiment(text):
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    return dict(zip(labels, probs))

def analyze_sentiment_for_gold():
    titles = fetch_news("gold market", 15)
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for title in titles:
        scores = finbert_sentiment(title)
        dominant = max(scores, key=scores.get)
        summary[dominant] += 1
    total = sum(summary.values())
    pos_pct = (summary["Positive"] / total) * 100 if total else 0
    neg_pct = (summary["Negative"] / total) * 100 if total else 0
    return pos_pct, neg_pct

# ──────────────────────────────
# TELEGRAM ALERT
# ──────────────────────────────
def send_alert(msg):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    except Exception as e:
        print(f"Failed to send alert: {e}")

# ──────────────────────────────
# MAIN LOOP
# ──────────────────────────────
WAT = timezone(timedelta(hours=1))  # UTC+1

def main():
    last_signal = None
    last_forced_alert_date = None  # Track 1AM alert

    while True:
        now_wat = datetime.now(WAT)
        df_1h = fetch_data("1h", 100)
        df_1d = fetch_data("1day", 50)

        # ────────── Strategy alert ──────────
        if df_1h is not None and df_1d is not None:
            signal, last = generate_signal(df_1h, df_1d)
            if signal and signal != last_signal:
                pos, neg = analyze_sentiment_for_gold()
                if (signal == "BUY" and pos >= 50) or (signal == "SELL" and neg >= 50):
                    msg = (
                        f"📈 Gold Signal Confirmed ({signal})\n"
                        f"Time: {last['datetime']}\n"
                        f"Close: ${last['close']:.2f}\n"
                        f"RSI: {last['rsi']:.2f}\n"
                        f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}%"
                    )
                    send_alert(msg)
                    last_signal = signal

        # ────────── Forced 1AM WAT alert ──────────
        if now_wat.hour == 1 and now_wat.weekday() < 5:
            if last_forced_alert_date != now_wat.date():
                msg = f"⏰ Gold 1AM WAT Status Alert\nTime: {now_wat}"
                send_alert(msg)
                last_forced_alert_date = now_wat.date()

        time.sleep(SLEEP_SECS)

if __name__ == "__main__":
    main()
