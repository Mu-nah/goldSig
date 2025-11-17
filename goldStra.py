import os, time, requests, pandas as pd, numpy as np, feedparser, torch, asyncio
from telegram import Bot
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from flask import Flask
import threading

# ──────────────────────────────
# ⚙️ CONFIG
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
SLEEP_SECS = 1200  # 20 minutes

bot = Bot(token=TELEGRAM_TOKEN)

# ──────────────────────────────
# 🧠 FINBERT SENTIMENT
# ──────────────────────────────
os.environ["HF_HOME"] = "/tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache"
labels = ["Positive", "Negative", "Neutral"]

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# ──────────────────────────────
# 🔄 DATA FETCH
# ──────────────────────────────
def fetch_data(interval, limit=100):
    base_url = "https://api.twelvedata.com/time_series"
    for key in API_KEYS:
        url = f"{base_url}?symbol={SYMBOL}&interval={interval}&outputsize={limit}&apikey={key.strip()}"
        try:
            print(f"📡 Fetching {interval} candles using key {key[:6]}...")
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if "values" in data:
                    df = pd.DataFrame(data["values"])
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.sort_values("datetime")
                    df = df.astype({"open": float, "high": float, "low": float, "close": float})
                    return df
            print(f"⚠️ Key {key[:6]} failed: {r.text[:80]}")
        except Exception as e:
            print(f"❌ Error using key {key[:6]} -> {e}")
    print("🚫 All TwelveData keys failed.")
    return None

# ──────────────────────────────
# 📊 INDICATORS
# ──────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower

# ──────────────────────────────
# 📊 STRATEGY
# ──────────────────────────────
def generate_signal(df_1h, df_1d):
    df_1h["rsi"] = rsi(df_1h["close"], RSI_PERIOD)
    df_1h["bb_upper"], df_1h["bb_mid"], df_1h["bb_lower"] = bollinger_bands(df_1h["close"], BB_PERIOD, BB_STDDEV)
    df_1d["bb_upper"], _, df_1d["bb_lower"] = bollinger_bands(df_1d["close"], BB_PERIOD, BB_STDDEV)

    last1h, last1d = df_1h.iloc[-1], df_1d.iloc[-1]
    direction = "BUY" if last1h["close"] > last1h["open"] else "SELL"

    trend = (direction == "BUY" and last1h["close"] > last1h["bb_mid"] + MIN_PIP_DISTANCE) or \
            (direction == "SELL" and last1h["close"] < last1h["bb_mid"] - MIN_PIP_DISTANCE)
    reversal = (direction == "BUY" and last1h["close"] < last1h["bb_mid"]) or \
               (direction == "SELL" and last1h["close"] > last1h["bb_mid"])
    confirm1d = (direction == "BUY" and last1d["close"] > last1d["open"]) or \
                (direction == "SELL" and last1d["close"] < last1d["open"])
    inside_bb1d = last1d["close"] < last1d["bb_upper"] and last1d["close"] > last1d["bb_lower"]

    if (trend or reversal) and confirm1d and inside_bb1d:
        return direction, last1h
    return None, last1h

# ──────────────────────────────
# 🧠 SENTIMENT ANALYSIS
# ──────────────────────────────
def fetch_news(query="gold price", num_articles=10):
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
    pos_pct = (summary["Positive"]/total)*100 if total else 0
    neg_pct = (summary["Negative"]/total)*100 if total else 0
    return pos_pct, neg_pct

# ──────────────────────────────
# 📬 TELEGRAM ALERT
# ──────────────────────────────
def send_alert(msg):
    try:
        asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg))
        print(f"✅ Alert sent: {msg}")
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")

# ──────────────────────────────
# 🚀 BOT LOOP
# ──────────────────────────────
WAT = timezone(timedelta(hours=1))  # UTC+1

def bot_loop():
    print("🤖 Bot loop started...")
    last_signal = None
    while True:
        try:
            now_wat = datetime.now(WAT)
            df_1h = fetch_data("1h", 100)
            df_1d = fetch_data("1day", 50)

            # Normal strategy alerts
            if df_1h is not None and df_1d is not None:
                signal, last = generate_signal(df_1h, df_1d)
                if signal and signal != last_signal:
                    pos, neg = analyze_sentiment_for_gold()
                    print(f"🧠 Sentiment → Pos: {pos:.1f}% | Neg: {neg:.1f}%")
                    if (signal == "BUY" and pos >= 30) or (signal == "SELL" and neg >= 30):
                        msg = (
                            f"📈 Gold Signal Confirmed ({signal})\n"
                            f"Time: {last['datetime']}\n"
                            f"Close: ${last['close']:.2f}\n"
                            f"RSI: {last['rsi']:.2f}\n"
                            f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}%"
                        )
                        send_alert(msg)
                        last_signal = signal

            # Forced 1AM WAT alert (weekdays) — always send
            if now_wat.hour == 1 and now_wat.weekday() < 5:
                sig_text = "No data"
                rsi_text = "N/A"
                close_text = "N/A"
                if df_1h is not None and df_1d is not None:
                    signal, last = generate_signal(df_1h, df_1d)
                    sig_text = signal if signal else "No clear signal"
                    rsi_text = f"{last['rsi']:.2f}"
                    close_text = f"${last['close']:.2f}"
                msg = (
                    f"⏰ Gold 1AM WAT Status\n"
                    f"Signal: {sig_text}\n"
                    f"Close: {close_text}\n"
                    f"RSI: {rsi_text}\n"
                    f"Time: {now_wat}"
                )
                send_alert(msg)
                print("📨 Forced 1AM alert sent.")

            time.sleep(SLEEP_SECS)

        except Exception as e:
            print(f"💥 Bot loop crashed: {e}")
            time.sleep(60)

# ──────────────────────────────
# FLASK FOR RAILWAY HEALTHCHECK
# ──────────────────────────────
app = Flask(__name__)
threading.Thread(target=bot_loop, daemon=True).start()

@app.route("/")
def health():
    return "Gold Bot running ✅", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
