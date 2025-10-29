import os, time, requests, pandas as pd, numpy as np, feedparser, torch, asyncio, threading, logging
from flask import Flask, jsonify
from telegram import Bot
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from urllib.parse import quote

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

# Only show warnings and errors in logs
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

# ──────────────────────────────
# 🧠 FINBERT SENTIMENT
# ──────────────────────────────
os.environ["HF_HOME"] = "/tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache"
labels = ["Positive", "Negative", "Neutral"]
finbert_tokenizer, finbert_model = None, None

def load_finbert():
    global finbert_tokenizer, finbert_model
    if finbert_model is None or finbert_tokenizer is None:
        logging.warning("Loading FinBERT model...")
        finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        logging.warning("FinBERT loaded.")

# ──────────────────────────────
# 🔄 DATA FETCH
# ──────────────────────────────
def fetch_data(interval, limit=100):
    base_url = "https://api.twelvedata.com/time_series"
    for key in API_KEYS:
        try:
            url = f"{base_url}?symbol={SYMBOL}&interval={interval}&outputsize={limit}&apikey={key.strip()}"
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and "values" in r.json():
                df = pd.DataFrame(r.json()["values"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime")
                df = df.astype({"open": float, "high": float, "low": float, "close": float})
                return df
        except Exception as e:
            logging.error(f"API error {key[:6]}: {e}")
    return None

# ──────────────────────────────
# 📊 INDICATORS
# ──────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std

# ──────────────────────────────
# 📈 STRATEGY
# ──────────────────────────────
def generate_signal(df_1h, df_1d):
    df_1h["rsi"] = rsi(df_1h["close"], RSI_PERIOD)
    df_1h["bb_upper"], df_1h["bb_mid"], df_1h["bb_lower"] = bollinger_bands(df_1h["close"])
    df_1d["bb_upper"], _, df_1d["bb_lower"] = bollinger_bands(df_1d["close"])

    last1h, last1d = df_1h.iloc[-1], df_1d.iloc[-1]
    direction = "BUY" if last1h["close"] > last1h["open"] else "SELL"
    trend = (direction == "BUY" and last1h["close"] > last1h["bb_mid"] + MIN_PIP_DISTANCE) or \
            (direction == "SELL" and last1h["close"] < last1h["bb_mid"] - MIN_PIP_DISTANCE)
    reversal = (direction == "BUY" and last1h["close"] < last1h["bb_mid"]) or \
               (direction == "SELL" and last1h["close"] > last1h["bb_mid"])
    confirm1d = (direction == "BUY" and last1d["close"] > last1d["open"]) or \
                (direction == "SELL" and last1d["close"] < last1d["open"])
    inside_bb1d = last1d["bb_lower"] < last1d["close"] < last1d["bb_upper"]

    if (trend or reversal) and confirm1d and inside_bb1d:
        return direction, last1h
    return None, last1h

# ──────────────────────────────
# 🧠 SENTIMENT
# ──────────────────────────────
def fetch_news(query="gold market", num_articles=10):
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={quote(query)}")
        return [entry.title for entry in feed.entries[:num_articles]]
    except Exception as e:
        logging.error(f"News fetch error: {e}")
        return []

def finbert_sentiment(text):
    load_finbert()
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    return dict(zip(labels, probs))

def analyze_sentiment_for_gold():
    titles = fetch_news("gold market", 15)
    if not titles:
        return 0, 0
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for title in titles:
        scores = finbert_sentiment(title)
        summary[max(scores, key=scores.get)] += 1
    total = sum(summary.values())
    return (summary["Positive"]/total)*100, (summary["Negative"]/total)*100

# ──────────────────────────────
# 📬 TELEGRAM ALERTS
# ──────────────────────────────
async def send_alert(msg):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def send_alert_sync(msg):
    asyncio.run(send_alert(msg))

# ──────────────────────────────
# 🖥 FLASK + BACKGROUND LOOP
# ──────────────────────────────
app = Flask(__name__)
last_signal = None

@app.route("/")
def home():
    return jsonify({"status": "running", "symbol": SYMBOL, "last_signal": last_signal})

def background_loop():
    global last_signal
    while True:
        try:
            df_1h, df_1d = fetch_data("1h", 100), fetch_data("1day", 50)
            if df_1h is None or df_1d is None:
                time.sleep(SLEEP_SECS)
                continue
            signal, last = generate_signal(df_1h, df_1d)
            if signal and signal != last_signal:
                pos, neg = analyze_sentiment_for_gold()
                if (signal == "BUY" and pos >= 30) or (signal == "SELL" and neg >= 30):
                    msg = (
                        f"📈 Gold Signal Confirmed ({signal})\n"
                        f"Time: {last['datetime']}\n"
                        f"Close: ${last['close']:.2f}\n"
                        f"RSI: {last['rsi']:.2f}\n"
                        f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}%"
                    )
                    send_alert_sync(msg)
                    last_signal = signal
            time.sleep(SLEEP_SECS)
        except Exception as e:
            logging.error(f"Loop error: {e}")
            time.sleep(SLEEP_SECS)

if __name__ == "__main__":
    threading.Thread(target=background_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
