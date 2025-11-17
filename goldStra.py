import os, time, requests, pandas as pd, numpy as np, feedparser, torch, asyncio, threading 
from telegram import Bot
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from flask import Flask

# ──────────────────────────────
# CONFIG
# ──────────────────────────────
load_dotenv()
SYMBOLS = ["XAU/USD", "AUD/USD"]
API_KEYS = os.getenv("TD_API_KEYS", "").split(",")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STDDEV = 2
MIN_PIP_DISTANCE = 1.0
SLEEP_SECS = 1200  # 20 minutes
SENTIMENT_BENCH = 30  # sentiment threshold for alerts

# ──────────────────────────────
# TELEGRAM SETUP
# ──────────────────────────────
bot = Bot(token=TELEGRAM_TOKEN)
loop = asyncio.new_event_loop()

def send_alert(msg):
    """Send message using background event loop"""
    asyncio.run_coroutine_threadsafe(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg), loop)

# ──────────────────────────────
# FINBERT SENTIMENT
# ──────────────────────────────
os.environ["HF_HOME"] = "/tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache"
labels = ["Positive", "Negative", "Neutral"]

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# ──────────────────────────────
# DATA FETCH
# ──────────────────────────────
def fetch_data(symbol, interval, limit=100):
    base_url = "https://api.twelvedata.com/time_series"
    for key in API_KEYS:
        url = f"{base_url}?symbol={symbol}&interval={interval}&outputsize={limit}&apikey={key.strip()}"
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
        except:
            continue
    return None

# ──────────────────────────────
# INDICATORS
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
# STRATEGY (now returns signal type: Trend / Reversal / None)
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
        sig_type = "Trend" if trend else "Reversal"
        return direction, last1h, sig_type

    return None, last1h, None

# ──────────────────────────────
# SENTIMENT ANALYSIS
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

def analyze_sentiment(symbol):
    query_map = {"XAU/USD": "gold market", "AUD/USD": "AUD USD forex"}
    titles = fetch_news(query_map.get(symbol, symbol), 15)
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for title in titles:
        scores = finbert_sentiment(title)
        dominant = max(scores, key=scores.get)
        summary[dominant] += 1
    total = sum(summary.values())
    pos_pct = (summary["Positive"]/total)*100 if total else 0
    neg_pct = (summary["Negative"]/total)*100 if total else 0
    neu_pct = (summary["Neutral"]/total)*100 if total else 0
    return pos_pct, neg_pct, neu_pct

# ──────────────────────────────
# BOT LOOP
# ──────────────────────────────
WAT = timezone(timedelta(hours=1))  # UTC+1
last_sent_date = None
last_signal_dict = {symbol: None for symbol in SYMBOLS}

def bot_loop():
    global last_sent_date, last_signal_dict

    # Send startup status alert
    now_wat = datetime.now(WAT)
    for symbol in SYMBOLS:
        df_1h = fetch_data(symbol, "1h", 100)
        df_1d = fetch_data(symbol, "1day", 50)
        if df_1h is None or df_1d is None:
            continue
        signal, last1h, sig_type = generate_signal(df_1h, df_1d)
        pos, neg, neu = analyze_sentiment(symbol)
        msg = (
            f"⚡ {symbol} Startup Status\n"
            f"Signal: {signal if signal else 'No clear signal'}" + (f" ({sig_type})" if sig_type else "") + "\n"
            f"Close: {last1h['close']:.4f}\n"
            f"RSI: {last1h['rsi']:.2f}\n"
            f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%\n"
            f"Time: {now_wat}"
        )
        send_alert(msg)
    last_sent_date = now_wat.date()

    while True:
        now_wat = datetime.now(WAT)

        # Normal signals
        for symbol in SYMBOLS:
            df_1h = fetch_data(symbol, "1h", 100)
            df_1d = fetch_data(symbol, "1day", 50)
            if df_1h is None or df_1d is None:
                continue
            signal, last1h, sig_type = generate_signal(df_1h, df_1d)
            pos, neg, neu = analyze_sentiment(symbol)
            if signal and signal != last_signal_dict[symbol]:
                if (signal == "BUY" and pos >= SENTIMENT_BENCH) or \
                   (signal == "SELL" and neg >= SENTIMENT_BENCH):
                    msg = (
                        f"📈 {symbol} Signal Confirmed ({signal})" + (f" [{sig_type}]" if sig_type else "") + "\n"
                        f"Time: {last1h['datetime']}\n"
                        f"Close: {last1h['close']:.4f}\n"
                        f"RSI: {last1h['rsi']:.2f}\n"
                        f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%"
                    )
                    send_alert(msg)
                    last_signal_dict[symbol] = signal

        # Forced 1AM WAT alert (weekdays, once per day)
        if now_wat.weekday() < 5 and last_sent_date != now_wat.date() and 1 <= now_wat.hour < 2:
            for symbol in SYMBOLS:
                df_1h = fetch_data(symbol, "1h", 100)
                df_1d = fetch_data(symbol, "1day", 50)
                if df_1h is None or df_1d is None:
                    continue
                signal, last1h, sig_type = generate_signal(df_1h, df_1d)
                pos, neg, neu = analyze_sentiment(symbol)
                msg = (
                    f"⏰ {symbol} 1AM WAT Status\n"
                    f"Signal: {signal if signal else 'No clear signal'}" + (f" ({sig_type})" if sig_type else "") + "\n"
                    f"Close: {last1h['close']:.4f}\n"
                    f"RSI: {last1h['rsi']:.2f}\n"
                    f"Sentiment → 🟢 {pos:.1f}% | 🔴 {neg:.1f}% | ⚪ {neu:.1f}%\n"
                    f"Time: {now_wat}"
                )
                send_alert(msg)
            last_sent_date = now_wat.date()

        time.sleep(SLEEP_SECS)

# ──────────────────────────────
# START BOT & TELEGRAM LOOP
# ──────────────────────────────
def start_telegram_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=start_telegram_loop, daemon=True).start()
threading.Thread(target=bot_loop, daemon=True).start()

# ──────────────────────────────
# FLASK HEALTHCHECK
# ──────────────────────────────
app = Flask(__name__)

@app.route("/")
def health():
    return "Trading Bot running ✅", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
