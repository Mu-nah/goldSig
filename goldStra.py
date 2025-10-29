import os, time, requests, pandas as pd, numpy as np, torch, threading, asyncio
from telegram import Bot
from dotenv import load_dotenv
from flask import Flask, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import feedparser
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ──────────────────────────────
# CONFIG
# ──────────────────────────────
load_dotenv()
SYMBOL = "XAU/USD"
API_KEYS = os.getenv("TD_API_KEYS", "").split(",")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SLEEP_SECS = 1200  # 20 minutes

bot = Bot(token=TELEGRAM_TOKEN)
last_signal = None

# ──────────────────────────────
# FINBERT SENTIMENT SETUP (optional)
# ──────────────────────────────
os.environ["HF_HOME"] = "/tmp/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache"
labels = ["Positive", "Negative", "Neutral"]

finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

vader = SentimentIntensityAnalyzer()

# ──────────────────────────────
# FETCH DATA
# ──────────────────────────────
def fetch_data(interval, limit=100):
    for key in API_KEYS:
        url = f"https://api.twelvedata.com/time_series?symbol={SYMBOL}&interval={interval}&outputsize={limit}&apikey={key.strip()}"
        try:
            r = requests.get(url, timeout=15)
            data = r.json()
            if "values" in data:
                df = pd.DataFrame(data["values"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime")
                df = df.astype({"open": float, "high": float, "low": float, "close": float})
                return df
        except Exception as e:
            print(f"❌ Error with key {key[:6]} -> {e}")
    return None

# ──────────────────────────────
# SENTIMENT ANALYSIS
# ──────────────────────────────
def fetch_news(query, num_articles=20):
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)
    now = datetime.now(timezone.utc)
    six_hours_ago = now - timedelta(hours=6)

    articles = []
    for entry in feed.entries[:num_articles]:
        try:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except:
            published = now
        if published < six_hours_ago:
            continue

        link = entry.link
        title = entry.title
        content = fetch_article_content(link)

        articles.append({"title": title, "link": link, "published": published, "content": content})

    return articles

def fetch_article_content(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs])
    except:
        return ""

def analyze_sentiment(text):
    scores = vader.polarity_scores(text)
    compound = scores['compound']
    if compound > 0.05:
        return "Positive", compound
    elif compound < -0.05:
        return "Negative", compound
    else:
        return "Neutral", compound

def summarize_sentiments(articles):
    summary = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for a in articles:
        sentiment, _ = analyze_sentiment(a["title"] + " " + a["content"])
        summary[sentiment] += 1
    total = sum(summary.values()) or 1
    return {k: (v, v/total*100) for k,v in summary.items()}

# ──────────────────────────────
# TELEGRAM ALERT
# ──────────────────────────────
def send_alert(msg):
    try:
        asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg))
        print(f"✅ Alert sent: {msg}")
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")

# ──────────────────────────────
# STRATEGY LOOP
# ──────────────────────────────
def strategy_loop():
    global last_signal
    while True:
        df_1h = fetch_data("1h", 100)
        df_1d = fetch_data("1day", 50)
        if df_1h is None or df_1d is None:
            print("⚠️ Could not fetch market data.")
            time.sleep(SLEEP_SECS)
            continue

        # --- Price action signal ---
        last1h = df_1h.iloc[-1]
        last1d = df_1d.iloc[-1]
        direction = "BUY" if last1h["close"] > last1h["open"] else "SELL"

        if last_signal == direction:
            print("📊 Duplicate signal. Skipping.")
            time.sleep(SLEEP_SECS)
            continue

        # --- News sentiment confirmation ---
        articles = []
        queries = ["gold market", "gold price", "gold news"]
        for q in queries:
            articles.extend(fetch_news(q))
        sentiment_summary = summarize_sentiments(articles)

        print(f"🧠 Sentiment: {sentiment_summary}")

        pos_pct = sentiment_summary["Positive"][1]
        neg_pct = sentiment_summary["Negative"][1]

        if (direction == "BUY" and pos_pct >= 30) or (direction == "SELL" and neg_pct >= 30):
            msg = (
                f"📈 Gold Signal ({direction})\n"
                f"Time: {last1h['datetime']}\n"
                f"Close: ${last1h['close']:.2f}\n"
                f"Sentiment 🟢 {pos_pct:.1f}% 🔴 {neg_pct:.1f}%"
            )
            send_alert(msg)
            last_signal = direction
        else:
            print("❌ Sentiment weak — signal ignored.")

        time.sleep(SLEEP_SECS)

# ──────────────────────────────
# FLASK APP (Render)
# ──────────────────────────────
app = Flask(__name__)

@app.route("/")
def home():
    return "🟢 Gold Strategy Bot Running!"

@app.route("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

# Start background thread
threading.Thread(target=strategy_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
