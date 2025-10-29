import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional: vader for fallback sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# HuggingFace cache
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

# Lazy-loaded variables
model = None
tokenizer = None
vader = SentimentIntensityAnalyzer()

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Loading FinBERT model...")
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone", use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone",
            device_map="cpu"  # force CPU
        )
        print("Model loaded!")

@app.route("/predict", methods=["POST"])
def predict():
    # Load model lazily
    load_model()

    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1).tolist()[0]

    result = {
        "positive": probs[0],
        "neutral": probs[1],
        "negative": probs[2]
    }

    # Optional: fallback sentiment
    vader_score = vader.polarity_scores(text)
    result["vader"] = vader_score

    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
