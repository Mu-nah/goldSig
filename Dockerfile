# ──────────────────────────────
# Base image
# ──────────────────────────────
FROM python:3.11-slim

# ──────────────────────────────
# Set working directory
# ──────────────────────────────
WORKDIR /app

# ──────────────────────────────
# Install system dependencies
# ──────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────
# Copy requirements
# ──────────────────────────────
COPY requirements.txt .

# ──────────────────────────────
# Install Python dependencies
# ──────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────
# Copy bot code
# ──────────────────────────────
COPY . .

# ──────────────────────────────
# Set environment variables for cache (FinBERT)
# ──────────────────────────────
ENV HF_HOME=/tmp/.cache
ENV TRANSFORMERS_CACHE=/tmp/.cache

# ──────────────────────────────
# Expose Flask port
# ──────────────────────────────
EXPOSE 8080

# ──────────────────────────────
# Run bot via Flask
# ──────────────────────────────
CMD ["python", "goldStra.py"]
