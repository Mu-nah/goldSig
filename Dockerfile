# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install build deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential curl git wget && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all bot code
COPY . .

# Hugging Face cache
ENV HF_HOME=/tmp/.cache
ENV TRANSFORMERS_CACHE=/tmp/.cache

EXPOSE 10000

CMD ["python", "goldStra.py"]
