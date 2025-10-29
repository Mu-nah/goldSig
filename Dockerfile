# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for pip and SSL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot code
COPY . .

# Set environment variables for Hugging Face caching
ENV HF_HOME=/tmp/.cache
ENV TRANSFORMERS_CACHE=/tmp/.cache

# Expose the port (Render expects 10000 by default or $PORT)
EXPOSE 10000

# Start the bot via Flask web service
CMD ["python", "goldStra.py"]
