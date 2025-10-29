# ──────────────────────────────
# Use slim Python 3.11 base
# ──────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code
COPY . .

# Set environment variables for HF cache
ENV HF_HOME=/tmp/.cache
ENV TRANSFORMERS_CACHE=/tmp/.cache

# Expose port (needed by Render for web services)
EXPOSE 10000

# Run bot
CMD ["python", "goldStra.py"]
