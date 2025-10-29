# Base image with Python 3.11
FROM python:3.11-slim

# Set environment variables for caching transformers
ENV HF_HOME=/tmp/.cache
ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot code
COPY . .

# Expose port (match your Flask port)
EXPOSE 10000

# Start the bot
CMD ["python", "goldStra.py"]
