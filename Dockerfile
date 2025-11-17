# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/.cache
ENV TRANSFORMERS_CACHE=/tmp/.cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements file (create this separately)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot source code into the image
COPY . .

# Set an entrypoint to run the bot
CMD ["python", "goldStra.py"]
