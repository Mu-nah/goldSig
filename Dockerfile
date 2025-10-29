# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Set environment variables for HuggingFace cache
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV PYTHONUNBUFFERED=1

# Copy your bot code
COPY . .

# Expose the port your Flask app will use
EXPOSE 5000

# Default command
CMD ["python", "goldStra.py"]
