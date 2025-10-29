# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port for Flask health check
EXPOSE 8080

# Environment variable (for unbuffered logs)
ENV PYTHONUNBUFFERED=1

# Start both bot and health server
CMD ["python", "goldStra.py"]
