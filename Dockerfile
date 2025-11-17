# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot code
COPY . .

# Set environment variables (if needed, or use Railway/Render env)
# ENV TELEGRAM_BOT_TOKEN=...
# ENV TELEGRAM_CHAT_ID=...

# Expose port for Flask health check
EXPOSE 8080

# Start the bot
CMD ["python", "goldStra.py"]
