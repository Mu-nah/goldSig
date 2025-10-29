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

# Expose port (Render sometimes expects it)
EXPOSE 8080

# Set environment variable (helps some platforms)
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "goldStra.py"]
