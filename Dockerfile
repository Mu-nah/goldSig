# Use a lightweight Python image
FROM python:3.11-slim

# Prevent bytecode creation and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port Flask will use
EXPOSE 8080

# Run the app
CMD ["python", "goldStra.py"]
