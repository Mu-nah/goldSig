# ---------- Base ----------
FROM python:3.11-slim

# ---------- Environment ----------
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# ---------- System deps ----------
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# ---------- Install deps ----------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ---------- Copy source ----------
COPY . .

# ---------- Port for Flask health check ----------
EXPOSE 10000

# ---------- Entrypoint ----------
CMD ["python", "goldStra.py"]
