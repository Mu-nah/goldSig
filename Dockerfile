# ---------- Base ----------
FROM python:3.11-slim

# ---------- Environment ----------
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# ---------- System deps ----------
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# ---------- Install deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy source ----------
COPY . .

# ---------- Entrypoint ----------
CMD ["python", "goldStra.py"]
