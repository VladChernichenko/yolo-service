FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py nutrition_map.json ./

# Model is NOT baked into the image.
# Provide it one of two ways:
#   • docker-compose: volume-mount best.pt → /app/best.pt
#   • Railway (or any cloud): set MODEL_URL env var to a direct download URL
ENV MODEL_PATH=/app/best.pt

EXPOSE 8001

CMD ["sh", "-c", "\
  if [ ! -f /app/best.pt ]; then \
    if [ -z \"$MODEL_URL\" ]; then echo 'ERROR: /app/best.pt not found and MODEL_URL not set' && exit 1; fi; \
    echo 'Downloading model from MODEL_URL...'; \
    curl -fL \"$MODEL_URL\" -o /app/best.pt; \
  fi; \
  exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}"]
