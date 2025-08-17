FROM python:3.11-slim

# System libs for pyzbar and some OpenCV routines
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY decoder.py server.py ./

# Gunicorn + Uvicorn workers (tweak WORKERS via env if you like)
ENV PORT=8000
ENV WORKERS=2
CMD exec gunicorn server:app \
    --bind 0.0.0.0:${PORT} \
    --workers ${WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 60
