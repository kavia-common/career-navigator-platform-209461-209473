# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src

EXPOSE 8001

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8001

CMD ["sh", "-c", "uvicorn src.api.main:app --host ${UVICORN_HOST} --port ${UVICORN_PORT}"]
