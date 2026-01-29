# API Service Dockerfile
# Lightweight container for FastAPI backend

FROM python:3.11-slim

# Labels
LABEL maintainer="Video Classifier Team"
LABEL description="FastAPI backend for Video Classifier"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_BASE_DIR=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY docker/requirements/api-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy application code
# 1. Common modules (io, data - for database, storage, queue)
COPY backend/common /app/common

# 2. API code
COPY backend/api /app/api
COPY backend/main.py /app/main.py

# 3. Config files
COPY config /app/config

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
