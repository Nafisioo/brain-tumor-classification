FROM python:3.11-slim

####################################################
# Metadata
####################################################

LABEL maintainer="Nafiseh"
LABEL project="Brain Tumor MRI Classification"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/Nafisioo/brain-tumor-classification"

####################################################
# Environment
####################################################

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

####################################################
# Working directory
####################################################

WORKDIR /app

####################################################
# System dependencies
####################################################

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

####################################################
# Python dependencies
####################################################

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

####################################################
# Application source
####################################################

COPY api/ ./api/
COPY inference/ ./inference/
COPY src/ ./src/
COPY configs/ ./configs/

####################################################
# Lightweight artifacts
####################################################

COPY artifacts/class_names.json ./artifacts/

####################################################
# Startup scripts
####################################################

COPY scripts/ ./scripts/

####################################################
# Create runtime user
####################################################

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/artifacts && \
    chmod +x /app/scripts/*.sh && \
    chown -R appuser:appuser /app

####################################################
# Runtime
####################################################

USER appuser

EXPOSE 8000

####################################################
# Health Check
####################################################

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

####################################################
# Container Entrypoint
####################################################

ENTRYPOINT ["/app/scripts/entrypoint.sh"]