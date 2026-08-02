FROM python:3.9-slim


LABEL maintainer="Nafiseh"
LABEL project="Brain Tumor MRI Classification"
LABEL version="1.0"


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app


WORKDIR /app



RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*



COPY requirements.txt .


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt



COPY api ./api
COPY inference ./inference
COPY src ./src
COPY configs ./configs
COPY artifacts ./artifacts



# create non-root user

RUN useradd -m appuser


RUN chown -R appuser:appuser /app


USER appuser



EXPOSE 8000



HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1



CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]