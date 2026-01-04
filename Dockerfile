# Single-stage build for debugging/reliability
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install dependencies (no venv to ensure global availability)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r gnosis && useradd -r -g gnosis gnosis
RUN mkdir -p /app/data /app/logs && chown -R gnosis:gnosis /app

USER gnosis

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    LOG_LEVEL=INFO

EXPOSE 8000

# Default command (overridden by railway.toml)
CMD ["uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "8000"]
