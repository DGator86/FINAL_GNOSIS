# =============================================================================
# Super Gnosis Trading System - Docker Image
# =============================================================================
#
# Multi-stage build for optimized production image
#
# Build:
#   docker build -t gnosis:latest .
#
# Run:
#   docker run -p 8000:8000 --env-file .env gnosis:latest
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.10-slim as runtime

# Labels
LABEL maintainer="Super Gnosis Trading System"
LABEL version="1.0.0"
LABEL description="Institutional-grade options trading platform"

# Create non-root user for security
RUN groupadd -r gnosis && useradd -r -g gnosis gnosis

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=gnosis:gnosis . .

# Create data directories
RUN mkdir -p /app/data /app/logs && \
    chown -R gnosis:gnosis /app/data /app/logs

# Switch to non-root user
USER gnosis

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run FastAPI with uvicorn
CMD ["uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
