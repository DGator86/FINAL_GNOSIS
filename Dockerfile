FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Setup Virtual Environment
# This is the critical fix: isolating dependencies in a VENV ensures they are found
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Permissions
RUN groupadd -r gnosis && useradd -r -g gnosis gnosis
RUN mkdir -p /app/data /app/logs && chown -R gnosis:gnosis /app /opt/venv

USER gnosis

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    LOG_LEVEL=INFO

EXPOSE 8000

# Run command
CMD ["uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "8000"]
