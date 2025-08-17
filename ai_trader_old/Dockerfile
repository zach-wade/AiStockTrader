# AI Trader Docker Configuration
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-prod.txt .

# Create wheels
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt -r requirements-prod.txt

# Runtime stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    AI_TRADER_ENV=prod

# Create non-root user
RUN groupadd -r aitrader && \
    useradd -r -g aitrader -d /app -s /sbin/nologin aitrader

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY --chown=aitrader:aitrader src/ /app/src/
COPY --chown=aitrader:aitrader pyproject.toml setup.py setup.cfg /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R aitrader:aitrader /app/logs /app/data

# Install the application
RUN pip install -e .

# Switch to non-root user
USER aitrader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from main.utils import core; print('OK')" || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command
CMD ["python", "src/main/app/run_trading.py"]
