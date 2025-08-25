# Multi-stage build for AI Trading System
FROM python:3.13-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements/ requirements/

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.13-slim as production

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading trading

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy built dependencies from builder stage
COPY --from=builder /root/.local /home/trading/.local

# Copy application source
COPY src/ src/
COPY tests/ tests/
COPY pyproject.toml .
COPY .env.example .env

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R trading:trading /app

# Switch to non-root user
USER trading

# Add local bins to PATH
ENV PATH=/home/trading/.local/bin:$PATH

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "src.main", "--mode", "paper"]

# Development stage
FROM production as development

USER root

# Install development dependencies
COPY requirements/requirements-dev.txt requirements/requirements-dev.txt
RUN pip install --no-cache-dir -r requirements/requirements-dev.txt

# Install additional dev tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER trading

# Override command for development
CMD ["python", "-m", "pytest", "--cov=src", "--cov-report=html"]

# Testing stage
FROM development as testing

# Copy test configuration
COPY .coveragerc .
COPY pytest.ini .

# Run tests by default
CMD ["python", "-m", "pytest", "-v", "--cov=src", "--cov-report=term-missing", "--cov-fail-under=80"]
