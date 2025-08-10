# Multi-stage Dockerfile for Tiny LLM Edge Profiler
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Run tests
RUN python -m pytest tests/ -v

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ /app/src/
COPY README.md LICENSE /app/

# Create non-root user
RUN useradd --create-home --shell /bin/bash profiler
USER profiler

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from src.tiny_llm_profiler.health_monitor import get_system_health; print(get_system_health()['status'])" || exit 1

# Expose default port (if web interface is added)
EXPOSE 8080

# Default command
CMD ["python", "-m", "src.tiny_llm_profiler.cli_lite", "benchmark", "--output=/app/data/benchmark_results.json"]