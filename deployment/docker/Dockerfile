# Multi-stage Dockerfile for Tiny LLM Edge Profiler
# Production-ready containerized deployment

# Build stage
FROM python:3.12-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./

# Build wheel
RUN pip install --no-cache-dir build
RUN python -m build --wheel

# Production stage
FROM python:3.12-slim as production

# Create non-root user for security
RUN groupadd -r profiler && useradd -r -g profiler profiler

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    curl \
    udev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /build/dist/*.whl ./

# Install the application
RUN pip install --no-cache-dir *.whl && rm *.whl

# Copy additional files
COPY scripts/ ./scripts/
COPY deployment/config/ ./config/

# Create directories for data and logs
RUN mkdir -p /data /logs /cache && \
    chown -R profiler:profiler /data /logs /cache /app

# Set environment variables
ENV PYTHONPATH=/app \
    TINY_LLM_LOG_DIR=/logs \
    TINY_LLM_OUTPUT_DIR=/data \
    TINY_LLM_MODEL_CACHE=/cache \
    TINY_LLM_LOG_LEVEL=INFO \
    TINY_LLM_LOG_JSON=true

# Expose ports (if running web interface)
EXPOSE 8080 8081

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from tiny_llm_profiler.health import run_health_checks; exit(0 if run_health_checks() else 1)"

# Switch to non-root user
USER profiler

# Set entrypoint
ENTRYPOINT ["python", "-m", "tiny_llm_profiler.cli"]
CMD ["--help"]

# Labels for container metadata
LABEL org.opencontainers.image.title="Tiny LLM Edge Profiler" \
      org.opencontainers.image.description="Comprehensive profiling toolkit for quantized LLMs on edge devices" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.authors="Terragon Labs <dev@terragon.dev>" \
      org.opencontainers.image.source="https://github.com/terragon-labs/tiny-llm-edge-profiler" \
      org.opencontainers.image.licenses="Apache-2.0"