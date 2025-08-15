# Multi-stage Docker build for Self-Healing Pipeline Guard
# Optimized for production deployment with security and performance

# Build stage
FROM python:3.12-slim AS builder

# Set build arguments
ARG BUILD_VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt requirements-lite.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /tmp/src/
COPY pyproject.toml /tmp/

# Install the package
WORKDIR /tmp
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.12-slim AS production

# Set metadata labels
LABEL maintainer="Terragon Labs <dev@terragon.dev>"
LABEL version="${BUILD_VERSION}"
LABEL build-date="${BUILD_DATE}"
LABEL vcs-ref="${VCS_REF}"
LABEL description="Self-Healing Pipeline Guard - Advanced CI/CD monitoring and autonomous healing"
LABEL org.opencontainers.image.title="self-healing-pipeline-guard"
LABEL org.opencontainers.image.description="Advanced self-healing pipeline monitoring system"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.version="${BUILD_VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.source="https://github.com/terragon-labs/self-healing-pipeline-guard"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r guardapp && useradd -r -g guardapp -d /app -s /bin/bash guardapp

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=guardapp:guardapp scripts/ /app/scripts/
COPY --chown=guardapp:guardapp config/ /app/config/
COPY --chown=guardapp:guardapp docker/self-healing-guard/entrypoint.sh /app/entrypoint.sh
COPY --chown=guardapp:guardapp docker/self-healing-guard/healthcheck.py /app/healthcheck.py

# Make scripts executable
RUN chmod +x /app/entrypoint.sh /app/scripts/*.sh

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/tmp && \
    chown -R guardapp:guardapp /app

# Set security-focused environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH" \
    USER=guardapp \
    UID=1000 \
    GID=1000

# Expose application port
EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

# Switch to non-root user
USER guardapp

# Set entrypoint
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["/app/entrypoint.sh"]

# Development stage (for development/testing)
FROM production AS development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    strace \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

# Switch back to app user
USER guardapp

# Override entrypoint for development
CMD ["/bin/bash"]