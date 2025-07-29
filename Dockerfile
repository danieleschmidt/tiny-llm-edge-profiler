# Multi-stage Docker build for tiny-llm-edge-profiler
# Supports development, testing, and production environments

ARG PYTHON_VERSION=3.11
ARG DEBIAN_VERSION=bookworm

# Base image with Python and system dependencies
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for embedded development
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libusb-1.0-0-dev \
    libusb-1.0-0 \
    udev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 profiler \
    && useradd --uid 1000 --gid profiler --shell /bin/bash --create-home profiler

# Set working directory
WORKDIR /app

# Development stage
FROM base as development

# Install additional development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY --chown=profiler:profiler . .

# Install package in development mode
RUN pip install -e .

# Switch to non-root user
USER profiler

# Set default command for development
CMD ["python", "-m", "tiny_llm_profiler.cli", "--help"]

# Testing stage
FROM development as testing

# Switch back to root for test setup
USER root

# Install additional testing tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    socat \
    minicom \
    && rm -rf /var/lib/apt/lists/*

# Create mock device nodes for testing
RUN mkdir -p /dev/mock && \
    chmod 755 /dev/mock

# Switch back to profiler user
USER profiler

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=src/tiny_llm_profiler"]

# Production stage
FROM base as production

# Copy only production requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=profiler:profiler src/ ./src/
COPY --chown=profiler:profiler pyproject.toml .
COPY --chown=profiler:profiler README.md .
COPY --chown=profiler:profiler LICENSE .

# Install package
RUN pip install --no-cache-dir .

# Switch to non-root user
USER profiler

# Set default command
CMD ["tiny-profiler", "--help"]

# Hardware testing stage
FROM development as hardware

# Switch to root for hardware setup
USER root

# Install hardware-specific dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    esptool \
    dfu-util \
    openocd \
    stlink-tools \
    && rm -rf /var/lib/apt/lists/*

# Install hardware Python dependencies
RUN pip install --no-cache-dir \
    pyftdi \
    adafruit-circuitpython-busdevice \
    adafruit-circuitpython-ina219

# Create udev rules for hardware access
COPY docker/99-embedded-devices.rules /etc/udev/rules.d/

# Add profiler user to dialout group for serial access
RUN usermod -a -G dialout profiler

# Switch back to profiler user
USER profiler

# Set environment for hardware testing
ENV HARDWARE_TEST_MODE=1

# Default command for hardware testing
CMD ["pytest", "tests/hardware/", "-v", "--hardware"]

# Documentation stage
FROM base as docs

# Install documentation dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install docs dependencies
RUN pip install --no-cache-dir \
    mkdocs \
    mkdocs-material \
    mkdocs-mermaid2-plugin \
    mkdocstrings[python]

# Copy source and docs
COPY --chown=profiler:profiler . .

# Install package for docs generation
RUN pip install -e .

# Switch to non-root user
USER profiler

# Expose port for docs server
EXPOSE 8000

# Default command to serve docs
CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8000"]