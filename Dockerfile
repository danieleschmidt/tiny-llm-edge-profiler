# Multi-stage Dockerfile for Tiny LLM Edge Profiler
# Supports both development and production environments

# =============================================================================
# Build stage: Development tools and compilation
# =============================================================================
FROM ubuntu:22.04 AS builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    # Python development
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Hardware toolchains
    gcc-arm-none-eabi \
    openocd \
    # Device communication
    libusb-1.0-0-dev \
    libudev-dev \
    # ESP32 dependencies
    flex \
    bison \
    gperf \
    libffi-dev \
    libssl-dev \
    dfu-util \
    libncurses5-dev \
    libncursesw5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlink
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install ESP-IDF
WORKDIR /opt/esp
RUN git clone --recursive --depth 1 --branch v5.1.2 https://github.com/espressif/esp-idf.git && \
    cd esp-idf && \
    ./install.sh esp32 && \
    rm -rf .git

# Install PlatformIO
RUN python3 -m pip install --upgrade pip && \
    pip3 install platformio

# Set up working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements*.txt pyproject.toml setup.py ./
COPY src/tiny_llm_profiler/__init__.py src/tiny_llm_profiler/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -r requirements-dev.txt

# =============================================================================
# Production stage: Minimal runtime environment
# =============================================================================
FROM ubuntu:22.04 AS production

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libusb-1.0-0 \
    libudev1 \
    # Serial communication tools
    minicom \
    picocom \
    screen \
    # Basic utilities
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlink
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Create non-root user
RUN groupadd -r profiler && useradd -r -g profiler -G dialout,plugdev profiler

# Set up working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY *.py ./
COPY *.toml *.txt ./
COPY README.md LICENSE ./

# Install application
RUN pip3 install --no-deps -e .

# Set up device permissions
COPY docker/99-platformio-udev.rules /etc/udev/rules.d/

# Create directories for data
RUN mkdir -p /app/data/{models,results,logs,firmware} && \
    chown -R profiler:profiler /app

# =============================================================================
# Development stage: Full development environment
# =============================================================================
FROM builder AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    # Additional dev tools
    vim \
    nano \
    htop \
    tree \
    jq \
    # Documentation tools
    pandoc \
    texlive-latex-base \
    # Debugging tools
    gdb \
    gdb-multiarch \
    && rm -rf /var/lib/apt/lists/*

# Install pre-commit and development tools
RUN pip3 install pre-commit black isort flake8 mypy pytest

# Copy ESP-IDF from builder
COPY --from=builder /opt/esp/esp-idf /opt/esp/esp-idf

# Set up ESP-IDF environment
ENV IDF_PATH=/opt/esp/esp-idf
ENV PATH="${IDF_PATH}/tools:${PATH}"

# Copy development configuration
COPY .devcontainer/ .devcontainer/
COPY .vscode/ .vscode/
COPY .pre-commit-config.yaml .editorconfig ./

# Install pre-commit hooks
RUN git init . && pre-commit install --install-hooks || true

# Set up aliases and environment
RUN echo 'alias ll="ls -alF"' >> /root/.bashrc && \
    echo 'alias la="ls -A"' >> /root/.bashrc && \
    echo 'alias l="ls -CF"' >> /root/.bashrc && \
    echo 'export PATH="/opt/esp/esp-idf/tools:$PATH"' >> /root/.bashrc && \
    echo 'alias get_idf=". /opt/esp/esp-idf/export.sh"' >> /root/.bashrc

# Create entrypoint script
COPY docker/entrypoint-dev.sh /entrypoint-dev.sh
RUN chmod +x /entrypoint-dev.sh

ENTRYPOINT ["/entrypoint-dev.sh"]
CMD ["bash"]

# =============================================================================
# CI/CD stage: Optimized for testing and building
# =============================================================================
FROM builder AS ci

# Install additional CI tools
RUN pip3 install \
    coverage[toml] \
    pytest-html \
    pytest-cov \
    pytest-xdist \
    bandit \
    safety \
    pip-audit

# Set up CI environment
ENV CI=true
ENV PYTEST_ADDOPTS="--tb=short --strict-markers"

# Copy all source code
COPY . .

# Install application in development mode
RUN pip3 install -e .

# Run tests and generate reports
RUN mkdir -p /workspace/reports

# Default command for CI
CMD ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=html", "--cov-report=xml", "--html=reports/pytest-report.html", "--self-contained-html"]

# =============================================================================
# Final stage selection based on build target
# =============================================================================
FROM ${BUILD_TARGET:-production} AS final

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import tiny_llm_profiler; print('OK')" || exit 1

# Labels for metadata
LABEL org.opencontainers.image.title="Tiny LLM Edge Profiler"
LABEL org.opencontainers.image.description="Comprehensive profiling toolkit for quantized LLMs on edge devices"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.vendor="Your Organization"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/your-org/tiny-llm-edge-profiler"
LABEL org.opencontainers.image.documentation="https://docs.your-org.com/tiny-llm-profiler"

# Default working directory and user for production
WORKDIR /app
USER profiler

# Expose default ports
EXPOSE 8000 8080

# Default command
CMD ["tiny-profiler", "--help"]