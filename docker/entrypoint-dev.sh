#!/bin/bash
set -e

# Development environment entrypoint script

echo "🚀 Starting Tiny LLM Edge Profiler development environment..."

# Initialize ESP-IDF environment if needed
if [ -f "/opt/esp/esp-idf/export.sh" ]; then
    echo "📦 Setting up ESP-IDF environment..."
    source /opt/esp/esp-idf/export.sh > /dev/null 2>&1
fi

# Set up device permissions
echo "🔧 Setting up device permissions..."
if [ -f "/etc/udev/rules.d/99-platformio-udev.rules" ]; then
    udevadm control --reload-rules || true
fi

# Install project in development mode if not already installed
if ! python -c "import tiny_llm_profiler" 2>/dev/null; then
    echo "📦 Installing project in development mode..."
    pip install -e .
fi

# Set up pre-commit hooks if git repository exists
if [ -d ".git" ] && [ -f ".pre-commit-config.yaml" ]; then
    echo "🪝 Setting up pre-commit hooks..."
    pre-commit install --install-hooks || true
fi

# Display environment information
echo "🌟 Development environment ready!"
echo ""
echo "📋 Available tools:"
echo "  - Python: $(python --version)"
echo "  - ESP-IDF: ${IDF_PATH:-Not available}"
echo "  - PlatformIO: $(pio --version 2>/dev/null || echo 'Not available')"
echo "  - OpenOCD: $(openocd --version 2>&1 | head -1 || echo 'Not available')"
echo ""
echo "🔧 Available commands:"
echo "  - tiny-profiler --help         # Main CLI tool"
echo "  - get_idf                      # Activate ESP-IDF environment"
echo "  - pytest tests/               # Run tests"
echo "  - black src/ tests/           # Format code"
echo "  - flake8 src/ tests/          # Lint code"
echo ""
echo "📖 Quick start:"
echo "  1. Connect your development board via USB"
echo "  2. List devices: ls /dev/tty*"
echo "  3. Start profiling: tiny-profiler profile --device /dev/ttyUSB0"
echo ""

# Execute the provided command or start interactive shell
if [ $# -eq 0 ]; then
    exec bash
else
    exec "$@"
fi