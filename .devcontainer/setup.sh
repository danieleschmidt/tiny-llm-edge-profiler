#!/bin/bash

# DevContainer post-creation setup script for Tiny LLM Edge Profiler

set -e

echo "🚀 Setting up Tiny LLM Edge Profiler development environment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libusb-1.0-0-dev \
    libudev-dev \
    pkg-config \
    curl \
    wget \
    git \
    screen \
    minicom \
    picocom

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
pip install \
    black \
    isort \
    flake8 \
    pylint \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    hypothesis \
    pre-commit \
    tox \
    sphinx \
    sphinx-rtd-theme

# Install project dependencies if requirements files exist
if [ -f "requirements.txt" ]; then
    echo "📋 Installing project requirements..."
    pip install -r requirements.txt
fi

if [ -f "requirements-dev.txt" ]; then
    echo "🔧 Installing development requirements..."
    pip install -r requirements-dev.txt
fi

# Install the project in development mode if setup.py exists
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "🏗️ Installing project in development mode..."
    pip install -e .
fi

# Install ESP-IDF for ESP32 development
echo "🔧 Installing ESP-IDF..."
mkdir -p /opt/esp
cd /opt/esp
wget -q https://dl.espressif.com/dl/xtensa-esp32-elf-gcc8_4_0-esp-2021r2-linux-amd64.tar.gz
tar -xzf xtensa-esp32-elf-gcc8_4_0-esp-2021r2-linux-amd64.tar.gz
git clone --recursive --depth 1 --branch v4.4.4 https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32
sudo chown -R vscode:vscode /opt/esp

# Install PlatformIO for multi-platform embedded development
echo "🛠️ Installing PlatformIO..."
python -c "$(curl -fsSL https://raw.githubusercontent.com/platformio/platformio/master/scripts/get-platformio.py)"
export PATH=$PATH:~/.platformio/penv/bin

# Setup ARM GCC toolchain
echo "🔧 Installing ARM GCC toolchain..."
sudo apt-get install -y gcc-arm-none-eabi

# Install OpenOCD for debugging
echo "🔍 Installing OpenOCD..."
sudo apt-get install -y openocd

# Setup udev rules for common development boards
echo "⚡ Setting up udev rules..."
sudo tee /etc/udev/rules.d/99-platformio-udev.rules << 'EOF'
# ESP32
SUBSYSTEMS=="usb", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE:="0666"

# STM32
SUBSYSTEMS=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="3748", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="374b", MODE:="0666"

# RP2040
SUBSYSTEMS=="usb", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="0003", MODE:="0666"

# Nordic nRF52
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1366", ATTRS{idProduct}=="1015", MODE:="0666"
EOF

sudo udevadm control --reload-rules
sudo usermod -a -G dialout vscode
sudo usermod -a -G plugdev vscode

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
fi

# Setup shell aliases and environment
echo "🐚 Setting up shell environment..."
cat >> ~/.bashrc << 'EOF'

# Tiny LLM Edge Profiler aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias pytest-cov='pytest --cov=src --cov-report=html --cov-report=term'
alias lint='flake8 src tests && pylint src && mypy src'
alias format='black src tests && isort src tests'

# ESP-IDF environment
alias get_idf='. /opt/esp/esp-idf/export.sh'

# PlatformIO environment
export PATH=$PATH:~/.platformio/penv/bin
EOF

# Create useful development directories
mkdir -p /workspaces/tiny-llm-edge-profiler/{logs,tmp,firmware,models,results}

echo "✅ Development environment setup complete!"
echo ""
echo "🔧 Available tools:"
echo "  - Python development: black, isort, flake8, pylint, mypy, pytest"
echo "  - Embedded development: ESP-IDF, PlatformIO, ARM GCC, OpenOCD"
echo "  - Hardware debugging: minicom, picocom, screen"
echo ""
echo "📖 Next steps:"
echo "  1. Run 'get_idf' to activate ESP-IDF environment"
echo "  2. Connect your development board via USB"
echo "  3. Run 'tiny-profiler --help' to get started"
echo ""