# Development container for Tiny LLM Edge Profiler
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Install system dependencies for hardware development
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        # Build tools
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        # Hardware development tools
        minicom \
        picocom \
        screen \
        usbutils \
        pciutils \
        # Cross-compilation toolchains
        gcc-arm-none-eabi \
        binutils-arm-none-eabi \
        # Hardware debugging
        openocd \
        gdb-multiarch \
        # Serial communication
        python3-serial \
        # USB device support
        libusb-1.0-0-dev \
        libudev-dev \
        # Additional development tools
        git-lfs \
        curl \
        wget \
        unzip \
        jq \
        # Documentation tools
        pandoc \
        texlive-latex-base \
        # Monitoring tools
        htop \
        iotop \
        # Network tools
        netcat \
        telnet \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install platform-specific toolchains
RUN mkdir -p /opt/toolchains

# ESP-IDF (ESP32 development)
ENV IDF_PATH="/opt/esp-idf"
RUN git clone --recursive https://github.com/espressif/esp-idf.git $IDF_PATH \
    && cd $IDF_PATH \
    && git checkout release/v5.0 \
    && ./install.sh esp32,esp32s2,esp32s3,esp32c3 \
    && rm -rf .git

# PlatformIO for multi-platform development
RUN pip3 install platformio \
    && pio platform install espressif32 \
    && pio platform install ststm32 \
    && pio platform install raspberrypi \
    && pio platform install nordicnrf52

# Install additional Python packages for development
RUN pip3 install --no-cache-dir \
        # Development tools
        ipython \
        jupyter \
        notebook \
        jupyterlab \
        # Hardware interface libraries
        pyserial \
        pyusb \
        libusb1 \
        # Embedded development
        esptool \
        adafruit-ampy \
        # Analysis and visualization
        matplotlib \
        seaborn \
        plotly \
        bokeh \
        # Data processing
        numpy \
        pandas \
        scipy \
        # Machine learning
        torch \
        transformers \
        # Testing and quality
        coverage \
        pytest-cov \
        pytest-benchmark \
        pytest-mock \
        pytest-asyncio \
        # Documentation
        sphinx \
        sphinx-rtd-theme \
        myst-parser \
        # Performance profiling
        memory-profiler \
        line-profiler \
        py-spy

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Set up user permissions for hardware access
RUN usermod -a -G dialout,plugdev,sudo vscode \
    && echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Create udev rules for common development boards
RUN echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="10c4", ATTR{idProduct}=="ea60", MODE="0666"' > /etc/udev/rules.d/99-esp32.rules \
    && echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6001", MODE="0666"' >> /etc/udev/rules.d/99-esp32.rules \
    && echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1a86", ATTR{idProduct}=="7523", MODE="0666"' >> /etc/udev/rules.d/99-esp32.rules \
    && echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2e8a", ATTR{idProduct}=="0003", MODE="0666"' >> /etc/udev/rules.d/99-rp2040.rules \
    && echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="3748", MODE="0666"' >> /etc/udev/rules.d/99-stm32.rules

# Set up shell environment
USER vscode
RUN echo 'export PATH="$PATH:/opt/esp-idf/tools"' >> ~/.bashrc \
    && echo 'export IDF_PATH="/opt/esp-idf"' >> ~/.bashrc \
    && echo 'source $IDF_PATH/export.sh' >> ~/.bashrc \
    && echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> ~/.bashrc

# Configure git (will be overridden by user config)
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false

# Create commonly used directories
RUN mkdir -p ~/.local/share/platformio \
    && mkdir -p ~/.cache/pip \
    && mkdir -p ~/.vscode-server

# Install Oh My Zsh for better shell experience
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && echo 'export PATH="$PATH:/opt/esp-idf/tools"' >> ~/.zshrc \
    && echo 'export IDF_PATH="/opt/esp-idf"' >> ~/.zshrc \
    && echo 'source $IDF_PATH/export.sh' >> ~/.zshrc \
    && echo 'export PYTHONPATH="/workspace/src:$PYTHONPATH"' >> ~/.zshrc

USER root

# Final cleanup
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Switch back to vscode user
USER vscode

# Set the default command
CMD ["/bin/bash"]