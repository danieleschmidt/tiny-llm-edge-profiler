# Makefile for Tiny LLM Edge Profiler
# Provides standardized build, test, and deployment commands

.PHONY: help install install-dev clean test lint format build docker docs

# =============================================================================
# Configuration
# =============================================================================
PYTHON := python3
PIP := pip3
PROJECT_NAME := tiny-llm-edge-profiler
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_TAG := latest

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo "$(PROJECT_NAME) - Makefile Commands"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment Variables:"
	@echo "  PYTHON      Python interpreter (default: python3)"
	@echo "  PIP         Pip command (default: pip3)"
	@echo "  DOCKER_TAG  Docker image tag (default: latest)"

# =============================================================================
# Development Setup
# =============================================================================
install: ## Install project dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,test,docs]"
	pre-commit install --install-hooks

install-hardware: ## Install hardware-specific dependencies
	$(PIP) install -e ".[hardware]"
	sudo cp docker/99-platformio-udev.rules /etc/udev/rules.d/
	sudo udevadm control --reload-rules

setup-esp32: ## Setup ESP32 development environment
	@echo "Setting up ESP32 development environment..."
	@if [ ! -d "/opt/esp/esp-idf" ]; then \
		echo "Installing ESP-IDF..."; \
		sudo mkdir -p /opt/esp; \
		cd /opt/esp && \
		sudo git clone --recursive --depth 1 --branch v5.1.2 https://github.com/espressif/esp-idf.git; \
		cd esp-idf && sudo ./install.sh esp32; \
	fi
	@echo "ESP-IDF installed. Run 'source /opt/esp/esp-idf/export.sh' to activate."

# =============================================================================
# Code Quality
# =============================================================================
format: ## Format code with black and isort
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

format-check: ## Check code formatting without making changes
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/

lint: ## Run linting with flake8, pylint, and mypy
	flake8 src/ tests/ scripts/
	pylint src/
	mypy src/ tests/

security: ## Run security checks with bandit and safety
	bandit -r src/ -f json
	safety check
	pip-audit

typecheck: ## Run type checking with mypy
	mypy src/ tests/

precommit: ## Run all pre-commit hooks
	pre-commit run --all-files

validate: format-check lint typecheck security ## Run all validation checks

# =============================================================================
# Testing
# =============================================================================
test: ## Run unit tests
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-hardware: ## Run hardware tests (requires real devices)
	pytest tests/hardware/ -v --hardware

test-all: ## Run all tests
	pytest tests/ -v

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=html --cov-report=term --cov-report=xml

test-parallel: ## Run tests in parallel
	pytest tests/ -n auto

test-watch: ## Run tests in watch mode
	ptw tests/

benchmark: ## Run performance benchmarks
	pytest tests/ --benchmark-only

# =============================================================================
# Building and Packaging
# =============================================================================
build: ## Build Python package
	$(PYTHON) -m build

build-wheel: ## Build wheel package only
	$(PYTHON) -m build --wheel

build-sdist: ## Build source distribution only
	$(PYTHON) -m build --sdist

build-firmware: ## Build all firmware binaries
	@echo "Building ESP32 firmware..."
	@if [ -d "firmware/esp32" ]; then \
		cd firmware/esp32 && \
		source /opt/esp/esp-idf/export.sh && \
		idf.py build; \
	fi
	@echo "Building STM32 firmware..."
	@if [ -d "firmware/stm32" ]; then \
		cd firmware/stm32 && \
		make clean && make all; \
	fi

build-docs: ## Build documentation
	cd docs && make html

serve-docs: ## Serve documentation locally
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# =============================================================================
# Docker
# =============================================================================
docker-build: ## Build Docker image for production
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-dev: ## Build Docker image for development
	docker build --target development -t $(DOCKER_IMAGE):dev .

docker-build-ci: ## Build Docker image for CI/CD
	docker build --target ci -t $(DOCKER_IMAGE):ci .

docker-run: ## Run Docker container in production mode
	docker run -it --rm --privileged \
		-v /dev:/dev \
		-v $(PWD)/results:/app/results \
		-p 8000:8000 \
		$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-dev: ## Run Docker container in development mode
	docker run -it --rm --privileged \
		-v /dev:/dev \
		-v $(PWD):/workspace \
		-p 8000:8000 \
		$(DOCKER_IMAGE):dev

docker-compose-dev: ## Start development environment with docker-compose
	docker-compose --profile dev up -d

docker-compose-prod: ## Start production environment with docker-compose
	docker-compose --profile prod up -d

docker-compose-monitoring: ## Start monitoring stack
	docker-compose --profile monitoring up -d

docker-compose-down: ## Stop all docker-compose services
	docker-compose down --remove-orphans

# =============================================================================
# Hardware Operations
# =============================================================================
flash-esp32: ## Flash ESP32 firmware
	@if [ -f "firmware/esp32/build/profiler.bin" ]; then \
		cd firmware/esp32 && \
		source /opt/esp/esp-idf/export.sh && \
		idf.py flash; \
	else \
		echo "ESP32 firmware not built. Run 'make build-firmware' first."; \
	fi

flash-stm32: ## Flash STM32 firmware
	@if [ -f "firmware/stm32/build/profiler.bin" ]; then \
		openocd -f interface/stlink.cfg -f target/stm32f7x.cfg \
			-c "program firmware/stm32/build/profiler.bin 0x08000000 verify reset exit"; \
	else \
		echo "STM32 firmware not built. Run 'make build-firmware' first."; \
	fi

monitor-esp32: ## Monitor ESP32 serial output
	@cd firmware/esp32 && \
	source /opt/esp/esp-idf/export.sh && \
	idf.py monitor

monitor-serial: ## Monitor serial port (generic)
	@PORT=$${SERIAL_PORT:-/dev/ttyUSB0}; \
	BAUD=$${SERIAL_BAUD:-921600}; \
	echo "Monitoring $$PORT at $$BAUD baud..."; \
	picocom -b $$BAUD $$PORT

list-devices: ## List available serial devices
	@echo "Available serial devices:"
	@ls -la /dev/tty* 2>/dev/null | grep -E "(USB|ACM)" || echo "No devices found"
	@echo ""
	@echo "USB devices:"
	@lsusb | grep -E "(10c4|0483|2e8a|1366)" || echo "No known development boards found"

# =============================================================================
# Data Management
# =============================================================================
setup-data-dirs: ## Create data directories
	mkdir -p data/{models,results,logs,firmware,configs}
	mkdir -p results/{profiling,benchmarks,reports}

download-sample-models: ## Download sample quantized models
	@echo "Downloading sample models..."
	@mkdir -p data/models
	@echo "Note: Add actual model download URLs when available"

clean-results: ## Clean result files
	rm -rf results/* || true
	rm -rf data/results/* || true

backup-results: ## Backup results to timestamped archive
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	tar -czf "results_backup_$$TIMESTAMP.tar.gz" results/ data/results/ || true; \
	echo "Results backed up to results_backup_$$TIMESTAMP.tar.gz"

# =============================================================================
# Maintenance
# =============================================================================
clean: ## Clean build artifacts and cache files
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete

clean-docker: ## Clean Docker images and containers
	docker system prune -f
	docker images $(DOCKER_IMAGE) -q | xargs -r docker rmi

clean-all: clean clean-docker clean-results ## Clean everything

deps-update: ## Update dependencies
	pip-compile requirements.in
	pip-compile requirements-dev.in

deps-install: ## Install exact dependencies from lock files
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

# =============================================================================
# Release Management
# =============================================================================
version: ## Show current version
	@$(PYTHON) -c "import tiny_llm_profiler; print(tiny_llm_profiler.__version__)"

release-check: ## Check if ready for release
	@echo "Checking release readiness..."
	@make validate
	@make test-all
	@make build
	@echo "Release checks passed!"

release: ## Create and publish release
	@echo "Creating release..."
	semantic-release version
	git push --follow-tags
	$(PYTHON) -m build
	twine upload dist/*

# =============================================================================
# Utilities
# =============================================================================
info: ## Show project information
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not available')"
	@echo "Git: $$(git --version 2>/dev/null || echo 'Not available')"
	@echo ""
	@echo "Project structure:"
	@tree -L 2 -I '__pycache__|*.pyc|.git|venv|.venv' . 2>/dev/null || find . -type d -not -path '*/.*' | head -20

check-deps: ## Check for dependency issues
	pip check
	pip list --outdated

serve: ## Start development server
	$(PYTHON) -m tiny_llm_profiler.cli serve --debug

# =============================================================================
# Platform-specific targets
# =============================================================================
ifeq ($(OS),Windows_NT)
    # Windows-specific commands
    SHELL := cmd.exe
    install-windows: ## Windows-specific installation
	@echo "Windows installation not yet implemented"
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        install-linux: install-hardware ## Linux-specific installation
	@echo "Linux installation complete"
    endif
    ifeq ($(UNAME_S),Darwin)
        install-macos: ## macOS-specific installation
	@echo "macOS installation"
	brew install libusb
	$(MAKE) install-dev
    endif
endif