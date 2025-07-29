# Makefile for tiny-llm-edge-profiler
# Provides convenient commands for development workflows

.PHONY: help install install-dev test test-unit test-integration test-hardware
.PHONY: lint format type-check security-check pre-commit clean build
.PHONY: docs docs-serve docs-build docker docker-dev docker-test
.PHONY: benchmark profile release

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

# Testing targets
test: ## Run all tests
	pytest tests/ -v --cov=src/tiny_llm_profiler --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	pytest tests/unit/ -v --cov=src/tiny_llm_profiler

test-integration: ## Run integration tests
	pytest tests/integration/ -v --timeout=300

test-hardware: ## Run hardware tests (requires physical devices)
	pytest tests/hardware/ -v --hardware --timeout=600

test-performance: ## Run performance benchmarks
	pytest tests/performance/ -v --benchmark-only

# Code quality targets
lint: ## Run all linters
	flake8 src/ tests/
	ruff check src/ tests/
	pydocstyle src/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

type-check: ## Run type checking with mypy
	mypy src/tiny_llm_profiler

security-check: ## Run security checks
	bandit -r src/
	safety check
	semgrep --config=auto src/

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Cleanup targets
clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Clean all generated files including docker
	docker system prune -f
	rm -rf node_modules/

# Build targets
build: clean ## Build wheel and source distribution
	python -m build

build-check: build ## Build and check package
	twine check dist/*

# Documentation targets
docs: ## Generate documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve --dev-addr=0.0.0.0:8000

docs-build: ## Build documentation for deployment
	mkdocs build --strict

# Docker targets
docker: ## Build production docker image
	docker build --target production -t tiny-llm-profiler:latest .

docker-dev: ## Build and run development environment
	docker-compose up -d dev

docker-test: ## Run tests in docker
	docker-compose up test

docker-hardware: ## Run hardware tests in docker (requires privileged mode)
	docker-compose up hardware-test

docker-docs: ## Serve documentation in docker
	docker-compose up -d docs

docker-quality: ## Run code quality checks in docker
	docker-compose up quality

docker-benchmark: ## Run benchmarks in docker
	docker-compose up benchmark

# Development workflow targets
dev-setup: install-dev ## Complete development environment setup
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify installation"

dev-test: format lint type-check test-unit ## Quick development test cycle
	@echo "Development tests passed!"

ci-test: lint type-check security-check test ## Full CI test suite
	@echo "CI tests passed!"

# Release targets
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version  
	bump2version minor

version-major: ## Bump major version
	bump2version major

release-check: ## Check if ready for release
	@echo "Checking release readiness..."
	@python -c "
import subprocess
import sys

checks = [
    ('Git clean', 'git status --porcelain'),
    ('Tests pass', 'python -m pytest tests/unit/ -q'),
    ('Security check', 'bandit -r src/ -q'),
    ('Build works', 'python -m build'),
]

for name, cmd in checks:
    result = subprocess.run(cmd.split(), capture_output=True)
    status = '✓' if result.returncode == 0 else '✗'
    print(f'{status} {name}')
    if result.returncode != 0:
        print(f'  Error: {result.stderr.decode()}')
        sys.exit(1)

print('✓ Ready for release!')
"

# Utility targets
hardware-check: ## Check for connected hardware devices
	@python -c "
from tiny_llm_profiler import EdgeProfiler
import sys

platforms = ['esp32', 'stm32f4', 'rp2040']
total_devices = 0

for platform in platforms:
    try:
        profiler = EdgeProfiler(platform=platform)
        devices = profiler.discover_devices()
        total_devices += len(devices)
        print(f'{platform}: {len(devices)} devices')
        for device in devices:
            print(f'  - {device.name} at {device.port}')
    except Exception as e:
        print(f'{platform}: Error - {e}')

if total_devices == 0:
    print('No hardware devices found. Hardware tests will be skipped.')
    sys.exit(1)
else:
    print(f'Total: {total_devices} devices available for testing')
"

deps-update: ## Update dependencies
	pip-compile requirements.in
	pip-compile requirements-dev.in
	pre-commit autoupdate

# Profiling targets
profile-memory: ## Profile memory usage
	python -m memory_profiler scripts/profile_memory.py

profile-cpu: ## Profile CPU usage
	python -m cProfile -s cumulative scripts/profile_cpu.py

# Database/Config management (if needed)
config-validate: ## Validate configuration files
	@echo "Validating configuration files..."
	python -c "
import yaml
import json
import sys
from pathlib import Path

config_files = [
    ('pyproject.toml', 'toml'),
    ('.pre-commit-config.yaml', 'yaml'),
    ('docker-compose.yml', 'yaml'),
]

for file_path, file_type in config_files:
    if Path(file_path).exists():
        try:
            if file_type == 'yaml':
                with open(file_path) as f:
                    yaml.safe_load(f)
            elif file_type == 'toml':
                import tomllib
                with open(file_path, 'rb') as f:
                    tomllib.load(f)
            print(f'✓ {file_path}')
        except Exception as e:
            print(f'✗ {file_path}: {e}')
            sys.exit(1)

print('All configuration files are valid!')
"

# Environment info
env-info: ## Display environment information
	@echo "Environment Information:"
	@echo "======================"
	@python --version
	@echo "Python executable: $$(which python)"
	@pip --version
	@echo "Git version: $$(git --version)"
	@echo "Current branch: $$(git branch --show-current)"
	@echo "Uncommitted changes: $$(git status --porcelain | wc -l)"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Make version: $$(make --version | head -1)"