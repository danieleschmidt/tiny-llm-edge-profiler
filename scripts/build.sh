#!/bin/bash
# Build script for tiny-llm-edge-profiler
# Provides automated build, test, and release workflows

set -euo pipefail

# Configuration
PROJECT_NAME="tiny-llm-edge-profiler"
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
BUILD_ENV=${BUILD_ENV:-development}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-ghcr.io}
DOCKER_NAMESPACE=${DOCKER_NAMESPACE:-danieleschmidt}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {  
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Build script for $PROJECT_NAME

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    install         Install dependencies and package
    install-dev     Install development dependencies
    test           Run all tests
    test-unit      Run unit tests only
    test-integration Run integration tests
    test-hardware  Run hardware tests (requires devices)
    lint           Run code linting
    format         Format code
    type-check     Run type checking
    security       Run security checks
    build          Build package distributions
    docker-build   Build Docker images
    docker-test    Test in Docker containers
    release        Build and prepare release
    clean          Clean build artifacts
    ci             Run full CI pipeline
    help           Show this help

Options:
    --python-version VERSION    Python version to use (default: $PYTHON_VERSION)
    --build-env ENV            Build environment: development|testing|production (default: $BUILD_ENV)
    --registry REGISTRY        Docker registry (default: $DOCKER_REGISTRY)
    --namespace NAMESPACE      Docker namespace (default: $DOCKER_NAMESPACE)
    --verbose                  Enable verbose output
    --no-cache                 Disable caching for builds
    --parallel                 Run tests in parallel where possible

Examples:
    $0 install-dev
    $0 test --parallel
    $0 docker-build --build-env production
    $0 ci --verbose

EOF
}

# Parse command line arguments
VERBOSE=false
NO_CACHE=false
PARALLEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --build-env)
            BUILD_ENV="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --namespace)
            DOCKER_NAMESPACE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        help|--help|-h)
            show_help
            exit 0
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Check dependencies
check_dependencies() {
    local deps=("python3" "pip" "git")
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is required but not installed"
            exit 1
        fi
    done
    
    # Check Python version
    local python_version
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Using Python $python_version"
}

# Install package and dependencies
install_package() {
    log_info "Installing $PROJECT_NAME..."
    
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "pyproject.toml not found. Are you in the project root?"
        exit 1
    fi
    
    pip install -e .
    log_success "Package installed successfully"
}

# Install development dependencies
install_dev_dependencies() {
    log_info "Installing development dependencies..."
    
    if [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
    fi
    
    install_package
    
    # Install pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        log_success "Pre-commit hooks installed"
    fi
    
    log_success "Development environment ready"
}

# Run tests
run_tests() {
    local test_type="${1:-all}"
    local extra_args=""
    
    if [[ "$PARALLEL" == "true" ]]; then
        extra_args="$extra_args -n auto"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        extra_args="$extra_args -v"
    fi
    
    log_info "Running $test_type tests..."
    
    case $test_type in
        "unit")
            pytest tests/unit/ $extra_args --cov=src/tiny_llm_profiler --cov-report=term-missing
            ;;
        "integration")
            pytest tests/integration/ $extra_args --timeout=300
            ;;
        "hardware")
            log_warning "Hardware tests require physical devices connected"
            pytest tests/hardware/ $extra_args --hardware --timeout=600
            ;;
        "performance")
            pytest tests/performance/ $extra_args --benchmark-only
            ;;
        "all"|*)
            pytest tests/ $extra_args --cov=src/tiny_llm_profiler --cov-report=html --cov-report=term-missing
            ;;
    esac
    
    log_success "$test_type tests completed"
}

# Code quality checks
run_linting() {
    log_info "Running code linting..."
    
    local exit_code=0
    
    # Run each linter and capture exit codes
    if ! flake8 src/ tests/; then
        log_error "flake8 found issues"
        exit_code=1
    fi
    
    if ! ruff check src/ tests/; then
        log_error "ruff found issues"
        exit_code=1
    fi
    
    if ! pydocstyle src/; then
        log_error "pydocstyle found issues"
        exit_code=1
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "All linting checks passed"
    else
        log_error "Linting failed"
        exit $exit_code
    fi
}

# Format code
format_code() {
    log_info "Formatting code..."
    
    black src/ tests/
    isort src/ tests/
    
    log_success "Code formatted successfully"
}

# Type checking
run_type_check() {
    log_info "Running type checking..."
    
    if ! mypy src/tiny_llm_profiler; then
        log_error "Type checking failed"
        exit 1
    fi
    
    log_success "Type checking passed"
}

# Security checks
run_security_checks() {
    log_info "Running security checks..."
    
    local exit_code=0
    
    if ! bandit -r src/; then
        log_error "bandit found security issues"
        exit_code=1
    fi
    
    if ! safety check; then
        log_error "safety found vulnerable dependencies"
        exit_code=1
    fi
    
    # Run semgrep if available
    if command -v semgrep &> /dev/null; then
        if ! semgrep --config=auto src/; then
            log_error "semgrep found security issues"
            exit_code=1
        fi
    else
        log_warning "semgrep not available, skipping advanced security scan"
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "All security checks passed"
    else
        log_error "Security checks failed"
        exit $exit_code
    fi
}

# Build package
build_package() {
    log_info "Building package..."
    
    # Clean previous builds
    rm -rf build/ dist/ *.egg-info/
    
    # Build wheel and source distribution
    python -m build
    
    # Check package
    twine check dist/*
    
    log_success "Package built successfully"
    ls -la dist/
}

# Docker build
docker_build() {
    local target="${1:-production}"
    local tag_suffix=""
    
    if [[ "$target" != "production" ]]; then
        tag_suffix="-$target"
    fi
    
    local image_tag="$DOCKER_REGISTRY/$DOCKER_NAMESPACE/$PROJECT_NAME:latest$tag_suffix"
    local build_args=""
    
    if [[ "$NO_CACHE" == "true" ]]; then
        build_args="--no-cache"
    fi
    
    log_info "Building Docker image: $image_tag (target: $target)"
    
    docker build $build_args --target "$target" -t "$image_tag" .
    
    # Also tag with git commit if in git repo
    if git rev-parse --git-dir > /dev/null 2>&1; then
        local git_hash
        git_hash=$(git rev-parse --short HEAD)
        local commit_tag="$DOCKER_REGISTRY/$DOCKER_NAMESPACE/$PROJECT_NAME:$git_hash$tag_suffix"
        docker tag "$image_tag" "$commit_tag"
        log_info "Also tagged as: $commit_tag"
    fi
    
    log_success "Docker image built successfully"
}

# Test in Docker
docker_test() {
    log_info "Running tests in Docker..."
    
    # Build test image
    docker_build "testing"
    
    # Run tests
    docker-compose up --abort-on-container-exit test
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Docker tests passed"
    else
        log_error "Docker tests failed"
        exit $exit_code
    fi
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    
    # Python artifacts
    rm -rf build/ dist/ *.egg-info/
    rm -rf .pytest_cache/ .coverage htmlcov/
    rm -rf .mypy_cache/ .ruff_cache/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    
    # Docker cleanup
    if command -v docker &> /dev/null; then
        docker system prune -f
    fi
    
    log_success "Build artifacts cleaned"
}

# Release preparation
prepare_release() {
    log_info "Preparing release..."
    
    # Run full test suite
    log_info "Running pre-release checks..."
    run_linting
    run_type_check
    run_security_checks
    run_tests "unit"
    
    # Build package
    build_package
    
    # Build production Docker image
    docker_build "production"
    
    log_success "Release preparation completed"
    log_info "Next steps:"
    log_info "1. Test the built package: pip install dist/*.whl"
    log_info "2. Test the Docker image: docker run $DOCKER_REGISTRY/$DOCKER_NAMESPACE/$PROJECT_NAME:latest"
    log_info "3. Push to registry: docker push $DOCKER_REGISTRY/$DOCKER_NAMESPACE/$PROJECT_NAME:latest"
}

# CI pipeline
run_ci() {
    log_info "Running CI pipeline..."
    
    local start_time
    start_time=$(date +%s)
    
    # Check dependencies
    check_dependencies
    
    # Install dependencies
    install_dev_dependencies
    
    # Code quality
    format_code
    run_linting
    run_type_check
    run_security_checks
    
    # Tests
    run_tests "unit"
    run_tests "integration"
    
    # Build
    build_package
    docker_build "production"
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "CI pipeline completed in ${duration}s"
}

# Main command dispatch
main() {
    local command="${COMMAND:-help}"
    
    case $command in
        "install")
            check_dependencies
            install_package
            ;;
        "install-dev")
            check_dependencies
            install_dev_dependencies
            ;;
        "test")
            run_tests "all"
            ;;
        "test-unit")
            run_tests "unit"
            ;;
        "test-integration")
            run_tests "integration"
            ;;
        "test-hardware")
            run_tests "hardware"
            ;;
        "test-performance")
            run_tests "performance"
            ;;
        "lint")
            run_linting
            ;;
        "format")
            format_code
            ;;
        "type-check")
            run_type_check
            ;;
        "security")
            run_security_checks
            ;;
        "build")
            build_package
            ;;
        "docker-build")
            docker_build "${BUILD_ENV}"
            ;;
        "docker-test")
            docker_test
            ;;
        "release")
            prepare_release
            ;;
        "clean")
            clean_build
            ;;
        "ci")
            run_ci
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"