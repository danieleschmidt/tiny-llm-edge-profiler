#!/bin/bash
# Production deployment script for Tiny LLM Edge Profiler
# Supports Docker, Kubernetes, and bare metal deployments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${REPO_ROOT}/deployment"

# Default values
DEPLOYMENT_TYPE="docker"
ENVIRONMENT="production"
VERSION="0.1.0"
NAMESPACE="profiler"
REGISTRY="terragon-labs"
DRY_RUN=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

# Show usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Tiny LLM Edge Profiler to various environments

OPTIONS:
    -t, --type TYPE         Deployment type: docker, kubernetes, bare-metal (default: docker)
    -e, --env ENV          Environment: development, staging, production (default: production)
    -v, --version VERSION  Version to deploy (default: 0.1.0)
    -n, --namespace NS     Kubernetes namespace (default: profiler)
    -r, --registry REG     Container registry (default: terragon-labs)
    --dry-run             Show what would be done without executing
    --verbose             Enable verbose output
    -h, --help            Show this help message

EXAMPLES:
    $0 --type docker --env production
    $0 --type kubernetes --namespace my-profiler --version 0.2.0
    $0 --type bare-metal --env development
    $0 --dry-run --verbose

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate deployment type
validate_deployment_type() {
    case $DEPLOYMENT_TYPE in
        docker|kubernetes|bare-metal)
            ;;
        *)
            error "Invalid deployment type: $DEPLOYMENT_TYPE"
            error "Valid types: docker, kubernetes, bare-metal"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for $DEPLOYMENT_TYPE deployment..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            if ! command -v docker &> /dev/null; then
                error "Docker is not installed or not in PATH"
                exit 1
            fi
            
            if ! command -v docker-compose &> /dev/null; then
                error "Docker Compose is not installed or not in PATH"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                error "kubectl is not installed or not in PATH"
                exit 1
            fi
            
            if ! kubectl cluster-info &> /dev/null; then
                error "Cannot connect to Kubernetes cluster"
                exit 1
            fi
            ;;
        bare-metal)
            if ! command -v python3 &> /dev/null; then
                error "Python 3 is not installed or not in PATH"
                exit 1
            fi
            
            if ! command -v pip3 &> /dev/null; then
                error "pip3 is not installed or not in PATH"
                exit 1
            fi
            ;;
    esac
    
    success "Prerequisites check passed"
}

# Build container image
build_image() {
    log "Building container image..."
    
    local image_tag="${REGISTRY}/tiny-llm-profiler:${VERSION}"
    
    if [[ $DRY_RUN == true ]]; then
        log "[DRY RUN] Would build image: $image_tag"
        return
    fi
    
    cd "$REPO_ROOT"
    
    docker build \
        -f deployment/docker/Dockerfile \
        -t "$image_tag" \
        --build-arg VERSION="$VERSION" \
        --build-arg ENVIRONMENT="$ENVIRONMENT" \
        .
    
    success "Container image built: $image_tag"
}

# Deploy with Docker Compose
deploy_docker() {
    log "Deploying with Docker Compose..."
    
    cd "${DEPLOYMENT_DIR}/docker"
    
    # Create necessary directories
    mkdir -p data logs cache
    
    # Set environment variables
    export PROFILER_VERSION="$VERSION"
    export PROFILER_ENV="$ENVIRONMENT"
    export PROFILER_REGISTRY="$REGISTRY"
    
    if [[ $DRY_RUN == true ]]; then
        log "[DRY RUN] Would run: docker-compose up -d"
        return
    fi
    
    # Pull and start services
    docker-compose pull
    docker-compose up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Check health
    if docker-compose ps | grep -q "Up (healthy)"; then
        success "Docker deployment completed successfully"
    else
        warn "Some services may not be healthy. Check with: docker-compose ps"
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log "Deploying to Kubernetes..."
    
    cd "${DEPLOYMENT_DIR}/kubernetes"
    
    # Create namespace if it doesn't exist
    if [[ $DRY_RUN == true ]]; then
        log "[DRY RUN] Would create namespace: $NAMESPACE"
        log "[DRY RUN] Would apply Kubernetes manifests"
        return
    fi
    
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply manifests
    find . -name "*.yaml" -exec kubectl apply -f {} -n "$NAMESPACE" \;
    
    # Wait for deployment to be ready
    log "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/tiny-llm-profiler -n "$NAMESPACE"
    
    # Get service information
    kubectl get services -n "$NAMESPACE"
    
    success "Kubernetes deployment completed successfully"
}

# Deploy bare metal
deploy_bare_metal() {
    log "Deploying on bare metal..."
    
    cd "$REPO_ROOT"
    
    if [[ $DRY_RUN == true ]]; then
        log "[DRY RUN] Would install Python package"
        log "[DRY RUN] Would create systemd service"
        return
    fi
    
    # Install the package
    pip3 install -e .
    
    # Create configuration directory
    sudo mkdir -p /etc/tiny-llm-profiler
    sudo cp "deployment/config/${ENVIRONMENT}.yaml" /etc/tiny-llm-profiler/config.yaml
    
    # Create systemd service
    sudo tee /etc/systemd/system/tiny-llm-profiler.service > /dev/null << EOF
[Unit]
Description=Tiny LLM Edge Profiler
After=network.target

[Service]
Type=simple
User=profiler
Group=profiler
WorkingDirectory=/opt/tiny-llm-profiler
ExecStart=/usr/local/bin/tiny-profiler --config /etc/tiny-llm-profiler/config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Create user and directories
    sudo useradd -r -s /bin/false profiler || true
    sudo mkdir -p /opt/tiny-llm-profiler /var/log/tiny-llm-profiler /var/lib/tiny-llm-profiler
    sudo chown -R profiler:profiler /opt/tiny-llm-profiler /var/log/tiny-llm-profiler /var/lib/tiny-llm-profiler
    
    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable tiny-llm-profiler
    sudo systemctl start tiny-llm-profiler
    
    success "Bare metal deployment completed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    case $DEPLOYMENT_TYPE in
        docker)
            if docker-compose ps | grep -q "Up"; then
                success "Docker services are running"
            else
                error "Docker services are not running properly"
                exit 1
            fi
            ;;
        kubernetes)
            if kubectl get pods -n "$NAMESPACE" | grep -q "Running"; then
                success "Kubernetes pods are running"
            else
                error "Kubernetes pods are not running properly"
                exit 1
            fi
            ;;
        bare-metal)
            if systemctl is-active --quiet tiny-llm-profiler; then
                success "Systemd service is running"
            else
                error "Systemd service is not running properly"
                exit 1
            fi
            ;;
    esac
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    # Add cleanup logic if needed
}

# Main deployment function
main() {
    log "Starting deployment of Tiny LLM Edge Profiler"
    log "Type: $DEPLOYMENT_TYPE, Environment: $ENVIRONMENT, Version: $VERSION"
    
    if [[ $DRY_RUN == true ]]; then
        warn "DRY RUN MODE - No actual changes will be made"
    fi
    
    check_prerequisites
    
    case $DEPLOYMENT_TYPE in
        docker)
            build_image
            deploy_docker
            ;;
        kubernetes)
            build_image
            deploy_kubernetes
            ;;
        bare-metal)
            deploy_bare_metal
            ;;
    esac
    
    if [[ $DRY_RUN == false ]]; then
        verify_deployment
        success "Deployment completed successfully!"
        
        case $DEPLOYMENT_TYPE in
            docker)
                log "Access the application at: http://localhost:8080"
                log "View logs with: docker-compose logs -f"
                ;;
            kubernetes)
                log "Forward port with: kubectl port-forward service/tiny-llm-profiler-service 8080:8080 -n $NAMESPACE"
                log "View logs with: kubectl logs -f deployment/tiny-llm-profiler -n $NAMESPACE"
                ;;
            bare-metal)
                log "View logs with: journalctl -u tiny-llm-profiler -f"
                log "View status with: systemctl status tiny-llm-profiler"
                ;;
        esac
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Parse arguments and run main
parse_args "$@"
validate_deployment_type
main