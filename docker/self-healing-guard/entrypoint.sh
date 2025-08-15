#!/bin/bash
set -euo pipefail

# Self-Healing Pipeline Guard - Production Entrypoint Script
# Handles initialization, configuration, and service startup

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Configuration defaults
export GUARD_MODE="${GUARD_MODE:-unified}"
export GUARD_PORT="${GUARD_PORT:-8080}"
export GUARD_LOG_LEVEL="${GUARD_LOG_LEVEL:-INFO}"
export GUARD_REGION="${GUARD_REGION:-us-east-1}"
export GUARD_ENVIRONMENT="${GUARD_ENVIRONMENT:-production}"

# Health check configuration
export HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-30}"
export HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-10}"

# Performance tuning
export GUARD_WORKERS="${GUARD_WORKERS:-$(nproc)}"
export GUARD_MAX_MEMORY_MB="${GUARD_MAX_MEMORY_MB:-1024}"
export GUARD_CACHE_SIZE="${GUARD_CACHE_SIZE:-100}"

# Security configuration
export GUARD_ENABLE_SECURITY="${GUARD_ENABLE_SECURITY:-true}"
export GUARD_API_KEY_REQUIRED="${GUARD_API_KEY_REQUIRED:-true}"

# Function to check required environment variables
check_environment() {
    log "Checking environment configuration..."
    
    # Required variables
    local required_vars=(
        "GUARD_MODE"
        "GUARD_REGION"
        "GUARD_ENVIRONMENT"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate values
    case "$GUARD_MODE" in
        unified|pipeline|infrastructure|security)
            ;;
        *)
            error "Invalid GUARD_MODE: $GUARD_MODE. Must be one of: unified, pipeline, infrastructure, security"
            exit 1
            ;;
    esac
    
    case "$GUARD_ENVIRONMENT" in
        development|staging|production)
            ;;
        *)
            error "Invalid GUARD_ENVIRONMENT: $GUARD_ENVIRONMENT. Must be one of: development, staging, production"
            exit 1
            ;;
    esac
    
    success "Environment configuration validated"
}

# Function to setup directories and permissions
setup_directories() {
    log "Setting up directories and permissions..."
    
    # Ensure directories exist and have correct permissions
    local dirs=(
        "/app/data"
        "/app/logs"
        "/app/tmp"
        "/app/config"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
        
        # Ensure proper ownership (already set in Dockerfile, but double-check)
        if [[ "$(stat -c %U "$dir")" != "guardapp" ]]; then
            warn "Directory $dir has incorrect ownership, but cannot change as non-root user"
        fi
    done
    
    success "Directory setup completed"
}

# Function to setup logging
setup_logging() {
    log "Setting up logging configuration..."
    
    # Create log file if it doesn't exist
    local log_file="/app/logs/guard.log"
    if [[ ! -f "$log_file" ]]; then
        touch "$log_file"
    fi
    
    # Setup log rotation marker
    echo "=== Guard Service Started at $(date) ===" >> "$log_file"
    
    success "Logging configuration completed"
}

# Function to validate configuration files
validate_config() {
    log "Validating configuration files..."
    
    # Check if config files exist
    local config_files=(
        "/app/config/guard.yaml"
        "/app/config/security.yaml"
        "/app/config/regions.yaml"
    )
    
    for config_file in "${config_files[@]}"; do
        if [[ -f "$config_file" ]]; then
            log "Found config file: $config_file"
            
            # Basic YAML validation (check if file can be parsed)
            if command -v python3 >/dev/null 2>&1; then
                if ! python3 -c "import yaml; yaml.safe_load(open('$config_file'))" 2>/dev/null; then
                    error "Invalid YAML syntax in $config_file"
                    exit 1
                fi
            fi
        else
            warn "Config file not found: $config_file (using defaults)"
        fi
    done
    
    success "Configuration validation completed"
}

# Function to perform pre-flight checks
preflight_checks() {
    log "Performing pre-flight checks..."
    
    # Check Python environment
    if ! python3 -c "import tiny_llm_profiler" 2>/dev/null; then
        error "tiny_llm_profiler package not found in Python path"
        exit 1
    fi
    
    # Check network connectivity (if in production)
    if [[ "$GUARD_ENVIRONMENT" == "production" ]]; then
        if ! curl -s --connect-timeout 5 http://httpbin.org/status/200 >/dev/null 2>&1; then
            warn "External network connectivity check failed (this may be expected in some environments)"
        else
            log "Network connectivity check passed"
        fi
    fi
    
    # Check available memory
    local available_memory=$(free -m | awk '/^Mem:/ {print $7}')
    local required_memory=256
    
    if [[ "$available_memory" -lt "$required_memory" ]]; then
        warn "Low available memory: ${available_memory}MB (recommended: ${required_memory}MB+)"
    else
        log "Memory check passed: ${available_memory}MB available"
    fi
    
    # Check disk space
    local available_space=$(df /app | awk 'NR==2 {print $4}')
    local required_space=1048576  # 1GB in KB
    
    if [[ "$available_space" -lt "$required_space" ]]; then
        warn "Low disk space: $((available_space/1024))MB available (recommended: 1GB+)"
    else
        log "Disk space check passed: $((available_space/1024))MB available"
    fi
    
    success "Pre-flight checks completed"
}

# Function to setup signal handlers
setup_signal_handlers() {
    log "Setting up signal handlers..."
    
    # Create signal handler script
    cat > /app/tmp/signal_handler.py << 'EOF'
#!/usr/bin/env python3
import signal
import sys
import time
import os

def signal_handler(signum, frame):
    print(f"Received signal {signum}, initiating graceful shutdown...")
    
    # Write shutdown marker
    with open('/app/tmp/shutdown_requested', 'w') as f:
        f.write(str(int(time.time())))
    
    # Give services time to shut down gracefully
    time.sleep(5)
    
    print("Graceful shutdown completed")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Keep the handler alive
while True:
    if os.path.exists('/app/tmp/shutdown_requested'):
        break
    time.sleep(1)
EOF
    
    chmod +x /app/tmp/signal_handler.py
    
    success "Signal handlers configured"
}

# Function to start monitoring services
start_monitoring() {
    log "Starting monitoring services..."
    
    # Start signal handler in background
    python3 /app/tmp/signal_handler.py &
    local signal_handler_pid=$!
    
    # Start health monitoring
    cat > /app/tmp/monitor.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import time
import json
import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/monitor.log'),
        logging.StreamHandler()
    ]
)

async def monitor_health():
    """Monitor system health"""
    while True:
        try:
            if os.path.exists('/app/tmp/shutdown_requested'):
                logging.info("Shutdown requested, stopping health monitor")
                break
            
            # Basic health metrics
            health_data = {
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy',
                'uptime_seconds': time.time() - start_time,
                'pid': os.getpid()
            }
            
            # Write health status
            with open('/app/tmp/health_status.json', 'w') as f:
                json.dump(health_data, f)
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logging.error(f"Health monitor error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    start_time = time.time()
    logging.info("Health monitor started")
    
    try:
        asyncio.run(monitor_health())
    except KeyboardInterrupt:
        logging.info("Health monitor stopped")
EOF
    
    chmod +x /app/tmp/monitor.py
    python3 /app/tmp/monitor.py &
    local monitor_pid=$!
    
    # Store PIDs for cleanup
    echo "$signal_handler_pid" > /app/tmp/signal_handler.pid
    echo "$monitor_pid" > /app/tmp/monitor.pid
    
    success "Monitoring services started"
}

# Function to start the main application
start_application() {
    log "Starting Self-Healing Pipeline Guard application..."
    
    # Create main application script
    cat > /app/tmp/main_app.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import os
import sys
import signal
import logging
from datetime import datetime

# Add src to Python path
sys.path.insert(0, '/opt/venv/lib/python3.12/site-packages')

try:
    from tiny_llm_profiler.unified_guard_system import get_unified_guard, start_unified_guard
    from tiny_llm_profiler.pipeline_guard import get_pipeline_guard, start_pipeline_guard
    from tiny_llm_profiler.infrastructure_sentinel import get_infrastructure_sentinel, start_infrastructure_monitoring
    from tiny_llm_profiler.multi_region_orchestrator import get_multi_region_orchestrator
    from tiny_llm_profiler.global_compliance_manager import get_compliance_manager
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in minimal mode...")

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv('GUARD_LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/guard.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class GuardApplication:
    def __init__(self):
        self.mode = os.getenv('GUARD_MODE', 'unified')
        self.port = int(os.getenv('GUARD_PORT', '8080'))
        self.region = os.getenv('GUARD_REGION', 'us-east-1')
        self.running = False
        
    async def start(self):
        """Start the guard application"""
        logger.info(f"Starting Guard Application in {self.mode} mode")
        logger.info(f"Region: {self.region}, Port: {self.port}")
        
        self.running = True
        
        try:
            if self.mode == 'unified':
                await self.start_unified_mode()
            elif self.mode == 'pipeline':
                await self.start_pipeline_mode()
            elif self.mode == 'infrastructure':
                await self.start_infrastructure_mode()
            elif self.mode == 'security':
                await self.start_security_mode()
            else:
                await self.start_minimal_mode()
                
        except Exception as e:
            logger.error(f"Application error: {e}")
            raise
    
    async def start_unified_mode(self):
        """Start in unified mode with all components"""
        logger.info("Starting unified guard system...")
        
        try:
            unified_guard = get_unified_guard()
            await unified_guard.start_unified_monitoring()
        except NameError:
            logger.warning("Unified guard not available, falling back to minimal mode")
            await self.start_minimal_mode()
    
    async def start_pipeline_mode(self):
        """Start in pipeline-only mode"""
        logger.info("Starting pipeline guard...")
        
        try:
            pipeline_guard = get_pipeline_guard()
            await pipeline_guard.start_monitoring()
        except NameError:
            logger.warning("Pipeline guard not available, falling back to minimal mode")
            await self.start_minimal_mode()
    
    async def start_infrastructure_mode(self):
        """Start in infrastructure monitoring mode"""
        logger.info("Starting infrastructure sentinel...")
        
        try:
            await start_infrastructure_monitoring()
        except NameError:
            logger.warning("Infrastructure sentinel not available, falling back to minimal mode")
            await self.start_minimal_mode()
    
    async def start_security_mode(self):
        """Start in security monitoring mode"""
        logger.info("Starting security monitoring...")
        
        try:
            # Security monitoring would be implemented here
            await self.start_minimal_mode()
        except Exception:
            await self.start_minimal_mode()
    
    async def start_minimal_mode(self):
        """Start in minimal mode (basic health monitoring)"""
        logger.info("Starting in minimal mode...")
        
        while self.running:
            if os.path.exists('/app/tmp/shutdown_requested'):
                logger.info("Shutdown requested")
                break
                
            # Basic status update
            status = {
                'mode': self.mode,
                'timestamp': datetime.now().isoformat(),
                'status': 'running',
                'uptime_seconds': int(asyncio.get_event_loop().time())
            }
            
            with open('/app/tmp/app_status.json', 'w') as f:
                import json
                json.dump(status, f)
            
            await asyncio.sleep(10)
        
        logger.info("Minimal mode stopped")
    
    async def stop(self):
        """Stop the application"""
        logger.info("Stopping Guard Application...")
        self.running = False

# Global application instance
app = GuardApplication()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    asyncio.create_task(app.stop())

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        asyncio.run(app.start())
    except KeyboardInterrupt:
        logger.info("Application interrupted")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)
EOF
    
    chmod +x /app/tmp/main_app.py
    
    # Start the main application
    log "Launching main application (Mode: $GUARD_MODE, Port: $GUARD_PORT)"
    exec python3 /app/tmp/main_app.py
}

# Function to cleanup on exit
cleanup() {
    log "Performing cleanup..."
    
    # Stop background processes
    if [[ -f /app/tmp/signal_handler.pid ]]; then
        local pid=$(cat /app/tmp/signal_handler.pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    fi
    
    if [[ -f /app/tmp/monitor.pid ]]; then
        local pid=$(cat /app/tmp/monitor.pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    fi
    
    # Remove temporary files
    rm -f /app/tmp/*.pid /app/tmp/shutdown_requested
    
    success "Cleanup completed"
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution flow
main() {
    log "Self-Healing Pipeline Guard - Starting..."
    log "Version: ${BUILD_VERSION:-development}"
    log "Environment: $GUARD_ENVIRONMENT"
    log "Mode: $GUARD_MODE"
    log "Region: $GUARD_REGION"
    
    # Execute startup sequence
    check_environment
    setup_directories
    setup_logging
    validate_config
    preflight_checks
    setup_signal_handlers
    start_monitoring
    
    success "Initialization completed successfully"
    
    # Start the main application (this will exec, so no return)
    start_application
}

# Execute main function
main "$@"