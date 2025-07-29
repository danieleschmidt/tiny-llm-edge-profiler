# Container Security Configuration

This document outlines the container security scanning and hardening configuration for the tiny-llm-edge-profiler project.

## Security Scanning Tools

### 1. Trivy Scanner

**Configuration**: Integrated into CI/CD pipeline

```yaml
# In GitHub Actions workflow
- name: Run Trivy security scan
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'tiny-llm-profiler:test'
    format: 'sarif'
    output: 'trivy-results.sarif'
    severity: 'CRITICAL,HIGH'
    exit-code: '1'
```

**Scan Coverage**:
- OS vulnerabilities (Alpine Linux base)
- Python package vulnerabilities 
- Known CVEs in dependencies
- Misconfigurations in Dockerfile

### 2. Docker Scout (Recommended)

**Setup for local development**:
```bash
# Enable Docker Scout
docker scout cves tiny-llm-profiler:latest

# Generate SBOM for container
docker scout sbom tiny-llm-profiler:latest
```

### 3. Grype Scanner (Alternative)

**Installation and usage**:
```bash
# Install Grype
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin

# Scan container image
grype tiny-llm-profiler:latest -o json > grype-report.json
```

## Dockerfile Security Hardening

### Current Security Measures

```dockerfile
# Multi-stage build to minimize attack surface
FROM python:3.11-alpine AS builder
# ... build stage

FROM python:3.11-alpine AS production

# Create non-root user
RUN addgroup -g 1001 -S profiler && \
    adduser -u 1001 -S profiler -G profiler -h /app

# Install security updates
RUN apk upgrade --no-cache && \
    apk add --no-cache \
        ca-certificates \
        tzdata && \
    rm -rf /var/cache/apk/*

# Set security-focused environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy application with proper ownership
COPY --from=builder --chown=profiler:profiler /app /app

# Drop to non-root user
USER profiler

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD tiny-profiler --version || exit 1
```

### Recommended Additional Hardening

#### 1. Distroless Base Image (Future)
```dockerfile
# More secure base image option
FROM gcr.io/distroless/python3:latest
```

#### 2. Read-Only Root Filesystem
```dockerfile
# In docker-compose.yml or deployment
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp:noexec,nosuid,size=100m
```

#### 3. Resource Limits
```dockerfile
# In docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '1.0'
      memory: 512M
    reservations:
      cpus: '0.5'
      memory: 256M
```

## Container Runtime Security

### 1. AppArmor/SELinux Profiles

**AppArmor profile** (`apparmor-tiny-llm-profiler`):
```apparmor
#include <tunables/global>

profile tiny-llm-profiler flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  #include <abstractions/python>
  
  # Allow network access for device communication
  network inet stream,
  network inet6 stream,
  network unix stream,
  
  # Serial device access
  /dev/ttyUSB* rw,
  /dev/ttyACM* rw,
  /dev/serial/** rw,
  
  # Application files
  /app/** r,
  /app/bin/* ix,
  
  # Temporary files
  /tmp/** rw,
  
  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_module,
}
```

### 2. Seccomp Profiles

**Custom seccomp profile** (`seccomp-tiny-llm-profiler.json`):
```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
  "syscalls": [
    {
      "names": [
        "read", "write", "open", "close", "stat", "fstat", "lstat",
        "poll", "lseek", "mmap", "mprotect", "munmap", "brk",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn",
        "ioctl", "access", "pipe", "select", "sched_yield",
        "mremap", "msync", "mincore", "madvise", "socket",
        "connect", "accept", "sendto", "recvfrom", "sendmsg",
        "recvmsg", "shutdown", "bind", "listen", "getsockname",
        "getpeername", "socketpair", "setsockopt", "getsockopt"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

### 3. Runtime Monitoring

**Falco rules** for runtime security:
```yaml
# /etc/falco/rules.d/tiny-llm-profiler.yaml
- rule: Suspicious tiny-llm-profiler network activity
  desc: Detect unexpected network connections from profiler
  condition: >
    spawned_process and
    proc.name=tiny-profiler and
    (outbound_connection and not allowed_profiler_destinations)
  output: >
    Unexpected network connection from tiny-llm-profiler
    (user=%user.name command=%proc.cmdline connection=%fd.name)
  priority: WARNING

- rule: Unauthorized file access from profiler
  desc: Detect access to sensitive files by profiler
  condition: >
    open_read and
    proc.name=tiny-profiler and
    fd.name pmatch (sensitive_file_paths)
  output: >
    Unauthorized file access by tiny-llm-profiler
    (user=%user.name file=%fd.name command=%proc.cmdline)
  priority: ERROR
```

## Vulnerability Management

### 1. Automated Scanning Schedule

```yaml
# GitHub Actions scheduled scan
on:
  schedule:
    # Run weekly security scans on Sundays at 2 AM UTC
    - cron: '0 2 * * 0'

jobs:
  security-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Pull latest base images
        run: docker pull python:3.11-alpine
        
      - name: Rebuild container
        run: docker build -t tiny-llm-profiler:security-scan .
        
      - name: Run comprehensive security scan
        run: |
          # Trivy scan with all severity levels
          trivy image --format json --output trivy-full-report.json tiny-llm-profiler:security-scan
          
          # Grype scan for additional coverage
          grype tiny-llm-profiler:security-scan -o json > grype-full-report.json
          
          # Docker Scout analysis
          docker scout cves tiny-llm-profiler:security-scan --format json > scout-report.json
```

### 2. Vulnerability Response Process

1. **Critical (CVSS 9.0-10.0)**: Immediate response within 24 hours
2. **High (CVSS 7.0-8.9)**: Response within 72 hours  
3. **Medium (CVSS 4.0-6.9)**: Response within 1 week
4. **Low (CVSS 0.1-3.9)**: Response within 1 month

### 3. Base Image Update Strategy

```bash
#!/bin/bash
# scripts/update-base-image.sh

# Check for Alpine security updates
alpine_version=$(docker run --rm python:3.11-alpine cat /etc/alpine-release)
echo "Current Alpine version: $alpine_version"

# Pull latest base image
docker pull python:3.11-alpine

# Rebuild and test
docker build -t tiny-llm-profiler:updated .
docker run --rm tiny-llm-profiler:updated tiny-profiler --version

# Run security scan on updated image
trivy image tiny-llm-profiler:updated
```

## Supply Chain Security

### 1. Image Signing with Cosign

```bash
# Generate signing key
cosign generate-key-pair

# Sign the container image
cosign sign --key cosign.key tiny-llm-profiler:latest

# Verify signature
cosign verify --key cosign.pub tiny-llm-profiler:latest
```

### 2. SLSA Provenance

```yaml
# GitHub Actions with SLSA provenance
- name: Generate SLSA provenance
  uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v1.7.0
  with:
    image: tiny-llm-profiler
    registry-username: ${{ github.actor }}
    registry-password: ${{ secrets.GITHUB_TOKEN }}
```

### 3. Software Bill of Materials (SBOM)

```bash
# Generate SBOM for container
syft tiny-llm-profiler:latest -o json > container-sbom.json

# Generate Python package SBOM
python scripts/generate_sbom.py --format json xml
```

## Security Testing

### 1. Container Security Tests

```python
# tests/security/test_container_security.py
import docker
import pytest

@pytest.fixture
def container():
    client = docker.from_env()
    container = client.containers.run(
        "tiny-llm-profiler:test",
        detach=True,
        remove=True
    )
    yield container
    container.stop()

def test_container_runs_as_non_root(container):
    """Test that container runs as non-root user."""
    result = container.exec_run("whoami")
    assert result.output.decode().strip() != "root"

def test_no_privileged_capabilities(container):
    """Test that container has no dangerous capabilities."""
    result = container.exec_run("capsh --print")
    output = result.output.decode()
    
    dangerous_caps = ["sys_admin", "sys_module", "dac_override"]
    for cap in dangerous_caps:
        assert cap not in output.lower()

def test_read_only_filesystem(container):
    """Test that filesystem modifications are restricted."""
    result = container.exec_run("touch /test-file")
    assert result.exit_code != 0  # Should fail due to read-only filesystem
```

### 2. Runtime Security Testing

```bash
# Test with security profiles
docker run --rm \
  --security-opt apparmor=tiny-llm-profiler \
  --security-opt seccomp=seccomp-tiny-llm-profiler.json \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp:noexec,nosuid \
  tiny-llm-profiler:latest \
  tiny-profiler --version
```

## Monitoring and Alerting

### 1. Security Event Monitoring

```yaml
# monitoring/security-alerts.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: security-alerts
data:
  rules.yml: |
    groups:
    - name: container-security
      rules:
      - alert: VulnerabilityDetected
        expr: container_vulnerability_count > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Container vulnerability detected"
          description: "{{ $labels.image }} has {{ $value }} vulnerabilities"
      
      - alert: SuspiciousContainerActivity
        expr: rate(container_syscall_failures[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Suspicious container activity detected"
```

### 2. Log Analysis

```bash
# Extract security-relevant logs
docker logs tiny-llm-profiler 2>&1 | grep -E "(ERROR|WARN|security|auth|fail)"

# Monitor for security events with jq
docker logs tiny-llm-profiler --since 1h 2>&1 | \
  grep -o '{.*}' | \
  jq 'select(.level == "ERROR" or .level == "WARN")'
```

## Compliance and Auditing

### 1. Security Compliance Checklist

- [ ] Container runs as non-root user
- [ ] Read-only root filesystem enabled
- [ ] No dangerous capabilities granted
- [ ] Resource limits configured
- [ ] Health checks implemented
- [ ] Security scanning integrated
- [ ] SBOM generation enabled
- [ ] Base image regularly updated
- [ ] Security profiles applied
- [ ] Runtime monitoring configured

### 2. Audit Logging

```python
# src/tiny_llm_profiler/security/audit.py
import logging
import json
from datetime import datetime

class SecurityAuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("security.audit")
        
    def log_security_event(self, event_type: str, details: dict):
        """Log security-relevant events."""
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
            "severity": self._determine_severity(event_type)
        }
        
        self.logger.info(json.dumps(audit_record))
    
    def _determine_severity(self, event_type: str) -> str:
        severity_map = {
            "device_connection": "INFO",
            "firmware_flash": "INFO",
            "authentication_failure": "WARN",
            "unauthorized_access": "ERROR",
            "privilege_escalation": "CRITICAL"
        }
        return severity_map.get(event_type, "INFO")
```

This comprehensive container security configuration ensures that the tiny-llm-edge-profiler maintains high security standards throughout its development and deployment lifecycle.