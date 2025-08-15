# Self-Healing Pipeline Guard - Deployment Guide

[![Production Ready](https://img.shields.io/badge/Production-Ready-green)](https://github.com/terragon-labs/self-healing-pipeline-guard)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](./docker-compose.production.yml)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Supported-blue)](./deployment/kubernetes/)
[![Security](https://img.shields.io/badge/Security-Hardened-green)](./SECURITY.md)

Comprehensive deployment guide for the Self-Healing Pipeline Guard system across multiple environments and platforms.

## 🚀 Quick Start

### Prerequisites

- **Docker**: Version 20.10+ with Docker Compose v2
- **Kubernetes**: Version 1.24+ (for Kubernetes deployment)
- **Resources**: Minimum 2GB RAM, 2 CPU cores, 10GB disk space
- **Network**: Outbound internet access for dependencies

### One-Command Production Deployment

```bash
# Clone repository
git clone https://github.com/terragon-labs/self-healing-pipeline-guard.git
cd self-healing-pipeline-guard

# Set environment variables
export GUARD_API_KEY="your-secure-api-key-here"
export GUARD_REGION="us-east-1"
export GRAFANA_ADMIN_PASSWORD="secure-password"

# Deploy with Docker Compose
docker-compose -f docker-compose.production.yml up -d
```

## 📋 Table of Contents

- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Provider Specific](#cloud-provider-specific)
- [Configuration](#configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Security](#security)
- [Scaling & Performance](#scaling--performance)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## 🐳 Docker Deployment

### Development Environment

```bash
# Quick development setup
docker-compose up -d

# View logs
docker-compose logs -f guard-unified

# Health check
curl http://localhost:8080/health
```

### Production Environment

#### 1. Environment Configuration

```bash
# Create .env file for production
cat > .env << 'EOF'
# Core Configuration
GUARD_REGION=us-east-1
GUARD_ENVIRONMENT=production
LOG_LEVEL=INFO
BUILD_VERSION=v1.0.0

# Security
GUARD_API_KEY=your-secure-api-key-here
GRAFANA_ADMIN_PASSWORD=secure-password-123
GRAFANA_SECRET_KEY=your-grafana-secret-key

# Performance
GUARD_WORKERS=4
MAX_MEMORY_MB=2048

# Monitoring
WATCHTOWER_EMAIL_FROM=alerts@yourcompany.com
WATCHTOWER_EMAIL_TO=devops@yourcompany.com
WATCHTOWER_EMAIL_SERVER=smtp.yourcompany.com:587
EOF
```

#### 2. Production Deployment

```bash
# Deploy production stack
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
docker-compose -f docker-compose.production.yml ps

# Check all services are healthy
docker-compose -f docker-compose.production.yml exec guard-unified python3 /app/healthcheck.py
```

#### 3. Service Access

| Service | URL | Purpose |
|---------|-----|---------|
| Main Guard | http://localhost:8080 | Unified guard system |
| Pipeline Guard | http://localhost:8081 | Pipeline monitoring |
| Infrastructure | http://localhost:8082 | Infrastructure monitoring |
| Grafana | http://localhost:3000 | Dashboards (admin/password) |
| Prometheus | http://localhost:9090 | Metrics collection |
| Redis | localhost:6379 | Caching & coordination |

### Docker Build Options

```bash
# Build custom image
docker build -t your-registry/self-healing-guard:latest \
  --build-arg BUILD_VERSION=v1.0.0 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse HEAD) \
  -f docker/self-healing-guard/Dockerfile .

# Multi-architecture build
docker buildx build --platform linux/amd64,linux/arm64 \
  -t your-registry/self-healing-guard:latest \
  --push .
```

## ☸️ Kubernetes Deployment

### Prerequisites

```bash
# Install required tools
kubectl version --client
helm version

# Verify cluster access
kubectl cluster-info
```

### 1. Quick Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/self-healing-guard.yaml

# Verify deployment
kubectl get pods -n self-healing-guard
kubectl get services -n self-healing-guard
```

### 2. Custom Configuration

```bash
# Create custom ConfigMap
kubectl create configmap guard-custom-config \
  --from-file=config/ \
  --namespace=self-healing-guard

# Update deployment to use custom config
kubectl patch deployment guard-unified \
  --namespace=self-healing-guard \
  --patch='{"spec":{"template":{"spec":{"volumes":[{"name":"config-volume","configMap":{"name":"guard-custom-config"}}]}}}}'
```

### 3. Scaling

```bash
# Manual scaling
kubectl scale deployment guard-unified --replicas=5 -n self-healing-guard

# Auto-scaling (HPA already configured)
kubectl get hpa -n self-healing-guard

# Vertical scaling (update resource limits)
kubectl patch deployment guard-unified -n self-healing-guard -p='
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "guard-unified",
          "resources": {
            "limits": {"cpu": "4", "memory": "4Gi"},
            "requests": {"cpu": "1", "memory": "1Gi"}
          }
        }]
      }
    }
  }
}'
```

### 4. Ingress Configuration

```bash
# Install NGINX Ingress Controller (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager (for TLS certificates)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Update ingress with your domain
kubectl patch ingress guard-ingress -n self-healing-guard --type='json' -p='[
  {"op": "replace", "path": "/spec/rules/0/host", "value": "guard.yourdomain.com"},
  {"op": "replace", "path": "/spec/rules/1/host", "value": "api.guard.yourdomain.com"}
]'
```

## ☁️ Cloud Provider Specific

### Amazon Web Services (AWS)

#### EKS Deployment

```bash
# Create EKS cluster
eksctl create cluster \
  --name self-healing-guard \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Deploy with AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

# Deploy application
kubectl apply -f deployment/kubernetes/self-healing-guard.yaml
```

#### ECS Deployment

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name self-healing-guard

# Register task definition
aws ecs register-task-definition --cli-input-json file://deployment/aws/ecs-task-definition.json

# Create service
aws ecs create-service \
  --cluster self-healing-guard \
  --service-name guard-unified \
  --task-definition guard-unified:1 \
  --desired-count 3
```

### Google Cloud Platform (GCP)

#### GKE Deployment

```bash
# Create GKE cluster
gcloud container clusters create self-healing-guard \
  --zone=us-central1-a \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials self-healing-guard --zone=us-central1-a

# Deploy
kubectl apply -f deployment/kubernetes/self-healing-guard.yaml
```

### Microsoft Azure

#### AKS Deployment

```bash
# Create resource group
az group create --name self-healing-guard-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group self-healing-guard-rg \
  --name self-healing-guard \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group self-healing-guard-rg --name self-healing-guard

# Deploy
kubectl apply -f deployment/kubernetes/self-healing-guard.yaml
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GUARD_MODE` | unified | Operation mode: unified, pipeline, infrastructure, security |
| `GUARD_REGION` | us-east-1 | AWS region or geographic region |
| `GUARD_ENVIRONMENT` | production | Environment: development, staging, production |
| `GUARD_LOG_LEVEL` | INFO | Logging level: DEBUG, INFO, WARNING, ERROR |
| `GUARD_API_KEY` | - | API key for authentication (required) |
| `GUARD_PORT` | 8080 | Main service port |
| `GUARD_WORKERS` | 4 | Number of worker processes |
| `GUARD_MAX_MEMORY_MB` | 2048 | Maximum memory usage |
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection URL |

### Configuration Files

#### `config/guard.yaml`

```yaml
guard:
  mode: unified
  region: us-east-1
  environment: production
  
  monitoring:
    health_check_interval: 30
    metrics_retention_hours: 168
    auto_scaling_enabled: true
    
  security:
    api_key_required: true
    rate_limiting_enabled: true
    max_requests_per_minute: 1000
    
  performance:
    max_workers: 4
    max_memory_mb: 2048
    cache_size: 1000
```

#### `config/security.yaml`

```yaml
security:
  encryption:
    algorithm: AES-256-GCM
    key_rotation_days: 90
    
  authentication:
    jwt_secret: your-jwt-secret
    token_expiry_hours: 24
    
  compliance:
    frameworks: [GDPR, CCPA, SOX, HIPAA]
    data_retention_days: 2555
    audit_logging: true
```

## 📊 Monitoring & Observability

### Metrics Collection

The system automatically exposes metrics at `/metrics` endpoint:

```bash
# View metrics
curl http://localhost:8080/metrics

# Prometheus configuration
scrape_configs:
  - job_name: 'self-healing-guard'
    static_configs:
      - targets: ['guard-unified:8080', 'guard-pipeline:8081']
```

### Grafana Dashboards

Pre-configured dashboards are available:

1. **System Overview**: Overall health and performance
2. **Pipeline Monitoring**: CI/CD pipeline health
3. **Infrastructure Metrics**: Resource utilization
4. **Security Events**: Threat detection and response
5. **Compliance Dashboard**: Regulatory compliance status

```bash
# Import dashboards
curl -X POST \
  http://admin:${GRAFANA_ADMIN_PASSWORD}@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @deployment/monitoring/grafana/dashboards/overview.json
```

### Log Aggregation

Configure log shipping to your preferred system:

```yaml
# Fluentd configuration
<source>
  @type tail
  path /app/logs/*.log
  pos_file /var/log/fluentd/guard.log.pos
  tag guard.*
  format json
</source>

<match guard.**>
  @type elasticsearch
  host elasticsearch.example.com
  port 9200
  index_name guard-logs
</match>
```

## 🔒 Security

### TLS/SSL Configuration

```bash
# Generate self-signed certificates for development
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/nginx/ssl/guard.key \
  -out deployment/nginx/ssl/guard.crt

# For production, use Let's Encrypt or your CA
certbot certonly --standalone -d guard.yourdomain.com
```

### API Authentication

```bash
# Generate secure API key
GUARD_API_KEY=$(openssl rand -hex 32)
echo "API Key: $GUARD_API_KEY"

# Use in requests
curl -H "Authorization: Bearer $GUARD_API_KEY" \
  http://localhost:8080/api/status
```

### Network Security

```yaml
# Docker network isolation
networks:
  guard-internal:
    driver: bridge
    internal: true
  guard-external:
    driver: bridge

# Kubernetes NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: guard-network-policy
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: self-healing-guard
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: nginx-ingress
    ports:
    - protocol: TCP
      port: 8080
```

## 📈 Scaling & Performance

### Horizontal Scaling

```bash
# Docker Swarm
docker service scale guard-unified=5

# Kubernetes
kubectl scale deployment guard-unified --replicas=5 -n self-healing-guard

# Auto-scaling based on metrics
kubectl autoscale deployment guard-unified \
  --cpu-percent=70 \
  --min=3 \
  --max=10 \
  -n self-healing-guard
```

### Vertical Scaling

```bash
# Update resource limits
docker-compose -f docker-compose.production.yml up -d \
  --scale guard-unified=3

# Kubernetes resource updates
kubectl patch deployment guard-unified -n self-healing-guard -p='
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "guard-unified",
          "resources": {
            "limits": {"cpu": "4", "memory": "4Gi"}
          }
        }]
      }
    }
  }
}'
```

### Performance Tuning

```yaml
# Optimize for high throughput
environment:
  - GUARD_WORKERS=8
  - GUARD_MAX_MEMORY_MB=4096
  - GUARD_CACHE_SIZE=2000
  - REDIS_MAXMEMORY=512mb
  - REDIS_MAXMEMORY_POLICY=allkeys-lru

# Optimize for low latency
environment:
  - GUARD_WORKERS=2
  - GUARD_MAX_MEMORY_MB=1024
  - GUARD_RESPONSE_TIMEOUT=5
  - REDIS_MAXMEMORY=256mb
```

## 🛠️ Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check logs
docker-compose logs guard-unified

# Check health
docker-compose exec guard-unified python3 /app/healthcheck.py

# Check configuration
docker-compose exec guard-unified cat /app/config/guard.yaml
```

#### High Memory Usage

```bash
# Monitor memory
docker stats guard-unified

# Adjust limits
export GUARD_MAX_MEMORY_MB=1024
docker-compose up -d guard-unified
```

#### Database Connection Issues

```bash
# Test Redis connectivity
docker-compose exec guard-unified redis-cli -h redis ping

# Check Redis logs
docker-compose logs redis
```

### Debug Mode

```bash
# Enable debug logging
export GUARD_LOG_LEVEL=DEBUG
docker-compose up -d guard-unified

# Access debug endpoints
curl http://localhost:8080/debug/status
curl http://localhost:8080/debug/metrics
```

### Performance Issues

```bash
# Profile application
docker-compose exec guard-unified python3 -m cProfile -o /app/logs/profile.stats /app/entrypoint.sh

# Monitor system resources
docker-compose exec guard-unified top
docker-compose exec guard-unified iostat -x 1
```

## 🔧 Maintenance

### Backup & Recovery

```bash
# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb ./backup/

# Backup configuration
tar -czf backup/config-$(date +%Y%m%d).tar.gz config/

# Restore Redis data
docker cp ./backup/dump.rdb $(docker-compose ps -q redis):/data/
docker-compose restart redis
```

### Updates

```bash
# Update to latest version
docker-compose pull
docker-compose up -d

# Rolling update (zero downtime)
docker-compose up -d --no-deps guard-unified
```

### Log Rotation

```bash
# Configure log rotation
cat > /etc/logrotate.d/self-healing-guard << 'EOF'
/var/lib/docker/volumes/guard-logs/_data/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 guardapp guardapp
    postrotate
        docker-compose exec guard-unified kill -USR1 1
    endscript
}
EOF
```

### Health Monitoring

```bash
# Automated health checks
#!/bin/bash
# health-monitor.sh
while true; do
    if ! curl -sf http://localhost:8080/health > /dev/null; then
        echo "$(date): Health check failed, restarting service..."
        docker-compose restart guard-unified
    fi
    sleep 60
done
```

## 📞 Support

### Getting Help

- 📧 **Email**: support@terragon.dev
- 💬 **Discord**: [Join our community](https://discord.gg/terragon)
- 📖 **Documentation**: [Full documentation](https://docs.terragon.dev/self-healing-guard)
- 🐛 **Issues**: [GitHub Issues](https://github.com/terragon-labs/self-healing-pipeline-guard/issues)

### Professional Support

- **Enterprise Support**: Available for production deployments
- **Training**: Custom training sessions for your team
- **Consulting**: Architecture and optimization consulting
- **SLA**: 99.9% uptime guarantee with enterprise support

---

**© 2024 Terragon Labs. All rights reserved.**