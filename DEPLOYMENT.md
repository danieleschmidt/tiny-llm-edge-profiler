# Deployment Guide

Complete deployment guide for the Tiny LLM Edge Profiler in production environments.

## Quick Start

Deploy with Docker Compose (recommended for getting started):

```bash
cd deployment/docker
docker-compose up -d
```

Access the application at `http://localhost:8080`

## Deployment Options

### 1. Docker Deployment

**Prerequisites:**
- Docker 20.10+
- Docker Compose 2.0+

**Deploy:**
```bash
./deployment/scripts/deploy.sh --type docker --env production
```

**Configuration:**
- Edit `deployment/docker/docker-compose.yml` for custom settings
- Modify `deployment/config/production.yaml` for application config
- Use environment variables for secrets

**Monitoring:**
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin_password)
- Application metrics: `http://localhost:8081/metrics`

### 2. Kubernetes Deployment

**Prerequisites:**
- Kubernetes 1.20+
- kubectl configured
- Helm 3.0+ (optional)

**Deploy:**
```bash
./deployment/scripts/deploy.sh --type kubernetes --namespace profiler
```

**Features:**
- Auto-scaling based on CPU/memory/custom metrics
- Rolling updates with zero downtime
- Pod disruption budgets for availability
- Resource limits and requests
- Security contexts and RBAC

**Access:**
```bash
kubectl port-forward service/tiny-llm-profiler-service 8080:8080 -n profiler
```

### 3. Bare Metal Deployment

**Prerequisites:**
- Python 3.8+
- systemd (Linux)
- pip3

**Deploy:**
```bash
./deployment/scripts/deploy.sh --type bare-metal --env production
```

**Features:**
- Systemd service management
- Automatic restarts
- Log rotation
- User isolation

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TINY_LLM_LOG_LEVEL` | Logging level | `INFO` |
| `TINY_LLM_LOG_DIR` | Log directory | `/logs` |
| `TINY_LLM_OUTPUT_DIR` | Output directory | `/data` |
| `TINY_LLM_MODEL_CACHE` | Model cache directory | `/cache` |
| `TINY_LLM_SECURITY_VALIDATION` | Enable security validation | `true` |
| `TINY_LLM_MAX_FILE_SIZE` | Max file size (MB) | `100` |

### Platform-Specific Settings

The profiler automatically optimizes settings for different platforms:

- **ESP32**: Balanced performance and stability
- **STM32F4**: Conservative settings for limited resources  
- **STM32F7**: Moderate settings with FPU support
- **RP2040**: Optimized for dual-core usage
- **Jetson Nano**: High-performance settings with GPU support

### Security Configuration

Production deployments include:

- Input validation and sanitization
- Path traversal protection
- File extension restrictions
- Secure temporary directories
- Non-root container execution
- RBAC (Kubernetes)

## Monitoring and Observability

### Metrics

The profiler exposes metrics at `/metrics`:

- **Profiling metrics**: Task completion rates, latency distributions
- **System metrics**: CPU, memory, disk usage
- **Application metrics**: Cache hit rates, error rates
- **Device metrics**: Connection status, communication errors

### Dashboards

Pre-configured Grafana dashboards:

- **Overview**: System health and key metrics
- **Profiling Performance**: Task execution and device metrics
- **Resource Utilization**: System resource usage
- **Error Tracking**: Error rates and failure analysis

### Alerting

Prometheus alerting rules for:

- High error rates
- System resource exhaustion
- Device connectivity issues
- Performance degradation

## Scaling

### Horizontal Scaling

**Docker Compose:**
```bash
docker-compose up -d --scale tiny-llm-profiler=3
```

**Kubernetes:**
- Automatic scaling based on CPU/memory/custom metrics
- Manual scaling: `kubectl scale deployment tiny-llm-profiler --replicas=5`

### Vertical Scaling

Adjust resource limits in deployment configuration:

**Docker:**
```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '4'
```

**Kubernetes:**
```yaml
resources:
  limits:
    memory: "4Gi"
    cpu: "4000m"
```

## Security

### Network Security

- All communications over HTTPS in production
- Network policies (Kubernetes)
- Firewall rules (bare metal)

### Container Security

- Non-root user execution
- Read-only root filesystem
- Minimal attack surface
- Security scanning in CI/CD

### Data Security

- Encrypted storage at rest
- Secure temporary file handling
- Input validation and sanitization
- Audit logging

## Backup and Recovery

### Data Backup

**Docker volumes:**
```bash
docker run --rm -v profiler_data:/data -v $(pwd):/backup busybox tar czf /backup/data-backup.tar.gz -C /data .
```

**Kubernetes:**
```bash
kubectl exec -n profiler deployment/tiny-llm-profiler -- tar czf - /data | gzip > data-backup.tar.gz
```

### Configuration Backup

```bash
# Docker
docker-compose config > docker-compose-backup.yml

# Kubernetes  
kubectl get all -n profiler -o yaml > k8s-backup.yaml
```

### Recovery

1. Stop services
2. Restore data from backup
3. Restart services
4. Verify functionality

## Troubleshooting

### Common Issues

**Container won't start:**
- Check logs: `docker-compose logs tiny-llm-profiler`
- Verify configuration files
- Check resource limits

**Device connection failures:**
- Verify device permissions
- Check USB device mounting
- Validate device paths

**High memory usage:**
- Adjust cache settings
- Check for memory leaks
- Monitor garbage collection

**Performance issues:**
- Review resource limits
- Check system load
- Analyze profiling metrics

### Health Checks

**Docker:**
```bash
docker-compose exec tiny-llm-profiler python -c "from tiny_llm_profiler.health import run_health_checks; print('OK' if run_health_checks() else 'FAIL')"
```

**Kubernetes:**
```bash
kubectl exec -n profiler deployment/tiny-llm-profiler -- python -c "from tiny_llm_profiler.health import run_health_checks; print('OK' if run_health_checks() else 'FAIL')"
```

### Log Analysis

**View real-time logs:**
```bash
# Docker
docker-compose logs -f tiny-llm-profiler

# Kubernetes
kubectl logs -f deployment/tiny-llm-profiler -n profiler

# Bare metal
journalctl -u tiny-llm-profiler -f
```

**Search logs:**
```bash
# Error messages
grep -i error /logs/tiny_llm_profiler.log

# Security events
grep -i security /logs/tiny_llm_profiler.log
```

## Maintenance

### Updates

**Docker:**
```bash
docker-compose pull
docker-compose up -d
```

**Kubernetes:**
```bash
kubectl set image deployment/tiny-llm-profiler profiler=terragon-labs/tiny-llm-profiler:0.2.0 -n profiler
```

### Log Rotation

Logs are automatically rotated:
- Docker: 10MB max, 3 files
- Kubernetes: Container logs handled by cluster
- Bare metal: systemd journal rotation

### Cache Maintenance

Clear caches if needed:
```bash
# Docker
docker-compose exec tiny-llm-profiler rm -rf /cache/*

# Kubernetes
kubectl exec -n profiler deployment/tiny-llm-profiler -- rm -rf /cache/*
```

## Performance Tuning

### Resource Optimization

1. **CPU**: Match core count to concurrent profiling tasks
2. **Memory**: Size based on model cache and concurrent operations
3. **Storage**: Use SSD for better I/O performance
4. **Network**: Ensure low latency for device communication

### Configuration Tuning

- Adjust `sample_rate_hz` based on system capability
- Tune `measurement_iterations` for accuracy vs speed
- Configure cache sizes based on available memory
- Set appropriate timeouts for device communication

## Production Checklist

Before going to production:

- [ ] Security configuration reviewed
- [ ] Resource limits set appropriately
- [ ] Monitoring and alerting configured
- [ ] Backup procedures tested
- [ ] Update procedures documented
- [ ] Health checks configured
- [ ] Log aggregation setup
- [ ] Performance baseline established
- [ ] Security scanning completed
- [ ] Documentation updated

## Support

For deployment issues:

1. Check logs for error messages
2. Verify configuration settings
3. Test in development environment
4. Review this documentation
5. Check GitHub issues
6. Contact support team

## License

Apache License 2.0 - see LICENSE file for details.