# Operational Procedures

Comprehensive operational procedures for maintaining and operating the tiny-llm-edge-profiler system.

## Daily Operations

### Morning Health Check Routine
```bash
#!/bin/bash
# scripts/daily_health_check.sh

echo "=== Daily Health Check - $(date) ==="

# 1. Application Health
echo "Checking application health..."
curl -f http://localhost:8000/health/detailed | jq '.'
if [ $? -ne 0 ]; then
    echo "❌ Application health check failed"
    exit 1
fi

# 2. Device Connectivity
echo "Checking device connectivity..."
curl -s http://localhost:8000/devices/status | jq '.devices[] | select(.connected == false)'
disconnected_devices=$(curl -s http://localhost:8000/devices/status | jq '.devices[] | select(.connected == false)' | wc -l)
if [ $disconnected_devices -gt 0 ]; then
    echo "⚠️  $disconnected_devices devices disconnected"
fi

# 3. Resource Utilization
echo "Checking resource utilization..."
df -h
free -h
docker stats --no-stream

# 4. Recent Errors
echo "Checking recent errors..."
grep -i error /var/log/tiny-llm-profiler/*.log | tail -10

# 5. Performance Metrics
echo "Performance summary..."
curl -s http://localhost:9090/api/v1/query?query=rate(profiling_sessions_total[1h]) | jq '.data.result[0].value[1]'

echo "✅ Daily health check complete"
```

### System Startup Procedures
```bash
#!/bin/bash
# scripts/system_startup.sh

echo "Starting tiny-llm-edge-profiler system..."

# 1. Start infrastructure services
echo "Starting monitoring stack..."
docker-compose -f docker-compose.monitoring.yml up -d

# 2. Wait for services to be ready
echo "Waiting for services to initialize..."
sleep 30

# 3. Verify monitoring services
curl -f http://localhost:9090/-/healthy || exit 1
curl -f http://localhost:3000/api/health || exit 1

# 4. Start main application
echo "Starting profiler application..."
docker-compose up -d profiler

# 5. Verify application startup
sleep 15
curl -f http://localhost:8000/health || exit 1

# 6. Initialize device connections
echo "Initializing device connections..."
curl -X POST http://localhost:8000/devices/scan

echo "✅ System startup complete"
```

## Backup and Recovery

### Database Backup Procedures
```bash
#!/bin/bash
# scripts/backup_database.sh

BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
mkdir -p $BACKUP_DIR

echo "Creating database backup..."

# Backup profiling results database
pg_dump -U profiler_user -h localhost profiler_db > $BACKUP_DIR/profiler_db.sql

# Backup configuration files
tar -czf $BACKUP_DIR/config_backup.tar.gz \
    /etc/tiny-llm-profiler/ \
    /opt/tiny-llm-profiler/config/ \
    monitoring/

# Backup model files
rsync -av /opt/models/ $BACKUP_DIR/models/

# Create backup manifest
cat > $BACKUP_DIR/backup_manifest.json << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "database_size": "$(stat -c%s $BACKUP_DIR/profiler_db.sql)",
    "config_size": "$(stat -c%s $BACKUP_DIR/config_backup.tar.gz)",
    "models_count": "$(find $BACKUP_DIR/models -type f | wc -l)"
}
EOF

echo "✅ Backup complete: $BACKUP_DIR"
```

### System Recovery Procedures
```bash
#!/bin/bash
# scripts/system_recovery.sh

BACKUP_DIR=$1
if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

echo "Starting system recovery from $BACKUP_DIR..."

# 1. Stop all services
docker-compose down
docker-compose -f docker-compose.monitoring.yml down

# 2. Restore database
echo "Restoring database..."
psql -U profiler_user -h localhost -c "DROP DATABASE IF EXISTS profiler_db;"
psql -U profiler_user -h localhost -c "CREATE DATABASE profiler_db;"
psql -U profiler_user -h localhost profiler_db < $BACKUP_DIR/profiler_db.sql

# 3. Restore configuration
echo "Restoring configuration..."
tar -xzf $BACKUP_DIR/config_backup.tar.gz -C /

# 4. Restore models
echo "Restoring models..."
rsync -av $BACKUP_DIR/models/ /opt/models/

# 5. Restart services
echo "Restarting services..."
docker-compose -f docker-compose.monitoring.yml up -d
sleep 30
docker-compose up -d

# 6. Verify recovery
curl -f http://localhost:8000/health/detailed

echo "✅ System recovery complete"
```

## Deployment Procedures

### Blue-Green Deployment
```bash
#!/bin/bash
# scripts/blue_green_deploy.sh

NEW_VERSION=$1
CURRENT_ENV=$(curl -s http://localhost:8000/health | jq -r '.environment // "blue"')
TARGET_ENV=$([ "$CURRENT_ENV" = "blue" ] && echo "green" || echo "blue")

echo "Deploying version $NEW_VERSION to $TARGET_ENV environment..."

# 1. Build new version
docker build -t tiny-llm-profiler:$NEW_VERSION .

# 2. Start target environment
docker-compose -f docker-compose.$TARGET_ENV.yml up -d

# 3. Health check target environment
echo "Waiting for $TARGET_ENV environment to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:800${TARGET_ENV:0:1}/health; then
        break
    fi
    sleep 10
done

# 4. Run smoke tests
echo "Running smoke tests..."
pytest tests/smoke/ -v --target-env=$TARGET_ENV
if [ $? -ne 0 ]; then
    echo "❌ Smoke tests failed, rolling back..."
    docker-compose -f docker-compose.$TARGET_ENV.yml down
    exit 1
fi

# 5. Switch traffic
echo "Switching traffic to $TARGET_ENV..."
# Update load balancer configuration
kubectl patch service profiler-service -p '{"spec":{"selector":{"environment":"'$TARGET_ENV'"}}}'

# 6. Stop old environment
sleep 30
docker-compose -f docker-compose.$CURRENT_ENV.yml down

echo "✅ Deployment complete - now serving from $TARGET_ENV"
```

### Rollback Procedures
```bash
#!/bin/bash
# scripts/rollback.sh

echo "Initiating rollback procedure..."

# 1. Identify current deployment
CURRENT_VERSION=$(curl -s http://localhost:8000/health | jq -r '.version')
echo "Current version: $CURRENT_VERSION"

# 2. Get previous stable version
PREVIOUS_VERSION=$(git describe --tags --abbrev=0 HEAD~1)
echo "Rolling back to: $PREVIOUS_VERSION"

# 3. Quick rollback - switch to previous image
docker tag tiny-llm-profiler:$PREVIOUS_VERSION tiny-llm-profiler:latest
docker-compose restart profiler

# 4. Verify rollback
sleep 15
curl -f http://localhost:8000/health
if [ $? -eq 0 ]; then
    echo "✅ Rollback successful"
else
    echo "❌ Rollback failed, manual intervention required"
    exit 1
fi

# 5. Notify team
curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"🔄 Rollback complete: '$CURRENT_VERSION' → '$PREVIOUS_VERSION'"}' \
    $SLACK_WEBHOOK_URL

echo "✅ Rollback procedure complete"
```

## Maintenance Procedures

### Log Rotation and Cleanup
```bash
#!/bin/bash
# scripts/log_cleanup.sh

echo "Starting log cleanup..."

# 1. Rotate application logs
find /var/log/tiny-llm-profiler/ -name "*.log" -size +100M -exec gzip {} \;
find /var/log/tiny-llm-profiler/ -name "*.log.gz" -mtime +30 -delete

# 2. Clean Docker logs
docker system prune -f --filter "until=72h"

# 3. Clean profiling cache
find /tmp/profiling_cache -mtime +7 -delete

# 4. Archive old profiling results
psql -U profiler_user -c "
    INSERT INTO profiling_results_archive 
    SELECT * FROM profiling_results 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    DELETE FROM profiling_results 
    WHERE created_at < NOW() - INTERVAL '90 days';
"

echo "✅ Log cleanup complete"
```

### Security Updates
```bash
#!/bin/bash
# scripts/security_update.sh

echo "Performing security updates..."

# 1. Update system packages
apt update && apt upgrade -y

# 2. Update Docker images
docker-compose pull
docker system prune -f

# 3. Scan for vulnerabilities
trivy image tiny-llm-profiler:latest

# 4. Update Python dependencies
pip-audit
pip install --upgrade -r requirements.txt

# 5. Restart services with new images
docker-compose down
docker-compose up -d

echo "✅ Security updates complete"
```

## Monitoring and Alerting Maintenance

### Monitoring Stack Health Check
```bash
#!/bin/bash
# scripts/monitoring_health.sh

echo "Checking monitoring stack health..."

# Prometheus
if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ Prometheus healthy"
else
    echo "❌ Prometheus unhealthy"
    docker-compose -f docker-compose.monitoring.yml restart prometheus
fi

# Grafana
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana healthy"
else
    echo "❌ Grafana unhealthy"
    docker-compose -f docker-compose.monitoring.yml restart grafana
fi

# AlertManager
if curl -f http://localhost:9093/-/healthy > /dev/null 2>&1; then
    echo "✅ AlertManager healthy"
else
    echo "❌ AlertManager unhealthy"
    docker-compose -f docker-compose.monitoring.yml restart alertmanager
fi

echo "Monitoring stack health check complete"
```

### Performance Optimization
```bash
#!/bin/bash
# scripts/performance_optimization.sh

echo "Running performance optimization..."

# 1. Database optimization
psql -U profiler_user -c "VACUUM ANALYZE;"
psql -U profiler_user -c "REINDEX DATABASE profiler_db;"

# 2. Clear caches
redis-cli FLUSHALL

# 3. Optimize device connections
curl -X POST http://localhost:8000/devices/optimize-connections

# 4. Review resource allocation
docker stats --no-stream | awk 'NR>1 {print $1, $3, $4}'

echo "✅ Performance optimization complete"
```

## Incident Response Procedures

### Critical Incident Response
1. **Immediate Response (0-5 minutes)**
   - Acknowledge alert
   - Check system status dashboard
   - Execute immediate mitigation steps

2. **Investigation (5-15 minutes)**
   - Gather logs and metrics
   - Identify root cause
   - Implement temporary fixes

3. **Resolution (15-60 minutes)**
   - Deploy permanent fix
   - Verify system stability
   - Update incident tracking

4. **Post-Incident (1-24 hours)**
   - Conduct post-mortem
   - Update runbooks
   - Implement preventive measures

These operational procedures ensure reliable, maintainable operation of the tiny-llm-edge-profiler system across all environments.