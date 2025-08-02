# Alerting Configuration

Comprehensive alerting setup for the tiny-llm-edge-profiler project, covering application health, device monitoring, and performance degradation.

## Alert Categories

### Critical Alerts
High-severity issues requiring immediate attention:

```yaml
# monitoring/alerts/critical.yml
groups:
  - name: critical_alerts
    rules:
      - alert: ApplicationDown
        expr: up{job="tiny-llm-profiler"} == 0
        for: 30s
        labels:
          severity: critical
          component: application
        annotations:
          summary: "Profiler application is down"
          description: "The tiny-llm-edge-profiler application has been down for more than 30 seconds"
          runbook_url: "https://docs.company.com/runbooks/application-down"

      - alert: DeviceDisconnected
        expr: device_connection_status == 0
        for: 1m
        labels:
          severity: critical
          component: device
        annotations:
          summary: "Device {{ $labels.device_id }} disconnected"
          description: "Device {{ $labels.device_id }} ({{ $labels.platform }}) has been disconnected for more than 1 minute"
          runbook_url: "https://docs.company.com/runbooks/device-connectivity"

      - alert: DeviceOverheating
        expr: device_temperature_celsius > 85
        for: 2m
        labels:
          severity: critical
          component: device
        annotations:
          summary: "Device {{ $labels.device_id }} overheating"
          description: "Device {{ $labels.device_id }} temperature is {{ $value }}°C, exceeding safe limits"
          runbook_url: "https://docs.company.com/runbooks/device-overheating"

      - alert: ProfilingSessionsFailingCritical
        expr: increase(profiling_errors_total[10m]) > 10
        for: 0s
        labels:
          severity: critical
          component: profiling
        annotations:
          summary: "High profiling session failure rate"
          description: "More than 10 profiling sessions have failed in the last 10 minutes"
          runbook_url: "https://docs.company.com/runbooks/profiling-failures"

      - alert: MemoryExhaustionCritical
        expr: device_memory_usage_bytes / device_memory_total_bytes > 0.95
        for: 1m
        labels:
          severity: critical
          component: device
        annotations:
          summary: "Device {{ $labels.device_id }} memory exhaustion"
          description: "Device {{ $labels.device_id }} memory usage is {{ $value | humanizePercentage }}"
          runbook_url: "https://docs.company.com/runbooks/memory-exhaustion"
```

### Warning Alerts
Medium-severity issues requiring attention:

```yaml
# monitoring/alerts/warnings.yml
groups:
  - name: warning_alerts
    rules:
      - alert: HighResourceUtilization
        expr: |
          (
            rate(container_cpu_usage_seconds_total{container="profiler"}[5m]) > 0.8
          ) or (
            container_memory_usage_bytes{container="profiler"} / container_spec_memory_limit_bytes > 0.8
          )
        for: 5m
        labels:
          severity: warning
          component: application
        annotations:
          summary: "High resource utilization in profiler application"
          description: "CPU or memory usage is above 80% for more than 5 minutes"

      - alert: SlowProfilingPerformance
        expr: histogram_quantile(0.95, rate(profiling_duration_seconds_bucket[10m])) > 300
        for: 3m
        labels:
          severity: warning
          component: profiling
        annotations:
          summary: "Slow profiling performance detected"
          description: "95th percentile profiling duration is {{ $value }}s, exceeding 5 minutes"

      - alert: DeviceTemperatureWarning
        expr: device_temperature_celsius > 70 and device_temperature_celsius <= 85
        for: 5m
        labels:
          severity: warning
          component: device
        annotations:
          summary: "Device {{ $labels.device_id }} temperature elevated"
          description: "Device {{ $labels.device_id }} temperature is {{ $value }}°C"

      - alert: LowModelThroughput
        expr: model_tokens_per_second < 5
        for: 3m
        labels:
          severity: warning
          component: model
        annotations:
          summary: "Low model throughput on {{ $labels.platform }}"
          description: "Model {{ $labels.model_name }} throughput is {{ $value }} tokens/second"

      - alert: HighDiskUsage
        expr: (node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes > 0.8
        for: 5m
        labels:
          severity: warning
          component: infrastructure
        annotations:
          summary: "High disk usage detected"
          description: "Disk usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}"
```

### Performance Alerts
Performance degradation monitoring:

```yaml
# monitoring/alerts/performance.yml
groups:
  - name: performance_alerts
    rules:
      - alert: HighLatencyP95
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2.0
        for: 3m
        labels:
          severity: warning
          component: api
        annotations:
          summary: "High API latency detected"
          description: "95th percentile API latency is {{ $value }}s"

      - alert: ModelAccuracyDegradation
        expr: model_accuracy_score < 0.85
        for: 5m
        labels:
          severity: warning
          component: model
        annotations:
          summary: "Model accuracy degraded"
          description: "Model {{ $labels.model_name }} accuracy dropped to {{ $value }}"

      - alert: QuantizationErrorHigh
        expr: quantization_error_rate > 0.1
        for: 2m
        labels:
          severity: warning
          component: quantization
        annotations:
          summary: "High quantization error rate"
          description: "Quantization error rate is {{ $value }} for {{ $labels.model_name }}"

      - alert: DevicePowerConsumptionHigh
        expr: device_power_consumption_mw > 5000
        for: 10m
        labels:
          severity: info
          component: device
        annotations:
          summary: "High power consumption on device {{ $labels.device_id }}"
          description: "Power consumption is {{ $value }}mW, consider optimizing"
```

## AlertManager Configuration

### Main Configuration
```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@tiny-llm-profiler.com'
  smtp_auth_username: 'alerts@tiny-llm-profiler.com'
  smtp_auth_password: 'password'

route:
  group_by: ['alertname', 'component']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default-receiver'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 10s
      repeat_interval: 1h
    - match:
        severity: warning
      receiver: 'warning-alerts'
      repeat_interval: 2h
    - match:
        component: device
      receiver: 'device-alerts'

receivers:
  - name: 'default-receiver'
    email_configs:
      - to: 'team@tiny-llm-profiler.com'
        subject: '[ALERT] {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Severity: {{ .Labels.severity }}
          Component: {{ .Labels.component }}
          
          {{ if .Annotations.runbook_url }}
          Runbook: {{ .Annotations.runbook_url }}
          {{ end }}
          {{ end }}

  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@tiny-llm-profiler.com'
        subject: '[CRITICAL] {{ .GroupLabels.alertname }}'
        body: |
          🚨 CRITICAL ALERT 🚨
          
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Time: {{ .StartsAt }}
          
          Runbook: {{ .Annotations.runbook_url }}
          {{ end }}
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts-critical'
        title: '[CRITICAL] {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ .Annotations.description }}
          {{ end }}

  - name: 'warning-alerts'
    email_configs:
      - to: 'team@tiny-llm-profiler.com'
        subject: '[WARNING] {{ .GroupLabels.alertname }}'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts-warning'
        title: '[WARNING] {{ .GroupLabels.alertname }}'

  - name: 'device-alerts'
    email_configs:
      - to: 'hardware-team@tiny-llm-profiler.com'
        subject: '[DEVICE] {{ .GroupLabels.alertname }}'
    webhook_configs:
      - url: 'http://localhost:8080/webhook/device-alerts'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
```

## Alert Testing

### Alert Testing Framework
```python
# tests/monitoring/test_alerts.py
import pytest
import requests
import time
from prometheus_api_client import PrometheusConnect

class TestAlerts:
    def __init__(self):
        self.prometheus = PrometheusConnect(url="http://localhost:9090")
    
    def test_device_disconnection_alert(self):
        """Test that device disconnection triggers alert"""
        # Simulate device disconnection
        self.simulate_device_disconnect("test_device_1")
        
        # Wait for alert evaluation
        time.sleep(65)  # Wait longer than alert 'for' duration
        
        # Check if alert is firing
        alerts = self.prometheus.get_current_metric_value(
            metric_name="ALERTS",
            label_config={"alertname": "DeviceDisconnected"}
        )
        
        assert len(alerts) > 0, "DeviceDisconnected alert should be firing"
        assert alerts[0]["metric"]["device_id"] == "test_device_1"
    
    def test_high_temperature_alert(self):
        """Test temperature alert threshold"""
        # Simulate high temperature
        self.set_device_temperature("test_device_1", 90)
        
        time.sleep(125)  # Wait for alert
        
        alerts = self.prometheus.get_current_metric_value(
            metric_name="ALERTS",
            label_config={"alertname": "DeviceOverheating"}
        )
        
        assert len(alerts) > 0, "DeviceOverheating alert should be firing"
    
    def test_profiling_failure_alert(self):
        """Test profiling failure rate alert"""
        # Simulate multiple profiling failures
        for _ in range(15):
            self.simulate_profiling_failure()
            time.sleep(1)
        
        time.sleep(30)
        
        alerts = self.prometheus.get_current_metric_value(
            metric_name="ALERTS",
            label_config={"alertname": "ProfilingSessionsFailingCritical"}
        )
        
        assert len(alerts) > 0, "ProfilingSessionsFailingCritical alert should be firing"
    
    def simulate_device_disconnect(self, device_id: str):
        """Simulate device disconnection for testing"""
        # This would interact with your device management system
        pass
    
    def set_device_temperature(self, device_id: str, temperature: float):
        """Set device temperature for testing"""
        # This would interact with your device monitoring system
        pass
    
    def simulate_profiling_failure(self):
        """Simulate profiling session failure"""
        # This would trigger a profiling error metric increment
        pass
```

### Alert Validation Script
```bash
#!/bin/bash
# scripts/validate_alerts.sh

echo "Validating Alert Configuration..."

# Check Prometheus rules syntax
promtool check rules monitoring/alerts/*.yml
if [ $? -ne 0 ]; then
    echo "❌ Alert rules validation failed"
    exit 1
fi

# Check AlertManager configuration
amtool check-config monitoring/alertmanager.yml
if [ $? -ne 0 ]; then
    echo "❌ AlertManager configuration validation failed"
    exit 1
fi

echo "✅ All alert configurations are valid"

# Test alert endpoints
echo "Testing alert endpoints..."

# Test Prometheus alerts API
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.type=="alerting") | .name' | wc -l
echo "Alert rules loaded: $(curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[] | select(.type=="alerting") | .name' | wc -l)"

# Test AlertManager API
curl -s http://localhost:9093/api/v1/status | jq '.data.versionInfo.version'
echo "AlertManager version: $(curl -s http://localhost:9093/api/v1/status | jq -r '.data.versionInfo.version')"

echo "✅ Alert validation complete"
```

## Runbook Integration

### Device Connectivity Runbook
```markdown
# Device Connectivity Issues

## Alert: DeviceDisconnected

### Immediate Actions
1. **Check Physical Connection**
   - Verify USB/Serial cable connection
   - Check device power status
   - Inspect for physical damage

2. **Verify Device Status**
   ```bash
   # List connected devices
   lsusb
   ls /dev/ttyUSB* /dev/ttyACM*
   
   # Check device permissions
   ls -la /dev/ttyUSB0
   ```

3. **Restart Device Connection**
   ```bash
   # Restart profiler service
   systemctl restart tiny-llm-profiler
   
   # Or via Docker
   docker-compose restart profiler
   ```

### Investigation Steps
1. Check device logs: `tail -f /var/log/device-connection.log`
2. Review system dmesg: `dmesg | tail -20`
3. Test manual connection: `screen /dev/ttyUSB0 115200`

### Escalation
If device remains disconnected after 10 minutes:
- Contact hardware team: hardware-team@company.com
- Check for hardware failure
- Consider device replacement
```

### Performance Degradation Runbook
```markdown
# Performance Degradation

## Alert: SlowProfilingPerformance

### Immediate Actions
1. **Check System Resources**
   ```bash
   # CPU and memory usage
   top -n 1
   free -h
   
   # Disk I/O
   iostat -x 1 5
   ```

2. **Review Active Profiling Sessions**
   ```bash
   # Check active sessions
   curl http://localhost:8000/sessions/active
   
   # Review session queue
   curl http://localhost:8000/sessions/queue
   ```

3. **Optimize Resources**
   - Reduce concurrent profiling sessions
   - Clear temporary files: `rm -rf /tmp/profiling_cache/*`
   - Restart profiler if memory usage > 80%

### Investigation Steps
1. Check profiling logs for bottlenecks
2. Review model complexity and size
3. Analyze device performance metrics
4. Consider scaling horizontally

### Prevention
- Implement session queuing limits
- Add automatic resource cleanup
- Monitor trends and capacity planning
```

This comprehensive alerting configuration ensures rapid detection and response to issues across the entire tiny-llm-edge-profiler system.