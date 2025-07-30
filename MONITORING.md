# Monitoring and Observability Configuration

This document outlines the monitoring, logging, and observability strategy for the tiny-llm-edge-profiler project, covering both the profiling application and the target embedded devices.

## Overview

Comprehensive monitoring is essential for:
- **Performance Tracking**: Monitor profiling accuracy and efficiency
- **Device Health**: Track embedded device status and performance
- **Operational Insights**: Understand usage patterns and bottlenecks
- **Incident Response**: Rapid detection and resolution of issues
- **Capacity Planning**: Predict resource needs and scaling requirements

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Edge Devices  │    │  Profiler App   │    │  Observability  │
│                 │    │                 │    │     Stack       │
│ ESP32, STM32,   │───▶│ Metrics         │───▶│                 │
│ RISC-V, etc.    │    │ Collection      │    │ Prometheus      │
│                 │    │                 │    │ Grafana         │
│ Hardware        │    │ Log             │    │ Jaeger          │
│ Metrics         │    │ Aggregation     │    │ ELK Stack       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Metrics Collection

### Application Metrics

#### Profiling Performance Metrics
```python
# Example metrics collection in profiler
from prometheus_client import Counter, Histogram, Gauge

# Profiling session metrics
profiling_sessions_total = Counter(
    'profiling_sessions_total',
    'Total number of profiling sessions',
    ['platform', 'model_type', 'quantization']
)

profiling_duration_seconds = Histogram(
    'profiling_duration_seconds',
    'Time spent profiling models',
    ['platform', 'test_type'],
    buckets=[1, 5, 10, 30, 60, 300, 600]
)

device_connection_status = Gauge(
    'device_connection_status',
    'Device connection status (1=connected, 0=disconnected)',
    ['device_id', 'platform', 'port']
)

# Model performance metrics
model_tokens_per_second = Gauge(
    'model_tokens_per_second',
    'Model inference speed',
    ['model_name', 'platform', 'quantization']
)

model_memory_usage_kb = Gauge(
    'model_memory_usage_kb', 
    'Model memory consumption',
    ['model_name', 'platform', 'memory_type']
)
```

#### Hardware Device Metrics
```python
# Device-specific metrics
device_cpu_utilization = Gauge(
    'device_cpu_utilization_percent',
    'CPU utilization on target device',
    ['device_id', 'platform']
)

device_memory_usage = Gauge(
    'device_memory_usage_bytes',
    'Memory usage on target device',
    ['device_id', 'platform', 'memory_type']
)

device_power_consumption = Gauge(
    'device_power_consumption_mw',
    'Power consumption of target device',
    ['device_id', 'platform']
)

device_temperature = Gauge(
    'device_temperature_celsius',
    'Operating temperature of device',
    ['device_id', 'platform', 'sensor_location']
)
```

### Custom Metrics for Edge AI

#### Model Quality Metrics
```python
model_accuracy_score = Gauge(
    'model_accuracy_score',
    'Model accuracy on test dataset',
    ['model_name', 'dataset', 'platform']
)

model_latency_p95 = Gauge(
    'model_latency_p95_ms',
    '95th percentile inference latency',
    ['model_name', 'platform', 'input_length']
)

quantization_error_rate = Gauge(
    'quantization_error_rate',
    'Error rate introduced by quantization',
    ['model_name', 'quantization_bits', 'method']
)
```

## Logging Configuration

### Structured Logging Setup
```python
# logging_config.py
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add profiling-specific context
        if hasattr(record, 'device_id'):
            log_entry['device_id'] = record.device_id
        if hasattr(record, 'platform'):
            log_entry['platform'] = record.platform
        if hasattr(record, 'model_name'):
            log_entry['model_name'] = record.model_name
            
        return json.dumps(log_entry)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/profiler.jsonl')
    ],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Log Categories

#### Profiling Operations
```python
logger = logging.getLogger('profiler.operations')

# Device connection events
logger.info("Device connected", extra={
    'device_id': device.id,
    'platform': device.platform,
    'port': device.port,
    'firmware_version': device.firmware_version
})

# Profiling session events  
logger.info("Profiling session started", extra={
    'session_id': session.id,
    'model_name': model.name,
    'platform': device.platform,
    'test_duration': session.duration
})
```

#### Hardware Events
```python
hardware_logger = logging.getLogger('profiler.hardware')

# Device health monitoring
hardware_logger.warning("High device temperature", extra={
    'device_id': device.id,
    'temperature_celsius': temp,
    'threshold_celsius': TEMP_THRESHOLD
})

# Performance anomalies
hardware_logger.error("Memory allocation failed", extra={
    'device_id': device.id,
    'requested_bytes': size,
    'available_bytes': available
})
```

## Observability Stack Setup

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'tiny-llm-profiler'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
    metrics_path: /metrics
    
  - job_name: 'device-metrics'
    static_configs:
      - targets: ['device-exporter:9100']
    scrape_interval: 10s
    
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

#### Main Profiling Dashboard
```json
{
  "dashboard": {
    "title": "Tiny LLM Edge Profiler - Main Dashboard",
    "panels": [
      {
        "title": "Active Profiling Sessions",
        "type": "stat",
        "targets": [
          {
            "expr": "profiling_sessions_active",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Model Performance by Platform",
        "type": "graph", 
        "targets": [
          {
            "expr": "avg by (platform) (model_tokens_per_second)",
            "refId": "A"
          }
        ]
      },
      {
        "title": "Device Connection Status",
        "type": "table",
        "targets": [
          {
            "expr": "device_connection_status",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

#### Device Health Dashboard
```json
{
  "dashboard": {
    "title": "Device Health Monitoring",
    "panels": [
      {
        "title": "Device Temperature",
        "type": "graph",
        "targets": [
          {
            "expr": "device_temperature_celsius",
            "refId": "A"
          }
        ],
        "alert": {
          "conditions": [
            {
              "query": {"queryType": "", "refId": "A"},
              "reducer": {"type": "last", "params": []},
              "evaluator": {"params": [85], "type": "gt"}
            }
          ]
        }
      }
    ]
  }
}
```

### Jaeger Tracing Configuration
```yaml
# jaeger-config.yml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:1.35
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=memory
```

#### Distributed Tracing Setup
```python
# tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Trace profiling operations
@tracer.start_as_current_span("profile_model")
def profile_model(model, device):
    with tracer.start_as_current_span("device_connection"):
        device.connect()
        
    with tracer.start_as_current_span("model_loading"):
        device.load_model(model)
        
    with tracer.start_as_current_span("inference_profiling"):
        results = device.run_profiling()
        
    return results
```

## Alerting Configuration

### Critical Alerts
```yaml
# alerts/critical.yml
groups:
  - name: device_critical
    rules:
      - alert: DeviceDisconnected
        expr: device_connection_status == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Device {{ $labels.device_id }} disconnected"
          
      - alert: DeviceOverheating
        expr: device_temperature_celsius > 85
        for: 60s
        labels:
          severity: critical
        annotations:
          summary: "Device {{ $labels.device_id }} overheating"
          
      - alert: ProfilingSessionFailed
        expr: increase(profiling_errors_total[5m]) > 3
        for: 0s
        labels:
          severity: critical
        annotations:
          summary: "Multiple profiling session failures"
```

### Performance Alerts
```yaml
# alerts/performance.yml
groups:
  - name: performance_degradation
    rules:
      - alert: LowThroughput
        expr: model_tokens_per_second < 1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Low model throughput on {{ $labels.platform }}"
          
      - alert: HighMemoryUsage
        expr: device_memory_usage_bytes / device_memory_total_bytes > 0.9
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.device_id }}"
```

## Docker Compose Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.40.0
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/rules:/etc/prometheus/rules
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:9.3.0
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

  jaeger:
    image: jaegertracing/all-in-one:1.35
    container_name: jaeger
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    container_name: kibana
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200

volumes:
  grafana-storage:
  elasticsearch-data:
```

## Performance Monitoring Integration

### Makefile Targets
```makefile
# monitoring targets
monitor-start: ## Start monitoring stack
	docker-compose -f docker-compose.monitoring.yml up -d

monitor-stop: ## Stop monitoring stack
	docker-compose -f docker-compose.monitoring.yml down

monitor-logs: ## View monitoring logs
	docker-compose -f docker-compose.monitoring.yml logs -f

monitor-dashboards: ## Open Grafana dashboards
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jaeger: http://localhost:16686"
	@echo "Kibana: http://localhost:5601"
```

### CI/CD Integration
```yaml
# .github/workflows/monitoring.yml
name: Monitoring Setup

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  monitoring-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start monitoring stack
        run: make monitor-start
        
      - name: Wait for services
        run: |
          sleep 30
          curl -f http://localhost:9090/api/v1/status/config
          curl -f http://localhost:3000/api/health
          
      - name: Run monitoring tests
        run: pytest tests/monitoring/ -v
        
      - name: Cleanup
        run: make monitor-stop
```

This comprehensive monitoring setup provides full observability into both the profiling application and the target embedded devices, enabling effective performance tracking, debugging, and operational insights.