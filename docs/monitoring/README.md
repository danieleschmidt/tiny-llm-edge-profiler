# Monitoring Documentation

This directory contains comprehensive monitoring and observability documentation for the tiny-llm-edge-profiler project.

## Contents

- **[Health Checks](./health-checks.md)** - Health check endpoint configurations and monitoring
- **[Structured Logging](./structured-logging.md)** - Logging configuration and best practices
- **[Metrics Collection](./metrics-collection.md)** - Prometheus metrics and custom monitoring
- **[Alerting](./alerting.md)** - Alert configuration and incident response procedures
- **[Performance Monitoring](./performance-monitoring.md)** - Performance tracking and optimization
- **[Dashboard Templates](./dashboards/)** - Grafana dashboard configurations
- **[Observability Best Practices](./best-practices.md)** - Guidelines for effective monitoring

## Quick Start

1. **Start Monitoring Stack**:
   ```bash
   make monitor-start
   ```

2. **Access Dashboards**:
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
   - Jaeger: http://localhost:16686
   - Kibana: http://localhost:5601

3. **Verify Health**:
   ```bash
   curl http://localhost:8000/health
   ```

## Architecture Overview

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

## Integration Points

- **Application Health**: `/health` and `/metrics` endpoints
- **Device Monitoring**: Hardware-specific metric collection
- **Distributed Tracing**: OpenTelemetry integration with Jaeger
- **Log Aggregation**: Structured logging with ELK stack
- **Alerting**: Prometheus AlertManager integration

For detailed implementation guides, see the individual documentation files in this directory.