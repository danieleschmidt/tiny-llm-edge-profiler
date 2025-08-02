# Health Check Configuration

Health checks are essential for monitoring the availability and operational status of the tiny-llm-edge-profiler application and connected devices.

## Application Health Endpoints

### Basic Health Check
```python
# src/tiny_llm_profiler/health.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time
import psutil
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "tiny-llm-edge-profiler",
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with system metrics"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Device connection status
        device_status = await check_device_connections()
        
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "tiny-llm-edge-profiler",
            "version": "1.0.0",
            "system": {
                "cpu_percent": cpu_percent,
                "memory_used_percent": memory.percent,
                "memory_available_mb": memory.available // (1024 * 1024),
                "disk_used_percent": (disk.used / disk.total) * 100,
                "disk_free_gb": disk.free // (1024 * 1024 * 1024)
            },
            "devices": device_status,
            "dependencies": await check_dependencies()
        }
        
        # Determine overall health
        if cpu_percent > 90 or memory.percent > 90:
            health_data["status"] = "degraded"
            health_data["warnings"] = ["High resource utilization"]
        
        if not any(device["connected"] for device in device_status.values()):
            health_data["status"] = "unhealthy"
            health_data["errors"] = ["No devices connected"]
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

async def check_device_connections() -> Dict[str, Dict[str, Any]]:
    """Check status of connected devices"""
    # This would integrate with your device manager
    return {
        "esp32_dev_1": {
            "connected": True,
            "platform": "ESP32",
            "firmware_version": "1.2.3",
            "last_seen": time.time() - 30,
            "temperature_celsius": 45.2
        },
        "stm32_dev_2": {
            "connected": False,
            "platform": "STM32",
            "firmware_version": "unknown",
            "last_seen": time.time() - 300,
            "temperature_celsius": None
        }
    }

async def check_dependencies() -> Dict[str, str]:
    """Check external dependencies"""
    dependencies = {}
    
    # Check database connection
    try:
        # Add your database health check here
        dependencies["database"] = "healthy"
    except Exception:
        dependencies["database"] = "unhealthy"
    
    # Check external APIs
    try:
        # Add external service checks here
        dependencies["model_registry"] = "healthy"
    except Exception:
        dependencies["model_registry"] = "unhealthy"
    
    return dependencies
```

### Kubernetes Health Probes
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tiny-llm-profiler
spec:
  template:
    spec:
      containers:
      - name: profiler
        image: tiny-llm-profiler:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/detailed
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 10
```

## Device Health Monitoring

### Hardware Health Checks
```python
# Device health monitoring
class DeviceHealthMonitor:
    def __init__(self, device):
        self.device = device
        self.health_metrics = {}
    
    async def check_device_health(self) -> Dict[str, Any]:
        """Comprehensive device health check"""
        health_status = {
            "device_id": self.device.id,
            "platform": self.device.platform,
            "timestamp": time.time(),
            "status": "healthy",
            "metrics": {},
            "alerts": []
        }
        
        try:
            # Temperature monitoring
            temp = await self.device.get_temperature()
            health_status["metrics"]["temperature_celsius"] = temp
            if temp > 85:
                health_status["status"] = "warning"
                health_status["alerts"].append("High temperature detected")
            
            # Memory usage
            memory_info = await self.device.get_memory_info()
            health_status["metrics"]["memory_used_percent"] = memory_info["used_percent"]
            if memory_info["used_percent"] > 90:
                health_status["status"] = "critical"
                health_status["alerts"].append("Memory usage critical")
            
            # CPU utilization
            cpu_percent = await self.device.get_cpu_usage()
            health_status["metrics"]["cpu_percent"] = cpu_percent
            
            # Power consumption
            power_mw = await self.device.get_power_consumption()
            health_status["metrics"]["power_consumption_mw"] = power_mw
            
            # Connectivity test
            connectivity = await self.device.test_connectivity()
            health_status["metrics"]["connectivity_ms"] = connectivity["latency_ms"]
            if not connectivity["success"]:
                health_status["status"] = "unhealthy"
                health_status["alerts"].append("Device connectivity failed")
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
            health_status["alerts"].append(f"Health check error: {e}")
        
        return health_status
    
    async def continuous_monitoring(self, interval_seconds: int = 30):
        """Continuous health monitoring loop"""
        while True:
            health_data = await self.check_device_health()
            
            # Log health status
            logger.info("Device health check", extra={
                "device_id": health_data["device_id"],
                "status": health_data["status"],
                "metrics": health_data["metrics"]
            })
            
            # Update Prometheus metrics
            device_health_status.labels(
                device_id=health_data["device_id"],
                platform=health_data["platform"]
            ).set(1 if health_data["status"] == "healthy" else 0)
            
            await asyncio.sleep(interval_seconds)
```

## Health Check Integration

### Docker Compose Health Checks
```yaml
# docker-compose.yml
version: '3.8'
services:
  profiler:
    build: .
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    depends_on:
      prometheus:
        condition: service_healthy
        
  prometheus:
    image: prom/prometheus:v2.40.0
    ports:
      - "9090:9090"
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Makefile Health Check Targets
```makefile
# Health check targets
health-check: ## Check application health
	@echo "Checking application health..."
	@curl -s http://localhost:8000/health | jq '.'

health-detailed: ## Detailed health check
	@echo "Running detailed health check..."
	@curl -s http://localhost:8000/health/detailed | jq '.'

health-devices: ## Check device health status
	@echo "Checking device health..."
	@curl -s http://localhost:8000/devices/health | jq '.'

health-monitoring: ## Check monitoring stack health
	@echo "Prometheus:"
	@curl -s http://localhost:9090/-/healthy || echo "Prometheus not healthy"
	@echo "Grafana:"
	@curl -s http://localhost:3000/api/health | jq '.database' || echo "Grafana not healthy"
	@echo "Jaeger:"
	@curl -s http://localhost:16686/ > /dev/null && echo "Jaeger healthy" || echo "Jaeger not healthy"
```

### Automated Health Monitoring
```python
# scripts/health_monitor.py
#!/usr/bin/env python3
"""
Automated health monitoring script for continuous deployment
"""
import asyncio
import aiohttp
import logging
import time
from typing import Dict, List

class HealthMonitor:
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.logger = logging.getLogger(__name__)
    
    async def check_endpoint(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Check a single health endpoint"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"url": url, "status": "healthy", "data": data}
                else:
                    return {"url": url, "status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"url": url, "status": "error", "error": str(e)}
    
    async def run_health_checks(self) -> List[Dict]:
        """Run health checks on all endpoints"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.check_endpoint(session, url) for url in self.endpoints]
            return await asyncio.gather(*tasks)
    
    async def continuous_monitoring(self, interval: int = 60):
        """Continuously monitor health endpoints"""
        while True:
            results = await self.run_health_checks()
            
            healthy_count = sum(1 for r in results if r["status"] == "healthy")
            total_count = len(results)
            
            self.logger.info(f"Health check complete: {healthy_count}/{total_count} healthy")
            
            for result in results:
                if result["status"] != "healthy":
                    self.logger.warning(f"Unhealthy endpoint: {result}")
            
            await asyncio.sleep(interval)

if __name__ == "__main__":
    endpoints = [
        "http://localhost:8000/health",
        "http://localhost:8000/health/detailed",
        "http://localhost:9090/-/healthy",
        "http://localhost:3000/api/health"
    ]
    
    monitor = HealthMonitor(endpoints)
    asyncio.run(monitor.continuous_monitoring())
```

This health check configuration provides comprehensive monitoring of both the application and connected devices, ensuring reliable operation and early detection of issues.