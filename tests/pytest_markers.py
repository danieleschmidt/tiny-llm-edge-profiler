"""
Pytest marker definitions and helper functions for test categorization.
Provides utilities for conditional test execution based on hardware availability.
"""

import pytest
import subprocess
import sys
from typing import List, Dict, Any
from pathlib import Path


def skip_if_no_hardware(platform: str = None) -> pytest.MarkDecorator:
    """Skip test if specified hardware platform is not available."""
    try:
        if platform:
            # Check for specific platform
            result = subprocess.run(
                [sys.executable, "-c", f"""
import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
found = any('{platform}' in p.description.lower() for p in ports)
sys.exit(0 if found else 1)
"""],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                return pytest.mark.skip(f"No {platform} hardware detected")
        else:
            # Check for any hardware
            result = subprocess.run(
                [sys.executable, "-c", """
import serial.tools.list_ports
ports = list(serial.tools.list_ports.comports())
sys.exit(0 if ports else 1)
"""],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                return pytest.mark.skip("No hardware devices detected")
        
        return pytest.mark.skipif(False, reason="")
    except Exception:
        return pytest.mark.skip(f"Hardware detection failed for {platform or 'any'}")


def slow_test(reason: str = "Test takes longer than 5 seconds") -> pytest.MarkDecorator:
    """Mark test as slow with optional reason."""
    return pytest.mark.slow


def requires_network() -> pytest.MarkDecorator:
    """Mark test as requiring network connectivity."""
    return pytest.mark.network


def performance_test(benchmark_group: str = "default") -> pytest.MarkDecorator:
    """Mark test as performance benchmark with optional group."""
    return pytest.mark.performance


def hardware_test(platform: str) -> pytest.MarkDecorator:
    """Mark test as requiring specific hardware platform."""
    marker = getattr(pytest.mark, platform.lower())
    return pytest.mark.combine(pytest.mark.hardware, marker)


def integration_test(service: str = None) -> pytest.MarkDecorator:
    """Mark test as integration test with optional service specification."""
    if service:
        return pytest.mark.combine(pytest.mark.integration, 
                                 pytest.mark.skipif(
                                     not _service_available(service),
                                     reason=f"Service {service} not available"
                                 ))
    return pytest.mark.integration


def _service_available(service: str) -> bool:
    """Check if external service is available."""
    # Implement service availability checks
    service_checks = {
        "mqtt": _check_mqtt_service,
        "serial": _check_serial_service,
        "docker": _check_docker_service,
    }
    
    checker = service_checks.get(service.lower())
    if checker:
        return checker()
    return True


def _check_mqtt_service() -> bool:
    """Check if MQTT service is available."""
    try:
        import asyncio_mqtt
        return True
    except ImportError:
        return False


def _check_serial_service() -> bool:
    """Check if serial devices are available."""
    try:
        import serial.tools.list_ports
        return len(list(serial.tools.list_ports.comports())) > 0
    except ImportError:
        return False


def _check_docker_service() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


# Marker combinations for common patterns
esp32_hardware = pytest.mark.combine(pytest.mark.hardware, pytest.mark.esp32)
stm32_hardware = pytest.mark.combine(pytest.mark.hardware, pytest.mark.stm32)
riscv_hardware = pytest.mark.combine(pytest.mark.hardware, pytest.mark.riscv)
rp2040_hardware = pytest.mark.combine(pytest.mark.hardware, pytest.mark.rp2040)

slow_integration = pytest.mark.combine(pytest.mark.slow, pytest.mark.integration)
performance_memory = pytest.mark.combine(pytest.mark.performance, pytest.mark.memory)
security_unit = pytest.mark.combine(pytest.mark.security, pytest.mark.unit)