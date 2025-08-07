"""Shared test configuration and fixtures for tiny-llm-edge-profiler."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Generator

import numpy as np


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_model_path(tmp_path: Path) -> Path:
    """Create a sample model file for testing."""
    model_path = tmp_path / "test_model.bin"
    # Create a minimal binary file representing a quantized model
    model_data = np.random.randint(0, 255, size=1024, dtype=np.uint8)
    model_path.write_bytes(model_data.tobytes())
    return model_path


@pytest.fixture
def mock_serial_device() -> Mock:
    """Mock serial device for testing hardware communication."""
    device = Mock()
    device.write.return_value = None
    device.read.return_value = b"OK\n"
    device.readline.return_value = b"READY\n"
    device.is_open = True
    device.close.return_value = None
    return device


@pytest.fixture
def esp32_platform_config() -> Dict[str, Any]:
    """Configuration for ESP32 platform testing."""
    return {
        "name": "esp32",
        "architecture": "xtensa-lx6",
        "ram_kb": 520,
        "flash_mb": 4,
        "cpu_freq_mhz": 240,
        "has_psram": True,
        "uart_baudrate": 921600,
        "supported_quantizations": [2, 3, 4],
    }


@pytest.fixture
def stm32_platform_config() -> Dict[str, Any]:
    """Configuration for STM32F4 platform testing."""
    return {
        "name": "stm32f4",
        "architecture": "arm-cortex-m4",
        "ram_kb": 192,
        "flash_mb": 1,
        "cpu_freq_mhz": 168,
        "has_fpu": True,
        "uart_baudrate": 115200,
        "supported_quantizations": [4],
    }


@pytest.fixture
def mock_power_sensor() -> Mock:
    """Mock power measurement sensor."""
    sensor = Mock()
    sensor.voltage = 3.3
    sensor.current_ma = 125.5
    sensor.power_mw = 414.15
    sensor.read_voltage.return_value = 3.3
    sensor.read_current.return_value = 0.1255  # Amperes
    sensor.read_power.return_value = 0.41415   # Watts
    return sensor


@pytest.fixture
def sample_profiling_results() -> Dict[str, Any]:
    """Sample profiling results for testing analysis functions."""
    return {
        "platform": "esp32",
        "model_size_bytes": 2_800_000,
        "quantization_bits": 2,
        "test_prompts": ["Hello", "How are you?", "Explain AI"],
        "metrics": {
            "first_token_latency_ms": 95.2,
            "inter_token_latency_ms": 45.1,
            "tokens_per_second": 22.15,
            "peak_memory_kb": 380,
            "average_power_mw": 205.5,
            "energy_per_token_mj": 9.3,
            "total_inference_time_ms": 892.5,
            "accuracy_score": 0.87,
        },
        "system_info": {
            "cpu_usage_percent": 85.2,
            "memory_usage_percent": 73.1,
            "temperature_celsius": 62.8,
            "frequency_mhz": 240,
        },
        "timestamp": "2024-01-15T10:30:00Z",
    }


@pytest.fixture
def mock_quantized_model() -> Mock:
    """Mock quantized model for testing."""
    model = Mock()
    model.size_bytes = 2_800_000
    model.quantization_bits = 2
    model.vocab_size = 32000
    model.layers = 22
    model.hidden_size = 2048
    model.is_loaded = True
    model.optimize_for_platform.return_value = model
    model.export.return_value = None
    return model


@pytest.fixture
def hardware_test_skip() -> Generator[None, None, None]:
    """Skip tests that require actual hardware when not available."""
    import sys
    import os
    
    # Check if we're in a CI environment or hardware is not available
    if "CI" in os.environ or "--hardware" not in sys.argv:
        pytest.skip("Hardware tests skipped - use --hardware flag to enable")
    yield


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """Create temporary configuration directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create sample platform configs
    esp32_config = config_dir / "esp32.yaml"
    esp32_config.write_text("""
name: esp32
architecture: xtensa-lx6
memory:
  ram_kb: 520
  flash_mb: 4
cpu:
  freq_mhz: 240
  cores: 2
features:
  - psram
  - dual_core
  - wifi
""")
    
    return config_dir


@pytest.fixture
def mock_file_system() -> Dict[str, bytes]:
    """Mock file system for testing file operations."""
    return {
        "/tmp/model.bin": b"\x00" * 1024,
        "/tmp/firmware.bin": b"\xFF" * 2048,
        "/tmp/config.yaml": b"test: true\n",
    }


class MockAsyncSerial:
    """Mock async serial connection for testing."""
    
    def __init__(self, responses=None):
        self.responses = responses or [b"OK\n"]
        self.response_index = 0
        self.written_data = []
        
    async def write(self, data: bytes):
        self.written_data.append(data)
        
    async def read(self, size: int = 1) -> bytes:
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        return b""
        
    async def close(self):
        pass
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


@pytest.fixture
def mock_async_serial() -> MockAsyncSerial:
    """Mock async serial connection."""
    return MockAsyncSerial()


# Pytest markers for different test categories
pytestmark = [
    pytest.mark.asyncio,
]