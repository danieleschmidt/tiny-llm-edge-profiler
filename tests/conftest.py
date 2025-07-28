"""
Pytest configuration and shared fixtures for Tiny LLM Edge Profiler tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch

import pytest
import numpy as np
import pandas as pd
from pyfakefs.fake_filesystem_unittest import Patcher


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring hardware"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "esp32: mark test as ESP32 specific"
    )
    config.addinivalue_line(
        "markers", "stm32: mark test as STM32 specific"
    )
    config.addinivalue_line(
        "markers", "riscv: mark test as RISC-V specific"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "hardware" in str(item.fspath):
            item.add_marker(pytest.mark.hardware)
            
        # Add platform-specific markers
        if "esp32" in str(item.fspath).lower():
            item.add_marker(pytest.mark.esp32)
        elif "stm32" in str(item.fspath).lower():
            item.add_marker(pytest.mark.stm32)
        elif "riscv" in str(item.fspath).lower():
            item.add_marker(pytest.mark.riscv)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_fs():
    """Create a mock filesystem for testing."""
    with Patcher() as patcher:
        yield patcher.fs


@pytest.fixture
def sample_model_config() -> Dict[str, Any]:
    """Sample model configuration for testing."""
    return {
        "name": "test-model",
        "architecture": "llama",
        "quantization": {
            "bits": 4,
            "group_size": 128,
            "symmetric": True
        },
        "size_mb": 3.2,
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_layers": 22,
        "num_attention_heads": 32
    }


@pytest.fixture
def sample_platform_config() -> Dict[str, Any]:
    """Sample platform configuration for testing."""
    return {
        "name": "esp32",
        "architecture": "xtensa",
        "memory": {
            "ram_kb": 520,
            "flash_mb": 4,
            "psram_mb": 8
        },
        "communication": {
            "protocol": "serial",
            "baudrate": 921600,
            "port": "/dev/ttyUSB0"
        },
        "optimization": {
            "use_psram": True,
            "use_dual_core": True,
            "cpu_freq_mhz": 240
        }
    }


@pytest.fixture
def sample_profiling_results() -> Dict[str, Any]:
    """Sample profiling results for testing."""
    return {
        "model_name": "test-model",
        "platform": "esp32",
        "timestamp": "2024-01-01T00:00:00Z",
        "metrics": {
            "latency": {
                "first_token_ms": 150.5,
                "inter_token_ms": 45.2,
                "total_ms": 892.7,
                "tokens_per_second": 11.2
            },
            "memory": {
                "peak_ram_kb": 380.5,
                "flash_usage_kb": 3200,
                "heap_free_kb": 140.0
            },
            "power": {
                "avg_power_mw": 780.5,
                "peak_power_mw": 1200.0,
                "energy_per_token_mj": 69.7
            }
        }
    }


@pytest.fixture
def mock_serial_device():
    """Mock serial device for testing device communication."""
    mock_device = Mock()
    mock_device.is_open = True
    mock_device.baudrate = 921600
    mock_device.port = "/dev/ttyUSB0"
    mock_device.read.return_value = b"OK\n"
    mock_device.write.return_value = 10
    mock_device.readline.return_value = b"READY\n"
    return mock_device


@pytest.fixture
def mock_power_sensor():
    """Mock power sensor for testing power measurement."""
    mock_sensor = Mock()
    mock_sensor.voltage = 3.3
    mock_sensor.current_ma = 230.5
    mock_sensor.power_mw = 760.65
    mock_sensor.shunt_voltage = 0.023
    return mock_sensor


@pytest.fixture
def sample_model_binary(temp_dir: Path) -> Path:
    """Create a sample model binary file for testing."""
    model_file = temp_dir / "test_model.bin"
    # Create a dummy binary file with some content
    with open(model_file, "wb") as f:
        # Write a simple header
        f.write(b"TINYMODEL")
        f.write((123456).to_bytes(4, byteorder='little'))  # Model size
        f.write((4).to_bytes(1, byteorder='little'))       # Quantization bits
        # Write some dummy weights
        f.write(b"\x00" * 1000)
    return model_file


@pytest.fixture
def sample_firmware_binary(temp_dir: Path) -> Path:
    """Create a sample firmware binary for testing."""
    firmware_file = temp_dir / "profiler.bin"
    with open(firmware_file, "wb") as f:
        # Write ESP32 bootloader header
        f.write(b"\xe9\x00\x00\x02")  # ESP32 image header
        f.write(b"\x00" * 100)       # Dummy firmware content
    return firmware_file


@pytest.fixture
def sample_metrics_data() -> pd.DataFrame:
    """Generate sample metrics data for testing analysis."""
    np.random.seed(42)  # For reproducible tests
    
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=1000, freq="1ms"),
        "cpu_usage_percent": np.random.normal(45, 10, 1000),
        "memory_usage_kb": np.random.normal(350, 50, 1000),
        "power_mw": np.random.normal(800, 100, 1000),
        "temperature_c": np.random.normal(55, 5, 1000),
        "inference_latency_ms": np.random.exponential(50, 1000)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_model_runtime():
    """Mock model runtime for testing inference."""
    mock_runtime = Mock()
    mock_runtime.load_model.return_value = True
    mock_runtime.predict.return_value = {
        "tokens": ["hello", "world"],
        "latency_ms": 95.5,
        "memory_used_kb": 420
    }
    mock_runtime.get_model_info.return_value = {
        "size_mb": 3.2,
        "quantization": "4-bit",
        "layers": 22
    }
    return mock_runtime


@pytest.fixture
def mock_platform_adapter():
    """Mock platform adapter for testing."""
    mock_adapter = Mock()
    mock_adapter.connect.return_value = True
    mock_adapter.disconnect.return_value = True
    mock_adapter.deploy_model.return_value = True
    mock_adapter.start_profiling.return_value = "session_123"
    mock_adapter.stop_profiling.return_value = {"status": "success"}
    mock_adapter.collect_metrics.return_value = iter([
        {"timestamp": 1000, "cpu": 45.0, "memory": 350},
        {"timestamp": 1001, "cpu": 47.0, "memory": 355},
    ])
    return mock_adapter


@pytest.fixture(autouse=True)
def mock_hardware_dependencies():
    """Automatically mock hardware dependencies for all tests unless hardware marker is used."""
    def should_mock(request):
        return not any(
            marker.name == "hardware" 
            for marker in request.node.iter_markers()
        )
    
    # Only apply mocks if this is not a hardware test
    if should_mock(pytest.current_request if hasattr(pytest, 'current_request') else None):
        with patch('serial.Serial'), \
             patch('usb.core.find'), \
             patch('smbus2.SMBus'):
            yield
    else:
        yield


@pytest.fixture
def environment_variables():
    """Set up test environment variables."""
    test_env = {
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "MOCK_HARDWARE": "true",
        "DEFAULT_SERIAL_PORT": "/dev/ttyUSB0",
        "DEFAULT_BAUDRATE": "921600",
        "MODEL_CACHE_DIR": "/tmp/test_models",
        "RESULTS_OUTPUT_DIR": "/tmp/test_results"
    }
    
    # Store original values
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_env
    
    # Restore original values
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def caplog_debug(caplog):
    """Capture debug logs for testing."""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


class MockHardwareFixtures:
    """Collection of mock hardware fixtures for different platforms."""
    
    @staticmethod
    @pytest.fixture
    def mock_esp32_device():
        """Mock ESP32 device for testing."""
        device = Mock()
        device.chip_id = "ESP32"
        device.mac_address = "24:0a:c4:12:34:56"
        device.flash_size = 4 * 1024 * 1024  # 4MB
        device.ram_size = 520 * 1024          # 520KB
        device.cpu_freq_mhz = 240
        return device
    
    @staticmethod
    @pytest.fixture
    def mock_stm32_device():
        """Mock STM32 device for testing."""
        device = Mock()
        device.chip_id = "STM32F767"
        device.flash_size = 2 * 1024 * 1024   # 2MB
        device.ram_size = 512 * 1024          # 512KB
        device.cpu_freq_mhz = 216
        return device


# Performance testing fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "min_rounds": 5,
        "max_time": 10.0,
        "warmup": True,
        "disable_gc": True,
    }


# Integration test fixtures
@pytest.fixture(scope="session")
def integration_test_environment():
    """Set up integration test environment."""
    # This would set up test databases, external services, etc.
    # For now, just return a configuration
    return {
        "test_mode": True,
        "external_services": {
            "model_repository": "mock://models",
            "metrics_storage": "memory://"
        }
    }