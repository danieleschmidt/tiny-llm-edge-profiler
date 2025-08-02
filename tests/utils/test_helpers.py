"""Test utility functions and helpers."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Generator
from contextlib import contextmanager
import time
import threading
import json
import yaml
import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestEnvironment:
    """Test environment manager for consistent test setup."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.temp_dir: Optional[Path] = None
        self.original_env = {}
        self.mocks = {}
        
    def __enter__(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"test_{self.test_name}_"))
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        # Restore environment variables
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    def set_env_var(self, key: str, value: str) -> None:
        """Set environment variable and remember original value."""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    def create_file(self, relative_path: str, content: Union[str, bytes]) -> Path:
        """Create a file in the test directory."""
        if not self.temp_dir:
            raise RuntimeError("Test environment not initialized")
            
        file_path = self.temp_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, str):
            file_path.write_text(content)
        else:
            file_path.write_bytes(content)
            
        return file_path
    
    def create_config_file(self, filename: str, config: Dict[str, Any]) -> Path:
        """Create a configuration file."""
        if filename.endswith('.json'):
            content = json.dumps(config, indent=2)
        elif filename.endswith(('.yaml', '.yml')):
            content = yaml.dump(config, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config format: {filename}")
            
        return self.create_file(filename, content)
    
    def create_mock_model(self, size_mb: float = 5.0, quantization: int = 4) -> Path:
        """Create a mock model file."""
        model_data = np.random.randint(0, 255, size=int(size_mb * 1024 * 1024), dtype=np.uint8)
        return self.create_file("model.bin", model_data.tobytes())


class MockDevice:
    """Mock hardware device for testing."""
    
    def __init__(self, platform: str = "esp32", responses: List[bytes] = None):
        self.platform = platform
        self.responses = responses or [b"OK\n"]
        self.response_index = 0
        self.written_data = []
        self.is_connected = True
        self.firmware_version = "1.0.0"
        
    def write(self, data: bytes) -> int:
        """Mock write operation."""
        if not self.is_connected:
            raise RuntimeError("Device not connected")
        self.written_data.append(data)
        return len(data)
    
    def read(self, size: int = 1024) -> bytes:
        """Mock read operation."""
        if not self.is_connected:
            raise RuntimeError("Device not connected")
            
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        return b""
    
    def readline(self) -> bytes:
        """Mock readline operation."""
        return self.read()
    
    def reset(self) -> None:
        """Mock device reset."""
        self.response_index = 0
        self.written_data.clear()
        time.sleep(0.1)  # Simulate reset time
    
    def flash_firmware(self, firmware_data: bytes) -> bool:
        """Mock firmware flashing."""
        time.sleep(2.0)  # Simulate flash time
        return True
    
    def get_system_info(self) -> Dict[str, Any]:
        """Mock system information."""
        return {
            "platform": self.platform,
            "firmware_version": self.firmware_version,
            "free_heap": 250000,
            "cpu_freq_mhz": 240,
            "flash_size_mb": 4,
            "uptime_seconds": 12345
        }


class MockPowerSensor:
    """Mock power measurement sensor."""
    
    def __init__(self, base_power_mw: float = 200.0):
        self.base_power_mw = base_power_mw
        self.is_measuring = False
        self.measurements = []
        self._measurement_thread = None
        
    def start_measurement(self, sample_rate_hz: int = 1000) -> None:
        """Start power measurement."""
        self.is_measuring = True
        self.measurements.clear()
        
        def measure():
            while self.is_measuring:
                # Generate realistic power consumption with noise
                power_mw = self.base_power_mw * (1.0 + np.random.normal(0, 0.1))
                power_mw = max(10.0, power_mw)  # Minimum power
                
                voltage_v = 3.3 + np.random.normal(0, 0.05)
                current_ma = power_mw / voltage_v
                
                self.measurements.append({
                    "timestamp": time.time(),
                    "voltage_v": round(voltage_v, 3),
                    "current_ma": round(current_ma, 2),
                    "power_mw": round(power_mw, 2)
                })
                
                time.sleep(1.0 / sample_rate_hz)
        
        self._measurement_thread = threading.Thread(target=measure)
        self._measurement_thread.daemon = True
        self._measurement_thread.start()
    
    def stop_measurement(self) -> List[Dict[str, float]]:
        """Stop measurement and return results."""
        self.is_measuring = False
        if self._measurement_thread:
            self._measurement_thread.join(timeout=1.0)
        return self.measurements.copy()
    
    def get_instant_reading(self) -> Dict[str, float]:
        """Get instant power reading."""
        power_mw = self.base_power_mw * (1.0 + np.random.normal(0, 0.05))
        voltage_v = 3.3 + np.random.normal(0, 0.02)
        current_ma = power_mw / voltage_v
        
        return {
            "voltage_v": round(voltage_v, 3),
            "current_ma": round(current_ma, 2),
            "power_mw": round(power_mw, 2)
        }


@contextmanager
def temporary_file(content: Union[str, bytes], suffix: str = ".tmp") -> Generator[Path, None, None]:
    """Context manager for temporary files."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        if isinstance(content, str):
            tmp_file.write(content.encode())
        else:
            tmp_file.write(content)
        tmp_file.flush()
        
        try:
            yield Path(tmp_file.name)
        finally:
            Path(tmp_file.name).unlink(missing_ok=True)


@contextmanager
def mock_serial_port(responses: List[bytes] = None) -> Generator[Mock, None, None]:
    """Context manager for mocking serial ports."""
    responses = responses or [b"OK\n"]
    
    mock_serial = Mock()
    mock_serial.write.return_value = None
    mock_serial.read.side_effect = responses
    mock_serial.readline.side_effect = responses
    mock_serial.is_open = True
    mock_serial.baudrate = 921600
    mock_serial.timeout = 1.0
    
    with patch('serial.Serial', return_value=mock_serial):
        yield mock_serial


@contextmanager
def capture_logs(logger_name: str = None) -> Generator[List[str], None, None]:
    """Context manager to capture log messages."""
    import logging
    
    log_capture = []
    
    class TestHandler(logging.Handler):
        def emit(self, record):
            log_capture.append(self.format(record))
    
    handler = TestHandler()
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    try:
        yield log_capture
    finally:
        logger.removeHandler(handler)


def assert_within_tolerance(actual: float, expected: float, tolerance_percent: float = 5.0) -> None:
    """Assert that actual value is within tolerance of expected value."""
    tolerance = abs(expected * tolerance_percent / 100.0)
    assert abs(actual - expected) <= tolerance, (
        f"Value {actual} not within {tolerance_percent}% of expected {expected} "
        f"(tolerance: ±{tolerance})"
    )


def assert_performance_acceptable(
    execution_time: float,
    max_time: float,
    operation_name: str = "operation"
) -> None:
    """Assert that execution time meets performance requirements."""
    assert execution_time <= max_time, (
        f"{operation_name} took {execution_time:.3f}s, "
        f"which exceeds maximum allowed time of {max_time:.3f}s"
    )


def create_sample_profiling_data(
    platform: str = "esp32",
    model_size_mb: float = 5.0,
    num_tokens: int = 100
) -> Dict[str, Any]:
    """Create sample profiling data for testing."""
    return {
        "metadata": {
            "platform": platform,
            "model_size_mb": model_size_mb,
            "test_timestamp": time.time(),
            "profiler_version": "1.0.0"
        },
        "metrics": {
            "tokens_generated": num_tokens,
            "total_time_ms": 2500.0,
            "first_token_latency_ms": 95.2,
            "tokens_per_second": 40.0,
            "peak_memory_kb": 380,
            "average_power_mw": 205.5
        },
        "per_token_data": [
            {
                "token_index": i,
                "latency_ms": 25.0 + np.random.normal(0, 5),
                "memory_kb": 300 + np.random.randint(-50, 50),
                "power_mw": 200 + np.random.normal(0, 20)
            }
            for i in range(num_tokens)
        ]
    }


def wait_for_condition(
    condition_func: callable,
    timeout_seconds: float = 10.0,
    check_interval: float = 0.1,
    error_message: str = "Condition not met within timeout"
) -> bool:
    """Wait for a condition to become true."""
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        if condition_func():
            return True
        time.sleep(check_interval)
    
    raise TimeoutError(error_message)


class AsyncMockDevice:
    """Async mock device for testing async operations."""
    
    def __init__(self, responses: List[bytes] = None, delay: float = 0.01):
        self.responses = responses or [b"OK\n"]
        self.response_index = 0
        self.delay = delay
        self.written_data = []
        
    async def write(self, data: bytes) -> None:
        """Async write operation."""
        await self._simulate_io_delay()
        self.written_data.append(data)
        
    async def read(self, size: int = 1024) -> bytes:
        """Async read operation."""
        await self._simulate_io_delay()
        
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        return b""
    
    async def _simulate_io_delay(self) -> None:
        """Simulate I/O delay."""
        import asyncio
        await asyncio.sleep(self.delay)


def parametrize_platforms(test_func):
    """Decorator to parametrize tests across multiple platforms."""
    platforms = ["esp32", "stm32f4", "stm32f7", "rp2040", "nrf52840"]
    return pytest.mark.parametrize("platform", platforms)(test_func)


def parametrize_quantization_levels(test_func):
    """Decorator to parametrize tests across quantization levels."""
    quantization_levels = [2, 3, 4, 8]
    return pytest.mark.parametrize("quantization", quantization_levels)(test_func)


def skip_if_no_hardware(test_func):
    """Decorator to skip tests if hardware is not available."""
    def wrapper(*args, **kwargs):
        if os.environ.get("SKIP_HARDWARE_TESTS", "false").lower() == "true":
            pytest.skip("Hardware tests disabled")
        return test_func(*args, **kwargs)
    
    return wrapper


def requires_device(platform: str):
    """Decorator to mark tests as requiring specific hardware."""
    def decorator(test_func):
        return pytest.mark.parametrize("required_platform", [platform])(test_func)
    return decorator


class MemoryLeakDetector:
    """Utility to detect memory leaks in tests."""
    
    def __init__(self):
        self.initial_memory = 0
        self.peak_memory = 0
        
    def __enter__(self):
        """Start memory monitoring."""
        import psutil
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Check for memory leaks."""
        import psutil
        process = psutil.Process()
        final_memory = process.memory_info().rss
        
        # Allow for some memory growth (10MB threshold)
        memory_growth_mb = (final_memory - self.initial_memory) / 1024 / 1024
        
        if memory_growth_mb > 10:
            pytest.fail(f"Potential memory leak detected: {memory_growth_mb:.2f} MB growth")


def create_test_fixtures_directory(base_path: Path) -> Path:
    """Create a directory with common test fixtures."""
    fixtures_dir = base_path / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    
    # Create sample model files
    for quantization in [2, 4, 8]:
        model_data = np.random.randint(0, 255, size=1024*1024, dtype=np.uint8)  # 1MB
        model_file = fixtures_dir / f"test_model_{quantization}bit.bin"
        model_file.write_bytes(model_data.tobytes())
    
    # Create sample firmware files
    firmware_data = b"\xFF" * (512 * 1024)  # 512KB firmware
    firmware_file = fixtures_dir / "test_firmware.bin"
    firmware_file.write_bytes(firmware_data)
    
    # Create configuration files
    config_data = {
        "platform": "esp32",
        "baudrate": 921600,
        "timeout": 30,
        "optimization_level": "O2"
    }
    config_file = fixtures_dir / "test_config.yaml"
    config_file.write_text(yaml.dump(config_data))
    
    return fixtures_dir


# Global test utilities

def get_test_data_path() -> Path:
    """Get the path to test data directory."""
    return Path(__file__).parent.parent / "fixtures"


def load_test_config(config_name: str) -> Dict[str, Any]:
    """Load test configuration by name."""
    config_path = get_test_data_path() / f"{config_name}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def generate_mock_metrics(duration_seconds: int = 30, sample_rate: int = 10) -> List[Dict]:
    """Generate mock performance metrics."""
    samples = []
    for i in range(duration_seconds * sample_rate):
        timestamp = i / sample_rate
        samples.append({
            "timestamp": timestamp,
            "cpu_percent": 70 + np.random.normal(0, 10),
            "memory_kb": 300 + np.random.randint(-20, 50),
            "power_mw": 200 + np.random.normal(0, 25),
            "temperature_c": 55 + np.random.normal(0, 5)
        })
    return samples