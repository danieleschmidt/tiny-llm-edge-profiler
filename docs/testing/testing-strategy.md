# Testing Strategy - Tiny LLM Edge Profiler

## 🧪 Comprehensive Testing Framework

This document outlines the testing strategy for ensuring reliable profiling across diverse edge hardware platforms.

## Testing Pyramid

### Unit Tests (70%)
- **Fast execution** (< 1ms per test)
- **No hardware dependencies**
- **High code coverage** (>90%)
- **Mock external dependencies**

### Integration Tests (20%)
- **Real hardware communication**
- **End-to-end workflows**
- **Multi-component interactions**
- **Performance validation**

### Hardware-in-the-Loop Tests (10%)
- **Physical device testing**
- **Real model profiling**
- **Power measurement validation**
- **Long-duration stress tests**

## Test Categories

### 1. Unit Tests

#### Core Components
```python
# Model handling
tests/unit/test_model_loader.py
tests/unit/test_model_quantizer.py
tests/unit/test_model_optimizer.py

# Platform abstraction
tests/unit/test_platform_manager.py
tests/unit/test_device_detector.py
tests/unit/test_communication_layer.py

# Profiling logic
tests/unit/test_metrics_calculator.py
tests/unit/test_performance_analyzer.py
tests/unit/test_result_formatter.py
```

#### Test Structure
```python
import pytest
from unittest.mock import Mock, patch
from tiny_llm_profiler import ModelLoader

class TestModelLoader:
    def test_load_valid_model(self):
        """Test loading a valid quantized model."""
        loader = ModelLoader()
        model = loader.load("tests/fixtures/valid_model.bin")
        
        assert model.size_mb > 0
        assert model.quantization_bits in [2, 3, 4, 8]
        assert model.vocab_size > 0
        
    def test_load_invalid_model_raises_error(self):
        """Test that invalid model raises appropriate error."""
        loader = ModelLoader()
        
        with pytest.raises(ModelFormatError) as exc_info:
            loader.load("tests/fixtures/invalid_model.bin")
            
        assert "Unsupported model format" in str(exc_info.value)
        
    @patch('tiny_llm_profiler.model_loader.os.path.getsize')
    def test_model_size_calculation(self, mock_getsize):
        """Test model size calculation with mocked file system."""
        mock_getsize.return_value = 1024 * 1024  # 1MB
        
        loader = ModelLoader()
        model = loader.load("fake_model.bin")
        
        assert model.size_mb == 1.0
```

### 2. Integration Tests

#### Platform Integration
```python
# Device communication
tests/integration/test_device_communication.py
tests/integration/test_firmware_flashing.py
tests/integration/test_real_time_monitoring.py

# End-to-end workflows
tests/integration/test_complete_profiling.py
tests/integration/test_benchmark_pipeline.py
tests/integration/test_multi_device_profiling.py
```

#### Test Configuration
```python
# tests/integration/conftest.py
import pytest
from tiny_llm_profiler import EdgeProfiler

@pytest.fixture(scope="session")
def available_devices():
    """Detect available hardware devices for testing."""
    from tiny_llm_profiler import DeviceDetector
    
    detector = DeviceDetector()
    devices = detector.detect_all()
    
    if not devices:
        pytest.skip("No hardware devices available for integration testing")
    
    return devices

@pytest.fixture
def esp32_profiler(available_devices):
    """Create profiler for ESP32 if available."""
    esp32_devices = [d for d in available_devices if d.platform == "esp32"]
    
    if not esp32_devices:
        pytest.skip("No ESP32 device available")
    
    return EdgeProfiler(
        platform="esp32",
        device=esp32_devices[0].port
    )
```

#### Integration Test Examples
```python
class TestDeviceCommunication:
    def test_device_detection(self, available_devices):
        """Test that devices are properly detected."""
        assert len(available_devices) > 0
        
        for device in available_devices:
            assert device.platform in SUPPORTED_PLATFORMS
            assert device.port is not None
            assert device.is_responsive()
    
    def test_firmware_flashing(self, esp32_profiler):
        """Test firmware flashing process."""
        result = esp32_profiler.flash_firmware()
        
        assert result.success
        assert result.flash_time_seconds < 30
        assert esp32_profiler.device.is_responsive()
    
    def test_model_profiling_e2e(self, esp32_profiler):
        """Test complete model profiling workflow."""
        from tiny_llm_profiler import QuantizedModel
        
        # Load test model
        model = QuantizedModel.from_file("tests/fixtures/tinyllama_2bit.bin")
        
        # Profile model
        results = esp32_profiler.profile_model(
            model=model,
            test_prompts=["Hello", "Test"],
            duration_seconds=30
        )
        
        # Validate results
        assert results.tokens_per_second > 0
        assert results.peak_memory_kb > 0
        assert results.average_latency_ms > 0
        assert len(results.per_token_metrics) > 0
```

### 3. Hardware-in-the-Loop (HIL) Tests

#### HIL Test Structure
```python
# tests/hardware/test_esp32_integration.py
import pytest
import time
from tiny_llm_profiler import EdgeProfiler, PowerProfiler

@pytest.mark.hardware
@pytest.mark.esp32
class TestESP32HardwareIntegration:
    @pytest.fixture(scope="class")
    def esp32_setup(self):
        """Setup ESP32 device for testing."""
        profiler = EdgeProfiler(platform="esp32", device="/dev/ttyUSB0")
        
        # Flash test firmware
        flash_result = profiler.flash_firmware(
            firmware_path="tests/fixtures/test_firmware.bin"
        )
        assert flash_result.success
        
        # Wait for device ready
        time.sleep(2)
        
        yield profiler
        
        # Cleanup
        profiler.reset_device()
    
    def test_memory_stress_test(self, esp32_setup):
        """Test device under memory stress conditions."""
        profiler = esp32_setup
        
        # Load large model near memory limit
        large_model = QuantizedModel.from_file(
            "tests/fixtures/large_model_4mb.bin"
        )
        
        results = profiler.profile_model(
            model=large_model,
            test_prompts=["Long test prompt"] * 100,
            duration_seconds=300  # 5 minute stress test
        )
        
        # Validate no memory leaks
        assert results.final_memory_kb <= results.peak_memory_kb * 1.1
        assert results.average_memory_kb < 400  # ESP32 limit
        
    def test_power_consumption_validation(self, esp32_setup):
        """Test power consumption measurements."""
        profiler = esp32_setup
        
        # Setup power measurement
        power_profiler = PowerProfiler(
            sensor="ina219",
            i2c_addr=0x40
        )
        
        model = QuantizedModel.from_file("tests/fixtures/test_model.bin")
        
        # Profile with power measurement
        with power_profiler.start_measurement():
            results = profiler.profile_model(
                model=model,
                test_prompts=["Test"] * 50
            )
        
        power_results = power_profiler.get_results()
        
        # Validate power consumption
        assert power_results.average_power_mw < 500  # Reasonable limit
        assert power_results.peak_power_mw < 800
        assert len(power_results.per_second_measurements) > 0
    
    def test_thermal_stability(self, esp32_setup):
        """Test device stability under thermal stress."""
        profiler = esp32_setup
        model = QuantizedModel.from_file("tests/fixtures/test_model.bin")
        
        # Run continuous profiling for thermal test
        start_time = time.time()
        thermal_results = []
        
        while time.time() - start_time < 1800:  # 30 minutes
            result = profiler.profile_model(
                model=model,
                test_prompts=["Thermal test"],
                metrics=["latency", "temperature"]
            )
            thermal_results.append(result)
            
            # Check for thermal throttling
            if result.cpu_temperature_c > 85:
                pytest.fail(f"Device overheating: {result.cpu_temperature_c}°C")
        
        # Validate thermal stability
        temps = [r.cpu_temperature_c for r in thermal_results]
        temp_variance = max(temps) - min(temps)
        assert temp_variance < 20  # Reasonable thermal stability
```

### 4. Performance Tests

#### Benchmark Suite
```python
# tests/performance/test_benchmarks.py
import pytest
from tiny_llm_profiler import StandardBenchmarks

@pytest.mark.performance
class TestPerformanceBenchmarks:
    def test_standard_benchmark_suite(self):
        """Run standard TinyML performance benchmarks."""
        benchmarks = StandardBenchmarks()
        
        results = benchmarks.run_tiny_ml_perf(
            models=["tinyllama-2bit", "phi-2-4bit"],
            platforms=["esp32", "stm32f7"],
            tasks=["text_generation", "summarization"]
        )
        
        # Validate benchmark results
        for result in results:
            assert result.tokens_per_second > 0
            assert result.memory_efficiency > 0
            assert result.energy_per_token_mj > 0
            
    def test_scaling_performance(self):
        """Test performance scaling across model sizes."""
        from tiny_llm_profiler import ScalingAnalyzer
        
        analyzer = ScalingAnalyzer()
        scaling_data = analyzer.analyze_scaling(
            model_sizes=[125e6, 350e6, 1.1e9],
            quantizations=[2, 4],
            platform="esp32"
        )
        
        # Validate scaling behavior
        latencies = [d.avg_latency_ms for d in scaling_data]
        assert latencies == sorted(latencies)  # Should increase with size
        
    @pytest.mark.slow
    def test_endurance_testing(self):
        """Long-duration endurance testing."""
        profiler = EdgeProfiler(platform="esp32", device="/dev/ttyUSB0")
        model = QuantizedModel.from_file("tests/fixtures/endurance_model.bin")
        
        # Run for 24 hours
        endurance_results = profiler.profile_endurance(
            model=model,
            duration_hours=24,
            sample_interval_minutes=5
        )
        
        # Validate system stability
        assert endurance_results.uptime_percentage > 99.5
        assert endurance_results.memory_leak_detected == False
        assert endurance_results.performance_degradation < 5  # Less than 5%
```

## Test Configuration

### Pytest Configuration
```ini
# pytest.ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra
    -q
    --strict-markers
    --strict-config
    --cov=src/tiny_llm_profiler
    --cov-report=html
    --cov-report=term-missing
    --cov-report=xml
    --cov-fail-under=90

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: Unit tests (fast, no hardware)
    integration: Integration tests (require hardware)
    hardware: Hardware-in-the-loop tests
    performance: Performance and benchmark tests
    slow: Slow tests (>30 seconds)
    esp32: ESP32-specific tests
    stm32: STM32-specific tests
    rp2040: RP2040-specific tests
    power: Power measurement tests
```

### Test Environment Setup
```bash
# tests/conftest.py environment setup
import os
import pytest
from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables and paths."""
    os.environ["TINY_PROFILER_TEST_MODE"] = "1"
    os.environ["TINY_PROFILER_LOG_LEVEL"] = "DEBUG"
    os.environ["TINY_PROFILER_TEST_DATA"] = str(TEST_DATA_DIR)
    
    # Ensure test data exists
    if not TEST_DATA_DIR.exists():
        pytest.skip("Test data directory not found")
    
    yield
    
    # Cleanup
    os.environ.pop("TINY_PROFILER_TEST_MODE", None)
```

## Test Data Management

### Test Fixtures
```python
# tests/fixtures/model_fixtures.py
import pytest
from tiny_llm_profiler import QuantizedModel

@pytest.fixture
def tinyllama_2bit():
    """Small 2-bit quantized TinyLLaMA model for testing."""
    return QuantizedModel.from_file("tests/fixtures/tinyllama_2bit.bin")

@pytest.fixture
def test_prompts():
    """Standard test prompts for consistency."""
    return [
        "Hello world",
        "What is artificial intelligence?",
        "Write a short story about robots",
        "Explain quantum computing",
        "1 + 1 = "
    ]

@pytest.fixture
def power_measurement_config():
    """Standard power measurement configuration."""
    return {
        "sensor": "ina219",
        "i2c_addr": 0x40,
        "shunt_ohms": 0.1,
        "sample_rate_hz": 1000
    }
```

## Continuous Integration

### Test Automation Pipeline
```yaml
# .github/workflows/test.yml (example structure)
name: Test Suite

on:
  pull_request:
  push:
    branches: [main, develop]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev,test]"
      
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src/tiny_llm_profiler
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: self-hosted  # Hardware runner
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      - name: Setup test hardware
        run: |
          # Setup hardware devices
          ./scripts/setup_test_hardware.sh
      
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v --device-timeout=300
```

## Coverage Requirements

### Coverage Targets
- **Overall Coverage**: >90%
- **Core Components**: >95%
- **Platform Adapters**: >85%
- **Hardware Interfaces**: >80%

### Coverage Configuration
```toml
# pyproject.toml
[tool.coverage.run]
source = ["src/tiny_llm_profiler"]
omit = [
    "*/tests/*",
    "*/fixtures/*",
    "*/vendor/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:"
]
```

## Test Maintenance

### Regular Tasks
1. **Weekly**: Update test fixtures with latest models
2. **Monthly**: Review and update hardware test suite
3. **Quarterly**: Performance benchmark baseline updates
4. **Release**: Full regression test suite execution

### Test Quality Metrics
- Test execution time trends
- Test flakiness detection
- Coverage trend analysis
- Hardware test success rates

---

This comprehensive testing strategy ensures reliable profiling across all supported edge platforms while maintaining development velocity and code quality.