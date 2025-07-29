"""Unit tests for EdgeProfiler class."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

import pytest
from tiny_llm_profiler.profiler import EdgeProfiler
from tiny_llm_profiler.models import QuantizedModel
from tiny_llm_profiler.exceptions import PlatformError, ModelError


class TestEdgeProfiler:
    """Test cases for EdgeProfiler class."""
    
    def test_profiler_initialization(self, esp32_platform_config):
        """Test EdgeProfiler initialization with valid configuration."""
        profiler = EdgeProfiler(
            platform="esp32",
            device="/dev/ttyUSB0",
            baudrate=921600
        )
        
        assert profiler.platform == "esp32"
        assert profiler.device == "/dev/ttyUSB0"
        assert profiler.baudrate == 921600
    
    def test_profiler_invalid_platform(self):
        """Test EdgeProfiler initialization with invalid platform."""
        with pytest.raises(PlatformError):
            EdgeProfiler(platform="invalid_platform")
    
    @patch('tiny_llm_profiler.profiler.serial.Serial')
    def test_device_connection(self, mock_serial, mock_serial_device):
        """Test device connection establishment."""
        mock_serial.return_value = mock_serial_device
        
        profiler = EdgeProfiler(platform="esp32")
        profiler.connect()
        
        assert profiler.is_connected()
        mock_serial.assert_called_once()
    
    def test_model_validation(self, mock_quantized_model):
        """Test model validation before profiling."""
        profiler = EdgeProfiler(platform="esp32")
        
        # Valid model should pass
        assert profiler._validate_model(mock_quantized_model) is True
        
        # Invalid model should fail
        mock_quantized_model.size_bytes = 10_000_000  # Too large
        with pytest.raises(ModelError):
            profiler._validate_model(mock_quantized_model)
    
    @pytest.mark.asyncio
    async def test_async_profiling(self, mock_quantized_model, mock_async_serial):
        """Test asynchronous profiling workflow."""
        profiler = EdgeProfiler(platform="esp32")
        
        with patch.object(profiler, '_get_serial_connection', return_value=mock_async_serial):
            results = await profiler.profile_model_async(
                model=mock_quantized_model,
                test_prompts=["Hello", "Test"]
            )
            
            assert results is not None
            assert hasattr(results, 'tokens_per_second')
    
    def test_metrics_collection(self, sample_profiling_results):
        """Test metrics collection and parsing."""
        profiler = EdgeProfiler(platform="esp32")
        
        parsed_metrics = profiler._parse_metrics(sample_profiling_results)
        
        assert 'first_token_latency_ms' in parsed_metrics
        assert 'tokens_per_second' in parsed_metrics
        assert parsed_metrics['tokens_per_second'] > 0
    
    def test_error_handling_device_not_found(self):
        """Test error handling when device is not found."""
        profiler = EdgeProfiler(platform="esp32", device="/dev/nonexistent")
        
        with pytest.raises(PlatformError, match="Device not found"):
            profiler.connect()
    
    @patch('tiny_llm_profiler.profiler.time.time')
    def test_timeout_handling(self, mock_time):
        """Test timeout handling during profiling."""
        mock_time.side_effect = [0, 0, 0, 1000]  # Simulate timeout
        
        profiler = EdgeProfiler(platform="esp32", timeout=5)
        
        with pytest.raises(TimeoutError):
            profiler._wait_for_response("READY", timeout=5)
    
    def test_power_measurement_integration(self, mock_power_sensor):
        """Test integration with power measurement sensors."""
        profiler = EdgeProfiler(platform="esp32")
        profiler.power_sensor = mock_power_sensor
        
        power_data = profiler._collect_power_metrics(duration=1.0)
        
        assert 'voltage' in power_data
        assert 'current_ma' in power_data
        assert 'power_mw' in power_data
    
    def test_cleanup_on_error(self, mock_serial_device):
        """Test proper cleanup when errors occur."""
        profiler = EdgeProfiler(platform="esp32")
        profiler.connection = mock_serial_device
        
        try:
            profiler._simulate_error()
        except Exception:
            pass
        
        # Cleanup should have been called
        mock_serial_device.close.assert_called()
    
    def test_concurrent_profiling_prevention(self):
        """Test prevention of concurrent profiling operations."""
        profiler = EdgeProfiler(platform="esp32")
        profiler._profiling_active = True
        
        with pytest.raises(RuntimeError, match="Profiling already in progress"):
            profiler.profile_model(Mock())
    
    @pytest.mark.parametrize("platform,expected_config", [
        ("esp32", {"baudrate": 921600, "timeout": 10}),
        ("stm32f4", {"baudrate": 115200, "timeout": 5}),
        ("rp2040", {"baudrate": 115200, "timeout": 3}),
    ])
    def test_platform_specific_configs(self, platform, expected_config):
        """Test platform-specific configuration handling."""
        profiler = EdgeProfiler(platform=platform)
        
        assert profiler.baudrate == expected_config["baudrate"]
        assert profiler.timeout == expected_config["timeout"]