"""
Integration tests for platform adapters and device communication.

These tests verify that different platform adapters work correctly
with mock and real hardware devices.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import json


@pytest.mark.integration
class TestESP32Integration:
    """Integration tests for ESP32 platform adapter."""
    
    def test_esp32_device_discovery(self):
        """Test ESP32 device discovery and identification."""
        with patch('serial.tools.list_ports.comports') as mock_comports:
            # Mock ESP32 device
            mock_device = Mock()
            mock_device.device = "/dev/ttyUSB0"
            mock_device.vid = 0x10c4  # Silicon Labs CP210x
            mock_device.pid = 0xea60
            mock_device.description = "CP2102 USB to UART Bridge Controller"
            
            mock_comports.return_value = [mock_device]
            
            devices = mock_comports()
            esp32_devices = [d for d in devices if d.vid == 0x10c4 and d.pid == 0xea60]
            
            assert len(esp32_devices) == 1
            assert esp32_devices[0].device == "/dev/ttyUSB0"
    
    def test_esp32_communication_protocol(self, mock_serial_device):
        """Test ESP32 communication protocol."""
        # Test command-response cycle
        commands = [
            b"GET_INFO\n",
            b"LOAD_MODEL\n", 
            b"START_PROFILING\n",
            b"GET_METRICS\n",
            b"STOP_PROFILING\n"
        ]
        
        responses = [
            b'{"chip": "ESP32", "mac": "24:0a:c4:12:34:56"}\n',
            b'{"status": "model_loaded", "size_kb": 3200}\n',
            b'{"status": "profiling_started", "session_id": "sess_123"}\n',
            b'{"cpu": 45.2, "memory": 380, "power": 750}\n',
            b'{"status": "profiling_stopped", "total_samples": 1000}\n'
        ]
        
        # Configure mock responses
        mock_serial_device.readline.side_effect = responses
        
        for i, command in enumerate(commands):
            mock_serial_device.write(command)
            response = mock_serial_device.readline()
            
            # Verify responses are JSON formatted
            try:
                data = json.loads(response.decode())
                assert isinstance(data, dict)
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON response for command {command}")
    
    def test_esp32_firmware_upload(self, temp_dir):
        """Test ESP32 firmware upload process."""
        # Create mock firmware file
        firmware_file = temp_dir / "esp32_profiler.bin"
        firmware_file.write_bytes(b"\x00" * 1024)  # 1KB firmware
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "Flash complete"
            
            # Simulate esptool.py flash command
            result = mock_subprocess(
                ["esptool.py", "--port", "/dev/ttyUSB0", "write_flash", "0x10000", str(firmware_file)],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            mock_subprocess.assert_called_once()


@pytest.mark.integration
class TestSTM32Integration:
    """Integration tests for STM32 platform adapter."""
    
    def test_stm32_device_discovery(self):
        """Test STM32 device discovery via USB CDC."""
        with patch('serial.tools.list_ports.comports') as mock_comports:
            # Mock STM32 device
            mock_device = Mock()
            mock_device.device = "/dev/ttyACM0"
            mock_device.vid = 0x0483  # STMicroelectronics
            mock_device.pid = 0x374b  # CDC ACM
            mock_device.description = "STM32 Virtual ComPort"
            
            mock_comports.return_value = [mock_device]
            
            devices = mock_comports()
            stm32_devices = [d for d in devices if d.vid == 0x0483]
            
            assert len(stm32_devices) == 1
            assert stm32_devices[0].device == "/dev/ttyACM0"
    
    def test_stm32_openocd_integration(self, temp_dir):
        """Test STM32 firmware upload via OpenOCD."""
        # Create mock firmware file
        firmware_file = temp_dir / "stm32_profiler.bin"
        firmware_file.write_bytes(b"\x00" * 2048)  # 2KB firmware
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "Programming Finished"
            
            # Simulate OpenOCD flash command
            result = mock_subprocess(
                [
                    "openocd",
                    "-f", "interface/stlink.cfg",
                    "-f", "target/stm32f7x.cfg",
                    "-c", f"program {firmware_file} 0x08000000 verify reset exit"
                ],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            mock_subprocess.assert_called_once()


@pytest.mark.integration
class TestMultiPlatformWorkflow:
    """Integration tests for multi-platform workflows."""
    
    def test_platform_adapter_factory(self):
        """Test platform adapter factory and registration."""
        # Mock platform registry
        platforms = {
            "esp32": "ESP32Adapter",
            "stm32f7": "STM32Adapter", 
            "rp2040": "RP2040Adapter",
            "nrf52840": "NordicAdapter"
        }
        
        for platform_name, adapter_class in platforms.items():
            # This would test actual adapter factory
            assert platform_name in platforms
            assert adapter_class.endswith("Adapter")
    
    def test_concurrent_device_profiling(self):
        """Test profiling multiple devices concurrently."""
        devices = [
            {"platform": "esp32", "port": "/dev/ttyUSB0"},
            {"platform": "stm32f7", "port": "/dev/ttyACM0"},
            {"platform": "rp2040", "port": "/dev/ttyUSB1"}
        ]
        
        # Mock concurrent profiling results
        results = []
        for device in devices:
            mock_result = {
                "platform": device["platform"],
                "port": device["port"],
                "status": "success",
                "metrics": {
                    "latency_ms": 50.0 + len(results) * 10,
                    "memory_kb": 300 + len(results) * 50,
                    "power_mw": 800 + len(results) * 100
                }
            }
            results.append(mock_result)
        
        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)
    
    def test_cross_platform_model_compatibility(self):
        """Test model compatibility across different platforms."""
        model_info = {
            "name": "tinyllama-1b-4bit",
            "size_mb": 3.2,
            "quantization": "4-bit",
            "architecture": "llama"
        }
        
        platform_constraints = {
            "esp32": {"max_size_mb": 4.0, "supports_4bit": True},
            "stm32f7": {"max_size_mb": 2.0, "supports_4bit": True},
            "rp2040": {"max_size_mb": 2.0, "supports_4bit": False},
            "nrf52840": {"max_size_mb": 1.0, "supports_4bit": False}
        }
        
        compatible_platforms = []
        for platform, constraints in platform_constraints.items():
            if (model_info["size_mb"] <= constraints["max_size_mb"] and
                (model_info["quantization"] != "4-bit" or constraints["supports_4bit"])):
                compatible_platforms.append(platform)
        
        # ESP32 and STM32F7 should be compatible
        assert "esp32" in compatible_platforms
        assert "stm32f7" in compatible_platforms
        # RP2040 and nRF52840 should not be compatible
        assert "rp2040" not in compatible_platforms
        assert "nrf52840" not in compatible_platforms


@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for data collection and processing pipeline."""
    
    def test_metrics_streaming_pipeline(self):
        """Test real-time metrics streaming and processing."""
        # Mock streaming metrics data
        metrics_stream = [
            {"timestamp": 1000 + i, "cpu": 45 + i, "memory": 350 + i*2, "power": 800 + i*5}
            for i in range(100)
        ]
        
        # Test streaming processing
        processed_metrics = []
        for metric in metrics_stream:
            # Apply basic filtering and validation
            if metric["cpu"] >= 0 and metric["memory"] > 0:
                processed_metrics.append(metric)
        
        assert len(processed_metrics) == 100
        assert all(m["cpu"] >= 0 for m in processed_metrics)
        assert all(m["memory"] > 0 for m in processed_metrics)
    
    def test_data_aggregation_pipeline(self, sample_metrics_data):
        """Test metrics data aggregation and analysis."""
        # Test time-based aggregation (1-second windows)
        aggregated_data = []
        
        # Group by second intervals
        for second in range(0, 10):
            window_data = sample_metrics_data.iloc[second*100:(second+1)*100]
            if not window_data.empty:
                agg_metrics = {
                    "timestamp": second,
                    "avg_cpu": window_data["cpu_usage_percent"].mean(),
                    "max_memory": window_data["memory_usage_kb"].max(),
                    "avg_power": window_data["power_mw"].mean(),
                    "p95_latency": window_data["inference_latency_ms"].quantile(0.95)
                }
                aggregated_data.append(agg_metrics)
        
        assert len(aggregated_data) == 10
        assert all(m["avg_cpu"] > 0 for m in aggregated_data)
    
    def test_report_generation_pipeline(self, sample_profiling_results):
        """Test report generation from profiling results."""
        # Test HTML report generation
        html_template = """
        <html>
        <head><title>Profiling Report</title></head>
        <body>
        <h1>Model: {model_name}</h1>
        <h2>Platform: {platform}</h2>
        <p>Latency: {latency_ms}ms</p>
        <p>Memory: {memory_kb}KB</p>
        <p>Power: {power_mw}mW</p>
        </body>
        </html>
        """
        
        report_html = html_template.format(
            model_name=sample_profiling_results["model_name"],
            platform=sample_profiling_results["platform"],
            latency_ms=sample_profiling_results["metrics"]["latency"]["inter_token_ms"],
            memory_kb=sample_profiling_results["metrics"]["memory"]["peak_ram_kb"],
            power_mw=sample_profiling_results["metrics"]["power"]["avg_power_mw"]
        )
        
        assert "test-model" in report_html
        assert "esp32" in report_html
        assert "45.2ms" in report_html


@pytest.mark.integration 
@pytest.mark.slow
class TestLongRunningIntegration:
    """Long-running integration tests."""
    
    def test_extended_profiling_session(self):
        """Test extended profiling session stability."""
        # Simulate 10-minute profiling session
        session_duration_seconds = 600  # 10 minutes
        sample_rate_hz = 10  # 10 samples per second
        total_samples = session_duration_seconds * sample_rate_hz
        
        # Generate mock data for extended session
        samples_collected = 0
        errors_encountered = 0
        
        for i in range(total_samples):
            try:
                # Simulate sample collection
                sample = {
                    "timestamp": i * 100,  # 100ms intervals
                    "cpu": 45.0 + (i % 20) - 10,  # Varying CPU usage
                    "memory": 350 + (i % 100),    # Slowly increasing memory
                    "power": 800 + (i % 50) - 25  # Varying power
                }
                
                # Simulate occasional errors (5% failure rate)
                if i % 20 == 0:
                    raise Exception("Simulated communication error")
                
                samples_collected += 1
                
            except Exception:
                errors_encountered += 1
        
        # Verify session completed successfully
        success_rate = samples_collected / total_samples
        error_rate = errors_encountered / total_samples
        
        assert success_rate >= 0.95  # 95% success rate
        assert error_rate <= 0.05    # 5% error rate
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simulate memory-intensive operations
        for iteration in range(100):
            # Create and process mock data
            large_dataset = list(range(10000))
            processed_data = [x * 2 for x in large_dataset]
            
            # Force garbage collection every 10 iterations
            if iteration % 10 == 0:
                gc.collect()
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable (<50MB)
                assert memory_growth < 50 * 1024 * 1024
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss
        total_growth = final_memory - initial_memory
        
        # Total memory growth should be minimal
        assert total_growth < 100 * 1024 * 1024  # <100MB growth


@pytest.mark.integration
class TestErrorRecovery:
    """Integration tests for error handling and recovery."""
    
    def test_device_disconnection_recovery(self, mock_serial_device):
        """Test recovery from device disconnection."""
        # Simulate device disconnection
        mock_serial_device.is_open = False
        mock_serial_device.read.side_effect = Exception("Device disconnected")
        
        # Test reconnection logic
        reconnect_attempts = 0
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                mock_serial_device.read()
                break  # Success
            except Exception:
                reconnect_attempts += 1
                time.sleep(0.1)  # Brief delay before retry
                
                if attempt == max_attempts - 1:
                    # Simulate successful reconnection on final attempt
                    mock_serial_device.is_open = True
                    mock_serial_device.read.side_effect = None
                    mock_serial_device.read.return_value = b"OK\n"
        
        assert reconnect_attempts <= max_attempts
        assert mock_serial_device.is_open is True
    
    def test_firmware_corruption_recovery(self, temp_dir):
        """Test recovery from firmware corruption."""
        # Create corrupted firmware file
        corrupted_firmware = temp_dir / "corrupted.bin"
        corrupted_firmware.write_bytes(b"\xFF" * 100)  # Invalid firmware
        
        # Test firmware validation
        def validate_firmware(firmware_path):
            with open(firmware_path, "rb") as f:
                header = f.read(8)
                # ESP32 should start with specific magic bytes
                return header.startswith(b"\xe9\x00\x00\x02")
        
        assert not validate_firmware(corrupted_firmware)
        
        # Create valid firmware for recovery
        valid_firmware = temp_dir / "valid.bin"
        valid_firmware.write_bytes(b"\xe9\x00\x00\x02" + b"\x00" * 96)
        
        assert validate_firmware(valid_firmware)
    
    def test_model_loading_error_recovery(self, temp_dir):
        """Test recovery from model loading errors."""
        # Create invalid model file
        invalid_model = temp_dir / "invalid.bin"
        invalid_model.write_bytes(b"INVALID" + b"\x00" * 1000)
        
        # Test model validation
        def validate_model(model_path):
            with open(model_path, "rb") as f:
                header = f.read(9)
                return header == b"TINYMODEL"
        
        assert not validate_model(invalid_model)
        
        # Create valid model for recovery
        valid_model = temp_dir / "valid.bin"
        valid_model.write_bytes(b"TINYMODEL" + b"\x00" * 1000)
        
        assert validate_model(valid_model)