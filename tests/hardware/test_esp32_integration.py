"""Hardware-in-the-loop tests for ESP32 integration."""

import pytest
import time
from pathlib import Path

from tiny_llm_profiler import EdgeProfiler, QuantizedModel


@pytest.mark.hardware
@pytest.mark.esp32
class TestESP32Integration:
    """Integration tests requiring actual ESP32 hardware."""
    
    @pytest.fixture(autouse=True)
    def setup_hardware(self, hardware_test_skip):
        """Setup hardware testing environment."""
        # This will skip if hardware is not available
        pass
    
    def test_esp32_device_discovery(self):
        """Test automatic ESP32 device discovery."""
        profiler = EdgeProfiler(platform="esp32")
        
        # Attempt to discover connected ESP32 devices
        devices = profiler.discover_devices()
        
        # Should find at least one ESP32 device for hardware tests
        assert len(devices) > 0
        assert any("ESP32" in device.name for device in devices)
    
    def test_esp32_firmware_flash(self):
        """Test flashing profiling firmware to ESP32."""
        profiler = EdgeProfiler(platform="esp32")
        devices = profiler.discover_devices()
        
        if not devices:
            pytest.skip("No ESP32 devices found")
        
        device = devices[0]
        result = profiler.flash_firmware(device.port)
        
        assert result.success is True
        assert "Firmware flashed successfully" in result.message
    
    def test_esp32_basic_communication(self):
        """Test basic communication with ESP32 device."""
        profiler = EdgeProfiler(platform="esp32")
        
        with profiler:
            # Test basic command response
            response = profiler.send_command("PING")
            assert response == "PONG"
            
            # Test device info query
            info = profiler.get_device_info()
            assert "ESP32" in info.chip_model
            assert info.free_heap > 0
    
    def test_esp32_model_profiling(self, sample_model_path):
        """Test actual model profiling on ESP32."""
        # Load a tiny test model
        model = QuantizedModel.from_file(
            sample_model_path,
            quantization=4,  # Use 4-bit for compatibility
            vocab_size=1000   # Small vocab for testing
        )
        
        profiler = EdgeProfiler(platform="esp32")
        
        with profiler:
            # Profile the model with short test prompts
            results = profiler.profile_model(
                model=model,
                test_prompts=["Hi", "Test"],
                iterations=3
            )
            
            # Verify we got meaningful results
            assert results.tokens_per_second > 0
            assert results.first_token_latency_ms > 0
            assert results.peak_memory_kb > 0
            assert results.peak_memory_kb < 520  # ESP32 RAM limit
    
    def test_esp32_power_profiling(self, sample_model_path):
        """Test power profiling on ESP32 (requires power sensor)."""
        try:
            from tiny_llm_profiler import PowerProfiler
        except ImportError:
            pytest.skip("Power profiling dependencies not available")
        
        model = QuantizedModel.from_file(sample_model_path, quantization=4)
        
        profiler = EdgeProfiler(platform="esp32")
        power_profiler = PowerProfiler(sensor="ina219", i2c_addr=0x40)
        
        with profiler, power_profiler:
            # Start power monitoring
            power_profiler.start_monitoring()
            
            # Run inference
            results = profiler.profile_model(
                model=model,
                test_prompts=["Power test"],
                iterations=1
            )
            
            # Stop monitoring and get power data
            power_data = power_profiler.stop_monitoring()
            
            # Verify power measurements
            assert power_data.average_power_mw > 50   # ESP32 should use >50mW
            assert power_data.average_power_mw < 1000 # But less than 1W
            assert power_data.energy_per_token_mj > 0
    
    def test_esp32_memory_stress(self):
        """Test ESP32 behavior under memory stress."""
        profiler = EdgeProfiler(platform="esp32")
        
        with profiler:
            # Get initial memory state
            initial_memory = profiler.get_memory_info()
            
            # Allocate memory progressively
            memory_chunks = []
            for chunk_size in [50, 100, 150, 200]:  # KB
                try:
                    chunk = profiler.allocate_memory(chunk_size * 1024)
                    memory_chunks.append(chunk)
                    
                    current_memory = profiler.get_memory_info()
                    assert current_memory.free_heap < initial_memory.free_heap
                    
                except MemoryError:
                    # Expected when we run out of memory
                    break
            
            # Clean up allocated memory
            for chunk in memory_chunks:
                profiler.free_memory(chunk)
    
    def test_esp32_concurrent_operations(self):
        """Test ESP32 handling of concurrent operations."""
        profiler = EdgeProfiler(platform="esp32")
        
        with profiler:
            # Start background memory monitoring
            monitor_task = profiler.start_memory_monitoring(interval_ms=100)
            
            # Perform model operations while monitoring
            time.sleep(2.0)
            
            # Send commands during monitoring
            for i in range(10):
                response = profiler.send_command(f"ECHO {i}")
                assert response == f"ECHO {i}"
                time.sleep(0.1)
            
            # Stop monitoring
            monitor_data = profiler.stop_memory_monitoring(monitor_task)
            
            # Verify we collected monitoring data
            assert len(monitor_data.samples) > 10
            assert all(sample.timestamp > 0 for sample in monitor_data.samples)
    
    def test_esp32_error_recovery(self):
        """Test ESP32 error recovery mechanisms."""
        profiler = EdgeProfiler(platform="esp32")
        
        with profiler:
            # Send invalid command to trigger error
            with pytest.raises(Exception):
                profiler.send_command("INVALID_COMMAND_XYZ")
            
            # Verify device recovers and can handle normal commands
            response = profiler.send_command("PING")
            assert response == "PONG"
            
            # Test recovery from communication timeout
            profiler._simulate_communication_timeout()
            
            # Device should reconnect automatically
            response = profiler.send_command("PING")
            assert response == "PONG"
    
    @pytest.mark.slow
    def test_esp32_long_running_profiling(self, sample_model_path):
        """Test long-running profiling session on ESP32."""
        model = QuantizedModel.from_file(sample_model_path, quantization=4)
        profiler = EdgeProfiler(platform="esp32")
        
        test_prompts = [
            "Hello world",
            "How are you today?",
            "Tell me about embedded systems",
            "What is artificial intelligence?",
            "Explain machine learning concepts"
        ]
        
        with profiler:
            results = profiler.profile_model(
                model=model,
                test_prompts=test_prompts,
                iterations=20,  # Long test
                collect_detailed_metrics=True
            )
            
            # Verify consistency over long run
            assert len(results.detailed_metrics) == 100  # 20 iterations * 5 prompts
            
            # Check for memory leaks (memory usage should be stable)
            memory_usage = [m.memory_kb for m in results.detailed_metrics]
            memory_variance = max(memory_usage) - min(memory_usage)
            assert memory_variance < 50  # Less than 50KB variance suggests no major leaks
            
            # Performance should be consistent
            latencies = [m.latency_ms for m in results.detailed_metrics]
            latency_variance = max(latencies) - min(latencies)
            assert latency_variance < results.first_token_latency_ms  # Reasonable variance