"""
Core EdgeProfiler implementation for measuring LLM performance on edge devices.
"""

import time
import asyncio
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import serial
import psutil
from pydantic import BaseModel, ConfigDict

from .models import QuantizedModel
from .results import ProfileResults, LatencyProfile, MemoryProfile, PowerProfile
from .platforms import PlatformManager


@dataclass
class ProfilingConfig:
    """Configuration for profiling sessions."""
    sample_rate_hz: int = 100
    duration_seconds: int = 60
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    timeout_seconds: int = 300
    enable_power_profiling: bool = False
    enable_memory_profiling: bool = True
    enable_latency_profiling: bool = True


class EdgeProfiler:
    """
    Main profiler class for measuring LLM performance on edge devices.
    
    Supports multiple platforms including ESP32, STM32, RISC-V, and ARM Cortex-M.
    """
    
    def __init__(
        self,
        platform: str,
        device: Optional[str] = None,
        baudrate: int = 921600,
        connection: str = "serial",
        **kwargs
    ):
        """
        Initialize EdgeProfiler for a specific platform.
        
        Args:
            platform: Target platform (esp32, stm32f4, stm32f7, rp2040, etc.)
            device: Device path (/dev/ttyUSB0, COM3, etc.)
            baudrate: Serial communication baudrate
            connection: Connection type (serial, network, local)
        """
        self.platform = platform
        self.device = device
        self.baudrate = baudrate
        self.connection = connection
        
        self.platform_manager = PlatformManager(platform)
        self.serial_conn: Optional[serial.Serial] = None
        self.is_connected = False
        
        # Initialize platform-specific settings
        self.platform_config = self.platform_manager.get_config()
        
    def connect(self) -> bool:
        """Establish connection to the target device."""
        try:
            if self.connection == "serial" and self.device:
                self.serial_conn = serial.Serial(
                    self.device,
                    self.baudrate,
                    timeout=5.0
                )
                self.is_connected = True
                return True
            elif self.connection == "local":
                # For local profiling (e.g., Raspberry Pi)
                self.is_connected = True
                return True
            else:
                raise ValueError(f"Unsupported connection type: {self.connection}")
                
        except Exception as e:
            print(f"Failed to connect to device: {e}")
            return False
    
    def disconnect(self):
        """Close connection to the target device."""
        if self.serial_conn:
            self.serial_conn.close()
        self.is_connected = False
    
    def profile_model(
        self,
        model: QuantizedModel,
        test_prompts: List[str],
        metrics: List[str] = None,
        config: Optional[ProfilingConfig] = None
    ) -> ProfileResults:
        """
        Profile a quantized model on the target platform.
        
        Args:
            model: QuantizedModel to profile
            test_prompts: List of test prompts for inference
            metrics: Metrics to collect (latency, memory, power)
            config: Profiling configuration
            
        Returns:
            ProfileResults containing all measured metrics
        """
        if not self.is_connected:
            if not self.connect():
                raise RuntimeError("Failed to connect to device")
        
        config = config or ProfilingConfig()
        metrics = metrics or ["latency", "memory"]
        
        # Initialize results structure
        results = ProfileResults(
            platform=self.platform,
            model_name=model.name,
            model_size_mb=model.size_mb,
            quantization=model.quantization
        )
        
        # Run profiling based on requested metrics
        if "latency" in metrics:
            latency_profile = self._profile_latency(model, test_prompts, config)
            results.add_latency_profile(latency_profile)
        
        if "memory" in metrics:
            memory_profile = self._profile_memory(model, test_prompts, config)
            results.add_memory_profile(memory_profile)
            
        if "power" in metrics and config.enable_power_profiling:
            power_profile = self._profile_power(model, test_prompts, config)
            results.add_power_profile(power_profile)
        
        return results
    
    def _profile_latency(
        self,
        model: QuantizedModel,
        test_prompts: List[str],
        config: ProfilingConfig
    ) -> LatencyProfile:
        """Profile inference latency for the model."""
        latencies = []
        first_token_latencies = []
        inter_token_latencies = []
        
        # Warmup runs
        for _ in range(config.warmup_iterations):
            self._run_inference(model, test_prompts[0])
        
        # Measurement runs
        for prompt in test_prompts:
            for _ in range(config.measurement_iterations):
                start_time = time.perf_counter()
                
                # Measure first token latency
                first_token_time = self._measure_first_token(model, prompt)
                first_token_latencies.append(first_token_time)
                
                # Complete inference and measure total time
                total_tokens = self._run_inference(model, prompt)
                end_time = time.perf_counter()
                
                total_latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(total_latency)
                
                # Calculate inter-token latency
                if total_tokens > 1:
                    inter_token_latency = (total_latency - first_token_time) / (total_tokens - 1)
                    inter_token_latencies.append(inter_token_latency)
        
        return LatencyProfile(
            first_token_latency_ms=np.mean(first_token_latencies),
            inter_token_latency_ms=np.mean(inter_token_latencies),
            total_latency_ms=np.mean(latencies),
            tokens_per_second=1000.0 / np.mean(inter_token_latencies) if inter_token_latencies else 0.0,
            latency_std_ms=np.std(latencies)
        )
    
    def _profile_memory(
        self,
        model: QuantizedModel,
        test_prompts: List[str],
        config: ProfilingConfig
    ) -> MemoryProfile:
        """Profile memory usage during inference."""
        if self.connection == "local":
            return self._profile_memory_local(model, test_prompts, config)
        else:
            return self._profile_memory_remote(model, test_prompts, config)
    
    def _profile_memory_local(
        self,
        model: QuantizedModel,
        test_prompts: List[str],
        config: ProfilingConfig
    ) -> MemoryProfile:
        """Profile memory usage on local system."""
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024  # Convert to KB
        
        peak_memory = baseline_memory
        memory_samples = []
        
        # Monitor memory during inference
        for prompt in test_prompts[:3]:  # Sample subset for memory profiling
            start_memory = process.memory_info().rss / 1024
            
            # Run inference while monitoring memory
            self._run_inference(model, prompt)
            
            current_memory = process.memory_info().rss / 1024
            memory_samples.append(current_memory - baseline_memory)
            peak_memory = max(peak_memory, current_memory)
        
        return MemoryProfile(
            baseline_memory_kb=baseline_memory,
            peak_memory_kb=peak_memory,
            memory_usage_kb=np.mean(memory_samples),
            memory_efficiency_tokens_per_kb=0.0  # Calculate based on throughput
        )
    
    def _profile_memory_remote(
        self,
        model: QuantizedModel,
        test_prompts: List[str],
        config: ProfilingConfig
    ) -> MemoryProfile:
        """Profile memory usage on remote device via serial communication."""
        # Send memory profiling commands to device
        memory_readings = []
        
        for prompt in test_prompts[:3]:
            # Request memory status before inference
            baseline = self._get_device_memory()
            
            # Run inference
            self._run_inference(model, prompt)
            
            # Request memory status after inference
            peak = self._get_device_memory()
            memory_readings.append(peak - baseline)
        
        return MemoryProfile(
            baseline_memory_kb=baseline,
            peak_memory_kb=max(memory_readings) + baseline,
            memory_usage_kb=np.mean(memory_readings),
            memory_efficiency_tokens_per_kb=0.0
        )
    
    def _profile_power(
        self,
        model: QuantizedModel,
        test_prompts: List[str],
        config: ProfilingConfig
    ) -> PowerProfile:
        """Profile power consumption during inference."""
        # Placeholder for power profiling implementation
        # Would integrate with hardware power measurement tools
        return PowerProfile(
            idle_power_mw=50.0,
            active_power_mw=150.0,
            peak_power_mw=200.0,
            energy_per_token_mj=2.5,
            total_energy_mj=100.0
        )
    
    def _run_inference(self, model: QuantizedModel, prompt: str) -> int:
        """Run inference and return number of generated tokens."""
        if self.connection == "local":
            # Simulate local inference
            time.sleep(0.05)  # Simulate computation time
            return len(prompt.split()) + 10  # Simulate token generation
        
        elif self.serial_conn:
            # Send inference command to remote device
            command = f"INFER:{prompt}\n"
            self.serial_conn.write(command.encode())
            
            # Read response
            response = self.serial_conn.readline().decode().strip()
            
            # Parse response for token count
            if response.startswith("TOKENS:"):
                return int(response.split(":")[1])
            
        return 0
    
    def _measure_first_token(self, model: QuantizedModel, prompt: str) -> float:
        """Measure time to first token generation."""
        start_time = time.perf_counter()
        
        # Implementation depends on platform
        if self.connection == "local":
            time.sleep(0.02)  # Simulate first token delay
        elif self.serial_conn:
            # Send command and wait for first token
            command = f"FIRST_TOKEN:{prompt}\n"
            self.serial_conn.write(command.encode())
            self.serial_conn.readline()  # Wait for response
        
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # Convert to ms
    
    def _get_device_memory(self) -> float:
        """Get current memory usage from remote device."""
        if self.serial_conn:
            self.serial_conn.write(b"MEM_STATUS\n")
            response = self.serial_conn.readline().decode().strip()
            
            if response.startswith("MEM:"):
                return float(response.split(":")[1])
        
        return 0.0
    
    def stream_metrics(self, duration_seconds: int = 60):
        """Stream real-time metrics from the device."""
        if not self.is_connected:
            raise RuntimeError("Device not connected")
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            if self.connection == "local":
                # Local metrics
                process = psutil.Process()
                yield {
                    "timestamp": time.time(),
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / (1024 * 1024),
                    "tokens_per_second": np.random.uniform(8, 12)  # Simulated
                }
            elif self.serial_conn:
                # Remote metrics
                self.serial_conn.write(b"METRICS\n")
                response = self.serial_conn.readline().decode().strip()
                
                if response.startswith("METRICS:"):
                    # Parse metrics from device
                    metrics_data = response.split(":")[1].split(",")
                    yield {
                        "timestamp": time.time(),
                        "cpu_percent": float(metrics_data[0]) if len(metrics_data) > 0 else 0,
                        "memory_kb": float(metrics_data[1]) if len(metrics_data) > 1 else 0,
                        "tokens_per_second": float(metrics_data[2]) if len(metrics_data) > 2 else 0
                    }
            
            time.sleep(0.1)  # 10Hz sampling rate
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()