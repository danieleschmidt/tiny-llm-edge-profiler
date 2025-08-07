"""
Power profiling and measurement tools for edge devices.
"""

import time
import asyncio
from typing import Dict, List, Optional, Iterator, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

from .results import PowerProfile


@dataclass
class PowerMeasurement:
    """Single power measurement sample."""
    timestamp: float
    voltage_v: float
    current_ma: float
    power_mw: float
    temperature_c: Optional[float] = None


@dataclass
class PowerConfig:
    """Configuration for power profiling."""
    sensor_type: str = "ina219"
    i2c_address: int = 0x40
    shunt_resistance_ohms: float = 0.1
    sample_rate_hz: int = 1000
    voltage_range_v: float = 16.0
    current_range_ma: float = 400.0
    calibration_factor: float = 1.0


class PowerSensor(ABC):
    """Abstract base class for power sensors."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the power sensor."""
        pass
    
    @abstractmethod
    def read_power(self) -> PowerMeasurement:
        """Read current power measurement."""
        pass
    
    @abstractmethod
    def set_sample_rate(self, sample_rate_hz: int):
        """Set the sampling rate."""
        pass
    
    @abstractmethod
    def close(self):
        """Close sensor connection."""
        pass


class INA219Sensor(PowerSensor):
    """INA219 current/power sensor implementation."""
    
    def __init__(self, config: PowerConfig):
        self.config = config
        self.sensor = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize INA219 sensor."""
        try:
            # Try to import and initialize INA219
            # This would normally require the actual hardware library
            # For now, we'll simulate the sensor
            self.sensor = self._create_simulated_sensor()
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize INA219 sensor: {e}")
            return False
    
    def read_power(self) -> PowerMeasurement:
        """Read power measurement from INA219."""
        if not self.is_initialized:
            raise RuntimeError("Sensor not initialized")
        
        # Simulate reading from actual sensor
        voltage = self._read_voltage()
        current = self._read_current()
        power = voltage * current
        
        return PowerMeasurement(
            timestamp=time.time(),
            voltage_v=voltage,
            current_ma=current,
            power_mw=power,
            temperature_c=self._read_temperature()
        )
    
    def set_sample_rate(self, sample_rate_hz: int):
        """Set sensor sampling rate."""
        self.config.sample_rate_hz = sample_rate_hz
        # Configure sensor hardware sampling rate
    
    def close(self):
        """Close sensor connection."""
        self.is_initialized = False
        self.sensor = None
    
    def _create_simulated_sensor(self):
        """Create simulated sensor for testing."""
        return {
            "voltage_base": 3.3,
            "current_base": 50,
            "noise_level": 0.1
        }
    
    def _read_voltage(self) -> float:
        """Read voltage from sensor."""
        if self.sensor:
            base_voltage = self.sensor["voltage_base"]
            noise = np.random.normal(0, self.sensor["noise_level"] * 0.1)
            return max(0, base_voltage + noise)
        return 3.3
    
    def _read_current(self) -> float:
        """Read current from sensor."""
        if self.sensor:
            # Simulate varying current based on activity
            base_current = self.sensor["current_base"]
            activity_variation = np.random.uniform(0.5, 2.0)  # 50% to 200% of base
            noise = np.random.normal(0, self.sensor["noise_level"] * 5)
            return max(0, base_current * activity_variation + noise)
        return 50.0
    
    def _read_temperature(self) -> Optional[float]:
        """Read temperature if available."""
        # Simulate temperature reading
        base_temp = 25.0
        variation = np.random.normal(0, 5)
        return base_temp + variation


class SimulatedPowerSensor(PowerSensor):
    """Simulated power sensor for testing."""
    
    def __init__(self, config: PowerConfig):
        self.config = config
        self.is_initialized = False
        self.base_power_mw = 100.0
        self.idle_power_mw = 50.0
    
    def initialize(self) -> bool:
        """Initialize simulated sensor."""
        self.is_initialized = True
        return True
    
    def read_power(self) -> PowerMeasurement:
        """Generate simulated power measurement."""
        if not self.is_initialized:
            raise RuntimeError("Sensor not initialized")
        
        # Simulate power consumption pattern
        current_time = time.time()
        
        # Add some periodic variation to simulate compute load
        load_factor = 1.0 + 0.5 * np.sin(current_time * 0.5)  # Slow oscillation
        noise = np.random.normal(1.0, 0.1)  # 10% noise
        
        power_mw = self.idle_power_mw + (self.base_power_mw - self.idle_power_mw) * load_factor * noise
        voltage_v = 3.3 + np.random.normal(0, 0.05)  # Voltage with small variation
        current_ma = power_mw / voltage_v
        
        return PowerMeasurement(
            timestamp=current_time,
            voltage_v=voltage_v,
            current_ma=current_ma,
            power_mw=power_mw,
            temperature_c=25.0 + np.random.normal(0, 2)
        )
    
    def set_sample_rate(self, sample_rate_hz: int):
        """Set sampling rate."""
        self.config.sample_rate_hz = sample_rate_hz
    
    def close(self):
        """Close sensor."""
        self.is_initialized = False


class PowerProfiler:
    """Main power profiling class."""
    
    def __init__(
        self,
        sensor: str = "simulated",
        i2c_addr: int = 0x40,
        shunt_ohms: float = 0.1,
        **kwargs
    ):
        """
        Initialize power profiler.
        
        Args:
            sensor: Sensor type ("ina219", "simulated")
            i2c_addr: I2C address for hardware sensors
            shunt_ohms: Shunt resistor value in ohms
        """
        self.config = PowerConfig(
            sensor_type=sensor,
            i2c_address=i2c_addr,
            shunt_resistance_ohms=shunt_ohms,
            **kwargs
        )
        
        self.sensor = self._create_sensor()
        self.measurements: List[PowerMeasurement] = []
        self.is_profiling = False
    
    def _create_sensor(self) -> PowerSensor:
        """Create appropriate sensor instance."""
        if self.config.sensor_type == "ina219":
            return INA219Sensor(self.config)
        elif self.config.sensor_type == "simulated":
            return SimulatedPowerSensor(self.config)
        else:
            raise ValueError(f"Unsupported sensor type: {self.config.sensor_type}")
    
    def initialize(self) -> bool:
        """Initialize the power profiler."""
        return self.sensor.initialize()
    
    def profile(self, duration_seconds: int, sample_rate_hz: int = 1000) -> PowerProfile:
        """
        Profile power consumption for specified duration.
        
        Args:
            duration_seconds: Duration to profile
            sample_rate_hz: Sampling rate in Hz
            
        Returns:
            PowerProfile with analysis results
        """
        if not self.sensor.initialize():
            raise RuntimeError("Failed to initialize power sensor")
        
        self.sensor.set_sample_rate(sample_rate_hz)
        self.measurements.clear()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        self.is_profiling = True
        
        try:
            while time.time() < end_time and self.is_profiling:
                measurement = self.sensor.read_power()
                self.measurements.append(measurement)
                
                # Sleep to achieve target sample rate
                sleep_time = 1.0 / sample_rate_hz
                time.sleep(max(0, sleep_time - 0.001))  # Account for processing time
                
        finally:
            self.is_profiling = False
            self.sensor.close()
        
        return self._analyze_measurements()
    
    async def profile_async(self, duration_seconds: int, sample_rate_hz: int = 1000) -> PowerProfile:
        """Asynchronous power profiling."""
        if not self.sensor.initialize():
            raise RuntimeError("Failed to initialize power sensor")
        
        self.sensor.set_sample_rate(sample_rate_hz)
        self.measurements.clear()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        self.is_profiling = True
        
        try:
            while time.time() < end_time and self.is_profiling:
                measurement = self.sensor.read_power()
                self.measurements.append(measurement)
                
                # Async sleep
                sleep_time = 1.0 / sample_rate_hz
                await asyncio.sleep(max(0, sleep_time - 0.001))
                
        finally:
            self.is_profiling = False
            self.sensor.close()
        
        return self._analyze_measurements()
    
    def stream_measurements(self, sample_rate_hz: int = 100) -> Iterator[PowerMeasurement]:
        """Stream real-time power measurements."""
        if not self.sensor.initialize():
            raise RuntimeError("Failed to initialize power sensor")
        
        self.sensor.set_sample_rate(sample_rate_hz)
        self.is_profiling = True
        
        try:
            while self.is_profiling:
                measurement = self.sensor.read_power()
                yield measurement
                
                sleep_time = 1.0 / sample_rate_hz
                time.sleep(max(0, sleep_time - 0.001))
                
        finally:
            self.sensor.close()
    
    def stop_profiling(self):
        """Stop ongoing profiling."""
        self.is_profiling = False
    
    def _analyze_measurements(self) -> PowerProfile:
        """Analyze collected measurements."""
        if not self.measurements:
            return PowerProfile(
                idle_power_mw=0.0,
                active_power_mw=0.0,
                peak_power_mw=0.0,
                energy_per_token_mj=0.0,
                total_energy_mj=0.0
            )
        
        power_values = [m.power_mw for m in self.measurements]
        voltage_values = [m.voltage_v for m in self.measurements]
        current_values = [m.current_ma for m in self.measurements]
        
        # Calculate statistics
        avg_power = np.mean(power_values)
        peak_power = np.max(power_values)
        min_power = np.min(power_values)  # Approximate idle power
        
        # Calculate total energy (integrate power over time)
        total_duration_s = self.measurements[-1].timestamp - self.measurements[0].timestamp
        total_energy_mj = avg_power * total_duration_s  # mW * s = mJ
        
        # Estimate energy per token (simplified)
        # This would need actual token generation data for accuracy
        estimated_tokens = max(1, int(total_duration_s * 10))  # Assume 10 tokens/second
        energy_per_token_mj = total_energy_mj / estimated_tokens
        
        return PowerProfile(
            idle_power_mw=min_power,
            active_power_mw=avg_power,
            peak_power_mw=peak_power,
            energy_per_token_mj=energy_per_token_mj,
            total_energy_mj=total_energy_mj,
            avg_voltage_v=np.mean(voltage_values),
            avg_current_ma=np.mean(current_values),
            power_std_mw=np.std(power_values),
            sample_count=len(self.measurements),
            duration_s=total_duration_s
        )
    
    def export_measurements(self, output_path: Path):
        """Export measurements to CSV file."""
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'voltage_v', 'current_ma', 'power_mw', 'temperature_c']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for measurement in self.measurements:
                writer.writerow({
                    'timestamp': measurement.timestamp,
                    'voltage_v': measurement.voltage_v,
                    'current_ma': measurement.current_ma,
                    'power_mw': measurement.power_mw,
                    'temperature_c': measurement.temperature_c or ''
                })
    
    def plot_measurements(self, output_path: Path):
        """Create power consumption plot."""
        if not self.measurements:
            return
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Prepare data
        timestamps = [m.timestamp for m in self.measurements]
        power_values = [m.power_mw for m in self.measurements]
        voltage_values = [m.voltage_v for m in self.measurements]
        current_values = [m.current_ma for m in self.measurements]
        
        # Convert timestamps to relative time
        start_time = timestamps[0]
        relative_times = [(t - start_time) for t in timestamps]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Power Consumption", "Voltage", "Current"),
            vertical_spacing=0.08
        )
        
        # Power plot
        fig.add_trace(
            go.Scatter(x=relative_times, y=power_values, mode='lines', name='Power (mW)'),
            row=1, col=1
        )
        
        # Voltage plot
        fig.add_trace(
            go.Scatter(x=relative_times, y=voltage_values, mode='lines', name='Voltage (V)'),
            row=2, col=1
        )
        
        # Current plot
        fig.add_trace(
            go.Scatter(x=relative_times, y=current_values, mode='lines', name='Current (mA)'),
            row=3, col=1
        )
        
        fig.update_layout(
            title="Power Profiling Results",
            xaxis_title="Time (seconds)",
            height=800,
            showlegend=False
        )
        
        fig.write_html(str(output_path))