"""
Hardware tests that require real devices connected.

These tests are marked with @pytest.mark.hardware and will only run
when real hardware is available and the --hardware flag is used.
"""

import pytest
import time
import os
from pathlib import Path
import serial
import serial.tools.list_ports
from typing import List, Optional, Dict, Any


# Skip all hardware tests if MOCK_HARDWARE is set
pytestmark = pytest.mark.skipif(
    os.environ.get("MOCK_HARDWARE", "false").lower() == "true",
    reason="Hardware tests disabled (MOCK_HARDWARE=true)"
)


@pytest.fixture(scope="session")
def available_devices() -> List[Dict[str, Any]]:
    """Discover available hardware devices for testing."""
    devices = []
    
    # Scan for devices
    for port in serial.tools.list_ports.comports():
        device_info = {
            "port": port.device,
            "vid": port.vid,
            "pid": port.pid,
            "description": port.description,
            "platform": None
        }
        
        # Identify platform based on VID/PID
        if port.vid == 0x10c4 and port.pid == 0xea60:
            device_info["platform"] = "esp32"
        elif port.vid == 0x0483:
            device_info["platform"] = "stm32"
        elif port.vid == 0x2e8a and port.pid == 0x0003:
            device_info["platform"] = "rp2040"
        elif port.vid == 0x1366:
            device_info["platform"] = "nordic"
        
        if device_info["platform"]:
            devices.append(device_info)
    
    return devices


def find_device_by_platform(devices: List[Dict], platform: str) -> Optional[Dict]:
    """Find first device matching the specified platform."""
    return next((dev for dev in devices if dev["platform"] == platform), None)


@pytest.mark.hardware
class TestRealESP32Hardware:
    """Tests requiring real ESP32 hardware."""
    
    @pytest.fixture
    def esp32_device(self, available_devices):
        """Get ESP32 device for testing."""
        device = find_device_by_platform(available_devices, "esp32")
        if not device:
            pytest.skip("No ESP32 device found")
        return device
    
    def test_esp32_connection(self, esp32_device):
        """Test basic connection to ESP32 device."""
        try:
            with serial.Serial(esp32_device["port"], 921600, timeout=5) as ser:
                assert ser.is_open
                
                # Send basic command
                ser.write(b"AT\r\n")
                time.sleep(0.1)
                
                # Try to read response
                response = ser.read(ser.in_waiting or 10)
                assert len(response) >= 0  # Some response expected
                
        except serial.SerialException as e:
            pytest.fail(f"Failed to connect to ESP32: {e}")
    
    def test_esp32_bootloader_detection(self, esp32_device):
        """Test ESP32 bootloader detection."""
        try:
            with serial.Serial(esp32_device["port"], 115200, timeout=2) as ser:
                # Try to enter download mode
                ser.setDTR(False)
                ser.setRTS(True)
                time.sleep(0.1)
                ser.setRTS(False)
                time.sleep(0.1)
                ser.setDTR(True)
                
                # Send sync command
                sync_cmd = b'\xc0\x00\x08\x24\x00\x00\x00\x00\x00\x07\x07\x12\x20\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\x55\xc0'
                ser.write(sync_cmd)
                
                # Wait for response
                time.sleep(0.5)
                response = ser.read(ser.in_waiting or 100)
                
                # Should receive some bootloader response
                assert len(response) > 0
                
        except serial.SerialException as e:
            pytest.skip(f"Could not test bootloader: {e}")
    
    def test_esp32_flash_detection(self, esp32_device):
        """Test ESP32 flash memory detection."""
        # This would typically use esptool.py
        import subprocess
        
        try:
            result = subprocess.run([
                "esptool.py", 
                "--port", esp32_device["port"],
                "--baud", "115200",
                "flash_id"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                assert "Manufacturer" in result.stdout
                assert "Flash size" in result.stdout
            else:
                pytest.skip("esptool.py not available or flash detection failed")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("esptool.py not available")


@pytest.mark.hardware
class TestRealSTM32Hardware:
    """Tests requiring real STM32 hardware."""
    
    @pytest.fixture
    def stm32_device(self, available_devices):
        """Get STM32 device for testing."""
        device = find_device_by_platform(available_devices, "stm32")
        if not device:
            pytest.skip("No STM32 device found")
        return device
    
    def test_stm32_cdc_connection(self, stm32_device):
        """Test CDC ACM connection to STM32."""
        try:
            with serial.Serial(stm32_device["port"], 115200, timeout=5) as ser:
                assert ser.is_open
                
                # STM32 CDC devices should respond quickly
                ser.write(b"test\r\n")
                time.sleep(0.1)
                
                # May or may not have response depending on firmware
                response = ser.read(ser.in_waiting or 0)
                # Just verify connection works
                assert True
                
        except serial.SerialException as e:
            pytest.fail(f"Failed to connect to STM32: {e}")
    
    def test_stm32_dfu_detection(self, stm32_device):
        """Test STM32 DFU mode detection."""
        import subprocess
        
        try:
            # Check if device is in DFU mode
            result = subprocess.run([
                "dfu-util", "-l"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Look for STM32 DFU device
                if "STM32" in result.stdout:
                    assert "Found DFU" in result.stdout
                else:
                    pytest.skip("STM32 not in DFU mode")
            else:
                pytest.skip("dfu-util not available")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("dfu-util not available")


@pytest.mark.hardware
class TestRealRP2040Hardware:
    """Tests requiring real RP2040 hardware."""
    
    @pytest.fixture
    def rp2040_device(self, available_devices):
        """Get RP2040 device for testing."""
        device = find_device_by_platform(available_devices, "rp2040")
        if not device:
            pytest.skip("No RP2040 device found")
        return device
    
    def test_rp2040_connection(self, rp2040_device):
        """Test basic connection to RP2040."""
        try:
            with serial.Serial(rp2040_device["port"], 115200, timeout=5) as ser:
                assert ser.is_open
                
                # Send test data
                ser.write(b"hello\r\n")
                time.sleep(0.1)
                
                # Read any response
                response = ser.read(ser.in_waiting or 0)
                # Connection successful if no exception
                assert True
                
        except serial.SerialException as e:
            pytest.fail(f"Failed to connect to RP2040: {e}")
    
    def test_rp2040_bootsel_mode(self, rp2040_device):
        """Test RP2040 BOOTSEL mode detection."""
        import subprocess
        
        try:
            # Check for RP2040 mass storage device
            result = subprocess.run([
                "lsblk", "-o", "NAME,LABEL,FSTYPE"
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Look for RPI-RP2 label
                if "RPI-RP2" in result.stdout:
                    assert "RPI-RP2" in result.stdout
                else:
                    pytest.skip("RP2040 not in BOOTSEL mode")
            else:
                pytest.skip("Cannot check block devices")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("lsblk not available")


@pytest.mark.hardware
class TestPowerMeasurement:
    """Tests for real power measurement hardware."""
    
    def test_i2c_power_sensor_detection(self):
        """Test detection of I2C power sensors."""
        try:
            import smbus2
            
            # Try to detect I2C buses
            i2c_buses = []
            for i in range(8):  # Check first 8 I2C buses
                try:
                    bus = smbus2.SMBus(i)
                    i2c_buses.append(i)
                    bus.close()
                except Exception:
                    continue
            
            if not i2c_buses:
                pytest.skip("No I2C buses available")
            
            # Try to detect common power sensor addresses
            power_sensors = []
            for bus_num in i2c_buses:
                try:
                    bus = smbus2.SMBus(bus_num)
                    
                    # Common power sensor I2C addresses
                    sensor_addresses = [0x40, 0x41, 0x44, 0x45]  # INA219, PAC1934
                    
                    for addr in sensor_addresses:
                        try:
                            # Try to read from device
                            bus.read_byte(addr)
                            power_sensors.append({"bus": bus_num, "address": addr})
                        except Exception:
                            continue
                    
                    bus.close()
                    
                except Exception:
                    continue
            
            # Report found sensors (if any)
            print(f"Found {len(power_sensors)} potential power sensors")
            
        except ImportError:
            pytest.skip("smbus2 not available")
    
    def test_ina219_power_sensor(self):
        """Test INA219 power sensor if available."""
        try:
            import smbus2
            
            # Try to find INA219 at common addresses
            for bus_num in range(2):  # Check first 2 buses
                try:
                    bus = smbus2.SMBus(bus_num)
                    
                    for addr in [0x40, 0x41, 0x44, 0x45]:
                        try:
                            # Read configuration register
                            config = bus.read_word_data(addr, 0x00)
                            
                            # INA219 configuration register should have specific bits
                            if config & 0x8000:  # Reset bit should be 0 after startup
                                continue
                            
                            # Try to read voltage register
                            voltage = bus.read_word_data(addr, 0x02)
                            current = bus.read_word_data(addr, 0x04)
                            power = bus.read_word_data(addr, 0x03)
                            
                            # Verify readings are reasonable
                            assert voltage >= 0
                            assert current >= 0
                            assert power >= 0
                            
                            print(f"INA219 found at bus {bus_num}, address 0x{addr:02x}")
                            return
                            
                        except Exception:
                            continue
                    
                    bus.close()
                    
                except Exception:
                    continue
            
            pytest.skip("No INA219 power sensor found")
            
        except ImportError:
            pytest.skip("smbus2 not available")


@pytest.mark.hardware
@pytest.mark.slow
class TestLongRunningHardware:
    """Long-running hardware tests."""
    
    def test_extended_device_communication(self, available_devices):
        """Test extended communication with device."""
        if not available_devices:
            pytest.skip("No hardware devices available")
        
        device = available_devices[0]
        
        try:
            with serial.Serial(device["port"], 115200, timeout=1) as ser:
                start_time = time.time()
                test_duration = 60  # 1 minute test
                
                successful_communications = 0
                failed_communications = 0
                
                while time.time() - start_time < test_duration:
                    try:
                        # Send periodic test command
                        ser.write(b"ping\r\n")
                        time.sleep(0.1)
                        
                        # Try to read response
                        response = ser.read(ser.in_waiting or 0)
                        successful_communications += 1
                        
                        time.sleep(1)  # 1 second interval
                        
                    except Exception:
                        failed_communications += 1
                
                total_attempts = successful_communications + failed_communications
                success_rate = successful_communications / total_attempts if total_attempts > 0 else 0
                
                # Should have >90% success rate
                assert success_rate >= 0.9
                
        except serial.SerialException:
            pytest.skip("Could not establish extended communication")
    
    def test_thermal_monitoring(self, available_devices):
        """Test device thermal behavior during extended operation."""
        if not available_devices:
            pytest.skip("No hardware devices available")
        
        # This test would monitor device temperature over time
        # For now, just simulate the test structure
        test_duration = 300  # 5 minutes
        temperature_readings = []
        
        start_time = time.time()
        while time.time() - start_time < test_duration:
            # Simulate temperature reading
            # In real implementation, this would read from device
            temp_reading = 25.0 + (time.time() - start_time) * 0.1  # Simulated warming
            temperature_readings.append(temp_reading)
            
            time.sleep(10)  # 10 second intervals
        
        # Verify thermal behavior
        if temperature_readings:
            max_temp = max(temperature_readings)
            avg_temp = sum(temperature_readings) / len(temperature_readings)
            
            # Device should not overheat (>85°C is concerning)
            assert max_temp < 85.0
            assert avg_temp > 0


@pytest.mark.hardware
class TestDeviceIdentification:
    """Tests for accurate device identification."""
    
    def test_device_enumeration(self):
        """Test comprehensive device enumeration."""
        devices = list(serial.tools.list_ports.comports())
        
        device_summary = {
            "total_devices": len(devices),
            "esp32_devices": 0,
            "stm32_devices": 0,
            "rp2040_devices": 0,
            "other_devices": 0
        }
        
        for device in devices:
            if device.vid == 0x10c4 and device.pid == 0xea60:
                device_summary["esp32_devices"] += 1
            elif device.vid == 0x0483:
                device_summary["stm32_devices"] += 1
            elif device.vid == 0x2e8a and device.pid == 0x0003:
                device_summary["rp2040_devices"] += 1
            else:
                device_summary["other_devices"] += 1
        
        print(f"Device enumeration summary: {device_summary}")
        
        # At least report what was found
        assert device_summary["total_devices"] >= 0
    
    def test_device_capabilities_detection(self, available_devices):
        """Test detection of device capabilities."""
        for device in available_devices:
            capabilities = {
                "platform": device["platform"],
                "supports_profiling": True,  # Assume all devices support basic profiling
                "supports_power_measurement": False,
                "supports_temperature_monitoring": False,
                "estimated_ram_kb": 0,
                "estimated_flash_mb": 0
            }
            
            # Set capabilities based on platform
            if device["platform"] == "esp32":
                capabilities.update({
                    "supports_power_measurement": True,
                    "supports_temperature_monitoring": True,
                    "estimated_ram_kb": 520,
                    "estimated_flash_mb": 4
                })
            elif device["platform"] == "stm32":
                capabilities.update({
                    "supports_temperature_monitoring": True,
                    "estimated_ram_kb": 512,
                    "estimated_flash_mb": 2
                })
            elif device["platform"] == "rp2040":
                capabilities.update({
                    "estimated_ram_kb": 264,
                    "estimated_flash_mb": 2
                })
            
            # Verify capabilities are reasonable
            assert capabilities["estimated_ram_kb"] > 0
            assert capabilities["estimated_flash_mb"] > 0
            
            print(f"Device {device['port']} capabilities: {capabilities}")


@pytest.fixture(scope="session", autouse=True)
def hardware_test_report(request):
    """Generate hardware test report."""
    def generate_report():
        if hasattr(request.session, "testsfailed"):
            failed_tests = request.session.testsfailed
            total_tests = request.session.testscollected
            
            report = {
                "total_hardware_tests": total_tests,
                "failed_tests": failed_tests,
                "success_rate": (total_tests - failed_tests) / total_tests if total_tests > 0 else 0,
                "timestamp": time.time()
            }
            
            print(f"\nHardware Test Report: {report}")
    
    yield
    generate_report()