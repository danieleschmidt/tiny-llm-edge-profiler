# Quick Start Guide - Tiny LLM Edge Profiler

## 🚀 Get Started in 5 Minutes

This guide will help you get the Tiny LLM Edge Profiler running on your first device in under 5 minutes.

## Prerequisites

- Python 3.8+ installed
- Hardware device (ESP32, STM32, RP2040, etc.)
- USB cable for device connection

## Step 1: Installation

### Option A: pip install (Recommended)
```bash
pip install tiny-llm-edge-profiler
```

### Option B: Development Install
```bash
git clone https://github.com/your-org/tiny-llm-edge-profiler
cd tiny-llm-edge-profiler
pip install -e ".[dev]"
```

## Step 2: Verify Installation

```bash
tiny-profiler --version
tiny-profiler list-platforms
```

## Step 3: Connect Your Device

1. **ESP32**: Connect via USB, note the port (e.g., `/dev/ttyUSB0` or `COM3`)
2. **STM32**: Use ST-Link or USB-UART adapter
3. **RP2040**: Connect Raspberry Pi Pico via USB

```bash
# Check connected devices
tiny-profiler detect-devices
```

## Step 4: Flash Profiler Firmware

```bash
# Flash to ESP32
tiny-profiler flash --device /dev/ttyUSB0 --platform esp32

# Flash to STM32F4
tiny-profiler flash --device /dev/ttyACM0 --platform stm32f4

# Flash to RP2040
tiny-profiler flash --device /dev/ttyACM0 --platform rp2040
```

## Step 5: Run Your First Profile

### Quick Test
```bash
# Test with built-in demo model
tiny-profiler demo --device /dev/ttyUSB0 --platform esp32
```

### Profile a Real Model
```python
from tiny_llm_profiler import EdgeProfiler, QuantizedModel

# Load a quantized model
model = QuantizedModel.from_file("tinyllama-2bit.bin")

# Initialize profiler
profiler = EdgeProfiler(
    platform="esp32",
    device="/dev/ttyUSB0"
)

# Run profiling
results = profiler.profile_model(
    model=model,
    test_prompts=["Hello world", "What is AI?"],
    metrics=["latency", "memory", "power"]
)

# Print results
print(f"Tokens/sec: {results.tokens_per_second:.2f}")
print(f"Memory: {results.peak_memory_kb:.1f} KB")
print(f"Energy: {results.energy_per_token_mj:.2f} mJ/token")
```

## Step 6: Explore Results

The profiler generates:
- **Console output**: Real-time metrics
- **HTML report**: Detailed analysis at `profiling_report.html`
- **JSON data**: Raw metrics for further analysis
- **Performance badges**: For your README

## What's Next?

### Explore Advanced Features
- [Power Profiling Guide](./power-profiling-guide.md)
- [Platform Optimization](./platform-optimization.md)
- [Model Quantization](./model-quantization.md)

### Try Different Platforms
```bash
# List all supported platforms
tiny-profiler list-platforms

# Get platform-specific setup
tiny-profiler platform-info esp32
tiny-profiler platform-info stm32f7
```

### Benchmark Multiple Devices
```bash
tiny-profiler benchmark \
  --model tinyllama-2bit.bin \
  --devices esp32,stm32f7,rp2040 \
  --output benchmark_results.json
```

## Troubleshooting

### Device Not Detected
1. Check USB cable connection
2. Verify correct drivers installed
3. Try different USB port
4. Check device permissions: `sudo usermod -a -G dialout $USER`

### Flashing Failed
1. Put device in bootloader mode (if required)
2. Check device-specific flashing procedure
3. Try different baud rate: `--baudrate 115200`

### Low Performance
1. Ensure adequate power supply (especially ESP32)
2. Check for thermal throttling
3. Verify model size fits in device memory
4. Try lower quantization (4-bit → 2-bit)

## Common Issues

| Issue | Solution |
|-------|----------|
| `Permission denied` on device | Add user to dialout group |
| `Model too large` error | Use more aggressive quantization |
| Slow token generation | Check CPU frequency, enable optimizations |
| High power consumption | Enable low-power mode, reduce frequency |

## Next Steps

- 📚 Read the [User Guide](./user-guide.md)
- 🔧 Check [Platform-Specific Setup](./platform-setup.md)
- ⚡ Learn [Performance Optimization](./performance-optimization.md)
- 🤝 Join our [Community Discord](https://discord.gg/your-org)

## Need Help?

- 📖 [Full Documentation](../README.md)
- 🐛 [Report Issues](https://github.com/your-org/tiny-llm-edge-profiler/issues)
- 💬 [Community Support](https://discord.gg/your-org)
- 📧 [Email Support](mailto:support@your-org.com)