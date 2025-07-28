# ADR-0003: Modular Platform Architecture with Adapter Pattern

## Status
Accepted

## Context
The profiler needs to support multiple edge platforms (ESP32, STM32, RP2040, RISC-V, etc.) with different:
- Communication protocols (UART, USB, WiFi, Ethernet)
- Hardware capabilities (memory, sensors, performance counters)
- Toolchains and development environments
- Model optimization requirements

We need an architecture that allows adding new platforms without modifying core profiling logic.

## Decision
We will implement a modular platform architecture using the Adapter pattern, where each supported platform has a dedicated adapter that implements a common interface.

## Architecture

```python
class PlatformAdapter(ABC):
    @abstractmethod
    def connect(self, connection_params: Dict) -> bool
    
    @abstractmethod
    def deploy_model(self, model: QuantizedModel) -> bool
    
    @abstractmethod
    def start_profiling(self, config: ProfilingConfig) -> str
    
    @abstractmethod
    def collect_metrics(self) -> Iterator[Metrics]
    
    @abstractmethod
    def stop_profiling(self) -> ProfilingResults
    
    @abstractmethod
    def disconnect(self) -> bool

class ESP32Adapter(PlatformAdapter):
    # ESP32-specific implementation
    
class STM32Adapter(PlatformAdapter):
    # STM32-specific implementation
```

## Consequences

### Positive
- **Extensibility**: New platforms can be added without changing core code
- **Maintainability**: Platform-specific code is isolated and easier to maintain
- **Testing**: Each adapter can be unit tested independently
- **Flexibility**: Different optimization strategies per platform
- **Parallel Development**: Multiple team members can work on different platforms

### Negative
- **Complexity**: Additional abstraction layer increases complexity
- **Performance**: Small overhead from abstraction layer
- **Initial Setup**: More upfront work to establish adapter infrastructure

## Implementation Details

### Platform Registration
```python
class PlatformRegistry:
    def register_platform(self, name: str, adapter_class: Type[PlatformAdapter])
    def get_adapter(self, name: str) -> PlatformAdapter
    def list_supported_platforms(self) -> List[str]
```

### Configuration Management
Each platform will have its own configuration schema:
```yaml
platforms:
  esp32:
    communication: serial
    baudrate: 921600
    flash_tool: esptool
    optimization_flags: ["use_psram", "dual_core"]
    
  stm32f7:
    communication: usb_cdc
    flash_tool: openocd
    optimization_flags: ["use_fpu", "dsp_instructions"]
```

### Firmware Management
Each platform adapter will manage its own firmware:
- Compilation toolchain
- Flashing procedures
- Version compatibility
- Hardware-specific optimizations