"""
Platform abstraction layer for supporting multiple microcontroller and edge platforms.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from pydantic import BaseModel


class Architecture(str, Enum):
    """Supported CPU architectures."""
    ARM_CORTEX_M0 = "arm_cortex_m0"
    ARM_CORTEX_M0_PLUS = "arm_cortex_m0_plus"
    ARM_CORTEX_M3 = "arm_cortex_m3"
    ARM_CORTEX_M4 = "arm_cortex_m4"
    ARM_CORTEX_M7 = "arm_cortex_m7"
    ARM_CORTEX_A7 = "arm_cortex_a7"
    ARM_CORTEX_A53 = "arm_cortex_a53"
    XTENSA_LX6 = "xtensa_lx6"
    XTENSA_LX7 = "xtensa_lx7"
    RISCV_RV32 = "riscv_rv32"
    RISCV_RV64 = "riscv_rv64"
    X86_64 = "x86_64"


@dataclass
class PlatformCapabilities:
    """Platform capabilities and constraints."""
    has_fpu: bool = False
    has_dsp: bool = False
    has_simd: bool = False
    has_vector_unit: bool = False
    has_dedicated_ai_accelerator: bool = False
    supports_dual_core: bool = False
    supports_wifi: bool = False
    supports_bluetooth: bool = False
    max_cpu_freq_mhz: int = 80
    min_cpu_freq_mhz: int = 1


@dataclass
class MemoryConfiguration:
    """Memory configuration for a platform."""
    ram_kb: int
    flash_kb: int
    external_ram_kb: int = 0
    cache_kb: int = 0
    has_mmu: bool = False
    has_psram: bool = False


@dataclass
class PlatformConfig:
    """Complete platform configuration."""
    name: str
    display_name: str
    architecture: Architecture
    memory: MemoryConfiguration
    capabilities: PlatformCapabilities
    toolchain: str
    flash_command: Optional[str] = None
    debug_interface: Optional[str] = None
    default_baudrate: int = 115200
    supported_quantizations: List[str] = None
    optimization_flags: List[str] = None


class PlatformManager:
    """
    Manages platform-specific configurations and operations.
    
    Provides abstraction layer for different microcontroller and edge computing
    platforms, handling their specific capabilities and constraints.
    """
    
    def __init__(self, platform_name: str):
        """
        Initialize PlatformManager for a specific platform.
        
        Args:
            platform_name: Name of the target platform
        """
        self.platform_name = platform_name.lower()
        self.config = self._load_platform_config(self.platform_name)
        
    def _load_platform_config(self, platform_name: str) -> PlatformConfig:
        """Load configuration for the specified platform."""
        # Built-in platform configurations
        configs = self._get_builtin_configs()
        
        if platform_name in configs:
            return configs[platform_name]
        else:
            # Try to load from external config file
            config_path = Path(__file__).parent / "configs" / f"{platform_name}.json"
            if config_path.exists():
                return self._load_config_from_file(config_path)
            else:
                raise ValueError(f"Unsupported platform: {platform_name}")
    
    def _get_builtin_configs(self) -> Dict[str, PlatformConfig]:
        """Get built-in platform configurations."""
        return {
            # ESP32 Family
            "esp32": PlatformConfig(
                name="esp32",
                display_name="ESP32",
                architecture=Architecture.XTENSA_LX6,
                memory=MemoryConfiguration(
                    ram_kb=520,
                    flash_kb=4096,
                    external_ram_kb=8192,  # PSRAM
                    has_psram=True
                ),
                capabilities=PlatformCapabilities(
                    has_fpu=True,
                    supports_dual_core=True,
                    supports_wifi=True,
                    supports_bluetooth=True,
                    max_cpu_freq_mhz=240,
                    min_cpu_freq_mhz=10
                ),
                toolchain="xtensa-esp32-elf",
                flash_command="esptool.py",
                debug_interface="jtag",
                default_baudrate=921600,
                supported_quantizations=["2bit", "3bit", "4bit"],
                optimization_flags=["-O2", "-ffast-math", "-DESP32_OPTIMIZED"]
            ),
            
            "esp32s3": PlatformConfig(
                name="esp32s3",
                display_name="ESP32-S3",
                architecture=Architecture.XTENSA_LX7,
                memory=MemoryConfiguration(
                    ram_kb=512,
                    flash_kb=8192,
                    external_ram_kb=32768,  # Up to 32MB PSRAM
                    has_psram=True
                ),
                capabilities=PlatformCapabilities(
                    has_fpu=True,
                    has_vector_unit=True,  # AI acceleration
                    supports_dual_core=True,
                    supports_wifi=True,
                    supports_bluetooth=True,
                    max_cpu_freq_mhz=240,
                    min_cpu_freq_mhz=10
                ),
                toolchain="xtensa-esp32s3-elf",
                flash_command="esptool.py",
                default_baudrate=921600,
                supported_quantizations=["2bit", "3bit", "4bit", "8bit"],
                optimization_flags=["-O2", "-ffast-math", "-DESP32S3_OPTIMIZED", "-DESP_NN"]
            ),
            
            # STM32 Family
            "stm32f4": PlatformConfig(
                name="stm32f4",
                display_name="STM32F4",
                architecture=Architecture.ARM_CORTEX_M4,
                memory=MemoryConfiguration(
                    ram_kb=192,
                    flash_kb=2048,
                    cache_kb=8
                ),
                capabilities=PlatformCapabilities(
                    has_fpu=True,
                    has_dsp=True,
                    max_cpu_freq_mhz=168,
                    min_cpu_freq_mhz=1
                ),
                toolchain="arm-none-eabi-gcc",
                flash_command="st-flash",
                debug_interface="swd",
                default_baudrate=115200,
                supported_quantizations=["2bit", "3bit", "4bit"],
                optimization_flags=["-O2", "-mthumb", "-mcpu=cortex-m4", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16"]
            ),
            
            "stm32f7": PlatformConfig(
                name="stm32f7",
                display_name="STM32F7",
                architecture=Architecture.ARM_CORTEX_M7,
                memory=MemoryConfiguration(
                    ram_kb=512,
                    flash_kb=2048,
                    cache_kb=32  # I-cache + D-cache
                ),
                capabilities=PlatformCapabilities(
                    has_fpu=True,
                    has_dsp=True,
                    max_cpu_freq_mhz=216,
                    min_cpu_freq_mhz=1
                ),
                toolchain="arm-none-eabi-gcc",
                flash_command="st-flash",
                debug_interface="swd",
                default_baudrate=115200,
                supported_quantizations=["2bit", "3bit", "4bit", "8bit"],
                optimization_flags=["-O2", "-mthumb", "-mcpu=cortex-m7", "-mfloat-abi=hard", "-mfpu=fpv5-d16"]
            ),
            
            # Raspberry Pi Pico
            "rp2040": PlatformConfig(
                name="rp2040",
                display_name="Raspberry Pi Pico (RP2040)",
                architecture=Architecture.ARM_CORTEX_M0_PLUS,
                memory=MemoryConfiguration(
                    ram_kb=264,
                    flash_kb=16384,  # 16MB external flash typical
                    external_ram_kb=0
                ),
                capabilities=PlatformCapabilities(
                    supports_dual_core=True,
                    max_cpu_freq_mhz=133,
                    min_cpu_freq_mhz=1
                ),
                toolchain="arm-none-eabi-gcc",
                flash_command="picotool",
                debug_interface="swd",
                default_baudrate=115200,
                supported_quantizations=["2bit", "3bit", "4bit"],
                optimization_flags=["-O2", "-mthumb", "-mcpu=cortex-m0plus"]
            ),
            
            # Nordic nRF52840
            "nrf52840": PlatformConfig(
                name="nrf52840",
                display_name="Nordic nRF52840",
                architecture=Architecture.ARM_CORTEX_M4,
                memory=MemoryConfiguration(
                    ram_kb=256,
                    flash_kb=1024
                ),
                capabilities=PlatformCapabilities(
                    has_fpu=True,
                    supports_bluetooth=True,
                    max_cpu_freq_mhz=64,
                    min_cpu_freq_mhz=1
                ),
                toolchain="arm-none-eabi-gcc",
                flash_command="nrfjprog",
                debug_interface="swd",
                default_baudrate=115200,
                supported_quantizations=["2bit", "3bit", "4bit"],
                optimization_flags=["-O2", "-mthumb", "-mcpu=cortex-m4", "-mfloat-abi=hard", "-mfpu=fpv4-sp-d16"]
            ),
            
            # RISC-V Platforms
            "k210": PlatformConfig(
                name="k210",
                display_name="Kendryte K210",
                architecture=Architecture.RISCV_RV64,
                memory=MemoryConfiguration(
                    ram_kb=8192,  # 8MB SRAM
                    flash_kb=16384,  # 16MB flash typical
                    has_mmu=False
                ),
                capabilities=PlatformCapabilities(
                    has_fpu=True,
                    has_dedicated_ai_accelerator=True,  # KPU
                    supports_dual_core=True,
                    max_cpu_freq_mhz=400,
                    min_cpu_freq_mhz=1
                ),
                toolchain="riscv64-unknown-elf-gcc",
                flash_command="kflash",
                debug_interface="jtag",
                default_baudrate=115200,
                supported_quantizations=["2bit", "3bit", "4bit", "8bit"],
                optimization_flags=["-O2", "-march=rv64imafc", "-mabi=lp64f", "-DCONFIG_K210"]
            ),
            
            "bl602": PlatformConfig(
                name="bl602",
                display_name="Bouffalo Lab BL602",
                architecture=Architecture.RISCV_RV32,
                memory=MemoryConfiguration(
                    ram_kb=276,
                    flash_kb=2048
                ),
                capabilities=PlatformCapabilities(
                    has_fpu=True,
                    supports_wifi=True,
                    supports_bluetooth=True,
                    max_cpu_freq_mhz=192,
                    min_cpu_freq_mhz=1
                ),
                toolchain="riscv32-unknown-elf-gcc",
                flash_command="blflash",
                default_baudrate=2000000,
                supported_quantizations=["2bit", "3bit", "4bit"],
                optimization_flags=["-O2", "-march=rv32imafc", "-mabi=ilp32f"]
            ),
            
            # Single Board Computers
            "rpi_zero": PlatformConfig(
                name="rpi_zero",
                display_name="Raspberry Pi Zero",
                architecture=Architecture.ARM_CORTEX_A7,
                memory=MemoryConfiguration(
                    ram_kb=512 * 1024,  # 512MB
                    flash_kb=32 * 1024 * 1024,  # 32GB SD card typical
                    has_mmu=True
                ),
                capabilities=PlatformCapabilities(
                    has_fpu=True,
                    max_cpu_freq_mhz=1000,
                    min_cpu_freq_mhz=700
                ),
                toolchain="arm-linux-gnueabihf-gcc",
                default_baudrate=115200,
                supported_quantizations=["2bit", "3bit", "4bit", "8bit", "fp16"],
                optimization_flags=["-O2", "-mcpu=arm1176jzf-s", "-mfloat-abi=hard", "-mfpu=vfp"]
            ),
            
            "jetson_nano": PlatformConfig(
                name="jetson_nano",
                display_name="NVIDIA Jetson Nano",
                architecture=Architecture.ARM_CORTEX_A53,
                memory=MemoryConfiguration(
                    ram_kb=4 * 1024 * 1024,  # 4GB
                    flash_kb=32 * 1024 * 1024,  # 32GB eMMC typical
                    has_mmu=True
                ),
                capabilities=PlatformCapabilities(
                    has_fpu=True,
                    has_simd=True,
                    has_dedicated_ai_accelerator=True,  # GPU
                    supports_dual_core=True,
                    max_cpu_freq_mhz=1479,
                    min_cpu_freq_mhz=102
                ),
                toolchain="aarch64-linux-gnu-gcc",
                default_baudrate=115200,
                supported_quantizations=["2bit", "3bit", "4bit", "8bit", "fp16"],
                optimization_flags=["-O2", "-mcpu=cortex-a57", "-mtune=cortex-a57"]
            )
        }
    
    def _load_config_from_file(self, config_path: Path) -> PlatformConfig:
        """Load platform configuration from JSON file."""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Convert JSON data to PlatformConfig
        memory_config = MemoryConfiguration(**data['memory'])
        capabilities = PlatformCapabilities(**data['capabilities'])
        
        return PlatformConfig(
            name=data['name'],
            display_name=data['display_name'],
            architecture=Architecture(data['architecture']),
            memory=memory_config,
            capabilities=capabilities,
            toolchain=data['toolchain'],
            flash_command=data.get('flash_command'),
            debug_interface=data.get('debug_interface'),
            default_baudrate=data.get('default_baudrate', 115200),
            supported_quantizations=data.get('supported_quantizations', ["4bit"]),
            optimization_flags=data.get('optimization_flags', [])
        )
    
    def get_config(self) -> PlatformConfig:
        """Get the platform configuration."""
        return self.config
    
    def get_memory_constraints(self) -> Dict[str, int]:
        """Get memory constraints for the platform."""
        return {
            "total_ram_kb": self.config.memory.ram_kb,
            "available_ram_kb": int(self.config.memory.ram_kb * 0.8),  # Reserve 20% for system
            "flash_kb": self.config.memory.flash_kb,
            "external_ram_kb": self.config.memory.external_ram_kb,
            "cache_kb": self.config.memory.cache_kb
        }
    
    def get_optimization_recommendations(self, model_size_mb: float) -> Dict[str, Any]:
        """Get optimization recommendations for a model on this platform."""
        memory_constraints = self.get_memory_constraints()
        available_ram_mb = memory_constraints["available_ram_kb"] / 1024
        
        recommendations = {
            "quantization": "4bit",
            "context_length": 2048,
            "batch_size": 1,
            "use_external_memory": False,
            "optimization_level": "balanced"
        }
        
        # Adjust based on model size vs available memory
        memory_ratio = model_size_mb / available_ram_mb
        
        if memory_ratio > 0.8:
            recommendations["quantization"] = "2bit"
            recommendations["context_length"] = 512
            recommendations["optimization_level"] = "aggressive"
            
            if self.config.memory.external_ram_kb > 0:
                recommendations["use_external_memory"] = True
        
        elif memory_ratio > 0.5:
            recommendations["quantization"] = "3bit"
            recommendations["context_length"] = 1024
            recommendations["optimization_level"] = "moderate"
        
        # Platform-specific adjustments
        if self.config.architecture == Architecture.ARM_CORTEX_M0_PLUS:
            # M0+ has no FPU, prefer integer operations
            recommendations["prefer_integer_ops"] = True
            recommendations["context_length"] = min(recommendations["context_length"], 512)
        
        elif self.config.capabilities.has_dedicated_ai_accelerator:
            # Can handle larger models with AI acceleration
            recommendations["quantization"] = "4bit"
            recommendations["use_ai_accelerator"] = True
        
        return recommendations
    
    def validate_model_compatibility(self, model_size_mb: float, quantization: str) -> tuple[bool, List[str]]:
        """
        Validate if a model is compatible with this platform.
        
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        issues = []
        memory_constraints = self.get_memory_constraints()
        available_ram_mb = memory_constraints["available_ram_kb"] / 1024
        
        # Check model size vs available RAM
        if model_size_mb > available_ram_mb:
            if self.config.memory.external_ram_kb > 0:
                total_available_mb = (memory_constraints["available_ram_kb"] + 
                                    self.config.memory.external_ram_kb) / 1024
                if model_size_mb > total_available_mb:
                    issues.append(f"Model size ({model_size_mb:.1f}MB) exceeds total available memory ({total_available_mb:.1f}MB)")
            else:
                issues.append(f"Model size ({model_size_mb:.1f}MB) exceeds available RAM ({available_ram_mb:.1f}MB)")
        
        # Check quantization support
        if quantization not in self.config.supported_quantizations:
            issues.append(f"Quantization '{quantization}' not supported on {self.config.display_name}")
        
        # Platform-specific checks
        if self.config.architecture in [Architecture.ARM_CORTEX_M0, Architecture.ARM_CORTEX_M0_PLUS]:
            if model_size_mb > 1.0:  # Very conservative for M0/M0+
                issues.append("Models larger than 1MB may be too slow on Cortex-M0/M0+ platforms")
        
        return len(issues) == 0, issues
    
    def get_flash_command(self, firmware_path: str, device_port: str = None) -> str:
        """Generate platform-specific flash command."""
        if not self.config.flash_command:
            return f"# No flash command configured for {self.config.name}"
        
        device_port = device_port or "/dev/ttyUSB0"
        
        command_templates = {
            "esptool.py": f"esptool.py --chip {self.config.name} --port {device_port} write_flash 0x1000 {firmware_path}",
            "st-flash": f"st-flash write {firmware_path} 0x8000000",
            "picotool": f"picotool load {firmware_path} --execute",
            "nrfjprog": f"nrfjprog --program {firmware_path} --sectorerase --verify --reset",
            "kflash": f"kflash -p {device_port} -b 1500000 {firmware_path}",
            "blflash": f"blflash flash {firmware_path}"
        }
        
        return command_templates.get(self.config.flash_command, 
                                   f"{self.config.flash_command} {firmware_path}")
    
    def get_build_flags(self) -> List[str]:
        """Get compiler flags optimized for this platform."""
        base_flags = self.config.optimization_flags or []
        
        # Add platform-specific defines
        platform_defines = [
            f"-DPLATFORM_{self.config.name.upper()}",
            f"-DARCH_{self.config.architecture.value.upper()}"
        ]
        
        # Add capability-based defines
        if self.config.capabilities.has_fpu:
            platform_defines.append("-DHAS_FPU=1")
        
        if self.config.capabilities.has_dsp:
            platform_defines.append("-DHAS_DSP=1")
        
        if self.config.capabilities.supports_dual_core:
            platform_defines.append("-DSUPPORTS_DUAL_CORE=1")
        
        if self.config.capabilities.has_dedicated_ai_accelerator:
            platform_defines.append("-DHAS_AI_ACCELERATOR=1")
        
        return base_flags + platform_defines
    
    @classmethod
    def list_supported_platforms(cls) -> List[str]:
        """Get list of all supported platform names."""
        temp_manager = cls.__new__(cls)  # Create instance without calling __init__
        configs = temp_manager._get_builtin_configs()
        return list(configs.keys())
    
    @classmethod
    def get_platform_info(cls, platform_name: str) -> Dict[str, Any]:
        """Get summary information about a platform."""
        try:
            manager = cls(platform_name)
            config = manager.get_config()
            
            return {
                "name": config.name,
                "display_name": config.display_name,
                "architecture": config.architecture.value,
                "ram_kb": config.memory.ram_kb,
                "flash_kb": config.memory.flash_kb,
                "max_freq_mhz": config.capabilities.max_cpu_freq_mhz,
                "has_fpu": config.capabilities.has_fpu,
                "dual_core": config.capabilities.supports_dual_core,
                "ai_accelerator": config.capabilities.has_dedicated_ai_accelerator,
                "supported_quantizations": config.supported_quantizations
            }
        except ValueError:
            return None