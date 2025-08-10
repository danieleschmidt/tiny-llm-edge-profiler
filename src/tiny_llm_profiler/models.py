"""
QuantizedModel implementation for handling compressed LLM models on edge devices.
"""

import json
import struct
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class QuantizationType(str, Enum):
    """Supported quantization types."""
    INT2 = "2bit"
    INT3 = "3bit"
    INT4 = "4bit"
    INT8 = "8bit"
    FLOAT16 = "fp16"


class ModelFormat(str, Enum):
    """Supported model file formats."""
    GGUF = "gguf"
    GGML = "ggml"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    CUSTOM_BIN = "custom_bin"


@dataclass
class ModelMetadata:
    """Metadata for a quantized model."""
    name: str
    architecture: str
    vocab_size: int
    context_length: int
    parameter_count: int
    quantization: QuantizationType
    model_format: ModelFormat
    creation_date: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None


class QuantizedModel:
    """
    Represents a quantized LLM model optimized for edge deployment.
    
    Handles loading, validation, and optimization of compressed models
    for various microcontroller and edge computing platforms.
    """
    
    def __init__(
        self,
        name: str,
        model_path: Optional[Path] = None,
        quantization: Union[str, QuantizationType] = QuantizationType.INT4,
        vocab_size: int = 32000,
        context_length: int = 2048,
        **kwargs
    ):
        """
        Initialize a QuantizedModel.
        
        Args:
            name: Model name/identifier
            model_path: Path to model file
            quantization: Quantization level (2bit, 3bit, 4bit, 8bit, fp16)
            vocab_size: Vocabulary size
            context_length: Maximum context length
        """
        self.name = name
        self.model_path = Path(model_path) if model_path else None
        self.quantization = QuantizationType(quantization) if isinstance(quantization, str) else quantization
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        # Model data
        self.weights: Optional[Dict[str, np.ndarray]] = None
        self.config: Dict[str, Any] = kwargs
        self.metadata: Optional[ModelMetadata] = None
        
        # Computed properties
        self._size_mb: Optional[float] = None
        self._parameter_count: Optional[int] = None
        
        # Add estimated performance attributes for optimization
        self.estimated_tokens_per_second = 10.0  # Default estimate
        
        # Load model if path provided
        if self.model_path and self.model_path.exists():
            self.load()
    
    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        quantization: Union[str, QuantizationType] = QuantizationType.INT4,
        vocab_size: int = 32000,
        **kwargs
    ) -> "QuantizedModel":
        """
        Load a quantized model from file.
        
        Args:
            path: Path to model file
            quantization: Quantization level
            vocab_size: Vocabulary size
            
        Returns:
            QuantizedModel instance
        """
        model_path = Path(path)
        name = model_path.stem
        
        return cls(
            name=name,
            model_path=model_path,
            quantization=quantization,
            vocab_size=vocab_size,
            **kwargs
        )
    
    @classmethod
    def from_huggingface(
        cls,
        model_id: str,
        quantization: Union[str, QuantizationType] = QuantizationType.INT4,
        **kwargs
    ) -> "QuantizedModel":
        """
        Load a model from Hugging Face Hub and quantize it.
        
        Args:
            model_id: Hugging Face model identifier
            quantization: Target quantization level
            
        Returns:
            QuantizedModel instance
        """
        # Placeholder for HF integration
        return cls(
            name=model_id.replace("/", "_"),
            quantization=quantization,
            **kwargs
        )
    
    def load(self) -> None:
        """Load model weights and metadata from file."""
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Detect format based on file extension
        format_map = {
            ".gguf": ModelFormat.GGUF,
            ".ggml": ModelFormat.GGML,
            ".bin": ModelFormat.CUSTOM_BIN,
            ".safetensors": ModelFormat.SAFETENSORS,
            ".pt": ModelFormat.PYTORCH,
            ".pth": ModelFormat.PYTORCH,
        }
        
        file_format = format_map.get(self.model_path.suffix.lower(), ModelFormat.CUSTOM_BIN)
        
        # Load based on detected format
        if file_format == ModelFormat.GGUF:
            self._load_gguf()
        elif file_format == ModelFormat.GGML:
            self._load_ggml()
        elif file_format == ModelFormat.CUSTOM_BIN:
            self._load_custom_binary()
        else:
            raise ValueError(f"Unsupported model format: {file_format}")
        
        # Calculate derived properties
        self._calculate_size()
        self._calculate_parameter_count()
    
    def _load_gguf(self) -> None:
        """Load GGUF format model."""
        # Simplified GGUF loader - would need full implementation
        with open(self.model_path, 'rb') as f:
            # Read GGUF header
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError("Invalid GGUF file")
            
            version = struct.unpack('<I', f.read(4))[0]
            
            # Read metadata (simplified)
            self.metadata = ModelMetadata(
                name=self.name,
                architecture="llama",  # Would parse from file
                vocab_size=self.vocab_size,
                context_length=self.context_length,
                parameter_count=0,  # Would calculate from file
                quantization=self.quantization,
                model_format=ModelFormat.GGUF
            )
            
            # Placeholder for weight loading
            self.weights = {}
    
    def _load_ggml(self) -> None:
        """Load GGML format model."""
        # Simplified GGML loader
        with open(self.model_path, 'rb') as f:
            # Read GGML header
            magic = f.read(4)
            if magic != b'ggml':
                raise ValueError("Invalid GGML file")
            
            # Placeholder implementation
            self.weights = {}
            self.metadata = ModelMetadata(
                name=self.name,
                architecture="llama",
                vocab_size=self.vocab_size,
                context_length=self.context_length,
                parameter_count=0,
                quantization=self.quantization,
                model_format=ModelFormat.GGML
            )
    
    def _load_custom_binary(self) -> None:
        """Load custom binary format model."""
        # Custom format for edge-optimized models
        with open(self.model_path, 'rb') as f:
            # Read header
            header_size = struct.unpack('<I', f.read(4))[0]
            header_data = f.read(header_size)
            
            try:
                header = json.loads(header_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback for binary-only models
                header = {
                    "name": self.name,
                    "quantization": self.quantization.value,
                    "vocab_size": self.vocab_size
                }
            
            # Load weights (simplified)
            self.weights = {}
            self.metadata = ModelMetadata(
                name=header.get("name", self.name),
                architecture=header.get("architecture", "unknown"),
                vocab_size=header.get("vocab_size", self.vocab_size),
                context_length=header.get("context_length", self.context_length),
                parameter_count=header.get("parameter_count", 0),
                quantization=QuantizationType(header.get("quantization", self.quantization.value)),
                model_format=ModelFormat.CUSTOM_BIN
            )
    
    def _calculate_size(self) -> None:
        """Calculate model size in MB."""
        if self.model_path and self.model_path.exists():
            self._size_mb = self.model_path.stat().st_size / (1024 * 1024)
        else:
            # Estimate from weights if loaded
            if self.weights:
                total_bytes = sum(
                    array.nbytes for array in self.weights.values()
                    if isinstance(array, np.ndarray)
                )
                self._size_mb = total_bytes / (1024 * 1024)
            else:
                self._size_mb = 0.0
    
    def _calculate_parameter_count(self) -> None:
        """Calculate total parameter count."""
        if self.weights:
            self._parameter_count = sum(
                array.size for array in self.weights.values()
                if isinstance(array, np.ndarray)
            )
        elif self.metadata and self.metadata.parameter_count:
            self._parameter_count = self.metadata.parameter_count
        else:
            # Estimate based on model size and quantization
            bits_per_param = {
                QuantizationType.INT2: 2,
                QuantizationType.INT3: 3,
                QuantizationType.INT4: 4,
                QuantizationType.INT8: 8,
                QuantizationType.FLOAT16: 16,
            }
            
            if self._size_mb:
                bits = bits_per_param.get(self.quantization, 4)
                bytes_per_param = bits / 8
                self._parameter_count = int((self._size_mb * 1024 * 1024) / bytes_per_param)
            else:
                self._parameter_count = 0
    
    @property
    def size_mb(self) -> float:
        """Model size in megabytes."""
        if self._size_mb is None:
            self._calculate_size()
        return self._size_mb or 0.0
    
    @property
    def parameter_count(self) -> int:
        """Total parameter count."""
        if self._parameter_count is None:
            self._calculate_parameter_count()
        return self._parameter_count or 0
    
    def optimize_for_platform(self, platform: str, constraints: Optional[Dict[str, Any]] = None) -> "QuantizedModel":
        """
        Optimize model for a specific platform.
        
        Args:
            platform: Target platform (esp32, stm32f4, etc.)
            constraints: Platform-specific constraints
            
        Returns:
            Optimized QuantizedModel instance
        """
        constraints = constraints or {}
        
        # Create optimized copy
        optimized = QuantizedModel(
            name=f"{self.name}_optimized_{platform}",
            quantization=self.quantization,
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            **self.config
        )
        
        # Apply platform-specific optimizations
        if platform == "esp32":
            optimized = self._optimize_for_esp32(optimized, constraints)
        elif platform.startswith("stm32"):
            optimized = self._optimize_for_stm32(optimized, constraints)
        elif platform == "rp2040":
            optimized = self._optimize_for_rp2040(optimized, constraints)
        
        return optimized
    
    def _optimize_for_esp32(self, model: "QuantizedModel", constraints: Dict[str, Any]) -> "QuantizedModel":
        """Apply ESP32-specific optimizations."""
        max_memory_kb = constraints.get("max_memory_kb", 400)
        use_psram = constraints.get("use_psram", True)
        
        # Adjust context length if memory constrained
        if not use_psram and max_memory_kb < 300:
            model.context_length = min(model.context_length, 1024)
        
        # Apply quantization if not aggressive enough
        if model.size_mb * 1024 > max_memory_kb and model.quantization != QuantizationType.INT2:
            model.quantization = QuantizationType.INT2
        
        return model
    
    def _optimize_for_stm32(self, model: "QuantizedModel", constraints: Dict[str, Any]) -> "QuantizedModel":
        """Apply STM32-specific optimizations."""
        max_memory_kb = constraints.get("max_memory_kb", 200)
        use_fpu = constraints.get("use_fpu", True)
        
        # Very aggressive memory constraints for STM32
        model.context_length = min(model.context_length, 512)
        
        if model.size_mb * 1024 > max_memory_kb:
            model.quantization = QuantizationType.INT2
        
        return model
    
    def _optimize_for_rp2040(self, model: "QuantizedModel", constraints: Dict[str, Any]) -> "QuantizedModel":
        """Apply RP2040-specific optimizations."""
        max_memory_kb = constraints.get("max_memory_kb", 200)
        
        # RP2040 has limited RAM but good flash storage
        model.context_length = min(model.context_length, 1024)
        
        return model
    
    def export(self, output_path: Union[str, Path], format: ModelFormat = ModelFormat.CUSTOM_BIN) -> None:
        """
        Export model to file in specified format.
        
        Args:
            output_path: Output file path
            format: Export format
        """
        output_path = Path(output_path)
        
        if format == ModelFormat.CUSTOM_BIN:
            self._export_custom_binary(output_path)
        else:
            raise ValueError(f"Export format {format} not yet supported")
    
    def _export_custom_binary(self, output_path: Path) -> None:
        """Export to custom binary format optimized for edge devices."""
        # Create header with metadata
        header = {
            "name": self.name,
            "architecture": self.metadata.architecture if self.metadata else "unknown",
            "quantization": self.quantization.value,
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "parameter_count": self.parameter_count,
            "format_version": "1.0"
        }
        
        header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
        
        with open(output_path, 'wb') as f:
            # Write header size and header
            f.write(struct.pack('<I', len(header_json)))
            f.write(header_json)
            
            # Write weights (simplified - would need actual weight serialization)
            if self.weights:
                for name, weight in self.weights.items():
                    if isinstance(weight, np.ndarray):
                        # Write weight name length and name
                        name_bytes = name.encode('utf-8')
                        f.write(struct.pack('<I', len(name_bytes)))
                        f.write(name_bytes)
                        
                        # Write weight shape and data
                        f.write(struct.pack('<I', len(weight.shape)))
                        f.write(struct.pack(f'<{len(weight.shape)}I', *weight.shape))
                        f.write(weight.tobytes())
    
    def get_memory_requirements(self, platform: str) -> Dict[str, float]:
        """
        Estimate memory requirements for a specific platform.
        
        Args:
            platform: Target platform
            
        Returns:
            Dictionary with memory requirements in KB
        """
        base_model_kb = self.size_mb * 1024
        
        # Estimate additional memory for inference
        context_kb = (self.context_length * 4) / 1024  # 4 bytes per token (simplified)
        kv_cache_kb = (self.context_length * 2048) / 1024  # Simplified KV cache estimate
        
        # Platform-specific overhead
        platform_overhead = {
            "esp32": 50,      # ESP32 system overhead
            "stm32f4": 20,    # STM32 system overhead
            "stm32f7": 30,
            "rp2040": 15,
            "nrf52840": 10,
        }
        
        overhead_kb = platform_overhead.get(platform, 30)
        
        return {
            "model_weights_kb": base_model_kb,
            "context_buffer_kb": context_kb,
            "kv_cache_kb": kv_cache_kb,
            "system_overhead_kb": overhead_kb,
            "total_estimated_kb": base_model_kb + context_kb + kv_cache_kb + overhead_kb
        }
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate model integrity and compatibility.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check file existence
        if self.model_path and not self.model_path.exists():
            issues.append(f"Model file not found: {self.model_path}")
        
        # Check size constraints
        if self.size_mb > 10:  # 10MB limit for edge devices
            issues.append(f"Model too large for edge deployment: {self.size_mb:.1f}MB")
        
        # Check quantization
        if self.quantization not in [QuantizationType.INT2, QuantizationType.INT3, QuantizationType.INT4]:
            issues.append(f"Quantization {self.quantization} may not be optimal for edge devices")
        
        # Check context length
        if self.context_length > 2048:
            issues.append(f"Context length {self.context_length} may exceed memory limits on edge devices")
        
        return len(issues) == 0, issues
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"QuantizedModel(name='{self.name}', "
            f"quantization={self.quantization.value}, "
            f"size={self.size_mb:.1f}MB, "
            f"params={self.parameter_count:,})"
        )