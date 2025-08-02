"""Test data factory for generating realistic test data."""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np


class TestDataFactory:
    """Factory for generating consistent test data across test suites."""
    
    PLATFORMS = ["esp32", "esp32s3", "stm32f4", "stm32f7", "rp2040", "nrf52840", "k210"]
    ARCHITECTURES = {
        "esp32": "xtensa-lx6",
        "esp32s3": "xtensa-lx7", 
        "stm32f4": "arm-cortex-m4",
        "stm32f7": "arm-cortex-m7",
        "rp2040": "arm-cortex-m0+",
        "nrf52840": "arm-cortex-m4",
        "k210": "risc-v-rv64"
    }
    
    MODEL_TYPES = ["tinyllama", "phi-1.5", "opt-125m", "distilbert", "gpt2-small"]
    QUANTIZATION_BITS = [2, 3, 4, 8]
    
    @classmethod
    def create_platform_config(cls, platform: str = None) -> Dict[str, Any]:
        """Create platform configuration for testing."""
        if platform is None:
            platform = random.choice(cls.PLATFORMS)
            
        base_configs = {
            "esp32": {
                "name": "esp32",
                "architecture": "xtensa-lx6",
                "ram_kb": 520,
                "flash_mb": 4,
                "cpu_freq_mhz": 240,
                "has_psram": True,
                "uart_baudrate": 921600,
                "supported_quantizations": [2, 3, 4],
                "power_profiles": {
                    "active": {"min_mw": 150, "max_mw": 400},
                    "idle": {"min_mw": 10, "max_mw": 50}
                }
            },
            "stm32f7": {
                "name": "stm32f7",
                "architecture": "arm-cortex-m7",
                "ram_kb": 512,
                "flash_mb": 2,
                "cpu_freq_mhz": 216,
                "has_fpu": True,
                "uart_baudrate": 115200,
                "supported_quantizations": [4, 8],
                "power_profiles": {
                    "active": {"min_mw": 80, "max_mw": 200},
                    "idle": {"min_mw": 5, "max_mw": 20}
                }
            },
            "rp2040": {
                "name": "rp2040",
                "architecture": "arm-cortex-m0+",
                "ram_kb": 264,
                "flash_mb": 2,
                "cpu_freq_mhz": 133,
                "has_dual_core": True,
                "uart_baudrate": 115200,
                "supported_quantizations": [4, 8],
                "power_profiles": {
                    "active": {"min_mw": 30, "max_mw": 100},
                    "idle": {"min_mw": 2, "max_mw": 10}
                }
            }
        }
        
        return base_configs.get(platform, base_configs["esp32"])
    
    @classmethod
    def create_model_metadata(
        cls, 
        model_type: str = None,
        quantization: int = None,
        size_mb: float = None
    ) -> Dict[str, Any]:
        """Create model metadata for testing."""
        if model_type is None:
            model_type = random.choice(cls.MODEL_TYPES)
        if quantization is None:
            quantization = random.choice(cls.QUANTIZATION_BITS)
        if size_mb is None:
            size_mb = random.uniform(1.0, 10.0)
            
        return {
            "name": f"{model_type}-{quantization}bit",
            "type": model_type,
            "quantization_bits": quantization,
            "size_mb": round(size_mb, 2),
            "size_bytes": int(size_mb * 1024 * 1024),
            "vocab_size": random.choice([32000, 50000, 65000]),
            "layers": random.randint(12, 32),
            "hidden_size": random.choice([768, 1024, 2048, 4096]),
            "architecture": "transformer",
            "optimization_level": random.choice(["o0", "o1", "o2", "o3"]),
            "checksum": f"sha256:{random.getrandbits(256):064x}"
        }
    
    @classmethod
    def create_profiling_results(
        cls,
        platform: str = None,
        model_type: str = None,
        duration_seconds: int = 30,
        num_prompts: int = 5
    ) -> Dict[str, Any]:
        """Create realistic profiling results for testing."""
        platform_config = cls.create_platform_config(platform)
        model_metadata = cls.create_model_metadata(model_type)
        
        # Generate realistic performance metrics based on platform capabilities
        base_latency = cls._calculate_base_latency(platform_config, model_metadata)
        base_power = cls._calculate_base_power(platform_config, model_metadata)
        
        prompts = [f"Test prompt {i+1}" for i in range(num_prompts)]
        
        # Generate per-token metrics
        token_metrics = []
        total_tokens = 0
        for prompt in prompts:
            prompt_tokens = random.randint(5, 20)
            response_tokens = random.randint(10, 50)
            total_tokens += response_tokens
            
            for token_idx in range(response_tokens):
                is_first_token = token_idx == 0
                latency_ms = base_latency * (2.5 if is_first_token else 1.0)
                latency_ms *= random.uniform(0.8, 1.2)  # Add variance
                
                token_metrics.append({
                    "token_index": token_idx,
                    "latency_ms": round(latency_ms, 2),
                    "memory_kb": round(random.uniform(
                        platform_config["ram_kb"] * 0.3,
                        platform_config["ram_kb"] * 0.8
                    ), 1),
                    "power_mw": round(base_power * random.uniform(0.9, 1.1), 1),
                    "cpu_usage_percent": round(random.uniform(70, 95), 1),
                    "temperature_c": round(random.uniform(45, 75), 1)
                })
        
        # Calculate aggregate metrics
        latencies = [t["latency_ms"] for t in token_metrics]
        memory_usage = [t["memory_kb"] for t in token_metrics]
        power_consumption = [t["power_mw"] for t in token_metrics]
        
        first_token_latencies = [token_metrics[0]["latency_ms"]]  # Simplified
        inter_token_latencies = [t["latency_ms"] for t in token_metrics[1:]]
        
        total_time_ms = sum(latencies)
        tokens_per_second = (total_tokens * 1000) / total_time_ms if total_time_ms > 0 else 0
        
        return {
            "metadata": {
                "platform": platform_config["name"],
                "model": model_metadata,
                "test_configuration": {
                    "duration_seconds": duration_seconds,
                    "num_prompts": num_prompts,
                    "warmup_iterations": 3,
                    "sample_rate_hz": 100
                },
                "timestamp": datetime.now().isoformat(),
                "profiler_version": "1.0.0"
            },
            "prompts": prompts,
            "aggregate_metrics": {
                "total_inference_time_ms": round(total_time_ms, 2),
                "first_token_latency_ms": round(np.mean(first_token_latencies), 2),
                "inter_token_latency_ms": round(np.mean(inter_token_latencies), 2),
                "tokens_per_second": round(tokens_per_second, 2),
                "peak_memory_kb": round(max(memory_usage), 1),
                "average_memory_kb": round(np.mean(memory_usage), 1),
                "peak_power_mw": round(max(power_consumption), 1),
                "average_power_mw": round(np.mean(power_consumption), 1),
                "total_energy_mj": round(sum(power_consumption) * total_time_ms / 1e6, 2),
                "energy_per_token_mj": round(
                    (sum(power_consumption) * total_time_ms / 1e6) / total_tokens, 2
                ),
                "cpu_efficiency": round(tokens_per_second / np.mean([
                    t["cpu_usage_percent"] for t in token_metrics
                ]), 3),
                "memory_efficiency": round(tokens_per_second / np.mean(memory_usage), 4)
            },
            "detailed_metrics": {
                "per_token": token_metrics,
                "latency_distribution": {
                    "p50": round(np.percentile(latencies, 50), 2),
                    "p90": round(np.percentile(latencies, 90), 2),
                    "p95": round(np.percentile(latencies, 95), 2),
                    "p99": round(np.percentile(latencies, 99), 2)
                },
                "memory_distribution": {
                    "p50": round(np.percentile(memory_usage, 50), 1),
                    "p90": round(np.percentile(memory_usage, 90), 1),
                    "p95": round(np.percentile(memory_usage, 95), 1),
                    "p99": round(np.percentile(memory_usage, 99), 1)
                }
            },
            "system_metrics": {
                "platform_utilization": {
                    "cpu_usage_percent": round(np.mean([
                        t["cpu_usage_percent"] for t in token_metrics
                    ]), 1),
                    "memory_usage_percent": round(
                        (np.mean(memory_usage) / platform_config["ram_kb"]) * 100, 1
                    ),
                    "frequency_mhz": platform_config["cpu_freq_mhz"],
                    "temperature_celsius": round(np.mean([
                        t["temperature_c"] for t in token_metrics
                    ]), 1)
                },
                "thermal_profile": {
                    "max_temperature_c": round(max([t["temperature_c"] for t in token_metrics]), 1),
                    "thermal_throttling_detected": False,
                    "cooling_efficiency": "normal"
                }
            },
            "quality_metrics": {
                "accuracy_score": round(random.uniform(0.75, 0.95), 3),
                "perplexity": round(random.uniform(15, 35), 2),
                "bleu_score": round(random.uniform(0.6, 0.9), 3),
                "response_coherence": round(random.uniform(0.7, 0.95), 3)
            }
        }
    
    @classmethod
    def create_benchmark_results(
        cls,
        platforms: List[str] = None,
        models: List[str] = None,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Create benchmark comparison results."""
        if platforms is None:
            platforms = random.sample(cls.PLATFORMS, 3)
        if models is None:
            models = random.sample(cls.MODEL_TYPES, 2)
            
        results = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "iterations": iterations,
                "platforms_tested": platforms,
                "models_tested": models,
                "test_duration_minutes": 45
            },
            "results": []
        }
        
        for platform in platforms:
            for model in models:
                for iteration in range(iterations):
                    result = cls.create_profiling_results(
                        platform=platform,
                        model_type=model,
                        duration_seconds=60
                    )
                    result["iteration"] = iteration + 1
                    results["results"].append(result)
        
        # Add comparative analysis
        results["comparative_analysis"] = cls._generate_comparative_analysis(results["results"])
        
        return results
    
    @classmethod
    def create_power_profile(
        cls,
        platform: str = None,
        duration_seconds: int = 300,
        sample_rate_hz: int = 1000
    ) -> Dict[str, Any]:
        """Create power consumption profile data."""
        platform_config = cls.create_platform_config(platform)
        power_profile = platform_config["power_profiles"]
        
        samples = []
        total_samples = duration_seconds * sample_rate_hz
        
        # Generate power samples with realistic patterns
        base_power = random.uniform(
            power_profile["active"]["min_mw"],
            power_profile["active"]["max_mw"]
        )
        
        for i in range(total_samples):
            timestamp = i / sample_rate_hz
            
            # Add inference spikes every few seconds
            if i % (sample_rate_hz * 3) < (sample_rate_hz * 0.5):  # Inference burst
                power_mw = base_power * random.uniform(1.2, 1.8)
            else:  # Idle periods
                idle_power = random.uniform(
                    power_profile["idle"]["min_mw"],
                    power_profile["idle"]["max_mw"]
                )
                power_mw = idle_power * random.uniform(0.8, 1.2)
            
            # Add noise
            power_mw *= random.uniform(0.95, 1.05)
            
            samples.append({
                "timestamp_s": round(timestamp, 3),
                "voltage_v": round(random.uniform(3.25, 3.35), 3),
                "current_ma": round(power_mw / 3.3, 2),
                "power_mw": round(power_mw, 2)
            })
        
        # Calculate statistics
        power_values = [s["power_mw"] for s in samples]
        
        return {
            "metadata": {
                "platform": platform_config["name"],
                "duration_seconds": duration_seconds,
                "sample_rate_hz": sample_rate_hz,
                "total_samples": len(samples),
                "sensor_type": "ina219",
                "measurement_accuracy": "±1%"
            },
            "statistics": {
                "average_power_mw": round(np.mean(power_values), 2),
                "peak_power_mw": round(max(power_values), 2),
                "min_power_mw": round(min(power_values), 2),
                "total_energy_mj": round(sum(power_values) * (1/sample_rate_hz), 2),
                "power_efficiency": round(1000 / np.mean(power_values), 2),  # ops/W
                "duty_cycle_percent": round(
                    len([p for p in power_values if p > base_power * 0.5]) / len(power_values) * 100, 1
                )
            },
            "samples": samples[::100] if len(samples) > 1000 else samples  # Downsample if needed
        }
    
    @classmethod
    def create_stress_test_data(
        cls,
        platform: str = None,
        stress_duration_hours: int = 24
    ) -> Dict[str, Any]:
        """Create long-duration stress test data."""
        platform_config = cls.create_platform_config(platform)
        
        # Simulate degradation over time
        hours_data = []
        for hour in range(stress_duration_hours):
            # Performance degrades slightly over time due to thermal effects
            degradation_factor = 1.0 - (hour * 0.001)  # 0.1% per hour
            
            base_result = cls.create_profiling_results(
                platform=platform,
                duration_seconds=300  # 5-minute samples each hour
            )
            
            # Apply degradation
            metrics = base_result["aggregate_metrics"]
            metrics["tokens_per_second"] *= degradation_factor
            metrics["first_token_latency_ms"] /= degradation_factor
            metrics["inter_token_latency_ms"] /= degradation_factor
            
            # Temperature increases over time
            temp_increase = hour * 0.5  # 0.5°C per hour
            base_temp = base_result["system_metrics"]["platform_utilization"]["temperature_celsius"]
            base_result["system_metrics"]["platform_utilization"]["temperature_celsius"] = round(
                base_temp + temp_increase, 1
            )
            
            hours_data.append({
                "hour": hour,
                "timestamp": (datetime.now() + timedelta(hours=hour)).isoformat(),
                "metrics": base_result["aggregate_metrics"],
                "system": base_result["system_metrics"]
            })
        
        return {
            "test_metadata": {
                "platform": platform_config["name"],
                "test_type": "endurance_stress_test",
                "duration_hours": stress_duration_hours,
                "sampling_interval_minutes": 60
            },
            "stability_analysis": {
                "performance_degradation_percent": round(
                    (1 - hours_data[-1]["metrics"]["tokens_per_second"] / 
                     hours_data[0]["metrics"]["tokens_per_second"]) * 100, 2
                ),
                "thermal_stability": "stable" if hours_data[-1]["system"]["platform_utilization"]["temperature_celsius"] < 85 else "concerning",
                "memory_leaks_detected": False,
                "system_crashes": 0,
                "uptime_percentage": 99.8
            },
            "hourly_data": hours_data
        }
    
    @classmethod
    def save_test_data(cls, data: Dict[str, Any], filepath: Path) -> None:
        """Save test data to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def _calculate_base_latency(platform_config: Dict, model_metadata: Dict) -> float:
        """Calculate base latency based on platform and model characteristics."""
        # Simple heuristic: larger models and slower platforms = higher latency
        base_latency = 50.0  # Base latency in ms
        
        # Platform factor
        freq_factor = 240 / platform_config["cpu_freq_mhz"]  # Normalized to ESP32
        ram_factor = max(1.0, model_metadata["size_mb"] / (platform_config["ram_kb"] / 1024))
        
        # Model complexity factor
        complexity_factor = model_metadata["layers"] / 20.0  # Normalized
        quantization_factor = 8.0 / model_metadata["quantization_bits"]  # Lower bits = faster
        
        return base_latency * freq_factor * ram_factor * complexity_factor / quantization_factor
    
    @staticmethod
    def _calculate_base_power(platform_config: Dict, model_metadata: Dict) -> float:
        """Calculate base power consumption."""
        power_profiles = platform_config.get("power_profiles", {
            "active": {"min_mw": 100, "max_mw": 300}
        })
        
        base_power = (power_profiles["active"]["min_mw"] + power_profiles["active"]["max_mw"]) / 2
        
        # Adjust for model complexity
        complexity_factor = 1.0 + (model_metadata["layers"] - 20) * 0.02
        freq_factor = platform_config["cpu_freq_mhz"] / 200.0
        
        return base_power * complexity_factor * freq_factor
    
    @staticmethod
    def _generate_comparative_analysis(results: List[Dict]) -> Dict[str, Any]:
        """Generate comparative analysis of benchmark results."""
        if not results:
            return {}
        
        # Group by platform and model
        platform_performance = {}
        model_performance = {}
        
        for result in results:
            platform = result["metadata"]["platform"]
            model = result["metadata"]["model"]["name"]
            tps = result["aggregate_metrics"]["tokens_per_second"]
            
            if platform not in platform_performance:
                platform_performance[platform] = []
            platform_performance[platform].append(tps)
            
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(tps)
        
        # Calculate averages
        platform_averages = {
            platform: round(np.mean(values), 2)
            for platform, values in platform_performance.items()
        }
        
        model_averages = {
            model: round(np.mean(values), 2)
            for model, values in model_performance.items()
        }
        
        return {
            "platform_ranking": sorted(
                platform_averages.items(),
                key=lambda x: x[1],
                reverse=True
            ),
            "model_ranking": sorted(
                model_averages.items(),
                key=lambda x: x[1],
                reverse=True
            ),
            "best_combination": {
                "platform": max(platform_averages, key=platform_averages.get),
                "model": max(model_averages, key=model_averages.get),
                "performance": max(platform_averages.values())
            },
            "performance_variance": {
                platform: round(np.std(values), 2)
                for platform, values in platform_performance.items()
            }
        }