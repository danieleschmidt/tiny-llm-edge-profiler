"""
Lightweight core functionality without heavy dependencies.
This module provides basic functionality that works without numpy, scipy, etc.
"""

import time
import statistics
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SimplifiedProfile:
    """Simplified profiling results without numpy dependencies."""

    tokens_per_second: float = 0.0
    latency_ms: float = 0.0
    memory_kb: float = 0.0
    power_mw: float = 0.0
    success: bool = True
    error_message: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class SimplePlatformManager:
    """Simplified platform manager without external dependencies."""

    PLATFORM_CONFIGS = {
        "esp32": {
            "name": "ESP32",
            "ram_kb": 520,
            "flash_kb": 4096,
            "max_freq_mhz": 240,
            "has_wifi": True,
            "dual_core": True,
        },
        "stm32f4": {
            "name": "STM32F4",
            "ram_kb": 192,
            "flash_kb": 2048,
            "max_freq_mhz": 168,
            "has_wifi": False,
            "dual_core": False,
        },
        "stm32f7": {
            "name": "STM32F7",
            "ram_kb": 512,
            "flash_kb": 2048,
            "max_freq_mhz": 216,
            "has_wifi": False,
            "dual_core": False,
        },
        "rp2040": {
            "name": "RP2040",
            "ram_kb": 264,
            "flash_kb": 2048,
            "max_freq_mhz": 133,
            "has_wifi": False,
            "dual_core": True,
        },
    }

    @classmethod
    def get_supported_platforms(cls) -> List[str]:
        return list(cls.PLATFORM_CONFIGS.keys())

    @classmethod
    def get_platform_info(cls, platform: str) -> Optional[Dict[str, Any]]:
        return cls.PLATFORM_CONFIGS.get(platform)


class SimpleProfiler:
    """Simple profiler for basic functionality testing."""

    def __init__(self, platform: str, connection: str = "local", max_retries: int = 3):
        self.platform = platform
        self.connection = connection
        self.max_retries = max_retries
        self.platform_info = SimplePlatformManager.get_platform_info(platform)

        if not self.platform_info:
            raise ValueError(f"Unsupported platform: {platform}")

        logger.info(f"Initialized profiler for {platform} with {connection} connection")

    def simulate_profiling(self, test_prompts: List[str]) -> SimplifiedProfile:
        """Simulate profiling for testing purposes with error handling."""
        logger.info(f"Starting profiling simulation with {len(test_prompts)} prompts")

        for attempt in range(self.max_retries):
            try:
                # Validate inputs
                if not test_prompts:
                    logger.warning("No test prompts provided, using defaults")
                    test_prompts = ["Hello", "Test", "AI"]

                # Simulate some processing time
                time.sleep(0.1)

                # Generate realistic but simulated results
                prompt_length = (
                    sum(len(p) for p in test_prompts) / len(test_prompts)
                    if test_prompts
                    else 10
                )

                # Base performance varies by platform
                platform_multiplier = {
                    "esp32": 1.0,
                    "stm32f7": 0.8,
                    "stm32f4": 0.6,
                    "rp2040": 0.4,
                }.get(self.platform, 1.0)

                base_tps = 10.0 * platform_multiplier
                latency = (1000.0 / base_tps) * (prompt_length / 10.0)
                memory = self.platform_info["ram_kb"] * 0.3  # Use 30% of available RAM
                power = 50 + (base_tps * 10)  # Base power + dynamic power

                # Add small random variation for realism
                import random

                base_tps *= random.uniform(0.9, 1.1)
                latency *= random.uniform(0.9, 1.1)
                memory *= random.uniform(0.8, 1.0)
                power *= random.uniform(0.95, 1.05)

                logger.info(
                    f"Profiling completed: {base_tps:.1f} tok/s, {memory:.0f}KB"
                )

                return SimplifiedProfile(
                    tokens_per_second=base_tps,
                    latency_ms=latency,
                    memory_kb=memory,
                    power_mw=power,
                    success=True,
                )

            except Exception as e:
                logger.warning(f"Profiling attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("All profiling attempts failed")
                    return SimplifiedProfile(
                        tokens_per_second=0.0,
                        latency_ms=0.0,
                        memory_kb=0.0,
                        power_mw=0.0,
                        success=False,
                        error_message=str(e),
                    )
                time.sleep(0.5)  # Wait before retry


def run_basic_benchmark() -> Dict[str, Any]:
    """Run a basic benchmark across supported platforms."""
    results = {}
    platforms = SimplePlatformManager.get_supported_platforms()
    test_prompts = ["Hello world", "Generate code", "Explain AI"]

    for platform in platforms:
        try:
            profiler = SimpleProfiler(platform)
            profile = profiler.simulate_profiling(test_prompts)
            results[platform] = {
                "success": profile.success,
                "tokens_per_second": profile.tokens_per_second,
                "latency_ms": profile.latency_ms,
                "memory_kb": profile.memory_kb,
                "power_mw": profile.power_mw,
            }
        except Exception as e:
            results[platform] = {"success": False, "error": str(e)}

    return results


def analyze_benchmark_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze benchmark results and provide insights."""
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}

    if not successful_results:
        return {"error": "No successful benchmark results"}

    # Calculate statistics
    tps_values = [r["tokens_per_second"] for r in successful_results.values()]
    memory_values = [r["memory_kb"] for r in successful_results.values()]
    power_values = [r["power_mw"] for r in successful_results.values()]

    analysis = {
        "platforms_tested": len(successful_results),
        "best_performer": max(
            successful_results.keys(),
            key=lambda k: successful_results[k]["tokens_per_second"],
        ),
        "most_efficient": min(
            successful_results.keys(), key=lambda k: successful_results[k]["memory_kb"]
        ),
        "lowest_power": min(
            successful_results.keys(), key=lambda k: successful_results[k]["power_mw"]
        ),
        "performance_stats": {
            "avg_tokens_per_second": statistics.mean(tps_values),
            "max_tokens_per_second": max(tps_values),
            "min_tokens_per_second": min(tps_values),
            "avg_memory_kb": statistics.mean(memory_values),
            "avg_power_mw": statistics.mean(power_values),
        },
    }

    return analysis


if __name__ == "__main__":
    print("Running lightweight benchmark...")
    results = run_basic_benchmark()

    print("\nBenchmark Results:")
    for platform, result in results.items():
        if result.get("success", False):
            print(
                f"  {platform}: {result['tokens_per_second']:.1f} tok/s, "
                f"{result['memory_kb']:.0f}KB, {result['power_mw']:.0f}mW"
            )
        else:
            print(f"  {platform}: FAILED - {result.get('error', 'Unknown error')}")

    print("\nAnalysis:")
    analysis = analyze_benchmark_results(results)
    if "error" not in analysis:
        stats = analysis["performance_stats"]
        print(f"  Best performer: {analysis['best_performer']}")
        print(f"  Most efficient: {analysis['most_efficient']}")
        print(f"  Average performance: {stats['avg_tokens_per_second']:.1f} tok/s")
        print(f"  Average memory usage: {stats['avg_memory_kb']:.0f}KB")
    else:
        print(f"  {analysis['error']}")
