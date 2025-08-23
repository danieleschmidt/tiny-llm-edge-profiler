"""
Lightweight CLI implementation that works without heavy dependencies.
"""

import sys
import json
import time
from typing import List, Dict, Any
from pathlib import Path

try:
    import click

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False
    print("Warning: click not available, using basic CLI")

try:
    from .core_lite import (
        SimpleProfiler,
        SimplePlatformManager,
        run_basic_benchmark,
        analyze_benchmark_results,
    )
except ImportError:
    # Handle direct execution
    from core_lite import (
        SimpleProfiler,
        SimplePlatformManager,
        run_basic_benchmark,
        analyze_benchmark_results,
    )


def print_table(headers: List[str], rows: List[List[str]], title: str = None):
    """Print a simple ASCII table."""
    if title:
        print(f"\n{title}")
        print("=" * len(title))

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    # Print header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))

    # Print rows
    for row in rows:
        data_row = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(data_row)

    print()


def cmd_platforms():
    """List supported platforms."""
    platforms = SimplePlatformManager.get_supported_platforms()

    headers = [
        "Platform",
        "Name",
        "RAM (KB)",
        "Flash (KB)",
        "Max Freq (MHz)",
        "Features",
    ]
    rows = []

    for platform in platforms:
        info = SimplePlatformManager.get_platform_info(platform)
        if info:
            features = []
            if info.get("has_wifi"):
                features.append("WiFi")
            if info.get("dual_core"):
                features.append("Dual-Core")

            rows.append(
                [
                    platform,
                    info["name"],
                    str(info["ram_kb"]),
                    str(info["flash_kb"]),
                    str(info["max_freq_mhz"]),
                    ", ".join(features) or "Basic",
                ]
            )

    print_table(headers, rows, "Supported Platforms")


def cmd_benchmark(platforms: List[str] = None, output: str = None):
    """Run benchmark across platforms."""
    print("Running TinyML Edge Profiler Benchmark...")
    print("This is a simulated benchmark for demonstration purposes.")
    print()

    # Run benchmark
    start_time = time.time()
    results = run_basic_benchmark()
    benchmark_time = time.time() - start_time

    # Display results
    headers = [
        "Platform",
        "Tokens/sec",
        "Latency (ms)",
        "Memory (KB)",
        "Power (mW)",
        "Status",
    ]
    rows = []

    for platform, result in results.items():
        if result.get("success", False):
            rows.append(
                [
                    platform,
                    f"{result['tokens_per_second']:.1f}",
                    f"{result['latency_ms']:.1f}",
                    f"{result['memory_kb']:.0f}",
                    f"{result['power_mw']:.0f}",
                    "✓ Success",
                ]
            )
        else:
            rows.append([platform, "N/A", "N/A", "N/A", "N/A", f"✗ Failed"])

    print_table(
        headers, rows, f"Benchmark Results (completed in {benchmark_time:.1f}s)"
    )

    # Analysis
    analysis = analyze_benchmark_results(results)
    if "error" not in analysis:
        stats = analysis["performance_stats"]
        print("Performance Analysis:")
        print(f"  • Best performer: {analysis['best_performer']}")
        print(f"  • Most memory efficient: {analysis['most_efficient']}")
        print(f"  • Lowest power consumption: {analysis['lowest_power']}")
        print(
            f"  • Average performance: {stats['avg_tokens_per_second']:.1f} tokens/sec"
        )
        print(f"  • Average memory usage: {stats['avg_memory_kb']:.0f} KB")
        print(f"  • Average power consumption: {stats['avg_power_mw']:.0f} mW")

    # Save results if requested
    if output:
        output_path = Path(output)
        output_data = {
            "timestamp": time.time(),
            "benchmark_duration_s": benchmark_time,
            "results": results,
            "analysis": analysis,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_path}")


def cmd_profile(platform: str, prompts: List[str] = None):
    """Profile a specific platform."""
    prompts = prompts or ["Hello world", "Generate code", "Explain AI"]

    print(f"Profiling platform: {platform}")
    print(f"Test prompts: {len(prompts)} prompts")
    print()

    try:
        profiler = SimpleProfiler(platform)
        result = profiler.simulate_profiling(prompts)

        if result.success:
            print("Profiling Results:")
            print(f"  • Tokens per second: {result.tokens_per_second:.2f}")
            print(f"  • Average latency: {result.latency_ms:.1f} ms")
            print(f"  • Memory usage: {result.memory_kb:.0f} KB")
            print(f"  • Power consumption: {result.power_mw:.0f} mW")

            # Platform info
            info = SimplePlatformManager.get_platform_info(platform)
            if info:
                memory_percent = (result.memory_kb / info["ram_kb"]) * 100
                print(f"  • Memory utilization: {memory_percent:.1f}%")
        else:
            print(f"Profiling failed: {result.error_message}")

    except Exception as e:
        print(f"Error: {e}")


def cmd_recommend(platform: str, model_size_mb: float = 2.0):
    """Get optimization recommendations."""
    info = SimplePlatformManager.get_platform_info(platform)

    if not info:
        print(f"Error: Unsupported platform '{platform}'")
        return

    print(f"Optimization Recommendations for {info['name']}")
    print("=" * 50)

    print(f"Platform Specifications:")
    print(f"  • RAM: {info['ram_kb']} KB")
    print(f"  • Flash: {info['flash_kb']} KB")
    print(f"  • Max Frequency: {info['max_freq_mhz']} MHz")

    # Memory analysis
    available_ram_kb = info["ram_kb"] * 0.8  # 80% available for model
    model_size_kb = model_size_mb * 1024

    print(f"\nModel Compatibility:")
    print(f"  • Model size: {model_size_mb:.1f} MB ({model_size_kb:.0f} KB)")
    print(f"  • Available RAM: {available_ram_kb:.0f} KB")

    if model_size_kb > available_ram_kb:
        print(f"  ⚠️  Model too large for available RAM!")
        print(f"  Recommendations:")
        print(f"    - Use 2-bit quantization to reduce size by ~75%")
        print(f"    - Reduce context length to 512 tokens")
        if info.get("has_wifi"):
            print(f"    - Consider model streaming from external storage")
    else:
        memory_usage = (model_size_kb / available_ram_kb) * 100
        print(f"  ✓ Model fits in RAM ({memory_usage:.1f}% utilization)")

        if memory_usage > 70:
            print(f"  Recommendations:")
            print(f"    - Consider 3-bit quantization for better headroom")
        else:
            print(f"  Recommendations:")
            print(f"    - Current configuration should work well")
            print(f"    - 4-bit quantization provides good quality/size balance")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Tiny LLM Edge Profiler - Lightweight CLI")
        print("\nCommands:")
        print("  platforms              List supported platforms")
        print("  benchmark [--output=file.json]  Run benchmark")
        print("  profile <platform>     Profile specific platform")
        print("  recommend <platform>   Get optimization recommendations")
        print("\nExamples:")
        print("  python -m tiny_llm_profiler.cli_lite platforms")
        print("  python -m tiny_llm_profiler.cli_lite benchmark --output=results.json")
        print("  python -m tiny_llm_profiler.cli_lite profile esp32")
        print("  python -m tiny_llm_profiler.cli_lite recommend stm32f4")
        return

    command = sys.argv[1]

    if command == "platforms":
        cmd_platforms()

    elif command == "benchmark":
        output_file = None
        for arg in sys.argv[2:]:
            if arg.startswith("--output="):
                output_file = arg.split("=", 1)[1]
        cmd_benchmark(output=output_file)

    elif command == "profile":
        if len(sys.argv) < 3:
            print("Error: platform required for profile command")
            return
        platform = sys.argv[2]
        cmd_profile(platform)

    elif command == "recommend":
        if len(sys.argv) < 3:
            print("Error: platform required for recommend command")
            return
        platform = sys.argv[2]

        model_size = 2.0
        for arg in sys.argv[3:]:
            if arg.startswith("--size="):
                try:
                    model_size = float(arg.split("=", 1)[1])
                except ValueError:
                    print("Warning: Invalid model size, using default 2.0 MB")

        cmd_recommend(platform, model_size)

    else:
        print(f"Unknown command: {command}")
        print("Use 'python -m tiny_llm_profiler.cli_lite' for help")


if __name__ == "__main__":
    main()
