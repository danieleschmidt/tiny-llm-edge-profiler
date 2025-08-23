"""
Command-line interface for the Tiny LLM Edge Profiler.
"""

import sys
import json
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from .profiler import EdgeProfiler, ProfilingConfig
from .models import QuantizedModel
from .platforms import PlatformManager
from .results import ProfileResults


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="tiny-profiler")
def cli():
    """
    Tiny LLM Edge Profiler - Profile quantized LLMs on microcontrollers and edge devices.
    """
    pass


@cli.command()
@click.option(
    "--platform", "-p", required=True, help="Target platform (esp32, stm32f4, etc.)"
)
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to quantized model file",
)
@click.option("--device", "-d", help="Device path (e.g., /dev/ttyUSB0)")
@click.option("--baudrate", "-b", default=921600, help="Serial baudrate")
@click.option("--prompts", multiple=True, help="Test prompts (can be repeated)")
@click.option(
    "--prompts-file", type=click.Path(exists=True), help="File containing test prompts"
)
@click.option(
    "--metrics", multiple=True, default=["latency", "memory"], help="Metrics to collect"
)
@click.option("--output", "-o", help="Output file for results (JSON format)")
@click.option("--duration", default=60, help="Profiling duration in seconds")
@click.option("--iterations", default=10, help="Number of measurement iterations")
@click.option("--warmup", default=3, help="Number of warmup iterations")
def profile(
    platform,
    model,
    device,
    baudrate,
    prompts,
    prompts_file,
    metrics,
    output,
    duration,
    iterations,
    warmup,
):
    """Profile a model on a specific platform."""

    # Load test prompts
    test_prompts = list(prompts) if prompts else []

    if prompts_file:
        with open(prompts_file, "r") as f:
            file_prompts = [line.strip() for line in f if line.strip()]
            test_prompts.extend(file_prompts)

    if not test_prompts:
        test_prompts = [
            "Hello, how are you?",
            "Write a simple Python function",
            "Explain quantum computing in simple terms",
        ]

    console.print(f"[bold blue]Profiling Model on {platform.upper()}[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Platform: {platform}")
    console.print(f"Device: {device or 'auto-detect'}")
    console.print(f"Metrics: {', '.join(metrics)}")
    console.print()

    try:
        # Load model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading model...", total=None)

            quantized_model = QuantizedModel.from_file(model)
            console.print(
                f"✓ Loaded model: {quantized_model.name} ({quantized_model.size_mb:.1f}MB)"
            )

            # Validate model compatibility
            progress.update(task, description="Validating platform compatibility...")
            platform_manager = PlatformManager(platform)
            is_compatible, issues = platform_manager.validate_model_compatibility(
                quantized_model.size_mb, quantized_model.quantization.value
            )

            if not is_compatible:
                console.print("[bold red]⚠ Compatibility Issues:[/bold red]")
                for issue in issues:
                    console.print(f"  • {issue}")

                if not click.confirm("Continue anyway?"):
                    sys.exit(1)

            # Initialize profiler
            progress.update(task, description="Initializing profiler...")
            profiler = EdgeProfiler(platform=platform, device=device, baudrate=baudrate)

            # Configure profiling
            config = ProfilingConfig(
                duration_seconds=duration,
                measurement_iterations=iterations,
                warmup_iterations=warmup,
            )

            # Run profiling
            progress.update(task, description="Running profiling...")

        results = profiler.profile_model(
            model=quantized_model,
            test_prompts=test_prompts,
            metrics=metrics,
            config=config,
        )

        # Display results
        _display_results(results)

        # Save results if requested
        if output:
            results.export_json(output)
            console.print(f"✓ Results saved to {output}")

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to model file",
)
@click.option("--devices", multiple=True, help="Target devices/platforms")
@click.option("--output", "-o", help="Output directory for results")
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "csv", "html"]),
)
def benchmark(model, devices, output, output_format):
    """Run comprehensive benchmarks across multiple devices."""

    if not devices:
        devices = ["esp32", "stm32f7", "rp2040"]

    output_dir = Path(output) if output else Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    console.print(f"[bold blue]Running Benchmarks[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Devices: {', '.join(devices)}")
    console.print()

    results = []

    for device in devices:
        console.print(f"[bold yellow]Testing on {device}...[/bold yellow]")

        try:
            # Load and optimize model for platform
            quantized_model = QuantizedModel.from_file(model)
            optimized_model = quantized_model.optimize_for_platform(device)

            # Run profiling
            profiler = EdgeProfiler(platform=device, connection="local")

            device_results = profiler.profile_model(
                model=optimized_model,
                test_prompts=["Hello world", "Generate code", "Explain AI"],
                metrics=["latency", "memory"],
            )

            results.append(device_results)

            # Show quick summary
            summary = device_results.get_summary()
            if "latency" in summary:
                console.print(
                    f"  ✓ {summary['latency']['tokens_per_second']:.1f} tok/s"
                )

        except Exception as e:
            console.print(f"  ✗ Failed: {e}")

    # Generate comparison report
    _generate_benchmark_report(results, output_dir, output_format)
    console.print(f"\n✓ Benchmark complete. Results in {output_dir}")


@cli.command()
@click.option("--platform", "-p", required=True, help="Target platform")
@click.option("--device", "-d", help="Device path")
@click.option("--duration", default=300, help="Monitoring duration in seconds")
def monitor(platform, device, duration):
    """Monitor real-time performance metrics."""

    console.print(f"[bold blue]Real-time Monitoring - {platform.upper()}[/bold blue]")
    console.print(f"Duration: {duration} seconds")
    console.print("Press Ctrl+C to stop early\n")

    try:
        profiler = EdgeProfiler(platform=platform, device=device)

        with profiler:
            table = Table()
            table.add_column("Time", style="cyan")
            table.add_column("CPU %", style="green")
            table.add_column("Memory", style="yellow")
            table.add_column("Tokens/s", style="magenta")

            console.print("Starting monitoring...")

            for i, metrics in enumerate(profiler.stream_metrics(duration)):
                if i % 10 == 0:  # Update display every 10 samples
                    console.clear()
                    console.print(
                        f"[bold blue]Real-time Monitoring - {platform.upper()}[/bold blue]"
                    )

                    new_table = Table()
                    new_table.add_column("Time", style="cyan")
                    new_table.add_column("CPU %", style="green")
                    new_table.add_column("Memory", style="yellow")
                    new_table.add_column("Tokens/s", style="magenta")

                    timestamp = f"{i//10:02d}:{(i%10)*6:02d}"
                    cpu_percent = f"{metrics.get('cpu_percent', 0):.1f}%"

                    if "memory_mb" in metrics:
                        memory = f"{metrics['memory_mb']:.1f}MB"
                    elif "memory_kb" in metrics:
                        memory = f"{metrics['memory_kb']:.0f}KB"
                    else:
                        memory = "N/A"

                    tokens_per_sec = f"{metrics.get('tokens_per_second', 0):.1f}"

                    new_table.add_row(timestamp, cpu_percent, memory, tokens_per_sec)
                    console.print(new_table)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def platforms():
    """List supported platforms and their capabilities."""

    console.print("[bold blue]Supported Platforms[/bold blue]\n")

    table = Table()
    table.add_column("Platform", style="cyan", no_wrap=True)
    table.add_column("Display Name", style="white")
    table.add_column("Architecture", style="yellow")
    table.add_column("RAM", style="green")
    table.add_column("Max Freq", style="magenta")
    table.add_column("Features", style="blue")

    platform_names = PlatformManager.list_supported_platforms()

    for platform_name in sorted(platform_names):
        info = PlatformManager.get_platform_info(platform_name)
        if info:
            features = []
            if info["has_fpu"]:
                features.append("FPU")
            if info["dual_core"]:
                features.append("Dual-Core")
            if info["ai_accelerator"]:
                features.append("AI-Accel")

            table.add_row(
                platform_name,
                info["display_name"],
                info["architecture"],
                f"{info['ram_kb']}KB",
                f"{info['max_freq_mhz']}MHz",
                ", ".join(features) if features else "Basic",
            )

    console.print(table)


@cli.command()
@click.option("--platform", "-p", required=True, help="Target platform")
@click.option("--model-size", type=float, help="Model size in MB")
@click.option("--quantization", default="4bit", help="Quantization level")
def recommend(platform, model_size, quantization):
    """Get optimization recommendations for a platform."""

    try:
        manager = PlatformManager(platform)

        console.print(
            f"[bold blue]Optimization Recommendations for {platform.upper()}[/bold blue]\n"
        )

        # Show platform info
        info = manager.get_config()
        memory_constraints = manager.get_memory_constraints()

        panel_content = f"""
Platform: {info.display_name}
Architecture: {info.architecture.value}
Available RAM: {memory_constraints['available_ram_kb']}KB
Flash Storage: {memory_constraints['flash_kb']}KB
Max CPU Frequency: {info.capabilities.max_cpu_freq_mhz}MHz
        """.strip()

        console.print(Panel(panel_content, title="Platform Information"))

        # Model compatibility check
        if model_size:
            is_compatible, issues = manager.validate_model_compatibility(
                model_size, quantization
            )

            if is_compatible:
                console.print("[green]✓ Model is compatible with this platform[/green]")
            else:
                console.print("[red]⚠ Compatibility Issues:[/red]")
                for issue in issues:
                    console.print(f"  • {issue}")

        # Get optimization recommendations
        recommendations = manager.get_optimization_recommendations(model_size or 2.0)

        console.print("\n[bold yellow]Optimization Recommendations:[/bold yellow]")
        for key, value in recommendations.items():
            console.print(f"  • {key.replace('_', ' ').title()}: {value}")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option("--device", "-d", required=True, help="Device path (e.g., /dev/ttyUSB0)")
@click.option("--platform", "-p", required=True, help="Platform type")
@click.option("--firmware", type=click.Path(exists=True), help="Firmware file to flash")
def flash(device, platform, firmware):
    """Flash profiler firmware to a device."""

    try:
        manager = PlatformManager(platform)

        if not firmware:
            # Use default firmware for platform
            firmware_dir = Path(__file__).parent / "firmware"
            firmware = firmware_dir / f"{platform}_profiler.bin"

            if not firmware.exists():
                console.print(f"[red]No firmware found for {platform}[/red]")
                console.print("Available firmware:")
                for fw_file in firmware_dir.glob("*.bin"):
                    console.print(f"  • {fw_file.name}")
                sys.exit(1)

        console.print(f"[bold blue]Flashing Firmware to {platform.upper()}[/bold blue]")
        console.print(f"Device: {device}")
        console.print(f"Firmware: {firmware}")

        # Generate flash command
        flash_command = manager.get_flash_command(str(firmware), device)

        console.print(f"\nFlash command:")
        console.print(f"[cyan]{flash_command}[/cyan]")

        if click.confirm("Execute flash command?"):
            import subprocess

            result = subprocess.run(
                flash_command, shell=True, capture_output=True, text=True
            )

            if result.returncode == 0:
                console.print("[green]✓ Firmware flashed successfully[/green]")
            else:
                console.print(f"[red]✗ Flash failed: {result.stderr}[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def _display_results(results: ProfileResults):
    """Display profiling results in a formatted table."""

    console.print(f"\n[bold green]Profiling Results[/bold green]")

    # Summary panel
    summary = results.get_summary()
    efficiency_score = results.calculate_efficiency_score()

    summary_content = f"""
Model: {results.model_name} ({results.model_size_mb:.1f}MB)
Platform: {results.platform}
Quantization: {results.quantization}
Efficiency Score: {efficiency_score:.1f}/100
    """.strip()

    console.print(Panel(summary_content, title="Summary"))

    # Detailed metrics table
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_column("Unit", style="green")

    if results.latency_profile:
        lat = results.latency_profile
        table.add_row("Tokens per Second", f"{lat.tokens_per_second:.2f}", "tok/s")
        table.add_row("First Token Latency", f"{lat.first_token_latency_ms:.1f}", "ms")
        table.add_row("Inter-token Latency", f"{lat.inter_token_latency_ms:.1f}", "ms")
        table.add_row("Total Latency", f"{lat.total_latency_ms:.1f}", "ms")

    if results.memory_profile:
        mem = results.memory_profile
        table.add_row("Peak Memory", f"{mem.peak_memory_kb:.0f}", "KB")
        table.add_row("Memory Usage", f"{mem.memory_usage_kb:.0f}", "KB")
        table.add_row("Baseline Memory", f"{mem.baseline_memory_kb:.0f}", "KB")

    if results.power_profile:
        pwr = results.power_profile
        table.add_row("Active Power", f"{pwr.active_power_mw:.1f}", "mW")
        table.add_row("Energy per Token", f"{pwr.energy_per_token_mj:.2f}", "mJ")
        table.add_row("Total Energy", f"{pwr.total_energy_mj:.1f}", "mJ")

    console.print(table)

    # Recommendations
    recommendations = results.get_recommendations()
    if recommendations:
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        for i, rec in enumerate(recommendations, 1):
            console.print(f"  {i}. {rec}")


def _generate_benchmark_report(
    results: List[ProfileResults], output_dir: Path, format: str
):
    """Generate benchmark comparison report."""

    if format == "json":
        # Save individual results
        for result in results:
            filename = f"{result.model_name}_{result.platform}_benchmark.json"
            result.export_json(output_dir / filename)

        # Create comparison summary
        comparison = {
            "timestamp": results[0].timestamp.isoformat() if results else None,
            "model": results[0].model_name if results else None,
            "platforms": [],
        }

        for result in results:
            summary = result.get_summary()
            summary["efficiency_score"] = result.calculate_efficiency_score()
            comparison["platforms"].append(summary)

        with open(output_dir / "benchmark_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    elif format == "csv":
        # Export all results to a single CSV
        import pandas as pd

        data_rows = []
        for result in results:
            summary = result.get_summary()

            # Flatten the summary
            flat_row = {"platform": result.platform}
            for key, value in summary.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_row[f"{key}_{subkey}"] = subvalue
                else:
                    flat_row[key] = value

            flat_row["efficiency_score"] = result.calculate_efficiency_score()
            data_rows.append(flat_row)

        df = pd.DataFrame(data_rows)
        df.to_csv(output_dir / "benchmark_results.csv", index=False)


def main():
    """Main CLI entry point."""
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
