"""
Generation 1 Enhancement: Quick-start CLI for immediate profiling
Simple command-line interface for getting started quickly.
"""

import sys
import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.prompt import Prompt, Confirm

from .core_lite import (
    QuickStartProfiler,
    quick_check,
    get_recommended_platform,
    print_platform_comparison,
    SimplePlatformManager
)

console = Console()


@click.group()
@click.version_option(version="0.4.0", prog_name="tiny-profiler-quickstart")
def quickstart_cli():
    """
    🚀 Tiny LLM Edge Profiler - Quick Start CLI
    
    Get started profiling LLMs on edge devices in seconds!
    """
    pass


@quickstart_cli.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--platform", "-p", help="Target platform (esp32, stm32f4, etc.)")
@click.option("--output", "-o", help="Output file for results")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed information")
def quick_profile(model_path: str, platform: Optional[str], output: Optional[str], verbose: bool):
    """
    Quick profile a model - one command to get results!
    
    MODEL_PATH: Path to your quantized model file
    """
    console.print("🚀 [bold blue]Quick Profile Starting...[/bold blue]")
    
    # Auto-detect platform if not specified
    if not platform:
        console.print("📡 [yellow]Auto-detecting best platform...[/yellow]")
        platform = get_recommended_platform(model_path)
        console.print(f"✅ Recommended platform: [bold green]{platform}[/bold green]")
    
    # Quick compatibility check first
    console.print(f"🔍 [yellow]Checking compatibility with {platform}...[/yellow]")
    
    if not quick_check(model_path, platform):
        console.print("❌ [bold red]Compatibility check failed![/bold red]")
        console.print("💡 Try these solutions:")
        console.print("   • Use a smaller quantized model")
        console.print("   • Try a different platform")
        console.print("   • Run 'tiny-profiler-quickstart platforms' to see options")
        sys.exit(1)
    
    console.print("✅ [bold green]Compatibility check passed![/bold green]")
    
    # Run profiling
    profiler = QuickStartProfiler()
    
    with console.status("[bold green]Running profiling analysis...") as status:
        results = profiler.quick_profile(model_path, platform)
    
    # Display results
    display_results(results, verbose)
    
    # Save results if requested
    if output:
        save_results(results, output)
        console.print(f"💾 Results saved to: [bold blue]{output}[/bold blue]")
    
    # Show next steps
    show_next_steps(results, platform)


@quickstart_cli.command()
@click.argument("model_path", type=click.Path(exists=True))
def check(model_path: str):
    """
    Quick compatibility check for all platforms.
    
    MODEL_PATH: Path to your quantized model file
    """
    console.print("🔍 [bold blue]Checking Model Compatibility[/bold blue]")
    console.print(f"Model: [yellow]{model_path}[/yellow]")
    console.print()
    
    profiler = QuickStartProfiler()
    platforms = SimplePlatformManager.get_supported_platforms()
    
    table = Table(title="Platform Compatibility Check")
    table.add_column("Platform", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Model Size", style="blue")
    table.add_column("Flash Usage", style="green")
    
    compatible_platforms = []
    
    for platform in track(platforms, description="Checking platforms..."):
        result = profiler.check_model_compatibility(model_path, platform)
        
        if result["compatible"]:
            status = "✅ Compatible"
            compatible_platforms.append(platform)
            color = "green"
        else:
            status = "❌ Too large"
            color = "red"
        
        table.add_row(
            platform,
            f"[{color}]{status}[/{color}]",
            f"{result.get('model_size_mb', 0):.1f} MB",
            f"{result.get('utilization', {}).get('flash_percent', 0):.1f}%"
        )
    
    console.print(table)
    
    if compatible_platforms:
        console.print(f"\n✨ [bold green]Compatible platforms:[/bold green] {', '.join(compatible_platforms)}")
        recommended = get_recommended_platform(model_path)
        console.print(f"🎯 [bold blue]Recommended:[/bold blue] {recommended}")
    else:
        console.print("\n❌ [bold red]No compatible platforms found![/bold red]")
        console.print("💡 Try using a smaller quantized model (2-bit or 3-bit)")


@quickstart_cli.command()
def platforms():
    """Show all supported platforms with specifications."""
    print_platform_comparison()


@quickstart_cli.command()
@click.option("--platform", "-p", default="esp32", help="Target platform for guide")
def guide(platform: str):
    """
    Show getting started guide for a platform.
    """
    profiler = QuickStartProfiler()
    guide_data = profiler.get_getting_started_guide(platform)
    
    if "error" in guide_data:
        console.print(f"❌ [bold red]{guide_data['error']}[/bold red]")
        console.print(f"Supported platforms: {', '.join(guide_data.get('supported_platforms', []))}")
        return
    
    console.print(f"📖 [bold blue]Getting Started with {platform.upper()}[/bold blue]")
    console.print()
    
    # Show platform info
    info = guide_data["platform_info"]
    console.print(Panel(
        f"[bold]{info['name']}[/bold]\n"
        f"RAM: {info['ram_kb']} KB\n"
        f"Flash: {info['flash_kb']} KB\n"
        f"Max Frequency: {info['max_freq_mhz']} MHz\n"
        f"Cores: {'Dual' if info['dual_core'] else 'Single'}\n"
        f"WiFi: {'Yes' if info['has_wifi'] else 'No'}",
        title="Platform Specifications",
        border_style="blue"
    ))
    
    # Show steps
    for step_data in guide_data["quick_start_steps"]:
        console.print(f"\n[bold cyan]Step {step_data['step']}: {step_data['title']}[/bold cyan]")
        console.print(step_data['description'])
        
        if 'details' in step_data:
            for detail in step_data['details']:
                console.print(f"  • {detail}")
        
        if 'code_example' in step_data:
            console.print("[dim]Code example:[/dim]")
            console.print(Panel(step_data['code_example'], border_style="dim"))
    
    # Show common issues
    console.print("\n[bold yellow]Common Issues & Solutions:[/bold yellow]")
    for issue in guide_data["common_issues"]:
        console.print(f"❓ [yellow]{issue['issue']}[/yellow]")
        console.print(f"   💡 {issue['solution']}")


@quickstart_cli.command()
def interactive():
    """Interactive setup wizard for new users."""
    console.print("🧙 [bold magenta]Interactive Setup Wizard[/bold magenta]")
    console.print("Let's get you set up step by step!\n")
    
    # Step 1: Model file
    model_path = Prompt.ask("📁 [bold]Path to your model file[/bold]")
    
    if not Path(model_path).exists():
        console.print("❌ [bold red]File not found![/bold red]")
        console.print("Please check the path and try again.")
        return
    
    # Step 2: Platform selection
    platforms = SimplePlatformManager.get_supported_platforms()
    console.print(f"\n🔧 [bold]Available platforms:[/bold] {', '.join(platforms)}")
    
    auto_detect = Confirm.ask("🤖 Auto-detect best platform?", default=True)
    
    if auto_detect:
        platform = get_recommended_platform(model_path)
        console.print(f"✅ Selected: [bold green]{platform}[/bold green]")
    else:
        platform = Prompt.ask("🎯 [bold]Choose platform[/bold]", choices=platforms, default="esp32")
    
    # Step 3: Compatibility check
    console.print(f"\n🔍 [yellow]Checking compatibility...[/yellow]")
    
    profiler = QuickStartProfiler()
    compat_result = profiler.check_model_compatibility(model_path, platform)
    
    if not compat_result["compatible"]:
        console.print(f"❌ [bold red]Not compatible:[/bold red] {compat_result['reason']}")
        console.print("Recommendations:")
        for rec in compat_result.get("recommendations", []):
            console.print(f"  💡 {rec}")
        
        if not Confirm.ask("Continue anyway?", default=False):
            return
    else:
        console.print("✅ [bold green]Compatible![/bold green]")
    
    # Step 4: Run profiling
    run_profiling = Confirm.ask("\n🚀 Run quick profiling?", default=True)
    
    if run_profiling:
        console.print("\n⏳ [yellow]Running profiling...[/yellow]")
        results = profiler.quick_profile(model_path, platform)
        display_results(results, verbose=True)
        
        # Save results?
        save_results_prompt = Confirm.ask("\n💾 Save results to file?", default=False)
        if save_results_prompt:
            output_file = Prompt.ask("📝 Output filename", default="profile_results.json")
            save_results(results, output_file)
            console.print(f"✅ Saved to: [bold blue]{output_file}[/bold blue]")
    
    console.print("\n🎉 [bold green]Setup complete![/bold green]")
    console.print("💡 Next time, use: [bold]tiny-profiler-quickstart quick-profile[/bold] for instant profiling")


def display_results(results: dict, verbose: bool = False):
    """Display profiling results in a nice format."""
    
    if results["status"] == "success":
        perf = results["performance"]
        
        # Main results table
        table = Table(title="🎯 Profiling Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold green")
        table.add_column("Assessment", style="yellow")
        
        # Add performance metrics
        table.add_row(
            "Tokens/Second",
            f"{perf['tokens_per_second']:.2f}",
            "🚀 Excellent" if perf['tokens_per_second'] > 10 else "🟡 Good" if perf['tokens_per_second'] > 5 else "🔴 Needs optimization"
        )
        
        table.add_row(
            "Latency",
            f"{perf['latency_ms']:.1f} ms",
            "⚡ Fast" if perf['latency_ms'] < 100 else "🟡 Acceptable" if perf['latency_ms'] < 500 else "⏱️ Slow"
        )
        
        table.add_row(
            "Memory Usage",
            f"{perf['memory_usage_kb']:.0f} KB",
            "✅ Efficient" if perf['memory_usage_kb'] < 200 else "🟡 Moderate" if perf['memory_usage_kb'] < 400 else "⚠️ High"
        )
        
        table.add_row(
            "Power Consumption",
            f"{perf['power_consumption_mw']:.0f} mW",
            "🔋 Low" if perf['power_consumption_mw'] < 100 else "🟡 Moderate" if perf['power_consumption_mw'] < 200 else "⚠️ High"
        )
        
        console.print(table)
        
        # Recommendations
        if "recommendations" in results:
            console.print("\n🎯 [bold blue]Recommendations:[/bold blue]")
            for rec in results["recommendations"]:
                console.print(f"  {rec}")
        
        # Verbose details
        if verbose and "compatibility" in results:
            compat = results["compatibility"]
            console.print(f"\n📊 [bold]Model Details:[/bold]")
            console.print(f"  Size: {compat.get('model_size_mb', 0):.2f} MB")
            console.print(f"  Platform: {results['platform']}")
            console.print(f"  Flash Usage: {compat.get('utilization', {}).get('flash_percent', 0):.1f}%")
            console.print(f"  Estimated RAM Usage: {compat.get('utilization', {}).get('estimated_ram_percent', 0):.1f}%")
    
    elif results["status"] == "incompatible":
        console.print("❌ [bold red]Model Incompatible[/bold red]")
        console.print(f"Reason: {results['compatibility']['reason']}")
        console.print("\n💡 [bold]Suggestions:[/bold]")
        for rec in results.get("recommendations", []):
            console.print(f"  • {rec}")
    
    else:  # error
        console.print("❌ [bold red]Profiling Failed[/bold red]")
        console.print(f"Error: {results.get('error', 'Unknown error')}")


def save_results(results: dict, output_path: str):
    """Save results to JSON file."""
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        console.print(f"❌ [bold red]Failed to save results:[/bold red] {e}")


def show_next_steps(results: dict, platform: str):
    """Show helpful next steps based on results."""
    console.print(f"\n🎯 [bold blue]Next Steps:[/bold blue]")
    
    if results["status"] == "success":
        perf = results["performance"]
        
        if perf["tokens_per_second"] > 8:
            console.print("  🚀 Great performance! Ready for deployment")
            console.print(f"  📖 Check out the deployment guide: [bold]tiny-profiler-quickstart guide {platform}[/bold]")
        else:
            console.print("  🔧 Consider optimization:")
            console.print(f"  📖 Get optimization tips: [bold]tiny-profiler-quickstart guide {platform}[/bold]")
            console.print("  🔍 Try different quantization levels")
        
        console.print(f"  📊 Run full analysis: [bold]tiny-profiler profile --platform {platform} [MODEL][/bold]")
        
    else:
        console.print("  🔍 Try compatibility check: [bold]tiny-profiler-quickstart check [MODEL][/bold]")
        console.print("  🔧 Use model optimization tools")
        console.print("  📖 Read getting started guide: [bold]tiny-profiler-quickstart guide[/bold]")


if __name__ == "__main__":
    quickstart_cli()