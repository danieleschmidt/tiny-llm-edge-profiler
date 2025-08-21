#!/usr/bin/env python3
"""
Experimental Reproduction Script

This script reproduces all experimental results from the breakthrough research paper.
Supports full experimental suite, individual algorithms, and platform-specific testing.

Usage:
    python scripts/reproduce_experiments.py --full-suite
    python scripts/reproduce_experiments.py --platform esp32 --algorithm HAQIP
    python scripts/reproduce_experiments.py --validate-statistics --generate-figures
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Import breakthrough algorithms
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.tiny_llm_profiler.breakthrough_research_algorithms import (
    HardwareProfile,
    HardwareArchitecture,
    BreakthroughProfilingEngine,
    run_breakthrough_research_experiment,
    compare_breakthrough_vs_traditional
)
from src.tiny_llm_profiler.experimental_validation_engine import (
    ExperimentalValidationEngine,
    ValidationConfiguration,
    ValidationMethod,
    assess_research_quality_gates,
    validate_breakthrough_experiment
)


class ExperimentalReproducer:
    """
    Main class for reproducing experimental results from the research paper.
    """
    
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.results_dir = Path("experimental_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Experimental configuration
        self.hardware_platforms = self._get_hardware_platforms()
        self.algorithms = ['HAQIP', 'AEPCO', 'MOPEP']
        self.replications = args.replications
        
        # Results storage
        self.experimental_results = {}
        self.statistical_results = {}
        self.validation_results = {}
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        handlers = [logging.StreamHandler()]
        if self.args.log_file:
            handlers.append(logging.FileHandler(self.args.log_file))
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _get_hardware_platforms(self) -> Dict[str, HardwareProfile]:
        """Get hardware platform configurations used in the paper."""
        platforms = {
            'esp32': HardwareProfile(
                architecture=HardwareArchitecture.ESP32_XTENSA,
                clock_frequency_mhz=240,
                ram_kb=520,
                flash_kb=4096,
                cache_kb=64,
                fpu_available=True,
                simd_available=False,
                power_domain_count=2,
                thermal_design_power_mw=500,
                voltage_domains=[3.3, 1.8],
                instruction_sets=["Xtensa", "LX6"]
            ),
            'stm32f7': HardwareProfile(
                architecture=HardwareArchitecture.STM32F7,
                clock_frequency_mhz=480,
                ram_kb=1024,
                flash_kb=2048,
                cache_kb=128,
                fpu_available=True,
                simd_available=True,
                power_domain_count=4,
                thermal_design_power_mw=800,
                voltage_domains=[3.3, 1.8, 1.2],
                instruction_sets=["ARMv7", "Thumb-2", "DSP"]
            ),
            'rp2040': HardwareProfile(
                architecture=HardwareArchitecture.RP2040,
                clock_frequency_mhz=133,
                ram_kb=264,
                flash_kb=2048,
                cache_kb=16,
                fpu_available=False,
                simd_available=False,
                power_domain_count=1,
                thermal_design_power_mw=150,
                voltage_domains=[3.3],
                instruction_sets=["ARMv6", "Thumb"]
            ),
            'riscv': HardwareProfile(
                architecture=HardwareArchitecture.RISC_V_RV32,
                clock_frequency_mhz=320,
                ram_kb=512,
                flash_kb=1024,
                cache_kb=32,
                fpu_available=False,
                simd_available=False,
                power_domain_count=1,
                thermal_design_power_mw=300,
                voltage_domains=[3.3],
                instruction_sets=["RV32I", "RV32M"]
            )
        }
        
        # Filter based on command line arguments
        if self.args.platform:
            if self.args.platform in platforms:
                return {self.args.platform: platforms[self.args.platform]}
            else:
                raise ValueError(f"Unknown platform: {self.args.platform}")
        
        return platforms
    
    async def run_full_experimental_suite(self):
        """Run the complete experimental suite from the paper."""
        self.logger.info("🚀 Starting full experimental suite reproduction")
        start_time = time.time()
        
        # Run experiments for each platform
        for platform_name, hardware_profile in self.hardware_platforms.items():
            self.logger.info(f"📱 Testing platform: {platform_name}")
            
            platform_results = await self._run_platform_experiments(
                platform_name, hardware_profile
            )
            self.experimental_results[platform_name] = platform_results
        
        # Statistical validation
        if self.args.validate_statistics:
            self.logger.info("📊 Running statistical validation")
            await self._run_statistical_validation()
        
        # Generate figures
        if self.args.generate_figures:
            self.logger.info("📈 Generating figures")
            await self._generate_figures()
        
        # Save results
        await self._save_results()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"✅ Experimental suite completed in {elapsed_time:.2f} seconds")
        
        return self.experimental_results
    
    async def _run_platform_experiments(
        self, 
        platform_name: str, 
        hardware_profile: HardwareProfile
    ) -> Dict[str, Any]:
        """Run experiments for a specific platform."""
        platform_results = {
            'hardware_profile': hardware_profile,
            'algorithm_results': {},
            'replication_results': {},
            'summary_statistics': {}
        }
        
        # Run multiple replications
        all_replications = []
        
        for replication in range(self.replications):
            self.logger.info(f"  Replication {replication + 1}/{self.replications}")
            
            # Set random seed for reproducibility
            np.random.seed(42 + replication)
            
            experiment_config = {
                'experiment_name': f'{platform_name}_replication_{replication}',
                'replication_id': replication,
                'platform': platform_name,
                'algorithms': self.algorithms if not self.args.algorithm else [self.args.algorithm],
                'iterations': 20 if self.args.quick else 100,
                'statistical_validation': True
            }
            
            # Run breakthrough experiment
            replication_result = await run_breakthrough_research_experiment(
                hardware_profile, experiment_config
            )
            
            all_replications.append(replication_result)
            
            # Log intermediate results
            if 'algorithm_results' in replication_result:
                for alg, results in replication_result['algorithm_results'].items():
                    score = results.get('optimal_score', 'N/A')
                    time_taken = results.get('execution_time', 'N/A')
                    self.logger.debug(f"    {alg}: score={score}, time={time_taken}s")
        
        # Aggregate replication results
        platform_results['replication_results'] = all_replications
        platform_results['algorithm_results'] = self._aggregate_algorithm_results(all_replications)
        platform_results['summary_statistics'] = self._calculate_summary_statistics(all_replications)
        
        return platform_results
    
    def _aggregate_algorithm_results(self, replications: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across replications."""
        aggregated = {}
        
        # Extract algorithm results from all replications
        for replication in replications:
            if 'algorithm_results' not in replication:
                continue
                
            for algorithm, results in replication['algorithm_results'].items():
                if algorithm not in aggregated:
                    aggregated[algorithm] = {
                        'optimal_scores': [],
                        'execution_times': [],
                        'quantum_advantages': [],
                        'convergence_histories': []
                    }
                
                if 'optimal_score' in results:
                    aggregated[algorithm]['optimal_scores'].append(results['optimal_score'])
                if 'execution_time' in results:
                    aggregated[algorithm]['execution_times'].append(results['execution_time'])
                if 'quantum_advantage_factor' in results:
                    aggregated[algorithm]['quantum_advantages'].append(results['quantum_advantage_factor'])
                if 'convergence_history' in results:
                    aggregated[algorithm]['convergence_histories'].append(results['convergence_history'])
        
        # Calculate aggregate statistics
        for algorithm in aggregated:
            alg_data = aggregated[algorithm]
            
            # Calculate mean and std for each metric
            for metric in ['optimal_scores', 'execution_times', 'quantum_advantages']:
                if alg_data[metric]:
                    values = alg_data[metric]
                    alg_data[f'{metric}_mean'] = np.mean(values)
                    alg_data[f'{metric}_std'] = np.std(values)
                    alg_data[f'{metric}_ci_95'] = self._calculate_confidence_interval(values, 0.95)
        
        return aggregated
    
    def _calculate_summary_statistics(self, replications: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics across all replications."""
        summary = {
            'total_replications': len(replications),
            'successful_replications': 0,
            'average_execution_time': 0.0,
            'reproducibility_metrics': {},
            'quality_assessment': {}
        }
        
        # Count successful replications
        successful_replications = [
            r for r in replications 
            if 'algorithm_results' in r and len(r['algorithm_results']) > 0
        ]
        summary['successful_replications'] = len(successful_replications)
        
        # Calculate average execution times
        if successful_replications:
            total_times = []
            for replication in successful_replications:
                replication_time = 0
                for alg_results in replication['algorithm_results'].values():
                    if 'execution_time' in alg_results:
                        replication_time += alg_results['execution_time']
                total_times.append(replication_time)
            
            summary['average_execution_time'] = np.mean(total_times)
        
        # Calculate reproducibility metrics
        for algorithm in self.algorithms:
            if self.args.algorithm and algorithm != self.args.algorithm:
                continue
                
            scores = []
            for replication in successful_replications:
                if (algorithm in replication.get('algorithm_results', {}) and
                    'optimal_score' in replication['algorithm_results'][algorithm]):
                    scores.append(replication['algorithm_results'][algorithm]['optimal_score'])
            
            if scores:
                cv = np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else float('inf')
                summary['reproducibility_metrics'][algorithm] = {
                    'coefficient_of_variation': cv,
                    'reproducibility_quality': (
                        'Excellent' if cv < 0.1 else 
                        'Good' if cv < 0.2 else 
                        'Acceptable' if cv < 0.3 else 
                        'Poor'
                    )
                }
        
        return summary
    
    def _calculate_confidence_interval(self, values: List[float], confidence_level: float) -> tuple:
        """Calculate confidence interval for a list of values."""
        if not values:
            return (0.0, 0.0)
        
        values_array = np.array(values)
        alpha = 1 - confidence_level
        
        mean = np.mean(values_array)
        std_error = np.std(values_array, ddof=1) / np.sqrt(len(values_array))
        
        # Use t-distribution for small samples
        from scipy import stats
        t_critical = stats.t.ppf(1 - alpha/2, len(values_array) - 1)
        margin_error = t_critical * std_error
        
        return (mean - margin_error, mean + margin_error)
    
    async def _run_statistical_validation(self):
        """Run comprehensive statistical validation."""
        self.logger.info("Running statistical validation of experimental results")
        
        for platform_name, platform_results in self.experimental_results.items():
            self.logger.info(f"  Validating {platform_name} results")
            
            # Extract experimental data for validation
            experimental_data = {}
            baseline_data = {}
            
            aggregated_results = platform_results['algorithm_results']
            
            for algorithm, results in aggregated_results.items():
                if 'optimal_scores' in results and results['optimal_scores']:
                    experimental_data[algorithm] = results['optimal_scores']
                    # Use a baseline score (simulated traditional method performance)
                    baseline_score = 100.0  # Normalized baseline
                    baseline_data[algorithm] = [baseline_score] * len(results['optimal_scores'])
            
            if experimental_data:
                # Configure validation
                validation_config = ValidationConfiguration(
                    validation_method=ValidationMethod.BOOTSTRAP,
                    n_bootstrap_samples=1000,
                    confidence_level=0.95,
                    min_effect_size=0.5,
                    max_p_value=0.05
                )
                
                # Run validation
                validation_engine = ExperimentalValidationEngine(validation_config)
                
                # Create experimental conditions
                from src.tiny_llm_profiler.experimental_validation_engine import ExperimentalCondition
                conditions = [
                    ExperimentalCondition(
                        name=algorithm,
                        parameters={'platform': platform_name},
                        control_group=False
                    ) for algorithm in experimental_data.keys()
                ]
                
                validation_report = await validation_engine.validate_experimental_results(
                    experimental_data, baseline_data, conditions
                )
                
                # Assess quality gates
                quality_assessment = assess_research_quality_gates(validation_report)
                
                self.statistical_results[platform_name] = {
                    'validation_report': validation_report,
                    'quality_assessment': quality_assessment
                }
                
                # Log key findings
                overall_quality = quality_assessment['overall_status']
                passed_gates = quality_assessment['passed_gates']
                total_gates = quality_assessment['total_gates']
                
                self.logger.info(f"    Quality: {overall_quality} ({passed_gates}/{total_gates} gates passed)")
    
    async def _generate_figures(self):
        """Generate figures reproducing those in the paper."""
        self.logger.info("Generating publication figures")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set publication style
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
            
            figures_dir = Path("figures")
            figures_dir.mkdir(exist_ok=True)
            
            # Figure 1: Performance Comparison
            await self._generate_performance_comparison_figure(figures_dir)
            
            # Figure 2: Statistical Validation
            if self.statistical_results:
                await self._generate_statistical_validation_figure(figures_dir)
            
            # Figure 3: Platform Scalability
            if len(self.experimental_results) > 1:
                await self._generate_platform_scalability_figure(figures_dir)
            
            # Figure 4: Reproducibility Analysis
            await self._generate_reproducibility_figure(figures_dir)
            
            self.logger.info(f"  Figures saved to {figures_dir}/")
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping figure generation")
    
    async def _generate_performance_comparison_figure(self, figures_dir: Path):
        """Generate performance comparison figure."""
        import matplotlib.pyplot as plt
        
        # Aggregate data across platforms
        algorithm_performance = {}
        
        for platform_name, platform_results in self.experimental_results.items():
            aggregated = platform_results['algorithm_results']
            
            for algorithm, results in aggregated.items():
                if algorithm not in algorithm_performance:
                    algorithm_performance[algorithm] = []
                
                if 'optimal_scores_mean' in results:
                    # Calculate improvement percentage (lower score is better)
                    baseline_score = 100.0
                    improvement = (baseline_score - results['optimal_scores_mean']) / baseline_score * 100
                    algorithm_performance[algorithm].append(improvement)
        
        # Calculate mean improvements
        mean_improvements = {}
        std_improvements = {}
        
        for algorithm, improvements in algorithm_performance.items():
            if improvements:
                mean_improvements[algorithm] = np.mean(improvements)
                std_improvements[algorithm] = np.std(improvements)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = list(mean_improvements.keys())
        improvements = [mean_improvements[alg] for alg in algorithms]
        errors = [std_improvements[alg] for alg in algorithms]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(algorithms)]
        bars = ax.bar(algorithms, improvements, yerr=errors, capsize=5, color=colors)
        
        ax.set_ylabel('Performance Improvement (%)')
        ax.set_title('Algorithm Performance Comparison\n(Higher is Better)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax.annotate(f'{improvement:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'performance_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(figures_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    async def _generate_statistical_validation_figure(self, figures_dir: Path):
        """Generate statistical validation figure."""
        import matplotlib.pyplot as plt
        
        # Collect statistical data
        p_values = {}
        effect_sizes = {}
        
        for platform_name, stats in self.statistical_results.items():
            validation_report = stats['validation_report']
            
            if 'statistical_tests' in validation_report:
                for condition, tests in validation_report['statistical_tests'].items():
                    for test_type, result in tests.items():
                        if condition not in p_values:
                            p_values[condition] = []
                            effect_sizes[condition] = []
                        
                        p_values[condition].append(result.p_value)
                        effect_sizes[condition].append(result.effect_size)
        
        # Calculate means
        mean_p_values = {alg: np.mean(vals) for alg, vals in p_values.items() if vals}
        mean_effect_sizes = {alg: np.mean(vals) for alg, vals in effect_sizes.items() if vals}
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # P-values (log scale)
        algorithms = list(mean_p_values.keys())
        p_vals = [mean_p_values[alg] for alg in algorithms]
        
        bars1 = ax1.bar(algorithms, [-np.log10(p) for p in p_vals], 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(algorithms)])
        ax1.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='α = 0.05')
        ax1.set_ylabel('-log₁₀(p-value)')
        ax1.set_title('Statistical Significance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Effect sizes
        effect_vals = [mean_effect_sizes[alg] for alg in algorithms]
        bars2 = ax2.bar(algorithms, effect_vals, 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(algorithms)])
        ax2.axhline(y=0.8, color='red', linestyle='--', label='Large effect')
        ax2.axhline(y=0.5, color='orange', linestyle='--', label='Medium effect')
        ax2.set_ylabel("Cohen's d")
        ax2.set_title('Effect Size Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'statistical_validation.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(figures_dir / 'statistical_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    async def _generate_platform_scalability_figure(self, figures_dir: Path):
        """Generate platform scalability figure."""
        import matplotlib.pyplot as plt
        
        # Prepare data
        platforms = list(self.experimental_results.keys())
        algorithms = self.algorithms
        
        # Create performance matrix
        performance_matrix = np.zeros((len(algorithms), len(platforms)))
        
        for j, platform in enumerate(platforms):
            platform_results = self.experimental_results[platform]['algorithm_results']
            
            for i, algorithm in enumerate(algorithms):
                if algorithm in platform_results and 'optimal_scores_mean' in platform_results[algorithm]:
                    baseline_score = 100.0
                    score = platform_results[algorithm]['optimal_scores_mean']
                    improvement = (baseline_score - score) / baseline_score * 100
                    performance_matrix[i, j] = improvement
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        import seaborn as sns
        sns.heatmap(performance_matrix, 
                   xticklabels=platforms, 
                   yticklabels=algorithms,
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlGn',
                   ax=ax)
        
        ax.set_title('Algorithm Performance Across Platforms\n(% Improvement)')
        ax.set_xlabel('Hardware Platform')
        ax.set_ylabel('Algorithm')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'platform_scalability.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(figures_dir / 'platform_scalability.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    async def _generate_reproducibility_figure(self, figures_dir: Path):
        """Generate reproducibility analysis figure."""
        import matplotlib.pyplot as plt
        
        # Collect reproducibility data
        reproducibility_data = {}
        
        for platform_name, platform_results in self.experimental_results.items():
            summary_stats = platform_results['summary_statistics']
            
            if 'reproducibility_metrics' in summary_stats:
                for algorithm, metrics in summary_stats['reproducibility_metrics'].items():
                    if algorithm not in reproducibility_data:
                        reproducibility_data[algorithm] = []
                    
                    cv = metrics['coefficient_of_variation']
                    reproducibility_data[algorithm].append(cv)
        
        # Calculate mean reproducibility
        mean_cv = {alg: np.mean(cvs) for alg, cvs in reproducibility_data.items() if cvs}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        algorithms = list(mean_cv.keys())
        cv_values = [mean_cv[alg] for alg in algorithms]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(algorithms)]
        bars = ax.bar(algorithms, cv_values, color=colors)
        
        # Add quality thresholds
        ax.axhline(y=0.1, color='green', linestyle='--', label='Excellent (CV < 0.1)')
        ax.axhline(y=0.2, color='orange', linestyle='--', label='Good (CV < 0.2)')
        ax.axhline(y=0.3, color='red', linestyle='--', label='Acceptable (CV < 0.3)')
        
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Reproducibility Analysis\n(Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, cv in zip(bars, cv_values):
            height = bar.get_height()
            ax.annotate(f'{cv:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'reproducibility_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(figures_dir / 'reproducibility_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    async def _save_results(self):
        """Save all experimental results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save experimental results
        results_file = self.results_dir / f"experimental_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.experimental_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Experimental results saved to {results_file}")
        
        # Save statistical results
        if self.statistical_results:
            stats_file = self.results_dir / f"statistical_results_{timestamp}.json"
            serializable_stats = self._make_json_serializable(self.statistical_results)
            
            with open(stats_file, 'w') as f:
                json.dump(serializable_stats, f, indent=2)
            
            self.logger.info(f"Statistical results saved to {stats_file}")
        
        # Generate summary report
        await self._generate_summary_report(timestamp)
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            # Handle custom objects
            return {key: self._make_json_serializable(value) for key, value in obj.__dict__.items()}
        else:
            return obj
    
    async def _generate_summary_report(self, timestamp: str):
        """Generate a human-readable summary report."""
        report_file = self.results_dir / f"summary_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Experimental Reproduction Summary Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Replications**: {self.replications}\n")
            f.write(f"**Platforms**: {list(self.hardware_platforms.keys())}\n")
            f.write(f"**Algorithms**: {self.algorithms}\n\n")
            
            # Results summary
            f.write("## Results Summary\n\n")
            
            for platform_name, platform_results in self.experimental_results.items():
                f.write(f"### Platform: {platform_name}\n\n")
                
                summary_stats = platform_results['summary_statistics']
                f.write(f"- **Successful replications**: {summary_stats['successful_replications']}/{summary_stats['total_replications']}\n")
                f.write(f"- **Average execution time**: {summary_stats['average_execution_time']:.2f}s\n\n")
                
                # Algorithm results
                algorithm_results = platform_results['algorithm_results']
                f.write("#### Algorithm Performance\n\n")
                f.write("| Algorithm | Mean Score | Std Dev | 95% CI | Reproducibility |\n")
                f.write("|-----------|------------|---------|--------|-----------------|\n")
                
                for algorithm, results in algorithm_results.items():
                    mean_score = results.get('optimal_scores_mean', 'N/A')
                    std_score = results.get('optimal_scores_std', 'N/A')
                    ci = results.get('optimal_scores_ci_95', ('N/A', 'N/A'))
                    
                    # Get reproducibility
                    repro_metrics = summary_stats.get('reproducibility_metrics', {})
                    repro = repro_metrics.get(algorithm, {}).get('reproducibility_quality', 'N/A')
                    
                    f.write(f"| {algorithm} | {mean_score:.2f} | {std_score:.2f} | ({ci[0]:.2f}, {ci[1]:.2f}) | {repro} |\n")
                
                f.write("\n")
            
            # Statistical validation summary
            if self.statistical_results:
                f.write("## Statistical Validation\n\n")
                
                for platform_name, stats in self.statistical_results.items():
                    quality_assessment = stats['quality_assessment']
                    f.write(f"### {platform_name}\n\n")
                    f.write(f"- **Overall Quality**: {quality_assessment['overall_status']}\n")
                    f.write(f"- **Quality Gates Passed**: {quality_assessment['passed_gates']}/{quality_assessment['total_gates']}\n\n")
            
            # Reproducibility summary
            f.write("## Reproducibility Assessment\n\n")
            
            for platform_name, platform_results in self.experimental_results.items():
                summary_stats = platform_results['summary_statistics']
                repro_metrics = summary_stats.get('reproducibility_metrics', {})
                
                f.write(f"### {platform_name}\n\n")
                for algorithm, metrics in repro_metrics.items():
                    cv = metrics['coefficient_of_variation']
                    quality = metrics['reproducibility_quality']
                    f.write(f"- **{algorithm}**: CV = {cv:.4f} ({quality})\n")
                f.write("\n")
        
        self.logger.info(f"Summary report saved to {report_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce experimental results from breakthrough research paper"
    )
    
    # Execution modes
    parser.add_argument('--full-suite', action='store_true',
                       help='Run complete experimental suite')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick version with fewer iterations')
    
    # Platform and algorithm selection
    parser.add_argument('--platform', type=str, choices=['esp32', 'stm32f7', 'rp2040', 'riscv'],
                       help='Run experiments for specific platform only')
    parser.add_argument('--algorithm', type=str, choices=['HAQIP', 'AEPCO', 'MOPEP'],
                       help='Run experiments for specific algorithm only')
    
    # Experimental parameters
    parser.add_argument('--replications', type=int, default=10,
                       help='Number of experimental replications (default: 10)')
    parser.add_argument('--parallel-jobs', type=int, default=4,
                       help='Number of parallel jobs (default: 4)')
    
    # Validation and output
    parser.add_argument('--validate-statistics', action='store_true',
                       help='Run comprehensive statistical validation')
    parser.add_argument('--generate-figures', action='store_true',
                       help='Generate publication figures')
    
    # Logging and output
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str,
                       help='Save logs to file')
    
    return parser.parse_args()


async def main():
    """Main entry point for experimental reproduction."""
    args = parse_arguments()
    
    # Create reproducer
    reproducer = ExperimentalReproducer(args)
    
    # Run experiments
    if args.full_suite or (not args.platform and not args.algorithm):
        results = await reproducer.run_full_experimental_suite()
    else:
        # Run subset of experiments
        reproducer.logger.info("Running subset of experiments")
        results = await reproducer.run_full_experimental_suite()
    
    print("\n🎉 Experimental reproduction completed successfully!")
    print(f"📊 Results saved to: {reproducer.results_dir}/")
    
    if args.generate_figures:
        print(f"📈 Figures saved to: figures/")
    
    return results


if __name__ == "__main__":
    # Run experimental reproduction
    results = asyncio.run(main())