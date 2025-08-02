#!/usr/bin/env python3
"""
Performance tracking and benchmarking automation.

Tracks performance metrics, runs benchmarks, and monitors
performance trends for the tiny-llm-edge-profiler project.
"""

import json
import time
import psutil
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import statistics
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class BenchmarkResult:
    """Performance benchmark result"""
    name: str
    duration_seconds: float
    memory_peak_mb: float
    cpu_percent: float
    success: bool
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io_mb: Tuple[float, float]  # (sent, received)
    timestamp: datetime


class PerformanceTracker:
    """Track and analyze performance metrics"""
    
    def __init__(self, data_dir: str = "performance_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        
        # Initialize performance history
        self.benchmarks_file = self.data_dir / "benchmarks.json"
        self.system_metrics_file = self.data_dir / "system_metrics.json"
        self.trends_file = self.data_dir / "trends.json"
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        # CPU usage (average over 1 second)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io_mb = (
            network.bytes_sent / (1024 * 1024),
            network.bytes_recv / (1024 * 1024)
        )
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            network_io_mb=network_io_mb,
            timestamp=datetime.utcnow()
        )
    
    def run_benchmark(self, name: str, command: List[str], 
                     timeout: int = 300, metadata: Dict[str, Any] = None) -> BenchmarkResult:
        """Run a performance benchmark"""
        self.logger.info(f"Running benchmark: {name}")
        
        if metadata is None:
            metadata = {}
        
        # Record initial system state
        initial_metrics = self.collect_system_metrics()
        
        start_time = time.time()
        success = False
        peak_memory = 0
        cpu_samples = []
        
        try:
            # Start the process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor the process
            while process.poll() is None:
                try:
                    # Monitor resource usage
                    proc_info = psutil.Process(process.pid)
                    memory_mb = proc_info.memory_info().rss / (1024 * 1024)
                    cpu_percent = proc_info.cpu_percent()
                    
                    peak_memory = max(peak_memory, memory_mb)
                    cpu_samples.append(cpu_percent)
                    
                    time.sleep(0.1)  # Sample every 100ms
                    
                    # Check for timeout
                    if time.time() - start_time > timeout:
                        process.kill()
                        raise subprocess.TimeoutExpired(command, timeout)
                        
                except psutil.NoSuchProcess:
                    break
            
            # Wait for process completion
            stdout, stderr = process.communicate(timeout=10)
            
            success = process.returncode == 0
            if not success:
                self.logger.error(f"Benchmark failed: {stderr}")
                metadata['error'] = stderr
            else:
                metadata['stdout'] = stdout
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Benchmark timed out: {name}")
            metadata['error'] = 'Timeout expired'
            try:
                process.kill()
            except:
                pass
        except Exception as e:
            self.logger.error(f"Benchmark error: {e}")
            metadata['error'] = str(e)
        
        duration = time.time() - start_time
        avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
        
        result = BenchmarkResult(
            name=name,
            duration_seconds=duration,
            memory_peak_mb=peak_memory,
            cpu_percent=avg_cpu,
            success=success,
            timestamp=datetime.utcnow(),
            metadata=metadata
        )
        
        self.logger.info(f"Benchmark completed: {name} - Success: {success}, Duration: {duration:.2f}s")
        return result
    
    def run_standard_benchmarks(self) -> List[BenchmarkResult]:
        """Run standard performance benchmarks"""
        benchmarks = []
        
        # Test suite benchmark
        if Path('tests').exists():
            benchmarks.append(self.run_benchmark(
                "test_suite",
                ["python", "-m", "pytest", "tests/", "-v"],
                timeout=600,
                metadata={"type": "testing", "scope": "full"}
            ))
        
        # Build benchmark
        if Path('setup.py').exists() or Path('pyproject.toml').exists():
            benchmarks.append(self.run_benchmark(
                "build",
                ["python", "-m", "pip", "install", "-e", "."],
                timeout=300,
                metadata={"type": "build", "scope": "development"}
            ))
        
        # Docker build benchmark (if Dockerfile exists)
        if Path('Dockerfile').exists():
            benchmarks.append(self.run_benchmark(
                "docker_build",
                ["docker", "build", "-t", "tiny-llm-profiler:benchmark", "."],
                timeout=900,
                metadata={"type": "containerization", "scope": "full"}
            ))
        
        # Linting benchmark
        linting_commands = [
            (["python", "-m", "flake8", "src/"], "flake8"),
            (["python", "-m", "black", "--check", "src/"], "black"),
            (["python", "-m", "isort", "--check-only", "src/"], "isort")
        ]
        
        for command, tool in linting_commands:
            try:
                benchmarks.append(self.run_benchmark(
                    f"lint_{tool}",
                    command,
                    timeout=120,
                    metadata={"type": "code_quality", "tool": tool}
                ))
            except:
                self.logger.warning(f"Skipping {tool} benchmark - tool not available")
        
        return benchmarks
    
    def save_benchmarks(self, benchmarks: List[BenchmarkResult]) -> None:
        """Save benchmark results to file"""
        # Load existing benchmarks
        existing_benchmarks = []
        if self.benchmarks_file.exists():
            with open(self.benchmarks_file, 'r') as f:
                existing_data = json.load(f)
                existing_benchmarks = [
                    BenchmarkResult(
                        name=b['name'],
                        duration_seconds=b['duration_seconds'],
                        memory_peak_mb=b['memory_peak_mb'],
                        cpu_percent=b['cpu_percent'],
                        success=b['success'],
                        timestamp=datetime.fromisoformat(b['timestamp']),
                        metadata=b['metadata']
                    ) for b in existing_data
                ]
        
        # Add new benchmarks
        all_benchmarks = existing_benchmarks + benchmarks
        
        # Keep only last 100 results per benchmark type
        benchmark_groups = {}
        for benchmark in all_benchmarks:
            if benchmark.name not in benchmark_groups:
                benchmark_groups[benchmark.name] = []
            benchmark_groups[benchmark.name].append(benchmark)
        
        # Sort by timestamp and keep last 100
        filtered_benchmarks = []
        for name, group in benchmark_groups.items():
            sorted_group = sorted(group, key=lambda x: x.timestamp)
            filtered_benchmarks.extend(sorted_group[-100:])
        
        # Save to file
        benchmark_data = []
        for benchmark in filtered_benchmarks:
            data = asdict(benchmark)
            data['timestamp'] = benchmark.timestamp.isoformat()
            benchmark_data.append(data)
        
        with open(self.benchmarks_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        self.logger.info(f"Saved {len(benchmarks)} new benchmark results")
    
    def analyze_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends over specified period"""
        if not self.benchmarks_file.exists():
            return {"error": "No benchmark data available"}
        
        # Load benchmark data
        with open(self.benchmarks_file, 'r') as f:
            benchmark_data = json.load(f)
        
        # Filter to specified time period
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_benchmarks = [
            b for b in benchmark_data
            if datetime.fromisoformat(b['timestamp']) >= cutoff_date
        ]
        
        if not recent_benchmarks:
            return {"error": f"No benchmark data available for last {days} days"}
        
        # Group by benchmark name
        trends = {}
        for benchmark in recent_benchmarks:
            name = benchmark['name']
            if name not in trends:
                trends[name] = {
                    'durations': [],
                    'memory_peaks': [],
                    'cpu_usage': [],
                    'success_rate': [],
                    'timestamps': []
                }
            
            trends[name]['durations'].append(benchmark['duration_seconds'])
            trends[name]['memory_peaks'].append(benchmark['memory_peak_mb'])
            trends[name]['cpu_usage'].append(benchmark['cpu_percent'])
            trends[name]['success_rate'].append(1 if benchmark['success'] else 0)
            trends[name]['timestamps'].append(benchmark['timestamp'])
        
        # Calculate trend analysis
        analysis = {}
        for name, data in trends.items():
            if len(data['durations']) < 2:
                continue
                
            analysis[name] = {
                'duration': {
                    'mean': statistics.mean(data['durations']),
                    'median': statistics.median(data['durations']),
                    'std_dev': statistics.stdev(data['durations']) if len(data['durations']) > 1 else 0,
                    'trend': self._calculate_trend(data['durations'])
                },
                'memory': {
                    'mean': statistics.mean(data['memory_peaks']),
                    'median': statistics.median(data['memory_peaks']),
                    'std_dev': statistics.stdev(data['memory_peaks']) if len(data['memory_peaks']) > 1 else 0,
                    'trend': self._calculate_trend(data['memory_peaks'])
                },
                'cpu': {
                    'mean': statistics.mean(data['cpu_usage']),
                    'median': statistics.median(data['cpu_usage']),
                    'std_dev': statistics.stdev(data['cpu_usage']) if len(data['cpu_usage']) > 1 else 0,
                    'trend': self._calculate_trend(data['cpu_usage'])
                },
                'success_rate': {
                    'current': statistics.mean(data['success_rate'][-5:]) if len(data['success_rate']) >= 5 else statistics.mean(data['success_rate']),
                    'overall': statistics.mean(data['success_rate']),
                    'trend': self._calculate_trend(data['success_rate'])
                },
                'data_points': len(data['durations']),
                'time_span_days': (datetime.fromisoformat(max(data['timestamps'])) - 
                                 datetime.fromisoformat(min(data['timestamps']))).days
            }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction (improving, degrading, stable)"""
        if len(values) < 3:
            return "insufficient_data"
        
        # Use linear regression slope to determine trend
        x = list(range(len(values)))
        y = values
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x_squared = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
        
        # Determine trend based on slope
        if abs(slope) < 0.01:  # Threshold for stability
            return "stable"
        elif slope > 0:
            return "degrading"  # For duration/memory/cpu, increasing is degrading
        else:
            return "improving"
    
    def generate_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        self.logger.info(f"Generating performance report for last {days} days")
        
        trends = self.analyze_performance_trends(days)
        
        if "error" in trends:
            return trends
        
        # Calculate overall health score
        health_score = 100
        issues = []
        recommendations = []
        
        for benchmark_name, analysis in trends.items():
            # Check for degrading trends
            if analysis['duration']['trend'] == 'degrading':
                health_score -= 10
                issues.append(f"{benchmark_name}: Duration performance degrading")
                recommendations.append(f"Investigate performance regression in {benchmark_name}")
            
            if analysis['memory']['trend'] == 'degrading':
                health_score -= 5
                issues.append(f"{benchmark_name}: Memory usage increasing")
                recommendations.append(f"Review memory usage patterns in {benchmark_name}")
            
            # Check success rates
            if analysis['success_rate']['current'] < 0.9:
                health_score -= 15
                issues.append(f"{benchmark_name}: Low success rate ({analysis['success_rate']['current']:.1%})")
                recommendations.append(f"Fix reliability issues in {benchmark_name}")
            
            # Check for high variability
            if analysis['duration']['std_dev'] > analysis['duration']['mean'] * 0.5:
                health_score -= 5
                issues.append(f"{benchmark_name}: High duration variability")
                recommendations.append(f"Investigate inconsistent performance in {benchmark_name}")
        
        health_score = max(0, health_score)
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days,
            'health_score': health_score,
            'overall_status': self._get_health_status(health_score),
            'benchmarks_analyzed': len(trends),
            'issues': issues,
            'recommendations': recommendations,
            'detailed_analysis': trends,
            'summary': self._generate_summary(trends)
        }
        
        # Save report
        report_file = self.data_dir / f"performance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report saved to {report_file}")
        return report
    
    def _get_health_status(self, score: int) -> str:
        """Get health status based on score"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        elif score >= 40:
            return "poor"
        else:
            return "critical"
    
    def _generate_summary(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not trends:
            return {}
        
        all_durations = []
        all_memory = []
        all_success_rates = []
        
        for analysis in trends.values():
            all_durations.append(analysis['duration']['mean'])
            all_memory.append(analysis['memory']['mean'])
            all_success_rates.append(analysis['success_rate']['overall'])
        
        return {
            'average_duration_seconds': statistics.mean(all_durations),
            'average_memory_mb': statistics.mean(all_memory),
            'overall_success_rate': statistics.mean(all_success_rates),
            'fastest_benchmark': min(trends.keys(), key=lambda k: trends[k]['duration']['mean']),
            'slowest_benchmark': max(trends.keys(), key=lambda k: trends[k]['duration']['mean']),
            'most_reliable_benchmark': max(trends.keys(), key=lambda k: trends[k]['success_rate']['overall']),
            'least_reliable_benchmark': min(trends.keys(), key=lambda k: trends[k]['success_rate']['overall'])
        }
    
    def create_performance_charts(self, output_dir: str = "performance_charts") -> List[str]:
        """Create performance visualization charts"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.benchmarks_file.exists():
            self.logger.warning("No benchmark data available for charts")
            return []
        
        # Load data
        with open(self.benchmarks_file, 'r') as f:
            benchmark_data = json.load(f)
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(benchmark_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        created_charts = []
        
        # Duration trends chart
        plt.figure(figsize=(12, 8))
        for benchmark_name in df['name'].unique():
            benchmark_df = df[df['name'] == benchmark_name].sort_values('timestamp')
            plt.plot(benchmark_df['timestamp'], benchmark_df['duration_seconds'], 
                    label=benchmark_name, marker='o')
        
        plt.title('Benchmark Duration Trends')
        plt.xlabel('Date')
        plt.ylabel('Duration (seconds)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        duration_chart = output_path / 'duration_trends.png'
        plt.savefig(duration_chart)
        plt.close()
        created_charts.append(str(duration_chart))
        
        # Memory usage trends chart
        plt.figure(figsize=(12, 8))
        for benchmark_name in df['name'].unique():
            benchmark_df = df[df['name'] == benchmark_name].sort_values('timestamp')
            plt.plot(benchmark_df['timestamp'], benchmark_df['memory_peak_mb'], 
                    label=benchmark_name, marker='s')
        
        plt.title('Memory Usage Trends')
        plt.xlabel('Date')
        plt.ylabel('Peak Memory (MB)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        memory_chart = output_path / 'memory_trends.png'
        plt.savefig(memory_chart)
        plt.close()
        created_charts.append(str(memory_chart))
        
        # Success rate chart
        success_rates = df.groupby(['name', df['timestamp'].dt.date])['success'].mean().reset_index()
        
        plt.figure(figsize=(12, 8))
        for benchmark_name in success_rates['name'].unique():
            benchmark_df = success_rates[success_rates['name'] == benchmark_name]
            plt.plot(benchmark_df['timestamp'], benchmark_df['success'], 
                    label=benchmark_name, marker='^')
        
        plt.title('Success Rate Trends')
        plt.xlabel('Date')
        plt.ylabel('Success Rate')
        plt.legend()
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        
        success_chart = output_path / 'success_trends.png'
        plt.savefig(success_chart)
        plt.close()
        created_charts.append(str(success_chart))
        
        self.logger.info(f"Created {len(created_charts)} performance charts")
        return created_charts


def main():
    """Main entry point for performance tracking"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance tracking and benchmarking')
    parser.add_argument('--run-benchmarks', action='store_true',
                       help='Run standard benchmarks')
    parser.add_argument('--analyze-trends', type=int, default=30,
                       help='Analyze trends over N days')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate performance report')
    parser.add_argument('--create-charts', action='store_true',
                       help='Create performance charts')
    parser.add_argument('--data-dir', default='performance_data',
                       help='Data directory for performance data')
    
    args = parser.parse_args()
    
    tracker = PerformanceTracker(args.data_dir)
    
    if args.run_benchmarks:
        benchmarks = tracker.run_standard_benchmarks()
        tracker.save_benchmarks(benchmarks)
        
        print(f"Completed {len(benchmarks)} benchmarks:")
        for benchmark in benchmarks:
            status = "✅" if benchmark.success else "❌"
            print(f"{status} {benchmark.name}: {benchmark.duration_seconds:.2f}s")
    
    if args.analyze_trends or args.generate_report:
        if args.generate_report:
            report = tracker.generate_performance_report(args.analyze_trends)
            print(f"Performance Report (Health Score: {report.get('health_score', 'N/A')})")
            print(f"Status: {report.get('overall_status', 'Unknown').upper()}")
            
            if report.get('issues'):
                print("\nIssues:")
                for issue in report['issues']:
                    print(f"  ⚠️  {issue}")
            
            if report.get('recommendations'):
                print("\nRecommendations:")
                for rec in report['recommendations']:
                    print(f"  💡 {rec}")
        else:
            trends = tracker.analyze_performance_trends(args.analyze_trends)
            print(f"Analyzed trends for {len(trends)} benchmarks over {args.analyze_trends} days")
    
    if args.create_charts:
        charts = tracker.create_performance_charts()
        print(f"Created {len(charts)} performance charts:")
        for chart in charts:
            print(f"  📊 {chart}")


if __name__ == '__main__':
    main()