#!/usr/bin/env python3
"""
Automated metrics collection script for tiny-llm-edge-profiler project.

Collects various project metrics including code quality, performance,
security, and development velocity metrics.
"""

import json
import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

import requests
import subprocess
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


class MetricsCollector:
    """Main metrics collection orchestrator"""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo = self.config['project']['repository']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load metrics configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all configured metrics"""
        self.logger.info("Starting comprehensive metrics collection")
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'project': self.config['project']['name'],
            'version': self.config['project']['version']
        }
        
        # Collect different metric categories
        if self.config['tracking']['enabled']:
            metrics['code_quality'] = self.collect_code_quality_metrics()
            metrics['performance'] = self.collect_performance_metrics()
            metrics['security'] = self.collect_security_metrics()
            metrics['reliability'] = self.collect_reliability_metrics()
            metrics['development'] = self.collect_development_metrics()
            metrics['custom'] = self.collect_custom_metrics()
        
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics"""
        self.logger.info("Collecting code quality metrics")
        
        metrics = {}
        
        try:
            # Test coverage from pytest-cov
            result = subprocess.run(
                ['python', '-m', 'pytest', '--cov=src', '--cov-report=json', '--quiet'],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            
            if result.returncode == 0 and os.path.exists('coverage.json'):
                with open('coverage.json', 'r') as f:
                    coverage_data = json.load(f)
                    metrics['coverage_percentage'] = coverage_data['totals']['percent_covered']
            
            # Code complexity using radon
            try:
                result = subprocess.run(
                    ['radon', 'cc', 'src/', '--json'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    complexity_data = json.loads(result.stdout)
                    # Calculate average complexity
                    total_complexity = sum(
                        sum(func['complexity'] for func in file_data)
                        for file_data in complexity_data.values()
                    )
                    total_functions = sum(
                        len(file_data) for file_data in complexity_data.values()
                    )
                    metrics['average_complexity'] = total_complexity / max(total_functions, 1)
                    
            except (subprocess.SubprocessError, json.JSONDecodeError):
                self.logger.warning("Could not collect complexity metrics")
            
            # Lines of code
            result = subprocess.run(
                ['find', 'src/', '-name', '*.py', '-exec', 'wc', '-l', '{}', '+'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_lines = sum(int(line.split()[0]) for line in lines[:-1])
                metrics['lines_of_code'] = total_lines
                
        except Exception as e:
            self.logger.error(f"Error collecting code quality metrics: {e}")
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics"""
        self.logger.info("Collecting performance metrics")
        
        metrics = {}
        
        try:
            # Build time measurement
            start_time = datetime.utcnow()
            result = subprocess.run(
                ['python', '-m', 'pip', 'install', '-e', '.'],
                capture_output=True, text=True
            )
            build_time = (datetime.utcnow() - start_time).total_seconds()
            metrics['build_time_seconds'] = build_time
            
            # Test execution time
            start_time = datetime.utcnow()
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/', '--quiet'],
                capture_output=True, text=True
            )
            test_time = (datetime.utcnow() - start_time).total_seconds()
            metrics['test_execution_time_seconds'] = test_time
            
            # Docker image size
            result = subprocess.run(
                ['docker', 'images', '--format', 'table {{.Size}}', 'tiny-llm-profiler:latest'],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                size_line = result.stdout.strip().split('\n')[-1]
                metrics['docker_image_size'] = size_line
                
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics"""
        self.logger.info("Collecting security metrics")
        
        metrics = {}
        
        try:
            # Run safety check for known vulnerabilities
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                try:
                    safety_data = json.loads(result.stdout)
                    metrics['vulnerability_count'] = len(safety_data)
                    metrics['critical_vulnerabilities'] = sum(
                        1 for vuln in safety_data 
                        if vuln.get('severity', '').lower() == 'critical'
                    )
                except json.JSONDecodeError:
                    metrics['vulnerability_count'] = 0
            else:
                # Safety returns non-zero when vulnerabilities found
                try:
                    safety_data = json.loads(result.stdout)
                    metrics['vulnerability_count'] = len(safety_data)
                except json.JSONDecodeError:
                    metrics['vulnerability_count'] = 0
            
            # Check for outdated dependencies
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                outdated_data = json.loads(result.stdout)
                metrics['outdated_dependencies'] = len(outdated_data)
                
        except Exception as e:
            self.logger.error(f"Error collecting security metrics: {e}")
        
        return metrics
    
    def collect_reliability_metrics(self) -> Dict[str, Any]:
        """Collect reliability metrics from monitoring systems"""
        self.logger.info("Collecting reliability metrics")
        
        metrics = {}
        
        try:
            # Query Prometheus for uptime and error rates
            if self.config['integrations']['prometheus']['enabled']:
                prometheus_url = self.config['integrations']['prometheus']['endpoint']
                
                # Uptime query
                uptime_query = 'up{job="tiny-llm-profiler"}'
                response = requests.get(f"{prometheus_url}/api/v1/query", 
                                      params={'query': uptime_query})
                
                if response.status_code == 200:
                    data = response.json()
                    if data['data']['result']:
                        metrics['current_uptime'] = float(data['data']['result'][0]['value'][1])
                
                # Error rate query
                error_query = 'rate(http_requests_total{status=~"5.."}[5m])'
                response = requests.get(f"{prometheus_url}/api/v1/query",
                                      params={'query': error_query})
                
                if response.status_code == 200:
                    data = response.json()
                    if data['data']['result']:
                        metrics['error_rate'] = float(data['data']['result'][0]['value'][1])
                        
        except Exception as e:
            self.logger.error(f"Error collecting reliability metrics: {e}")
        
        return metrics
    
    def collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development velocity metrics from GitHub API"""
        self.logger.info("Collecting development metrics from GitHub")
        
        metrics = {}
        
        if not self.github_token:
            self.logger.warning("GitHub token not available, skipping GitHub metrics")
            return metrics
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            # Pull requests metrics (last 30 days)
            since_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
            
            # Closed PRs
            pr_url = f"https://api.github.com/repos/{self.repo}/pulls"
            response = requests.get(pr_url, headers=headers, params={
                'state': 'closed',
                'since': since_date,
                'per_page': 100
            })
            
            if response.status_code == 200:
                prs = response.json()
                metrics['closed_prs_30d'] = len(prs)
                
                # Calculate average PR lifecycle time
                if prs:
                    lifecycle_times = []
                    for pr in prs:
                        if pr['merged_at']:
                            created = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
                            merged = datetime.fromisoformat(pr['merged_at'].replace('Z', '+00:00'))
                            lifecycle_times.append((merged - created).total_seconds() / 3600)  # hours
                    
                    if lifecycle_times:
                        metrics['avg_pr_lifecycle_hours'] = sum(lifecycle_times) / len(lifecycle_times)
            
            # Issues metrics
            issues_url = f"https://api.github.com/repos/{self.repo}/issues"
            response = requests.get(issues_url, headers=headers, params={
                'state': 'closed',
                'since': since_date,
                'per_page': 100
            })
            
            if response.status_code == 200:
                issues = response.json()
                # Filter out PRs (GitHub treats PRs as issues)
                actual_issues = [issue for issue in issues if not issue.get('pull_request')]
                metrics['closed_issues_30d'] = len(actual_issues)
            
            # Commits metrics
            commits_url = f"https://api.github.com/repos/{self.repo}/commits"
            response = requests.get(commits_url, headers=headers, params={
                'since': since_date,
                'per_page': 100
            })
            
            if response.status_code == 200:
                commits = response.json()
                metrics['commits_30d'] = len(commits)
                
        except Exception as e:
            self.logger.error(f"Error collecting development metrics: {e}")
        
        return metrics
    
    def collect_custom_metrics(self) -> Dict[str, Any]:
        """Collect custom profiling-specific metrics"""
        self.logger.info("Collecting custom profiling metrics")
        
        metrics = {}
        
        try:
            # These would be implemented based on your specific profiling system
            # For now, we'll simulate some metrics
            
            # Device compatibility (simulated)
            metrics['device_compatibility_score'] = 96.5
            
            # Model optimization ratio (simulated)
            metrics['model_optimization_ratio'] = 1.8
            
            # Edge deployment success rate (simulated)
            metrics['edge_deployment_success_rate'] = 98.2
            
            # Profiling accuracy (simulated)
            metrics['profiling_accuracy'] = 94.1
            
        except Exception as e:
            self.logger.error(f"Error collecting custom metrics: {e}")
        
        return metrics
    
    def export_metrics(self, metrics: Dict[str, Any]) -> None:
        """Export metrics to configured destinations"""
        self.logger.info("Exporting metrics")
        
        # Export to JSON file
        output_dir = Path('metrics_output')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        json_file = output_dir / f'metrics_{timestamp}.json'
        
        with open(json_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics exported to {json_file}")
        
        # Export to Prometheus Push Gateway if enabled
        if self.config['integrations']['prometheus']['enabled']:
            self._export_to_prometheus(metrics)
    
    def _export_to_prometheus(self, metrics: Dict[str, Any]) -> None:
        """Export metrics to Prometheus Push Gateway"""
        try:
            registry = CollectorRegistry()
            
            # Create Prometheus metrics
            for category, category_metrics in metrics.items():
                if isinstance(category_metrics, dict):
                    for metric_name, value in category_metrics.items():
                        if isinstance(value, (int, float)):
                            gauge = Gauge(
                                f'tiny_llm_profiler_{category}_{metric_name}',
                                f'{category} {metric_name}',
                                registry=registry
                            )
                            gauge.set(value)
            
            # Push to gateway
            push_gateway_url = self.config['integrations']['prometheus'].get('push_gateway')
            if push_gateway_url:
                push_to_gateway(push_gateway_url, job='metrics_collector', registry=registry)
                self.logger.info("Metrics pushed to Prometheus")
                
        except Exception as e:
            self.logger.error(f"Error exporting to Prometheus: {e}")
    
    def check_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """Check metrics against configured thresholds"""
        alerts = []
        
        thresholds = self.config['metrics']
        
        # Check code quality thresholds
        if 'code_quality' in metrics:
            cq_metrics = metrics['code_quality']
            cq_thresholds = thresholds['code_quality']
            
            if 'coverage_percentage' in cq_metrics:
                if cq_metrics['coverage_percentage'] < cq_thresholds['coverage_threshold']:
                    alerts.append(f"Code coverage below threshold: {cq_metrics['coverage_percentage']}% < {cq_thresholds['coverage_threshold']}%")
        
        # Check security thresholds
        if 'security' in metrics:
            sec_metrics = metrics['security']
            sec_thresholds = thresholds['security']
            
            if 'critical_vulnerabilities' in sec_metrics:
                if sec_metrics['critical_vulnerabilities'] > sec_thresholds['critical_vulnerabilities_threshold']:
                    alerts.append(f"Critical vulnerabilities detected: {sec_metrics['critical_vulnerabilities']}")
        
        return alerts


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--config', default='.github/project-metrics.json',
                       help='Path to metrics configuration file')
    parser.add_argument('--export-only', action='store_true',
                       help='Only export metrics, skip collection')
    parser.add_argument('--check-thresholds', action='store_true',
                       help='Check metrics against thresholds and alert')
    
    args = parser.parse_args()
    
    try:
        collector = MetricsCollector(args.config)
        
        if not args.export_only:
            metrics = collector.collect_all_metrics()
            collector.export_metrics(metrics)
            
            if args.check_thresholds:
                alerts = collector.check_thresholds(metrics)
                if alerts:
                    print("THRESHOLD ALERTS:")
                    for alert in alerts:
                        print(f"  - {alert}")
                    sys.exit(1)
        
        collector.logger.info("Metrics collection completed successfully")
        
    except Exception as e:
        logging.error(f"Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()