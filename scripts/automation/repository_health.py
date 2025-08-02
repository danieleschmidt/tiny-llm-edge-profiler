#!/usr/bin/env python3
"""
Repository health monitoring and automated maintenance script.

Monitors repository health, performs automated maintenance tasks,
and generates health reports for the tiny-llm-edge-profiler project.
"""

import json
import os
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from pathlib import Path
import requests
import yaml


class RepositoryHealthMonitor:
    """Monitor and maintain repository health"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.logger = self._setup_logging()
        self.github_token = os.getenv('GITHUB_TOKEN')
        
        # Get repository info from git
        self.repo_url = self._get_repo_url()
        self.repo_name = self._extract_repo_name(self.repo_url)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _get_repo_url(self) -> str:
        """Get repository URL from git remote"""
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except subprocess.SubprocessError:
            return ""
    
    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL"""
        if 'github.com' in repo_url:
            # Handle both SSH and HTTPS URLs
            if repo_url.startswith('git@'):
                # SSH: git@github.com:owner/repo.git
                return repo_url.split(':')[1].replace('.git', '')
            else:
                # HTTPS: https://github.com/owner/repo.git
                return '/'.join(repo_url.split('/')[-2:]).replace('.git', '')
        return ""
    
    def check_repository_health(self) -> Dict[str, Any]:
        """Perform comprehensive repository health check"""
        self.logger.info("Starting repository health check")
        
        health_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'repository': self.repo_name,
            'overall_health': 'healthy',
            'score': 0,
            'max_score': 100,
            'checks': {}
        }
        
        # Perform various health checks
        checks = [
            ('git_health', self._check_git_health),
            ('file_structure', self._check_file_structure),
            ('documentation', self._check_documentation),
            ('testing', self._check_testing_setup),
            ('security', self._check_security_setup),
            ('ci_cd', self._check_ci_cd_setup),
            ('dependencies', self._check_dependencies),
            ('code_quality', self._check_code_quality)
        ]
        
        total_score = 0
        for check_name, check_func in checks:
            try:
                result = check_func()
                health_report['checks'][check_name] = result
                total_score += result.get('score', 0)
                self.logger.info(f"✅ {check_name}: {result.get('score', 0)}/10")
            except Exception as e:
                self.logger.error(f"❌ Error in {check_name}: {e}")
                health_report['checks'][check_name] = {
                    'score': 0,
                    'status': 'error',
                    'error': str(e)
                }
        
        health_report['score'] = total_score
        
        # Determine overall health status
        if total_score >= 80:
            health_report['overall_health'] = 'excellent'
        elif total_score >= 60:
            health_report['overall_health'] = 'good'
        elif total_score >= 40:
            health_report['overall_health'] = 'fair'
        else:
            health_report['overall_health'] = 'poor'
        
        return health_report
    
    def _check_git_health(self) -> Dict[str, Any]:
        """Check Git repository health"""
        result = {'score': 0, 'max_score': 10, 'issues': [], 'status': 'healthy'}
        
        try:
            # Check if we're in a git repository
            subprocess.run(['git', 'status'], cwd=self.repo_path, 
                         capture_output=True, check=True)
            result['score'] += 2
            
            # Check for uncommitted changes
            status_result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repo_path,
                capture_output=True, text=True
            )
            
            if not status_result.stdout.strip():
                result['score'] += 2
            else:
                result['issues'].append("Uncommitted changes detected")
            
            # Check recent commits
            commits_result = subprocess.run(
                ['git', 'log', '--oneline', '-10'],
                cwd=self.repo_path,
                capture_output=True, text=True
            )
            
            if commits_result.stdout.strip():
                result['score'] += 2
                
                # Check commit message quality
                commit_lines = commits_result.stdout.strip().split('\n')
                good_commits = sum(1 for line in commit_lines 
                                 if len(line.split(' ', 1)[1]) > 10)
                if good_commits >= len(commit_lines) * 0.8:
                    result['score'] += 2
                else:
                    result['issues'].append("Poor commit message quality")
            
            # Check for proper branching
            branches_result = subprocess.run(
                ['git', 'branch', '-r'],
                cwd=self.repo_path,
                capture_output=True, text=True
            )
            
            if 'origin/main' in branches_result.stdout or 'origin/master' in branches_result.stdout:
                result['score'] += 2
            else:
                result['issues'].append("No main/master branch detected")
                
        except subprocess.SubprocessError as e:
            result['issues'].append(f"Git command failed: {e}")
        
        return result
    
    def _check_file_structure(self) -> Dict[str, Any]:
        """Check repository file structure"""
        result = {'score': 0, 'max_score': 10, 'issues': [], 'status': 'healthy'}
        
        required_files = [
            'README.md',
            'LICENSE',
            'requirements.txt',
            '.gitignore'
        ]
        
        recommended_files = [
            'CONTRIBUTING.md',
            'CODE_OF_CONDUCT.md',
            'SECURITY.md',
            'CHANGELOG.md'
        ]
        
        # Check required files
        for file in required_files:
            if (self.repo_path / file).exists():
                result['score'] += 1.5
            else:
                result['issues'].append(f"Missing required file: {file}")
        
        # Check recommended files
        for file in recommended_files:
            if (self.repo_path / file).exists():
                result['score'] += 1
            else:
                result['issues'].append(f"Missing recommended file: {file}")
        
        # Check directory structure
        expected_dirs = ['src', 'tests', 'docs']
        for dir_name in expected_dirs:
            if (self.repo_path / dir_name).exists():
                result['score'] += 0.5
            else:
                result['issues'].append(f"Missing directory: {dir_name}")
        
        return result
    
    def _check_documentation(self) -> Dict[str, Any]:
        """Check documentation quality"""
        result = {'score': 0, 'max_score': 10, 'issues': [], 'status': 'healthy'}
        
        # Check README.md
        readme_path = self.repo_path / 'README.md'
        if readme_path.exists():
            readme_content = readme_path.read_text()
            
            if len(readme_content) > 500:
                result['score'] += 3
            else:
                result['issues'].append("README.md is too short")
            
            # Check for common sections
            sections = ['installation', 'usage', 'contributing', 'license']
            found_sections = sum(1 for section in sections 
                               if section.lower() in readme_content.lower())
            result['score'] += found_sections * 0.5
            
        else:
            result['issues'].append("README.md missing")
        
        # Check docs directory
        docs_path = self.repo_path / 'docs'
        if docs_path.exists():
            doc_files = list(docs_path.rglob('*.md'))
            if len(doc_files) > 3:
                result['score'] += 3
            elif len(doc_files) > 0:
                result['score'] += 1.5
            else:
                result['issues'].append("Docs directory exists but is empty")
        else:
            result['issues'].append("No docs directory found")
        
        # Check for API documentation
        if (self.repo_path / 'docs' / 'api').exists():
            result['score'] += 2
        
        return result
    
    def _check_testing_setup(self) -> Dict[str, Any]:
        """Check testing infrastructure"""
        result = {'score': 0, 'max_score': 10, 'issues': [], 'status': 'healthy'}
        
        # Check tests directory
        tests_path = self.repo_path / 'tests'
        if tests_path.exists():
            test_files = list(tests_path.rglob('test_*.py')) + list(tests_path.rglob('*_test.py'))
            
            if len(test_files) > 10:
                result['score'] += 4
            elif len(test_files) > 5:
                result['score'] += 3
            elif len(test_files) > 0:
                result['score'] += 2
            else:
                result['issues'].append("Tests directory exists but no test files found")
        else:
            result['issues'].append("No tests directory found")
        
        # Check test configuration
        test_configs = ['pytest.ini', 'pyproject.toml', 'setup.cfg']
        if any((self.repo_path / config).exists() for config in test_configs):
            result['score'] += 2
        else:
            result['issues'].append("No test configuration found")
        
        # Check for test coverage
        if (self.repo_path / '.coverage').exists() or 'coverage' in (self.repo_path / 'requirements-dev.txt').read_text() if (self.repo_path / 'requirements-dev.txt').exists() else '':
            result['score'] += 2
        
        # Try running tests
        try:
            test_result = subprocess.run(
                ['python', '-m', 'pytest', '--version'],
                cwd=self.repo_path,
                capture_output=True,
                timeout=10
            )
            if test_result.returncode == 0:
                result['score'] += 2
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            result['issues'].append("Cannot run pytest")
        
        return result
    
    def _check_security_setup(self) -> Dict[str, Any]:
        """Check security configuration"""
        result = {'score': 0, 'max_score': 10, 'issues': [], 'status': 'healthy'}
        
        # Check SECURITY.md
        if (self.repo_path / 'SECURITY.md').exists():
            result['score'] += 2
        else:
            result['issues'].append("No SECURITY.md found")
        
        # Check for security scanning tools
        security_tools = ['safety', 'bandit', 'semgrep']
        requirements_files = ['requirements-dev.txt', 'requirements.txt']
        
        found_tools = 0
        for req_file in requirements_files:
            req_path = self.repo_path / req_file
            if req_path.exists():
                content = req_path.read_text()
                found_tools += sum(1 for tool in security_tools if tool in content)
        
        result['score'] += min(found_tools * 2, 4)
        
        # Check .gitignore for secrets
        gitignore_path = self.repo_path / '.gitignore'
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            secret_patterns = ['.env', '*.key', '*.pem', 'secrets']
            if any(pattern in content for pattern in secret_patterns):
                result['score'] += 2
            else:
                result['issues'].append("Gitignore missing secret file patterns")
        
        # Check for hardcoded secrets (basic check)
        try:
            secret_scan = subprocess.run(
                ['grep', '-r', '-i', 'password\\|secret\\|key\\|token', 'src/', '--include=*.py'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if secret_scan.returncode != 0:  # No matches found
                result['score'] += 2
            else:
                result['issues'].append("Potential hardcoded secrets found")
        except subprocess.SubprocessError:
            pass
        
        return result
    
    def _check_ci_cd_setup(self) -> Dict[str, Any]:
        """Check CI/CD configuration"""
        result = {'score': 0, 'max_score': 10, 'issues': [], 'status': 'healthy'}
        
        # Check GitHub Actions
        github_workflows = self.repo_path / '.github' / 'workflows'
        if github_workflows.exists():
            workflow_files = list(github_workflows.glob('*.yml')) + list(github_workflows.glob('*.yaml'))
            
            if len(workflow_files) >= 3:
                result['score'] += 4
            elif len(workflow_files) >= 1:
                result['score'] += 2
            
            # Check for common workflow types
            workflow_content = ""
            for wf in workflow_files:
                workflow_content += wf.read_text()
            
            workflow_types = ['test', 'build', 'deploy', 'security']
            found_types = sum(1 for wf_type in workflow_types 
                            if wf_type in workflow_content.lower())
            result['score'] += found_types
            
        else:
            result['issues'].append("No GitHub Actions workflows found")
        
        # Check for other CI systems
        ci_files = ['.travis.yml', '.circleci/config.yml', 'azure-pipelines.yml']
        if any((self.repo_path / ci_file).exists() for ci_file in ci_files):
            result['score'] += 2
        
        # Check Docker setup
        if (self.repo_path / 'Dockerfile').exists():
            result['score'] += 2
        
        if (self.repo_path / 'docker-compose.yml').exists():
            result['score'] += 1
        
        return result
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check dependency management"""
        result = {'score': 0, 'max_score': 10, 'issues': [], 'status': 'healthy'}
        
        # Check requirements files
        req_files = ['requirements.txt', 'requirements-dev.txt', 'pyproject.toml']
        found_req_files = sum(1 for req in req_files if (self.repo_path / req).exists())
        
        if found_req_files >= 2:
            result['score'] += 3
        elif found_req_files >= 1:
            result['score'] += 2
        else:
            result['issues'].append("No dependency files found")
        
        # Check for version pinning
        if (self.repo_path / 'requirements.txt').exists():
            content = (self.repo_path / 'requirements.txt').read_text()
            lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
            pinned_deps = sum(1 for line in lines if '==' in line or '>=' in line)
            
            if pinned_deps >= len(lines) * 0.8:
                result['score'] += 3
            elif pinned_deps >= len(lines) * 0.5:
                result['score'] += 2
            else:
                result['issues'].append("Dependencies not properly pinned")
        
        # Check for dependency scanning
        if (self.repo_path / 'renovate.json').exists() or (self.repo_path / '.dependabot').exists():
            result['score'] += 2
        else:
            result['issues'].append("No automated dependency updates configured")
        
        # Check for virtual environment
        if (self.repo_path / 'venv').exists() or os.getenv('VIRTUAL_ENV'):
            result['score'] += 2
        
        return result
    
    def _check_code_quality(self) -> Dict[str, Any]:
        """Check code quality setup"""
        result = {'score': 0, 'max_score': 10, 'issues': [], 'status': 'healthy'}
        
        # Check for linting configuration
        lint_configs = ['.flake8', '.pylintrc', 'pyproject.toml', 'setup.cfg']
        if any((self.repo_path / config).exists() for config in lint_configs):
            result['score'] += 2
        else:
            result['issues'].append("No linting configuration found")
        
        # Check for formatting configuration
        format_configs = ['.black', 'pyproject.toml', '.isort.cfg']
        if any((self.repo_path / config).exists() for config in format_configs):
            result['score'] += 2
        
        # Check for pre-commit hooks
        if (self.repo_path / '.pre-commit-config.yaml').exists():
            result['score'] += 3
        else:
            result['issues'].append("No pre-commit hooks configured")
        
        # Check for type checking
        if 'mypy' in (self.repo_path / 'requirements-dev.txt').read_text() if (self.repo_path / 'requirements-dev.txt').exists() else '':
            result['score'] += 2
        
        # Check code complexity
        try:
            radon_result = subprocess.run(
                ['radon', 'cc', 'src/', '--min', 'B'],
                cwd=self.repo_path,
                capture_output=True
            )
            if radon_result.returncode == 0:
                result['score'] += 1
        except subprocess.SubprocessError:
            pass
        
        return result
    
    def generate_health_report(self, health_data: Dict[str, Any]) -> str:
        """Generate human-readable health report"""
        report = []
        report.append("=" * 50)
        report.append("REPOSITORY HEALTH REPORT")
        report.append("=" * 50)
        report.append(f"Repository: {health_data['repository']}")
        report.append(f"Timestamp: {health_data['timestamp']}")
        report.append(f"Overall Health: {health_data['overall_health'].upper()}")
        report.append(f"Score: {health_data['score']}/{health_data['max_score']}")
        report.append("")
        
        # Detailed checks
        for check_name, check_data in health_data['checks'].items():
            status_emoji = "✅" if check_data.get('score', 0) >= check_data.get('max_score', 10) * 0.8 else "⚠️"
            report.append(f"{status_emoji} {check_name.replace('_', ' ').title()}: {check_data.get('score', 0)}/{check_data.get('max_score', 10)}")
            
            if check_data.get('issues'):
                for issue in check_data['issues']:
                    report.append(f"   - {issue}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        
        total_issues = sum(len(check.get('issues', [])) for check in health_data['checks'].values())
        if total_issues == 0:
            report.append("🎉 Excellent! Your repository is in great health!")
        else:
            report.append("Address the issues above to improve repository health:")
            for check_name, check_data in health_data['checks'].items():
                if check_data.get('issues'):
                    report.append(f"\n{check_name.replace('_', ' ').title()}:")
                    for issue in check_data['issues']:
                        report.append(f"  • {issue}")
        
        return "\n".join(report)
    
    def run_maintenance_tasks(self) -> Dict[str, Any]:
        """Run automated maintenance tasks"""
        self.logger.info("Running automated maintenance tasks")
        
        maintenance_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'tasks': {}
        }
        
        # Clean up temporary files
        maintenance_results['tasks']['cleanup'] = self._cleanup_temp_files()
        
        # Update dependencies
        maintenance_results['tasks']['dependencies'] = self._update_dependencies()
        
        # Run security scan
        maintenance_results['tasks']['security'] = self._run_security_scan()
        
        # Optimize git repository
        maintenance_results['tasks']['git_optimization'] = self._optimize_git_repo()
        
        return maintenance_results
    
    def _cleanup_temp_files(self) -> Dict[str, Any]:
        """Clean up temporary files"""
        result = {'status': 'success', 'cleaned_files': 0, 'errors': []}
        
        try:
            # Remove Python cache files
            cache_dirs = list(self.repo_path.rglob('__pycache__'))
            for cache_dir in cache_dirs:
                subprocess.run(['rm', '-rf', str(cache_dir)])
                result['cleaned_files'] += 1
            
            # Remove .pyc files
            pyc_files = list(self.repo_path.rglob('*.pyc'))
            for pyc_file in pyc_files:
                pyc_file.unlink()
                result['cleaned_files'] += 1
            
            # Remove test artifacts
            test_artifacts = ['.coverage', '.pytest_cache', 'htmlcov']
            for artifact in test_artifacts:
                artifact_path = self.repo_path / artifact
                if artifact_path.exists():
                    subprocess.run(['rm', '-rf', str(artifact_path)])
                    result['cleaned_files'] += 1
                    
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
        
        return result
    
    def _update_dependencies(self) -> Dict[str, Any]:
        """Check for outdated dependencies"""
        result = {'status': 'success', 'outdated_packages': [], 'errors': []}
        
        try:
            # Check for outdated packages
            outdated_result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True, text=True
            )
            
            if outdated_result.returncode == 0:
                outdated_data = json.loads(outdated_result.stdout)
                result['outdated_packages'] = outdated_data
                
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
        
        return result
    
    def _run_security_scan(self) -> Dict[str, Any]:
        """Run security scan"""
        result = {'status': 'success', 'vulnerabilities': 0, 'errors': []}
        
        try:
            # Run safety check
            safety_result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True, text=True
            )
            
            if safety_result.stdout:
                safety_data = json.loads(safety_result.stdout)
                result['vulnerabilities'] = len(safety_data)
                
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(str(e))
        
        return result
    
    def _optimize_git_repo(self) -> Dict[str, Any]:
        """Optimize git repository"""
        result = {'status': 'success', 'optimizations': [], 'errors': []}
        
        try:
            # Run git gc
            gc_result = subprocess.run(['git', 'gc'], cwd=self.repo_path, capture_output=True)
            if gc_result.returncode == 0:
                result['optimizations'].append('Garbage collection completed')
            
            # Prune remote tracking branches
            prune_result = subprocess.run(['git', 'remote', 'prune', 'origin'], 
                                        cwd=self.repo_path, capture_output=True)
            if prune_result.returncode == 0:
                result['optimizations'].append('Remote branches pruned')
                
        except Exception as e:
            result['status'] = 'error'  
            result['errors'].append(str(e))
        
        return result


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Repository health monitoring')
    parser.add_argument('--check-health', action='store_true',
                       help='Run health check')
    parser.add_argument('--maintenance', action='store_true',
                       help='Run maintenance tasks')
    parser.add_argument('--output', default='health_report.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    monitor = RepositoryHealthMonitor()
    
    if args.check_health:
        health_data = monitor.check_repository_health()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(health_data, f, indent=2)
        
        # Print report
        report = monitor.generate_health_report(health_data)
        print(report)
        
        # Exit with error code if health is poor
        if health_data['score'] < 40:
            exit(1)
    
    if args.maintenance:
        maintenance_results = monitor.run_maintenance_tasks()
        
        print("Maintenance Results:")
        for task, result in maintenance_results['tasks'].items():
            status = "✅" if result['status'] == 'success' else "❌"
            print(f"{status} {task}: {result['status']}")


if __name__ == '__main__':
    main()