#!/usr/bin/env python3
"""
Health Check Script for Self-Healing Pipeline Guard
Comprehensive health monitoring for container orchestration
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path


class HealthChecker:
    def __init__(self):
        self.checks = []
        self.timeout = int(os.getenv('HEALTH_CHECK_TIMEOUT', '10'))
        self.port = int(os.getenv('GUARD_PORT', '8080'))
        
    def add_check(self, name, check_func, critical=True):
        """Add a health check"""
        self.checks.append({
            'name': name,
            'func': check_func,
            'critical': critical
        })
    
    def run_all_checks(self):
        """Run all health checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        critical_failures = 0
        
        for check in self.checks:
            try:
                start_time = time.time()
                status, message, details = check['func']()
                duration = time.time() - start_time
                
                results['checks'][check['name']] = {
                    'status': status,
                    'message': message,
                    'details': details,
                    'duration_ms': round(duration * 1000, 2),
                    'critical': check['critical']
                }
                
                if status != 'healthy' and check['critical']:
                    critical_failures += 1
                    
            except Exception as e:
                results['checks'][check['name']] = {
                    'status': 'error',
                    'message': f"Health check failed: {str(e)}",
                    'details': {},
                    'duration_ms': 0,
                    'critical': check['critical']
                }
                
                if check['critical']:
                    critical_failures += 1
        
        # Determine overall status
        if critical_failures > 0:
            results['overall_status'] = 'unhealthy'
        elif any(check['status'] != 'healthy' for check in results['checks'].values()):
            results['overall_status'] = 'degraded'
        
        return results
    
    def check_process_health(self):
        """Check if main process is running"""
        try:
            # Check if app status file exists and is recent
            app_status_file = Path('/app/tmp/app_status.json')
            
            if not app_status_file.exists():
                return 'unhealthy', 'Application status file not found', {}
            
            # Check if file is recent (updated within last 60 seconds)
            file_age = time.time() - app_status_file.stat().st_mtime
            if file_age > 60:
                return 'unhealthy', f'Application status file is stale ({file_age:.1f}s old)', {'file_age': file_age}
            
            # Try to read and parse the status file
            with open(app_status_file, 'r') as f:
                status_data = json.load(f)
            
            app_status = status_data.get('status', 'unknown')
            if app_status != 'running':
                return 'unhealthy', f'Application status is {app_status}', status_data
            
            return 'healthy', 'Application process is running', status_data
            
        except Exception as e:
            return 'error', f'Failed to check process health: {str(e)}', {}
    
    def check_disk_space(self):
        """Check available disk space"""
        try:
            import shutil
            
            # Check disk usage for /app directory
            total, used, free = shutil.disk_usage('/app')
            
            # Convert to MB
            total_mb = total // (1024 * 1024)
            used_mb = used // (1024 * 1024)
            free_mb = free // (1024 * 1024)
            
            usage_percent = (used / total) * 100
            
            details = {
                'total_mb': total_mb,
                'used_mb': used_mb,
                'free_mb': free_mb,
                'usage_percent': round(usage_percent, 1)
            }
            
            # Critical if less than 100MB free or >95% usage
            if free_mb < 100 or usage_percent > 95:
                return 'unhealthy', f'Low disk space: {free_mb}MB free ({usage_percent:.1f}% used)', details
            
            # Warning if less than 500MB free or >85% usage
            elif free_mb < 500 or usage_percent > 85:
                return 'degraded', f'Disk space warning: {free_mb}MB free ({usage_percent:.1f}% used)', details
            
            return 'healthy', f'Disk space OK: {free_mb}MB free ({usage_percent:.1f}% used)', details
            
        except Exception as e:
            return 'error', f'Failed to check disk space: {str(e)}', {}
    
    def check_memory_usage(self):
        """Check memory usage"""
        try:
            # Read memory info from /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            # Parse memory values
            mem_total = 0
            mem_available = 0
            
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1]) * 1024  # Convert KB to bytes
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) * 1024  # Convert KB to bytes
            
            if mem_total == 0:
                return 'error', 'Could not read memory information', {}
            
            mem_used = mem_total - mem_available
            usage_percent = (mem_used / mem_total) * 100
            
            details = {
                'total_mb': mem_total // (1024 * 1024),
                'used_mb': mem_used // (1024 * 1024),
                'available_mb': mem_available // (1024 * 1024),
                'usage_percent': round(usage_percent, 1)
            }
            
            # Critical if >95% memory usage
            if usage_percent > 95:
                return 'unhealthy', f'High memory usage: {usage_percent:.1f}%', details
            
            # Warning if >85% memory usage
            elif usage_percent > 85:
                return 'degraded', f'Memory usage warning: {usage_percent:.1f}%', details
            
            return 'healthy', f'Memory usage OK: {usage_percent:.1f}%', details
            
        except Exception as e:
            return 'error', f'Failed to check memory usage: {str(e)}', {}
    
    def check_log_files(self):
        """Check log file health"""
        try:
            log_dir = Path('/app/logs')
            
            if not log_dir.exists():
                return 'degraded', 'Log directory does not exist', {}
            
            # Check main log file
            main_log = log_dir / 'guard.log'
            monitor_log = log_dir / 'monitor.log'
            
            details = {'log_files': []}
            
            for log_file in [main_log, monitor_log]:
                if log_file.exists():
                    stat = log_file.stat()
                    file_age = time.time() - stat.st_mtime
                    
                    details['log_files'].append({
                        'file': str(log_file),
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'age_seconds': round(file_age, 1),
                        'exists': True
                    })
                    
                    # Check if log files are growing too large (>100MB)
                    if stat.st_size > 100 * 1024 * 1024:
                        return 'degraded', f'Large log file: {log_file.name} ({stat.st_size // (1024*1024)}MB)', details
                        
                else:
                    details['log_files'].append({
                        'file': str(log_file),
                        'exists': False
                    })
            
            return 'healthy', 'Log files are healthy', details
            
        except Exception as e:
            return 'error', f'Failed to check log files: {str(e)}', {}
    
    def check_health_monitor(self):
        """Check if health monitoring process is working"""
        try:
            health_status_file = Path('/app/tmp/health_status.json')
            
            if not health_status_file.exists():
                return 'degraded', 'Health monitor status file not found', {}
            
            # Check if file is recent
            file_age = time.time() - health_status_file.stat().st_mtime
            if file_age > 90:  # Health monitor updates every 30s, allow 90s tolerance
                return 'degraded', f'Health monitor status is stale ({file_age:.1f}s old)', {'file_age': file_age}
            
            # Read health status
            with open(health_status_file, 'r') as f:
                health_data = json.load(f)
            
            return 'healthy', 'Health monitor is active', health_data
            
        except Exception as e:
            return 'degraded', f'Health monitor check failed: {str(e)}', {}
    
    def check_configuration(self):
        """Check configuration validity"""
        try:
            config_dir = Path('/app/config')
            details = {'config_files': []}
            
            # Expected config files (optional)
            config_files = ['guard.yaml', 'security.yaml', 'regions.yaml']
            
            for config_file in config_files:
                file_path = config_dir / config_file
                if file_path.exists():
                    details['config_files'].append({
                        'file': config_file,
                        'exists': True,
                        'size': file_path.stat().st_size
                    })
                else:
                    details['config_files'].append({
                        'file': config_file,
                        'exists': False
                    })
            
            # Check environment variables
            required_env_vars = ['GUARD_MODE', 'GUARD_REGION', 'GUARD_ENVIRONMENT']
            missing_vars = []
            
            for var in required_env_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                return 'degraded', f'Missing environment variables: {", ".join(missing_vars)}', details
            
            return 'healthy', 'Configuration is valid', details
            
        except Exception as e:
            return 'error', f'Configuration check failed: {str(e)}', {}
    
    def check_dependencies(self):
        """Check that required dependencies are available"""
        try:
            import sys
            
            # Check Python version
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            # Try to import core modules
            missing_modules = []
            required_modules = [
                'asyncio',
                'json',
                'datetime',
                'logging',
                'pathlib'
            ]
            
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)
            
            details = {
                'python_version': python_version,
                'missing_modules': missing_modules
            }
            
            if missing_modules:
                return 'unhealthy', f'Missing required modules: {", ".join(missing_modules)}', details
            
            return 'healthy', f'Dependencies OK (Python {python_version})', details
            
        except Exception as e:
            return 'error', f'Dependency check failed: {str(e)}', {}


def main():
    """Main health check execution"""
    
    # Create health checker
    checker = HealthChecker()
    
    # Add all health checks
    checker.add_check('process', checker.check_process_health, critical=True)
    checker.add_check('disk_space', checker.check_disk_space, critical=True)
    checker.add_check('memory', checker.check_memory_usage, critical=True)
    checker.add_check('dependencies', checker.check_dependencies, critical=True)
    checker.add_check('configuration', checker.check_configuration, critical=False)
    checker.add_check('logs', checker.check_log_files, critical=False)
    checker.add_check('health_monitor', checker.check_health_monitor, critical=False)
    
    # Run all checks
    results = checker.run_all_checks()
    
    # Output results in JSON format for debugging
    if os.getenv('HEALTH_CHECK_VERBOSE', '').lower() in ['true', '1', 'yes']:
        print(json.dumps(results, indent=2))
    
    # Write results to file
    try:
        with open('/app/tmp/health_check_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass  # Don't fail health check if we can't write results
    
    # Determine exit code
    if results['overall_status'] == 'healthy':
        print(f"Health check PASSED - Status: {results['overall_status']}")
        sys.exit(0)
    elif results['overall_status'] == 'degraded':
        print(f"Health check WARNING - Status: {results['overall_status']}")
        # Exit 0 for degraded state (still considered healthy for container orchestration)
        sys.exit(0)
    else:
        print(f"Health check FAILED - Status: {results['overall_status']}")
        sys.exit(1)


if __name__ == "__main__":
    main()