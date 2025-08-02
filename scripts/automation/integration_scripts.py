#!/usr/bin/env python3
"""
Integration scripts for external tools and services.

Provides integration capabilities for monitoring, alerting, and
development tools used with the tiny-llm-edge-profiler project.
"""

import json
import os
import requests
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart


@dataclass
class IntegrationConfig:
    """Configuration for external integrations"""
    slack_webhook_url: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    github_token: Optional[str] = None
    prometheus_url: Optional[str] = None
    grafana_url: Optional[str] = None
    jira_url: Optional[str] = None
    jira_username: Optional[str] = None
    jira_token: Optional[str] = None


class IntegrationManager:
    """Manage integrations with external tools and services"""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or self._load_config_from_env()
        self.logger = self._setup_logging()
    
    def _load_config_from_env(self) -> IntegrationConfig:
        """Load configuration from environment variables"""
        return IntegrationConfig(
            slack_webhook_url=os.getenv('SLACK_WEBHOOK_URL'),
            email_smtp_server=os.getenv('EMAIL_SMTP_SERVER'),
            email_smtp_port=int(os.getenv('EMAIL_SMTP_PORT', 587)),
            email_username=os.getenv('EMAIL_USERNAME'),
            email_password=os.getenv('EMAIL_PASSWORD'),
            github_token=os.getenv('GITHUB_TOKEN'),
            prometheus_url=os.getenv('PROMETHEUS_URL', 'http://localhost:9090'),
            grafana_url=os.getenv('GRAFANA_URL', 'http://localhost:3000'),
            jira_url=os.getenv('JIRA_URL'),
            jira_username=os.getenv('JIRA_USERNAME'),
            jira_token=os.getenv('JIRA_TOKEN')
        )
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)


class SlackIntegration:
    """Slack integration for notifications"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)
    
    def send_message(self, message: str, channel: Optional[str] = None, 
                    username: str = "Repository Bot", 
                    emoji: str = ":robot_face:") -> bool:
        """Send a message to Slack"""
        try:
            payload = {
                "text": message,
                "username": username,
                "icon_emoji": emoji
            }
            
            if channel:
                payload["channel"] = channel
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info("Slack message sent successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack message: {e}")
            return False
    
    def send_alert(self, alert_type: str, title: str, description: str, 
                  severity: str = "info") -> bool:
        """Send formatted alert to Slack"""
        
        emoji_map = {
            "critical": ":rotating_light:",
            "warning": ":warning:",
            "info": ":information_source:",
            "success": ":white_check_mark:"
        }
        
        color_map = {
            "critical": "#ff0000",
            "warning": "#ffaa00", 
            "info": "#0099ff",
            "success": "#00ff00"
        }
        
        attachment = {
            "color": color_map.get(severity, "#0099ff"),
            "title": f"{emoji_map.get(severity, ':bell:')} {title}",
            "text": description,
            "fields": [
                {
                    "title": "Alert Type",
                    "value": alert_type,
                    "short": True
                },
                {
                    "title": "Severity", 
                    "value": severity.upper(),
                    "short": True
                },
                {
                    "title": "Timestamp",
                    "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "short": True
                }
            ]
        }
        
        payload = {
            "attachments": [attachment]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
            return False


class EmailIntegration:
    """Email integration for notifications"""
    
    def __init__(self, smtp_server: str, smtp_port: int, 
                 username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)
    
    def send_email(self, to_addresses: List[str], subject: str, 
                  body: str, html_body: Optional[str] = None) -> bool:
        """Send email notification"""
        try:
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.username
            msg['To'] = ', '.join(to_addresses)
            
            # Add plain text part
            text_part = MimeText(body, 'plain')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MimeText(html_body, 'html')
                msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(f"Email sent to {', '.join(to_addresses)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def send_report_email(self, to_addresses: List[str], 
                         report_type: str, report_data: Dict[str, Any]) -> bool:
        """Send formatted report email"""
        
        subject = f"Repository Report - {report_type} - {datetime.utcnow().strftime('%Y-%m-%d')}"
        
        # Generate plain text body
        body_lines = [
            f"Repository Report: {report_type}",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "Summary:",
        ]
        
        if 'score' in report_data:
            body_lines.append(f"Overall Score: {report_data['score']}/{report_data.get('max_score', 100)}")
        
        if 'checks' in report_data:
            body_lines.append("\nDetailed Results:")
            for check_name, check_data in report_data['checks'].items():
                status = "✅" if check_data.get('score', 0) >= check_data.get('max_score', 10) * 0.8 else "❌"
                body_lines.append(f"{status} {check_name}: {check_data.get('score', 0)}/{check_data.get('max_score', 10)}")
        
        body = "\n".join(body_lines)
        
        # Generate HTML body
        html_body = self._generate_html_report(report_type, report_data)
        
        return self.send_email(to_addresses, subject, body, html_body)
    
    def _generate_html_report(self, report_type: str, report_data: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; margin: 10px 0; }}
                .check {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; }}
                .success {{ border-left-color: #28a745; }}
                .warning {{ border-left-color: #ffc107; }}
                .error {{ border-left-color: #dc3545; }}
                .score {{ font-size: 1.2em; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1 class="header">Repository Report: {report_type}</h1>
            <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
        """
        
        if 'score' in report_data:
            score_class = "success" if report_data['score'] >= 80 else "warning" if report_data['score'] >= 60 else "error"
            html += f'<p class="score {score_class}">Overall Score: {report_data["score"]}/{report_data.get("max_score", 100)}</p>'
        
        html += "</div>"
        
        if 'checks' in report_data:
            html += "<h2>Detailed Results</h2>"
            for check_name, check_data in report_data['checks'].items():
                check_class = "success" if check_data.get('score', 0) >= check_data.get('max_score', 10) * 0.8 else "warning"
                html += f'''
                <div class="check {check_class}">
                    <h3>{check_name.replace('_', ' ').title()}</h3>
                    <p>Score: {check_data.get('score', 0)}/{check_data.get('max_score', 10)}</p>
                '''
                
                if check_data.get('issues'):
                    html += "<ul>"
                    for issue in check_data['issues']:
                        html += f"<li>{issue}</li>"
                    html += "</ul>"
                
                html += "</div>"
        
        html += """
            </body>
        </html>
        """
        
        return html


class GitHubIntegration:
    """GitHub API integration"""
    
    def __init__(self, token: str, repository: str):
        self.token = token
        self.repository = repository  # format: owner/repo
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.logger = logging.getLogger(__name__)
    
    def create_issue(self, title: str, body: str, labels: List[str] = None) -> Optional[int]:
        """Create a GitHub issue"""
        try:
            url = f"https://api.github.com/repos/{self.repository}/issues"
            
            data = {
                'title': title,
                'body': body
            }
            
            if labels:
                data['labels'] = labels
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            issue_data = response.json()
            issue_number = issue_data['number']
            
            self.logger.info(f"Created GitHub issue #{issue_number}")
            return issue_number
            
        except Exception as e:
            self.logger.error(f"Failed to create GitHub issue: {e}")
            return None
    
    def add_comment(self, issue_number: int, comment: str) -> bool:
        """Add comment to existing issue"""
        try:
            url = f"https://api.github.com/repos/{self.repository}/issues/{issue_number}/comments"
            
            data = {'body': comment}
            
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            self.logger.info(f"Added comment to issue #{issue_number}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add comment to issue #{issue_number}: {e}")
            return False
    
    def get_repository_metrics(self) -> Dict[str, Any]:
        """Get repository metrics from GitHub API"""
        try:
            # Get repository info
            repo_url = f"https://api.github.com/repos/{self.repository}"
            repo_response = requests.get(repo_url, headers=self.headers)
            repo_response.raise_for_status()
            repo_data = repo_response.json()
            
            # Get pull requests
            pr_url = f"https://api.github.com/repos/{self.repository}/pulls"
            pr_response = requests.get(pr_url, headers=self.headers, 
                                     params={'state': 'all', 'per_page': 100})
            pr_response.raise_for_status()
            pr_data = pr_response.json()
            
            # Get issues
            issues_url = f"https://api.github.com/repos/{self.repository}/issues"
            issues_response = requests.get(issues_url, headers=self.headers,
                                         params={'state': 'all', 'per_page': 100})
            issues_response.raise_for_status()
            issues_data = issues_response.json()
            
            # Filter out PRs from issues (GitHub treats PRs as issues)
            actual_issues = [issue for issue in issues_data if not issue.get('pull_request')]
            
            metrics = {
                'stars': repo_data['stargazers_count'],
                'forks': repo_data['forks_count'],
                'open_issues': repo_data['open_issues_count'],
                'total_prs': len(pr_data),
                'total_issues': len(actual_issues),
                'language': repo_data['language'],
                'size_kb': repo_data['size'],
                'created_at': repo_data['created_at'],
                'updated_at': repo_data['updated_at'],
                'has_wiki': repo_data['has_wiki'],
                'has_pages': repo_data['has_pages']
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get repository metrics: {e}")
            return {}


class PrometheusIntegration:
    """Prometheus integration for metrics"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
    
    def query_metric(self, query: str) -> Optional[Dict[str, Any]]:
        """Query Prometheus metric"""
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            response = requests.get(url, params={'query': query})
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to query Prometheus: {e}")
            return None
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics"""
        metrics = {}
        
        # Define queries for application metrics
        queries = {
            'uptime': 'up{job="tiny-llm-profiler"}',
            'error_rate': 'rate(http_requests_total{status=~"5.."}[5m])',
            'response_time': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
            'memory_usage': 'process_resident_memory_bytes',
            'cpu_usage': 'rate(process_cpu_seconds_total[5m])'
        }
        
        for metric_name, query in queries.items():
            result = self.query_metric(query)
            if result and result['data']['result']:
                metrics[metric_name] = result['data']['result'][0]['value'][1]
        
        return metrics


class NotificationService:
    """Unified notification service"""
    
    def __init__(self, integration_manager: IntegrationManager):
        self.integration_manager = integration_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize integrations
        if integration_manager.config.slack_webhook_url:
            self.slack = SlackIntegration(integration_manager.config.slack_webhook_url)
        else:
            self.slack = None
            
        if all([integration_manager.config.email_smtp_server,
                integration_manager.config.email_username,
                integration_manager.config.email_password]):
            self.email = EmailIntegration(
                integration_manager.config.email_smtp_server,
                integration_manager.config.email_smtp_port,
                integration_manager.config.email_username,
                integration_manager.config.email_password
            )
        else:
            self.email = None
            
        if integration_manager.config.github_token:
            # Repository name would need to be configured
            self.github = GitHubIntegration(
                integration_manager.config.github_token,
                "owner/repo"  # This should be configurable
            )
        else:
            self.github = None
    
    def send_alert(self, alert_type: str, title: str, description: str,
                  severity: str = "info", channels: List[str] = None) -> Dict[str, bool]:
        """Send alert through multiple channels"""
        results = {}
        
        if not channels:
            channels = ['slack', 'email']
        
        if 'slack' in channels and self.slack:
            results['slack'] = self.slack.send_alert(alert_type, title, description, severity)
        
        if 'email' in channels and self.email:
            # Email configuration would need recipient list
            results['email'] = self.email.send_email(
                ['team@example.com'],  # This should be configurable
                f"ALERT: {title}",
                f"Alert Type: {alert_type}\nSeverity: {severity}\n\n{description}"
            )
        
        if 'github' in channels and self.github and severity in ['critical', 'warning']:
            # Create GitHub issue for critical/warning alerts
            labels = ['alert', severity]
            issue_number = self.github.create_issue(
                f"[ALERT] {title}",
                f"**Alert Type:** {alert_type}\n**Severity:** {severity}\n\n{description}",
                labels
            )
            results['github'] = issue_number is not None
        
        return results
    
    def send_report(self, report_type: str, report_data: Dict[str, Any],
                   channels: List[str] = None) -> Dict[str, bool]:
        """Send report through multiple channels"""
        results = {}
        
        if not channels:
            channels = ['email']
        
        if 'email' in channels and self.email:
            results['email'] = self.email.send_report_email(
                ['team@example.com'],  # This should be configurable
                report_type,
                report_data
            )
        
        if 'slack' in channels and self.slack:
            # Send summary to Slack
            summary = f"📊 {report_type} Report Generated\n"
            if 'score' in report_data:
                summary += f"Score: {report_data['score']}/{report_data.get('max_score', 100)}\n"
            summary += f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            
            results['slack'] = self.slack.send_message(summary)
        
        return results


def main():
    """Example usage of integration services"""
    # Initialize integration manager
    integration_manager = IntegrationManager()
    
    # Initialize notification service
    notification_service = NotificationService(integration_manager)
    
    # Example: Send test alert
    notification_service.send_alert(
        alert_type="test",
        title="Integration Test",
        description="Testing integration services",
        severity="info"
    )
    
    # Example: Send test report
    test_report = {
        'score': 85,
        'max_score': 100,
        'checks': {
            'test_check': {
                'score': 8,
                'max_score': 10,
                'issues': ['Minor issue found']
            }
        }
    }
    
    notification_service.send_report("Health Check", test_report)


if __name__ == '__main__':
    main()