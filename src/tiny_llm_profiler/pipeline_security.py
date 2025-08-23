"""
Pipeline Security and Threat Detection System
Advanced security monitoring and threat prevention for self-healing pipelines
"""

import hashlib
import hmac
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
import json
import ipaddress
from pathlib import Path

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    INJECTION = "injection"
    MALICIOUS_CODE = "malicious_code"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    RESOURCE_ABUSE = "resource_abuse"
    SUPPLY_CHAIN = "supply_chain"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


class SecurityAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    TERMINATE = "terminate"


@dataclass
class SecurityEvent:
    threat_type: ThreatType
    threat_level: ThreatLevel
    source: str
    description: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    action_taken: SecurityAction = SecurityAction.WARN
    blocked: bool = False


@dataclass
class SecurityRule:
    rule_id: str
    name: str
    description: str
    threat_type: ThreatType
    pattern: str
    action: SecurityAction
    enabled: bool = True


class SecurityDetector(ABC):
    @abstractmethod
    def detect_threats(self, data: Dict[str, Any]) -> List[SecurityEvent]:
        pass


class InjectionDetector(SecurityDetector):
    def __init__(self):
        self.sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)",
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\b(OR|AND)\b\s+\d+\s*=\s*\d+)",
            r"(--|#|/\*|\*/)",
            r"(\bEXEC\b|\bEXECUTE\b)",
        ]

        self.command_patterns = [
            r"(\b(rm|del|format|mkfs)\b)",
            r"(\b(wget|curl|nc|netcat)\b)",
            r"(\b(sudo|su|chmod|chown)\b)",
            r"(\$\(.*\)|\`.*\`)",
            r"(&&|\|\||;|\|)",
        ]

        self.script_patterns = [
            r"(<script[^>]*>.*</script>)",
            r"(javascript:|vbscript:|data:)",
            r"(eval\s*\(|setTimeout\s*\(|setInterval\s*\()",
            r"(document\.cookie|window\.location)",
        ]

    def detect_threats(self, data: Dict[str, Any]) -> List[SecurityEvent]:
        events = []

        # Check all string values in data
        for key, value in data.items():
            if isinstance(value, str):
                events.extend(self._check_injection_patterns(key, value))

        return events

    def _check_injection_patterns(self, field: str, value: str) -> List[SecurityEvent]:
        events = []

        # SQL Injection
        for pattern in self.sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                events.append(
                    SecurityEvent(
                        threat_type=ThreatType.INJECTION,
                        threat_level=ThreatLevel.HIGH,
                        source=f"field:{field}",
                        description=f"Potential SQL injection in {field}",
                        details={
                            "field": field,
                            "value": value[:100],  # Truncate for security
                            "pattern": pattern,
                            "injection_type": "sql",
                        },
                        action_taken=SecurityAction.BLOCK,
                    )
                )

        # Command Injection
        for pattern in self.command_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                events.append(
                    SecurityEvent(
                        threat_type=ThreatType.INJECTION,
                        threat_level=ThreatLevel.CRITICAL,
                        source=f"field:{field}",
                        description=f"Potential command injection in {field}",
                        details={
                            "field": field,
                            "value": value[:100],
                            "pattern": pattern,
                            "injection_type": "command",
                        },
                        action_taken=SecurityAction.TERMINATE,
                    )
                )

        # Script Injection (XSS)
        for pattern in self.script_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                events.append(
                    SecurityEvent(
                        threat_type=ThreatType.INJECTION,
                        threat_level=ThreatLevel.MEDIUM,
                        source=f"field:{field}",
                        description=f"Potential script injection in {field}",
                        details={
                            "field": field,
                            "value": value[:100],
                            "pattern": pattern,
                            "injection_type": "script",
                        },
                        action_taken=SecurityAction.BLOCK,
                    )
                )

        return events


class MaliciousCodeDetector(SecurityDetector):
    def __init__(self):
        self.malicious_patterns = [
            r"(eval\s*\(.*\))",
            r"(exec\s*\(.*\))",
            r"(os\.system|subprocess\.call|subprocess\.run)",
            r"(import\s+os|from\s+os\s+import)",
            r"(__import__|getattr|setattr|delattr)",
            r"(compile\s*\(|execfile\s*\()",
            r"(base64\.decode|base64\.b64decode)",
        ]

        self.suspicious_imports = [
            "socket",
            "urllib",
            "requests",
            "subprocess",
            "os",
            "sys",
            "pickle",
            "marshal",
            "types",
            "importlib",
        ]

    def detect_threats(self, data: Dict[str, Any]) -> List[SecurityEvent]:
        events = []

        # Check for malicious code patterns
        code_content = data.get("code", "")
        if isinstance(code_content, str):
            events.extend(self._analyze_code(code_content))

        # Check uploaded files
        if "files" in data:
            for file_info in data["files"]:
                events.extend(self._analyze_file(file_info))

        return events

    def _analyze_code(self, code: str) -> List[SecurityEvent]:
        events = []

        for pattern in self.malicious_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                events.append(
                    SecurityEvent(
                        threat_type=ThreatType.MALICIOUS_CODE,
                        threat_level=ThreatLevel.HIGH,
                        source="code_analysis",
                        description=f"Potentially malicious code pattern detected",
                        details={
                            "pattern": pattern,
                            "match": match.group(),
                            "line": code[: match.start()].count("\n") + 1,
                        },
                        action_taken=SecurityAction.QUARANTINE,
                    )
                )

        # Check for suspicious imports
        import_matches = re.finditer(r"import\s+(\w+)|from\s+(\w+)\s+import", code)
        for match in import_matches:
            module = match.group(1) or match.group(2)
            if module in self.suspicious_imports:
                events.append(
                    SecurityEvent(
                        threat_type=ThreatType.MALICIOUS_CODE,
                        threat_level=ThreatLevel.MEDIUM,
                        source="code_analysis",
                        description=f"Suspicious import detected: {module}",
                        details={
                            "module": module,
                            "line": code[: match.start()].count("\n") + 1,
                        },
                        action_taken=SecurityAction.WARN,
                    )
                )

        return events

    def _analyze_file(self, file_info: Dict[str, Any]) -> List[SecurityEvent]:
        events = []

        filename = file_info.get("filename", "")
        file_size = file_info.get("size", 0)
        file_hash = file_info.get("hash", "")

        # Check file extension
        dangerous_extensions = [".exe", ".bat", ".cmd", ".scr", ".pif", ".jar"]
        if any(filename.lower().endswith(ext) for ext in dangerous_extensions):
            events.append(
                SecurityEvent(
                    threat_type=ThreatType.MALICIOUS_CODE,
                    threat_level=ThreatLevel.HIGH,
                    source="file_upload",
                    description=f"Potentially dangerous file uploaded: {filename}",
                    details={
                        "filename": filename,
                        "size": file_size,
                        "hash": file_hash,
                    },
                    action_taken=SecurityAction.BLOCK,
                )
            )

        # Check file size
        if file_size > 100 * 1024 * 1024:  # 100MB
            events.append(
                SecurityEvent(
                    threat_type=ThreatType.RESOURCE_ABUSE,
                    threat_level=ThreatLevel.MEDIUM,
                    source="file_upload",
                    description=f"Large file upload detected: {file_size} bytes",
                    details={"filename": filename, "size": file_size},
                    action_taken=SecurityAction.WARN,
                )
            )

        return events


class AccessControlDetector(SecurityDetector):
    def __init__(self):
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, List[float]] = {}
        self.max_requests_per_minute = 60
        self.failed_attempts: Dict[str, int] = {}
        self.max_failed_attempts = 5

    def detect_threats(self, data: Dict[str, Any]) -> List[SecurityEvent]:
        events = []

        client_ip = data.get("client_ip", "")
        user_id = data.get("user_id", "")
        request_type = data.get("request_type", "")

        # IP-based detection
        if client_ip:
            events.extend(self._check_ip_threats(client_ip, request_type))

        # User-based detection
        if user_id:
            events.extend(self._check_user_threats(user_id, data))

        # Rate limiting
        if client_ip or user_id:
            identifier = client_ip or user_id
            events.extend(self._check_rate_limiting(identifier))

        return events

    def _check_ip_threats(self, ip: str, request_type: str) -> List[SecurityEvent]:
        events = []

        # Check if IP is blocked
        if ip in self.blocked_ips:
            events.append(
                SecurityEvent(
                    threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                    threat_level=ThreatLevel.HIGH,
                    source=f"ip:{ip}",
                    description=f"Request from blocked IP: {ip}",
                    details={"ip": ip, "request_type": request_type},
                    action_taken=SecurityAction.BLOCK,
                    blocked=True,
                )
            )

        # Check for private/internal IP ranges in external context
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private and request_type == "external":
                events.append(
                    SecurityEvent(
                        threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                        threat_level=ThreatLevel.MEDIUM,
                        source=f"ip:{ip}",
                        description=f"Private IP in external context: {ip}",
                        details={"ip": ip, "is_private": True},
                        action_taken=SecurityAction.WARN,
                    )
                )
        except ValueError:
            # Invalid IP format
            events.append(
                SecurityEvent(
                    threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.LOW,
                    source=f"ip:{ip}",
                    description=f"Invalid IP format: {ip}",
                    details={"ip": ip, "invalid_format": True},
                    action_taken=SecurityAction.WARN,
                )
            )

        return events

    def _check_user_threats(
        self, user_id: str, data: Dict[str, Any]
    ) -> List[SecurityEvent]:
        events = []

        # Check for authentication failures
        if data.get("auth_failed", False):
            self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1

            if self.failed_attempts[user_id] >= self.max_failed_attempts:
                events.append(
                    SecurityEvent(
                        threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                        threat_level=ThreatLevel.HIGH,
                        source=f"user:{user_id}",
                        description=f"Multiple failed login attempts: {user_id}",
                        details={
                            "user_id": user_id,
                            "failed_attempts": self.failed_attempts[user_id],
                        },
                        action_taken=SecurityAction.BLOCK,
                    )
                )
        else:
            # Reset failed attempts on successful auth
            self.failed_attempts.pop(user_id, None)

        # Check for privilege escalation attempts
        if data.get("privilege_escalation", False):
            events.append(
                SecurityEvent(
                    threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                    threat_level=ThreatLevel.CRITICAL,
                    source=f"user:{user_id}",
                    description=f"Privilege escalation attempt: {user_id}",
                    details={"user_id": user_id},
                    action_taken=SecurityAction.TERMINATE,
                )
            )

        return events

    def _check_rate_limiting(self, identifier: str) -> List[SecurityEvent]:
        events = []
        current_time = time.time()

        # Initialize rate tracking
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []

        # Add current request
        self.rate_limits[identifier].append(current_time)

        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - 60
        self.rate_limits[identifier] = [
            t for t in self.rate_limits[identifier] if t > cutoff_time
        ]

        # Check rate limit
        request_count = len(self.rate_limits[identifier])
        if request_count > self.max_requests_per_minute:
            events.append(
                SecurityEvent(
                    threat_type=ThreatType.RESOURCE_ABUSE,
                    threat_level=ThreatLevel.MEDIUM,
                    source=f"rate_limit:{identifier}",
                    description=f"Rate limit exceeded: {request_count} requests/minute",
                    details={
                        "identifier": identifier,
                        "request_count": request_count,
                        "limit": self.max_requests_per_minute,
                    },
                    action_taken=SecurityAction.BLOCK,
                )
            )

        return events

    def block_ip(self, ip: str) -> None:
        self.blocked_ips.add(ip)
        logger.warning(f"IP blocked: {ip}")

    def unblock_ip(self, ip: str) -> None:
        self.blocked_ips.discard(ip)
        logger.info(f"IP unblocked: {ip}")


class DataExfiltrationDetector(SecurityDetector):
    def __init__(self):
        self.sensitive_patterns = [
            r"(\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b)",  # Credit card
            r"(\b\d{3}-\d{2}-\d{4}\b)",  # SSN
            r"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)",  # Email
            r"(\b(?:password|passwd|pwd|secret|key|token)\s*[:=]\s*\S+)",  # Credentials
            r"(\b(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*\S+)",  # API keys
        ]

        self.data_volume_threshold = 10 * 1024 * 1024  # 10MB
        self.export_tracking: Dict[str, List[Tuple[float, int]]] = {}

    def detect_threats(self, data: Dict[str, Any]) -> List[SecurityEvent]:
        events = []

        # Check for sensitive data in output
        output_data = data.get("output", "")
        if isinstance(output_data, str):
            events.extend(self._check_sensitive_data(output_data))

        # Check for large data exports
        if "export_size" in data:
            events.extend(self._check_data_export(data))

        # Check for unusual data access patterns
        if "data_access" in data:
            events.extend(self._check_access_patterns(data["data_access"]))

        return events

    def _check_sensitive_data(self, content: str) -> List[SecurityEvent]:
        events = []

        for pattern in self.sensitive_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                events.append(
                    SecurityEvent(
                        threat_type=ThreatType.DATA_EXFILTRATION,
                        threat_level=ThreatLevel.HIGH,
                        source="output_analysis",
                        description="Sensitive data detected in output",
                        details={
                            "pattern_type": self._get_pattern_type(pattern),
                            "match_snippet": match.group()[:20] + "***",
                        },
                        action_taken=SecurityAction.BLOCK,
                    )
                )

        return events

    def _check_data_export(self, data: Dict[str, Any]) -> List[SecurityEvent]:
        events = []

        user_id = data.get("user_id", "unknown")
        export_size = data.get("export_size", 0)
        current_time = time.time()

        # Track export volumes
        if user_id not in self.export_tracking:
            self.export_tracking[user_id] = []

        self.export_tracking[user_id].append((current_time, export_size))

        # Clean old exports (older than 1 hour)
        cutoff_time = current_time - 3600
        self.export_tracking[user_id] = [
            (t, size) for t, size in self.export_tracking[user_id] if t > cutoff_time
        ]

        # Check total export volume
        total_exported = sum(size for _, size in self.export_tracking[user_id])

        if total_exported > self.data_volume_threshold:
            events.append(
                SecurityEvent(
                    threat_type=ThreatType.DATA_EXFILTRATION,
                    threat_level=ThreatLevel.CRITICAL,
                    source=f"user:{user_id}",
                    description=f"Large data export detected: {total_exported} bytes",
                    details={
                        "user_id": user_id,
                        "total_exported": total_exported,
                        "threshold": self.data_volume_threshold,
                        "export_count": len(self.export_tracking[user_id]),
                    },
                    action_taken=SecurityAction.BLOCK,
                )
            )

        return events

    def _check_access_patterns(
        self, access_data: Dict[str, Any]
    ) -> List[SecurityEvent]:
        events = []

        # Check for unusual access times
        access_hour = datetime.now().hour
        if access_hour < 6 or access_hour > 22:  # Outside business hours
            events.append(
                SecurityEvent(
                    threat_type=ThreatType.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.MEDIUM,
                    source="access_pattern",
                    description=f"Data access outside business hours: {access_hour}:00",
                    details={"access_hour": access_hour},
                    action_taken=SecurityAction.WARN,
                )
            )

        # Check for rapid sequential access
        access_count = access_data.get("sequential_access_count", 0)
        if access_count > 100:  # More than 100 rapid accesses
            events.append(
                SecurityEvent(
                    threat_type=ThreatType.DATA_EXFILTRATION,
                    threat_level=ThreatLevel.HIGH,
                    source="access_pattern",
                    description=f"Rapid sequential data access: {access_count} accesses",
                    details={"access_count": access_count},
                    action_taken=SecurityAction.WARN,
                )
            )

        return events

    def _get_pattern_type(self, pattern: str) -> str:
        if "credit" in pattern or r"\d{4}" in pattern:
            return "credit_card"
        elif "ssn" in pattern or r"\d{3}-\d{2}-\d{4}" in pattern:
            return "ssn"
        elif "@" in pattern:
            return "email"
        elif "password" in pattern:
            return "credentials"
        elif "api" in pattern or "token" in pattern:
            return "api_key"
        else:
            return "unknown"


class PipelineSecurityManager:
    def __init__(self):
        self.detectors = {
            ThreatType.INJECTION: InjectionDetector(),
            ThreatType.MALICIOUS_CODE: MaliciousCodeDetector(),
            ThreatType.UNAUTHORIZED_ACCESS: AccessControlDetector(),
            ThreatType.DATA_EXFILTRATION: DataExfiltrationDetector(),
        }

        self.security_events: List[SecurityEvent] = []
        self.blocked_requests: Set[str] = set()
        self.quarantined_items: List[Dict[str, Any]] = []

        # Configuration
        self.auto_block_enabled = True
        self.event_retention_hours = 24
        self.max_events = 10000

        # Statistics
        self.total_threats_detected = 0
        self.total_requests_blocked = 0
        self.total_items_quarantined = 0

    def analyze_request(
        self, request_data: Dict[str, Any]
    ) -> Tuple[bool, List[SecurityEvent]]:
        all_events = []
        should_block = False

        # Run all security detectors
        for threat_type, detector in self.detectors.items():
            try:
                events = detector.detect_threats(request_data)
                all_events.extend(events)

                # Check if any event requires blocking
                for event in events:
                    if event.action_taken in [
                        SecurityAction.BLOCK,
                        SecurityAction.TERMINATE,
                    ]:
                        should_block = True
                    elif event.action_taken == SecurityAction.QUARANTINE:
                        self._quarantine_item(request_data, event)

            except Exception as e:
                logger.error(f"Error in {threat_type.value} detector: {str(e)}")

        # Store events
        for event in all_events:
            self._record_security_event(event)

        # Block request if needed
        if should_block and self.auto_block_enabled:
            request_id = request_data.get("request_id", "unknown")
            self.blocked_requests.add(request_id)
            self.total_requests_blocked += 1

        return not should_block, all_events

    def _record_security_event(self, event: SecurityEvent) -> None:
        self.security_events.append(event)
        self.total_threats_detected += 1

        # Clean up old events
        cutoff_time = datetime.now() - timedelta(hours=self.event_retention_hours)
        self.security_events = [
            e for e in self.security_events if e.timestamp > cutoff_time
        ]

        # Limit total events
        if len(self.security_events) > self.max_events:
            excess = len(self.security_events) - self.max_events
            self.security_events = self.security_events[excess:]

        logger.warning(f"Security event: {event.description}")

    def _quarantine_item(
        self, request_data: Dict[str, Any], event: SecurityEvent
    ) -> None:
        quarantine_item = {
            "timestamp": datetime.now().isoformat(),
            "event": {
                "threat_type": event.threat_type.value,
                "description": event.description,
                "details": event.details,
            },
            "request_data": request_data,
            "quarantine_id": f"q_{int(time.time())}_{len(self.quarantined_items)}",
        }

        self.quarantined_items.append(quarantine_item)
        self.total_items_quarantined += 1

        logger.warning(f"Item quarantined: {quarantine_item['quarantine_id']}")

    def is_request_blocked(self, request_id: str) -> bool:
        return request_id in self.blocked_requests

    def unblock_request(self, request_id: str) -> bool:
        if request_id in self.blocked_requests:
            self.blocked_requests.remove(request_id)
            return True
        return False

    def get_security_summary(self) -> Dict[str, Any]:
        active_events = [e for e in self.security_events if not e.blocked]

        return {
            "total_threats_detected": self.total_threats_detected,
            "total_requests_blocked": self.total_requests_blocked,
            "total_items_quarantined": self.total_items_quarantined,
            "active_events_count": len(active_events),
            "blocked_requests_count": len(self.blocked_requests),
            "quarantined_items_count": len(self.quarantined_items),
            "events_by_threat_type": {
                threat_type.value: len(
                    [e for e in active_events if e.threat_type == threat_type]
                )
                for threat_type in ThreatType
            },
            "events_by_level": {
                level.value: len([e for e in active_events if e.threat_level == level])
                for level in ThreatLevel
            },
        }

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        recent_events = self.security_events[-limit:]

        return [
            {
                "threat_type": event.threat_type.value,
                "threat_level": event.threat_level.value,
                "source": event.source,
                "description": event.description,
                "details": event.details,
                "timestamp": event.timestamp.isoformat(),
                "action_taken": event.action_taken.value,
                "blocked": event.blocked,
            }
            for event in recent_events
        ]

    def get_quarantined_items(self) -> List[Dict[str, Any]]:
        return self.quarantined_items.copy()

    def release_quarantined_item(self, quarantine_id: str) -> bool:
        for i, item in enumerate(self.quarantined_items):
            if item.get("quarantine_id") == quarantine_id:
                self.quarantined_items.pop(i)
                logger.info(f"Released quarantined item: {quarantine_id}")
                return True
        return False


# Global security manager instance
_global_security_manager: Optional[PipelineSecurityManager] = None


def get_security_manager() -> PipelineSecurityManager:
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = PipelineSecurityManager()
    return _global_security_manager


def analyze_security_threat(
    request_data: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    manager = get_security_manager()
    allowed, events = manager.analyze_request(request_data)

    event_dicts = [
        {
            "threat_type": e.threat_type.value,
            "threat_level": e.threat_level.value,
            "description": e.description,
            "action_taken": e.action_taken.value,
        }
        for e in events
    ]

    return allowed, event_dicts


def get_security_status() -> Dict[str, Any]:
    manager = get_security_manager()
    return manager.get_security_summary()


def block_ip_address(ip: str) -> None:
    manager = get_security_manager()
    access_detector = manager.detectors.get(ThreatType.UNAUTHORIZED_ACCESS)
    if isinstance(access_detector, AccessControlDetector):
        access_detector.block_ip(ip)
