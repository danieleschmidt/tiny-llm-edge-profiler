"""
Global Compliance Manager for Self-Healing Pipeline Guard
Multi-region compliance, data sovereignty, and regulatory adherence
"""

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import hashlib
import base64

logger = logging.getLogger(__name__)


class ComplianceRegion(Enum):
    EU = "eu"  # European Union (GDPR)
    US = "us"  # United States (CCPA, HIPAA)
    APAC = "apac"  # Asia-Pacific (PDPA, etc.)
    CANADA = "canada"  # PIPEDA
    BRAZIL = "brazil"  # LGPD
    AUSTRALIA = "australia"  # Privacy Act
    CHINA = "china"  # PIPL
    GLOBAL = "global"  # General compliance


class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


class ComplianceFramework(Enum):
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOX = "sox"  # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"  # Information Security Management
    NIST = "nist"  # NIST Cybersecurity Framework
    PDPA = "pdpa"  # Personal Data Protection Act
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados


@dataclass
class DataFlowRecord:
    data_id: str
    source_region: ComplianceRegion
    destination_region: ComplianceRegion
    data_classification: DataClassification
    processing_purpose: str
    legal_basis: str
    timestamp: datetime = field(default_factory=datetime.now)
    encrypted: bool = False
    anonymized: bool = False
    retention_period_days: int = 365


@dataclass
class ComplianceRule:
    rule_id: str
    framework: ComplianceFramework
    region: ComplianceRegion
    name: str
    description: str
    requirements: List[str]
    validation_func: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class ComplianceViolation:
    violation_id: str
    rule_id: str
    framework: ComplianceFramework
    description: str
    severity: str
    detected_at: datetime
    data_involved: List[str]
    remediation_steps: List[str]
    resolved: bool = False


class ComplianceValidator(ABC):
    @abstractmethod
    def validate(
        self, data_flow: DataFlowRecord, context: Dict[str, Any]
    ) -> List[ComplianceViolation]:
        pass


class GDPRValidator(ComplianceValidator):
    def __init__(self):
        self.lawful_bases = {
            "consent",
            "contract",
            "legal_obligation",
            "vital_interests",
            "public_task",
            "legitimate_interests",
        }
        self.special_categories = {
            "health",
            "race",
            "religion",
            "politics",
            "sexual_orientation",
            "biometric",
            "genetic",
        }

    def validate(
        self, data_flow: DataFlowRecord, context: Dict[str, Any]
    ) -> List[ComplianceViolation]:
        violations = []

        # Check lawful basis
        if data_flow.legal_basis not in self.lawful_bases:
            violations.append(
                ComplianceViolation(
                    violation_id=f"gdpr_lawful_basis_{data_flow.data_id}",
                    rule_id="gdpr_article_6",
                    framework=ComplianceFramework.GDPR,
                    description=f"Invalid lawful basis: {data_flow.legal_basis}",
                    severity="high",
                    detected_at=datetime.now(),
                    data_involved=[data_flow.data_id],
                    remediation_steps=[
                        "Review and establish valid lawful basis",
                        "Update privacy policy if necessary",
                        "Obtain consent if applicable",
                    ],
                )
            )

        # Check cross-border transfer requirements
        if (
            data_flow.source_region == ComplianceRegion.EU
            and data_flow.destination_region
            not in [ComplianceRegion.EU, ComplianceRegion.GLOBAL]
        ):

            adequacy_countries = {"us", "canada", "australia"}  # Simplified list
            if data_flow.destination_region.value not in adequacy_countries:
                if not data_flow.encrypted:
                    violations.append(
                        ComplianceViolation(
                            violation_id=f"gdpr_transfer_{data_flow.data_id}",
                            rule_id="gdpr_chapter_5",
                            framework=ComplianceFramework.GDPR,
                            description="Cross-border transfer without adequate protection",
                            severity="critical",
                            detected_at=datetime.now(),
                            data_involved=[data_flow.data_id],
                            remediation_steps=[
                                "Implement Standard Contractual Clauses (SCCs)",
                                "Enable encryption for data in transit",
                                "Conduct Transfer Impact Assessment (TIA)",
                                "Consider data localization",
                            ],
                        )
                    )

        # Check retention period
        if data_flow.retention_period_days > 2555:  # 7 years max for most data
            violations.append(
                ComplianceViolation(
                    violation_id=f"gdpr_retention_{data_flow.data_id}",
                    rule_id="gdpr_article_5",
                    framework=ComplianceFramework.GDPR,
                    description=f"Retention period too long: {data_flow.retention_period_days} days",
                    severity="medium",
                    detected_at=datetime.now(),
                    data_involved=[data_flow.data_id],
                    remediation_steps=[
                        "Review data retention policy",
                        "Implement automated data deletion",
                        "Document business justification for retention",
                    ],
                )
            )

        return violations


class CCPAValidator(ComplianceValidator):
    def validate(
        self, data_flow: DataFlowRecord, context: Dict[str, Any]
    ) -> List[ComplianceViolation]:
        violations = []

        # Check for sale of personal information
        if data_flow.processing_purpose.lower() in [
            "sale",
            "sharing",
            "advertising",
        ] and data_flow.data_classification in [
            DataClassification.PERSONAL,
            DataClassification.SENSITIVE_PERSONAL,
        ]:

            consumer_rights_notice = context.get("consumer_rights_notice", False)
            opt_out_mechanism = context.get("opt_out_mechanism", False)

            if not consumer_rights_notice or not opt_out_mechanism:
                violations.append(
                    ComplianceViolation(
                        violation_id=f"ccpa_sale_{data_flow.data_id}",
                        rule_id="ccpa_section_1798.120",
                        framework=ComplianceFramework.CCPA,
                        description="Sale of personal information without proper notices",
                        severity="high",
                        detected_at=datetime.now(),
                        data_involved=[data_flow.data_id],
                        remediation_steps=[
                            "Implement 'Do Not Sell My Personal Information' link",
                            "Update privacy policy with consumer rights",
                            "Provide opt-out mechanism",
                            "Train staff on consumer requests",
                        ],
                    )
                )

        return violations


class HIPAAValidator(ComplianceValidator):
    def validate(
        self, data_flow: DataFlowRecord, context: Dict[str, Any]
    ) -> List[ComplianceViolation]:
        violations = []

        # Check for PHI handling
        is_healthcare_context = context.get("healthcare_context", False)
        if (
            is_healthcare_context
            and data_flow.data_classification == DataClassification.SENSITIVE_PERSONAL
        ):

            # Encryption requirement
            if not data_flow.encrypted:
                violations.append(
                    ComplianceViolation(
                        violation_id=f"hipaa_encryption_{data_flow.data_id}",
                        rule_id="hipaa_164.312",
                        framework=ComplianceFramework.HIPAA,
                        description="PHI transmitted without encryption",
                        severity="critical",
                        detected_at=datetime.now(),
                        data_involved=[data_flow.data_id],
                        remediation_steps=[
                            "Implement AES-256 encryption",
                            "Use secure transmission protocols",
                            "Establish Business Associate Agreements",
                            "Conduct security risk assessment",
                        ],
                    )
                )

            # Access logging requirement
            access_logging_enabled = context.get("access_logging", False)
            if not access_logging_enabled:
                violations.append(
                    ComplianceViolation(
                        violation_id=f"hipaa_logging_{data_flow.data_id}",
                        rule_id="hipaa_164.312",
                        framework=ComplianceFramework.HIPAA,
                        description="Insufficient access logging for PHI",
                        severity="high",
                        detected_at=datetime.now(),
                        data_involved=[data_flow.data_id],
                        remediation_steps=[
                            "Enable comprehensive audit logging",
                            "Implement access monitoring",
                            "Regular log review procedures",
                            "Incident response procedures",
                        ],
                    )
                )

        return violations


class GlobalComplianceManager:
    def __init__(self):
        self.validators = {
            ComplianceFramework.GDPR: GDPRValidator(),
            ComplianceFramework.CCPA: CCPAValidator(),
            ComplianceFramework.HIPAA: HIPAAValidator(),
        }

        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.data_flows: List[DataFlowRecord] = []
        self.violations: List[ComplianceViolation] = []

        # Regional framework mappings
        self.region_frameworks = {
            ComplianceRegion.EU: [
                ComplianceFramework.GDPR,
                ComplianceFramework.ISO27001,
            ],
            ComplianceRegion.US: [
                ComplianceFramework.CCPA,
                ComplianceFramework.HIPAA,
                ComplianceFramework.SOX,
            ],
            ComplianceRegion.APAC: [
                ComplianceFramework.PDPA,
                ComplianceFramework.ISO27001,
            ],
            ComplianceRegion.CANADA: [ComplianceFramework.PIPEDA],
            ComplianceRegion.BRAZIL: [ComplianceFramework.LGPD],
            ComplianceRegion.GLOBAL: [
                ComplianceFramework.ISO27001,
                ComplianceFramework.NIST,
            ],
        }

        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setup default compliance rules"""
        default_rules = [
            ComplianceRule(
                rule_id="gdpr_article_6",
                framework=ComplianceFramework.GDPR,
                region=ComplianceRegion.EU,
                name="Lawful Basis for Processing",
                description="Processing must have a lawful basis under Article 6",
                requirements=[
                    "Identify lawful basis before processing",
                    "Document legal basis in privacy policy",
                    "Review basis if processing purpose changes",
                ],
                severity="high",
            ),
            ComplianceRule(
                rule_id="ccpa_consumer_rights",
                framework=ComplianceFramework.CCPA,
                region=ComplianceRegion.US,
                name="Consumer Rights Notice",
                description="Provide clear notice of consumer rights",
                requirements=[
                    "Right to know about personal information",
                    "Right to delete personal information",
                    "Right to opt-out of sale",
                    "Right to non-discrimination",
                ],
                severity="medium",
            ),
            ComplianceRule(
                rule_id="hipaa_safeguards",
                framework=ComplianceFramework.HIPAA,
                region=ComplianceRegion.US,
                name="Administrative, Physical, and Technical Safeguards",
                description="Implement required safeguards for PHI",
                requirements=[
                    "Access controls and user authentication",
                    "Encryption of PHI in transit and at rest",
                    "Audit logs and monitoring",
                    "Incident response procedures",
                ],
                severity="critical",
            ),
        ]

        for rule in default_rules:
            self.compliance_rules[rule.rule_id] = rule

    def register_data_flow(self, data_flow: DataFlowRecord) -> None:
        """Register a new data flow for compliance monitoring"""
        self.data_flows.append(data_flow)
        logger.info(f"Registered data flow: {data_flow.data_id}")

    def validate_data_flow(
        self, data_flow: DataFlowRecord, context: Optional[Dict[str, Any]] = None
    ) -> List[ComplianceViolation]:
        """Validate a data flow against applicable compliance frameworks"""
        context = context or {}
        violations = []

        # Determine applicable frameworks based on regions
        applicable_frameworks = set()

        for region in [data_flow.source_region, data_flow.destination_region]:
            if region in self.region_frameworks:
                applicable_frameworks.update(self.region_frameworks[region])

        # Run validation for each applicable framework
        for framework in applicable_frameworks:
            if framework in self.validators:
                framework_violations = self.validators[framework].validate(
                    data_flow, context
                )
                violations.extend(framework_violations)

        # Store violations
        self.violations.extend(violations)

        if violations:
            logger.warning(
                f"Compliance violations detected for data flow {data_flow.data_id}: {len(violations)}"
            )

        return violations

    def check_cross_border_transfer(
        self,
        source_region: ComplianceRegion,
        dest_region: ComplianceRegion,
        data_classification: DataClassification,
    ) -> Dict[str, Any]:
        """Check if cross-border transfer is compliant"""

        # Adequacy decisions (simplified)
        adequacy_mappings = {
            ComplianceRegion.EU: {
                ComplianceRegion.US,
                ComplianceRegion.CANADA,
                ComplianceRegion.AUSTRALIA,
                ComplianceRegion.GLOBAL,
            },
            ComplianceRegion.US: {
                ComplianceRegion.EU,
                ComplianceRegion.CANADA,
                ComplianceRegion.AUSTRALIA,
                ComplianceRegion.GLOBAL,
            },
        }

        is_adequate = (
            source_region == dest_region
            or dest_region in adequacy_mappings.get(source_region, set())
            or source_region in adequacy_mappings.get(dest_region, set())
        )

        requirements = []
        if not is_adequate:
            if source_region == ComplianceRegion.EU:
                requirements.extend(
                    [
                        "Standard Contractual Clauses (SCCs)",
                        "Transfer Impact Assessment (TIA)",
                        "Encryption in transit and at rest",
                        "Data Processing Agreement (DPA)",
                    ]
                )

            if data_classification in [
                DataClassification.PERSONAL,
                DataClassification.SENSITIVE_PERSONAL,
            ]:
                requirements.extend(
                    [
                        "Enhanced security measures",
                        "Regular compliance audits",
                        "Data subject rights mechanism",
                    ]
                )

        return {
            "is_adequate": is_adequate,
            "requirements": requirements,
            "risk_level": (
                "high"
                if not is_adequate
                and data_classification == DataClassification.SENSITIVE_PERSONAL
                else "medium"
            ),
        }

    def generate_privacy_notice(
        self,
        region: ComplianceRegion,
        data_types: List[DataClassification],
        processing_purposes: List[str],
    ) -> Dict[str, Any]:
        """Generate region-specific privacy notice"""

        notice_sections = {
            "data_collection": [],
            "processing_purposes": processing_purposes,
            "legal_basis": [],
            "data_sharing": [],
            "retention": [],
            "rights": [],
            "contact": [],
        }

        # Region-specific requirements
        if region == ComplianceRegion.EU:
            notice_sections["legal_basis"] = [
                "Consent (Article 6(1)(a))",
                "Contract performance (Article 6(1)(b))",
                "Legal obligation (Article 6(1)(c))",
                "Legitimate interests (Article 6(1)(f))",
            ]
            notice_sections["rights"] = [
                "Right of access (Article 15)",
                "Right to rectification (Article 16)",
                "Right to erasure (Article 17)",
                "Right to restrict processing (Article 18)",
                "Right to data portability (Article 20)",
                "Right to object (Article 21)",
            ]
            notice_sections["contact"] = [
                "Data Protection Officer contact",
                "Supervisory authority contact",
            ]

        elif region == ComplianceRegion.US:
            notice_sections["rights"] = [
                "Right to know about personal information collected",
                "Right to delete personal information",
                "Right to opt-out of the sale of personal information",
                "Right to non-discrimination",
            ]
            notice_sections["contact"] = [
                "Privacy Officer contact",
                "Consumer rights request process",
            ]

        # Data-specific sections
        for data_type in data_types:
            if data_type == DataClassification.SENSITIVE_PERSONAL:
                notice_sections["data_collection"].append(
                    "Special category personal data (with explicit consent)"
                )
            elif data_type == DataClassification.PERSONAL:
                notice_sections["data_collection"].append(
                    "Personal identification information"
                )

        return {
            "region": region.value,
            "sections": notice_sections,
            "last_updated": datetime.now().isoformat(),
            "applicable_frameworks": [
                f.value for f in self.region_frameworks.get(region, [])
            ],
        }

    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Generate compliance dashboard data"""

        # Calculate violation statistics
        violation_stats = {
            "total": len(self.violations),
            "resolved": len([v for v in self.violations if v.resolved]),
            "by_severity": {},
            "by_framework": {},
            "recent": [],
        }

        for violation in self.violations:
            # By severity
            if violation.severity not in violation_stats["by_severity"]:
                violation_stats["by_severity"][violation.severity] = 0
            violation_stats["by_severity"][violation.severity] += 1

            # By framework
            framework = violation.framework.value
            if framework not in violation_stats["by_framework"]:
                violation_stats["by_framework"][framework] = 0
            violation_stats["by_framework"][framework] += 1

        # Recent violations (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent_violations = [
            {
                "id": v.violation_id,
                "framework": v.framework.value,
                "severity": v.severity,
                "description": v.description,
                "detected_at": v.detected_at.isoformat(),
            }
            for v in self.violations
            if v.detected_at > week_ago
        ]
        violation_stats["recent"] = recent_violations[:10]  # Last 10

        # Data flow statistics
        flow_stats = {
            "total_flows": len(self.data_flows),
            "by_region": {},
            "by_classification": {},
            "cross_border_transfers": 0,
        }

        for flow in self.data_flows:
            # By region
            source = flow.source_region.value
            dest = flow.destination_region.value

            if source not in flow_stats["by_region"]:
                flow_stats["by_region"][source] = {"outbound": 0, "inbound": 0}
            if dest not in flow_stats["by_region"]:
                flow_stats["by_region"][dest] = {"outbound": 0, "inbound": 0}

            flow_stats["by_region"][source]["outbound"] += 1
            flow_stats["by_region"][dest]["inbound"] += 1

            # Cross-border transfers
            if source != dest:
                flow_stats["cross_border_transfers"] += 1

            # By classification
            classification = flow.data_classification.value
            if classification not in flow_stats["by_classification"]:
                flow_stats["by_classification"][classification] = 0
            flow_stats["by_classification"][classification] += 1

        return {
            "timestamp": datetime.now().isoformat(),
            "violation_stats": violation_stats,
            "flow_stats": flow_stats,
            "compliance_score": self._calculate_compliance_score(),
            "active_frameworks": list(
                set(
                    f.value
                    for frameworks in self.region_frameworks.values()
                    for f in frameworks
                )
            ),
            "monitored_regions": [r.value for r in self.region_frameworks.keys()],
        }

    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)"""
        if not self.data_flows:
            return 100.0

        total_flows = len(self.data_flows)

        # Count unresolved violations
        unresolved_violations = len([v for v in self.violations if not v.resolved])

        # Weight by severity
        severity_weights = {"low": 1, "medium": 2, "high": 4, "critical": 8}
        weighted_violations = sum(
            severity_weights.get(v.severity, 2)
            for v in self.violations
            if not v.resolved
        )

        # Calculate score
        max_possible_violations = total_flows * 8  # Assuming worst case
        if max_possible_violations == 0:
            return 100.0

        violation_impact = min(
            100, (weighted_violations / max_possible_violations) * 100
        )
        compliance_score = max(0, 100 - violation_impact)

        return round(compliance_score, 1)

    def export_compliance_report(self, format: str = "json") -> str:
        """Export comprehensive compliance report"""

        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_period": "all_time",
                "format_version": "1.0",
            },
            "summary": self.get_compliance_dashboard(),
            "data_flows": [
                {
                    "data_id": flow.data_id,
                    "source_region": flow.source_region.value,
                    "destination_region": flow.destination_region.value,
                    "classification": flow.data_classification.value,
                    "purpose": flow.processing_purpose,
                    "legal_basis": flow.legal_basis,
                    "encrypted": flow.encrypted,
                    "anonymized": flow.anonymized,
                    "retention_days": flow.retention_period_days,
                    "timestamp": flow.timestamp.isoformat(),
                }
                for flow in self.data_flows
            ],
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "rule_id": v.rule_id,
                    "framework": v.framework.value,
                    "description": v.description,
                    "severity": v.severity,
                    "detected_at": v.detected_at.isoformat(),
                    "data_involved": v.data_involved,
                    "remediation_steps": v.remediation_steps,
                    "resolved": v.resolved,
                }
                for v in self.violations
            ],
            "recommendations": self._generate_recommendations(),
        }

        if format.lower() == "json":
            return json.dumps(report_data, indent=2)
        else:
            # Could add other formats like CSV, PDF, etc.
            return json.dumps(report_data, indent=2)

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate compliance improvement recommendations"""
        recommendations = []

        # Analyze violation patterns
        unresolved_violations = [v for v in self.violations if not v.resolved]

        if unresolved_violations:
            # Framework-specific recommendations
            framework_counts = {}
            for violation in unresolved_violations:
                framework = violation.framework.value
                if framework not in framework_counts:
                    framework_counts[framework] = 0
                framework_counts[framework] += 1

            for framework, count in framework_counts.items():
                if count >= 3:
                    recommendations.append(
                        {
                            "priority": "high",
                            "category": "framework_compliance",
                            "title": f"Address {framework.upper()} compliance gaps",
                            "description": f"Multiple violations detected for {framework.upper()}",
                            "action_items": [
                                f"Review {framework.upper()} requirements",
                                "Conduct compliance audit",
                                "Implement missing controls",
                                "Train staff on compliance requirements",
                            ],
                        }
                    )

        # Cross-border transfer recommendations
        cross_border_flows = [
            flow
            for flow in self.data_flows
            if flow.source_region != flow.destination_region
        ]

        if cross_border_flows:
            unencrypted_transfers = [
                flow for flow in cross_border_flows if not flow.encrypted
            ]

            if unencrypted_transfers:
                recommendations.append(
                    {
                        "priority": "critical",
                        "category": "data_protection",
                        "title": "Encrypt cross-border data transfers",
                        "description": f"{len(unencrypted_transfers)} unencrypted international transfers detected",
                        "action_items": [
                            "Implement end-to-end encryption",
                            "Use secure transmission protocols",
                            "Review data transfer agreements",
                            "Consider data localization options",
                        ],
                    }
                )

        return recommendations


# Global compliance manager instance
_global_compliance_manager: Optional[GlobalComplianceManager] = None


def get_compliance_manager() -> GlobalComplianceManager:
    global _global_compliance_manager
    if _global_compliance_manager is None:
        _global_compliance_manager = GlobalComplianceManager()
    return _global_compliance_manager


def validate_data_processing(
    data_id: str,
    source_region: str,
    dest_region: str,
    data_classification: str,
    purpose: str,
    legal_basis: str = "legitimate_interests",
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Validate data processing for compliance"""
    manager = get_compliance_manager()

    data_flow = DataFlowRecord(
        data_id=data_id,
        source_region=ComplianceRegion(source_region),
        destination_region=ComplianceRegion(dest_region),
        data_classification=DataClassification(data_classification),
        processing_purpose=purpose,
        legal_basis=legal_basis,
    )

    violations = manager.validate_data_flow(data_flow, context)

    return [
        {
            "violation_id": v.violation_id,
            "framework": v.framework.value,
            "description": v.description,
            "severity": v.severity,
            "remediation_steps": v.remediation_steps,
        }
        for v in violations
    ]


def get_compliance_status() -> Dict[str, Any]:
    """Get current compliance status"""
    manager = get_compliance_manager()
    return manager.get_compliance_dashboard()


def generate_privacy_notice(
    region: str, data_types: List[str], purposes: List[str]
) -> Dict[str, Any]:
    """Generate privacy notice for specific region"""
    manager = get_compliance_manager()

    region_enum = ComplianceRegion(region)
    classification_enums = [DataClassification(dt) for dt in data_types]

    return manager.generate_privacy_notice(region_enum, classification_enums, purposes)
