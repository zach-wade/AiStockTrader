"""
Compliance-specific features and regulatory reporting for audit logs.

This module provides specialized functionality for regulatory compliance,
including automated compliance checks, regulatory reporting, and compliance
violation detection and remediation.
"""

import csv
import json
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from io import StringIO
from typing import Any

from .config import AuditConfig
from .events import ComplianceRegulation
from .exceptions import AuditComplianceError
from .logger import AuditLogger
from .storage import AuditStorage


class ViolationSeverity(Enum):
    """Compliance violation severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceViolation:
    """Represents a compliance violation detected in audit logs."""

    violation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    regulation: ComplianceRegulation = ComplianceRegulation.SOX
    rule_id: str = ""
    rule_description: str = ""
    severity: ViolationSeverity = ViolationSeverity.MEDIUM
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_ids: list[str] = field(default_factory=list)
    affected_users: list[str] = field(default_factory=list)
    affected_resources: list[str] = field(default_factory=list)
    violation_details: dict[str, Any] = field(default_factory=dict)
    remediation_required: bool = True
    remediation_deadline: datetime | None = None
    remediation_actions: list[str] = field(default_factory=list)
    status: str = "open"  # open, investigating, remediated, closed
    assigned_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            "violation_id": self.violation_id,
            "regulation": self.regulation.value,
            "rule_id": self.rule_id,
            "rule_description": self.rule_description,
            "severity": self.severity.value,
            "detected_at": self.detected_at.isoformat(),
            "event_ids": self.event_ids,
            "affected_users": self.affected_users,
            "affected_resources": self.affected_resources,
            "violation_details": self.violation_details,
            "remediation_required": self.remediation_required,
            "remediation_deadline": (
                self.remediation_deadline.isoformat() if self.remediation_deadline else None
            ),
            "remediation_actions": self.remediation_actions,
            "status": self.status,
            "assigned_to": self.assigned_to,
        }


class ComplianceChecker:
    """
    Base class for compliance checking implementations.

    Provides framework for implementing regulation-specific compliance
    checks and violation detection.
    """

    def __init__(self, regulation: ComplianceRegulation) -> None:
        """
        Initialize compliance checker.

        Args:
            regulation: Target compliance regulation
        """
        self.regulation = regulation

    def check_compliance(
        self, events: Iterator[dict[str, Any]], start_date: datetime, end_date: datetime
    ) -> list[ComplianceViolation]:
        """
        Check events for compliance violations.

        Args:
            events: Iterator of audit events to check
            start_date: Start of compliance period
            end_date: End of compliance period

        Returns:
            List of compliance violations found
        """
        violations = []

        # Group events by type for efficient processing
        events_by_type = defaultdict(list)
        for event in events:
            events_by_type[event.get("event_type", "unknown")].append(event)

        # Run regulation-specific checks
        violations.extend(self._check_financial_controls(events_by_type, start_date, end_date))
        violations.extend(self._check_access_controls(events_by_type, start_date, end_date))
        violations.extend(self._check_data_integrity(events_by_type, start_date, end_date))
        violations.extend(self._check_audit_trails(events_by_type, start_date, end_date))

        return violations

    def _check_financial_controls(
        self,
        events_by_type: dict[str, list[dict[str, Any]]],
        start_date: datetime,
        end_date: datetime,
    ) -> list[ComplianceViolation]:
        """Check financial control compliance."""
        return []  # Override in subclasses

    def _check_access_controls(
        self,
        events_by_type: dict[str, list[dict[str, Any]]],
        start_date: datetime,
        end_date: datetime,
    ) -> list[ComplianceViolation]:
        """Check access control compliance."""
        return []  # Override in subclasses

    def _check_data_integrity(
        self,
        events_by_type: dict[str, list[dict[str, Any]]],
        start_date: datetime,
        end_date: datetime,
    ) -> list[ComplianceViolation]:
        """Check data integrity compliance."""
        return []  # Override in subclasses

    def _check_audit_trails(
        self,
        events_by_type: dict[str, list[dict[str, Any]]],
        start_date: datetime,
        end_date: datetime,
    ) -> list[ComplianceViolation]:
        """Check audit trail compliance."""
        return []  # Override in subclasses


class SOXComplianceChecker(ComplianceChecker):
    """Sarbanes-Oxley Act compliance checker."""

    def __init__(self) -> None:
        super().__init__(ComplianceRegulation.SOX)

    def _check_financial_controls(
        self,
        events_by_type: dict[str, list[dict[str, Any]]],
        start_date: datetime,
        end_date: datetime,
    ) -> list[ComplianceViolation]:
        """Check SOX financial control requirements."""
        violations = []

        # Check for unauthorized financial transactions
        order_events = events_by_type.get("order_operation", [])
        for event in order_events:
            if self._is_unauthorized_financial_transaction(event):
                violation = ComplianceViolation(
                    regulation=ComplianceRegulation.SOX,
                    rule_id="SOX-302",
                    rule_description="Unauthorized financial transaction detected",
                    severity=ViolationSeverity.HIGH,
                    event_ids=(
                        [str(event.get("event_id"))] if event.get("event_id") is not None else []
                    ),
                    affected_users=(
                        [str(event.get("user_id"))] if event.get("user_id") is not None else []
                    ),
                    violation_details={
                        "transaction_amount": event.get("resource_details", {}).get("quantity")
                    },
                    remediation_actions=[
                        "Review transaction authorization",
                        "Audit user permissions",
                    ],
                )
                violations.append(violation)

        # Check for financial data modifications without proper controls
        config_events = events_by_type.get("configuration_change", [])
        for event in config_events:
            if self._is_financial_config_change_without_approval(event):
                violation = ComplianceViolation(
                    regulation=ComplianceRegulation.SOX,
                    rule_id="SOX-404",
                    rule_description="Financial configuration changed without proper approval",
                    severity=ViolationSeverity.CRITICAL,
                    event_ids=(
                        [str(event.get("event_id"))] if event.get("event_id") is not None else []
                    ),
                    affected_users=(
                        [str(event.get("user_id"))] if event.get("user_id") is not None else []
                    ),
                    violation_details={
                        "config_key": event.get("resource_details", {}).get("config_key")
                    },
                    remediation_actions=[
                        "Verify approval process",
                        "Review change management controls",
                    ],
                )
                violations.append(violation)

        return violations

    def _check_access_controls(
        self,
        events_by_type: dict[str, list[dict[str, Any]]],
        start_date: datetime,
        end_date: datetime,
    ) -> list[ComplianceViolation]:
        """Check SOX access control requirements."""
        violations = []

        # Check for failed authentication attempts (potential security breach)
        auth_events = events_by_type.get("authentication_attempt", [])
        failed_attempts = [
            e for e in auth_events if not e.get("resource_details", {}).get("login_success")
        ]

        # Group by user and check for excessive failed attempts
        failed_by_user = defaultdict(list)
        for event in failed_attempts:
            user_id = event.get("user_id")
            if user_id:
                failed_by_user[user_id].append(event)

        for user_id, failures in failed_by_user.items():
            if len(failures) > 5:  # More than 5 failed attempts
                violation = ComplianceViolation(
                    regulation=ComplianceRegulation.SOX,
                    rule_id="SOX-Access-01",
                    rule_description="Excessive failed login attempts detected",
                    severity=ViolationSeverity.HIGH,
                    event_ids=[
                        str(e.get("event_id")) for e in failures if e.get("event_id") is not None
                    ],
                    affected_users=[user_id],
                    violation_details={"failed_attempts": len(failures)},
                    remediation_actions=[
                        "Review user access",
                        "Investigate potential security breach",
                    ],
                )
                violations.append(violation)

        return violations

    def _check_audit_trails(
        self,
        events_by_type: dict[str, list[dict[str, Any]]],
        start_date: datetime,
        end_date: datetime,
    ) -> list[ComplianceViolation]:
        """Check SOX audit trail requirements."""
        violations = []

        # Check for gaps in audit trail (missing events)
        all_events = []
        for event_list in events_by_type.values():
            all_events.extend(event_list)

        # Sort events by timestamp
        sorted_events = sorted(all_events, key=lambda e: e.get("timestamp", ""))

        # Check for significant time gaps (more than 1 hour with no events)
        for i in range(1, len(sorted_events)):
            current_time = datetime.fromisoformat(
                sorted_events[i].get("timestamp", "").replace("Z", "+00:00")
            )
            prev_time = datetime.fromisoformat(
                sorted_events[i - 1].get("timestamp", "").replace("Z", "+00:00")
            )

            time_gap = current_time - prev_time
            if time_gap > timedelta(hours=1):
                violation = ComplianceViolation(
                    regulation=ComplianceRegulation.SOX,
                    rule_id="SOX-Audit-01",
                    rule_description="Significant gap in audit trail detected",
                    severity=ViolationSeverity.MEDIUM,
                    event_ids=[],
                    violation_details={
                        "gap_start": prev_time.isoformat(),
                        "gap_end": current_time.isoformat(),
                        "gap_duration_hours": time_gap.total_seconds() / 3600,
                    },
                    remediation_actions=[
                        "Investigate audit logging system",
                        "Verify system availability",
                    ],
                )
                violations.append(violation)

        return violations

    def _is_unauthorized_financial_transaction(self, event: dict[str, Any]) -> bool:
        """Check if transaction is unauthorized."""
        # Example logic - check for high-value transactions without proper authorization
        resource_details = event.get("resource_details", {})
        quantity = resource_details.get("quantity")

        if quantity:
            try:
                amount = float(quantity)
                # Flag transactions over $100,000 without explicit approval
                return amount > 100000 and not resource_details.get("approved_by")
            except (ValueError, TypeError):
                pass

        return False

    def _is_financial_config_change_without_approval(self, event: dict[str, Any]) -> bool:
        """Check if financial configuration change lacks proper approval."""
        resource_details = event.get("resource_details", {})
        config_key = resource_details.get("config_key", "")

        # Check if this is a financial configuration
        financial_configs = ["risk_limit", "position_limit", "trading_limit", "commission_rate"]
        is_financial_config = any(fc in config_key.lower() for fc in financial_configs)

        if is_financial_config:
            return not resource_details.get("approved_by")

        return False


class GDPRComplianceChecker(ComplianceChecker):
    """GDPR compliance checker."""

    def __init__(self) -> None:
        super().__init__(ComplianceRegulation.GDPR)

    def _check_data_integrity(
        self,
        events_by_type: dict[str, list[dict[str, Any]]],
        start_date: datetime,
        end_date: datetime,
    ) -> list[ComplianceViolation]:
        """Check GDPR data protection requirements."""
        violations = []

        # Check for data retention violations
        all_events = []
        for event_list in events_by_type.values():
            all_events.extend(event_list)

        # Check events older than GDPR retention limits
        retention_limit = datetime.now(UTC) - timedelta(days=365)  # 1 year default

        old_events = [
            e
            for e in all_events
            if self._has_personal_data(e)
            and datetime.fromisoformat(e.get("timestamp", "").replace("Z", "+00:00"))
            < retention_limit
        ]

        if old_events:
            violation = ComplianceViolation(
                regulation=ComplianceRegulation.GDPR,
                rule_id="GDPR-Art5",
                rule_description="Personal data retained beyond lawful period",
                severity=ViolationSeverity.HIGH,
                event_ids=[
                    str(e.get("event_id")) for e in old_events[:10] if e.get("event_id") is not None
                ],  # Limit to first 10
                violation_details={"expired_events_count": len(old_events)},
                remediation_actions=["Delete expired personal data", "Review retention policies"],
            )
            violations.append(violation)

        return violations

    def _has_personal_data(self, event: dict[str, Any]) -> bool:
        """Check if event contains personal data."""
        # Check for common personal data fields
        personal_data_indicators = ["user_id", "email", "phone", "ip_address"]

        for field in personal_data_indicators:
            if event.get(field) or event.get("context", {}).get(field):
                return True

        return False


class ComplianceReporter:
    """
    Generates compliance reports for various regulatory requirements.

    Provides automated report generation, violation summaries, and
    regulatory filing support.
    """

    def __init__(self, audit_logger: AuditLogger, storage: AuditStorage, config: AuditConfig):
        """
        Initialize compliance reporter.

        Args:
            audit_logger: Audit logger instance
            storage: Audit storage backend
            config: Audit configuration
        """
        self.audit_logger = audit_logger
        self.storage = storage
        self.config = config

        # Initialize compliance checkers
        self.checkers = {
            ComplianceRegulation.SOX: SOXComplianceChecker(),
            ComplianceRegulation.GDPR: GDPRComplianceChecker(),
            # Add more checkers as needed
        }

    def generate_compliance_report(
        self,
        regulation: ComplianceRegulation,
        start_date: datetime,
        end_date: datetime,
        output_format: str = "json",
    ) -> dict[str, Any]:
        """
        Generate comprehensive compliance report.

        Args:
            regulation: Target regulation
            start_date: Report start date
            end_date: Report end date
            output_format: Output format ('json', 'csv', 'xml')

        Returns:
            Compliance report data
        """
        try:
            # Query audit events for the period
            events = self.storage.query(start_time=start_date, end_time=end_date)

            # Convert iterator to list for processing
            events_list = list(events)

            # Run compliance checks
            checker = self.checkers.get(regulation)
            violations = (
                checker.check_compliance(iter(events_list), start_date, end_date) if checker else []
            )

            # Generate report data
            report_data = {
                "report_id": str(uuid.uuid4()),
                "regulation": regulation.value,
                "period": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
                "generated_at": datetime.now(UTC).isoformat(),
                "summary": {
                    "total_events_reviewed": len(events_list),
                    "violations_found": len(violations),
                    "critical_violations": len(
                        [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
                    ),
                    "high_violations": len(
                        [v for v in violations if v.severity == ViolationSeverity.HIGH]
                    ),
                    "medium_violations": len(
                        [v for v in violations if v.severity == ViolationSeverity.MEDIUM]
                    ),
                    "low_violations": len(
                        [v for v in violations if v.severity == ViolationSeverity.LOW]
                    ),
                },
                "violations": [v.to_dict() for v in violations],
                "event_statistics": self._generate_event_statistics(events_list),
                "recommendations": self._generate_recommendations(violations),
                "format": output_format,
            }

            return report_data

        except Exception as e:
            raise AuditComplianceError(
                f"Failed to generate compliance report: {e}",
                regulation=regulation.value,
                requirement="report_generation",
            )

    def generate_violation_summary(self, violations: list[ComplianceViolation]) -> dict[str, Any]:
        """
        Generate summary of compliance violations.

        Args:
            violations: List of compliance violations

        Returns:
            Violation summary data
        """
        summary: dict[str, Any] = {
            "total_violations": len(violations),
            "by_regulation": defaultdict(int),
            "by_severity": defaultdict(int),
            "by_status": defaultdict(int),
            "remediation_required": 0,
            "overdue_remediations": 0,
        }

        now = datetime.now(UTC)

        for violation in violations:
            summary["by_regulation"][violation.regulation.value] += 1
            summary["by_severity"][violation.severity.value] += 1
            summary["by_status"][violation.status] += 1

            if violation.remediation_required:
                summary["remediation_required"] += 1

                if violation.remediation_deadline and violation.remediation_deadline < now:
                    summary["overdue_remediations"] += 1

        return summary

    def export_report(
        self, report_data: dict[str, Any], output_format: str, file_path: str | None = None
    ) -> str:
        """
        Export report to specified format.

        Args:
            report_data: Report data to export
            output_format: Export format ('json', 'csv', 'xml')
            file_path: Optional file path to save report

        Returns:
            Exported report content as string
        """
        if output_format.lower() == "json":
            content = json.dumps(report_data, indent=2, default=str)
        elif output_format.lower() == "csv":
            content = self._export_csv(report_data)
        elif output_format.lower() == "xml":
            content = self._export_xml(report_data)
        else:
            raise AuditComplianceError(
                f"Unsupported export format: {output_format}", requirement="report_export"
            )

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def _generate_event_statistics(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate statistics about audit events."""
        stats: dict[str, Any] = {
            "events_by_type": defaultdict(int),
            "events_by_user": defaultdict(int),
            "events_by_severity": defaultdict(int),
            "critical_events": 0,
            "authentication_events": 0,
            "financial_operations": 0,
        }

        for event in events:
            event_type = event.get("event_type", "unknown")
            user_id = event.get("user_id", "unknown")
            severity = event.get("severity", "unknown")
            is_critical = event.get("is_critical", False)

            stats["events_by_type"][event_type] += 1
            stats["events_by_user"][user_id] += 1
            stats["events_by_severity"][severity] += 1

            if is_critical:
                stats["critical_events"] += 1

            if "authentication" in event_type:
                stats["authentication_events"] += 1

            if event_type in ["order_operation", "position_operation", "portfolio_operation"]:
                stats["financial_operations"] += 1

        return stats

    def _generate_recommendations(self, violations: list[ComplianceViolation]) -> list[str]:
        """Generate recommendations based on violations found."""
        recommendations = []

        violation_counts: defaultdict[str, int] = defaultdict(int)
        for violation in violations:
            violation_counts[violation.rule_id] += 1

        # Generate recommendations based on violation patterns
        if violation_counts.get("SOX-302", 0) > 0:
            recommendations.append(
                "Implement stronger financial transaction authorization controls"
            )

        if violation_counts.get("SOX-404", 0) > 0:
            recommendations.append(
                "Enhance change management processes for financial configurations"
            )

        if violation_counts.get("GDPR-Art5", 0) > 0:
            recommendations.append("Review and update data retention policies")

        if any("Access" in rule_id for rule_id in violation_counts.keys()):
            recommendations.append("Strengthen access control and authentication mechanisms")

        if any("Audit" in rule_id for rule_id in violation_counts.keys()):
            recommendations.append("Improve audit logging system reliability and coverage")

        return recommendations

    def _export_csv(self, report_data: dict[str, Any]) -> str:
        """Export report data as CSV."""
        output = StringIO()

        # Write summary
        writer = csv.writer(output)
        writer.writerow(["Compliance Report Summary"])
        writer.writerow(["Report ID", report_data["report_id"]])
        writer.writerow(["Regulation", report_data["regulation"]])
        writer.writerow(["Period Start", report_data["period"]["start_date"]])
        writer.writerow(["Period End", report_data["period"]["end_date"]])
        writer.writerow([])

        # Write violations
        writer.writerow(["Violations"])
        writer.writerow(
            ["Violation ID", "Rule ID", "Severity", "Status", "Description", "Affected Users"]
        )

        for violation in report_data["violations"]:
            writer.writerow(
                [
                    violation["violation_id"],
                    violation["rule_id"],
                    violation["severity"],
                    violation["status"],
                    violation["rule_description"],
                    ", ".join(violation["affected_users"]),
                ]
            )

        return output.getvalue()

    def _export_xml(self, report_data: dict[str, Any]) -> str:
        """Export report data as XML."""
        root = ET.Element("ComplianceReport")

        # Add header
        header = ET.SubElement(root, "Header")
        ET.SubElement(header, "ReportId").text = report_data["report_id"]
        ET.SubElement(header, "Regulation").text = report_data["regulation"]
        ET.SubElement(header, "GeneratedAt").text = report_data["generated_at"]

        # Add period
        period = ET.SubElement(root, "Period")
        ET.SubElement(period, "StartDate").text = report_data["period"]["start_date"]
        ET.SubElement(period, "EndDate").text = report_data["period"]["end_date"]

        # Add violations
        violations = ET.SubElement(root, "Violations")
        for violation_data in report_data["violations"]:
            violation_elem = ET.SubElement(violations, "Violation")
            ET.SubElement(violation_elem, "Id").text = violation_data["violation_id"]
            ET.SubElement(violation_elem, "RuleId").text = violation_data["rule_id"]
            ET.SubElement(violation_elem, "Severity").text = violation_data["severity"]
            ET.SubElement(violation_elem, "Status").text = violation_data["status"]
            ET.SubElement(violation_elem, "Description").text = violation_data["rule_description"]

        return ET.tostring(root, encoding="unicode")


class RegulatoryExporter:
    """
    Exports audit data for regulatory submissions and filings.

    Provides format-specific exports required by various regulatory
    bodies and compliance frameworks.
    """

    def __init__(self, storage: AuditStorage, config: AuditConfig) -> None:
        """
        Initialize regulatory exporter.

        Args:
            storage: Audit storage backend
            config: Audit configuration
        """
        self.storage = storage
        self.config = config

    def export_for_regulation(
        self,
        regulation: ComplianceRegulation,
        start_date: datetime,
        end_date: datetime,
        output_format: str = "json",
    ) -> str:
        """
        Export audit data in format required by specific regulation.

        Args:
            regulation: Target regulation
            start_date: Export start date
            end_date: Export end date
            output_format: Output format

        Returns:
            Formatted export data
        """
        # Query relevant events
        events = self.storage.query(start_time=start_date, end_time=end_date)

        # Filter events relevant to regulation
        relevant_events = []
        for event in events:
            if self._is_relevant_to_regulation(event, regulation):
                relevant_events.append(event)

        # Format according to regulation requirements
        if regulation == ComplianceRegulation.SEC:
            return self._export_sec_format(relevant_events, output_format)
        elif regulation == ComplianceRegulation.CFTC:
            return self._export_cftc_format(relevant_events, output_format)
        elif regulation == ComplianceRegulation.MIFID_II:
            return self._export_mifid_format(relevant_events, output_format)
        else:
            # Generic export
            return self._export_generic_format(relevant_events, output_format)

    def _is_relevant_to_regulation(
        self, event: dict[str, Any], regulation: ComplianceRegulation
    ) -> bool:
        """Check if event is relevant to specific regulation."""
        event_regulations = event.get("compliance_regulations", [])
        return regulation.value in event_regulations

    def _export_sec_format(self, events: list[dict[str, Any]], output_format: str) -> str:
        """Export in SEC-required format."""
        # SEC requires specific fields and formats for trading records
        export_data: dict[str, Any] = {
            "report_type": "SEC_AUDIT_TRAIL",
            "submission_date": datetime.now(UTC).isoformat(),
            "firm_id": self.config.system_name,
            "records": [],
        }

        for event in events:
            if event.get("resource_type") in ["order", "position"]:
                record = {
                    "record_id": event.get("event_id"),
                    "timestamp": event.get("timestamp"),
                    "event_type": event.get("event_type"),
                    "security_identifier": event.get("resource_details", {}).get("symbol"),
                    "transaction_type": event.get("action"),
                    "quantity": event.get("resource_details", {}).get("quantity"),
                    "price": event.get("resource_details", {}).get("price"),
                    "customer_id": event.get("user_id"),
                }
                export_data["records"].append(record)

        if output_format == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            return str(export_data)

    def _export_cftc_format(self, events: list[dict[str, Any]], output_format: str) -> str:
        """Export in CFTC-required format."""
        # CFTC has specific requirements for swap and derivatives reporting
        export_data: dict[str, Any] = {
            "report_type": "CFTC_SWAP_DATA",
            "reporting_entity": self.config.system_name,
            "report_date": datetime.now(UTC).date().isoformat(),
            "transactions": [],
        }

        for event in events:
            # Only include derivative/swap related events
            if "derivative" in event.get("resource_details", {}).get("instrument_type", "").lower():
                transaction = {
                    "transaction_id": event.get("event_id"),
                    "execution_timestamp": event.get("timestamp"),
                    "product_type": event.get("resource_details", {}).get("instrument_type"),
                    "notional_amount": event.get("resource_details", {}).get("notional_amount"),
                    "counterparty": event.get("resource_details", {}).get("counterparty"),
                }
                export_data["transactions"].append(transaction)

        return json.dumps(export_data, indent=2, default=str)

    def _export_mifid_format(self, events: list[dict[str, Any]], output_format: str) -> str:
        """Export in MiFID II-required format."""
        # MiFID II requires transaction reporting with specific fields
        export_data: dict[str, Any] = {
            "report_type": "MIFID_TRANSACTION_REPORT",
            "reporting_firm": self.config.system_name,
            "report_period": datetime.now(UTC).date().isoformat(),
            "transactions": [],
        }

        for event in events:
            if event.get("resource_type") == "order" and event.get("action") in ["create", "fill"]:
                transaction = {
                    "transaction_reference": event.get("event_id"),
                    "trading_date_time": event.get("timestamp"),
                    "instrument_identification": event.get("resource_details", {}).get("symbol"),
                    "buy_sell_indicator": event.get("resource_details", {}).get("side"),
                    "quantity": event.get("resource_details", {}).get("quantity"),
                    "unit_price": event.get("resource_details", {}).get("price"),
                    "trading_venue": "INTERNAL",
                    "investment_decision_within_firm": event.get("user_id"),
                }
                export_data["transactions"].append(transaction)

        return json.dumps(export_data, indent=2, default=str)

    def _export_generic_format(self, events: list[dict[str, Any]], output_format: str) -> str:
        """Export in generic regulatory format."""
        export_data = {
            "export_type": "GENERIC_REGULATORY_EXPORT",
            "export_timestamp": datetime.now(UTC).isoformat(),
            "system_identifier": self.config.system_name,
            "events": events,
        }

        return json.dumps(export_data, indent=2, default=str)
