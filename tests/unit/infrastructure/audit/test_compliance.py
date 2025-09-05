"""
Unit tests for audit compliance functionality.

Tests cover compliance checking, violation detection, reporting,
and regulatory export capabilities.
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from src.infrastructure.audit.compliance import (
    ComplianceReporter,
    ComplianceViolation,
    GDPRComplianceChecker,
    RegulatoryExporter,
    SOXComplianceChecker,
    ViolationSeverity,
)
from src.infrastructure.audit.config import (
    AuditConfig,
    SecurityConfig,
    StorageBackend,
    StorageConfig,
)
from src.infrastructure.audit.events import ComplianceRegulation
from src.infrastructure.audit.exceptions import AuditComplianceError
from src.infrastructure.audit.logger import AuditLogger
from src.infrastructure.audit.storage import AuditStorage


@pytest.fixture
def sample_violation():
    """Create sample compliance violation."""
    return ComplianceViolation(
        regulation=ComplianceRegulation.SOX,
        rule_id="SOX-302",
        rule_description="Unauthorized financial transaction",
        severity=ViolationSeverity.HIGH,
        event_ids=["event_123", "event_456"],
        affected_users=["user_789"],
        affected_resources=["order_123"],
        violation_details={"amount": 150000},
        remediation_actions=["Review authorization", "Audit permissions"],
    )


@pytest.fixture
def sample_audit_events():
    """Create sample audit events for testing."""
    base_time = datetime.now(UTC)

    return [
        {
            "event_id": "order_event_1",
            "event_type": "order_operation",
            "resource_type": "order",
            "action": "create",
            "user_id": "user_123",
            "timestamp": base_time.isoformat(),
            "resource_details": {
                "quantity": "150000",  # Large order
                "approved_by": None,  # No approval
            },
        },
        {
            "event_id": "auth_event_1",
            "event_type": "authentication_attempt",
            "resource_type": "authentication",
            "action": "login",
            "user_id": "user_456",
            "timestamp": base_time.isoformat(),
            "resource_details": {"login_success": False, "failure_reason": "invalid_password"},
        },
        {
            "event_id": "auth_event_2",
            "event_type": "authentication_attempt",
            "resource_type": "authentication",
            "action": "login",
            "user_id": "user_456",
            "timestamp": (base_time + timedelta(minutes=1)).isoformat(),
            "resource_details": {"login_success": False, "failure_reason": "invalid_password"},
        },
        {
            "event_id": "config_event_1",
            "event_type": "configuration_change",
            "resource_type": "configuration",
            "action": "update",
            "user_id": "user_789",
            "timestamp": base_time.isoformat(),
            "resource_details": {
                "config_key": "risk_limit",
                "old_value": "100000",
                "new_value": "200000",
                "approved_by": None,  # No approval
            },
        },
        {
            "event_id": "personal_data_event",
            "event_type": "data_access",
            "resource_type": "user_data",
            "action": "access",
            "user_id": "user_personal",
            "timestamp": (base_time - timedelta(days=400)).isoformat(),  # Old event
            "context": {"user_id": "user_personal", "ip_address": "192.168.1.1"},
        },
    ]


@pytest.fixture
def mock_storage():
    """Create mock storage for testing."""
    storage = Mock(spec=AuditStorage)
    return storage


@pytest.fixture
def mock_logger():
    """Create mock audit logger."""
    return Mock(spec=AuditLogger)


class TestComplianceViolation:
    """Test suite for ComplianceViolation class."""

    def test_init(self):
        """Test violation initialization."""
        violation = ComplianceViolation(
            regulation=ComplianceRegulation.GDPR,
            rule_id="GDPR-Art5",
            rule_description="Data retention violation",
            severity=ViolationSeverity.CRITICAL,
        )

        assert violation.regulation == ComplianceRegulation.GDPR
        assert violation.rule_id == "GDPR-Art5"
        assert violation.rule_description == "Data retention violation"
        assert violation.severity == ViolationSeverity.CRITICAL
        assert len(violation.violation_id) > 0
        assert isinstance(violation.detected_at, datetime)
        assert violation.status == "open"
        assert violation.remediation_required == True

    def test_to_dict(self, sample_violation):
        """Test conversion to dictionary."""
        violation_dict = sample_violation.to_dict()

        assert violation_dict["regulation"] == "sox"
        assert violation_dict["rule_id"] == "SOX-302"
        assert violation_dict["severity"] == "high"
        assert violation_dict["event_ids"] == ["event_123", "event_456"]
        assert violation_dict["affected_users"] == ["user_789"]
        assert violation_dict["violation_details"] == {"amount": 150000}
        assert "detected_at" in violation_dict
        assert "violation_id" in violation_dict


class TestSOXComplianceChecker:
    """Test suite for SOX compliance checker."""

    def test_init(self):
        """Test SOX checker initialization."""
        checker = SOXComplianceChecker()
        assert checker.regulation == ComplianceRegulation.SOX

    def test_check_unauthorized_financial_transaction(self, sample_audit_events):
        """Test detection of unauthorized financial transactions."""
        checker = SOXComplianceChecker()

        start_date = datetime.now(UTC) - timedelta(days=1)
        end_date = datetime.now(UTC) + timedelta(days=1)

        violations = checker.check_compliance(iter(sample_audit_events), start_date, end_date)

        # Should find unauthorized financial transaction
        financial_violations = [v for v in violations if v.rule_id == "SOX-302"]
        assert len(financial_violations) == 1

        violation = financial_violations[0]
        assert violation.severity == ViolationSeverity.HIGH
        assert "user_123" in violation.affected_users
        assert "Unauthorized financial transaction" in violation.rule_description

    def test_check_config_change_without_approval(self, sample_audit_events):
        """Test detection of financial config changes without approval."""
        checker = SOXComplianceChecker()

        start_date = datetime.now(UTC) - timedelta(days=1)
        end_date = datetime.now(UTC) + timedelta(days=1)

        violations = checker.check_compliance(iter(sample_audit_events), start_date, end_date)

        # Should find unapproved config change
        config_violations = [v for v in violations if v.rule_id == "SOX-404"]
        assert len(config_violations) == 1

        violation = config_violations[0]
        assert violation.severity == ViolationSeverity.CRITICAL
        assert "user_789" in violation.affected_users

    def test_check_excessive_failed_logins(self, mock_logger):
        """Test detection of excessive failed login attempts."""
        checker = SOXComplianceChecker()

        # Create many failed login events for same user
        failed_events = []
        base_time = datetime.now(UTC)

        for i in range(10):  # More than 5 failures
            event = {
                "event_id": f"failed_login_{i}",
                "event_type": "authentication_attempt",
                "resource_type": "authentication",
                "action": "login",
                "user_id": "suspicious_user",
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
                "resource_details": {"login_success": False, "failure_reason": "invalid_password"},
            }
            failed_events.append(event)

        start_date = base_time - timedelta(hours=1)
        end_date = base_time + timedelta(hours=1)

        violations = checker.check_compliance(iter(failed_events), start_date, end_date)

        # Should find excessive failed attempts violation
        access_violations = [v for v in violations if "Access" in v.rule_id]
        assert len(access_violations) == 1

        violation = access_violations[0]
        assert violation.severity == ViolationSeverity.HIGH
        assert "suspicious_user" in violation.affected_users
        assert violation.violation_details["failed_attempts"] == 10

    def test_check_audit_trail_gaps(self, mock_logger):
        """Test detection of audit trail gaps."""
        checker = SOXComplianceChecker()

        # Create events with significant time gap
        base_time = datetime.now(UTC)
        events_with_gap = [
            {
                "event_id": "event_before_gap",
                "event_type": "order_operation",
                "timestamp": base_time.isoformat(),
            },
            {
                "event_id": "event_after_gap",
                "event_type": "order_operation",
                "timestamp": (base_time + timedelta(hours=2)).isoformat(),  # 2 hour gap
            },
        ]

        start_date = base_time - timedelta(hours=1)
        end_date = base_time + timedelta(hours=3)

        violations = checker.check_compliance(iter(events_with_gap), start_date, end_date)

        # Should find audit trail gap
        audit_violations = [v for v in violations if "Audit" in v.rule_id]
        assert len(audit_violations) == 1

        violation = audit_violations[0]
        assert violation.severity == ViolationSeverity.MEDIUM
        assert "gap_duration_hours" in violation.violation_details
        assert violation.violation_details["gap_duration_hours"] == 2.0


class TestGDPRComplianceChecker:
    """Test suite for GDPR compliance checker."""

    def test_init(self):
        """Test GDPR checker initialization."""
        checker = GDPRComplianceChecker()
        assert checker.regulation == ComplianceRegulation.GDPR

    def test_check_data_retention_violations(self, sample_audit_events):
        """Test detection of data retention violations."""
        checker = GDPRComplianceChecker()

        start_date = datetime.now(UTC) - timedelta(days=500)
        end_date = datetime.now(UTC)

        violations = checker.check_compliance(iter(sample_audit_events), start_date, end_date)

        # Should find data retention violation
        retention_violations = [v for v in violations if v.rule_id == "GDPR-Art5"]
        assert len(retention_violations) == 1

        violation = retention_violations[0]
        assert violation.severity == ViolationSeverity.HIGH
        assert "expired_events_count" in violation.violation_details
        assert violation.violation_details["expired_events_count"] == 1

    def test_personal_data_detection(self):
        """Test detection of personal data in events."""
        checker = GDPRComplianceChecker()

        # Event with personal data
        personal_event = {
            "event_id": "personal_event",
            "user_id": "user_123",
            "context": {"ip_address": "192.168.1.1", "email": "user@example.com"},
        }

        # Event without personal data
        system_event = {"event_id": "system_event", "event_type": "system_startup"}

        assert checker._has_personal_data(personal_event) == True
        assert checker._has_personal_data(system_event) == False


class TestComplianceReporter:
    """Test suite for ComplianceReporter class."""

    def test_init(self, mock_logger, mock_storage):
        """Test compliance reporter initialization."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        reporter = ComplianceReporter(mock_logger, mock_storage, config)

        assert reporter.audit_logger == mock_logger
        assert reporter.storage == mock_storage
        assert reporter.config == config
        assert ComplianceRegulation.SOX in reporter.checkers
        assert ComplianceRegulation.GDPR in reporter.checkers

    def test_generate_compliance_report(self, mock_logger, mock_storage, sample_audit_events):
        """Test compliance report generation."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        reporter = ComplianceReporter(mock_logger, mock_storage, config)

        # Mock storage query to return sample events
        mock_storage.query.return_value = iter(sample_audit_events)

        start_date = datetime.now(UTC) - timedelta(days=1)
        end_date = datetime.now(UTC)

        report = reporter.generate_compliance_report(
            regulation=ComplianceRegulation.SOX, start_date=start_date, end_date=end_date
        )

        assert report["regulation"] == "sox"
        assert "report_id" in report
        assert "generated_at" in report
        assert "summary" in report
        assert "violations" in report
        assert "event_statistics" in report
        assert "recommendations" in report

        # Should have found violations in sample events
        assert report["summary"]["total_events_reviewed"] == len(sample_audit_events)
        assert report["summary"]["violations_found"] > 0

    def test_generate_violation_summary(self, mock_logger, mock_storage, sample_violation):
        """Test violation summary generation."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        reporter = ComplianceReporter(mock_logger, mock_storage, config)

        violations = [
            sample_violation,
            ComplianceViolation(
                regulation=ComplianceRegulation.GDPR,
                rule_id="GDPR-Art5",
                severity=ViolationSeverity.CRITICAL,
                status="investigating",
            ),
            ComplianceViolation(
                regulation=ComplianceRegulation.SOX,
                rule_id="SOX-404",
                severity=ViolationSeverity.MEDIUM,
                remediation_required=True,
                remediation_deadline=datetime.now(UTC) - timedelta(days=1),  # Overdue
            ),
        ]

        summary = reporter.generate_violation_summary(violations)

        assert summary["total_violations"] == 3
        assert summary["by_regulation"]["sox"] == 2
        assert summary["by_regulation"]["gdpr"] == 1
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_severity"]["medium"] == 1
        assert summary["by_status"]["open"] == 2
        assert summary["by_status"]["investigating"] == 1
        assert summary["remediation_required"] == 2
        assert summary["overdue_remediations"] == 1

    def test_export_report_json(self, mock_logger, mock_storage):
        """Test JSON report export."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        reporter = ComplianceReporter(mock_logger, mock_storage, config)

        report_data = {
            "report_id": "test_report",
            "regulation": "sox",
            "violations": [
                {"violation_id": "v1", "severity": "high"},
                {"violation_id": "v2", "severity": "medium"},
            ],
        }

        exported = reporter.export_report(report_data, "json")

        # Should be valid JSON
        parsed = json.loads(exported)
        assert parsed["report_id"] == "test_report"
        assert parsed["regulation"] == "sox"
        assert len(parsed["violations"]) == 2

    def test_export_report_csv(self, mock_logger, mock_storage):
        """Test CSV report export."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        reporter = ComplianceReporter(mock_logger, mock_storage, config)

        report_data = {
            "report_id": "test_report",
            "regulation": "sox",
            "period": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            "violations": [
                {
                    "violation_id": "v1",
                    "rule_id": "SOX-302",
                    "severity": "high",
                    "status": "open",
                    "rule_description": "Test violation",
                    "affected_users": ["user1", "user2"],
                }
            ],
        }

        exported = reporter.export_report(report_data, "csv")

        # Should contain CSV headers and data
        assert "Compliance Report Summary" in exported
        assert "Violations" in exported
        assert "Violation ID" in exported
        assert "v1" in exported
        assert "SOX-302" in exported

    def test_export_report_xml(self, mock_logger, mock_storage):
        """Test XML report export."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        reporter = ComplianceReporter(mock_logger, mock_storage, config)

        report_data = {
            "report_id": "test_report",
            "regulation": "sox",
            "generated_at": "2023-01-01T00:00:00Z",
            "period": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            "violations": [
                {
                    "violation_id": "v1",
                    "rule_id": "SOX-302",
                    "severity": "high",
                    "status": "open",
                    "rule_description": "Test violation",
                }
            ],
        }

        exported = reporter.export_report(report_data, "xml")

        # Should contain XML structure
        assert "<ComplianceReport>" in exported
        assert "<ReportId>test_report</ReportId>" in exported
        assert "<Violations>" in exported
        assert "<Violation>" in exported

    def test_unsupported_export_format(self, mock_logger, mock_storage):
        """Test unsupported export format error."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        reporter = ComplianceReporter(mock_logger, mock_storage, config)

        report_data = {"report_id": "test"}

        with pytest.raises(AuditComplianceError) as exc_info:
            reporter.export_report(report_data, "unsupported_format")

        assert "Unsupported export format" in str(exc_info)

    def test_event_statistics_generation(self, mock_logger, mock_storage, sample_audit_events):
        """Test event statistics generation."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        reporter = ComplianceReporter(mock_logger, mock_storage, config)

        stats = reporter._generate_event_statistics(sample_audit_events)

        assert stats["events_by_type"]["order_operation"] == 1
        assert stats["events_by_type"]["authentication_attempt"] == 2
        assert stats["events_by_type"]["configuration_change"] == 1
        assert stats["events_by_user"]["user_123"] == 1
        assert stats["events_by_user"]["user_456"] == 2
        assert stats["authentication_events"] == 2
        assert stats["financial_operations"] == 1

    def test_recommendations_generation(self, mock_logger, mock_storage):
        """Test recommendations generation."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        reporter = ComplianceReporter(mock_logger, mock_storage, config)

        violations = [
            ComplianceViolation(rule_id="SOX-302"),
            ComplianceViolation(rule_id="SOX-404"),
            ComplianceViolation(rule_id="GDPR-Art5"),
            ComplianceViolation(rule_id="SOX-Access-01"),
            ComplianceViolation(rule_id="SOX-Audit-01"),
        ]

        recommendations = reporter._generate_recommendations(violations)

        assert len(recommendations) == 5
        assert any("financial transaction authorization" in rec for rec in recommendations)
        assert any("change management" in rec for rec in recommendations)
        assert any("data retention" in rec for rec in recommendations)
        assert any("access control" in rec for rec in recommendations)
        assert any("audit logging system" in rec for rec in recommendations)


class TestRegulatoryExporter:
    """Test suite for RegulatoryExporter class."""

    def test_init(self, mock_storage):
        """Test regulatory exporter initialization."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        exporter = RegulatoryExporter(mock_storage, config)

        assert exporter.storage == mock_storage
        assert exporter.config == config

    def test_export_for_sec(self, mock_storage, sample_audit_events):
        """Test SEC format export."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        exporter = RegulatoryExporter(mock_storage, config)

        # Filter events to include compliance regulations
        sec_events = []
        for event in sample_audit_events:
            event["compliance_regulations"] = ["sec"]
            if event["resource_type"] in ["order", "position"]:
                sec_events.append(event)

        mock_storage.query.return_value = iter(sec_events)

        start_date = datetime.now(UTC) - timedelta(days=1)
        end_date = datetime.now(UTC)

        export_data = exporter.export_for_regulation(
            regulation=ComplianceRegulation.SEC, start_date=start_date, end_date=end_date
        )

        parsed = json.loads(export_data)

        assert parsed["report_type"] == "SEC_AUDIT_TRAIL"
        assert parsed["firm_id"] == config.system_name
        assert "submission_date" in parsed
        assert "records" in parsed

    def test_export_for_mifid(self, mock_storage, sample_audit_events):
        """Test MiFID II format export."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        exporter = RegulatoryExporter(mock_storage, config)

        # Create MiFID relevant events
        mifid_events = [
            {
                "event_id": "mifid_order_1",
                "event_type": "order_operation",
                "resource_type": "order",
                "action": "create",
                "compliance_regulations": ["mifid_ii"],
                "resource_details": {
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": "100",
                    "price": "150.00",
                },
                "user_id": "trader_123",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]

        mock_storage.query.return_value = iter(mifid_events)

        start_date = datetime.now(UTC) - timedelta(days=1)
        end_date = datetime.now(UTC)

        export_data = exporter.export_for_regulation(
            regulation=ComplianceRegulation.MIFID_II, start_date=start_date, end_date=end_date
        )

        parsed = json.loads(export_data)

        assert parsed["report_type"] == "MIFID_TRANSACTION_REPORT"
        assert parsed["reporting_firm"] == config.system_name
        assert len(parsed["transactions"]) == 1

        transaction = parsed["transactions"][0]
        assert transaction["instrument_identification"] == "AAPL"
        assert transaction["buy_sell_indicator"] == "buy"
        assert transaction["quantity"] == "100"

    def test_relevance_filtering(self, mock_storage):
        """Test filtering of events relevant to specific regulation."""
        config = AuditConfig(
            security_config=SecurityConfig(
                encryption_enabled=False,
                digital_signatures_enabled=False,
                access_control_enabled=False,
                integrity_checks_enabled=False,
                tls_enabled=False,
                tamper_detection_enabled=False,
            ),
            storage_config=StorageConfig(
                primary_backend=StorageBackend.MEMORY, file_storage_path="/tmp/test_audit"
            ),
        )
        exporter = RegulatoryExporter(mock_storage, config)

        # Event relevant to SEC
        sec_event = {"compliance_regulations": ["sec", "finra"]}

        # Event relevant to GDPR
        gdpr_event = {"compliance_regulations": ["gdpr"]}

        # Event with no compliance regulations
        generic_event = {"compliance_regulations": []}

        assert exporter._is_relevant_to_regulation(sec_event, ComplianceRegulation.SEC) == True
        assert exporter._is_relevant_to_regulation(sec_event, ComplianceRegulation.GDPR) == False
        assert exporter._is_relevant_to_regulation(gdpr_event, ComplianceRegulation.GDPR) == True
        assert exporter._is_relevant_to_regulation(generic_event, ComplianceRegulation.SEC) == False
