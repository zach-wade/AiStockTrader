"""
Unit tests for audit event classes.

Tests cover all event types, validation, serialization, and
compliance requirements.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.infrastructure.audit.events import (
    AuditEvent,
    AuthenticationEvent,
    ComplianceEvent,
    ComplianceRegulation,
    ConfigurationEvent,
    EventSeverity,
    OrderEvent,
    PortfolioEvent,
    PositionEvent,
    RiskEvent,
)
from src.infrastructure.audit.exceptions import AuditValidationError


class ConcreteAuditEvent(AuditEvent):
    """Concrete implementation for testing base AuditEvent."""

    def get_resource_details(self):
        return {"test": "details"}

    def _validate_resource_data(self):
        pass


class TestAuditEvent:
    """Test suite for base AuditEvent class."""

    def test_init_required_fields(self):
        """Test initialization with required fields."""
        event = ConcreteAuditEvent(
            event_type="test_event",
            resource_type="test_resource",
            resource_id="test_123",
            action="test_action",
        )

        assert event.event_type == "test_event"
        assert event.resource_type == "test_resource"
        assert event.resource_id == "test_123"
        assert event.action == "test_action"
        assert event.severity == EventSeverity.MEDIUM
        assert event.is_critical == False
        assert event.compliance_regulations == []
        assert event.tags == []
        assert event.metadata == {}
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo is not None

    def test_init_all_fields(self):
        """Test initialization with all fields."""
        timestamp = datetime.now(UTC)

        event = ConcreteAuditEvent(
            event_type="test_event",
            resource_type="test_resource",
            resource_id="test_123",
            action="test_action",
            user_id="user_456",
            session_id="session_789",
            timestamp=timestamp,
            severity=EventSeverity.HIGH,
            is_critical=True,
            compliance_regulations=[ComplianceRegulation.SOX],
            tags=["important", "financial"],
            metadata={"key": "value"},
        )

        assert event.user_id == "user_456"
        assert event.session_id == "session_789"
        assert event.timestamp == timestamp
        assert event.severity == EventSeverity.HIGH
        assert event.is_critical == True
        assert event.compliance_regulations == [ComplianceRegulation.SOX]
        assert event.tags == ["important", "financial"]
        assert event.metadata == {"key": "value"}

    def test_timestamp_timezone_awareness(self):
        """Test timestamp timezone handling."""
        # Test with naive datetime
        naive_timestamp = datetime(2023, 1, 1, 12, 0, 0)
        event = ConcreteAuditEvent(
            event_type="test",
            resource_type="test",
            resource_id="test",
            action="test",
            timestamp=naive_timestamp,
        )

        # Should be converted to UTC
        assert event.timestamp.tzinfo == UTC

    def test_validate_success(self):
        """Test successful validation."""
        event = ConcreteAuditEvent(
            event_type="test_event",
            resource_type="test_resource",
            resource_id="test_123",
            action="test_action",
        )

        # Should not raise exception
        event.validate()

    def test_validate_missing_event_type(self):
        """Test validation with missing event type."""
        event = ConcreteAuditEvent(
            event_type="",
            resource_type="test_resource",
            resource_id="test_123",
            action="test_action",
        )

        with pytest.raises(AuditValidationError) as exc_info:
            event.validate()

        assert "Event type is required" in str(exc_info)

    def test_validate_missing_resource_type(self):
        """Test validation with missing resource type."""
        event = ConcreteAuditEvent(
            event_type="test_event", resource_type="", resource_id="test_123", action="test_action"
        )

        with pytest.raises(AuditValidationError):
            event.validate()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
        event = ConcreteAuditEvent(
            event_type="test_event",
            resource_type="test_resource",
            resource_id="test_123",
            action="test_action",
            user_id="user_456",
            timestamp=timestamp,
            severity=EventSeverity.HIGH,
            is_critical=True,
            compliance_regulations=[ComplianceRegulation.SOX],
            tags=["tag1"],
            metadata={"key": "value"},
        )

        result = event.to_dict()

        assert result["event_type"] == "test_event"
        assert result["resource_type"] == "test_resource"
        assert result["resource_id"] == "test_123"
        assert result["action"] == "test_action"
        assert result["user_id"] == "user_456"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["severity"] == "high"
        assert result["is_critical"] == True
        assert result["compliance_regulations"] == ["sox"]
        assert result["tags"] == ["tag1"]
        assert result["metadata"] == {"key": "value"}
        assert result["resource_details"] == {"test": "details"}

    def test_add_tag(self):
        """Test adding tags."""
        event = ConcreteAuditEvent(
            event_type="test", resource_type="test", resource_id="test", action="test"
        )

        event.add_tag("important")
        event.add_tag("financial")
        event.add_tag("important")  # Duplicate should not be added

        assert event.tags == ["important", "financial"]

    def test_add_compliance_regulation(self):
        """Test adding compliance regulations."""
        event = ConcreteAuditEvent(
            event_type="test", resource_type="test", resource_id="test", action="test"
        )

        event.add_compliance_regulation(ComplianceRegulation.SOX)
        event.add_compliance_regulation(ComplianceRegulation.SEC)
        event.add_compliance_regulation(ComplianceRegulation.SOX)  # Duplicate

        assert len(event.compliance_regulations) == 2
        assert ComplianceRegulation.SOX in event.compliance_regulations
        assert ComplianceRegulation.SEC in event.compliance_regulations

    def test_set_metadata(self):
        """Test setting metadata."""
        event = ConcreteAuditEvent(
            event_type="test", resource_type="test", resource_id="test", action="test"
        )

        event.set_metadata("key1", "value1")
        event.set_metadata("key2", {"nested": "value"})

        assert event.metadata["key1"] == "value1"
        assert event.metadata["key2"] == {"nested": "value"}


class TestOrderEvent:
    """Test suite for OrderEvent class."""

    def test_init_basic(self):
        """Test basic initialization."""
        event = OrderEvent(event_type="order_create", order_id="order_123", symbol="AAPL")

        assert event.resource_type == "order"
        assert event.resource_id == "order_123"
        assert event.order_id == "order_123"
        assert event.symbol == "AAPL"
        assert event.severity == EventSeverity.HIGH
        assert ComplianceRegulation.SEC in event.compliance_regulations
        assert ComplianceRegulation.FINRA in event.compliance_regulations

    def test_init_with_financial_details(self):
        """Test initialization with financial details."""
        event = OrderEvent(
            event_type="order_create",
            order_id="order_123",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.25"),
            order_type="limit",
            time_in_force="DAY",
            status="pending",
            broker_order_id="broker_456",
        )

        assert event.side == "buy"
        assert event.quantity == Decimal("100")
        assert event.price == Decimal("150.25")
        assert event.order_type == "limit"
        assert event.time_in_force == "DAY"
        assert event.status == "pending"
        assert event.broker_order_id == "broker_456"

    def test_large_order_critical_flag(self):
        """Test critical flag for large orders."""
        event = OrderEvent(
            event_type="order_create",
            order_id="order_123",
            symbol="AAPL",
            quantity=Decimal("15000"),  # Large order
        )

        assert event.is_critical == True

    def test_rejected_order_critical_flag(self):
        """Test critical flag for rejected orders."""
        event = OrderEvent(
            event_type="order_rejected", order_id="order_123", symbol="AAPL", action="rejected"
        )

        assert event.is_critical == True

    def test_get_resource_details(self):
        """Test resource details extraction."""
        event = OrderEvent(
            event_type="order_create",
            order_id="order_123",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("150.25"),
            order_type="limit",
            fill_quantity=Decimal("50"),
            commission=Decimal("1.99"),
        )

        details = event.get_resource_details()

        assert details["order_id"] == "order_123"
        assert details["symbol"] == "AAPL"
        assert details["side"] == "buy"
        assert details["order_type"] == "limit"
        assert details["quantity"] == "100"
        assert details["price"] == "150.25"
        assert details["fill_quantity"] == "50"
        assert details["commission"] == "1.99"

    def test_validate_success(self):
        """Test successful validation."""
        event = OrderEvent(
            event_type="order_create",
            order_id="order_123",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
        )

        # Should not raise exception
        event._validate_resource_data()

    def test_validate_missing_order_id(self):
        """Test validation with missing order ID."""
        event = OrderEvent(event_type="order_create", order_id="", symbol="AAPL")

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()

    def test_validate_missing_symbol(self):
        """Test validation with missing symbol."""
        event = OrderEvent(event_type="order_create", order_id="order_123", symbol="")

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()

    def test_validate_invalid_side(self):
        """Test validation with invalid side."""
        event = OrderEvent(
            event_type="order_create", order_id="order_123", symbol="AAPL", side="invalid"
        )

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()

    def test_validate_negative_quantity(self):
        """Test validation with negative quantity."""
        event = OrderEvent(
            event_type="order_create", order_id="order_123", symbol="AAPL", quantity=Decimal("-100")
        )

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()


class TestPositionEvent:
    """Test suite for PositionEvent class."""

    def test_init_basic(self):
        """Test basic initialization."""
        event = PositionEvent(event_type="position_open", position_id="pos_123", symbol="GOOGL")

        assert event.resource_type == "position"
        assert event.resource_id == "pos_123"
        assert event.position_id == "pos_123"
        assert event.symbol == "GOOGL"
        assert event.severity == EventSeverity.HIGH
        assert ComplianceRegulation.SEC in event.compliance_regulations
        assert ComplianceRegulation.CFTC in event.compliance_regulations

    def test_init_with_financial_details(self):
        """Test initialization with financial details."""
        event = PositionEvent(
            event_type="position_update",
            position_id="pos_123",
            symbol="GOOGL",
            quantity=Decimal("50"),
            average_cost=Decimal("2500.00"),
            current_price=Decimal("2550.00"),
            unrealized_pnl=Decimal("2500.00"),
            realized_pnl=Decimal("500.00"),
            portfolio_id="port_456",
        )

        assert event.quantity == Decimal("50")
        assert event.average_cost == Decimal("2500.00")
        assert event.current_price == Decimal("2550.00")
        assert event.unrealized_pnl == Decimal("2500.00")
        assert event.realized_pnl == Decimal("500.00")
        assert event.portfolio_id == "port_456"

    def test_large_position_critical_flag(self):
        """Test critical flag for large positions."""
        event = PositionEvent(
            event_type="position_open",
            position_id="pos_123",
            symbol="GOOGL",
            quantity=Decimal("15000"),  # Large position
        )

        assert event.is_critical == True

    def test_large_pnl_critical_flag(self):
        """Test critical flag for large P&L."""
        event = PositionEvent(
            event_type="position_update",
            position_id="pos_123",
            symbol="GOOGL",
            unrealized_pnl=Decimal("150000"),  # Large P&L
        )

        assert event.is_critical == True

    def test_get_resource_details(self):
        """Test resource details extraction."""
        event = PositionEvent(
            event_type="position_update",
            position_id="pos_123",
            symbol="GOOGL",
            quantity=Decimal("50"),
            average_cost=Decimal("2500.00"),
            portfolio_id="port_456",
        )

        details = event.get_resource_details()

        assert details["position_id"] == "pos_123"
        assert details["symbol"] == "GOOGL"
        assert details["portfolio_id"] == "port_456"
        assert details["quantity"] == "50"
        assert details["average_cost"] == "2500.00"

    def test_validate_success(self):
        """Test successful validation."""
        event = PositionEvent(event_type="position_open", position_id="pos_123", symbol="GOOGL")

        # Should not raise exception
        event._validate_resource_data()

    def test_validate_missing_position_id(self):
        """Test validation with missing position ID."""
        event = PositionEvent(event_type="position_open", position_id="", symbol="GOOGL")

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()

    def test_validate_missing_symbol(self):
        """Test validation with missing symbol."""
        event = PositionEvent(event_type="position_open", position_id="pos_123", symbol="")

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()


class TestPortfolioEvent:
    """Test suite for PortfolioEvent class."""

    def test_init_basic(self):
        """Test basic initialization."""
        event = PortfolioEvent(event_type="portfolio_create", portfolio_id="port_123")

        assert event.resource_type == "portfolio"
        assert event.resource_id == "port_123"
        assert event.portfolio_id == "port_123"
        assert event.severity == EventSeverity.HIGH
        assert ComplianceRegulation.SEC in event.compliance_regulations
        assert ComplianceRegulation.SOX in event.compliance_regulations

    def test_init_with_financial_details(self):
        """Test initialization with financial details."""
        event = PortfolioEvent(
            event_type="portfolio_update",
            portfolio_id="port_123",
            portfolio_name="Test Portfolio",
            cash_balance=Decimal("50000.00"),
            market_value=Decimal("150000.00"),
            total_value=Decimal("200000.00"),
            day_pnl=Decimal("5000.00"),
            risk_metrics={"var": "10000", "beta": "1.2"},
        )

        assert event.portfolio_name == "Test Portfolio"
        assert event.cash_balance == Decimal("50000.00")
        assert event.market_value == Decimal("150000.00")
        assert event.total_value == Decimal("200000.00")
        assert event.day_pnl == Decimal("5000.00")
        assert event.risk_metrics == {"var": "10000", "beta": "1.2"}

    def test_large_portfolio_critical_flag(self):
        """Test critical flag for large portfolios."""
        event = PortfolioEvent(
            event_type="portfolio_update",
            portfolio_id="port_123",
            total_value=Decimal("2000000"),  # Large portfolio
        )

        assert event.is_critical == True

    def test_large_pnl_critical_flag(self):
        """Test critical flag for large daily P&L."""
        event = PortfolioEvent(
            event_type="portfolio_update",
            portfolio_id="port_123",
            day_pnl=Decimal("75000"),  # Large daily P&L
        )

        assert event.is_critical == True

    def test_validate_missing_portfolio_id(self):
        """Test validation with missing portfolio ID."""
        event = PortfolioEvent(event_type="portfolio_create", portfolio_id="")

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()


class TestRiskEvent:
    """Test suite for RiskEvent class."""

    def test_init_basic(self):
        """Test basic initialization."""
        event = RiskEvent(
            event_type="risk_breach",
            risk_type="position_limit",
            threshold_value=Decimal("100000"),
            current_value=Decimal("150000"),
        )

        assert event.resource_type == "risk"
        assert event.risk_type == "position_limit"
        assert event.threshold_value == Decimal("100000")
        assert event.current_value == Decimal("150000")
        assert event.severity == EventSeverity.HIGH
        assert ComplianceRegulation.SEC in event.compliance_regulations
        assert ComplianceRegulation.BASEL_III in event.compliance_regulations

    def test_critical_risk_level(self):
        """Test critical flag for critical risk level."""
        event = RiskEvent(event_type="risk_breach", risk_type="var_limit", risk_level="critical")

        assert event.is_critical == True

    def test_high_breach_percentage(self):
        """Test critical flag for high breach percentage."""
        event = RiskEvent(
            event_type="risk_breach", risk_type="position_limit", breach_percentage=Decimal("75")
        )

        assert event.is_critical == True

    def test_validate_missing_risk_type(self):
        """Test validation with missing risk type."""
        event = RiskEvent(event_type="risk_breach", risk_type="")

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()


class TestAuthenticationEvent:
    """Test suite for AuthenticationEvent class."""

    def test_init_successful_login(self):
        """Test initialization for successful login."""
        event = AuthenticationEvent(
            event_type="authentication_attempt",
            resource_id="user_123",
            resource_type="user",
            action="login",
            user_id="user_123",
            auth_method="password",
            ip_address="192.168.1.1",
            login_success=True,
        )

        assert event.resource_type == "authentication"
        assert event.auth_method == "password"
        assert event.ip_address == "192.168.1.1"
        assert event.login_success == True
        assert event.severity == EventSeverity.MEDIUM
        assert event.is_critical == False

    def test_init_failed_login(self):
        """Test initialization for failed login."""
        event = AuthenticationEvent(
            event_type="authentication_attempt",
            resource_id="user_123",
            resource_type="user",
            action="login",
            user_id="user_123",
            auth_method="password",
            login_success=False,
            failure_reason="invalid_password",
        )

        assert event.login_success == False
        assert event.failure_reason == "invalid_password"
        assert event.severity == EventSeverity.CRITICAL
        assert event.is_critical == True
        assert ComplianceRegulation.SOX in event.compliance_regulations
        assert ComplianceRegulation.GDPR in event.compliance_regulations

    def test_validate_missing_login_success(self):
        """Test validation with missing login success status."""
        event = AuthenticationEvent(
            event_type="authentication_attempt",
            resource_id="user_123",
            resource_type="user",
            action="login",
            login_success=None,
        )

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()


class TestConfigurationEvent:
    """Test suite for ConfigurationEvent class."""

    def test_init_basic(self):
        """Test basic initialization."""
        event = ConfigurationEvent(
            event_type="config_change",
            resource_id="config_risk_limit",
            action="update",
            config_key="risk_limit",
            old_value="100000",
            new_value="150000",
        )

        assert event.resource_type == "configuration"
        assert event.config_key == "risk_limit"
        assert event.old_value == "100000"
        assert event.new_value == "150000"
        assert event.severity == EventSeverity.HIGH

    def test_critical_config_categories(self):
        """Test critical flag for critical config categories."""
        event = ConfigurationEvent(
            event_type="config_change",
            resource_id="config_risk_setting",
            action="update",
            config_key="risk_setting",
            config_category="risk",
        )

        assert event.is_critical == True

    def test_validate_missing_config_key(self):
        """Test validation with missing config key."""
        event = ConfigurationEvent(
            event_type="config_change", resource_id="config_test", action="update", config_key=""
        )

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()


class TestComplianceEvent:
    """Test suite for ComplianceEvent class."""

    def test_init_compliance_pass(self):
        """Test initialization for passing compliance check."""
        event = ComplianceEvent(
            event_type="compliance_check",
            resource_id="compliance_sox_test",
            action="check",
            regulation=ComplianceRegulation.SOX,
            compliance_rule="SOX-404",
            check_result=True,
        )

        assert event.resource_type == "compliance"
        assert event.regulation == ComplianceRegulation.SOX
        assert event.compliance_rule == "SOX-404"
        assert event.check_result == True
        assert event.severity == EventSeverity.MEDIUM  # Default for passing
        assert event.is_critical == False
        assert ComplianceRegulation.SOX in event.compliance_regulations

    def test_init_compliance_violation(self):
        """Test initialization for compliance violation."""
        deadline = datetime(2023, 12, 31, tzinfo=UTC)

        event = ComplianceEvent(
            event_type="compliance_violation",
            resource_id="compliance_gdpr_violation",
            action="violate",
            regulation=ComplianceRegulation.GDPR,
            compliance_rule="GDPR-Art5",
            check_result=False,
            violation_details="Data retained beyond lawful period",
            remediation_actions=["Delete expired data"],
            deadline=deadline,
        )

        assert event.check_result == False
        assert event.violation_details == "Data retained beyond lawful period"
        assert event.remediation_actions == ["Delete expired data"]
        assert event.deadline == deadline
        assert event.severity == EventSeverity.CRITICAL
        assert event.is_critical == True

    def test_validate_missing_compliance_rule(self):
        """Test validation with missing compliance rule."""
        event = ComplianceEvent(
            event_type="compliance_check",
            resource_id="compliance_test",
            action="check",
            regulation=ComplianceRegulation.SOX,
            compliance_rule="",
        )

        with pytest.raises(AuditValidationError):
            event._validate_resource_data()
