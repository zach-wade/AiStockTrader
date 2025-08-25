"""
Audit event definitions for financial trading operations.

This module provides structured audit event classes for all types of
financial operations, ensuring comprehensive compliance tracking and
regulatory reporting capabilities.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from .exceptions import AuditValidationError


class EventSeverity(Enum):
    """Audit event severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceRegulation(Enum):
    """Supported compliance regulations."""

    SOX = "sox"
    GDPR = "gdpr"
    MIFID_II = "mifid_ii"
    CFTC = "cftc"
    SEC = "sec"
    FINRA = "finra"
    BASEL_III = "basel_iii"


@dataclass
class AuditEvent(ABC):
    """
    Base audit event class for all financial operations.

    Provides common fields and validation for all audit events,
    ensuring consistency and compliance across the system.
    """

    event_type: str
    resource_type: str
    resource_id: str
    action: str
    user_id: str | None = None
    session_id: str | None = None
    timestamp: datetime | None = None
    severity: EventSeverity = EventSeverity.MEDIUM
    is_critical: bool = False
    compliance_regulations: list[ComplianceRegulation] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization validation and defaults."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

        # Ensure timestamp is timezone-aware
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=UTC)

    @abstractmethod
    def get_resource_details(self) -> dict[str, Any]:
        """Get resource-specific details for the audit event."""
        pass

    def validate(self) -> None:
        """Validate audit event data."""
        if not self.event_type:
            raise AuditValidationError("Event type is required")

        if not self.resource_type:
            raise AuditValidationError("Resource type is required")

        if not self.resource_id:
            raise AuditValidationError("Resource ID is required")

        if not self.action:
            raise AuditValidationError("Action is required")

        # Validate resource-specific data
        self._validate_resource_data()

    @abstractmethod
    def _validate_resource_data(self) -> None:
        """Validate resource-specific data."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert audit event to dictionary representation."""
        base_data = {
            "event_type": self.event_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": (
                self.timestamp.isoformat() if self.timestamp else datetime.now(UTC).isoformat()
            ),
            "severity": self.severity.value,
            "is_critical": self.is_critical,
            "compliance_regulations": [reg.value for reg in self.compliance_regulations],
            "tags": self.tags,
            "metadata": self.metadata,
        }

        # Add resource-specific details
        base_data["resource_details"] = self.get_resource_details()

        return base_data

    def add_tag(self, tag: str) -> None:
        """Add tag to audit event."""
        if tag not in self.tags:
            self.tags.append(tag)

    def add_compliance_regulation(self, regulation: ComplianceRegulation) -> None:
        """Add compliance regulation to audit event."""
        if regulation not in self.compliance_regulations:
            self.compliance_regulations.append(regulation)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value


@dataclass
class OrderEvent(AuditEvent):
    """Audit event for order-related operations."""

    order_id: str = ""
    symbol: str = ""
    side: str | None = None  # "buy" or "sell"
    quantity: Decimal | None = None
    price: Decimal | None = None
    order_type: str | None = None  # "market", "limit", "stop"
    time_in_force: str | None = None
    status: str | None = None
    fill_quantity: Decimal | None = None
    fill_price: Decimal | None = None
    commission: Decimal | None = None
    broker_order_id: str | None = None

    def __post_init__(self) -> None:
        """Initialize order event."""
        super().__post_init__()
        if not self.resource_type:
            self.resource_type = "order"
        if not self.resource_id and self.order_id:
            self.resource_id = self.order_id

        # Order events are generally high importance
        if self.action in ["create", "modify", "cancel", "fill"]:
            self.severity = EventSeverity.HIGH

        # Critical for large orders or failures
        if (self.quantity and self.quantity > 10000) or self.action == "rejected":
            self.is_critical = True

        # Add relevant compliance regulations
        self.add_compliance_regulation(ComplianceRegulation.SEC)
        self.add_compliance_regulation(ComplianceRegulation.FINRA)

    def get_resource_details(self) -> dict[str, Any]:
        """Get order-specific details."""
        details = {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "time_in_force": self.time_in_force,
            "status": self.status,
        }

        # Add financial details if present
        if self.quantity is not None:
            details["quantity"] = str(self.quantity)
        if self.price is not None:
            details["price"] = str(self.price)
        if self.fill_quantity is not None:
            details["fill_quantity"] = str(self.fill_quantity)
        if self.fill_price is not None:
            details["fill_price"] = str(self.fill_price)
        if self.commission is not None:
            details["commission"] = str(self.commission)
        if self.broker_order_id:
            details["broker_order_id"] = self.broker_order_id

        return details

    def _validate_resource_data(self) -> None:
        """Validate order-specific data."""
        if not self.order_id:
            raise AuditValidationError("Order ID is required for order events")

        if not self.symbol:
            raise AuditValidationError("Symbol is required for order events")

        if self.side and self.side not in ["buy", "sell"]:
            raise AuditValidationError("Order side must be 'buy' or 'sell'")

        if self.quantity is not None and self.quantity <= 0:
            raise AuditValidationError("Order quantity must be positive")


@dataclass
class PositionEvent(AuditEvent):
    """Audit event for position-related operations."""

    position_id: str = ""
    symbol: str = ""
    quantity: Decimal | None = None
    average_cost: Decimal | None = None
    current_price: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    realized_pnl: Decimal | None = None
    portfolio_id: str | None = None

    def __post_init__(self) -> None:
        """Initialize position event."""
        super().__post_init__()
        if not self.resource_type:
            self.resource_type = "position"
        if not self.resource_id and self.position_id:
            self.resource_id = self.position_id

        # Position events are high importance
        self.severity = EventSeverity.HIGH

        # Critical for large positions or significant P&L
        if (self.quantity and abs(self.quantity) > 10000) or (
            self.unrealized_pnl and abs(self.unrealized_pnl) > 100000
        ):
            self.is_critical = True

        # Add relevant compliance regulations
        self.add_compliance_regulation(ComplianceRegulation.SEC)
        self.add_compliance_regulation(ComplianceRegulation.CFTC)

    def get_resource_details(self) -> dict[str, Any]:
        """Get position-specific details."""
        details = {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "portfolio_id": self.portfolio_id,
        }

        # Add financial details if present
        if self.quantity is not None:
            details["quantity"] = str(self.quantity)
        if self.average_cost is not None:
            details["average_cost"] = str(self.average_cost)
        if self.current_price is not None:
            details["current_price"] = str(self.current_price)
        if self.unrealized_pnl is not None:
            details["unrealized_pnl"] = str(self.unrealized_pnl)
        if self.realized_pnl is not None:
            details["realized_pnl"] = str(self.realized_pnl)

        return details

    def _validate_resource_data(self) -> None:
        """Validate position-specific data."""
        if not self.position_id:
            raise AuditValidationError("Position ID is required for position events")

        if not self.symbol:
            raise AuditValidationError("Symbol is required for position events")


@dataclass
class PortfolioEvent(AuditEvent):
    """Audit event for portfolio-related operations."""

    portfolio_id: str = ""
    portfolio_name: str | None = None
    cash_balance: Decimal | None = None
    market_value: Decimal | None = None
    total_value: Decimal | None = None
    buying_power: Decimal | None = None
    day_pnl: Decimal | None = None
    total_pnl: Decimal | None = None
    risk_metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize portfolio event."""
        super().__post_init__()
        if not self.resource_type:
            self.resource_type = "portfolio"
        if not self.resource_id and self.portfolio_id:
            self.resource_id = self.portfolio_id

        # Portfolio events are high importance
        self.severity = EventSeverity.HIGH

        # Critical for large portfolios or significant changes
        if (self.total_value and self.total_value > 1000000) or (
            self.day_pnl and abs(self.day_pnl) > 50000
        ):
            self.is_critical = True

        # Add relevant compliance regulations
        self.add_compliance_regulation(ComplianceRegulation.SEC)
        self.add_compliance_regulation(ComplianceRegulation.SOX)

    def get_resource_details(self) -> dict[str, Any]:
        """Get portfolio-specific details."""
        details: dict[str, Any] = {
            "portfolio_id": self.portfolio_id,
            "portfolio_name": self.portfolio_name,
        }

        # Add financial details if present
        if self.cash_balance is not None:
            details["cash_balance"] = str(self.cash_balance)
        if self.market_value is not None:
            details["market_value"] = str(self.market_value)
        if self.total_value is not None:
            details["total_value"] = str(self.total_value)
        if self.buying_power is not None:
            details["buying_power"] = str(self.buying_power)
        if self.day_pnl is not None:
            details["day_pnl"] = str(self.day_pnl)
        if self.total_pnl is not None:
            details["total_pnl"] = str(self.total_pnl)
        if self.risk_metrics:
            details["risk_metrics"] = self.risk_metrics

        return details

    def _validate_resource_data(self) -> None:
        """Validate portfolio-specific data."""
        if not self.portfolio_id:
            raise AuditValidationError("Portfolio ID is required for portfolio events")


@dataclass
class RiskEvent(AuditEvent):
    """Audit event for risk management operations."""

    risk_type: str = ""  # "position_limit", "loss_limit", "margin_call", etc.
    threshold_value: Decimal | None = None
    current_value: Decimal | None = None
    breach_percentage: Decimal | None = None
    risk_level: str | None = None  # "low", "medium", "high", "critical"
    affected_portfolios: list[str] = field(default_factory=list)
    mitigation_actions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize risk event."""
        super().__post_init__()
        if not self.resource_type:
            self.resource_type = "risk"
        if not self.resource_id and self.risk_type:
            self.resource_id = f"risk_{self.risk_type}_{uuid.uuid4().hex[:8]}"

        # Risk events are always high importance
        self.severity = EventSeverity.HIGH

        # Critical for significant risk breaches
        if self.risk_level == "critical" or (
            self.breach_percentage and self.breach_percentage > 50
        ):
            self.is_critical = True

        # Add relevant compliance regulations
        self.add_compliance_regulation(ComplianceRegulation.SEC)
        self.add_compliance_regulation(ComplianceRegulation.BASEL_III)

    def get_resource_details(self) -> dict[str, Any]:
        """Get risk-specific details."""
        details = {
            "risk_type": self.risk_type,
            "risk_level": self.risk_level,
            "affected_portfolios": self.affected_portfolios,
            "mitigation_actions": self.mitigation_actions,
        }

        # Add threshold details if present
        if self.threshold_value is not None:
            details["threshold_value"] = str(self.threshold_value)
        if self.current_value is not None:
            details["current_value"] = str(self.current_value)
        if self.breach_percentage is not None:
            details["breach_percentage"] = str(self.breach_percentage)

        return details

    def _validate_resource_data(self) -> None:
        """Validate risk-specific data."""
        if not self.risk_type:
            raise AuditValidationError("Risk type is required for risk events")


@dataclass
class AuthenticationEvent(AuditEvent):
    """Audit event for authentication and authorization operations."""

    auth_method: str | None = None  # "password", "api_key", "oauth", "mfa"
    ip_address: str | None = None
    user_agent: str | None = None
    login_success: bool | None = None
    failure_reason: str | None = None
    permissions_granted: list[str] = field(default_factory=list)
    session_duration: int | None = None  # seconds

    def __post_init__(self) -> None:
        """Initialize authentication event."""
        super().__post_init__()
        if not self.resource_type:
            self.resource_type = "authentication"
        if not self.resource_id and self.user_id:
            self.resource_id = f"auth_{self.user_id}_{uuid.uuid4().hex[:8]}"

        # Failed authentications are critical
        if self.login_success is False:
            self.severity = EventSeverity.CRITICAL
            self.is_critical = True

        # Add relevant compliance regulations
        self.add_compliance_regulation(ComplianceRegulation.SOX)
        self.add_compliance_regulation(ComplianceRegulation.GDPR)

    def get_resource_details(self) -> dict[str, Any]:
        """Get authentication-specific details."""
        details = {
            "auth_method": self.auth_method,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "login_success": self.login_success,
            "failure_reason": self.failure_reason,
            "permissions_granted": self.permissions_granted,
        }

        if self.session_duration is not None:
            details["session_duration"] = str(self.session_duration)

        return details

    def _validate_resource_data(self) -> None:
        """Validate authentication-specific data."""
        if self.login_success is None:
            raise AuditValidationError("Login success status is required for auth events")


@dataclass
class ConfigurationEvent(AuditEvent):
    """Audit event for system configuration changes."""

    config_key: str = ""
    old_value: Any | None = None
    new_value: Any | None = None
    config_category: str | None = None  # "risk", "trading", "compliance", etc.
    change_reason: str | None = None
    approval_required: bool = False
    approved_by: str | None = None

    def __post_init__(self) -> None:
        """Initialize configuration event."""
        super().__post_init__()
        if not self.resource_type:
            self.resource_type = "configuration"
        if not self.resource_id and self.config_key:
            self.resource_id = f"config_{self.config_key}"

        # Configuration changes are high importance
        self.severity = EventSeverity.HIGH

        # Critical configuration changes
        critical_categories = ["risk", "compliance", "security"]
        if self.config_category in critical_categories:
            self.is_critical = True

        # Add relevant compliance regulations
        self.add_compliance_regulation(ComplianceRegulation.SOX)

    def get_resource_details(self) -> dict[str, Any]:
        """Get configuration-specific details."""
        return {
            "config_key": self.config_key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "config_category": self.config_category,
            "change_reason": self.change_reason,
            "approval_required": self.approval_required,
            "approved_by": self.approved_by,
        }

    def _validate_resource_data(self) -> None:
        """Validate configuration-specific data."""
        if not self.config_key:
            raise AuditValidationError("Config key is required for configuration events")


@dataclass
class ComplianceEvent(AuditEvent):
    """Audit event for compliance-specific operations."""

    regulation: ComplianceRegulation | None = None
    compliance_rule: str = ""
    check_result: bool = False
    violation_details: str | None = None
    remediation_actions: list[str] = field(default_factory=list)
    report_required: bool = True
    deadline: datetime | None = None

    def __post_init__(self) -> None:
        """Initialize compliance event."""
        super().__post_init__()
        if not self.resource_type:
            self.resource_type = "compliance"
        if not self.resource_id:
            self.resource_id = f"compliance_{self.regulation.value if self.regulation else 'unknown'}_{uuid.uuid4().hex[:8]}"

        # Compliance violations are always critical
        if not self.check_result:
            self.severity = EventSeverity.CRITICAL
            self.is_critical = True

        # Add the specific regulation
        if self.regulation:
            self.add_compliance_regulation(self.regulation)

    def get_resource_details(self) -> dict[str, Any]:
        """Get compliance-specific details."""
        details = {
            "regulation": self.regulation.value if self.regulation else None,
            "compliance_rule": self.compliance_rule,
            "check_result": self.check_result,
            "violation_details": self.violation_details,
            "remediation_actions": self.remediation_actions,
            "report_required": self.report_required,
        }

        if self.deadline:
            details["deadline"] = self.deadline.isoformat()

        return details

    def _validate_resource_data(self) -> None:
        """Validate compliance-specific data."""
        if not self.compliance_rule:
            raise AuditValidationError("Compliance rule is required for compliance events")
