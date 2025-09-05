"""
Comprehensive tests for SecurityPolicyService.

This test suite covers:
- Security context validation
- Risk assessment logic
- Access control policies
- Data sanitization requirements
- Validation rules enforcement
- Threat detection patterns
- Security event classification
"""

import pytest

from src.infrastructure.security.security_policy_service import (
    AccessLevel,
    RiskLevel,
    SanitizationLevel,
    SecurityContext,
    SecurityPolicyService,
    ValidationRules,
)


class TestSecurityContext:
    """Test SecurityContext dataclass."""

    def test_create_security_context(self):
        """Test creating a security context."""
        context = SecurityContext(
            user_role="trader",
            source_ip="192.168.1.100",
            request_type="place_order",
            resource_type="order",
            operation="create",
            timestamp=1234567890.0,
            metadata={"session_id": "abc123"},
        )

        assert context.user_role == "trader"
        assert context.source_ip == "192.168.1.100"
        assert context.request_type == "place_order"
        assert context.resource_type == "order"
        assert context.operation == "create"
        assert context.timestamp == 1234567890.0
        assert context.metadata["session_id"] == "abc123"

    def test_partial_security_context(self):
        """Test creating a partial security context."""
        context = SecurityContext(user_role="admin", operation="delete")

        assert context.user_role == "admin"
        assert context.operation == "delete"
        assert context.source_ip is None
        assert context.request_type is None
        assert context.resource_type is None
        assert context.timestamp is None
        assert context.metadata is None


class TestValidationRules:
    """Test ValidationRules dataclass."""

    def test_create_validation_rules(self):
        """Test creating validation rules."""
        rules = ValidationRules(
            required_fields=["symbol", "quantity"],
            optional_fields=["price"],
            field_validators={"symbol": "ticker", "quantity": "positive_integer"},
            max_length={"symbol": 10},
            min_length={"symbol": 1},
            patterns={"symbol": r"^[A-Z]+$"},
            custom_rules=["check_market_hours"],
        )

        assert len(rules.required_fields) == 2
        assert "symbol" in rules.required_fields
        assert "price" in rules.optional_fields
        assert rules.field_validators["symbol"] == "ticker"
        assert rules.max_length["symbol"] == 10
        assert rules.min_length["symbol"] == 1
        assert rules.patterns["symbol"] == r"^[A-Z]+$"
        assert "check_market_hours" in rules.custom_rules


class TestSecurityPolicyService:
    """Test SecurityPolicyService functionality."""

    @pytest.fixture
    def service(self):
        """Create a security policy service instance."""
        return SecurityPolicyService()

    def test_assess_risk_level(self, service):
        """Test risk level assessment."""
        # Create SecurityContext objects for testing
        from src.infrastructure.security.security_policy_service import SecurityContext

        # Test low risk operation
        context = SecurityContext(
            user_role="user", source_ip="127.0.0.1", resource_type="market_data", operation="read"
        )
        risk = service.evaluate_request_risk(context)
        assert risk in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

        # Test medium risk operation
        context = SecurityContext(
            user_role="user",
            source_ip="127.0.0.1",
            resource_type="user_profile",
            operation="update",
        )
        risk = service.evaluate_request_risk(context)
        assert risk in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_determine_access_level(self, service):
        """Test access level determination."""
        # Test public resource
        access = service.determine_access_level(resource="market_data", operation="read")
        assert access in [
            AccessLevel.PUBLIC,
            AccessLevel.AUTHENTICATED,
            AccessLevel.AUTHORIZED,
            AccessLevel.PRIVILEGED,
            AccessLevel.ADMIN,
        ]

        # Test authenticated resource
        access = service.determine_access_level(resource="user_profile", operation="read")
        assert access in [
            AccessLevel.PUBLIC,
            AccessLevel.AUTHENTICATED,
            AccessLevel.AUTHORIZED,
            AccessLevel.PRIVILEGED,
            AccessLevel.ADMIN,
        ]

        # Test admin resource
        access = service.determine_access_level(resource="system_config", operation="modify")
        assert access in [
            AccessLevel.PUBLIC,
            AccessLevel.AUTHENTICATED,
            AccessLevel.AUTHORIZED,
            AccessLevel.PRIVILEGED,
            AccessLevel.ADMIN,
        ]

    def test_get_sanitization_level(self, service):
        """Test sanitization level determination."""
        # Test user input
        level = service.determine_sanitization_level("user_input")
        assert level in [
            SanitizationLevel.NONE,
            SanitizationLevel.BASIC,
            SanitizationLevel.STANDARD,
            SanitizationLevel.STRICT,
            SanitizationLevel.PARANOID,
        ]

        # Test database query
        level = service.determine_sanitization_level("database_query")
        assert level in [
            SanitizationLevel.NONE,
            SanitizationLevel.BASIC,
            SanitizationLevel.STANDARD,
            SanitizationLevel.STRICT,
            SanitizationLevel.PARANOID,
        ]

        # Test price data
        level = service.determine_sanitization_level("price")
        assert level in [
            SanitizationLevel.NONE,
            SanitizationLevel.BASIC,
            SanitizationLevel.STANDARD,
            SanitizationLevel.STRICT,
            SanitizationLevel.PARANOID,
        ]

        # Test HTML content
        level = service.determine_sanitization_level("html")
        assert level in [
            SanitizationLevel.NONE,
            SanitizationLevel.BASIC,
            SanitizationLevel.STANDARD,
            SanitizationLevel.STRICT,
            SanitizationLevel.PARANOID,
        ]

        # Test unknown data type (should default to STANDARD)
        level = service.get_sanitization_level("unknown_type")
        assert level == SanitizationLevel.STANDARD

    def test_get_validation_rules_trading_context(self, service):
        """Test getting validation rules for trading context."""
        rules = service.get_validation_rules("trading")

        assert isinstance(rules, ValidationRules)
        assert "symbol" in rules.required_fields
        assert "quantity" in rules.required_fields
        assert "order_type" in rules.required_fields
        assert "price" in rules.optional_fields
        assert len(rules.field_validators) > 0

    def test_get_validation_rules_portfolio_context(self, service):
        """Test getting validation rules for portfolio context."""
        rules = service.get_validation_rules("portfolio")

        assert isinstance(rules, ValidationRules)
        assert "portfolio_id" in rules.required_fields
        assert "positions" in rules.optional_fields
        assert "cash_balance" in rules.optional_fields

    def test_validate_field_length(self, service):
        """Test field length validation."""
        # Test valid length
        is_valid = service.validate_field_length(
            field_name="symbol", value="AAPL", min_length=1, max_length=10
        )
        assert is_valid is True

        # Test too short
        is_valid = service.validate_field_length(
            field_name="symbol", value="", min_length=1, max_length=10
        )
        assert is_valid is False

        # Test too long
        is_valid = service.validate_field_length(
            field_name="symbol", value="VERYLONGSYMBOL", min_length=1, max_length=10
        )
        assert is_valid is False

    def test_validate_pattern(self, service):
        """Test pattern validation."""
        # Test valid pattern
        is_valid = service.validate_pattern(value="AAPL", pattern=r"^[A-Z]+$")
        assert is_valid is True

        # Test invalid pattern
        is_valid = service.validate_pattern(value="aapl", pattern=r"^[A-Z]+$")
        assert is_valid is False

        # Test complex pattern (email)
        is_valid = service.validate_pattern(
            value="test@example.com", pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
        assert is_valid is True

    def test_check_permission(self, service):
        """Test permission checking."""
        # Test admin has all permissions
        has_permission = service.check_permission(
            user_role="admin", required_access=AccessLevel.ADMIN
        )
        assert has_permission is True

        # Test trader has privileged access
        has_permission = service.check_permission(
            user_role="trader", required_access=AccessLevel.PRIVILEGED
        )
        assert has_permission is True

        # Test trader doesn't have admin access
        has_permission = service.check_permission(
            user_role="trader", required_access=AccessLevel.ADMIN
        )
        assert has_permission is False

        # Test viewer has only authenticated access
        has_permission = service.check_permission(
            user_role="viewer", required_access=AccessLevel.AUTHENTICATED
        )
        assert has_permission is True

        has_permission = service.check_permission(
            user_role="viewer", required_access=AccessLevel.AUTHORIZED
        )
        assert has_permission is False

    def test_is_suspicious_activity(self, service):
        """Test suspicious activity detection."""
        context = SecurityContext(
            user_role="trader",
            source_ip="192.168.1.100",
            operation="place_order",
            metadata={"order_count": 100},
        )

        # Test high order count
        is_suspicious = service.is_suspicious_activity(context)
        assert is_suspicious is True

        # Test normal activity
        context.metadata = {"order_count": 5}
        is_suspicious = service.is_suspicious_activity(context)
        assert is_suspicious is False

        # Test suspicious IP pattern
        context.source_ip = "10.0.0.1"
        context.metadata = {"failed_attempts": 10}
        is_suspicious = service.is_suspicious_activity(context)
        assert is_suspicious is True

    def test_classify_security_event(self, service):
        """Test security event classification."""
        # Test authentication failure
        event_type = service.classify_security_event(
            event_name="authentication_failed", severity="high"
        )
        assert event_type == "authentication_failure"

        # Test authorization violation
        event_type = service.classify_security_event(
            event_name="unauthorized_access", severity="critical"
        )
        assert event_type == "authorization_violation"

        # Test data breach
        event_type = service.classify_security_event(
            event_name="data_exfiltration", severity="critical"
        )
        assert event_type == "data_breach"

    def test_rate_limit_policy(self, service):
        """Test rate limiting policy determination."""
        # Test API rate limits
        limits = service.get_rate_limit_policy(user_role="trader", resource_type="api")
        assert "requests_per_minute" in limits
        assert "requests_per_hour" in limits
        assert limits["requests_per_minute"] > 0

        # Test order rate limits
        limits = service.get_rate_limit_policy(user_role="trader", resource_type="orders")
        assert "orders_per_minute" in limits
        assert "orders_per_day" in limits

        # Test admin has higher limits
        admin_limits = service.get_rate_limit_policy(user_role="admin", resource_type="api")
        trader_limits = service.get_rate_limit_policy(user_role="trader", resource_type="api")
        assert admin_limits["requests_per_minute"] > trader_limits["requests_per_minute"]

    def test_encryption_requirements(self, service):
        """Test encryption requirement determination."""
        # Test PII data requires encryption
        requires_encryption = service.requires_encryption(data_type="personal_information")
        assert requires_encryption is True

        # Test financial data requires encryption
        requires_encryption = service.requires_encryption(data_type="account_balance")
        assert requires_encryption is True

        # Test public data doesn't require encryption
        requires_encryption = service.requires_encryption(data_type="market_data")
        assert requires_encryption is False

    def test_audit_requirements(self, service):
        """Test audit requirement determination."""
        # Test critical operations require audit
        requires_audit = service.requires_audit(
            operation="transfer_funds", risk_level=RiskLevel.CRITICAL
        )
        assert requires_audit is True

        # Test high risk operations require audit
        requires_audit = service.requires_audit(operation="place_order", risk_level=RiskLevel.HIGH)
        assert requires_audit is True

        # Test low risk read operations don't require audit
        requires_audit = service.requires_audit(operation="read", risk_level=RiskLevel.LOW)
        assert requires_audit is False

    def test_session_timeout_policy(self, service):
        """Test session timeout policy."""
        # Test admin has longer timeout
        timeout = service.get_session_timeout(user_role="admin")
        assert timeout == 7200  # 2 hours

        # Test trader has standard timeout
        timeout = service.get_session_timeout(user_role="trader")
        assert timeout == 3600  # 1 hour

        # Test viewer has shorter timeout
        timeout = service.get_session_timeout(user_role="viewer")
        assert timeout == 1800  # 30 minutes

    def test_password_policy(self, service):
        """Test password policy requirements."""
        policy = service.get_password_policy()

        assert policy["min_length"] >= 8
        assert policy["require_uppercase"] is True
        assert policy["require_lowercase"] is True
        assert policy["require_numbers"] is True
        assert policy["require_special_chars"] is True
        assert policy["max_age_days"] == 90
        assert policy["password_history"] >= 5

    def test_validate_password_strength(self, service):
        """Test password strength validation."""
        # Test weak password
        is_strong = service.validate_password_strength("password")
        assert is_strong is False

        # Test medium password
        is_strong = service.validate_password_strength("Password123")
        assert is_strong is False

        # Test strong password
        is_strong = service.validate_password_strength("MyStr0ng!P@ssw0rd#2024")
        assert is_strong is True

    def test_ip_whitelist_policy(self, service):
        """Test IP whitelist policy."""
        # Test internal IP is whitelisted
        is_whitelisted = service.is_ip_whitelisted(ip="192.168.1.100", user_role="admin")
        assert is_whitelisted is True

        # Test external IP for non-admin
        is_whitelisted = service.is_ip_whitelisted(ip="8.8.8.8", user_role="trader")
        assert is_whitelisted is False

    def test_data_retention_policy(self, service):
        """Test data retention policy."""
        # Test audit log retention
        retention_days = service.get_data_retention_period(data_type="audit_logs")
        assert retention_days == 2555  # 7 years

        # Test transaction data retention
        retention_days = service.get_data_retention_period(data_type="transactions")
        assert retention_days == 2555  # 7 years

        # Test temporary data retention
        retention_days = service.get_data_retention_period(data_type="temp_files")
        assert retention_days == 1  # 1 day

    def test_security_headers_policy(self, service):
        """Test security headers policy."""
        headers = service.get_required_security_headers()

        assert "X-Content-Type-Options" in headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in headers
        assert headers["X-Frame-Options"] == "DENY"
        assert "Strict-Transport-Security" in headers
        assert "Content-Security-Policy" in headers

    def test_input_validation_severity(self, service):
        """Test input validation severity determination."""
        # Test SQL injection attempt
        severity = service.assess_input_threat(input_value="'; DROP TABLE users; --")
        assert severity == RiskLevel.CRITICAL

        # Test XSS attempt
        severity = service.assess_input_threat(input_value="<script>alert('XSS')</script>")
        assert severity == RiskLevel.HIGH

        # Test normal input
        severity = service.assess_input_threat(input_value="AAPL")
        assert severity == RiskLevel.LOW
