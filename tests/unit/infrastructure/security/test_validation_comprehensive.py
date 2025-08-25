"""
Comprehensive unit tests for ValidationError and Security Validation.

Tests the enhanced validation system including security validation,
schema validation, and trading-specific validation.
"""

from decimal import Decimal

import pytest

from src.infrastructure.security.validation import (
    SchemaValidationError,
    SchemaValidator,
    SecurityValidationError,
    SecurityValidator,
    TradingInputValidator,
    ValidationError,
)


@pytest.mark.unit
class TestSecurityValidator:
    """Test SecurityValidator class."""

    def test_validate_sql_injection_safe(self):
        """Test SQL injection detection with safe strings."""
        assert SecurityValidator.validate_sql_injection("normal text") is True
        assert SecurityValidator.validate_sql_injection("user@example.com") is True
        assert SecurityValidator.validate_sql_injection("AAPL stock price") is True

    def test_validate_sql_injection_unsafe(self):
        """Test SQL injection detection with unsafe strings."""
        assert SecurityValidator.validate_sql_injection("SELECT * FROM users") is False
        assert SecurityValidator.validate_sql_injection("1' OR '1'='1") is False
        assert SecurityValidator.validate_sql_injection("'; DROP TABLE users; --") is False
        assert (
            SecurityValidator.validate_sql_injection("UNION SELECT password FROM accounts") is False
        )

    def test_validate_xss_safe(self):
        """Test XSS detection with safe strings."""
        assert SecurityValidator.validate_xss("normal text") is True
        assert SecurityValidator.validate_xss("Click here") is True
        assert SecurityValidator.validate_xss("Price: $100.50") is True

    def test_validate_xss_unsafe(self):
        """Test XSS detection with unsafe strings."""
        assert SecurityValidator.validate_xss("<script>alert('xss')</script>") is False
        assert SecurityValidator.validate_xss("javascript:alert(1)") is False
        assert SecurityValidator.validate_xss("onclick='malicious()'") is False
        assert SecurityValidator.validate_xss("data:text/html,<script>alert(1)</script>") is False

    def test_validate_trading_symbol(self):
        """Test trading symbol validation."""
        # Valid symbols
        assert SecurityValidator.validate_trading_symbol("AAPL") is True
        assert SecurityValidator.validate_trading_symbol("MSFT") is True
        assert SecurityValidator.validate_trading_symbol("A") is True
        assert SecurityValidator.validate_trading_symbol("GOOGL") is True

        # Invalid symbols
        assert SecurityValidator.validate_trading_symbol("") is False
        assert SecurityValidator.validate_trading_symbol("aapl") is False  # lowercase
        assert SecurityValidator.validate_trading_symbol("TOOLONG") is False  # too long
        assert SecurityValidator.validate_trading_symbol("A123") is False  # contains numbers
        assert SecurityValidator.validate_trading_symbol("A-B") is False  # contains dash

    def test_validate_currency_code(self):
        """Test currency code validation."""
        # Valid codes
        assert SecurityValidator.validate_currency_code("USD") is True
        assert SecurityValidator.validate_currency_code("EUR") is True
        assert SecurityValidator.validate_currency_code("GBP") is True

        # Invalid codes
        assert SecurityValidator.validate_currency_code("") is False
        assert SecurityValidator.validate_currency_code("US") is False  # too short
        assert SecurityValidator.validate_currency_code("USDX") is False  # too long
        assert SecurityValidator.validate_currency_code("usd") is False  # lowercase

    def test_validate_price(self):
        """Test price validation."""
        # Valid prices
        assert SecurityValidator.validate_price("100.50") is True
        assert SecurityValidator.validate_price("0.01") is True
        assert SecurityValidator.validate_price(100.0) is True
        assert SecurityValidator.validate_price(Decimal("50.25")) is True

        # Invalid prices
        assert SecurityValidator.validate_price("0") is False  # zero
        assert SecurityValidator.validate_price("-10") is False  # negative
        assert SecurityValidator.validate_price("invalid") is False  # non-numeric
        assert SecurityValidator.validate_price("1000000") is False  # too large

    def test_validate_quantity(self):
        """Test quantity validation."""
        # Valid quantities
        assert SecurityValidator.validate_quantity("100") is True
        assert SecurityValidator.validate_quantity("10.5") is True
        assert SecurityValidator.validate_quantity(100) is True
        assert SecurityValidator.validate_quantity(0.1) is True

        # Invalid quantities
        assert SecurityValidator.validate_quantity("0") is False  # zero
        assert SecurityValidator.validate_quantity("-5") is False  # negative
        assert SecurityValidator.validate_quantity("invalid") is False  # non-numeric
        assert SecurityValidator.validate_quantity("10000000") is False  # too large

    def test_validate_ip_address(self):
        """Test IP address validation."""
        # Valid IPs
        assert SecurityValidator.validate_ip_address("192.168.1.1") is True
        assert SecurityValidator.validate_ip_address("127.0.0.1") is True
        assert SecurityValidator.validate_ip_address("::1") is True  # IPv6
        assert SecurityValidator.validate_ip_address("2001:db8::1") is True  # IPv6

        # Invalid IPs
        assert SecurityValidator.validate_ip_address("invalid") is False
        assert SecurityValidator.validate_ip_address("256.256.256.256") is False
        assert SecurityValidator.validate_ip_address("192.168.1") is False  # incomplete

    def test_validate_url(self):
        """Test URL validation."""
        # Valid URLs
        assert SecurityValidator.validate_url("https://example.com") is True
        assert SecurityValidator.validate_url("http://localhost:8080") is True
        assert SecurityValidator.validate_url("ftp://ftp.example.com") is True

        # Invalid URLs
        assert SecurityValidator.validate_url("invalid") is False
        assert SecurityValidator.validate_url("javascript:alert(1)") is False  # unsafe scheme
        assert SecurityValidator.validate_url("file:///etc/passwd") is False  # unsafe scheme
        assert SecurityValidator.validate_url("https://") is False  # no netloc

    def test_validate_email(self):
        """Test email validation."""
        # Valid emails
        assert SecurityValidator.validate_email("user@example.com") is True
        assert SecurityValidator.validate_email("test.email+tag@example.co.uk") is True

        # Invalid emails
        assert SecurityValidator.validate_email("invalid") is False
        assert SecurityValidator.validate_email("@example.com") is False
        assert SecurityValidator.validate_email("user@") is False
        assert SecurityValidator.validate_email("user.example.com") is False

    def test_validate_json_structure(self):
        """Test JSON structure validation."""
        # Valid JSON
        assert SecurityValidator.validate_json_structure('{"key": "value"}') is True
        assert SecurityValidator.validate_json_structure("[]") is True
        assert SecurityValidator.validate_json_structure('{"nested": {"deep": "value"}}') is True

        # Invalid JSON
        assert SecurityValidator.validate_json_structure("invalid json") is False
        assert SecurityValidator.validate_json_structure('{"incomplete":') is False

        # Deep nesting (should fail with default max_depth=10)
        deep_json = '{"level": ' * 15 + '"value"' + "}" * 15
        assert SecurityValidator.validate_json_structure(deep_json) is False


@pytest.mark.unit
class TestSchemaValidator:
    """Test SchemaValidator class."""

    def test_validate_schema_valid(self):
        """Test schema validation with valid data."""
        schema = {
            "required": ["name", "age"],
            "fields": {
                "name": {"type": "string", "min_length": 1, "max_length": 50},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "email": {"type": "string", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
            },
        }

        data = {"name": "John", "age": 30, "email": "john@example.com"}
        errors = SchemaValidator.validate_schema(data, schema)
        assert errors == []

    def test_validate_schema_missing_required(self):
        """Test schema validation with missing required fields."""
        schema = {"required": ["name"], "fields": {}}
        data = {}
        errors = SchemaValidator.validate_schema(data, schema)
        assert len(errors) == 1
        assert "Required field 'name' is missing" in errors[0]

    def test_validate_schema_type_mismatch(self):
        """Test schema validation with type mismatches."""
        schema = {"fields": {"age": {"type": "integer"}}}
        data = {"age": "not_an_integer"}
        errors = SchemaValidator.validate_schema(data, schema)
        assert len(errors) == 1
        assert "must be of type integer" in errors[0]

    def test_validate_schema_string_constraints(self):
        """Test schema validation with string constraints."""
        schema = {
            "fields": {
                "name": {"type": "string", "min_length": 5, "max_length": 10},
                "code": {"type": "string", "pattern": r"^[A-Z]{3}$"},
            }
        }

        # Too short
        data = {"name": "Jo", "code": "ABC"}
        errors = SchemaValidator.validate_schema(data, schema)
        assert any("at least 5 characters" in error for error in errors)

        # Too long
        data = {"name": "VeryLongName", "code": "ABC"}
        errors = SchemaValidator.validate_schema(data, schema)
        assert any("at most 10 characters" in error for error in errors)

        # Pattern mismatch
        data = {"name": "ValidName", "code": "invalid"}
        errors = SchemaValidator.validate_schema(data, schema)
        assert any("does not match required pattern" in error for error in errors)

    def test_validate_schema_numeric_constraints(self):
        """Test schema validation with numeric constraints."""
        schema = {
            "fields": {
                "price": {"type": "float", "minimum": 0.0, "maximum": 1000.0},
                "quantity": {"type": "integer", "minimum": 1},
            }
        }

        # Below minimum
        data = {"price": -10.0, "quantity": 0}
        errors = SchemaValidator.validate_schema(data, schema)
        assert any("at least 0.0" in error for error in errors)
        assert any("at least 1" in error for error in errors)

        # Above maximum
        data = {"price": 1500.0, "quantity": 10}
        errors = SchemaValidator.validate_schema(data, schema)
        assert any("at most 1000.0" in error for error in errors)


@pytest.mark.unit
class TestTradingInputValidator:
    """Test TradingInputValidator class."""

    def test_validate_order_valid(self):
        """Test valid trading order validation."""
        order_data = {"symbol": "AAPL", "quantity": 100.0, "order_type": "market", "side": "buy"}
        errors = TradingInputValidator.validate_order(order_data)
        assert errors == []

    def test_validate_order_limit_with_price(self):
        """Test limit order validation with price."""
        order_data = {
            "symbol": "AAPL",
            "quantity": 100.0,
            "price": 150.0,
            "order_type": "limit",
            "side": "buy",
        }
        errors = TradingInputValidator.validate_order(order_data)
        assert errors == []

    def test_validate_order_limit_missing_price(self):
        """Test limit order validation missing price."""
        order_data = {"symbol": "AAPL", "quantity": 100.0, "order_type": "limit", "side": "buy"}
        errors = TradingInputValidator.validate_order(order_data)
        assert any("Price is required for limit" in error for error in errors)

    def test_validate_order_invalid_fields(self):
        """Test order validation with invalid fields."""
        order_data = {
            "symbol": "invalid_symbol",  # invalid format
            "quantity": -10,  # negative quantity
            "order_type": "invalid_type",  # invalid type
            "side": "invalid_side",  # invalid side
        }
        errors = TradingInputValidator.validate_order(order_data)
        assert len(errors) >= 4  # Should have multiple errors

    def test_validate_portfolio_data_valid(self):
        """Test valid portfolio data validation."""
        portfolio_data = {
            "positions": [{"symbol": "AAPL", "quantity": 100}, {"symbol": "MSFT", "quantity": 50}]
        }
        errors = TradingInputValidator.validate_portfolio_data(portfolio_data)
        assert errors == []

    def test_validate_portfolio_data_invalid_positions(self):
        """Test portfolio data validation with invalid positions."""
        portfolio_data = {
            "positions": [
                {"symbol": "invalid123", "quantity": -10},  # invalid symbol and quantity
                {"symbol": "AAPL"},  # missing quantity
            ]
        }
        errors = TradingInputValidator.validate_portfolio_data(portfolio_data)
        assert len(errors) >= 3  # Should have multiple errors


@pytest.mark.unit
class TestDecorators:
    """Test validation decorators."""

    def test_security_validate_decorator(self):
        """Test security_validate decorator function."""
        validator = security_validate("test_field", "string")

        # Test valid string
        result = validator("safe string")
        assert isinstance(result, str)

        # Test SQL injection detection
        with pytest.raises(SecurityValidationError):
            validator("SELECT * FROM users")

        # Test XSS detection
        with pytest.raises(SecurityValidationError):
            validator("<script>alert(1)</script>")

    def test_security_validate_email(self):
        """Test security_validate with email validation."""
        validator = security_validate("email_field", "email")

        # Valid email
        result = validator("user@example.com")
        assert isinstance(result, str)

        # Invalid email
        with pytest.raises(ValidationError):
            validator("invalid_email")

    def test_security_validate_symbol(self):
        """Test security_validate with symbol validation."""
        validator = security_validate("symbol_field", "symbol")

        # Valid symbol
        result = validator("AAPL")
        assert result == "AAPL"

        # Invalid symbol
        with pytest.raises(ValidationError):
            validator("invalid123")

    def test_security_validate_price(self):
        """Test security_validate with price validation."""
        validator = security_validate("price_field", "price")

        # Valid price
        result = validator("100.50")
        assert isinstance(result, str)

        # Invalid price
        with pytest.raises(ValidationError):
            validator("-10")


@pytest.mark.unit
class TestExceptions:
    """Test custom exception classes."""

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Test error", field="test_field", value="test_value")
        assert str(error) == "Test error"
        assert error.field == "test_field"
        assert error == "test_value"

    def test_schema_validation_error(self):
        """Test SchemaValidationError exception."""
        schema_errors = ["Error 1", "Error 2"]
        error = SchemaValidationError("Schema failed", schema_errors)
        assert str(error) == "Schema failed"
        assert error.schema_errors == schema_errors

    def test_security_validation_error(self):
        """Test SecurityValidationError exception."""
        error = SecurityValidationError("Security check failed")
        assert str(error) == "Security check failed"
        assert isinstance(error, ValidationError)  # Should inherit from ValidationError
