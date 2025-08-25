"""
Comprehensive unit tests for input validation module.

Tests all validation methods, SQL injection prevention, XSS prevention,
decorators, and sanitization functions.
"""

from typing import Any
from unittest.mock import patch

import pytest

from src.infrastructure.security.validation import (
    SchemaValidationError,
    SchemaValidator,
    SecurityValidationError,
    SecurityValidator,
    TradingInputValidator,
    ValidationError,
    check_and_sanitize,
    check_required,
    sanitize_input,
    sanitize_sql_params,
    security_check,
)


class TestValidationErrors:
    """Test validation error classes."""

    def test_validation_error_with_field_and_value(self):
        """Test ValidationError with all attributes."""
        error = ValidationError("Test error", field="username", value="test123")
        assert str(error) == "Test error"
        assert error.field == "username"
        assert error == "test123"

    def test_validation_error_without_field(self):
        """Test ValidationError without field attribute."""
        error = ValidationError("Test error")
        assert str(error) == "Test error"
        assert error.field is None
        assert error is None

    def test_schema_validation_error(self):
        """Test SchemaValidationError with schema errors."""
        schema_errors = ["Field 'name' is required", "Field 'age' must be integer"]
        error = SchemaValidationError("Schema validation failed", schema_errors)
        assert str(error) == "Schema validation failed"
        assert error.schema_errors == schema_errors

    def test_security_validation_error(self):
        """Test SecurityValidationError."""
        error = SecurityValidationError("Security check failed")
        assert str(error) == "Security check failed"
        assert isinstance(error, ValidationError)


class TestSanitizeInputDecorator:
    """Test sanitize_input decorator."""

    @patch("src.infrastructure.security.input_sanitizer.InputSanitizer.sanitize_string")
    def test_sanitize_input_success(self, mock_sanitize):
        """Test successful input sanitization."""
        mock_sanitize.return_value = "sanitized_value"

        @sanitize_input(user_input=lambda x: mock_sanitize(x, max_length=100))
        def process_data(user_input: str) -> str:
            return f"processed: {user_input}"

        result = process_data("test_input")
        assert result == "processed: sanitized_value"
        mock_sanitize.assert_called_once_with("test_input", max_length=100)

    @patch("src.infrastructure.security.input_sanitizer.InputSanitizer")
    def test_sanitize_input_with_sanitization_error(self, mock_sanitizer_class):
        """Test sanitization error handling."""
        from src.infrastructure.security.input_sanitizer import SanitizationError

        mock_sanitizer = mock_sanitizer_class.return_value
        mock_sanitizer.sanitize_string.side_effect = SanitizationError("Dangerous input")

        @sanitize_input(user_input=lambda x: mock_sanitizer.sanitize_string(x))
        def process_data(user_input: str) -> str:
            return f"processed: {user_input}"

        with pytest.raises(ValidationError) as exc_info:
            process_data("malicious_input")
        assert "Invalid user_input" in str(exc_info)

    def test_sanitize_input_multiple_parameters(self):
        """Test sanitizing multiple parameters."""

        @sanitize_input(param1=lambda x: f"clean_{x}", param2=lambda x: x.upper())
        def process_multiple(param1: str, param2: str, param3: str) -> str:
            return f"{param1}|{param2}|{param3}"

        result = process_multiple("test1", "test2", "test3")
        assert result == "clean_test1|TEST2|test3"

    def test_sanitize_input_with_defaults(self):
        """Test sanitizer with default parameters."""

        @sanitize_input(optional_param=lambda x: x.strip() if x else None)
        def process_with_defaults(required: str, optional_param: str = "  default  ") -> str:
            return f"{required}:{optional_param}"

        result = process_with_defaults("test")
        assert result == "test:default"


class TestCheckRequiredDecorator:
    """Test check_required decorator."""

    def test_check_required_valid_types(self):
        """Test type checking with valid types."""

        @check_required(api_key=str, port=int)
        def connect(api_key: str, port: int) -> str:
            return f"Connected to port {port}"

        result = connect("secret_key", 8080)
        assert result == "Connected to port 8080"

    def test_check_required_invalid_type(self):
        """Test type checking with invalid type."""

        @check_required(api_key=str, port=int)
        def connect(api_key: str, port: int) -> str:
            return f"Connected to port {port}"

        with pytest.raises(ValidationError) as exc_info:
            connect("secret_key", "not_an_int")
        assert "Parameter 'port' must be of type int" in str(exc_info)

    def test_check_required_with_none_values(self):
        """Test that None values are allowed."""

        @check_required(optional=str)
        def process(optional: str = None) -> str:
            return f"Value: {optional}"

        result = process()
        assert result == "Value: None"

    def test_check_required_with_subclass(self):
        """Test type checking with subclasses."""

        @check_required(value=int)
        def process(value: int) -> str:
            return f"Value: {value}"

        # bool is a subclass of int in Python
        result = process(True)
        assert result == "Value: True"


class TestSanitizeSqlParams:
    """Test SQL parameter sanitization."""

    def test_sanitize_sql_params_strings(self):
        """Test sanitizing string parameters."""
        with patch("src.infrastructure.security.input_sanitizer.InputSanitizer") as mock_cls:
            mock_sanitizer = mock_cls.return_value
            mock_sanitizer.sanitize_string.side_effect = lambda x: f"safe_{x}"

            params = {"name": "John", "description": "User description"}
            result = sanitize_sql_params(params)

            assert result == {"name": "safe_John", "description": "safe_User description"}
            assert mock_sanitizer.sanitize_string.call_count == 2

    def test_sanitize_sql_params_mixed_types(self):
        """Test sanitizing mixed parameter types."""
        params = {"id": 123, "name": "Test", "active": True, "balance": 99.99}

        with patch("src.infrastructure.security.input_sanitizer.InputSanitizer") as mock_cls:
            mock_sanitizer = mock_cls.return_value
            mock_sanitizer.sanitize_string.return_value = "safe_Test"

            result = sanitize_sql_params(params)

            # Non-string values should be kept as-is
            assert result["id"] == 123
            assert result["active"] is True
            assert result["balance"] == 99.99
            # String should be sanitized
            assert result["name"] == "safe_Test"

    @patch("src.infrastructure.security.input_sanitizer.InputSanitizer")
    def test_sanitize_sql_params_sanitization_error(self, mock_sanitizer_class):
        """Test handling of sanitization errors."""
        from src.infrastructure.security.input_sanitizer import SanitizationError

        mock_sanitizer = mock_sanitizer_class.return_value
        mock_sanitizer.sanitize_string.side_effect = SanitizationError("SQL injection detected")

        params = {"malicious": "'; DROP TABLE users; --"}

        with pytest.raises(ValidationError) as exc_info:
            sanitize_sql_params(params)
        assert "Invalid parameter malicious" in str(exc_info)


class TestSecurityValidator:
    """Test SecurityValidator class."""

    @patch("src.domain.services.validation_service.ValidationService.validate_sql_injection")
    def test_check_sql_injection(self, mock_validate):
        """Test SQL injection checking."""
        mock_validate.return_value = True
        assert SecurityValidator.check_sql_injection("SELECT * FROM users") is True
        mock_validate.assert_called_once_with("SELECT * FROM users")

        mock_validate.return_value = False
        assert SecurityValidator.check_sql_injection("'; DROP TABLE--") is False

    @patch("src.domain.services.validation_service.ValidationService.validate_xss")
    def test_check_xss(self, mock_validate):
        """Test XSS checking."""
        mock_validate.return_value = True
        assert SecurityValidator.check_xss("<b>Bold text</b>") is True
        mock_validate.assert_called_once_with("<b>Bold text</b>")

        mock_validate.return_value = False
        assert SecurityValidator.check_xss("<script>alert('XSS')</script>") is False

    @patch(
        "src.domain.services.trading_validation_service.TradingValidationService.validate_trading_symbol"
    )
    def test_check_trading_symbol(self, mock_validate):
        """Test trading symbol validation."""
        mock_validate.return_value = True
        assert SecurityValidator.check_trading_symbol("AAPL") is True
        mock_validate.assert_called_once_with("AAPL")

    @patch(
        "src.domain.services.trading_validation_service.TradingValidationService.validate_currency_code"
    )
    def test_check_currency_code(self, mock_validate):
        """Test currency code validation."""
        mock_validate.return_value = True
        assert SecurityValidator.check_currency_code("USD") is True
        mock_validate.assert_called_once_with("USD")

    @patch("src.domain.services.trading_validation_service.TradingValidationService.validate_price")
    def test_check_price(self, mock_validate):
        """Test price validation."""
        mock_validate.return_value = True
        assert SecurityValidator.check_price("100.50") is True
        assert SecurityValidator.check_price(100.50) is True
        mock_validate.assert_called_with(100.50)

    @patch(
        "src.domain.services.trading_validation_service.TradingValidationService.validate_quantity"
    )
    def test_check_quantity(self, mock_validate):
        """Test quantity validation."""
        mock_validate.return_value = True
        assert SecurityValidator.check_quantity(100) is True
        assert SecurityValidator.check_quantity("100") is True
        mock_validate.assert_called_with("100")

    @patch("src.domain.services.validation_service.ValidationService.validate_ip_address")
    def test_check_ip_address(self, mock_validate):
        """Test IP address validation."""
        mock_validate.return_value = True
        assert SecurityValidator.check_ip_address("192.168.1.1") is True
        mock_validate.assert_called_once_with("192.168.1.1")

    @patch("src.domain.services.validation_service.ValidationService.validate_url")
    def test_check_url(self, mock_validate):
        """Test URL validation."""
        mock_validate.return_value = True
        assert SecurityValidator.check_url("https://example.com") is True
        mock_validate.assert_called_once_with("https://example.com")

    @patch("src.domain.services.validation_service.ValidationService.validate_email")
    def test_check_email(self, mock_validate):
        """Test email validation."""
        mock_validate.return_value = True
        assert SecurityValidator.check_email("user@example.com") is True
        mock_validate.assert_called_once_with("user@example.com")

    @patch("src.domain.services.validation_service.ValidationService.validate_json_structure")
    def test_check_json_structure(self, mock_validate):
        """Test JSON structure validation."""
        mock_validate.return_value = True
        json_str = '{"key": "value"}'
        assert SecurityValidator.check_json_structure(json_str, max_depth=5) is True
        mock_validate.assert_called_once_with(json_str, 5)


class TestSchemaValidator:
    """Test SchemaValidator class."""

    @patch("src.domain.services.validation_service.ValidationService.validate_schema")
    def test_check_schema(self, mock_validate):
        """Test schema validation."""
        data = {"name": "John", "age": 30}
        schema = {"name": str, "age": int}
        mock_validate.return_value = []

        errors = SchemaValidator.check_schema(data, schema)
        assert errors == []
        mock_validate.assert_called_once_with(data, schema)

        mock_validate.return_value = ["Invalid field 'email'"]
        errors = SchemaValidator.check_schema(data, schema)
        assert errors == ["Invalid field 'email'"]


class TestTradingInputValidator:
    """Test TradingInputValidator class."""

    @patch("src.domain.services.trading_validation_service.TradingValidationService.validate_order")
    def test_check_order(self, mock_validate):
        """Test order validation."""
        order_data = {"symbol": "AAPL", "quantity": 100, "side": "buy", "order_type": "market"}
        mock_validate.return_value = []

        errors = TradingInputValidator.check_order(order_data)
        assert errors == []
        mock_validate.assert_called_once_with(order_data)

    @patch(
        "src.domain.services.trading_validation_service.TradingValidationService.validate_portfolio_data"
    )
    def test_check_portfolio_data(self, mock_validate):
        """Test portfolio data validation."""
        portfolio_data = {"cash": 10000, "positions": []}
        mock_validate.return_value = []

        errors = TradingInputValidator.check_portfolio_data(portfolio_data)
        assert errors == []
        mock_validate.assert_called_once_with(portfolio_data)

    @patch(
        "src.domain.services.trading_validation_service.TradingValidationService.get_order_schema"
    )
    def test_get_order_schema(self, mock_get_schema):
        """Test getting order schema."""
        expected_schema = {
            "symbol": {"type": "string", "required": True},
            "quantity": {"type": "integer", "required": True},
        }
        mock_get_schema.return_value = expected_schema

        schema = TradingInputValidator.get_order_schema()
        assert schema == expected_schema
        mock_get_schema.assert_called_once()


class TestCheckAndSanitizeDecorator:
    """Test check_and_sanitize decorator."""

    def test_check_and_sanitize_basic(self):
        """Test basic check and sanitize functionality."""

        def uppercase_validator(value):
            return value.upper() if isinstance(value, str) else value

        @check_and_sanitize(text=uppercase_validator)
        def process(text: str) -> str:
            return f"Result: {text}"

        result = process("hello")
        assert result == "Result: HELLO"

    def test_check_and_sanitize_with_validation_error(self):
        """Test validation error propagation."""

        def strict_validator(value):
            if not value or len(value) < 5:
                raise ValidationError("Value too short")
            return value

        @check_and_sanitize(text=strict_validator)
        def process(text: str) -> str:
            return f"Result: {text}"

        with pytest.raises(ValidationError) as exc_info:
            process("hi")
        assert "Value too short" in str(exc_info)

    def test_check_and_sanitize_none_result(self):
        """Test validator returning None."""

        def nullable_validator(value):
            return None  # Validator returns None, original value should be preserved

        @check_and_sanitize(text=nullable_validator)
        def process(text: str) -> str:
            return f"Result: {text}"

        result = process("test")
        assert result == "Result: test"

    def test_check_and_sanitize_multiple_validators(self):
        """Test multiple validators on different parameters."""

        @check_and_sanitize(
            name=lambda x: x.strip().title(),
            age=lambda x: max(0, min(150, x)),  # Clamp age between 0-150
        )
        def create_user(name: str, age: int) -> dict[str, Any]:
            return {"name": name, "age": age}

        result = create_user("  john doe  ", 200)
        assert result == {"name": "John Doe", "age": 150}


class TestSecurityCheckFunction:
    """Test security_check validator factory."""

    @patch("src.domain.services.validation_service.ValidationService.validate_field")
    @patch("src.infrastructure.security.input_sanitizer.InputSanitizer.sanitize_string")
    def test_security_check_success(self, mock_sanitize, mock_validate):
        """Test successful security check."""
        mock_validate.return_value = True
        mock_sanitize.return_value = "sanitized_value"

        validator = security_check("username", "string")
        result = validator("test_user")

        assert result == "sanitized_value"
        mock_validate.assert_called_once_with("test_user", "string")
        mock_sanitize.assert_called_once_with("test_user")

    @patch("src.domain.services.validation_service.ValidationService.validate_field")
    def test_security_check_validation_failure(self, mock_validate):
        """Test validation failure."""
        mock_validate.return_value = False

        validator = security_check("email", "email")

        with pytest.raises(ValidationError) as exc_info:
            validator("invalid-email")
        assert "Validation failed for email" in str(exc_info)

    @patch("src.domain.services.validation_service.ValidationService.validate_field")
    @patch("src.infrastructure.security.input_sanitizer.InputSanitizer.sanitize_string")
    def test_security_check_sanitization_failure(self, mock_sanitize, mock_validate):
        """Test sanitization failure."""
        from src.infrastructure.security.input_sanitizer import SanitizationError

        mock_validate.return_value = True
        mock_sanitize.side_effect = SanitizationError("Dangerous input")

        validator = security_check("comment", "text")

        with pytest.raises(SecurityValidationError) as exc_info:
            validator("<script>alert('xss')</script>")
        assert "Sanitization failed for comment" in str(exc_info)

    def test_security_check_with_none(self):
        """Test security check with None value."""
        validator = security_check("optional_field", "string")
        result = validator(None)
        assert result is None


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple validation features."""

    @patch("src.domain.services.validation_service.ValidationService.validate_field")
    @patch("src.infrastructure.security.input_sanitizer.InputSanitizer.sanitize_string")
    def test_combined_decorators(self, mock_sanitize, mock_validate):
        """Test combining multiple decorators."""
        mock_validate.return_value = True
        mock_sanitize.return_value = "safe_input"

        @check_required(api_key=str)
        @sanitize_input(data=lambda x: mock_sanitize(x))
        def api_call(api_key: str, data: str) -> str:
            return f"API: {api_key}, Data: {data}"

        result = api_call("key123", "user_input")
        assert result == "API: key123, Data: safe_input"

    def test_sql_injection_prevention_flow(self):
        """Test complete SQL injection prevention flow."""
        # Simulate a malicious SQL injection attempt
        malicious_params = {"user_id": "1 OR 1=1", "table": "users; DROP TABLE accounts;--"}

        with patch("src.infrastructure.security.input_sanitizer.InputSanitizer") as mock_cls:
            from src.infrastructure.security.input_sanitizer import SanitizationError

            mock_sanitizer = mock_cls.return_value

            # First param passes, second fails
            mock_sanitizer.sanitize_string.side_effect = [
                "safe_1_OR_1_1",
                SanitizationError("SQL injection detected"),
            ]

            with pytest.raises(ValidationError) as exc_info:
                sanitize_sql_params(malicious_params)
            assert "Invalid parameter table" in str(exc_info)

    def test_xss_prevention_flow(self):
        """Test complete XSS prevention flow."""

        @sanitize_input(comment=lambda x: x.replace("<script>", "").replace("</script>", ""))
        def post_comment(comment: str) -> str:
            return f"Comment posted: {comment}"

        # XSS attempt should be sanitized
        result = post_comment("<script>alert('XSS')</script>Hello")
        assert result == "Comment posted: alert('XSS')Hello"

    @patch("src.domain.services.trading_validation_service.TradingValidationService")
    def test_trading_validation_flow(self, mock_service):
        """Test complete trading validation flow."""
        # Setup mock responses
        mock_service.validate_trading_symbol.return_value = True
        mock_service.validate_price.return_value = True
        mock_service.validate_quantity.return_value = True
        mock_service.validate_order.return_value = []

        # Create order data
        order = {"symbol": "AAPL", "price": 150.50, "quantity": 100, "side": "buy"}

        # Validate individual fields
        assert SecurityValidator.check_trading_symbol(order["symbol"])
        assert SecurityValidator.check_price(order["price"])
        assert SecurityValidator.check_quantity(order["quantity"])

        # Validate complete order
        errors = TradingInputValidator.check_order(order)
        assert errors == []


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_decorator_with_no_matching_params(self):
        """Test decorator when specified param doesn't exist."""

        @sanitize_input(nonexistent=lambda x: x.upper())
        def process(actual_param: str) -> str:
            return actual_param

        # Should work normally as decorator ignores non-existent params
        result = process("test")
        assert result == "test"

    def test_empty_sql_params(self):
        """Test sanitizing empty SQL parameters."""
        result = sanitize_sql_params({})
        assert result == {}

    @patch("src.infrastructure.security.input_sanitizer.InputSanitizer")
    def test_unexpected_sanitization_error(self, mock_sanitizer_class):
        """Test handling of unexpected errors during sanitization."""
        mock_sanitizer = mock_sanitizer_class.return_value
        mock_sanitizer.sanitize_string.side_effect = RuntimeError("Unexpected error")

        @sanitize_input(data=lambda x: mock_sanitizer.sanitize_string(x))
        def process(data: str) -> str:
            return data

        with pytest.raises(ValidationError) as exc_info:
            process("test")
        assert "Error sanitizing data" in str(exc_info)

    def test_validator_with_non_callable(self):
        """Test check_and_sanitize with non-callable validator."""

        @check_and_sanitize(value="not_a_function")
        def process(value: str) -> str:
            return value

        # Should skip non-callable validators
        result = process("test")
        assert result == "test"

    @patch("src.domain.services.validation_service.ValidationService.validate_field")
    def test_security_check_with_empty_string(self, mock_validate):
        """Test security check with empty string."""
        mock_validate.return_value = True

        validator = security_check("field", "string")

        with patch(
            "src.infrastructure.security.input_sanitizer.InputSanitizer.sanitize_string"
        ) as mock_san:
            mock_san.return_value = ""
            result = validator("")
            assert result == ""
            mock_validate.assert_called_once_with("", "string")
