"""
Extended unit tests for validation module to achieve 80%+ coverage.

Focuses on security validation, schema validation, and decorator functionality.
"""

from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest

from src.infrastructure.security.input_sanitizer import SanitizationError
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
    """Test custom validation error types"""

    def test_validation_error_basic(self):
        """Test basic ValidationError"""
        error = ValidationError("Invalid input")
        assert str(error) == "Invalid input"
        assert error.field is None
        assert error is None

    def test_validation_error_with_field(self):
        """Test ValidationError with field information"""
        error = ValidationError("Invalid email", field="email", value="bad@")
        assert str(error) == "Invalid email"
        assert error.field == "email"
        assert error == "bad@"

    def test_schema_validation_error(self):
        """Test SchemaValidationError"""
        schema_errors = ["Field 'age' must be integer", "Field 'email' is required"]
        error = SchemaValidationError("Schema validation failed", schema_errors)
        assert str(error) == "Schema validation failed"
        assert error.schema_errors == schema_errors
        assert error.field is None

    def test_security_validation_error(self):
        """Test SecurityValidationError"""
        error = SecurityValidationError("SQL injection detected", field="query")
        assert str(error) == "SQL injection detected"
        assert error.field == "query"
        assert isinstance(error, ValidationError)


class TestSanitizeInputDecorator:
    """Test the sanitize_input decorator"""

    def test_sanitize_input_basic(self):
        """Test basic input sanitization"""

        @sanitize_input(name=lambda x: x.strip().upper())
        def process_name(name: str) -> str:
            return f"Hello, {name}"

        result = process_name("  john  ")
        assert result == "Hello, JOHN"

    def test_sanitize_input_multiple_params(self):
        """Test sanitizing multiple parameters"""

        @sanitize_input(
            first_name=lambda x: x.strip().title(), last_name=lambda x: x.strip().upper()
        )
        def create_user(first_name: str, last_name: str) -> str:
            return f"{first_name} {last_name}"

        result = create_user("  john  ", "  doe  ")
        assert result == "John DOE"

    def test_sanitize_input_with_kwargs(self):
        """Test sanitizing with keyword arguments"""

        @sanitize_input(email=lambda x: x.strip().lower())
        def register(username: str, email: str) -> tuple:
            return (username, email)

        result = register("User123", email="  USER@EXAMPLE.COM  ")
        assert result == ("User123", "user@example.com")

    def test_sanitize_input_sanitization_error(self):
        """Test handling of SanitizationError"""
        from src.infrastructure.security.input_sanitizer import InputSanitizer

        @sanitize_input(query=lambda x: InputSanitizer.sanitize_string(x))
        def search(query: str) -> str:
            return f"Searching for: {query}"

        with patch("src.infrastructure.security.validation.logger") as mock_logger:
            with pytest.raises(ValidationError) as exc_info:
                search("SELECT * FROM users")

            assert "Invalid query" in str(exc_info)
            mock_logger.warning.assert_called_once()

    def test_sanitize_input_general_exception(self):
        """Test handling of general exceptions"""

        @sanitize_input(value=lambda x: x.undefined_method())  # Will raise AttributeError
        def process(value: Any) -> Any:
            return value

        with patch("src.infrastructure.security.validation.logger") as mock_logger:
            with pytest.raises(ValidationError) as exc_info:
                process("test")

            assert "Error sanitizing value" in str(exc_info)
            mock_logger.error.assert_called_once()

    def test_sanitize_input_missing_param(self):
        """Test sanitizer for non-existent parameter"""

        @sanitize_input(nonexistent=lambda x: x.upper())
        def func(value: str) -> str:
            return value

        # Should work fine - sanitizer ignored for missing param
        result = func("test")
        assert result == "test"

    def test_sanitize_input_with_defaults(self):
        """Test sanitizing with default parameter values"""

        @sanitize_input(name=lambda x: x.upper() if x else "ANONYMOUS")
        def greet(name: str = "guest") -> str:
            return f"Hello, {name}"

        result = greet()
        assert result == "Hello, GUEST"

        result = greet("john")
        assert result == "Hello, JOHN"


class TestCheckRequiredDecorator:
    """Test the check_required decorator"""

    def test_check_required_valid_types(self):
        """Test type checking with valid types"""

        @check_required(api_key=str, port=int)
        def connect(api_key: str, port: int) -> str:
            return f"Connected to port {port}"

        result = connect("key123", 8080)
        assert result == "Connected to port 8080"

    def test_check_required_invalid_type(self):
        """Test type checking with invalid types"""

        @check_required(api_key=str, port=int)
        def connect(api_key: str, port: int) -> str:
            return "Connected"

        with pytest.raises(ValidationError) as exc_info:
            connect("key123", "8080")  # String instead of int

        assert "Parameter 'port' must be of type int" in str(exc_info)

    def test_check_required_none_value(self):
        """Test that None values are allowed"""

        @check_required(value=str)
        def process(value: str = None) -> str:
            return str(value)

        # None should pass - only non-None values are type-checked
        result = process(None)
        assert result == "None"

    def test_check_required_with_kwargs(self):
        """Test with keyword arguments"""

        @check_required(host=str, port=int, secure=bool)
        def connect(host: str, port: int = 80, secure: bool = False) -> tuple:
            return (host, port, secure)

        result = connect("example.com", secure=True)
        assert result == ("example.com", 80, True)

    def test_check_required_missing_param(self):
        """Test checking non-existent parameter"""

        @check_required(nonexistent=str)
        def func(value: int) -> int:
            return value * 2

        # Should work fine - check ignored for missing param
        result = func(5)
        assert result == 10


class TestSanitizeSQLParams:
    """Test SQL parameter sanitization"""

    def test_sanitize_sql_params_strings(self):
        """Test sanitizing string parameters"""
        params = {"username": "john_doe", "email": "user@example.com", "comment": "Hello world"}

        with patch("src.infrastructure.security.input_sanitizer.InputSanitizer") as mock_sanitizer:
            mock_sanitizer.return_value.sanitize_string.side_effect = lambda x: x.upper()

            result = sanitize_sql_params(params)

            assert result == {
                "username": "JOHN_DOE",
                "email": "USER@EXAMPLE.COM",
                "comment": "HELLO WORLD",
            }

    def test_sanitize_sql_params_mixed_types(self):
        """Test sanitizing mixed type parameters"""
        params = {"name": "test", "age": 25, "balance": 100.50, "active": True, "created": None}

        with patch("src.infrastructure.security.input_sanitizer.InputSanitizer") as mock_sanitizer:
            mock_sanitizer.return_value.sanitize_string.side_effect = lambda x: f"sanitized_{x}"

            result = sanitize_sql_params(params)

            assert result == {
                "name": "sanitized_test",
                "age": 25,  # Non-string unchanged
                "balance": 100.50,
                "active": True,
                "created": None,
            }

    def test_sanitize_sql_params_sanitization_error(self):
        """Test handling of sanitization errors"""
        params = {"query": "SELECT * FROM users"}

        with patch("src.infrastructure.security.input_sanitizer.InputSanitizer") as mock_sanitizer:
            mock_sanitizer.return_value.sanitize_string.side_effect = SanitizationError(
                "SQL injection"
            )

            with patch("src.infrastructure.security.validation.logger") as mock_logger:
                with pytest.raises(ValidationError) as exc_info:
                    sanitize_sql_params(params)

                assert "Invalid parameter query" in str(exc_info)
                mock_logger.warning.assert_called_once()

    def test_sanitize_sql_params_empty(self):
        """Test sanitizing empty parameters"""
        assert sanitize_sql_params({}) == {}


class TestSecurityValidator:
    """Test SecurityValidator class"""

    @patch("src.domain.services.validation_service.ValidationService")
    def test_check_sql_injection(self, mock_service):
        """Test SQL injection checking"""
        mock_service.validate_sql_injection.return_value = False

        result = SecurityValidator.check_sql_injection("SELECT * FROM users")

        assert result is False
        mock_service.validate_sql_injection.assert_called_once_with("SELECT * FROM users")

    @patch("src.domain.services.validation_service.ValidationService")
    def test_check_xss(self, mock_service):
        """Test XSS checking"""
        mock_service.validate_xss.return_value = True

        result = SecurityValidator.check_xss("<script>alert(1)</script>")

        assert result is True
        mock_service.validate_xss.assert_called_once_with("<script>alert(1)</script>")

    @patch("src.domain.services.trading_validation_service.TradingValidationService")
    def test_check_trading_symbol(self, mock_service):
        """Test trading symbol validation"""
        mock_service.validate_trading_symbol.return_value = True

        result = SecurityValidator.check_trading_symbol("AAPL")

        assert result is True
        mock_service.validate_trading_symbol.assert_called_once_with("AAPL")

    @patch("src.domain.services.trading_validation_service.TradingValidationService")
    def test_check_currency_code(self, mock_service):
        """Test currency code validation"""
        mock_service.validate_currency_code.return_value = True

        result = SecurityValidator.check_currency_code("USD")

        assert result is True
        mock_service.validate_currency_code.assert_called_once_with("USD")

    @patch("src.domain.services.trading_validation_service.TradingValidationService")
    def test_check_price(self, mock_service):
        """Test price validation"""
        mock_service.validate_price.return_value = True

        # Test with different types
        result = SecurityValidator.check_price("100.50")
        assert result is True

        result = SecurityValidator.check_price(100.50)
        assert result is True

        result = SecurityValidator.check_price(Decimal("100.50"))
        assert result is True

    @patch("src.domain.services.trading_validation_service.TradingValidationService")
    def test_check_quantity(self, mock_service):
        """Test quantity validation"""
        mock_service.validate_quantity.return_value = True

        result = SecurityValidator.check_quantity(100)
        assert result is True

        result = SecurityValidator.check_quantity("100")
        assert result is True

        result = SecurityValidator.check_quantity(100.5)
        assert result is True

    @patch("src.domain.services.validation_service.ValidationService")
    def test_check_ip_address(self, mock_service):
        """Test IP address validation"""
        mock_service.validate_ip_address.return_value = True

        result = SecurityValidator.check_ip_address("192.168.1.1")

        assert result is True
        mock_service.validate_ip_address.assert_called_once_with("192.168.1.1")

    @patch("src.domain.services.validation_service.ValidationService")
    def test_check_url(self, mock_service):
        """Test URL validation"""
        mock_service.validate_url.return_value = True

        result = SecurityValidator.check_url("https://example.com")

        assert result is True
        mock_service.validate_url.assert_called_once_with("https://example.com")

    @patch("src.domain.services.validation_service.ValidationService")
    def test_check_email(self, mock_service):
        """Test email validation"""
        mock_service.validate_email.return_value = True

        result = SecurityValidator.check_email("user@example.com")

        assert result is True
        mock_service.validate_email.assert_called_once_with("user@example.com")

    @patch("src.domain.services.validation_service.ValidationService")
    def test_check_json_structure(self, mock_service):
        """Test JSON structure validation"""
        mock_service.validate_json_structure.return_value = True

        json_str = '{"key": "value"}'
        result = SecurityValidator.check_json_structure(json_str, max_depth=5)

        assert result is True
        mock_service.validate_json_structure.assert_called_once_with(json_str, 5)


class TestSchemaValidator:
    """Test SchemaValidator class"""

    @patch("src.domain.services.validation_service.ValidationService")
    def test_check_schema(self, mock_service):
        """Test schema validation"""
        data = {"name": "John", "age": 30}
        schema = {"name": str, "age": int}
        mock_service.validate_schema.return_value = []

        errors = SchemaValidator.check_schema(data, schema)

        assert errors == []
        mock_service.validate_schema.assert_called_once_with(data, schema)

    @patch("src.domain.services.validation_service.ValidationService")
    def test_check_schema_with_errors(self, mock_service):
        """Test schema validation with errors"""
        data = {"name": "John"}
        schema = {"name": str, "age": int, "email": str}
        mock_service.validate_schema.return_value = [
            "Missing required field: age",
            "Missing required field: email",
        ]

        errors = SchemaValidator.check_schema(data, schema)

        assert len(errors) == 2
        assert "Missing required field: age" in errors


class TestTradingInputValidator:
    """Test TradingInputValidator class"""

    @patch("src.domain.services.trading_validation_service.TradingValidationService")
    def test_check_order(self, mock_service):
        """Test order validation"""
        order_data = {"symbol": "AAPL", "quantity": 100, "side": "buy", "order_type": "market"}
        mock_service.validate_order.return_value = []

        errors = TradingInputValidator.check_order(order_data)

        assert errors == []
        mock_service.validate_order.assert_called_once_with(order_data)

    @patch("src.domain.services.trading_validation_service.TradingValidationService")
    def test_check_order_with_errors(self, mock_service):
        """Test order validation with errors"""
        order_data = {"symbol": "INVALID"}
        mock_service.validate_order.return_value = [
            "Invalid symbol format",
            "Missing required field: quantity",
        ]

        errors = TradingInputValidator.check_order(order_data)

        assert len(errors) == 2
        assert "Invalid symbol format" in errors

    @patch("src.domain.services.trading_validation_service.TradingValidationService")
    def test_check_portfolio_data(self, mock_service):
        """Test portfolio data validation"""
        portfolio_data = {"cash": 10000.00, "positions": []}
        mock_service.validate_portfolio_data.return_value = []

        errors = TradingInputValidator.check_portfolio_data(portfolio_data)

        assert errors == []
        mock_service.validate_portfolio_data.assert_called_once_with(portfolio_data)

    @patch("src.domain.services.trading_validation_service.TradingValidationService")
    def test_get_order_schema(self, mock_service):
        """Test getting order schema"""
        expected_schema = {
            "symbol": {"type": "string", "required": True},
            "quantity": {"type": "integer", "required": True},
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "order_type": {"type": "string", "enum": ["market", "limit"]},
        }
        mock_service.get_order_schema.return_value = expected_schema

        schema = TradingInputValidator.get_order_schema()

        assert schema == expected_schema
        mock_service.get_order_schema.assert_called_once()


class TestCheckAndSanitizeDecorator:
    """Test the check_and_sanitize decorator"""

    def test_check_and_sanitize_basic(self):
        """Test basic validation and sanitization"""

        @check_and_sanitize(name=lambda x: x.strip().upper() if x else None)
        def process(name: str) -> str:
            return f"Processing: {name}"

        result = process("  john  ")
        assert result == "Processing: JOHN"

    def test_check_and_sanitize_none_result(self):
        """Test when validator returns None"""

        @check_and_sanitize(value=lambda x: None)  # Validator returns None
        def process(value: str) -> str:
            return f"Value: {value}"

        result = process("test")
        assert result == "Value: test"  # Original value used

    def test_check_and_sanitize_validation_error(self):
        """Test when validator raises ValidationError"""

        def validate_positive(x):
            if x <= 0:
                raise ValidationError("Must be positive")
            return x

        @check_and_sanitize(value=validate_positive)
        def process(value: int) -> int:
            return value * 2

        with pytest.raises(ValidationError) as exc_info:
            process(-5)

        assert "Must be positive" in str(exc_info)

    def test_check_and_sanitize_non_callable(self):
        """Test with non-callable validator (should be skipped)"""

        @check_and_sanitize(value="not_callable")  # Not a function
        def process(value: str) -> str:
            return value

        result = process("test")
        assert result == "test"


class TestSecurityCheckFunction:
    """Test the security_check validator factory"""

    def test_security_check_valid(self):
        """Test security check with valid input"""
        with patch("src.domain.services.validation_service.ValidationService") as mock_service:
            mock_service.validate_field.return_value = True

            with patch(
                "src.infrastructure.security.input_sanitizer.InputSanitizer"
            ) as mock_sanitizer:
                mock_sanitizer.sanitize_string.return_value = "sanitized_value"

                validator = security_check("username", "string")
                result = validator("test_user")

                assert result == "sanitized_value"
                mock_service.validate_field.assert_called_once_with("test_user", "string")
                mock_sanitizer.sanitize_string.assert_called_once_with("test_user")

    def test_security_check_none_value(self):
        """Test security check with None value"""
        validator = security_check("optional_field", "string")
        result = validator(None)
        assert result is None

    def test_security_check_validation_failure(self):
        """Test security check with validation failure"""
        with patch("src.domain.services.validation_service.ValidationService") as mock_service:
            mock_service.validate_field.return_value = False

            validator = security_check("field", "email")

            with pytest.raises(ValidationError) as exc_info:
                validator("invalid_email")

            assert "Validation failed for field of type email" in str(exc_info)

    def test_security_check_sanitization_error(self):
        """Test security check with sanitization error"""
        with patch("src.domain.services.validation_service.ValidationService") as mock_service:
            mock_service.validate_field.return_value = True

            with patch(
                "src.infrastructure.security.input_sanitizer.InputSanitizer"
            ) as mock_sanitizer:
                mock_sanitizer.sanitize_string.side_effect = SanitizationError("Dangerous input")

                validator = security_check("query", "string")

                with pytest.raises(SecurityValidationError) as exc_info:
                    validator("SELECT * FROM users")

                assert "Sanitization failed for query" in str(exc_info)

    def test_security_check_different_types(self):
        """Test security check with different validation types"""
        with patch("src.domain.services.validation_service.ValidationService") as mock_service:
            mock_service.validate_field.return_value = True

            with patch(
                "src.infrastructure.security.input_sanitizer.InputSanitizer"
            ) as mock_sanitizer:
                mock_sanitizer.sanitize_string.side_effect = lambda x: f"clean_{x}"

                # Test different validation types
                for validation_type in ["string", "email", "url", "ip"]:
                    validator = security_check("field", validation_type)
                    result = validator("test")
                    assert result == "clean_test"


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple decorators"""

    def test_combined_decorators(self):
        """Test combining multiple validation decorators"""

        @check_required(name=str, age=int)
        @sanitize_input(name=lambda x: x.strip().upper())
        def create_user(name: str, age: int) -> dict:
            return {"name": name, "age": age}

        result = create_user("  john  ", 30)
        assert result == {"name": "JOHN", "age": 30}

        with pytest.raises(ValidationError):
            create_user("john", "thirty")  # Invalid age type

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention in practice"""

        @sanitize_input(query=lambda x: x if not SecurityValidator.check_sql_injection(x) else None)
        def search(query: str) -> str:
            return f"Searching: {query}"

        with patch("src.domain.services.validation_service.ValidationService") as mock_service:
            mock_service.validate_sql_injection.return_value = False  # Contains SQL injection

            # Should set query to None when SQL injection detected
            result = search("SELECT * FROM users")
            assert result == "Searching: None"
