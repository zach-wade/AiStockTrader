"""
Comprehensive unit tests for ValidationService.

This test module provides comprehensive coverage for the DomainValidator,
OrderValidator, and DatabaseIdentifierValidator classes.
"""

import uuid
from decimal import Decimal
from uuid import UUID

import pytest

from src.domain.services.validation_service import (
    DatabaseIdentifierValidator,
    DomainValidator,
    OrderValidator,
    ValidationError,
)


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_is_exception(self):
        """Test that ValidationError is an Exception."""
        assert issubclass(ValidationError, Exception)

    def test_validation_error_message(self):
        """Test ValidationError with message."""
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"


class TestDomainValidatorSymbol:
    """Test symbol validation."""

    def test_validate_symbol_valid(self):
        """Test validation of valid symbols."""
        valid_symbols = ["AAPL", "GOOGL", "MSFT", "BRK.B", "BRK-A", "SPY", "A", "Z123"]

        for symbol in valid_symbols:
            result = DomainValidator.validate_symbol(symbol)
            assert result == symbol.upper()

    def test_validate_symbol_lowercase_normalized(self):
        """Test that lowercase symbols are normalized to uppercase."""
        assert DomainValidator.validate_symbol("aapl") == "AAPL"
        assert DomainValidator.validate_symbol("gOoGl") == "GOOGL"

    def test_validate_symbol_whitespace_stripped(self):
        """Test that whitespace is stripped from symbols."""
        assert DomainValidator.validate_symbol("  AAPL  ") == "AAPL"
        assert DomainValidator.validate_symbol("\tMSFT\n") == "MSFT"

    def test_validate_symbol_empty(self):
        """Test that empty symbol raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_symbol("")
        assert "Symbol cannot be empty" in str(exc_info)

        with pytest.raises(ValidationError):
            DomainValidator.validate_symbol("   ")

    def test_validate_symbol_too_short(self):
        """Test that symbols shorter than minimum length raise error."""
        # Since MIN_SYMBOL_LENGTH is 1, empty string is the only "too short" case
        # This is already covered by empty test above
        pass

    def test_validate_symbol_too_long(self):
        """Test that symbols longer than maximum length raise error."""
        long_symbol = "A" * (DomainValidator.MAX_SYMBOL_LENGTH + 1)

        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_symbol(long_symbol)
        assert "Symbol too long" in str(exc_info)

    def test_validate_symbol_invalid_characters(self):
        """Test that symbols with invalid characters raise error."""
        invalid_symbols = ["AA@PL", "GOO GL", "MS$FT", "BRK_B", "SPY!", "(AAPL)"]

        for symbol in invalid_symbols:
            with pytest.raises(ValidationError) as exc_info:
                DomainValidator.validate_symbol(symbol)
            assert "Invalid symbol format" in str(exc_info)

    def test_validate_symbol_special_valid_characters(self):
        """Test that allowed special characters work."""
        # Dots and hyphens are allowed
        assert DomainValidator.validate_symbol("BRK.B") == "BRK.B"
        assert DomainValidator.validate_symbol("BRK-A") == "BRK-A"
        assert DomainValidator.validate_symbol("A.B-C") == "A.B-C"


class TestDomainValidatorPrice:
    """Test price validation."""

    def test_validate_price_valid(self):
        """Test validation of valid prices."""
        valid_prices = [
            (100, Decimal("100")),
            ("50.25", Decimal("50.25")),
            (Decimal("999.99"), Decimal("999.99")),
            (0.01, Decimal("0.01")),
            ("0.01", Decimal("0.01")),
        ]

        for input_price, expected in valid_prices:
            result = DomainValidator.validate_price(input_price)
            assert result == expected

    def test_validate_price_none(self):
        """Test that None price raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_price(None)
        assert "Price cannot be None" in str(exc_info)

    def test_validate_price_invalid_format(self):
        """Test that invalid price format raises error."""
        invalid_prices = ["invalid", "abc", "", [100], {"price": 100}]

        for price in invalid_prices:
            with pytest.raises(ValidationError) as exc_info:
                DomainValidator.validate_price(price)
            assert "Invalid price format" in str(exc_info)

    def test_validate_price_too_low(self):
        """Test that prices below minimum raise error."""
        low_prices = [0, "0", "-1", "-100", "0.001", "0.009"]

        for price in low_prices:
            with pytest.raises(ValidationError) as exc_info:
                DomainValidator.validate_price(price)
            assert "Price too low" in str(exc_info)

    def test_validate_price_too_high(self):
        """Test that prices above maximum raise error."""
        high_price = str(DomainValidator.MAX_PRICE + Decimal("0.01"))

        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_price(high_price)
        assert "Price too high" in str(exc_info)

    def test_validate_price_boundary_values(self):
        """Test price validation at boundaries."""
        # Minimum valid price
        assert (
            DomainValidator.validate_price(DomainValidator.MIN_PRICE) == DomainValidator.MIN_PRICE
        )

        # Maximum valid price
        assert (
            DomainValidator.validate_price(DomainValidator.MAX_PRICE) == DomainValidator.MAX_PRICE
        )

    def test_validate_price_scientific_notation(self):
        """Test price validation with scientific notation."""
        result = DomainValidator.validate_price("1e2")
        assert result == Decimal("100")

        result = DomainValidator.validate_price("1.5e1")
        assert result == Decimal("15")


class TestDomainValidatorQuantity:
    """Test quantity validation."""

    def test_validate_quantity_valid(self):
        """Test validation of valid quantities."""
        valid_quantities = [
            (100, Decimal("100")),
            ("50.5", Decimal("50.5")),
            (Decimal("1000"), Decimal("1000")),
            (0.01, Decimal("0.01")),
            ("0.01", Decimal("0.01")),
        ]

        for input_qty, expected in valid_quantities:
            result = DomainValidator.validate_quantity(input_qty)
            assert result == expected

    def test_validate_quantity_none(self):
        """Test that None quantity raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_quantity(None)
        assert "Quantity cannot be None" in str(exc_info)

    def test_validate_quantity_invalid_format(self):
        """Test that invalid quantity format raises error."""
        invalid_quantities = ["invalid", "abc", "", [100], {"qty": 100}]

        for qty in invalid_quantities:
            with pytest.raises(ValidationError) as exc_info:
                DomainValidator.validate_quantity(qty)
            assert "Invalid quantity format" in str(exc_info)

    def test_validate_quantity_too_low(self):
        """Test that quantities below minimum raise error."""
        low_quantities = [0, "0", "-1", "-100", "0.001", "0.009"]

        for qty in low_quantities:
            with pytest.raises(ValidationError) as exc_info:
                DomainValidator.validate_quantity(qty)
            assert "Quantity too low" in str(exc_info)

    def test_validate_quantity_too_high(self):
        """Test that quantities above maximum raise error."""
        high_qty = str(DomainValidator.MAX_QUANTITY + Decimal("1"))

        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_quantity(high_qty)
        assert "Quantity too high" in str(exc_info)

    def test_validate_quantity_boundary_values(self):
        """Test quantity validation at boundaries."""
        # Minimum valid quantity
        assert (
            DomainValidator.validate_quantity(DomainValidator.MIN_QUANTITY)
            == DomainValidator.MIN_QUANTITY
        )

        # Maximum valid quantity
        assert (
            DomainValidator.validate_quantity(DomainValidator.MAX_QUANTITY)
            == DomainValidator.MAX_QUANTITY
        )


class TestDomainValidatorDecimal:
    """Test decimal validation with bounds."""

    def test_validate_decimal_valid(self):
        """Test validation of valid decimal values."""
        result = DomainValidator.validate_decimal(100)
        assert result == Decimal("100")

        result = DomainValidator.validate_decimal("50.25")
        assert result == Decimal("50.25")

        result = DomainValidator.validate_decimal(Decimal("999.99"))
        assert result == Decimal("999.99")

    def test_validate_decimal_none(self):
        """Test that None value raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_decimal(None)
        assert "Value cannot be None" in str(exc_info)

    def test_validate_decimal_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_decimal("invalid")
        assert "Invalid decimal format" in str(exc_info)

    def test_validate_decimal_with_min_bound(self):
        """Test validation with minimum bound."""
        # Valid: at or above minimum
        result = DomainValidator.validate_decimal(10, min_value=Decimal("10"))
        assert result == Decimal("10")

        result = DomainValidator.validate_decimal(11, min_value=Decimal("10"))
        assert result == Decimal("11")

        # Invalid: below minimum
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_decimal(9, min_value=Decimal("10"))
        assert "Value too low: 9 < 10" in str(exc_info)

    def test_validate_decimal_with_max_bound(self):
        """Test validation with maximum bound."""
        # Valid: at or below maximum
        result = DomainValidator.validate_decimal(100, max_value=Decimal("100"))
        assert result == Decimal("100")

        result = DomainValidator.validate_decimal(99, max_value=Decimal("100"))
        assert result == Decimal("99")

        # Invalid: above maximum
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_decimal(101, max_value=Decimal("100"))
        assert "Value too high: 101 > 100" in str(exc_info)

    def test_validate_decimal_with_both_bounds(self):
        """Test validation with both min and max bounds."""
        # Valid: within bounds
        result = DomainValidator.validate_decimal(
            50, min_value=Decimal("10"), max_value=Decimal("100")
        )
        assert result == Decimal("50")

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            DomainValidator.validate_decimal(5, min_value=Decimal("10"), max_value=Decimal("100"))

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            DomainValidator.validate_decimal(150, min_value=Decimal("10"), max_value=Decimal("100"))

    def test_validate_decimal_no_bounds(self):
        """Test validation without bounds."""
        # Should accept any valid decimal
        result = DomainValidator.validate_decimal("-999999")
        assert result == Decimal("-999999")

        result = DomainValidator.validate_decimal("999999")
        assert result == Decimal("999999")


class TestDomainValidatorUUID:
    """Test UUID validation."""

    def test_validate_uuid_valid_uuid_object(self):
        """Test validation with UUID object."""
        test_uuid = uuid.uuid4()
        result = DomainValidator.validate_uuid(test_uuid)
        assert result == test_uuid
        assert isinstance(result, UUID)

    def test_validate_uuid_valid_string(self):
        """Test validation with UUID string."""
        test_uuid = str(uuid.uuid4())
        result = DomainValidator.validate_uuid(test_uuid)
        assert str(result) == test_uuid
        assert isinstance(result, UUID)

    def test_validate_uuid_none(self):
        """Test that None UUID raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_uuid(None)
        assert "UUID cannot be None" in str(exc_info)

    def test_validate_uuid_invalid_format(self):
        """Test that invalid UUID format raises error."""
        invalid_uuids = [
            "invalid",
            "12345",
            "not-a-uuid",
            "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            123456,
            [],
        ]

        for invalid in invalid_uuids:
            with pytest.raises(ValidationError) as exc_info:
                DomainValidator.validate_uuid(invalid)
            assert "Invalid UUID format" in str(exc_info)

    def test_validate_uuid_various_formats(self):
        """Test validation with various UUID string formats."""
        base_uuid = "550e8400-e29b-41d4-a716-446655440000"

        # With hyphens
        result = DomainValidator.validate_uuid(base_uuid)
        assert isinstance(result, UUID)

        # Without hyphens
        no_hyphens = base_uuid.replace("-", "")
        result = DomainValidator.validate_uuid(no_hyphens)
        assert isinstance(result, UUID)

        # With braces
        with_braces = "{" + base_uuid + "}"
        result = DomainValidator.validate_uuid(with_braces)
        assert isinstance(result, UUID)


class TestDomainValidatorEmail:
    """Test email validation."""

    def test_validate_email_valid(self):
        """Test validation of valid email addresses."""
        valid_emails = [
            "user@example.com",
            "john.doe@company.co.uk",
            "test123@test-domain.org",
            "user+tag@example.com",
            "admin@localhost.localdomain",
        ]

        for email in valid_emails:
            result = DomainValidator.validate_email(email)
            assert result == email.lower()

    def test_validate_email_normalized_lowercase(self):
        """Test that emails are normalized to lowercase."""
        assert DomainValidator.validate_email("USER@EXAMPLE.COM") == "user@example.com"
        assert DomainValidator.validate_email("JoHn.DoE@CoMpAnY.cOm") == "john.doe@company.com"

    def test_validate_email_whitespace_stripped(self):
        """Test that whitespace is stripped from emails."""
        assert DomainValidator.validate_email("  user@example.com  ") == "user@example.com"
        assert DomainValidator.validate_email("\tuser@example.com\n") == "user@example.com"

    def test_validate_email_empty(self):
        """Test that empty email raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_email("")
        assert "Email cannot be empty" in str(exc_info)

        with pytest.raises(ValidationError):
            DomainValidator.validate_email("   ")

    def test_validate_email_invalid_format(self):
        """Test that invalid email formats raise error."""
        invalid_emails = [
            "notanemail",
            "@example.com",
            "user@",
            "user",
            "user@.com",
            "user@example",
            "user @example.com",
            "user@exam ple.com",
            "user@@example.com",
            "user.example.com",
        ]

        for email in invalid_emails:
            with pytest.raises(ValidationError) as exc_info:
                DomainValidator.validate_email(email)
            assert "Invalid email format" in str(exc_info)


class TestDomainValidatorPercentage:
    """Test percentage validation."""

    def test_validate_percentage_valid(self):
        """Test validation of valid percentages."""
        valid_percentages = [
            (0, Decimal("0")),
            (50, Decimal("50")),
            (100, Decimal("100")),
            ("25.5", Decimal("25.5")),
            (Decimal("99.99"), Decimal("99.99")),
        ]

        for input_pct, expected in valid_percentages:
            result = DomainValidator.validate_percentage(input_pct)
            assert result == expected

    def test_validate_percentage_boundaries(self):
        """Test percentage validation at boundaries."""
        # Minimum (0%)
        assert DomainValidator.validate_percentage(0) == Decimal("0")

        # Maximum (100%)
        assert DomainValidator.validate_percentage(100) == Decimal("100")

    def test_validate_percentage_below_zero(self):
        """Test that negative percentages raise error."""
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_percentage(-1)
        assert "Value too low" in str(exc_info)

    def test_validate_percentage_above_hundred(self):
        """Test that percentages above 100 raise error."""
        with pytest.raises(ValidationError) as exc_info:
            DomainValidator.validate_percentage(101)
        assert "Value too high" in str(exc_info)

    def test_validate_percentage_none(self):
        """Test that None percentage raises error."""
        with pytest.raises(ValidationError):
            DomainValidator.validate_percentage(None)


class TestOrderValidator:
    """Test OrderValidator class."""

    def test_order_validator_initialization(self):
        """Test OrderValidator initialization."""
        # Default initialization
        validator = OrderValidator()
        assert validator.domain_validator is not None

        # With custom domain validator
        custom_validator = DomainValidator()
        validator = OrderValidator(custom_validator)
        assert validator.domain_validator is custom_validator

    def test_validate_order_params_valid(self):
        """Test validation of valid order parameters."""
        validator = OrderValidator()

        symbol, quantity, price = validator.validate_order_params("AAPL", 100, 150.50)

        assert symbol == "AAPL"
        assert quantity == Decimal("100")
        assert price == Decimal("150.50")

    def test_validate_order_params_no_price(self):
        """Test validation with no price (market order)."""
        validator = OrderValidator()

        symbol, quantity, price = validator.validate_order_params("GOOGL", 50, None)

        assert symbol == "GOOGL"
        assert quantity == Decimal("50")
        assert price is None

    def test_validate_order_params_invalid_symbol(self):
        """Test validation with invalid symbol."""
        validator = OrderValidator()

        with pytest.raises(ValidationError):
            validator.validate_order_params("", 100, 150)

        with pytest.raises(ValidationError):
            validator.validate_order_params("INVALID@", 100, 150)

    def test_validate_order_params_invalid_quantity(self):
        """Test validation with invalid quantity."""
        validator = OrderValidator()

        with pytest.raises(ValidationError):
            validator.validate_order_params("AAPL", 0, 150)

        with pytest.raises(ValidationError):
            validator.validate_order_params("AAPL", -10, 150)

    def test_validate_order_params_invalid_price(self):
        """Test validation with invalid price."""
        validator = OrderValidator()

        with pytest.raises(ValidationError):
            validator.validate_order_params("AAPL", 100, 0)

        with pytest.raises(ValidationError):
            validator.validate_order_params("AAPL", 100, -50)

    def test_validate_order_params_normalization(self):
        """Test that order parameters are normalized."""
        validator = OrderValidator()

        symbol, quantity, price = validator.validate_order_params("aapl", "100", "150.50")

        assert symbol == "AAPL"  # Uppercase
        assert quantity == Decimal("100")  # Decimal
        assert price == Decimal("150.50")  # Decimal


class TestDatabaseIdentifierValidator:
    """Test DatabaseIdentifierValidator class."""

    def test_validate_identifier_valid(self):
        """Test validation of valid identifiers."""
        valid_identifiers = [
            "table_name",
            "column_name",
            "_private_table",
            "table123",
            "CamelCase",
            "UPPERCASE",
            "a",  # Single character
            "_",  # Just underscore
        ]

        for identifier in valid_identifiers:
            result = DatabaseIdentifierValidator.validate_identifier(identifier)
            assert result == identifier

    def test_validate_identifier_empty(self):
        """Test that empty identifier raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseIdentifierValidator.validate_identifier("")
        assert "identifier cannot be empty" in str(exc_info)

    def test_validate_identifier_too_long(self):
        """Test that identifier exceeding max length raises error."""
        long_identifier = "a" * (DatabaseIdentifierValidator.MAX_IDENTIFIER_LENGTH + 1)

        with pytest.raises(ValidationError) as exc_info:
            DatabaseIdentifierValidator.validate_identifier(long_identifier)
        assert "identifier too long" in str(exc_info)

    def test_validate_identifier_invalid_start(self):
        """Test that identifiers starting with invalid characters raise error."""
        invalid_starts = ["1table", "123", "-table", ".column", " table"]

        for identifier in invalid_starts:
            with pytest.raises(ValidationError) as exc_info:
                DatabaseIdentifierValidator.validate_identifier(identifier)
            assert "Invalid identifier format" in str(exc_info)
            # Check for the message case-insensitively
            error_msg_lower = str(exc_info).lower()
            assert "start with a letter or underscore" in error_msg_lower

    def test_validate_identifier_invalid_characters(self):
        """Test that identifiers with invalid characters raise error."""
        invalid_identifiers = [
            "table-name",  # Hyphen
            "table.name",  # Dot
            "table name",  # Space
            "table@name",  # Special char
            "table$name",  # Dollar sign
            "table#name",  # Hash
        ]

        for identifier in invalid_identifiers:
            with pytest.raises(ValidationError) as exc_info:
                DatabaseIdentifierValidator.validate_identifier(identifier)
            assert "Invalid identifier format" in str(exc_info)

    def test_validate_identifier_custom_type(self):
        """Test validation with custom identifier type in error message."""
        with pytest.raises(ValidationError) as exc_info:
            DatabaseIdentifierValidator.validate_identifier("", "Custom type")
        assert "Custom type cannot be empty" in str(exc_info)

        with pytest.raises(ValidationError) as exc_info:
            DatabaseIdentifierValidator.validate_identifier("1invalid", "Column")
        assert "Invalid Column format" in str(exc_info)

    def test_validate_schema_name(self):
        """Test schema name validation."""
        # Valid schema name
        result = DatabaseIdentifierValidator.validate_schema_name("public")
        assert result == "public"

        result = DatabaseIdentifierValidator.validate_schema_name("my_schema")
        assert result == "my_schema"

        # Invalid schema name
        with pytest.raises(ValidationError) as exc_info:
            DatabaseIdentifierValidator.validate_schema_name("schema-name")
        assert "Invalid Schema name format" in str(exc_info)

    def test_validate_table_name(self):
        """Test table name validation."""
        # Valid table name
        result = DatabaseIdentifierValidator.validate_table_name("users")
        assert result == "users"

        result = DatabaseIdentifierValidator.validate_table_name("order_items")
        assert result == "order_items"

        # Invalid table name
        with pytest.raises(ValidationError) as exc_info:
            DatabaseIdentifierValidator.validate_table_name("table.name")
        assert "Invalid Table name format" in str(exc_info)

    def test_validate_identifier_boundary_length(self):
        """Test identifier at maximum allowed length."""
        max_length_identifier = "a" * DatabaseIdentifierValidator.MAX_IDENTIFIER_LENGTH

        # Should be valid at exactly max length
        result = DatabaseIdentifierValidator.validate_identifier(max_length_identifier)
        assert result == max_length_identifier

        # Should fail at max + 1
        too_long = max_length_identifier + "a"
        with pytest.raises(ValidationError):
            DatabaseIdentifierValidator.validate_identifier(too_long)


class TestClassConstants:
    """Test class constants."""

    def test_domain_validator_constants(self):
        """Test DomainValidator constants."""
        assert DomainValidator.MIN_SYMBOL_LENGTH == 1
        assert DomainValidator.MAX_SYMBOL_LENGTH == 10
        assert Decimal("0.01") == DomainValidator.MIN_PRICE
        assert Decimal("999999.99") == DomainValidator.MAX_PRICE
        assert Decimal("0.01") == DomainValidator.MIN_QUANTITY
        assert Decimal("999999") == DomainValidator.MAX_QUANTITY

    def test_database_identifier_constants(self):
        """Test DatabaseIdentifierValidator constants."""
        assert DatabaseIdentifierValidator.MAX_IDENTIFIER_LENGTH == 63  # PostgreSQL limit


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_validate_symbol_with_numbers(self):
        """Test symbol validation with numbers."""
        assert DomainValidator.validate_symbol("3M") == "3M"
        assert DomainValidator.validate_symbol("401K") == "401K"

    def test_validate_price_with_many_decimals(self):
        """Test price validation with many decimal places."""
        result = DomainValidator.validate_price("123.456789")
        assert result == Decimal("123.456789")

    def test_validate_email_with_plus_sign(self):
        """Test email validation with plus sign (valid)."""
        result = DomainValidator.validate_email("user+tag@example.com")
        assert result == "user+tag@example.com"

    def test_validate_email_with_dots(self):
        """Test email validation with dots in local part."""
        result = DomainValidator.validate_email("first.last@example.com")
        assert result == "first.last@example.com"

    def test_validate_percentage_fractional(self):
        """Test percentage validation with fractional values."""
        result = DomainValidator.validate_percentage("33.333333")
        assert result == Decimal("33.333333")

    def test_database_identifier_reserved_words(self):
        """Test that reserved SQL words are still validated as valid identifiers."""
        # The validator doesn't check for reserved words, only format
        reserved_words = ["select", "from", "where", "table", "column"]

        for word in reserved_words:
            result = DatabaseIdentifierValidator.validate_identifier(word)
            assert result == word
