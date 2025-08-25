"""
Unit tests for broker constants module.

Tests cover:
- Constant values verification
- Type checking
- Usage validation
"""

from src.infrastructure.brokers.constants import (
    CURRENCY_CODE_LENGTH,
    DAYS_TO_MONDAY_FROM_FRIDAY,
    DAYS_TO_MONDAY_FROM_SATURDAY,
    HTTP_NOT_FOUND,
    HTTP_UNAUTHORIZED,
    RANDOM_MIN_FACTOR,
    WEEKDAY_FRIDAY,
    WEEKDAY_SATURDAY,
)


class TestHttpStatusCodes:
    """Test HTTP status code constants."""

    def test_http_unauthorized_value(self):
        """Test HTTP_UNAUTHORIZED has correct value."""
        assert HTTP_UNAUTHORIZED == 401
        assert isinstance(HTTP_UNAUTHORIZED, int)

    def test_http_not_found_value(self):
        """Test HTTP_NOT_FOUND has correct value."""
        assert HTTP_NOT_FOUND == 404
        assert isinstance(HTTP_NOT_FOUND, int)

    def test_status_codes_are_distinct(self):
        """Test status codes have different values."""
        assert HTTP_UNAUTHORIZED != HTTP_NOT_FOUND


class TestMarketHoursConstants:
    """Test market hours related constants."""

    def test_weekday_friday_value(self):
        """Test WEEKDAY_FRIDAY represents Friday."""
        assert WEEKDAY_FRIDAY == 4
        assert isinstance(WEEKDAY_FRIDAY, int)
        assert 0 <= WEEKDAY_FRIDAY <= 6  # Valid weekday range

    def test_weekday_saturday_value(self):
        """Test WEEKDAY_SATURDAY represents Saturday."""
        assert WEEKDAY_SATURDAY == 5
        assert isinstance(WEEKDAY_SATURDAY, int)
        assert 0 <= WEEKDAY_SATURDAY <= 6  # Valid weekday range

    def test_days_to_monday_from_friday(self):
        """Test days calculation from Friday to Monday."""
        assert DAYS_TO_MONDAY_FROM_FRIDAY == 3
        assert isinstance(DAYS_TO_MONDAY_FROM_FRIDAY, int)
        assert DAYS_TO_MONDAY_FROM_FRIDAY > 0

    def test_days_to_monday_from_saturday(self):
        """Test days calculation from Saturday to Monday."""
        assert DAYS_TO_MONDAY_FROM_SATURDAY == 2
        assert isinstance(DAYS_TO_MONDAY_FROM_SATURDAY, int)
        assert DAYS_TO_MONDAY_FROM_SATURDAY > 0

    def test_weekday_sequence(self):
        """Test weekday constants are sequential."""
        assert WEEKDAY_SATURDAY == WEEKDAY_FRIDAY + 1

    def test_days_to_monday_consistency(self):
        """Test days to Monday calculations are consistent."""
        # From Friday to Monday is one more day than Saturday to Monday
        assert DAYS_TO_MONDAY_FROM_FRIDAY == DAYS_TO_MONDAY_FROM_SATURDAY + 1


class TestCurrencyValidation:
    """Test currency validation constants."""

    def test_currency_code_length(self):
        """Test standard currency code length."""
        assert CURRENCY_CODE_LENGTH == 3
        assert isinstance(CURRENCY_CODE_LENGTH, int)
        assert CURRENCY_CODE_LENGTH > 0

    def test_currency_code_usage(self):
        """Test currency code length with standard codes."""
        standard_codes = ["USD", "EUR", "GBP", "JPY", "CHF"]
        for code in standard_codes:
            assert len(code) == CURRENCY_CODE_LENGTH


class TestRandomGeneration:
    """Test random generation constants."""

    def test_random_min_factor(self):
        """Test random minimum factor value."""
        assert RANDOM_MIN_FACTOR == 0.5
        assert isinstance(RANDOM_MIN_FACTOR, float)
        assert 0 < RANDOM_MIN_FACTOR < 1

    def test_random_factor_usage(self):
        """Test random factor in typical usage."""
        base_value = 100
        min_value = base_value * RANDOM_MIN_FACTOR
        assert min_value == 50.0
        assert min_value < base_value


class TestConstantsIntegration:
    """Test constants work together properly."""

    def test_all_constants_immutable(self):
        """Test that constants are not accidentally mutable types."""
        from src.infrastructure.brokers import constants

        # Get all constants
        const_names = [
            name for name in dir(constants) if name.isupper() and not name.startswith("_")
        ]

        # Verify each is a simple immutable type
        for name in const_names:
            value = getattr(constants, name)
            assert isinstance(value, (int, float, str, bool, type(None)))

    def test_constants_naming_convention(self):
        """Test constants follow UPPER_SNAKE_CASE convention."""
        from src.infrastructure.brokers import constants

        const_names = [name for name in dir(constants) if not name.startswith("_")]

        for name in const_names:
            if name.isupper():
                # Should be all uppercase with underscores
                assert name.replace("_", "").isupper()
                # Should not have consecutive underscores
                assert "__" not in name
                # Should not start or end with underscore
                assert not name.startswith("_")
                assert not name.endswith("_")
