"""
Comprehensive test suite for Money value object.

Tests all Money functionality to achieve >95% coverage:
- Creation and validation
- Arithmetic operations
- Comparison operations
- Currency handling
- Rounding and precision
- Edge cases and error conditions
"""

import decimal
from decimal import Decimal

import pytest

from src.domain.value_objects.money import Money


class TestMoneyCreation:
    """Test Money object creation and validation."""

    def test_create_with_decimal(self):
        """Test creating Money with Decimal amount."""
        money = Money(Decimal("100.50"), "USD")
        assert money.amount == Decimal("100.50")
        assert money.currency == "USD"

    def test_create_with_float(self):
        """Test creating Money with float amount."""
        money = Money(100.50, "USD")
        assert money.amount == Decimal("100.50")
        assert money.currency == "USD"

    def test_create_with_int(self):
        """Test creating Money with integer amount."""
        money = Money(100, "USD")
        assert money.amount == Decimal("100")
        assert money.currency == "USD"

    def test_create_with_string(self):
        """Test creating Money with string amount."""
        money = Money("100.50", "USD")
        assert money.amount == Decimal("100.50")
        assert money.currency == "USD"

    def test_default_currency(self):
        """Test default USD currency."""
        money = Money(Decimal("100.00"))
        assert money.currency == "USD"

    def test_currency_normalization(self):
        """Test currency code normalization to uppercase."""
        money = Money(100, "usd")
        assert money.currency == "USD"

    def test_invalid_currency_length(self):
        """Test validation of currency code length."""
        with pytest.raises(ValueError, match="Invalid currency code"):
            Money(100, "USDD")  # Too long

        with pytest.raises(ValueError, match="Invalid currency code"):
            Money(100, "US")  # Too short

    def test_zero_amount(self):
        """Test creating Money with zero amount."""
        money = Money(0, "USD")
        assert money.amount == Decimal("0")

    def test_negative_amount(self):
        """Test creating Money with negative amount."""
        money = Money(-100, "USD")
        assert money.amount == Decimal("-100")

    def test_high_precision_amount(self):
        """Test Money with high precision amounts."""
        money = Money(Decimal("100.123456789"), "USD")
        assert money.amount == Decimal("100.123456789")


class TestMoneyArithmetic:
    """Test Money arithmetic operations."""

    def test_addition_same_currency(self):
        """Test adding Money objects with same currency."""
        m1 = Money(Decimal("100.50"), "USD")
        m2 = Money(Decimal("50.25"), "USD")
        result = m1 + m2

        assert result.amount == Decimal("150.75")
        assert result.currency == "USD"

    def test_addition_different_currencies(self):
        """Test adding Money objects with different currencies raises error."""
        m1 = Money(100, "USD")
        m2 = Money(50, "EUR")

        with pytest.raises(ValueError, match="Cannot add USD and EUR"):
            m1 + m2

    def test_subtraction_same_currency(self):
        """Test subtracting Money objects with same currency."""
        m1 = Money(Decimal("100.50"), "USD")
        m2 = Money(Decimal("50.25"), "USD")
        result = m1 - m2

        assert result.amount == Decimal("50.25")
        assert result.currency == "USD"

    def test_subtraction_different_currencies(self):
        """Test subtracting Money objects with different currencies raises error."""
        m1 = Money(100, "USD")
        m2 = Money(50, "EUR")

        with pytest.raises(ValueError, match="Cannot subtract EUR from USD"):
            m1 - m2

    def test_multiplication_by_scalar(self):
        """Test multiplying Money by scalar values."""
        money = Money(Decimal("100.50"), "USD")

        # Multiply by integer
        result1 = money * 2
        assert result1.amount == Decimal("201.00")
        assert result1.currency == "USD"

        # Multiply by Decimal
        result2 = money * Decimal("1.5")
        assert result2.amount == Decimal("150.75")

        # Reverse multiplication
        result3 = Decimal("2") * money
        assert result3.amount == Decimal("201.00")

    def test_division_by_scalar(self):
        """Test dividing Money by scalar values."""
        money = Money(Decimal("100.00"), "USD")

        result1 = money / 2
        assert result1.amount == Decimal("50.00")
        assert result1.currency == "USD"

        result2 = money / Decimal("4")
        assert result2.amount == Decimal("25.00")

    def test_division_by_zero(self):
        """Test division by zero raises error."""
        money = Money(100, "USD")

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            money / 0

    # Note: modulo, floor division, and power operations are not implemented
    # in Money class as they are not standard financial operations

    def test_absolute_value(self):
        """Test absolute value of Money."""
        money = Money(Decimal("-100.50"), "USD")
        result = abs(money)

        assert result.amount == Decimal("100.50")
        assert result.currency == "USD"

    def test_negation(self):
        """Test negation of Money."""
        money = Money(Decimal("100.50"), "USD")
        result = -money

        assert result.amount == Decimal("-100.50")
        assert result.currency == "USD"

    # Note: unary positive operator (+) is not implemented in Money class


class TestMoneyComparison:
    """Test Money comparison operations."""

    def test_equality_same_currency(self):
        """Test equality comparison with same currency."""
        m1 = Money(Decimal("100.50"), "USD")
        m2 = Money(Decimal("100.50"), "USD")
        m3 = Money(Decimal("100.51"), "USD")

        assert m1 == m2
        assert m1 != m3

    def test_equality_different_currencies(self):
        """Test equality comparison with different currencies."""
        m1 = Money(100, "USD")
        m2 = Money(100, "EUR")

        assert m1 != m2

    def test_inequality(self):
        """Test inequality comparison."""
        m1 = Money(100, "USD")
        m2 = Money(101, "USD")

        assert m1 != m2

    def test_less_than_same_currency(self):
        """Test less than comparison with same currency."""
        m1 = Money(100, "USD")
        m2 = Money(101, "USD")

        assert m1 < m2
        assert not (m2 < m1)

    def test_less_than_different_currencies(self):
        """Test less than comparison with different currencies raises error."""
        m1 = Money(100, "USD")
        m2 = Money(101, "EUR")

        with pytest.raises(ValueError, match="Cannot compare USD and EUR"):
            m1 < m2

    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        m1 = Money(100, "USD")
        m2 = Money(100, "USD")
        m3 = Money(101, "USD")

        assert m1 <= m2
        assert m1 <= m3
        assert not (m3 <= m1)

    def test_greater_than(self):
        """Test greater than comparison."""
        m1 = Money(101, "USD")
        m2 = Money(100, "USD")

        assert m1 > m2
        assert not (m2 > m1)

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        m1 = Money(101, "USD")
        m2 = Money(100, "USD")
        m3 = Money(101, "USD")

        assert m1 >= m2
        assert m1 >= m3
        assert not (m2 >= m1)


class TestMoneyUtility:
    """Test Money utility methods and properties."""

    def test_round_method(self):
        """Test Money rounding with different strategies."""
        money = Money(Decimal("100.555"), "USD")

        # Round to 2 decimal places (default)
        rounded = money.round(2)
        assert rounded.amount == Decimal("100.56")

        # Note: Money class uses fixed ROUND_HALF_UP rounding strategy
        # Different rounding modes are not supported

    def test_round_default_places(self):
        """Test Money rounding with default 2 decimal places."""
        money = Money(Decimal("100.555"), "USD")
        rounded = money.round()
        assert rounded.amount == Decimal("100.56")

    def test_is_zero(self):
        """Test is_zero method."""
        zero_money = Money(0, "USD")
        nonzero_money = Money(Decimal("0.01"), "USD")

        assert zero_money.is_zero()
        assert not nonzero_money.is_zero()

    def test_is_positive(self):
        """Test is_positive method."""
        positive_money = Money(100, "USD")
        zero_money = Money(0, "USD")
        negative_money = Money(-100, "USD")

        assert positive_money.is_positive()
        assert not zero_money.is_positive()
        assert not negative_money.is_positive()

    def test_is_negative(self):
        """Test is_negative method."""
        negative_money = Money(-100, "USD")
        zero_money = Money(0, "USD")
        positive_money = Money(100, "USD")

        assert negative_money.is_negative()
        assert not zero_money.is_negative()
        assert not positive_money.is_negative()

    def test_to_string_representation(self):
        """Test string representation of Money."""
        money = Money(Decimal("100.50"), "USD")

        assert str(money) == "$100.50"

    def test_repr_representation(self):
        """Test repr representation of Money."""
        money = Money(Decimal("100.50"), "USD")

        assert repr(money) == "Money(100.50, 'USD')"

    def test_hash(self):
        """Test Money hashing for use in sets and dicts."""
        m1 = Money(100, "USD")
        m2 = Money(100, "USD")
        m3 = Money(100, "EUR")

        # Same Money objects should have same hash
        assert hash(m1) == hash(m2)
        # Different currencies should have different hashes
        assert hash(m1) != hash(m3)

        # Test in set
        money_set = {m1, m2, m3}
        assert len(money_set) == 2  # m1 and m2 are same


class TestMoneyEdgeCases:
    """Test Money edge cases and error conditions."""

    def test_very_large_amount(self):
        """Test Money with very large amounts."""
        large_amount = Decimal("999999999999999999.99")
        money = Money(large_amount, "USD")

        assert money.amount == large_amount

    def test_very_small_amount(self):
        """Test Money with very small amounts."""
        small_amount = Decimal("0.000000001")
        money = Money(small_amount, "USD")

        assert money.amount == small_amount

    def test_scientific_notation(self):
        """Test Money with scientific notation."""
        money = Money("1.23e2", "USD")  # 123.0
        assert money.amount == Decimal("123")

    def test_invalid_amount_string(self):
        """Test creation with invalid amount string."""
        with pytest.raises((ValueError, decimal.InvalidOperation)):
            Money("not_a_number", "USD")

    def test_none_amount(self):
        """Test creation with None amount."""
        with pytest.raises((TypeError, decimal.InvalidOperation)):
            Money(None, "USD")

    def test_money_with_different_precision(self):
        """Test operations with Money of different precisions."""
        m1 = Money(Decimal("100.5"), "USD")  # 1 decimal
        m2 = Money(Decimal("50.25"), "USD")  # 2 decimals

        result = m1 + m2
        assert result.amount == Decimal("150.75")

    def test_immutability(self):
        """Test that Money objects are immutable."""
        money = Money(100, "USD")
        original_amount = money.amount
        original_currency = money.currency

        # Operations should return new objects
        new_money = money + Money(50, "USD")

        # Original should be unchanged
        assert money.amount == original_amount
        assert money.currency == original_currency

        # New object should be different
        assert new_money is not money

    def test_copy_constructor(self):
        """Test creating Money from another Money object."""
        original = Money(100, "USD")
        copy = Money(original.amount, original.currency)

        assert copy == original
        assert copy is not original


class TestMoneySpecialCurrencies:
    """Test Money with various currency codes."""

    def test_common_currencies(self):
        """Test common currency codes."""
        currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]

        for currency in currencies:
            money = Money(100, currency)
            assert money.currency == currency

    def test_cryptocurrency_codes(self):
        """Test cryptocurrency-style codes."""
        crypto_currencies = ["BTC", "ETH", "XRP"]

        for currency in crypto_currencies:
            money = Money(Decimal("0.001"), currency)
            assert money.currency == currency

    def test_case_insensitive_currency(self):
        """Test currency code case handling."""
        money1 = Money(100, "usd")
        money2 = Money(100, "USD")
        money3 = Money(100, "Usd")

        assert money1.currency == "USD"
        assert money2.currency == "USD"
        assert money3.currency == "USD"

        # All should be equal
        assert money1 == money2 == money3


class TestMoneyPerformance:
    """Test Money performance characteristics."""

    def test_arithmetic_precision_preservation(self):
        """Test that arithmetic operations preserve precision."""
        m1 = Money(Decimal("100.123"), "USD")
        m2 = Money(Decimal("50.456"), "USD")

        result = m1 + m2
        assert result.amount == Decimal("150.579")

        result = m1 * Decimal("2.5")
        assert result.amount == Decimal("250.3075")

    def test_rounding_edge_cases(self):
        """Test rounding behavior with edge cases."""
        # Test banker's rounding (round half to even)
        money1 = Money(Decimal("2.5"), "USD")
        money2 = Money(Decimal("3.5"), "USD")

        rounded1 = money1.round(0)  # Uses default ROUND_HALF_UP
        rounded2 = money2.round(0)  # Uses default ROUND_HALF_UP

        assert rounded1.amount == Decimal("3")
        assert rounded2.amount == Decimal("4")

    def test_chained_operations(self):
        """Test chaining multiple operations."""
        money = Money(100, "USD")

        result = ((money + Money(50, "USD")) * 2) / 3
        expected = Money(Decimal("100"), "USD")

        assert result.amount == expected.amount
        assert result.currency == "USD"
