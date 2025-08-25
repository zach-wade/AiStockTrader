"""
Comprehensive unit tests for Money value object.

Tests all public methods, operators, edge cases, mathematical operations,
comparisons, validation, immutability, and string representations.
"""

# Standard library imports
from decimal import Decimal

# Third-party imports
import pytest

# Local imports
from src.domain.value_objects.money import Money


class TestMoneyCreation:
    """Test Money creation and initialization."""

    def test_create_money_with_decimal(self):
        """Test creating money with Decimal amount."""
        money = Money(Decimal("100.50"), "USD")
        assert money == Decimal("100.50")
        assert money.currency == "USD"

    def test_create_money_with_float(self):
        """Test creating money with float amount."""
        money = Decimal(100.50)
        assert money == Decimal("100.50")
        assert money.currency == "USD"

    def test_create_money_with_int(self):
        """Test creating money with integer amount."""
        money = Decimal(100)
        assert money == Decimal("100")
        assert money.currency == "USD"

    def test_create_money_with_string(self):
        """Test creating money with string amount."""
        money = Decimal("100.50")
        assert money == Decimal("100.50")
        assert money.currency == "USD"

    def test_create_money_default_currency(self):
        """Test creating money with default USD currency."""
        money = Money(Decimal("100.00"))
        assert money.currency == "USD"

    def test_currency_uppercase_normalization(self):
        """Test that currency is normalized to uppercase."""
        money = Money(Decimal("100.00"), "eur")
        assert money.currency == "EUR"

    def test_invalid_currency_code_length(self):
        """Test that invalid currency code length raises ValueError."""
        with pytest.raises(ValueError, match="Invalid currency code: US"):
            Money(Decimal("100.00"), "US")

        with pytest.raises(ValueError, match="Invalid currency code: USDD"):
            Money(Decimal("100.00"), "USDD")

    def test_negative_money_allowed(self):
        """Test that negative amounts are allowed (for debts/losses)."""
        money = Money(Decimal("-100.50"), "USD")
        assert money == Decimal("-100.50")
        assert money.is_negative()

    def test_zero_money_allowed(self):
        """Test that zero amount is allowed."""
        money = Money(Decimal("0"), "USD")
        assert money == Decimal("0")
        assert money.is_zero()

    def test_extreme_precision(self):
        """Test money with extreme decimal precision."""
        money = Money(Decimal("100.123456789"), "USD")
        assert money == Decimal("100.123456789")


class TestMoneyProperties:
    """Test Money property accessors."""

    def test_amount_property(self):
        """Test amount property returns correct value."""
        money = Money(Decimal("123.45"), "USD")
        assert money == Decimal("123.45")

    def test_currency_property(self):
        """Test currency property returns correct value."""
        money = Money(Decimal("100.00"), "EUR")
        assert money.currency == "EUR"


class TestMoneyArithmetic:
    """Test Money arithmetic operations."""

    def test_add_same_currency(self):
        """Test adding money with same currency."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "USD")

        result = money1.add(money2)
        assert isinstance(result, Money)
        assert result == Decimal("150.00")
        assert result.currency == "USD"

    def test_add_different_currency_raises_error(self):
        """Test adding money with different currencies raises error."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "EUR")

        with pytest.raises(ValueError, match="Cannot add USD and EUR"):
            money1.add(money2)

    def test_add_non_money_raises_error(self):
        """Test adding non-Money type raises error."""
        money = Money(Decimal("100.00"), "USD")

        with pytest.raises(TypeError, match="Cannot add Money and"):
            money.add(100)

    def test_subtract_same_currency(self):
        """Test subtracting money with same currency."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("30.00"), "USD")

        result = money1.subtract(money2)
        assert isinstance(result, Money)
        assert result == Decimal("70.00")
        assert result.currency == "USD"

    def test_subtract_resulting_in_negative(self):
        """Test subtracting larger amount results in negative money."""
        money1 = Money(Decimal("50.00"), "USD")
        money2 = Money(Decimal("100.00"), "USD")

        result = money1.subtract(money2)
        assert result == Decimal("-50.00")
        assert result.is_negative()

    def test_subtract_different_currency_raises_error(self):
        """Test subtracting money with different currencies raises error."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "EUR")

        with pytest.raises(ValueError, match="Cannot subtract EUR from USD"):
            money1.subtract(money2)

    def test_subtract_non_money_raises_error(self):
        """Test subtracting non-Money type raises error."""
        money = Money(Decimal("100.00"), "USD")

        with pytest.raises(TypeError, match="Cannot subtract .* from Money"):
            money.subtract(50)

    def test_multiply_by_decimal(self):
        """Test multiplying money by Decimal factor."""
        money = Money(Decimal("100.00"), "USD")

        result = money.multiply(Decimal("2.5"))
        assert isinstance(result, Money)
        assert result == Decimal("250.00")
        assert result.currency == "USD"

    def test_multiply_by_int(self):
        """Test multiplying money by integer factor."""
        money = Money(Decimal("100.00"), "USD")

        result = money.multiply(3)
        assert result == Decimal("300.00")

    def test_multiply_by_float(self):
        """Test multiplying money by float factor."""
        money = Money(Decimal("100.00"), "USD")

        result = money.multiply(0.5)
        assert result == Decimal("50.00")

    def test_multiply_by_zero(self):
        """Test multiplying money by zero."""
        money = Money(Decimal("100.00"), "USD")

        result = money.multiply(0)
        assert result == Decimal("0")
        assert result.is_zero()

    def test_multiply_by_negative(self):
        """Test multiplying money by negative factor."""
        money = Money(Decimal("100.00"), "USD")

        result = money.multiply(-2)
        assert result == Decimal("-200.00")
        assert result.is_negative()

    def test_divide_by_decimal(self):
        """Test dividing money by Decimal divisor."""
        money = Money(Decimal("100.00"), "USD")

        result = money.divide(Decimal("4"))
        assert isinstance(result, Money)
        assert result == Decimal("25.00")
        assert result.currency == "USD"

    def test_divide_by_int(self):
        """Test dividing money by integer divisor."""
        money = Money(Decimal("100.00"), "USD")

        result = money.divide(2)
        assert result == Decimal("50.00")

    def test_divide_by_float(self):
        """Test dividing money by float divisor."""
        money = Money(Decimal("100.00"), "USD")

        result = money.divide(2.5)
        assert result == Decimal("40.00")

    def test_divide_by_zero_raises_error(self):
        """Test dividing by zero raises ValueError."""
        money = Money(Decimal("100.00"), "USD")

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            money.divide(0)

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            money.divide(Decimal("0"))


class TestMoneyRounding:
    """Test Money rounding operations."""

    def test_round_to_two_decimal_places(self):
        """Test rounding to 2 decimal places (default)."""
        money = Money(Decimal("100.12678"), "USD")

        rounded = money.round(2)
        assert rounded == Decimal("100.13")
        assert isinstance(rounded, Money)

    def test_round_to_zero_decimal_places(self):
        """Test rounding to 0 decimal places."""
        money = Money(Decimal("100.678"), "USD")

        rounded = money.round(0)
        assert rounded == Decimal("101")

    def test_round_to_four_decimal_places(self):
        """Test rounding to 4 decimal places."""
        money = Money(Decimal("100.123456"), "USD")

        rounded = money.round(4)
        assert rounded == Decimal("100.1235")

    def test_round_default_decimal_places(self):
        """Test rounding with default decimal places."""
        money = Money(Decimal("100.5678"), "USD")

        rounded = money.round()
        assert rounded == Decimal("100.57")

    def test_round_half_up(self):
        """Test that rounding uses ROUND_HALF_UP."""
        money1 = Money(Decimal("100.125"), "USD")
        money2 = Money(Decimal("100.135"), "USD")

        assert money1.round(2) == Decimal("100.13")
        assert money2.round(2) == Decimal("100.14")


class TestMoneyComparison:
    """Test Money comparison operations."""

    def test_equality_same_currency_and_amount(self):
        """Test equality for same currency and amount."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("100.00"), "USD")

        assert money1 == money2
        assert money1 == money2

    def test_equality_different_amount(self):
        """Test inequality for different amounts."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("200.00"), "USD")

        assert money1 != money2
        assert money1 != money2

    def test_equality_different_currency(self):
        """Test inequality for different currencies."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("100.00"), "EUR")

        assert money1 != money2

    def test_equality_with_non_money(self):
        """Test equality comparison with non-Money types."""
        money = Money(Decimal("100.00"), "USD")

        assert money != 100
        assert money != "100"
        assert money != None

    def test_less_than_same_currency(self):
        """Test less than comparison with same currency."""
        money1 = Money(Decimal("50.00"), "USD")
        money2 = Money(Decimal("100.00"), "USD")

        assert money1 < money2
        assert not money2 < money1

    def test_less_than_different_currency_raises_error(self):
        """Test less than comparison with different currencies raises error."""
        money1 = Money(Decimal("50.00"), "USD")
        money2 = Money(Decimal("100.00"), "EUR")

        with pytest.raises(ValueError, match="Cannot compare USD and EUR"):
            money1 < money2

    def test_less_than_non_money_raises_error(self):
        """Test less than comparison with non-Money raises error."""
        money = Money(Decimal("100.00"), "USD")

        with pytest.raises(TypeError, match="Cannot compare Money and"):
            money < 50

    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        money1 = Money(Decimal("50.00"), "USD")
        money2 = Money(Decimal("100.00"), "USD")
        money3 = Money(Decimal("100.00"), "USD")

        assert money1 <= money2
        assert money2 <= money3
        assert not money2 <= money1

    def test_greater_than_same_currency(self):
        """Test greater than comparison with same currency."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "USD")

        assert money1 > money2
        assert not money2 > money1

    def test_greater_than_different_currency_raises_error(self):
        """Test greater than comparison with different currencies raises error."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "EUR")

        with pytest.raises(ValueError, match="Cannot compare USD and EUR"):
            money1 > money2

    def test_greater_than_non_money_raises_error(self):
        """Test greater than comparison with non-Money raises error."""
        money = Money(Decimal("100.00"), "USD")

        with pytest.raises(TypeError, match="Cannot compare Money and"):
            money > 150

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("50.00"), "USD")
        money3 = Money(Decimal("100.00"), "USD")

        assert money1 >= money2
        assert money1 >= money3
        assert not money2 >= money1


class TestMoneyUtilityMethods:
    """Test Money utility methods."""

    def test_is_positive(self):
        """Test is_positive method."""
        positive = Money(Decimal("100.00"), "USD")
        negative = Money(Decimal("-100.00"), "USD")
        zero = Money(Decimal("0"), "USD")

        assert positive.is_positive()
        assert not negative.is_positive()
        assert not zero.is_positive()

    def test_is_negative(self):
        """Test is_negative method."""
        positive = Money(Decimal("100.00"), "USD")
        negative = Money(Decimal("-100.00"), "USD")
        zero = Money(Decimal("0"), "USD")

        assert not positive.is_negative()
        assert negative.is_negative()
        assert not zero.is_negative()

    def test_is_zero(self):
        """Test is_zero method."""
        positive = Money(Decimal("100.00"), "USD")
        negative = Money(Decimal("-100.00"), "USD")
        zero = Money(Decimal("0"), "USD")

        assert not positive.is_zero()
        assert not negative.is_zero()
        assert zero.is_zero()

    def test_negation_operator(self):
        """Test negation operator."""
        positive = Money(Decimal("100.00"), "USD")
        negative = -positive

        assert negative == Decimal("-100.00")
        assert negative.currency == "USD"
        assert isinstance(negative, Money)

        # Double negation
        double_neg = -negative
        assert double_neg == Decimal("100.00")

    def test_abs_operator(self):
        """Test absolute value operator."""
        positive = Money(Decimal("100.00"), "USD")
        negative = Money(Decimal("-100.00"), "USD")

        assert abs(positive) == Decimal("100.00")
        assert abs(negative) == Decimal("100.00")
        assert isinstance(abs(negative), Money)

    def test_hash(self):
        """Test hash for use in sets and dicts."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("100.00"), "USD")
        money3 = Money(Decimal("200.00"), "USD")
        money4 = Money(Decimal("100.00"), "EUR")

        # Equal Money objects have same hash
        assert hash(money1) == hash(money2)

        # Can be used in sets
        money_set = {money1, money2, money3, money4}
        assert len(money_set) == 3  # money1 and money2 are equal


class TestMoneyFormatting:
    """Test Money formatting and display."""

    def test_format_usd_with_symbol(self):
        """Test formatting USD with dollar symbol."""
        money = Money(Decimal("1234.56"), "USD")

        formatted = money.format(include_currency=True, decimal_places=2)
        assert formatted == "$1,234.56"

    def test_format_usd_without_symbol(self):
        """Test formatting USD without currency symbol."""
        money = Money(Decimal("1234.56"), "USD")

        formatted = money.format(include_currency=False, decimal_places=2)
        assert formatted == "1,234.56"

    def test_format_other_currency(self):
        """Test formatting non-USD currency."""
        money = Money(Decimal("1234.56"), "EUR")

        formatted = money.format(include_currency=True, decimal_places=2)
        assert formatted == "1,234.56 EUR"

    def test_format_with_different_decimal_places(self):
        """Test formatting with different decimal places."""
        money = Money(Decimal("1234.5678"), "USD")

        assert money.format(decimal_places=0) == "$1,235"
        assert money.format(decimal_places=1) == "$1,234.6"
        assert money.format(decimal_places=3) == "$1,234.568"
        assert money.format(decimal_places=4) == "$1,234.5678"

    def test_format_large_numbers(self):
        """Test formatting large numbers with thousands separators."""
        money = Money(Decimal("1234567890.12"), "USD")

        formatted = money.format()
        assert formatted == "$1,234,567,890.12"

    def test_format_negative_amounts(self):
        """Test formatting negative amounts."""
        money = Money(Decimal("-1234.56"), "USD")

        formatted = money.format()
        assert formatted == "$-1,234.56"

    def test_str_representation(self):
        """Test string representation."""
        money = Money(Decimal("1234.56"), "USD")
        assert str(money) == "$1,234.56"

    def test_repr_representation(self):
        """Test repr representation."""
        money = Money(Decimal("100.50"), "USD")
        assert repr(money) == "Money(100.50, 'USD')"


class TestMoneyEdgeCases:
    """Test Money edge cases and extreme values."""

    def test_very_large_amount(self):
        """Test handling very large amounts."""
        large_money = Money(Decimal("999999999999999.99"), "USD")
        assert large_money == Decimal("999999999999999.99")

    def test_very_small_amount(self):
        """Test handling very small amounts."""
        small_money = Money(Decimal("0.0000000001"), "USD")
        assert small_money == Decimal("0.0000000001")

    def test_many_decimal_places(self):
        """Test handling many decimal places."""
        money = Money(Decimal("100.123456789123456789"), "USD")
        assert money == Decimal("100.123456789123456789")

    def test_scientific_notation(self):
        """Test handling scientific notation."""
        money = Decimal("1.5E+3")
        assert money == Decimal("1500")

    def test_immutability(self):
        """Test that Money is immutable."""
        money = Money(Decimal("100.00"), "USD")
        original_amount = money
        original_currency = money.currency

        # Operations return new objects
        new_money = money.add(Money(Decimal("50.00"), "USD"))
        assert money == original_amount
        assert money.currency == original_currency
        assert new_money == Decimal("150.00")

        # Properties can't be modified
        with pytest.raises(AttributeError):
            money = Decimal("200.00")

        with pytest.raises(AttributeError):
            money.currency = "EUR"

    def test_different_currency_codes(self):
        """Test various valid currency codes."""
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

        for currency in currencies:
            money = Money(Decimal("100.00"), currency)
            assert money.currency == currency

    def test_chain_operations(self):
        """Test chaining multiple operations."""
        money = Money(Decimal("100.00"), "USD")

        result = money.add(Money(Decimal("50.00"), "USD")).multiply(2).divide(3)
        assert result == Decimal("100.00")

        # Original unchanged
        assert money == Decimal("100.00")

    def test_zero_arithmetic(self):
        """Test arithmetic with zero amounts."""
        zero = Money(Decimal("0"), "USD")
        hundred = Money(Decimal("100.00"), "USD")

        # Adding zero
        assert hundred.add(zero) == Decimal("100.00")
        assert zero.add(hundred) == Decimal("100.00")

        # Subtracting zero
        assert hundred.subtract(zero) == Decimal("100.00")
        assert zero.subtract(hundred) == Decimal("-100.00")

        # Multiplying by zero
        assert hundred.multiply(0) == Decimal("0")
        assert zero.multiply(100) == Decimal("0")

    def test_precision_preservation(self):
        """Test that precision is preserved in operations."""
        money1 = Money(Decimal("0.001"), "USD")
        money2 = Money(Decimal("0.002"), "USD")

        result = money1.add(money2)
        assert result == Decimal("0.003")

        # Multiplication preserves precision
        result = money1.multiply(Decimal("3"))
        assert result == Decimal("0.003")
