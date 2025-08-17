"""Edge case tests for Money value object."""

# Standard library imports
from decimal import Decimal

# Third-party imports
import pytest

# Local imports
from src.domain.value_objects.money import Money


class TestMoneyEdgeCases:
    """Test edge cases for Money value object."""

    def test_money_division_by_zero(self):
        """Test that division by zero raises appropriate error."""
        money = Money(Decimal("100.00"), "USD")

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            money.divide(0)

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            money.divide(Decimal("0"))

    def test_money_negative_operations(self):
        """Test operations with negative money values."""
        positive = Money(Decimal("100.00"), "USD")
        negative = Money(Decimal("-50.00"), "USD")

        # Addition with negative
        result = positive.add(negative)
        assert result.amount == Decimal("50.00")

        # Subtraction resulting in negative
        result = negative.subtract(positive)
        assert result.amount == Decimal("-150.00")

        # Multiplication with negative
        result = negative.multiply(2)
        assert result.amount == Decimal("-100.00")

        # Division with negative
        result = negative.divide(-2)
        assert result.amount == Decimal("25.00")

    def test_money_extreme_values(self):
        """Test money with extremely large and small values."""
        # Very large amount
        large = Money(Decimal("999999999999.99"), "USD")
        assert large.amount == Decimal("999999999999.99")

        # Operations with large values
        result = large.add(large)
        assert result.amount == Decimal("1999999999999.98")

        # Very small amount (but not zero)
        small = Money(Decimal("0.01"), "USD")
        result = small.divide(100)
        assert result.amount == Decimal("0.0001")

    def test_money_currency_mismatch(self):
        """Test that operations with different currencies raise errors."""
        usd = Money(Decimal("100.00"), "USD")
        eur = Money(Decimal("100.00"), "EUR")

        with pytest.raises(ValueError, match="Cannot add"):
            usd.add(eur)

        with pytest.raises(ValueError, match="Cannot subtract"):
            usd.subtract(eur)

        with pytest.raises(ValueError, match="Cannot compare"):
            _ = usd < eur

    def test_money_comparison_edge_cases(self):
        """Test edge cases in money comparisons."""
        money1 = Money(Decimal("100.00"), "USD")
        money2 = Money(Decimal("100.00"), "USD")
        money3 = Money(Decimal("100.01"), "USD")

        # Equality with same value
        assert money1 == money2
        assert money1 == money2

        # Inequality
        assert money1 != money3
        assert money1 < money3
        assert money3 > money1
        assert money1 <= money2
        assert money1 >= money2

        # Comparison with None
        assert money1 is not None

        # Comparison with non-Money types
        assert money1 != 100
        assert money1 != "100.00"

    def test_money_string_representation(self):
        """Test string representation edge cases."""
        # Positive amount
        money = Money(Decimal("1234.56"), "USD")
        assert str(money) == "$1,234.56"

        # Negative amount
        money = Money(Decimal("-1234.56"), "USD")
        assert str(money) == "$-1,234.56"  # Formatting puts $ before negative sign

        # Zero amount
        money = Money(Decimal("0.00"), "USD")
        assert str(money) == "$0.00"

        # Very small amount
        money = Money(Decimal("0.01"), "USD")
        assert str(money) == "$0.01"

        # Large amount
        money = Money(Decimal("1000000.00"), "USD")
        assert str(money) == "$1,000,000.00"

    def test_money_type_errors(self):
        """Test type errors in money operations."""
        money = Money(Decimal("100.00"), "USD")

        # Test adding non-Money type
        with pytest.raises(TypeError, match="Cannot add Money"):
            money.add(100)

        with pytest.raises(TypeError, match="Cannot add Money"):
            money.add("100")

        # Test subtracting non-Money type
        with pytest.raises(TypeError, match="Cannot subtract"):
            money.subtract(50)

        # Test comparing with non-Money type
        with pytest.raises(TypeError, match="Cannot compare"):
            _ = money < 100

        with pytest.raises(TypeError, match="Cannot compare"):
            _ = money > "100"

    def test_money_rounding(self):
        """Test money rounding behavior."""
        # Test rounding with different decimal places
        money = Money(Decimal("100.12789"), "USD")
        assert money.amount == Decimal("100.12789")

        # Round to different decimal places
        rounded_2 = money.round(2)
        assert rounded_2.amount == Decimal("100.13")

        rounded_0 = money.round(0)
        assert rounded_0.amount == Decimal("100")

        rounded_4 = money.round(4)
        assert rounded_4.amount == Decimal("100.1279")

        # Test rounding at boundary (ROUND_HALF_UP)
        money = Money(Decimal("100.125"), "USD")
        rounded = money.round(2)
        assert rounded.amount == Decimal("100.13")

        money = Money(Decimal("100.124"), "USD")
        rounded = money.round(2)
        assert rounded.amount == Decimal("100.12")

        # Negative rounding
        money = Money(Decimal("-100.125"), "USD")
        rounded = money.round(2)
        assert rounded.amount == Decimal("-100.13")

    def test_money_abs_value(self):
        """Test absolute value operations."""
        positive = Money(Decimal("100.00"), "USD")
        negative = Money(Decimal("-100.00"), "USD")

        # Test __abs__ method
        assert positive.__abs__().amount == Decimal("100.00")
        assert negative.__abs__().amount == Decimal("100.00")
        assert Money(Decimal("0.00"), "USD").__abs__().amount == Decimal("0.00")

        # Test __neg__ method
        assert positive.__neg__().amount == Decimal("-100.00")
        assert negative.__neg__().amount == Decimal("100.00")
        assert Money(Decimal("0.00"), "USD").__neg__().amount == Decimal("0.00")

    def test_money_zero_operations(self):
        """Test operations with zero money."""
        zero = Money(Decimal("0.00"), "USD")
        hundred = Money(Decimal("100.00"), "USD")

        # Addition with zero
        assert zero.add(hundred).amount == Decimal("100.00")
        assert hundred.add(zero).amount == Decimal("100.00")

        # Subtraction with zero
        assert hundred.subtract(zero).amount == Decimal("100.00")
        assert zero.subtract(hundred).amount == Decimal("-100.00")

        # Multiplication with zero
        assert zero.multiply(5).amount == Decimal("0.00")
        assert hundred.multiply(0).amount == Decimal("0.00")

        # Division of zero
        assert zero.divide(5).amount == Decimal("0.00")

        # But division BY zero should fail
        with pytest.raises(ValueError):
            hundred.divide(0)

    def test_money_hash_and_repr(self):
        """Test hashing and repr for Money."""
        money1 = Money(Decimal("100.50"), "USD")
        money2 = Money(Decimal("100.50"), "USD")
        money3 = Money(Decimal("100.50"), "EUR")

        # Same money should have same hash
        assert hash(money1) == hash(money2)
        assert hash(money1) != hash(money3)

        # Should be usable in sets
        money_set = {money1, money2, money3}
        assert len(money_set) == 2  # money1 and money2 are equal

        # Test repr
        assert repr(money1) == "Money(100.50, 'USD')"
        assert repr(money3) == "Money(100.50, 'EUR')"

        # Test str
        assert str(money1) == "$100.50"

    def test_money_format_edge_cases(self):
        """Test edge cases in money formatting."""
        # Test formatting with different decimal places
        money = Money(Decimal("1234.567"), "USD")
        assert money.format(decimal_places=0) == "$1,235"
        assert money.format(decimal_places=1) == "$1,234.6"
        assert money.format(decimal_places=3) == "$1,234.567"

        # Test formatting without currency
        assert money.format(include_currency=False, decimal_places=2) == "1,234.57"

        # Test non-USD currency formatting
        eur = Money(Decimal("1234.56"), "EUR")
        assert eur.format() == "1,234.56 EUR"
        assert eur.format(include_currency=False) == "1,234.56"

        jpy = Money(Decimal("1234.56"), "JPY")
        assert jpy.format() == "1,234.56 JPY"

        # Test negative formatting
        negative = Money(Decimal("-1234.56"), "USD")
        assert negative.format() == "$-1,234.56"

        # Test very small values
        small = Money(Decimal("0.0001"), "USD")
        assert small.format(decimal_places=4) == "$0.0001"
        assert small.format(decimal_places=2) == "$0.00"
