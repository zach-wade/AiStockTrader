"""Edge case tests for Quantity value object."""

# Standard library imports
from decimal import Decimal

# Third-party imports
import pytest

# Local imports
from src.domain.value_objects.quantity import Quantity


class TestQuantityEdgeCases:
    """Test edge cases for Quantity value object."""

    def test_quantity_zero_handling(self):
        """Test that zero quantities are handled correctly."""
        zero = Quantity(0)
        assert zero.value == Decimal("0")
        assert zero.is_zero()
        assert not zero.is_long()
        assert not zero.is_short()
        assert zero.is_valid()

        # Zero from different types
        zero_float = Quantity(0.0)
        assert zero_float.is_zero()

        zero_str = Quantity("0.00000")
        assert zero_str.is_zero()

    def test_quantity_negative_handling(self):
        """Test negative quantities (short positions)."""
        short = Quantity(-100)
        assert short.value == Decimal("-100")
        assert short.is_short()
        assert not short.is_long()
        assert not short.is_zero()
        assert short.is_valid()

        # Very small negative
        tiny_short = Quantity(-0.0001)
        assert tiny_short.is_short()

    def test_quantity_extreme_values(self):
        """Test extreme quantity values."""
        # Very large quantity
        large = Quantity("999999999999999999")
        assert large.value == Decimal("999999999999999999")
        assert large.is_long()

        # Very small positive quantity
        tiny = Quantity("0.00000001")
        assert tiny.value == Decimal("0.00000001")
        assert tiny.is_long()
        assert not tiny.is_zero()

        # Very large negative
        large_short = Quantity("-999999999999999999")
        assert large_short.is_short()

    def test_quantity_arithmetic_edge_cases(self):
        """Test edge cases in quantity arithmetic."""
        # Adding opposite signs
        long_qty = Quantity(100)
        short_qty = Quantity(-50)
        result = long_qty.add(short_qty)
        assert result.value == Decimal("50")
        assert result.is_long()

        # Adding to zero
        result = short_qty.add(Quantity(50))
        assert result.value == Decimal("0")
        assert result.is_zero()

        # Subtracting to negative
        result = short_qty.subtract(long_qty)
        assert result.value == Decimal("-150")
        assert result.is_short()

        # Multiply by zero
        result = long_qty.multiply(0)
        assert result.value == Decimal("0")
        assert result.is_zero()

        # Multiply by negative
        result = long_qty.multiply(-1)
        assert result.value == Decimal("-100")
        assert result.is_short()

        # Divide by very small number
        result = long_qty.divide(0.001)
        assert result.value == Decimal("100000")

        # Division by zero
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            long_qty.divide(0)

    def test_quantity_type_errors(self):
        """Test type errors in quantity operations."""
        qty = Quantity(100)

        # Adding non-Quantity type
        with pytest.raises(TypeError, match="Cannot add Quantity"):
            qty.add(100)

        with pytest.raises(TypeError, match="Cannot add Quantity"):
            qty.add("100")

        # Subtracting non-Quantity type
        with pytest.raises(TypeError, match="Cannot subtract"):
            qty.subtract(50)

        # Comparing with non-Quantity type
        with pytest.raises(TypeError, match="Cannot compare"):
            _ = qty < 100

        with pytest.raises(TypeError, match="Cannot compare"):
            _ = qty > "100"

    def test_quantity_split_edge_cases(self):
        """Test edge cases in quantity splitting."""
        # Split into 1 part
        qty = Quantity(100)
        parts = qty.split(1)
        assert len(parts) == 1
        assert parts[0].value == Decimal("100")

        # Split with no remainder
        qty = Quantity(100)
        parts = qty.split(4)
        assert len(parts) == 4
        assert all(p.value == Decimal("25") for p in parts)

        # Split with remainder distribution
        qty = Quantity(103)
        parts = qty.split(4)
        assert len(parts) == 4
        # First part gets remainder
        assert parts[0].value == Decimal("28")
        assert all(p.value == Decimal("25") for p in parts[1:])
        # Total should equal original
        total = sum(p.value for p in parts)
        assert total == Decimal("103")

        # Split negative quantity
        qty = Quantity(-100)
        parts = qty.split(4)
        assert len(parts) == 4
        assert parts[0].value == Decimal("-25")
        assert all(p.value == Decimal("-25") for p in parts)

        # Split zero
        qty = Quantity(0)
        parts = qty.split(5)
        assert len(parts) == 5
        assert all(p.value == Decimal("0") for p in parts)

        # Split decimal quantity
        qty = Quantity("10.5")
        parts = qty.split(3)
        assert len(parts) == 3
        # With ROUND_DOWN, each gets 3, remainder is 1.5
        assert parts[0].value == Decimal("4.5")  # 3 + 1.5
        assert parts[1].value == Decimal("3")
        assert parts[2].value == Decimal("3")

        # Invalid split
        with pytest.raises(ValueError, match="Number of parts must be at least 1"):
            qty.split(0)

        with pytest.raises(ValueError, match="Number of parts must be at least 1"):
            qty.split(-1)

    def test_quantity_rounding_edge_cases(self):
        """Test edge cases in quantity rounding."""
        # Rounding positive
        qty = Quantity("123.456789")
        assert qty.round(0).value == Decimal("123")
        assert qty.round(2).value == Decimal("123.45")  # ROUND_DOWN
        assert qty.round(5).value == Decimal("123.45678")  # ROUND_DOWN

        # Rounding negative
        qty = Quantity("-123.456789")
        assert qty.round(0).value == Decimal("-123")
        assert qty.round(2).value == Decimal("-123.45")  # ROUND_DOWN toward zero

        # Rounding zero
        qty = Quantity(0)
        assert qty.round(0).value == Decimal("0")
        assert qty.round(5).value == Decimal("0.00000")

        # Invalid decimal places
        qty = Quantity(100)
        with pytest.raises(ValueError, match="Decimal places must be non-negative"):
            qty.round(-1)

        # Very precise rounding
        qty = Quantity("0.123456789012345678901234567890")
        rounded = qty.round(20)
        assert str(rounded.value).startswith("0.12345678901234567890")

    def test_quantity_abs_and_neg(self):
        """Test absolute value and negation edge cases."""
        # Positive quantity
        pos = Quantity(100)
        assert pos.abs().value == Decimal("100")
        assert (-pos).value == Decimal("-100")
        assert abs(pos).value == Decimal("100")

        # Negative quantity
        neg = Quantity(-50)
        assert neg.abs().value == Decimal("50")
        assert (-neg).value == Decimal("50")
        assert abs(neg).value == Decimal("50")

        # Zero
        zero = Quantity(0)
        assert zero.abs().value == Decimal("0")
        assert (-zero).value == Decimal("0")
        assert abs(zero).value == Decimal("0")

    def test_quantity_comparison_edge_cases(self):
        """Test edge cases in quantity comparisons."""
        # Very close values
        qty1 = Quantity("100")
        qty2 = Quantity("100.00000001")
        assert qty1 < qty2
        assert qty1 <= qty2
        assert qty2 > qty1
        assert qty2 >= qty1
        assert qty1 != qty2

        # Comparing different signs
        pos = Quantity(10)
        neg = Quantity(-10)
        zero = Quantity(0)

        assert neg < zero < pos
        assert pos > zero > neg

        # Equality with different representations
        qty1 = Quantity(100)
        qty2 = Quantity("100.00")
        qty3 = Quantity(Decimal("100"))

        assert qty1 == qty2 == qty3
        assert hash(qty1) == hash(qty2) == hash(qty3)

        # Use in sets
        qty_set = {qty1, qty2, qty3}
        assert len(qty_set) == 1  # All are equal

    def test_quantity_string_representations(self):
        """Test string representations with edge cases."""
        # Integer-like values
        qty = Quantity(100)
        assert str(qty) == "100"  # No unnecessary decimals
        assert repr(qty) == "Quantity(100)"

        # Decimal values
        qty = Quantity("100.50")
        assert str(qty) == "100.5"
        assert repr(qty) == "Quantity(100.50)"

        # Zero
        qty = Quantity(0)
        assert str(qty) == "0"
        assert repr(qty) == "Quantity(0)"

        # Negative
        qty = Quantity(-42.5)
        assert str(qty) == "-42.5"
        assert repr(qty) == "Quantity(-42.5)"

        # Very small
        qty = Quantity("0.00000001")
        assert str(qty) in ["0.00000001", "1E-8"]  # Accept either format

        # Scientific notation
        qty = Quantity("1.23E+10")
        # Should maintain precision
        assert "12300000000" in str(qty) or "1.23E+10" in str(qty)

    def test_quantity_is_valid(self):
        """Test is_valid method always returns True."""
        # All quantities are valid (including zero and negative)
        assert Quantity(100).is_valid()
        assert Quantity(0).is_valid()
        assert Quantity(-100).is_valid()
        assert Quantity("0.0001").is_valid()
        assert Quantity("-999999").is_valid()

    def test_quantity_hash_consistency(self):
        """Test hash consistency for quantities."""
        qty1 = Quantity(100)
        qty2 = Quantity(100)

        # Same value should have same hash
        assert hash(qty1) == hash(qty2)

        # Should work as dictionary keys
        qty_dict = {qty1: "value1"}
        assert qty_dict[qty2] == "value1"

        # Different values have different hashes
        qty3 = Quantity(101)
        assert hash(qty1) != hash(qty3)

    def test_quantity_decimal_precision(self):
        """Test that decimal precision is maintained."""
        # High precision should be maintained
        qty = Quantity("123.123456789012345678901234567890")
        assert str(qty.value) == "123.123456789012345678901234567890"

        # Operations should maintain precision
        result = qty.multiply(1)
        assert result == qty

        result = qty.add(Quantity(0))
        assert result == qty
