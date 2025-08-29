"""
Comprehensive unit tests for Quantity value object.

Tests all public methods, operators, edge cases, mathematical operations,
comparisons, validation, immutability, and string representations.
"""

# Standard library imports
from decimal import Decimal

# Third-party imports
import pytest

# Local imports
from src.domain.value_objects.quantity import Quantity


class TestQuantityCreation:
    """Test Quantity creation and initialization."""

    def test_create_quantity_with_decimal(self):
        """Test creating quantity with Decimal value."""
        quantity = Quantity("100.50")
        assert quantity.value == Decimal("100.50")

    def test_create_quantity_with_float(self):
        """Test creating quantity with float value."""
        quantity = Quantity(100.50)
        assert quantity.value == Decimal("100.50")

    def test_create_quantity_with_int(self):
        """Test creating quantity with integer value."""
        quantity = Quantity(100)
        assert quantity.value == Decimal("100")

    def test_create_quantity_with_string(self):
        """Test creating quantity with string value."""
        quantity = Quantity("100.50")
        assert quantity.value == Decimal("100.50")

    def test_create_positive_quantity(self):
        """Test creating positive quantity (long position)."""
        quantity = Quantity(Decimal("100"))
        assert quantity.value == Decimal("100")
        assert quantity.is_long()
        assert not quantity.is_short()

    def test_create_negative_quantity(self):
        """Test creating negative quantity (short position)."""
        quantity = Quantity("-100")
        assert quantity.value == Decimal("-100")
        assert quantity.is_short()
        assert not quantity.is_long()

    def test_create_zero_quantity(self):
        """Test creating zero quantity (for position closing)."""
        quantity = Quantity("0")
        assert quantity.value == Decimal("0")
        assert quantity.is_zero()
        assert not quantity.is_long()
        assert not quantity.is_short()

    def test_extreme_precision(self):
        """Test quantity with extreme decimal precision."""
        quantity = Quantity("100.123456789123456789")
        assert quantity.value == Decimal("100.123456789123456789")


class TestQuantityProperties:
    """Test Quantity property accessors."""

    def test_value_property(self):
        """Test value property returns correct value."""
        quantity = Quantity("123.45")
        assert quantity.value == Decimal("123.45")


class TestQuantityValidation:
    """Test Quantity validation methods."""

    def test_is_valid(self):
        """Test is_valid method always returns True."""
        positive = Quantity("100")
        negative = Quantity("-100")
        zero = Quantity("0")

        assert positive.is_valid()
        assert negative.is_valid()
        assert zero.is_valid()

    def test_is_long(self):
        """Test is_long method for long positions."""
        long_position = Quantity("100")
        short_position = Quantity("-100")
        zero_position = Quantity("0")

        assert long_position.is_long()
        assert not short_position.is_long()
        assert not zero_position.is_long()

    def test_is_short(self):
        """Test is_short method for short positions."""
        long_position = Quantity("100")
        short_position = Quantity("-100")
        zero_position = Quantity("0")

        assert not long_position.is_short()
        assert short_position.is_short()
        assert not zero_position.is_short()

    def test_is_zero(self):
        """Test is_zero method."""
        zero = Quantity("0")
        positive = Quantity("100")
        negative = Quantity("-100")
        very_small = Decimal("0.00000001")

        assert zero.is_zero()
        assert not positive.is_zero()
        assert not negative.is_zero()
        assert not very_small.is_zero()


class TestQuantityArithmetic:
    """Test Quantity arithmetic operations."""

    def test_add_quantities(self):
        """Test adding two quantities."""
        qty1 = Quantity("100")
        qty2 = Quantity("50")

        result = qty1.add(qty2)
        assert isinstance(result, Quantity)
        assert result.value == Decimal("150")

    def test_add_positive_and_negative(self):
        """Test adding positive and negative quantities."""
        long_qty = Quantity("100")
        short_qty = Quantity("-30")

        result = long_qty.add(short_qty)
        assert result.value == Decimal("70")

    def test_add_resulting_in_negative(self):
        """Test adding quantities resulting in negative."""
        qty1 = Quantity("50")
        qty2 = Quantity("-100")

        result = qty1.add(qty2)
        assert result.value == Decimal("-50")
        assert result.is_short()

    def test_add_non_quantity_raises_error(self):
        """Test adding non-Quantity type raises error."""
        quantity = Quantity("100")

        with pytest.raises(TypeError, match="Cannot add Quantity and"):
            quantity.add(100)

    def test_subtract_quantities(self):
        """Test subtracting two quantities."""
        qty1 = Quantity("100")
        qty2 = Quantity("30")

        result = qty1.subtract(qty2)
        assert isinstance(result, Quantity)
        assert result.value == Decimal("70")

    def test_subtract_resulting_in_negative(self):
        """Test subtracting larger quantity results in negative."""
        qty1 = Quantity("30")
        qty2 = Quantity("100")

        result = qty1.subtract(qty2)
        assert result.value == Decimal("-70")
        assert result.is_short()

    def test_subtract_negative_quantity(self):
        """Test subtracting negative quantity (double negative)."""
        qty1 = Quantity("100")
        qty2 = Quantity("-50")

        result = qty1.subtract(qty2)
        assert result.value == Decimal("150")  # 100 - (-50) = 150

    def test_subtract_non_quantity_raises_error(self):
        """Test subtracting non-Quantity type raises error."""
        quantity = Quantity("100")

        with pytest.raises(TypeError, match="Cannot subtract .* from Quantity"):
            quantity.subtract(50)

    def test_multiply_by_decimal(self):
        """Test multiplying quantity by Decimal factor."""
        quantity = Quantity("100")

        result = quantity.multiply(Decimal("2.5"))
        assert isinstance(result, Quantity)
        assert result.value == Decimal("250")

    def test_multiply_by_int(self):
        """Test multiplying quantity by integer factor."""
        quantity = Quantity("100")

        result = quantity.multiply(3)
        assert result.value == Decimal("300")

    def test_multiply_by_float(self):
        """Test multiplying quantity by float factor."""
        quantity = Quantity("100")

        result = quantity.multiply(0.5)
        assert result.value == Decimal("50")

    def test_multiply_by_one(self):
        """Test multiplying by 1 returns exact same value."""
        quantity = Quantity("100.123456")

        result = quantity.multiply(1)
        assert result.value == Decimal("100.123456")

    def test_multiply_by_zero(self):
        """Test multiplying quantity by zero."""
        quantity = Quantity("100")

        result = quantity.multiply(0)
        assert result.value == Decimal("0")
        assert result.is_zero()

    def test_multiply_by_negative(self):
        """Test multiplying quantity by negative factor."""
        positive = Quantity("100")
        negative = Quantity("-100")

        # Positive * negative = negative
        result = positive.multiply(-2)
        assert result.value == Decimal("-200")
        assert result.is_short()

        # Negative * negative = positive
        result = negative.multiply(-2)
        assert result.value == Decimal("200")
        assert result.is_long()

    def test_divide_by_decimal(self):
        """Test dividing quantity by Decimal divisor."""
        quantity = Quantity("100")

        result = quantity.divide(Decimal("4"))
        assert isinstance(result, Quantity)
        assert result.value == Decimal("25")

    def test_divide_by_int(self):
        """Test dividing quantity by integer divisor."""
        quantity = Quantity("100")

        result = quantity.divide(2)
        assert result.value == Decimal("50")

    def test_divide_by_float(self):
        """Test dividing quantity by float divisor."""
        quantity = Quantity("100")

        result = quantity.divide(2.5)
        assert result.value == Decimal("40")

    def test_divide_by_zero_raises_error(self):
        """Test dividing by zero raises ValueError."""
        quantity = Quantity("100")

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            quantity.divide(0)

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            quantity.divide(Decimal("0"))

    def test_abs_method(self):
        """Test abs method returns absolute value."""
        positive = Quantity("100")
        negative = Quantity("-100")

        assert positive.abs().value == Decimal("100")
        assert negative.abs().value == Decimal("100")
        assert isinstance(negative.abs(), Quantity)


class TestQuantitySplitting:
    """Test Quantity splitting functionality."""

    def test_split_into_one_part(self):
        """Test splitting into one part returns original."""
        quantity = Quantity("100")

        parts = quantity.split(1)
        assert len(parts) == 1
        assert parts[0].value == Decimal("100")
        assert parts[0] == quantity

    def test_split_into_equal_parts(self):
        """Test splitting into equal parts."""
        quantity = Quantity("100")

        parts = quantity.split(4)
        assert len(parts) == 4

        # All parts should sum to original
        total = sum(part for part in parts)
        assert total.value == Decimal("100")

        # Parts should be equal (or nearly equal with remainder)
        for part in parts:
            assert part.value == Decimal("25")

    def test_split_with_remainder(self):
        """Test splitting with remainder goes to first part."""
        quantity = Quantity("100")

        parts = quantity.split(3)
        assert len(parts) == 3

        # First part gets remainder: 33 + 1 = 34
        assert parts[0].value == Decimal("34")
        assert parts[1].value == Decimal("33")
        assert parts[2].value == Decimal("33")

        # Total should equal original
        total = sum(part for part in parts)
        assert total.value == Decimal("100")

    def test_split_negative_quantity(self):
        """Test splitting negative quantity."""
        quantity = Quantity("-100")

        parts = quantity.split(4)
        assert len(parts) == 4

        for part in parts:
            assert part.value == Decimal("-25")

        total = sum(part for part in parts)
        assert total.value == Decimal("-100")

    def test_split_with_decimals(self):
        """Test splitting quantity with decimals."""
        quantity = Quantity("100.5")

        parts = quantity.split(2)
        assert len(parts) == 2

        # First part gets remainder
        assert parts[0].value == Decimal("50.5")
        assert parts[1].value == Decimal("50")

        total = sum(part for part in parts)
        assert total.value == Decimal("100.5")

    def test_split_invalid_num_parts(self):
        """Test splitting with invalid number of parts raises error."""
        quantity = Quantity("100")

        with pytest.raises(ValueError, match="Number of parts must be at least 1"):
            quantity.split(0)

        with pytest.raises(ValueError, match="Number of parts must be at least 1"):
            quantity.split(-1)


class TestQuantityRounding:
    """Test Quantity rounding operations."""

    def test_round_to_zero_decimal_places(self):
        """Test rounding to 0 decimal places."""
        quantity = Quantity("100.678")

        rounded = quantity.round(0)
        assert rounded.value == Decimal("100")  # Rounds down
        assert isinstance(rounded, Quantity)

    def test_round_to_two_decimal_places(self):
        """Test rounding to 2 decimal places."""
        quantity = Quantity("100.12678")

        rounded = quantity.round(2)
        assert rounded.value == Decimal("100.12")  # Rounds down

    def test_round_negative_quantity(self):
        """Test rounding negative quantity."""
        quantity = Quantity("-100.678")

        rounded = quantity.round(0)
        assert rounded.value == Decimal("-100")  # Rounds towards zero (up for negative)

    def test_round_down_behavior(self):
        """Test that rounding uses ROUND_DOWN."""
        quantity1 = Quantity("100.999")
        quantity2 = Quantity("100.111")

        assert quantity1.round(0).value == Decimal("100")
        assert quantity2.round(0).value == Decimal("100")

    def test_round_invalid_decimal_places(self):
        """Test rounding with negative decimal places raises error."""
        quantity = Quantity("100")

        with pytest.raises(ValueError, match="Decimal places must be non-negative"):
            quantity.round(-1)


class TestQuantityComparison:
    """Test Quantity comparison operations."""

    def test_equality_same_value(self):
        """Test equality for same value."""
        qty1 = Quantity("100")
        qty2 = Quantity("100")

        assert qty1 == qty2
        assert qty1 == qty2

    def test_equality_different_value(self):
        """Test inequality for different values."""
        qty1 = Quantity("100")
        qty2 = Quantity("200")

        assert qty1 != qty2
        assert qty1 != qty2

    def test_equality_with_non_quantity(self):
        """Test equality comparison with non-Quantity types."""
        quantity = Quantity("100")

        assert quantity != 100
        assert quantity != "100"
        assert quantity != None

    def test_less_than(self):
        """Test less than comparison."""
        qty1 = Quantity("50")
        qty2 = Quantity("100")

        assert qty1 < qty2
        assert not qty2 < qty1

    def test_less_than_with_negative(self):
        """Test less than with negative quantities."""
        negative = Quantity("-100")
        positive = Quantity("50")

        assert negative < positive
        assert not positive < negative

    def test_less_than_non_quantity_raises_error(self):
        """Test less than comparison with non-Quantity raises error."""
        quantity = Quantity("100")

        with pytest.raises(TypeError, match="Cannot compare Quantity and"):
            quantity < "50"

    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        qty1 = Quantity("50")
        qty2 = Quantity("100")
        qty3 = Decimal("100")

        assert qty1 <= qty2
        assert qty2 <= qty3
        assert not qty2 <= qty1

    def test_greater_than(self):
        """Test greater than comparison."""
        qty1 = Quantity("100")
        qty2 = Quantity("50")

        assert qty1 > qty2
        assert not qty2 > qty1

    def test_greater_than_non_quantity_raises_error(self):
        """Test greater than comparison with non-Quantity raises error."""
        quantity = Quantity("100")

        with pytest.raises(TypeError, match="Cannot compare Quantity and"):
            quantity > "150"

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        qty1 = Quantity("100")
        qty2 = Quantity("50")
        qty3 = Decimal("100")

        assert qty1 >= qty2
        assert qty1 >= qty3
        assert not qty2 >= qty1


class TestQuantityOperators:
    """Test Quantity special operators."""

    def test_negation_operator(self):
        """Test negation operator."""
        positive = Quantity("100")
        negative = -positive

        assert negative.value == Decimal("-100")
        assert negative.is_short()
        assert isinstance(negative, Quantity)

        # Double negation
        double_neg = -negative
        assert double_neg.value == Decimal("100")
        assert double_neg.is_long()

    def test_abs_operator(self):
        """Test absolute value operator."""
        positive = Quantity("100")
        negative = Quantity("-100")

        assert abs(positive).value == Decimal("100")
        assert abs(negative).value == Decimal("100")
        assert isinstance(abs(negative), Quantity)

    def test_hash(self):
        """Test hash for use in sets and dicts."""
        qty1 = Quantity("100")
        qty2 = Quantity("100")
        qty3 = Decimal("200")

        # Equal quantities have same hash
        assert hash(qty1) == hash(qty2)

        # Can be used in sets
        qty_set = {qty1, qty2, qty3}
        assert len(qty_set) == 2  # qty1 and qty2 are equal


class TestQuantityFormatting:
    """Test Quantity formatting and display."""

    def test_str_representation_integer(self):
        """Test string representation for integer values."""
        quantity = Quantity("100")
        assert str(quantity) == "100"

    def test_str_representation_decimal(self):
        """Test string representation for decimal values."""
        quantity = Quantity("100.50")
        assert str(quantity) == "100.5"

    def test_str_representation_negative(self):
        """Test string representation for negative values."""
        quantity = Quantity("-100")
        assert str(quantity) == "-100"

    def test_repr_representation(self):
        """Test repr representation."""
        quantity = Quantity("100.50")
        assert repr(quantity) == "Quantity(100.50)"


class TestQuantityEdgeCases:
    """Test Quantity edge cases and extreme values."""

    def test_very_large_quantity(self):
        """Test handling very large quantities."""
        large_qty = Quantity("999999999999999999")
        assert large_qty.value == Decimal("999999999999999999")
        assert large_qty.is_valid()

    def test_very_small_quantity(self):
        """Test handling very small quantities."""
        small_qty = Quantity("0.00000001")
        assert small_qty.value == Decimal("0.00000001")
        assert small_qty.is_valid()
        assert not small_qty.is_zero()

    def test_many_decimal_places(self):
        """Test quantity with many decimal places."""
        quantity = Quantity("100.123456789123456789")
        assert quantity.value == Decimal("100.123456789123456789")

    def test_scientific_notation(self):
        """Test handling scientific notation."""
        quantity = Quantity("1.5E+3")
        assert quantity.value == Decimal("1500")

        quantity = Quantity("1.5E-3")
        assert quantity.value == Decimal("0.0015")

    def test_immutability(self):
        """Test that Quantity is immutable."""
        quantity = Quantity("100")
        original_value = quantity

        # Operations return new objects
        new_qty = quantity.add(Quantity("50"))
        assert quantity == original_value
        assert new_qty.value == Decimal("150")

        # Properties can't be modified
        with pytest.raises(AttributeError):
            quantity.value = Decimal("200")

    def test_chain_operations(self):
        """Test chaining multiple operations."""
        quantity = Quantity("100")

        result = quantity.add(Quantity("50")).multiply(2).divide(3)
        assert result.value == Decimal("100")

        # Original unchanged
        assert quantity.value == Decimal("100")

    def test_zero_arithmetic(self):
        """Test arithmetic with zero quantities."""
        zero = Quantity("0")
        hundred = Quantity("100")

        # Adding zero
        assert hundred.add(zero).value == Decimal("100")
        assert zero.add(hundred).value == Decimal("100")

        # Subtracting zero
        assert hundred.subtract(zero).value == Decimal("100")
        assert zero.subtract(hundred).value == Decimal("-100")

        # Multiplying by zero
        assert hundred.multiply(0).value == Decimal("0")
        assert zero.multiply(100).value == Decimal("0")

    def test_precision_preservation(self):
        """Test that precision is preserved in operations."""
        qty1 = Quantity("0.001")
        qty2 = Quantity("0.002")

        result = qty1.add(qty2)
        assert result.value == Decimal("0.003")

        # Multiplication preserves precision
        result = qty1.multiply(Decimal("3"))
        assert result.value == Decimal("0.003")

    def test_position_transitions(self):
        """Test transitions between long, short, and flat positions."""
        long_position = Quantity("100")
        short_position = Quantity("-50")

        # Long + Short can result in reduced long
        result = long_position.add(short_position)
        assert result.value == Decimal("50")
        assert result.is_long()

        # Long + larger Short results in net short
        large_short = Quantity("-150")
        result = long_position.add(large_short)
        assert result.value == Decimal("-50")
        assert result.is_short()

        # Exact offsetting results in zero
        offsetting = Quantity("-100")
        result = long_position.add(offsetting)
        assert result.value == Decimal("0")
        assert result.is_zero()
