"""
Ultra-Comprehensive Edge Case Tests for Domain Value Objects
==========================================================

Additional edge case tests to ensure maximum coverage of all value objects
with focus on boundary conditions, error handling, and precision.
"""

from decimal import Decimal

import pytest

from src.domain.value_objects import Money, Price, Quantity


class TestMoneyEdgeCases:
    """Comprehensive edge case tests for Money value object."""

    def test_money_extreme_precision(self):
        """Test Money with extreme decimal precision."""
        precise_amount = Decimal("123.456789012345678901234567890")
        money = Money(precise_amount)
        assert money.amount == precise_amount

    def test_money_very_large_amounts(self):
        """Test Money with very large amounts."""
        large_amount = Decimal("999999999999999999999.99")
        money = Money(large_amount)
        assert money.amount == large_amount

    def test_money_scientific_notation(self):
        """Test Money with scientific notation input."""
        scientific = Decimal("1.5E+6")  # 1,500,000
        money = Money(scientific)
        assert money.amount == Decimal("1500000")

    def test_money_zero_variations(self):
        """Test Money with different zero representations."""
        zero_variations = [
            Decimal("0"),
            Decimal("0.0"),
            Decimal("0.00000"),
            Decimal("-0"),
        ]

        for zero in zero_variations:
            money = Money(zero)
            assert money.amount == Decimal("0")

    def test_money_arithmetic_edge_cases(self):
        """Test Money arithmetic edge cases."""
        m1 = Money(Decimal("0.01"))
        m2 = Money(Decimal("0.02"))

        # Very small operations
        result = m1 + m2
        assert result.amount == Decimal("0.03")

        # Subtraction resulting in negative
        result = m1 - m2
        assert result.amount == Decimal("-0.01")

    def test_money_comparison_precision(self):
        """Test Money comparison with precision edge cases."""
        m1 = Money(Decimal("123.456789012345678901"))
        m2 = Money(Decimal("123.456789012345678902"))

        assert m1 != m2
        assert m1 < m2

    def test_money_hash_consistency(self):
        """Test Money hash consistency."""
        amount = Decimal("100.50")
        m1 = Money(amount)
        m2 = Money(amount)

        assert hash(m1) == hash(m2)
        assert {m1: "test"}[m2] == "test"  # Can be used as dict key

    def test_money_string_edge_cases(self):
        """Test Money string representation edge cases."""
        # Very large number - should format with currency and commas
        large_money = Money(Decimal("123456789.99"))
        str_repr = str(large_money)
        assert str_repr == "$123,456,789.99"

        # Very small number - should format with currency and default precision (2 decimal places)
        small_money = Money(Decimal("0.000001"))
        str_repr = str(small_money)
        assert str_repr == "$0.00"  # Default 2 decimal places rounds very small amounts

        # Test formatting with higher precision for small numbers
        str_repr_precise = small_money.format(decimal_places=6)
        assert str_repr_precise == "$0.000001"


class TestPriceEdgeCases:
    """Comprehensive edge case tests for Price value object."""

    def test_price_boundary_validation(self):
        """Test Price validation at boundaries."""
        # Just above zero should work
        Price(Decimal("0.000001"))

        # Exactly zero should fail
        with pytest.raises(ValueError):
            Price(Decimal("0"))

        # Negative should fail
        with pytest.raises(ValueError):
            Price(Decimal("-0.01"))

    def test_price_extreme_values(self):
        """Test Price with extreme but valid values."""
        # Very high price (like BRK.A)
        high_price = Price(Decimal("500000"))
        assert high_price.value == Decimal("500000")

        # Very low price (penny stocks)
        low_price = Price(Decimal("0.0001"))
        assert low_price.value == Decimal("0.0001")

    def test_price_precision_maintained(self):
        """Test that Price maintains full decimal precision."""
        precise_price = Price(Decimal("123.456789123456789"))
        assert precise_price.value == Decimal("123.456789123456789")

    def test_price_comparison_edge_cases(self):
        """Test Price comparison edge cases."""
        p1 = Price(Decimal("100.000000001"))
        p2 = Price(Decimal("100.000000002"))

        assert p1 < p2
        assert p1 != p2

    def test_price_arithmetic_operations(self):
        """Test Price arithmetic edge cases."""
        p1 = Price(Decimal("50.25"))
        p2 = Price(Decimal("25.75"))

        # Addition
        result = p1 + p2
        assert result.value == Decimal("76.00")

        # Subtraction (result must still be positive)
        result = p1 - Price(Decimal("10.25"))
        assert result.value == Decimal("40.00")

    def test_price_invalid_operations(self):
        """Test Price invalid operations."""
        p = Price(Decimal("100"))

        # Subtracting too much should fail if result would be negative
        with pytest.raises(ValueError):
            p - Price(Decimal("150"))


class TestQuantityEdgeCases:
    """Comprehensive edge case tests for Quantity value object."""

    def test_quantity_zero_allowed(self):
        """Test that Quantity allows zero."""
        q = Quantity(Decimal("0"))
        assert q.value == Decimal("0")

    def test_quantity_negative_values(self):
        """Test Quantity with negative values (short positions)."""
        q = Quantity(Decimal("-100"))
        assert q.value == Decimal("-100")

    def test_quantity_fractional_shares(self):
        """Test Quantity with fractional shares."""
        fractional_quantities = [
            Decimal("0.5"),
            Decimal("0.001"),
            Decimal("123.456789"),
            Decimal("-0.25"),
        ]

        for qty in fractional_quantities:
            q = Quantity(qty)
            assert q.value == qty

    def test_quantity_extreme_values(self):
        """Test Quantity with extreme values."""
        # Very large position
        large_q = Quantity(Decimal("999999999"))
        assert large_q.value == Decimal("999999999")

        # Very small fractional
        small_q = Quantity(Decimal("0.000000001"))
        assert small_q.value == Decimal("0.000000001")

    def test_quantity_precision_arithmetic(self):
        """Test Quantity arithmetic maintains precision."""
        q1 = Quantity(Decimal("123.456789"))
        q2 = Quantity(Decimal("456.789123"))

        result = q1 + q2
        expected = Decimal("123.456789") + Decimal("456.789123")
        assert result.value == expected

    def test_quantity_comparison_precision(self):
        """Test Quantity comparison with high precision."""
        q1 = Quantity(Decimal("100.000000000001"))
        q2 = Quantity(Decimal("100.000000000002"))

        assert q1 < q2
        assert q1 != q2


class TestValueObjectEdgeCases:
    """Test ValueObject edge cases and error conditions."""

    def test_value_object_equality(self):
        """Test BaseValueObject equality implementation."""
        m1 = Money(Decimal("100"))
        m2 = Money(Decimal("100"))
        m3 = Money(Decimal("200"))

        assert m1 == m2
        assert m1 != m3
        assert m1 != "not a money object"
        assert m1 != 100  # Different type

    def test_value_object_hash_immutability(self):
        """Test that value objects are hashable and immutable."""
        money = Money(Decimal("100"))
        price = Price(Decimal("50"))
        quantity = Quantity(Decimal("10"))

        # Should be hashable
        value_set = {money, price, quantity}
        assert len(value_set) == 3

        # Should work as dict keys
        value_dict = {money: "money", price: "price", quantity: "quantity"}
        assert value_dict[money] == "money"

    def test_value_object_repr_consistency(self):
        """Test value object __repr__ consistency."""
        money = Money(Decimal("123.45"))
        repr_str = repr(money)

        # Should contain class name and value
        assert "Money" in repr_str
        assert "123.45" in repr_str

    def test_value_object_type_safety(self):
        """Test value object type safety."""
        money = Money(Decimal("100"))
        price = Price(Decimal("50"))

        # Different types should not be equal even with same numeric value
        assert money != Price(Decimal("100"))
        assert price != Quantity(Decimal("50"))


class TestValueObjectIntegration:
    """Test value objects working together in complex scenarios."""

    def test_mixed_precision_calculations(self):
        """Test calculations with mixed precision value objects."""
        high_precision_price = Price(Decimal("123.456789012345"))
        high_precision_qty = Quantity(Decimal("987.654321098765"))

        # Calculate position value
        position_value = Money(high_precision_price.value * high_precision_qty.value)

        # Should maintain precision
        expected = Decimal("123.456789012345") * Decimal("987.654321098765")
        assert position_value.amount == expected

    def test_boundary_condition_calculations(self):
        """Test calculations at boundary conditions."""
        min_price = Price(Decimal("0.0001"))
        max_qty = Quantity(Decimal("999999999"))

        # Large position value calculation
        position_value = Money(min_price.value * max_qty.value)
        expected = Decimal("0.0001") * Decimal("999999999")
        assert position_value.amount == expected

    def test_edge_case_financial_calculations(self):
        """Test edge cases in financial calculations."""
        # Very small price movements
        entry_price = Price(Decimal("100.0000"))
        exit_price = Price(Decimal("100.0001"))
        quantity = Quantity(Decimal("1000000"))  # Large quantity

        # P&L calculation with tiny price movement but large quantity
        pnl = Money(quantity.value * (exit_price.value - entry_price.value))
        expected = Decimal("1000000") * Decimal("0.0001")  # Should be 100
        assert pnl.amount == expected

    def test_rounding_and_precision_consistency(self):
        """Test that rounding and precision are handled consistently."""
        # Division that might cause rounding
        total_cost = Money(Decimal("100.00"))
        shares = Quantity(Decimal("3"))

        # Average price calculation
        avg_price = total_cost.amount / shares.value
        price_per_share = Price(avg_price)

        # Should handle the recurring decimal correctly
        assert price_per_share.value == Decimal("100.00") / Decimal("3")

    def test_zero_handling_across_types(self):
        """Test zero value handling across all value object types."""
        zero_money = Money(Decimal("0"))
        zero_quantity = Quantity(Decimal("0"))

        # Zero quantity with any price should give zero position value
        any_price = Price(Decimal("100"))
        position_value = Money(zero_quantity.value * any_price.value)
        assert position_value == zero_money

    def test_negative_value_interactions(self):
        """Test interactions involving negative values."""
        positive_price = Price(Decimal("100"))
        negative_quantity = Quantity(Decimal("-50"))  # Short position

        # Position value should be positive (absolute value used)
        position_value = Money(abs(negative_quantity.value) * positive_price.value)
        assert position_value.amount == Decimal("5000")

        # P&L calculation for short position
        entry_price = Price(Decimal("100"))
        exit_price = Price(Decimal("90"))  # Price went down, profit for short
        pnl = Money(abs(negative_quantity.value) * (entry_price.value - exit_price.value))
        assert pnl.amount == Decimal("500")  # Profit
