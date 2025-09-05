"""Test to ensure Quantity cannot be compared with Money (critical bug fix)"""

from decimal import Decimal

import pytest

from src.domain.value_objects import Money, Quantity


class TestQuantityMoneyComparison:
    """Test that Quantity-Money comparisons are properly prevented."""

    def test_quantity_cannot_compare_less_than_money(self):
        """Quantity should not be comparable with Money using <"""
        quantity = Quantity(Decimal("100"))
        money = Money(Decimal("500"))

        with pytest.raises(TypeError, match="Cannot compare Quantity and"):
            _ = quantity < money

    def test_quantity_cannot_compare_less_equal_money(self):
        """Quantity should not be comparable with Money using <="""
        quantity = Quantity(Decimal("100"))
        money = Money(Decimal("500"))

        with pytest.raises(TypeError, match="Cannot compare Quantity and"):
            _ = quantity <= money

    def test_quantity_cannot_compare_greater_than_money(self):
        """Quantity should not be comparable with Money using >"""
        quantity = Quantity(Decimal("100"))
        money = Money(Decimal("500"))

        with pytest.raises(TypeError, match="Cannot compare Quantity and"):
            _ = quantity > money

    def test_quantity_cannot_compare_greater_equal_money(self):
        """Quantity should not be comparable with Money using >="""
        quantity = Quantity(Decimal("100"))
        money = Money(Decimal("500"))

        with pytest.raises(TypeError, match="Cannot compare Quantity and"):
            _ = quantity >= money

    def test_quantity_can_compare_with_quantity(self):
        """Quantity should be comparable with other Quantity instances"""
        q1 = Quantity(Decimal("100"))
        q2 = Quantity(Decimal("200"))

        assert q1 < q2
        assert q1 <= q2
        assert q2 > q1
        assert q2 >= q1
        assert q1 == q1
        assert q1 != q2

    def test_quantity_can_compare_with_numbers(self):
        """Quantity should be comparable with numeric values"""
        quantity = Quantity(Decimal("100"))

        assert quantity < 200
        assert quantity <= 100
        assert quantity > 50
        assert quantity >= 100
        assert quantity < Decimal("150")
        assert quantity > 50.5

    def test_quantity_cannot_divide_by_money(self):
        """Quantity should not be divisible by Money"""
        quantity = Quantity(Decimal("100"))
        money = Money(Decimal("10"))

        # The divide method doesn't accept Money, it will raise an error
        with pytest.raises(Exception):  # Could be InvalidOperation or TypeError
            _ = quantity / money
