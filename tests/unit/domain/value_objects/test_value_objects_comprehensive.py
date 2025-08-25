"""Comprehensive tests for domain value objects.

This test suite provides extensive coverage for all value objects,
including Money, Price, Quantity, Symbol, and their edge cases.
"""

import json
from decimal import Decimal

import pytest

from src.domain.value_objects.money import Money
from src.domain.value_objects.price import Price
from src.domain.value_objects.quantity import Quantity
from src.domain.value_objects.symbol import Symbol


class TestMoneyValueObject:
    """Comprehensive tests for Money value object."""

    def test_money_creation_valid(self):
        """Test creating Money with valid values."""
        money = Money(Decimal("100.50"), "USD")
        assert money.amount == Decimal("100.50")
        assert money.currency == "USD"

    def test_money_creation_from_string(self):
        """Test creating Money from string amount."""
        money = Money("100.50", "USD")
        assert money.amount == Decimal("100.50")

    def test_money_creation_from_int(self):
        """Test creating Money from integer."""
        money = Money(100, "USD")
        assert money.amount == Decimal("100")

    def test_money_creation_from_float(self):
        """Test creating Money from float."""
        money = Money(100.50, "USD")
        assert money.amount == Decimal("100.50")

    def test_money_negative_amount(self):
        """Test Money with negative amount."""
        money = Money(Decimal("-100.50"), "USD")
        assert money.amount == Decimal("-100.50")
        assert money.is_negative()

    def test_money_zero_amount(self):
        """Test Money with zero amount."""
        money = Money(Decimal("0"), "USD")
        assert money.amount == Decimal("0")
        assert money.is_zero()

    def test_money_invalid_currency(self):
        """Test Money with invalid currency."""
        with pytest.raises(ValueError, match="Invalid currency"):
            Money(Decimal("100"), "INVALID")

    def test_money_empty_currency(self):
        """Test Money with empty currency."""
        with pytest.raises(ValueError, match="Currency is required"):
            Money(Decimal("100"), "")

    def test_money_addition_same_currency(self):
        """Test adding Money with same currency."""
        money1 = Money(Decimal("100.50"), "USD")
        money2 = Money(Decimal("50.25"), "USD")
        result = money1 + money2
        assert result.amount == Decimal("150.75")
        assert result.currency == "USD"

    def test_money_addition_different_currency(self):
        """Test adding Money with different currencies."""
        money1 = Money(Decimal("100"), "USD")
        money2 = Money(Decimal("100"), "EUR")
        with pytest.raises(ValueError, match="Cannot add money with different currencies"):
            money1 + money2

    def test_money_subtraction_same_currency(self):
        """Test subtracting Money with same currency."""
        money1 = Money(Decimal("100.50"), "USD")
        money2 = Money(Decimal("50.25"), "USD")
        result = money1 - money2
        assert result.amount == Decimal("50.25")

    def test_money_subtraction_resulting_negative(self):
        """Test subtraction resulting in negative amount."""
        money1 = Money(Decimal("50"), "USD")
        money2 = Money(Decimal("100"), "USD")
        result = money1 - money2
        assert result.amount == Decimal("-50")
        assert result.is_negative()

    def test_money_multiplication_by_scalar(self):
        """Test multiplying Money by scalar."""
        money = Money(Decimal("100"), "USD")
        result = money * Decimal("1.5")
        assert result.amount == Decimal("150")
        assert result.currency == "USD"

    def test_money_multiplication_by_negative(self):
        """Test multiplying Money by negative scalar."""
        money = Money(Decimal("100"), "USD")
        result = money * Decimal("-1")
        assert result.amount == Decimal("-100")
        assert result.is_negative()

    def test_money_division_by_scalar(self):
        """Test dividing Money by scalar."""
        money = Money(Decimal("100"), "USD")
        result = money / Decimal("2")
        assert result.amount == Decimal("50")

    def test_money_division_by_zero(self):
        """Test dividing Money by zero."""
        money = Money(Decimal("100"), "USD")
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            money / Decimal("0")

    def test_money_comparison_equal(self):
        """Test Money equality comparison."""
        money1 = Money(Decimal("100.50"), "USD")
        money2 = Money(Decimal("100.50"), "USD")
        money3 = Money(Decimal("100.50"), "EUR")

        assert money1 == money2
        assert money1 != money3  # Different currency

    def test_money_comparison_ordering(self):
        """Test Money ordering comparisons."""
        money1 = Money(Decimal("50"), "USD")
        money2 = Money(Decimal("100"), "USD")

        assert money1 < money2
        assert money2 > money1
        assert money1 <= money2
        assert money2 >= money1

    def test_money_comparison_different_currency(self):
        """Test comparing Money with different currencies."""
        money1 = Money(Decimal("100"), "USD")
        money2 = Money(Decimal("100"), "EUR")

        with pytest.raises(ValueError, match="Cannot compare money with different currencies"):
            money1 < money2

    def test_money_absolute_value(self):
        """Test absolute value of Money."""
        money = Money(Decimal("-100.50"), "USD")
        result = abs(money)
        assert result.amount == Decimal("100.50")

    def test_money_negation(self):
        """Test negating Money."""
        money = Money(Decimal("100.50"), "USD")
        result = -money
        assert result.amount == Decimal("-100.50")

    def test_money_rounding(self):
        """Test rounding Money."""
        money = Money(Decimal("100.5555"), "USD")
        rounded = money.round(2)
        assert rounded.amount == Decimal("100.56")

    def test_money_floor(self):
        """Test floor operation on Money."""
        money = Money(Decimal("100.99"), "USD")
        result = money.floor()
        assert result.amount == Decimal("100")

    def test_money_ceiling(self):
        """Test ceiling operation on Money."""
        money = Money(Decimal("100.01"), "USD")
        result = money.ceiling()
        assert result.amount == Decimal("101")

    def test_money_string_representation(self):
        """Test string representation of Money."""
        money = Money(Decimal("1234.56"), "USD")
        assert str(money) == "$1,234.56"

        money_eur = Money(Decimal("1234.56"), "EUR")
        assert str(money_eur) == "€1,234.56"

    def test_money_repr(self):
        """Test repr of Money."""
        money = Money(Decimal("100.50"), "USD")
        assert repr(money) == "Money(amount=Decimal('100.50'), currency='USD')"

    def test_money_to_dict(self):
        """Test converting Money to dictionary."""
        money = Money(Decimal("100.50"), "USD")
        data = money.to_dict()
        assert data == {"amount": "100.50", "currency": "USD"}

    def test_money_from_dict(self):
        """Test creating Money from dictionary."""
        data = {"amount": "100.50", "currency": "USD"}
        money = Money.from_dict(data)
        assert money.amount == Decimal("100.50")
        assert money.currency == "USD"

    def test_money_json_serialization(self):
        """Test JSON serialization of Money."""
        money = Money(Decimal("100.50"), "USD")
        json_str = json.dumps(money.to_dict())
        data = json.loads(json_str)
        restored = Money.from_dict(data)
        assert restored == money


class TestPriceValueObject:
    """Comprehensive tests for Price value object."""

    def test_price_creation_valid(self):
        """Test creating Price with valid value."""
        price = Price(Decimal("150.50"))
        assert price.value == Decimal("150.50")

    def test_price_creation_from_string(self):
        """Test creating Price from string."""
        price = Price("150.50")
        assert price.value == Decimal("150.50")

    def test_price_creation_from_int(self):
        """Test creating Price from integer."""
        price = Price(150)
        assert price.value == Decimal("150")

    def test_price_creation_from_float(self):
        """Test creating Price from float."""
        price = Price(150.50)
        assert price.value == Decimal("150.50")

    def test_price_negative_value(self):
        """Test Price with negative value."""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            Price(Decimal("-10"))

    def test_price_zero_value(self):
        """Test Price with zero value."""
        price = Price(Decimal("0"))
        assert price.value == Decimal("0")
        assert price.is_zero()

    def test_price_very_small_value(self):
        """Test Price with very small value."""
        price = Price(Decimal("0.0001"))
        assert price.value == Decimal("0.0001")

    def test_price_very_large_value(self):
        """Test Price with very large value."""
        price = Price(Decimal("1000000000"))
        assert price.value == Decimal("1000000000")

    def test_price_addition(self):
        """Test adding prices."""
        price1 = Price(Decimal("100.50"))
        price2 = Price(Decimal("50.25"))
        result = price1 + price2
        assert result.value == Decimal("150.75")

    def test_price_subtraction(self):
        """Test subtracting prices."""
        price1 = Price(Decimal("100.50"))
        price2 = Price(Decimal("50.25"))
        result = price1 - price2
        assert result.value == Decimal("50.25")

    def test_price_subtraction_negative_result(self):
        """Test subtraction resulting in negative price."""
        price1 = Price(Decimal("50"))
        price2 = Price(Decimal("100"))
        with pytest.raises(ValueError, match="Price cannot be negative"):
            price1 - price2

    def test_price_multiplication(self):
        """Test multiplying price by scalar."""
        price = Price(Decimal("100"))
        result = price * Decimal("1.5")
        assert result.value == Decimal("150")

    def test_price_multiplication_negative(self):
        """Test multiplying price by negative scalar."""
        price = Price(Decimal("100"))
        with pytest.raises(ValueError, match="Price cannot be negative"):
            price * Decimal("-1")

    def test_price_division(self):
        """Test dividing price by scalar."""
        price = Price(Decimal("100"))
        result = price / Decimal("2")
        assert result.value == Decimal("50")

    def test_price_division_by_zero(self):
        """Test dividing price by zero."""
        price = Price(Decimal("100"))
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            price / Decimal("0")

    def test_price_comparison(self):
        """Test price comparisons."""
        price1 = Price(Decimal("50"))
        price2 = Price(Decimal("100"))
        price3 = Price(Decimal("100"))

        assert price1 < price2
        assert price2 > price1
        assert price2 == price3
        assert price1 != price2
        assert price1 <= price2
        assert price2 >= price1

    def test_price_percentage_change(self):
        """Test calculating percentage change."""
        old_price = Price(Decimal("100"))
        new_price = Price(Decimal("110"))

        change = new_price.percentage_change(old_price)
        assert change == Decimal("10")  # 10% increase

    def test_price_percentage_change_decrease(self):
        """Test percentage change for decrease."""
        old_price = Price(Decimal("100"))
        new_price = Price(Decimal("90"))

        change = new_price.percentage_change(old_price)
        assert change == Decimal("-10")  # 10% decrease

    def test_price_percentage_change_from_zero(self):
        """Test percentage change from zero."""
        old_price = Price(Decimal("0"))
        new_price = Price(Decimal("100"))

        with pytest.raises(ValueError, match="Cannot calculate percentage change from zero"):
            new_price.percentage_change(old_price)

    def test_price_rounding(self):
        """Test rounding prices."""
        price = Price(Decimal("100.5555"))

        rounded_2 = price.round(2)
        assert rounded_2.value == Decimal("100.56")

        rounded_0 = price.round(0)
        assert rounded_0.value == Decimal("101")

    def test_price_to_money(self):
        """Test converting price to money."""
        price = Price(Decimal("100.50"))
        money = price.to_money("USD")

        assert isinstance(money, Money)
        assert money.amount == Decimal("100.50")
        assert money.currency == "USD"

    def test_price_string_representation(self):
        """Test string representation of Price."""
        price = Price(Decimal("1234.56"))
        assert str(price) == "1234.56"

    def test_price_format(self):
        """Test formatting price."""
        price = Price(Decimal("1234.56"))
        assert price.format() == "$1,234.56"
        assert price.format(symbol="€") == "€1,234.56"
        assert price.format(decimal_places=0) == "$1,235"


class TestQuantityValueObject:
    """Comprehensive tests for Quantity value object."""

    def test_quantity_creation_valid(self):
        """Test creating Quantity with valid value."""
        qty = Quantity(100)
        assert qty.value == 100

    def test_quantity_creation_negative(self):
        """Test creating Quantity with negative value (for short positions)."""
        qty = Quantity(-100)
        assert qty.value == -100
        assert qty.is_short()

    def test_quantity_creation_zero(self):
        """Test creating Quantity with zero."""
        qty = Quantity(0)
        assert qty.value == 0
        assert qty.is_zero()

    def test_quantity_creation_from_float(self):
        """Test creating Quantity from float (should round)."""
        qty = Quantity(100.7)
        assert qty.value == 101  # Rounds to nearest integer

    def test_quantity_creation_from_decimal(self):
        """Test creating Quantity from Decimal."""
        qty = Quantity(Decimal("100"))
        assert qty.value == 100

    def test_quantity_is_long(self):
        """Test identifying long positions."""
        qty = Quantity(100)
        assert qty.is_long() is True
        assert qty.is_short() is False

    def test_quantity_is_short(self):
        """Test identifying short positions."""
        qty = Quantity(-100)
        assert qty.is_short() is True
        assert qty.is_long() is False

    def test_quantity_absolute(self):
        """Test absolute value of Quantity."""
        qty = Quantity(-100)
        assert qty.absolute() == 100
        assert abs(qty).value == 100

    def test_quantity_addition(self):
        """Test adding quantities."""
        qty1 = Quantity(100)
        qty2 = Quantity(50)
        result = qty1 + qty2
        assert result.value == 150

    def test_quantity_addition_long_short(self):
        """Test adding long and short quantities."""
        qty1 = Quantity(100)  # Long
        qty2 = Quantity(-30)  # Short
        result = qty1 + qty2
        assert result.value == 70

    def test_quantity_subtraction(self):
        """Test subtracting quantities."""
        qty1 = Quantity(100)
        qty2 = Quantity(30)
        result = qty1 - qty2
        assert result.value == 70

    def test_quantity_multiplication(self):
        """Test multiplying quantity."""
        qty = Quantity(100)
        result = qty * 2
        assert result.value == 200

    def test_quantity_division(self):
        """Test dividing quantity."""
        qty = Quantity(100)
        result = qty / 2
        assert result.value == 50

    def test_quantity_division_with_remainder(self):
        """Test dividing quantity with remainder (should round)."""
        qty = Quantity(100)
        result = qty / 3
        assert result.value == 33  # Rounds down

    def test_quantity_division_by_zero(self):
        """Test dividing quantity by zero."""
        qty = Quantity(100)
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            qty / 0

    def test_quantity_comparison(self):
        """Test quantity comparisons."""
        qty1 = Quantity(50)
        qty2 = Quantity(100)
        qty3 = Quantity(100)

        assert qty1 < qty2
        assert qty2 > qty1
        assert qty2 == qty3
        assert qty1 != qty2

    def test_quantity_to_lots(self):
        """Test converting quantity to lots."""
        qty = Quantity(250)

        lots = qty.to_lots(100)  # 100 shares per lot
        assert lots == 2  # 2 full lots

        remainder = qty.lot_remainder(100)
        assert remainder == 50  # 50 shares remaining

    def test_quantity_split(self):
        """Test splitting quantity."""
        qty = Quantity(100)

        parts = qty.split(4)  # Split into 4 parts
        assert len(parts) == 4
        assert all(p.value == 25 for p in parts)

    def test_quantity_split_with_remainder(self):
        """Test splitting quantity with remainder."""
        qty = Quantity(100)

        parts = qty.split(3)  # Split into 3 parts
        assert len(parts) == 3
        assert parts[0].value == 34  # First part gets remainder
        assert parts[1].value == 33
        assert parts[2].value == 33

    def test_quantity_string_representation(self):
        """Test string representation of Quantity."""
        qty = Quantity(100)
        assert str(qty) == "100"

        qty_short = Quantity(-100)
        assert str(qty_short) == "-100"


class TestSymbolValueObject:
    """Comprehensive tests for Symbol value object."""

    def test_symbol_creation_valid(self):
        """Test creating Symbol with valid value."""
        symbol = Symbol("AAPL")
        assert symbol.value == "AAPL"

    def test_symbol_creation_lowercase(self):
        """Test Symbol converts to uppercase."""
        symbol = Symbol("aapl")
        assert symbol.value == "AAPL"

    def test_symbol_creation_with_spaces(self):
        """Test Symbol strips spaces."""
        symbol = Symbol("  AAPL  ")
        assert symbol.value == "AAPL"

    def test_symbol_creation_empty(self):
        """Test creating Symbol with empty string."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            Symbol("")

    def test_symbol_creation_with_numbers(self):
        """Test Symbol with numbers."""
        symbol = Symbol("BRK.B")
        assert symbol.value == "BRK.B"

    def test_symbol_creation_special_chars(self):
        """Test Symbol with special characters."""
        symbol = Symbol("BRK-B")
        assert symbol.value == "BRK-B"

    def test_symbol_validation_too_long(self):
        """Test Symbol length validation."""
        with pytest.raises(ValueError, match="Symbol too long"):
            Symbol("A" * 20)  # Too long

    def test_symbol_validation_invalid_chars(self):
        """Test Symbol with invalid characters."""
        with pytest.raises(ValueError, match="Invalid symbol"):
            Symbol("AAPL$#@")

    def test_symbol_equality(self):
        """Test Symbol equality."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("AAPL")
        symbol3 = Symbol("aapl")  # Same after uppercase
        symbol4 = Symbol("MSFT")

        assert symbol1 == symbol2
        assert symbol1 == symbol3
        assert symbol1 != symbol4

    def test_symbol_hashing(self):
        """Test Symbol can be used as dict key."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("AAPL")

        data = {symbol1: "value1"}
        assert data[symbol2] == "value1"  # Same hash

    def test_symbol_ordering(self):
        """Test Symbol ordering."""
        symbols = [Symbol("MSFT"), Symbol("AAPL"), Symbol("GOOGL")]
        sorted_symbols = sorted(symbols)

        assert sorted_symbols[0].value == "AAPL"
        assert sorted_symbols[1].value == "GOOGL"
        assert sorted_symbols[2].value == "MSFT"

    def test_symbol_is_option(self):
        """Test identifying option symbols."""
        stock = Symbol("AAPL")
        option = Symbol("AAPL210115C00140000")  # Option format

        assert stock.is_option() is False
        assert option.is_option() is True

    def test_symbol_exchange_suffix(self):
        """Test Symbol with exchange suffix."""
        symbol = Symbol("AAPL.NASDAQ")
        assert symbol.base_symbol() == "AAPL"
        assert symbol.exchange() == "NASDAQ"

    def test_symbol_string_representation(self):
        """Test string representation of Symbol."""
        symbol = Symbol("AAPL")
        assert str(symbol) == "AAPL"
        assert repr(symbol) == "Symbol('AAPL')"


class TestValueObjectBase:
    """Test the base ValueObject class."""

    def test_value_object_immutability(self):
        """Test that value objects are immutable."""
        money = Money(Decimal("100"), "USD")

        with pytest.raises(AttributeError):
            money.amount = Decimal("200")

    def test_value_object_equality_by_value(self):
        """Test value objects are equal by value, not identity."""
        money1 = Money(Decimal("100"), "USD")
        money2 = Money(Decimal("100"), "USD")

        assert money1 == money2
        assert money1 is not money2  # Different objects

    def test_value_object_hashable(self):
        """Test value objects are hashable."""
        price = Price(Decimal("100"))
        symbol = Symbol("AAPL")

        # Should be usable as dict keys
        data = {price: "price_value", symbol: "symbol_value"}

        assert data[Price(Decimal("100"))] == "price_value"
        assert data[Symbol("AAPL")] == "symbol_value"

    def test_value_object_copy(self):
        """Test copying value objects."""
        import copy

        original = Money(Decimal("100"), "USD")
        copied = copy.copy(original)
        deep_copied = copy.deepcopy(original)

        assert original == copied
        assert original == deep_copied
        assert original is not copied
        assert original is not deep_copied


class TestValueObjectIntegration:
    """Test integration between different value objects."""

    def test_price_times_quantity_equals_money(self):
        """Test multiplying price by quantity gives money."""
        price = Price(Decimal("150.50"))
        quantity = Quantity(100)

        total = Money(price.value * quantity.value, "USD")
        assert total.amount == Decimal("15050")

    def test_position_value_calculation(self):
        """Test calculating position value."""
        symbol = Symbol("AAPL")
        quantity = Quantity(100)
        entry_price = Price(Decimal("150.00"))
        current_price = Price(Decimal("155.00"))

        # Entry value
        entry_value = Money(entry_price.value * abs(quantity.value), "USD")
        assert entry_value.amount == Decimal("15000")

        # Current value
        current_value = Money(current_price.value * abs(quantity.value), "USD")
        assert current_value.amount == Decimal("15500")

        # P&L
        pnl = current_value - entry_value
        assert pnl.amount == Decimal("500")

    def test_portfolio_calculations(self):
        """Test portfolio calculations with value objects."""
        positions = [
            {"symbol": Symbol("AAPL"), "quantity": Quantity(100), "price": Price(Decimal("150"))},
            {"symbol": Symbol("MSFT"), "quantity": Quantity(50), "price": Price(Decimal("300"))},
            {
                "symbol": Symbol("GOOGL"),
                "quantity": Quantity(-20),
                "price": Price(Decimal("2500")),
            },  # Short
        ]

        total_value = Money(Decimal("0"), "USD")
        for pos in positions:
            position_value = Money(pos["price"].value * abs(pos["quantity"].value), "USD")
            total_value = total_value + position_value

        assert total_value.amount == Decimal("80000")  # 15000 + 15000 + 50000
