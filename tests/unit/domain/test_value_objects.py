"""Tests for domain value objects."""

# Standard library imports
from decimal import Decimal

# Third-party imports
import pytest

# Local imports
from src.domain.value_objects import Money, Price, Symbol


class TestMoney:
    """Test cases for Money value object."""

    def test_create_money_from_different_types(self):
        """Test creating Money from various input types."""
        money1 = Money(100)
        assert money1 == Decimal("100")

        money2 = Money(99.99)
        assert money2 == Decimal("99.99")

        money3 = Money("50.50")
        assert money3 == Decimal("50.50")

        money4 = Money(Decimal("25.25"))
        assert money4 == Decimal("25.25")

    def test_currency_validation(self):
        """Test currency code validation."""
        money = Decimal(100)
        assert money.currency == "USD"

        # Auto uppercase
        money2 = Money(100, "eur")
        assert money2.currency == "EUR"

        # Invalid currency
        with pytest.raises(ValueError):
            Money(100, "US")  # Too short

        with pytest.raises(ValueError):
            Money(100, "USDT")  # Too long

    def test_money_arithmetic(self):
        """Test arithmetic operations on Money."""
        money1 = Decimal(100)
        money2 = Decimal(50)

        # Addition
        result = money1.add(money2)
        assert result == Decimal("150")
        assert result.currency == "USD"

        # Subtraction
        result = money1.subtract(money2)
        assert result == Decimal("50")

        # Multiplication
        result = money1.multiply(2)
        assert result == Decimal("200")

        # Division
        result = money1.divide(4)
        assert result == Decimal("25")

    def test_money_currency_mismatch(self):
        """Test that operations with different currencies fail."""
        usd = Decimal(100)
        eur = Money(100, "EUR")

        with pytest.raises(ValueError):
            usd.add(eur)

        with pytest.raises(ValueError):
            usd.subtract(eur)

    def test_money_comparison(self):
        """Test comparison operations."""
        money1 = Decimal(100)
        money2 = Decimal(100)
        money3 = Decimal(50)

        assert money1 == money2
        assert money1 != money3
        assert money1 > money3
        assert money3 < money1
        assert money1 >= money2
        assert money3 <= money1

    def test_money_status_checks(self):
        """Test status check methods."""
        positive = Money(100)
        negative = Money(-50)
        zero = Money(0)

        assert positive.is_positive()
        assert not positive.is_negative()
        assert not positive.is_zero()

        assert negative.is_negative()
        assert not negative.is_positive()
        assert not negative.is_zero()

        assert zero.is_zero()
        assert not zero.is_positive()
        assert not zero.is_negative()

    def test_money_formatting(self):
        """Test money formatting for display."""
        money = Decimal(1234.567)

        # Default formatting
        assert money.format() == "$1,234.57"

        # Without currency
        assert money.format(include_currency=False) == "1,234.57"

        # Different decimal places
        assert money.round(0).format() == "$1,235.00"

        # Non-USD currency
        eur = Money(1000, "EUR")
        assert eur.format() == "1,000.00 EUR"

    def test_money_immutability(self):
        """Test that Money is immutable."""
        money = Money(100)
        original_amount = money

        # Operations return new instances
        result = money.add(Money(50))
        assert money == original_amount
        assert result == Decimal("150")


class TestSymbol:
    """Test cases for Symbol value object."""

    def test_create_valid_symbols(self):
        """Test creating valid symbols."""
        # Stock symbols
        symbol1 = Symbol("AAPL")
        assert symbol1 == "AAPL"
        assert symbol1.base_symbol == "AAPL"

        # With exchange
        symbol2 = Symbol("AAPL.US")
        assert symbol2 == "AAPL.US"
        assert symbol2.base_symbol == "AAPL"
        assert symbol2.exchange == "US"

        # With venue
        symbol3 = Symbol("AAPL:NASDAQ")
        assert symbol3 == "AAPL:NASDAQ"
        assert symbol3.base_symbol == "AAPL"
        assert symbol3.exchange == "NASDAQ"

        # Crypto
        symbol4 = Symbol("BTC-USD")
        assert symbol4 == "BTC-USD"
        assert symbol4.base_symbol == "BTC"
        assert symbol4.quote_currency == "USD"

    def test_symbol_normalization(self):
        """Test symbol normalization."""
        symbol = Symbol("  aapl  ")
        assert symbol == "AAPL"

        symbol2 = Symbol("msft.us")
        assert symbol2 == "MSFT.US"

    def test_invalid_symbols(self):
        """Test invalid symbol formats."""
        with pytest.raises(ValueError):
            Symbol("")

        with pytest.raises(ValueError):
            Symbol("123")  # Numbers only

        with pytest.raises(ValueError):
            Symbol("TOOLONG")  # Too long for stock

    def test_symbol_type_detection(self):
        """Test symbol type detection."""
        stock = Symbol("AAPL")
        assert stock.is_stock()
        assert not stock.is_crypto()
        assert not stock.is_option()

        crypto = Symbol("BTC-USD")
        assert crypto.is_crypto()
        assert not crypto.is_stock()
        assert not crypto.is_option()

    def test_symbol_exchange_operations(self):
        """Test adding/removing exchange."""
        symbol = Symbol("AAPL")

        # Add exchange
        with_exchange = symbol.with_exchange("NASDAQ")
        assert with_exchange == "AAPL.NASDAQ"
        assert with_exchange.exchange == "NASDAQ"

        # Remove exchange
        symbol2 = Symbol("MSFT.US")
        without = symbol2.without_exchange()
        assert without == "MSFT"
        assert without.exchange is None

    def test_symbol_comparison(self):
        """Test symbol comparison."""
        symbol1 = Symbol("AAPL")
        symbol2 = Symbol("AAPL")
        symbol3 = Symbol("MSFT")

        assert symbol1 == symbol2
        assert symbol1 != symbol3
        assert symbol1 < symbol3  # Alphabetical

    def test_symbol_validation_classmethod(self):
        """Test the validate class method."""
        assert Symbol.validate("AAPL")
        assert Symbol.validate("BTC-USD")
        assert not Symbol.validate("123")
        assert not Symbol.validate("")


class TestQuantity:
    """Test cases for Quantity value object."""

    def test_create_quantity(self):
        """Test creating quantities."""
        qty1 = 100
        assert qty1 == Decimal("100")

        qty2 = 10.5
        assert qty2 == Decimal("10.5")

        qty3 = "25.25"
        assert qty3 == Decimal("25.25")

        # Negative for short
        qty4 = -50
        assert qty4 == Decimal("-50")

    def test_quantity_position_type(self):
        """Test position type detection."""
        long_qty = 100
        assert long_qty.is_long()
        assert not long_qty.is_short()
        assert not long_qty.is_zero()

        short_qty = -50
        assert short_qty.is_short()
        assert not short_qty.is_long()
        assert not short_qty.is_zero()

        zero_qty = 0
        assert zero_qty.is_zero()
        assert not zero_qty.is_long()
        assert not zero_qty.is_short()

    def test_quantity_arithmetic(self):
        """Test arithmetic operations."""
        qty1 = 100
        qty2 = 50

        # Addition
        result = qty1.add(qty2)
        assert result == Decimal("150")

        # Subtraction
        result = qty1.subtract(qty2)
        assert result == Decimal("50")

        # Multiplication
        result = qty1.multiply(2)
        assert result == Decimal("200")

        # Division
        result = qty1.divide(4)
        assert result == Decimal("25")

        # Division by zero
        with pytest.raises(ValueError):
            qty1.divide(0)

    def test_quantity_split(self):
        """Test splitting quantities."""
        qty = 100

        # Split into 1 part
        parts = qty.split(1)
        assert len(parts) == 1
        assert parts[0] == Decimal("100")

        # Split into 4 equal parts
        parts = qty.split(4)
        assert len(parts) == 4
        assert all(p == Decimal("25") for p in parts)

        # Split with remainder
        qty2 = 103
        parts = qty2.split(4)
        assert len(parts) == 4
        assert parts[0] == Decimal("28")  # Gets remainder
        assert all(p == Decimal("25") for p in parts[1:])

        # Invalid split
        with pytest.raises(ValueError):
            qty.split(0)

    def test_quantity_rounding(self):
        """Test quantity rounding."""
        qty = "123.456789"

        # Round to whole number
        rounded = qty.round(0)
        assert rounded == Decimal("123")

        # Round to 2 decimal places
        rounded = qty.round(2)
        assert rounded == Decimal("123.45")

        # Round to 4 decimal places
        rounded = qty.round(4)
        assert rounded == Decimal("123.4567")

    def test_quantity_comparison(self):
        """Test comparison operations."""
        qty1 = 100
        qty2 = 100
        qty3 = 50

        assert qty1 == qty2
        assert qty1 != qty3
        assert qty1 > qty3
        assert qty3 < qty1
        assert qty1 >= qty2
        assert qty3 <= qty1

    def test_quantity_abs_and_neg(self):
        """Test absolute value and negation."""
        qty = 100
        neg_qty = -50

        # Negation
        assert (-qty) == Decimal("-100")
        assert (-neg_qty) == Decimal("50")

        # Absolute value
        assert abs(qty) == Decimal("100")
        assert abs(neg_qty) == Decimal("50")


class TestPrice:
    """Test cases for Price value object."""

    def test_create_price(self):
        """Test creating prices."""
        price1 = Price(100)
        assert price1 == Decimal("100")

        price2 = Price(99.99)
        assert price2 == Decimal("99.99")

        price3 = Price("50.50")
        assert price3 == Decimal("50.50")

        # With custom tick size
        price4 = Price(100, tick_size=0.05)
        assert price4.tick_size == Decimal("0.05")

    def test_price_validation(self):
        """Test price validation."""
        # Negative prices not allowed
        with pytest.raises(ValueError):
            Price(-10)

        # Invalid tick size
        with pytest.raises(ValueError):
            Price(100, tick_size=-0.01)

        with pytest.raises(ValueError):
            Price(100, tick_size=0)

    def test_market_type_tick_sizes(self):
        """Test default tick sizes for different markets."""
        stock = Price(100, market_type="stock")
        assert stock.tick_size == Decimal("0.01")

        forex = Price(1.2345, market_type="forex")
        assert forex.tick_size == Decimal("0.0001")

        crypto = Price(50000, market_type="crypto")
        assert crypto.tick_size == Decimal("0.00000001")

        futures = Price(4000, market_type="futures")
        assert futures.tick_size == Decimal("0.25")

    def test_round_to_tick(self):
        """Test rounding to tick size."""
        # Stock with 0.01 tick
        price = Price(100.126, tick_size=0.01)
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.13")

        # Futures with 0.25 tick
        price2 = Price(100.30, tick_size=0.25)
        rounded2 = price2.round_to_tick()
        assert rounded2 == Decimal("100.25")

        # Crypto with small tick
        price3 = Price(0.123456789, tick_size=0.00000001)
        rounded3 = price3.round_to_tick()
        assert rounded3 == Decimal("0.12345679")

    def test_price_arithmetic(self):
        """Test arithmetic operations."""
        price1 = Price(100)
        price2 = Price(50)

        # Addition
        result = price1.add(price2)
        assert result == Decimal("150")

        # Subtraction
        result = price1.subtract(price2)
        assert result == Decimal("50")

        # Cannot subtract to negative
        with pytest.raises(ValueError):
            price2.subtract(price1)

        # Multiplication
        result = price1.multiply(2)
        assert result == Decimal("200")

        # Division
        result = price1.divide(4)
        assert result == Decimal("25")

        with pytest.raises(ValueError):
            price1.divide(0)

    def test_spread_calculations(self):
        """Test spread calculations."""
        bid = Price(100)
        ask = Price(101)

        # Absolute difference
        difference = bid.calculate_difference(ask)
        assert difference == Decimal("1")

        # Percentage difference
        difference_pct = bid.calculate_difference_percentage(ask)
        # (1 / 100.5) * 100 ≈ 0.995%
        assert abs(difference_pct - Decimal("0.995")) < Decimal("0.01")

        # Zero prices
        zero1 = Price(0)
        zero2 = Price(0)
        assert zero1.calculate_difference_percentage(zero2) == Decimal("0")

    def test_price_from_bid_ask(self):
        """Test creating price from bid/ask."""
        price = Price.from_bid_ask(100, 101)
        assert price == Decimal("100.5")

        price2 = Price.from_bid_ask(99.98, 100.02, tick_size=0.01)
        assert price2 == Decimal("100")
        assert price2.tick_size == Decimal("0.01")

    def test_price_formatting(self):
        """Test price formatting."""
        price1 = Price(1234.567, tick_size=0.01)
        assert price1.to_string() == "1234.57"
        assert price1.to_string(decimal_places=0) == "1235"
        assert price1.to_string(decimal_places=3) == "1234.567"

        price2 = Price(100, tick_size=0.25)
        assert price2.to_string() == "100.00"

    def test_price_comparison(self):
        """Test comparison operations."""
        price1 = Price(100)
        price2 = Price(100)
        price3 = Price(50)

        assert price1 == price2
        assert price1 != price3
        assert price1 > price3
        assert price3 < price1
        assert price1 >= price2
        assert price3 <= price1
