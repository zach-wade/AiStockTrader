"""
Comprehensive unit tests for Price value object
"""

# Standard library imports
from decimal import Decimal

# Third-party imports
import pytest

# Local imports
from src.domain.value_objects import Price


class TestPriceCreation:
    """Test Price creation and validation"""

    def test_create_price_with_positive_value(self):
        """Test creating a price with positive value"""
        price = Price(Decimal("100.50"))
        assert price == Decimal("100.50")
        assert str(price) == "100.50"

    def test_create_price_with_string(self):
        """Test creating a price from string"""
        price = Price("150.75")
        assert price == Decimal("150.75")

    def test_create_price_with_int(self):
        """Test creating a price from integer"""
        price = Price(100)
        assert price == Decimal("100")

    def test_create_price_with_float(self):
        """Test creating a price from float"""
        price = Price(99.99)
        assert price == Decimal("99.99")

    def test_cannot_create_negative_price(self):
        """Test that negative price raises ValueError"""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            Price(Decimal("-10.00"))

    def test_create_zero_price(self):
        """Test that zero price is allowed"""
        price = Price(Decimal("0"))
        assert price == Decimal("0")
        assert price.is_zero()

    def test_price_precision_handling(self):
        """Test that Price handles decimal precision correctly"""
        price = Price(Decimal("100.123456"))
        # Should maintain precision
        assert price == Decimal("100.123456")

    def test_price_with_tick_size(self):
        """Test creating price with custom tick size"""
        price = Price(Decimal("100.00"), tick_size=Decimal("0.05"))
        assert price.tick_size == Decimal("0.05")

    def test_price_with_market_type(self):
        """Test creating price with different market types"""
        stock_price = Price(Decimal("100.00"), market_type="stock")
        assert stock_price.tick_size == Decimal("0.01")

        forex_price = Price(Decimal("1.2345"), market_type="forex")
        assert forex_price.tick_size == Decimal("0.0001")

        crypto_price = Price(Decimal("50000.00"), market_type="crypto")
        assert crypto_price.tick_size == Decimal("0.00000001")

    def test_invalid_tick_size(self):
        """Test that invalid tick size raises error"""
        with pytest.raises(ValueError, match="Tick size must be positive"):
            Price(Decimal("100.00"), tick_size=Decimal("0"))

        with pytest.raises(ValueError, match="Tick size must be positive"):
            Price(Decimal("100.00"), tick_size=Decimal("-0.01"))


class TestPriceProperties:
    """Test Price properties"""

    def test_value_property(self):
        """Test value property"""
        price = Price(Decimal("100.50"))
        assert price == Decimal("100.50")

    def test_tick_size_property(self):
        """Test tick_size property"""
        price = Price(Decimal("100.00"), tick_size=Decimal("0.25"))
        assert price.tick_size == Decimal("0.25")

    def test_market_type_property(self):
        """Test market_type property"""
        price = Price(Decimal("100.00"), market_type="futures")
        assert price.market_type == "futures"


class TestPriceComparison:
    """Test Price comparison operations"""

    def test_price_equality(self):
        """Test price equality comparison"""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("100.00"))
        price3 = Price(Decimal("200.00"))

        assert price1 == price2
        assert price1 != price3

    def test_price_less_than(self):
        """Test price less than comparison"""
        price1 = Price(Decimal("50.00"))
        price2 = Price(Decimal("100.00"))

        assert price1 < price2
        assert not price2 < price1

    def test_price_less_than_or_equal(self):
        """Test price less than or equal comparison"""
        price1 = Price(Decimal("50.00"))
        price2 = Price(Decimal("100.00"))
        price3 = Price(Decimal("100.00"))

        assert price1 <= price2
        assert price2 <= price3
        assert not price2 <= price1

    def test_price_greater_than(self):
        """Test price greater than comparison"""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("50.00"))

        assert price1 > price2
        assert not price2 > price1

    def test_price_greater_than_or_equal(self):
        """Test price greater than or equal comparison"""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("50.00"))
        price3 = Price(Decimal("100.00"))

        assert price1 >= price2
        assert price1 >= price3
        assert not price2 >= price1

    def test_price_hash(self):
        """Test Price hashing for use in sets/dicts"""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("100.00"))
        price3 = Price(Decimal("200.00"))

        # Equal prices should have same hash
        assert hash(price1) == hash(price2)

        # Can be used in sets
        price_set = {price1, price2, price3}
        assert len(price_set) == 2  # price1 and price2 are equal


class TestPriceArithmetic:
    """Test Price arithmetic operations"""

    def test_add_prices(self):
        """Test adding two prices"""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("50.00"))

        result = price1.add(price2)
        assert isinstance(result, Price)
        assert result == Decimal("150.00")

    def test_subtract_prices(self):
        """Test subtracting two prices"""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("30.00"))

        result = price1.subtract(price2)
        assert isinstance(result, Price)
        assert result == Decimal("70.00")

    def test_subtract_larger_price_raises_error(self):
        """Test subtracting larger price raises error (no negative prices)"""
        price1 = Price(Decimal("30.00"))
        price2 = Price(Decimal("100.00"))

        with pytest.raises(ValueError, match="Price cannot be negative"):
            price1.subtract(price2)

    def test_multiply_price_by_scalar(self):
        """Test multiplying price by scalar"""
        price = Price(Decimal("100.00"))

        result = price.multiply(3)
        assert isinstance(result, Price)
        assert result == Decimal("300.00")

        result = price.multiply(Decimal("2.5"))
        assert isinstance(result, Price)
        assert result == Decimal("250.00")

        result = price.multiply(0.5)
        assert isinstance(result, Price)
        assert result == Decimal("50.00")

    def test_divide_price_by_scalar(self):
        """Test dividing price by scalar"""
        price = Price(Decimal("100.00"))

        result = price.divide(2)
        assert isinstance(result, Price)
        assert result == Decimal("50.00")

        result = price.divide(Decimal("4"))
        assert isinstance(result, Price)
        assert result == Decimal("25.00")

    def test_divide_by_zero_raises_error(self):
        """Test that dividing by zero raises ValueError"""
        price = Price(Decimal("100.00"))

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            price.divide(0)

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            price.divide(Decimal("0"))


class TestPriceRounding:
    """Test Price rounding operations"""

    def test_round_to_tick(self):
        """Test rounding price to tick size"""
        # Stock with $0.01 tick
        price = Price(Decimal("100.12678"), market_type="stock")
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.13")

        # Custom tick size of 0.25
        price = Price(Decimal("100.37"), tick_size=Decimal("0.25"))
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.25")

        price = Price(Decimal("100.38"), tick_size=Decimal("0.25"))
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.50")

    def test_round_forex_price(self):
        """Test rounding forex prices"""
        price = Price(Decimal("1.234567"), market_type="forex")
        rounded = price.round_to_tick()
        # Forex tick is 0.0001, so should round to 4 decimal places
        assert rounded == Decimal("1.2346")


class TestPriceUtilityMethods:
    """Test Price utility methods"""

    def test_is_valid(self):
        """Test is_valid method"""
        valid_price = Price(Decimal("100.00"))
        assert valid_price.is_valid()

        zero_price = Price(Decimal("0"))
        assert zero_price.is_valid()

    def test_is_zero(self):
        """Test is_zero method"""
        zero_price = Price(Decimal("0"))
        assert zero_price.is_zero()

        non_zero_price = Price(Decimal("0.01"))
        assert not non_zero_price.is_zero()

    def test_calculate_difference(self):
        """Test calculating difference between prices"""
        bid = Price(Decimal("100.00"))
        ask = Price(Decimal("100.50"))

        difference = ask.calculate_difference(bid)
        assert difference == Decimal("0.50")

        # Reverse should give same absolute difference
        difference = bid.calculate_difference(ask)
        assert difference == Decimal("0.50")

    def test_calculate_difference_percentage(self):
        """Test calculating difference percentage"""
        bid = Price(Decimal("100.00"))
        ask = Price(Decimal("101.00"))

        # Difference is 1.00, average price is 100.50
        # Percentage = (1.00 / 100.50) * 100 ≈ 0.995%
        difference_pct = ask.calculate_difference_percentage(bid)
        assert abs(difference_pct - Decimal("0.9950248756218906")) < Decimal("0.0001")

    def test_from_bid_ask(self):
        """Test creating price from bid/ask"""
        mid_price = Price.from_bid_ask(Decimal("100.00"), Decimal("100.50"))
        assert mid_price == Decimal("100.25")

        # Test with different types
        mid_price = Price.from_bid_ask(100.00, 101.00)
        assert mid_price == Decimal("100.50")


class TestPriceFormatting:
    """Test Price formatting and display"""

    def test_string_representation(self):
        """Test string representation of price"""
        price = Price(Decimal("1234.56"))
        assert str(price) == "1234.56"

    def test_repr_representation(self):
        """Test repr representation of price"""
        price = Price(Decimal("100.00"))
        assert repr(price) == "Price(100.00, tick_size=0.01)"

    def test_format_method(self):
        """Test format method with decimal places"""
        price = Price(Decimal("100.123456"))

        # Default formatting rounds to tick size (0.01 for stock)
        assert price.to_string() == "100.12"

        # Format to 2 decimal places
        assert price.to_string(decimal_places=2) == "100.12"

        # Format to 4 decimal places
        assert price.to_string(decimal_places=4) == "100.1235"

        # Format to 0 decimal places
        assert price.to_string(decimal_places=0) == "100"


class TestPriceEdgeCases:
    """Test Price edge cases"""

    def test_very_large_price(self):
        """Test handling very large prices"""
        large_price = Price(Decimal("999999999999.99"))
        assert large_price == Decimal("999999999999.99")
        assert large_price.is_valid()

    def test_very_small_price(self):
        """Test handling very small prices"""
        small_price = Price(Decimal("0.00000001"))
        assert small_price == Decimal("0.00000001")
        assert small_price.is_valid()
        assert not small_price.is_zero()

    def test_price_with_many_decimal_places(self):
        """Test price with many decimal places"""
        price = Price(Decimal("100.123456789"))
        assert price == Decimal("100.123456789")

        # Rounding should work correctly
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.12")  # Stock default tick is 0.01

    def test_price_immutability(self):
        """Test that Price is immutable"""
        price = Price(Decimal("100.00"))
        original_value = price

        # Operations should return new Price objects
        new_price = price.add(Price(Decimal("50.00")))
        assert price == original_value
        assert new_price == Decimal("150.00")

    def test_price_with_different_market_types(self):
        """Test prices with different market types maintain their properties"""
        stock = Price(Decimal("100.00"), market_type="stock")
        forex = Price(Decimal("1.2345"), market_type="forex")
        crypto = Price(Decimal("50000.00"), market_type="crypto")
        futures = Price(Decimal("100.00"), market_type="futures")

        assert stock.market_type == "stock"
        assert forex.market_type == "forex"
        assert crypto.market_type == "crypto"
        assert futures.market_type == "futures"

        # Check default tick sizes
        assert stock.tick_size == Decimal("0.01")
        assert forex.tick_size == Decimal("0.0001")
        assert crypto.tick_size == Decimal("0.00000001")
        assert futures.tick_size == Decimal("0.25")
