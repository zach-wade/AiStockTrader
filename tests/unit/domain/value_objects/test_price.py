"""
Comprehensive unit tests for Price value object.

Tests all public methods, operators, edge cases, mathematical operations,
comparisons, validation, immutability, and string representations.
"""

# Standard library imports
from decimal import Decimal

# Third-party imports
import pytest

# Local imports
from src.domain.value_objects.price import Price


class TestPriceCreation:
    """Test Price creation and initialization."""

    def test_create_price_with_decimal(self):
        """Test creating price with Decimal value."""
        price = Price(Decimal("100.50"))
        assert price == Decimal("100.50")
        assert price.tick_size == Decimal("0.01")  # Default stock tick
        assert price.market_type == "stock"

    def test_create_price_with_float(self):
        """Test creating price with float value."""
        price = Price(100.50)
        assert price == Decimal("100.50")

    def test_create_price_with_int(self):
        """Test creating price with integer value."""
        price = Price(100)
        assert price == Decimal("100")

    def test_create_price_with_string(self):
        """Test creating price with string value."""
        price = Price("100.50")
        assert price == Decimal("100.50")

    def test_create_price_with_custom_tick_size(self):
        """Test creating price with custom tick size."""
        price = Price(Decimal("100.00"), tick_size=Decimal("0.05"))
        assert price.tick_size == Decimal("0.05")

    def test_create_price_with_tick_size_as_float(self):
        """Test creating price with tick size as float."""
        price = Price(Decimal("100.00"), tick_size=0.25)
        assert price.tick_size == Decimal("0.25")

    def test_create_price_with_tick_size_as_string(self):
        """Test creating price with tick size as string."""
        price = Price(Decimal("100.00"), tick_size="0.10")
        assert price.tick_size == Decimal("0.10")

    def test_create_price_with_different_market_types(self):
        """Test creating price with different market types."""
        stock = Price(Decimal("100.00"), market_type="stock")
        assert stock.tick_size == Decimal("0.01")
        assert stock.market_type == "stock"

        forex = Price(Decimal("1.2345"), market_type="forex")
        assert forex.tick_size == Decimal("0.0001")
        assert forex.market_type == "forex"

        crypto = Price(Decimal("50000.00"), market_type="crypto")
        assert crypto.tick_size == Decimal("0.00000001")
        assert crypto.market_type == "crypto"

        futures = Price(Decimal("100.00"), market_type="futures")
        assert futures.tick_size == Decimal("0.25")
        assert futures.market_type == "futures"

    def test_unknown_market_type_uses_stock_default(self):
        """Test that unknown market type defaults to stock tick size."""
        price = Price(Decimal("100.00"), market_type="unknown")
        assert price.tick_size == Decimal("0.01")
        assert price.market_type == "unknown"

    def test_negative_price_raises_error(self):
        """Test that negative price raises ValueError."""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            Price(Decimal("-10.00"))

        with pytest.raises(ValueError, match="Price cannot be negative"):
            Price(-10)

    def test_zero_price_allowed(self):
        """Test that zero price is allowed."""
        price = Price(Decimal("0"))
        assert price == Decimal("0")
        assert price.is_zero()
        assert price.is_valid()

    def test_invalid_tick_size_raises_error(self):
        """Test that invalid tick size raises error."""
        with pytest.raises(ValueError, match="Tick size must be positive"):
            Price(Decimal("100.00"), tick_size=Decimal("0"))

        with pytest.raises(ValueError, match="Tick size must be positive"):
            Price(Decimal("100.00"), tick_size=Decimal("-0.01"))

    def test_extreme_precision(self):
        """Test price with extreme decimal precision."""
        price = Price(Decimal("100.123456789123456789"))
        assert price == Decimal("100.123456789123456789")


class TestPriceProperties:
    """Test Price property accessors."""

    def test_value_property(self):
        """Test value property returns correct value."""
        price = Price(Decimal("123.45"))
        assert price == Decimal("123.45")

    def test_tick_size_property(self):
        """Test tick_size property returns correct value."""
        price = Price(Decimal("100.00"), tick_size=Decimal("0.05"))
        assert price.tick_size == Decimal("0.05")

    def test_market_type_property(self):
        """Test market_type property returns correct value."""
        price = Price(Decimal("100.00"), market_type="forex")
        assert price.market_type == "forex"


class TestPriceArithmetic:
    """Test Price arithmetic operations."""

    def test_add_prices(self):
        """Test adding two prices."""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("50.00"))

        result = price1.add(price2)
        assert isinstance(result, Price)
        assert result == Decimal("150.00")
        assert result.tick_size == price1.tick_size
        assert result.market_type == price1.market_type

    def test_add_non_price_raises_error(self):
        """Test adding non-Price type raises error."""
        price = Price(Decimal("100.00"))

        with pytest.raises(TypeError, match="Cannot add Price and"):
            price.add(100)

    def test_subtract_prices(self):
        """Test subtracting two prices."""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("30.00"))

        result = price1.subtract(price2)
        assert isinstance(result, Price)
        assert result == Decimal("70.00")
        assert result.tick_size == price1.tick_size
        assert result.market_type == price1.market_type

    def test_subtract_resulting_in_negative_raises_error(self):
        """Test subtracting larger price raises error."""
        price1 = Price(Decimal("30.00"))
        price2 = Price(Decimal("100.00"))

        with pytest.raises(ValueError, match="Price cannot be negative"):
            price1.subtract(price2)

    def test_subtract_non_price_raises_error(self):
        """Test subtracting non-Price type raises error."""
        price = Price(Decimal("100.00"))

        with pytest.raises(TypeError, match="Cannot subtract .* from Price"):
            price.subtract(50)

    def test_multiply_by_decimal(self):
        """Test multiplying price by Decimal factor."""
        price = Price(Decimal("100.00"))

        result = price.multiply(Decimal("2.5"))
        assert isinstance(result, Price)
        assert result == Decimal("250.00")
        assert result.tick_size == price.tick_size
        assert result.market_type == price.market_type

    def test_multiply_by_int(self):
        """Test multiplying price by integer factor."""
        price = Price(Decimal("100.00"))

        result = price.multiply(3)
        assert result == Decimal("300.00")

    def test_multiply_by_float(self):
        """Test multiplying price by float factor."""
        price = Price(Decimal("100.00"))

        result = price.multiply(0.5)
        assert result == Decimal("50.00")

    def test_multiply_by_zero(self):
        """Test multiplying price by zero."""
        price = Price(Decimal("100.00"))

        result = price.multiply(0)
        assert result == Decimal("0")
        assert result.is_zero()

    def test_multiply_resulting_in_negative_raises_error(self):
        """Test multiplying by negative factor raises error since Price cannot be negative."""
        price = Price(Decimal("100.00"))

        # Price cannot be negative, so multiplying by negative should raise error
        with pytest.raises(ValueError, match="Price cannot be negative"):
            price.multiply(Decimal("-2"))

    def test_divide_by_decimal(self):
        """Test dividing price by Decimal divisor."""
        price = Price(Decimal("100.00"))

        result = price.divide(Decimal("4"))
        assert isinstance(result, Price)
        assert result == Decimal("25.00")
        assert result.tick_size == price.tick_size
        assert result.market_type == price.market_type

    def test_divide_by_int(self):
        """Test dividing price by integer divisor."""
        price = Price(Decimal("100.00"))

        result = price.divide(2)
        assert result == Decimal("50.00")

    def test_divide_by_float(self):
        """Test dividing price by float divisor."""
        price = Price(Decimal("100.00"))

        result = price.divide(2.5)
        assert result == Decimal("40.00")

    def test_divide_by_zero_raises_error(self):
        """Test dividing by zero raises ValueError."""
        price = Price(Decimal("100.00"))

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            price.divide(0)

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            price.divide(Decimal("0"))


class TestPriceRounding:
    """Test Price rounding operations."""

    def test_round_to_tick_stock(self):
        """Test rounding to tick size for stock market."""
        price = Price(Decimal("100.12678"), market_type="stock")
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.13")

    def test_round_to_tick_custom_tick_size(self):
        """Test rounding with custom tick size."""
        # Tick size of 0.25
        price = Price(Decimal("100.37"), tick_size=Decimal("0.25"))
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.25")

        price = Price(Decimal("100.38"), tick_size=Decimal("0.25"))
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.50")

        price = Price(Decimal("100.62"), tick_size=Decimal("0.25"))
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.50")

        price = Price(Decimal("100.63"), tick_size=Decimal("0.25"))
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.75")

    def test_round_to_tick_forex(self):
        """Test rounding for forex market."""
        price = Price(Decimal("1.234567"), market_type="forex")
        rounded = price.round_to_tick()
        assert rounded == Decimal("1.2346")

    def test_round_to_tick_crypto(self):
        """Test rounding for crypto market."""
        price = Price(Decimal("50000.123456789"), market_type="crypto")
        rounded = price.round_to_tick()
        # Crypto tick is 0.00000001 (satoshi level)
        assert rounded == Decimal("50000.12345679")

    def test_round_to_tick_futures(self):
        """Test rounding for futures market."""
        price = Price(Decimal("100.37"), market_type="futures")
        rounded = price.round_to_tick()
        assert rounded == Decimal("100.25")

    def test_round_to_tick_zero_tick_size(self):
        """Test that zero tick size (edge case) returns same price."""
        # This is a defensive test - zero tick size should not be allowed
        # but if it somehow exists, round_to_tick should handle it
        price = Price(Decimal("100.123"))
        price._tick_size = Decimal("0")  # Force zero tick size
        rounded = price.round_to_tick()
        assert rounded == price


class TestPriceComparison:
    """Test Price comparison operations."""

    def test_equality_same_value(self):
        """Test equality for same value."""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("100.00"))

        assert price1 == price2
        assert price1 == price2

    def test_equality_different_value(self):
        """Test inequality for different values."""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("200.00"))

        assert price1 != price2
        assert price1 != price2

    def test_equality_same_value_different_tick_size(self):
        """Test that prices with same value but different tick sizes are equal."""
        price1 = Price(Decimal("100.00"), tick_size=Decimal("0.01"))
        price2 = Price(Decimal("100.00"), tick_size=Decimal("0.05"))

        assert price1 == price2  # Equality is based on value only

    def test_equality_with_non_price(self):
        """Test equality comparison with non-Price types."""
        price = Price(Decimal("100.00"))

        assert price != 100
        assert price != "100"
        assert price != None

    def test_less_than(self):
        """Test less than comparison."""
        price1 = Price(Decimal("50.00"))
        price2 = Price(Decimal("100.00"))

        assert price1 < price2
        assert not price2 < price1

    def test_less_than_non_price_raises_error(self):
        """Test less than comparison with non-Price raises error."""
        price = Price(Decimal("100.00"))

        with pytest.raises(TypeError, match="Cannot compare Price and"):
            price < 50

    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        price1 = Price(Decimal("50.00"))
        price2 = Price(Decimal("100.00"))
        price3 = Price(Decimal("100.00"))

        assert price1 <= price2
        assert price2 <= price3
        assert not price2 <= price1

    def test_greater_than(self):
        """Test greater than comparison."""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("50.00"))

        assert price1 > price2
        assert not price2 > price1

    def test_greater_than_non_price_raises_error(self):
        """Test greater than comparison with non-Price raises error."""
        price = Price(Decimal("100.00"))

        with pytest.raises(TypeError, match="Cannot compare Price and"):
            price > 150

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        price1 = Price(Decimal("100.00"))
        price2 = Price(Decimal("50.00"))
        price3 = Price(Decimal("100.00"))

        assert price1 >= price2
        assert price1 >= price3
        assert not price2 >= price1

    def test_hash(self):
        """Test hash for use in sets and dicts."""
        price1 = Price(Decimal("100.00"), tick_size=Decimal("0.01"))
        price2 = Price(Decimal("100.00"), tick_size=Decimal("0.01"))
        price3 = Price(Decimal("200.00"), tick_size=Decimal("0.01"))
        price4 = Price(Decimal("100.00"), tick_size=Decimal("0.05"))

        # Equal prices with same tick size have same hash
        assert hash(price1) == hash(price2)

        # Different tick sizes have different hashes even if value is same
        assert hash(price1) != hash(price4)

        # Can be used in sets
        price_set = {price1, price2, price3, price4}
        assert len(price_set) == 3  # price1 and price2 are the same


class TestPriceUtilityMethods:
    """Test Price utility methods."""

    def test_is_valid(self):
        """Test is_valid method."""
        valid_price = Price(Decimal("100.00"))
        assert valid_price.is_valid()

        zero_price = Price(Decimal("0"))
        assert zero_price.is_valid()

    def test_is_zero(self):
        """Test is_zero method."""
        zero_price = Price(Decimal("0"))
        assert zero_price.is_zero()

        non_zero_price = Price(Decimal("0.01"))
        assert not non_zero_price.is_zero()

        very_small_price = Price(Decimal("0.00000001"))
        assert not very_small_price.is_zero()

    def test_calculate_difference(self):
        """Test calculating difference between prices."""
        bid = Price(Decimal("100.00"))
        ask = Price(Decimal("100.50"))

        difference = ask.calculate_difference(bid)
        assert difference == Decimal("0.50")

        # Reverse should give same absolute difference
        difference = bid.calculate_difference(ask)
        assert difference == Decimal("0.50")

    def test_calculate_difference_non_price_raises_error(self):
        """Test calculating difference with non-Price raises error."""
        price = Price(Decimal("100.00"))

        with pytest.raises(TypeError, match="Cannot calculate difference with"):
            price.calculate_difference(100)

    def test_calculate_difference_percentage(self):
        """Test calculating difference percentage."""
        bid = Price(Decimal("100.00"))
        ask = Price(Decimal("101.00"))

        # Difference is 1.00, average price is 100.50
        # Percentage = (1.00 / 100.50) * 100 ≈ 0.995%
        difference_pct = ask.calculate_difference_percentage(bid)
        expected = Decimal("1.00") / Decimal("100.50") * 100
        assert abs(difference_pct - expected) < Decimal("0.0001")

    def test_calculate_difference_percentage_zero_prices(self):
        """Test calculating difference percentage with zero prices."""
        zero1 = Price(Decimal("0"))
        zero2 = Price(Decimal("0"))

        difference_pct = zero1.calculate_difference_percentage(zero2)
        assert difference_pct == Decimal("0")

    def test_calculate_difference_percentage_one_zero(self):
        """Test calculating difference percentage with one zero price."""
        zero = Price(Decimal("0"))
        non_zero = Price(Decimal("100.00"))

        # When average is 50 and difference is 100
        difference_pct = non_zero.calculate_difference_percentage(zero)
        assert difference_pct == Decimal("200")

    def test_from_bid_ask(self):
        """Test creating price from bid/ask midpoint."""
        mid_price = Price.from_bid_ask(Decimal("100.00"), Decimal("100.50"))
        assert mid_price == Decimal("100.25")

    def test_from_bid_ask_with_floats(self):
        """Test creating price from bid/ask with floats."""
        mid_price = Price.from_bid_ask(100.00, 101.00)
        assert mid_price == Decimal("100.50")

    def test_from_bid_ask_with_kwargs(self):
        """Test creating price from bid/ask with additional arguments."""
        mid_price = Price.from_bid_ask(
            Decimal("100.00"), Decimal("100.50"), tick_size=Decimal("0.05"), market_type="futures"
        )
        assert mid_price == Decimal("100.25")
        assert mid_price.tick_size == Decimal("0.05")
        assert mid_price.market_type == "futures"


class TestPriceFormatting:
    """Test Price formatting and display."""

    def test_format_default(self):
        """Test default formatting."""
        price = Price(Decimal("1234.56"))
        formatted = price.to_string()
        assert formatted == "1234.56"

    def test_format_with_decimal_places(self):
        """Test formatting with specific decimal places."""
        price = Price(Decimal("100.123456"))

        assert price.to_string(decimal_places=0) == "100"
        assert price.to_string(decimal_places=2) == "100.12"
        assert price.to_string(decimal_places=4) == "100.1235"
        assert price.to_string(decimal_places=6) == "100.123456"

    def test_format_auto_decimal_places(self):
        """Test auto-detection of decimal places based on tick size."""
        stock_price = Price(Decimal("100.12"), market_type="stock")
        assert stock_price.to_string() == "100.12"  # 2 decimals for 0.01 tick

        forex_price = Price(Decimal("1.2345"), market_type="forex")
        assert forex_price.to_string() == "1.2345"  # 4 decimals for 0.0001 tick

        futures_price = Price(Decimal("100.25"), market_type="futures")
        assert futures_price.to_string() == "100.25"  # 2 decimals for 0.25 tick

    def test_format_zero(self):
        """Test formatting zero price."""
        zero_price = Price(Decimal("0"))
        assert zero_price.to_string() == "0.00"
        assert zero_price.to_string(decimal_places=4) == "0.0000"

    def test_str_representation(self):
        """Test string representation."""
        price = Price(Decimal("1234.56"))
        assert str(price) == "1234.56"

    def test_repr_representation(self):
        """Test repr representation."""
        price = Price(Decimal("100.00"), tick_size=Decimal("0.05"))
        assert repr(price) == "Price(100.00, tick_size=0.05)"


class TestPriceEdgeCases:
    """Test Price edge cases and extreme values."""

    def test_very_large_price(self):
        """Test handling very large prices."""
        large_price = Price(Decimal("999999999999999.99"))
        assert large_price == Decimal("999999999999999.99")
        assert large_price.is_valid()

    def test_very_small_price(self):
        """Test handling very small prices."""
        small_price = Price(Decimal("0.00000001"))
        assert small_price == Decimal("0.00000001")
        assert small_price.is_valid()
        assert not small_price.is_zero()

    def test_many_decimal_places(self):
        """Test price with many decimal places."""
        price = Price(Decimal("100.123456789123456789"))
        assert price == Decimal("100.123456789123456789")

    def test_scientific_notation(self):
        """Test handling scientific notation."""
        price = Price("1.5E+3")
        assert price == Decimal("1500")

        price = Price("1.5E-3")
        assert price == Decimal("0.0015")

    def test_immutability(self):
        """Test that Price is immutable."""
        price = Price(Decimal("100.00"))
        original_value = price
        original_tick = price.tick_size

        # Operations return new objects
        new_price = price.add(Price(Decimal("50.00")))
        assert price == original_value
        assert new_price == Decimal("150.00")

        # Properties can't be modified
        with pytest.raises(AttributeError):
            price = Decimal("200.00")

        with pytest.raises(AttributeError):
            price.tick_size = Decimal("0.05")

    def test_chain_operations(self):
        """Test chaining multiple operations."""
        price = Price(Decimal("100.00"))

        result = price.add(Price(Decimal("50.00"))).multiply(2).divide(3)
        assert result == Decimal("100.00")

        # Original unchanged
        assert price == Decimal("100.00")

    def test_zero_arithmetic(self):
        """Test arithmetic with zero prices."""
        zero = Price(Decimal("0"))
        hundred = Price(Decimal("100.00"))

        # Adding zero
        assert hundred.add(zero) == Decimal("100.00")
        assert zero.add(hundred) == Decimal("100.00")

        # Subtracting zero
        assert hundred.subtract(zero) == Decimal("100.00")

        # Can't subtract from zero (would be negative)
        with pytest.raises(ValueError):
            zero.subtract(hundred)

        # Multiplying by zero
        assert hundred.multiply(0) == Decimal("0")
        assert zero.multiply(100) == Decimal("0")

    def test_precision_preservation(self):
        """Test that precision is preserved in operations."""
        price1 = Price(Decimal("0.001"))
        price2 = Price(Decimal("0.002"))

        result = price1.add(price2)
        assert result == Decimal("0.003")

        # Multiplication preserves precision
        result = price1.multiply(Decimal("3"))
        assert result == Decimal("0.003")

    def test_tick_size_inheritance(self):
        """Test that tick size is inherited in operations."""
        price = Price(Decimal("100.00"), tick_size=Decimal("0.05"))

        # Add operation inherits tick size
        result = price.add(Price(Decimal("50.00")))
        assert result.tick_size == Decimal("0.05")

        # Multiply operation inherits tick size
        result = price.multiply(2)
        assert result.tick_size == Decimal("0.05")

        # Divide operation inherits tick size
        result = price.divide(2)
        assert result.tick_size == Decimal("0.05")
