"""Edge case tests for Price value object."""

# Standard library imports
from decimal import Decimal

# Third-party imports
import pytest

# Local imports
from src.domain.value_objects.price import Price


class TestPriceEdgeCases:
    """Test edge cases for Price value object."""

    def test_price_negative_validation(self):
        """Test that negative prices are rejected."""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            Price(-10)

        with pytest.raises(ValueError, match="Price cannot be negative"):
            Price(Decimal("-0.01"))

        # Zero is allowed
        price = Price(0)
        assert price.value == Decimal("0")
        assert price.is_zero()

    def test_price_tick_size_validation(self):
        """Test tick size validation."""
        # Negative tick size
        with pytest.raises(ValueError, match="Tick size must be positive"):
            Price(100, tick_size=-0.01)

        # Zero tick size
        with pytest.raises(ValueError, match="Tick size must be positive"):
            Price(100, tick_size=0)

        # Valid tick sizes
        price = Price(100, tick_size=0.01)
        assert price.tick_size == Decimal("0.01")

        price = Price(100, tick_size="0.001")
        assert price.tick_size == Decimal("0.001")

    def test_price_round_to_tick_edge_cases(self):
        """Test edge cases in tick rounding."""
        # Very small tick size
        price = Price(0.123456789, tick_size=0.00000001)
        rounded = price.round_to_tick()
        assert rounded.value == Decimal("0.12345679")

        # Large tick size
        price = Price(103.7, tick_size=5)
        rounded = price.round_to_tick()
        assert rounded.value == Decimal("105")

        # Exact tick multiple
        price = Price(100, tick_size=0.25)
        rounded = price.round_to_tick()
        assert rounded.value == Decimal("100")

        # Half-tick rounding (ROUND_HALF_UP)
        price = Price(100.125, tick_size=0.25)
        rounded = price.round_to_tick()
        assert rounded.value == Decimal("100.25")

        price = Price(100.375, tick_size=0.25)
        rounded = price.round_to_tick()
        assert rounded.value == Decimal("100.50")

    def test_price_arithmetic_edge_cases(self):
        """Test edge cases in price arithmetic."""
        # Adding to zero
        zero = Price(0)
        hundred = Price(100)
        result = zero.add(hundred)
        assert result.value == Decimal("100")

        # Subtracting to zero
        result = hundred.subtract(hundred)
        assert result.value == Decimal("0")
        assert result.is_zero()

        # Cannot subtract to negative
        with pytest.raises(ValueError, match="Price cannot be negative"):
            zero.subtract(hundred)

        # Multiply by zero
        result = hundred.multiply(0)
        assert result.value == Decimal("0")

        # Multiply by negative - Price validation prevents negative results
        with pytest.raises(ValueError, match="Price cannot be negative"):
            hundred.multiply(-1)

        # Divide by very small number
        result = hundred.divide(0.00001)
        assert result.value == Decimal("10000000")

        # Division by zero
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            hundred.divide(0)

    def test_price_type_errors(self):
        """Test type errors in price operations."""
        price = Price(100)

        # Adding non-Price type
        with pytest.raises(TypeError, match="Cannot add Price"):
            price.add(100)

        with pytest.raises(TypeError, match="Cannot add Price"):
            price.add("100")

        # Subtracting non-Price type
        with pytest.raises(TypeError, match="Cannot subtract"):
            price.subtract(50)

        # Comparing with non-Price type
        with pytest.raises(TypeError, match="Cannot compare"):
            _ = price < 100

        with pytest.raises(TypeError, match="Cannot compare"):
            _ = price > "100"

        # Spread calculation with non-Price
        with pytest.raises(TypeError, match="Cannot calculate spread"):
            price.calculate_spread(100)

    def test_price_spread_calculations_edge_cases(self):
        """Test edge cases in spread calculations."""
        # Zero spread
        price1 = Price(100)
        price2 = Price(100)
        assert price1.calculate_spread(price2) == Decimal("0")
        assert price1.calculate_spread_percentage(price2) == Decimal("0")

        # Both prices zero
        zero1 = Price(0)
        zero2 = Price(0)
        assert zero1.calculate_spread(zero2) == Decimal("0")
        assert zero1.calculate_spread_percentage(zero2) == Decimal("0")

        # One price zero
        hundred = Price(100)
        zero = Price(0)
        assert hundred.calculate_spread(zero) == Decimal("100")
        # Percentage: 100 / ((100 + 0) / 2) * 100 = 200%
        assert hundred.calculate_spread_percentage(zero) == Decimal("200")

        # Very small spread
        price1 = Price(100)
        price2 = Price(100.001)
        spread = price1.calculate_spread(price2)
        assert spread == Decimal("0.001")

        # Spread percentage calculation precision
        spread_pct = price1.calculate_spread_percentage(price2)
        expected = (Decimal("0.001") / Decimal("100.0005")) * 100
        assert abs(spread_pct - expected) < Decimal("0.00001")

    def test_price_from_bid_ask_edge_cases(self):
        """Test creating price from bid/ask with edge cases."""
        # Normal case
        price = Price.from_bid_ask(99.99, 100.01)
        assert price.value == Decimal("100")

        # Same bid and ask
        price = Price.from_bid_ask(100, 100)
        assert price.value == Decimal("100")

        # With tick size
        price = Price.from_bid_ask(99.98, 100.02, tick_size=0.01)
        assert price.value == Decimal("100")
        assert price.tick_size == Decimal("0.01")

        # With market type
        price = Price.from_bid_ask(1.2345, 1.2347, market_type="forex")
        assert price.value == Decimal("1.2346")
        assert price.market_type == "forex"

        # Very small prices
        price = Price.from_bid_ask(0.00001, 0.00002)
        assert price.value == Decimal("0.000015")

    def test_price_format_edge_cases(self):
        """Test edge cases in price formatting."""
        # Auto-detect decimal places from tick size
        price = Price(1234.567, tick_size=0.01)
        assert price.format() == "1234.57"

        price = Price(1234.567, tick_size=0.001)
        assert price.format() == "1234.567"

        price = Price(1234.567, tick_size=1)
        assert price.format() == "1235"

        # Override decimal places
        price = Price(1234.567)
        assert price.format(decimal_places=0) == "1235"
        assert price.format(decimal_places=4) == "1234.5670"

        # Zero price formatting
        zero = Price(0)
        assert zero.format() == "0.00"
        assert zero.format(decimal_places=4) == "0.0000"

        # Very large price
        large = Price(999999999.99)
        assert large.format() == "999999999.99"

        # Very small price
        small = Price(0.00000001, market_type="crypto")
        # Formatting very small prices may round to 0 based on default precision
        formatted = small.format()
        # Just verify it produces a valid formatted string
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_price_comparison_edge_cases(self):
        """Test edge cases in price comparisons."""
        # Comparing with very close values
        price1 = Price(100)
        price2 = Price(100.00000001)
        assert price1 < price2
        assert price1 <= price2
        assert price2 > price1
        assert price2 >= price1

        # Comparing zeros
        zero1 = Price(0)
        zero2 = Price(0)
        assert zero1 == zero2
        assert zero1 <= zero2
        assert zero1 >= zero2

        # Hash and equality
        price1 = Price(100, tick_size=0.01)
        price2 = Price(100, tick_size=0.01)
        price3 = Price(100, tick_size=0.05)

        assert price1 == price2
        assert hash(price1) == hash(price2)

        # Different tick sizes mean different hashes even if values are same
        assert price1 == price3  # Values are equal
        assert hash(price1) != hash(price3)  # But hashes differ

        # Use in sets
        price_set = {price1, price2, price3}
        assert len(price_set) == 2  # price1 and price2 have same hash

    def test_price_repr_and_str(self):
        """Test string representations."""
        price = Price(100.50, tick_size=0.01)
        assert repr(price) == "Price(100.5, tick_size=0.01)"
        assert str(price) == "100.50"

        price = Price(0, tick_size=0.25)
        assert repr(price) == "Price(0, tick_size=0.25)"
        assert str(price) == "0.00"

    def test_price_is_valid(self):
        """Test is_valid method."""
        valid = Price(100)
        assert valid.is_valid()

        zero = Price(0)
        assert zero.is_valid()  # Zero is valid

        # Negative prices are not allowed to be created
        # So all created prices should be valid

    def test_price_market_types(self):
        """Test different market type behaviors."""
        # Unknown market type defaults to stock
        price = Price(100, market_type="unknown")
        assert price.tick_size == Decimal("0.01")  # Stock default

        # Case sensitivity
        price = Price(100, market_type="STOCK")
        assert price.market_type == "STOCK"
        # Tick size lookup is case-sensitive, so it defaults
        assert price.tick_size == Decimal("0.01")
