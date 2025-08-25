"""
Comprehensive unit tests for TradingValidationService.

Tests all trading validation business logic including symbols, prices,
quantities, orders, and portfolio data validation.
"""

from decimal import Decimal

import pytest

from src.domain.services.trading_validation_service import TradingValidationService


class TestTradingValidationService:
    """Test suite for TradingValidationService."""

    @pytest.fixture
    def service(self):
        """Create a TradingValidationService instance."""
        return TradingValidationService()

    # Trading Symbol Validation Tests

    def test_validate_trading_symbol_valid(self, service):
        """Test validation of valid trading symbols."""
        valid_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "A",  # Single character
            "ABCDEFGHIJ",  # Max length (10)
            "BRK",
            "TSM",
        ]

        for symbol in valid_symbols:
            assert service.validate_trading_symbol(symbol) is True, f"Failed for {symbol}"

    def test_validate_trading_symbol_invalid(self, service):
        """Test validation of invalid trading symbols."""
        invalid_symbols = [
            "",  # Empty string
            "AAPL1",  # Contains number
            "AAPL-B",  # Contains hyphen
            "AAPL.B",  # Contains period
            "AAPL GOOGL",  # Contains space
            "ABCDEFGHIJK",  # Too long (11 chars)
            "123",  # Only numbers
            "@AAPL",  # Special character
        ]

        for symbol in invalid_symbols:
            assert service.validate_trading_symbol(symbol) is False, f"Failed for {symbol}"

        # Note: The implementation converts to uppercase, so lowercase symbols are actually valid
        assert service.validate_trading_symbol("aapl") is True  # Converts to AAPL

    def test_validate_trading_symbol_edge_cases(self, service):
        """Test edge cases for trading symbol validation."""
        # Test with mixed case (converts to uppercase so it's valid)
        assert service.validate_trading_symbol("AaPl") is True

        # Test boundary conditions
        assert service.validate_trading_symbol("A" * 10) is True  # Exactly 10 chars
        assert service.validate_trading_symbol("A" * 11) is False  # 11 chars

    # Currency Code Validation Tests

    def test_validate_currency_code_valid(self, service):
        """Test validation of valid currency codes."""
        valid_codes = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]

        for code in valid_codes:
            assert service.validate_currency_code(code) is True, f"Failed for {code}"

    def test_validate_currency_code_invalid(self, service):
        """Test validation of invalid currency codes."""
        invalid_codes = [
            "",  # Empty string
            "US",  # Too short
            "USDD",  # Too long
            "US1",  # Contains number
            "US$",  # Contains special char
            "123",  # Only numbers
        ]

        for code in invalid_codes:
            assert service.validate_currency_code(code) is False, f"Failed for {code}"

        # Lowercase is converted to uppercase, so it's valid
        assert service.validate_currency_code("usd") is True

    # Price Validation Tests

    def test_validate_price_valid(self, service):
        """Test validation of valid prices."""
        valid_prices = [
            Decimal("100.50"),
            Decimal("0.01"),
            Decimal("999999.99"),
            Decimal("1"),
            Decimal("0.0001"),
            100.50,  # Float
            100,  # Integer
            "100.50",  # String representation
        ]

        for price in valid_prices:
            assert service.validate_price(price) is True, f"Failed for {price}"

    def test_validate_price_invalid(self, service):
        """Test validation of invalid prices."""
        invalid_prices = [
            Decimal("-1"),  # Negative
            Decimal("0"),  # Zero (MIN_PRICE < price < MAX_PRICE)
            Decimal("1000000"),  # Equal to max (needs to be less than)
            Decimal("1000001"),  # Too high
            "invalid",  # Invalid string
            None,  # None value
            "",  # Empty string
            "NaN",  # Not a number
            float("inf"),  # Infinity
        ]

        for price in invalid_prices:
            assert service.validate_price(price) is False, f"Failed for {price}"

    def test_validate_price_boundary_conditions(self, service):
        """Test price validation at boundaries."""
        # Test at max price (exclusive boundary)
        assert service.validate_price(service.MAX_PRICE) is False

        # Test just below max price
        assert service.validate_price(service.MAX_PRICE - Decimal("0.01")) is True

        # Test just above max price
        assert service.validate_price(service.MAX_PRICE + Decimal("0.01")) is False

        # Test at min price (exclusive boundary)
        assert service.validate_price(Decimal("0")) is False

        # Test just above min price
        assert service.validate_price(Decimal("0.00001")) is True

    # Quantity Validation Tests

    def test_validate_quantity_valid(self, service):
        """Test validation of valid quantities."""
        valid_quantities = [
            1,
            100,
            1000,
            0.01,  # Minimum
            1000000,  # Maximum
            999999,  # Near maximum
            50.5,
            1.23456,
        ]

        for quantity in valid_quantities:
            assert service.validate_quantity(quantity) is True, f"Failed for {quantity}"

    def test_validate_quantity_invalid(self, service):
        """Test validation of invalid quantities."""
        invalid_quantities = [
            0,  # Zero (below minimum)
            0.009,  # Below minimum
            -1,  # Negative
            0.001,  # Below minimum
            1000001,  # Above maximum
            None,  # None value
            "invalid",  # Invalid string
            float("inf"),  # Infinity
            float("nan"),  # NaN
        ]

        for quantity in invalid_quantities:
            assert service.validate_quantity(quantity) is False, f"Failed for {quantity}"

    def test_validate_quantity_boundary_conditions(self, service):
        """Test quantity validation at boundaries."""
        # Test at min quantity (inclusive)
        assert service.validate_quantity(service.MIN_QUANTITY) is True

        # Test just below min quantity
        assert service.validate_quantity(service.MIN_QUANTITY - 0.001) is False

        # Test at max quantity (inclusive)
        assert service.validate_quantity(service.MAX_QUANTITY) is True

        # Test just above max quantity
        assert service.validate_quantity(service.MAX_QUANTITY + 1) is False

    # Order Type Validation Tests

    def test_validate_order_type_valid(self, service):
        """Test validation of valid order types."""
        for order_type in service.VALID_ORDER_TYPES:
            assert service.validate_order_type(order_type) is True

        # Uppercase/mixed case should be converted to lowercase and validated
        assert service.validate_order_type("MARKET") is True
        assert service.validate_order_type("Market") is True
        assert service.validate_order_type("LiMiT") is True

    def test_validate_order_type_invalid(self, service):
        """Test validation of invalid order types."""
        invalid_types = [
            "invalid",
            "buy",  # This is a side, not a type
            "sell",  # This is a side, not a type
        ]

        for order_type in invalid_types:
            assert service.validate_order_type(order_type) is False

    # Order Side Validation Tests

    def test_validate_order_side_valid(self, service):
        """Test validation of valid order sides."""
        for side in service.VALID_ORDER_SIDES:
            assert service.validate_order_side(side) is True

        # Uppercase/mixed case should be converted to lowercase and validated
        assert service.validate_order_side("BUY") is True
        assert service.validate_order_side("Buy") is True
        assert service.validate_order_side("SELL") is True

    def test_validate_order_side_invalid(self, service):
        """Test validation of invalid order sides."""
        invalid_sides = [
            "long",
            "short",
            "hold",
            "market",  # This is a type, not a side
        ]

        for side in invalid_sides:
            assert service.validate_order_side(side) is False

    # Time In Force Validation Tests

    def test_validate_time_in_force_valid(self, service):
        """Test validation of valid time in force values."""
        for tif in service.VALID_TIME_IN_FORCE:
            assert service.validate_time_in_force(tif) is True

        # Uppercase/mixed case should be converted to lowercase and validated
        assert service.validate_time_in_force("DAY") is True
        assert service.validate_time_in_force("Day") is True
        assert service.validate_time_in_force("GTC") is True

    def test_validate_time_in_force_invalid(self, service):
        """Test validation of invalid time in force values."""
        invalid_tifs = [
            "invalid",
            "good_till_canceled",
            "immediate",
        ]

        for tif in invalid_tifs:
            assert service.validate_time_in_force(tif) is False

    # Order Validation Tests

    def test_validate_order_valid(self, service):
        """Test validation of valid order data."""
        valid_order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "order_type": "limit",
            "price": Decimal("150.50"),
            "time_in_force": "day",
        }

        errors = service.validate_order(valid_order)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_validate_order_market_order(self, service):
        """Test validation of market order (no limit price required)."""
        market_order = {
            "symbol": "MSFT",
            "side": "sell",
            "quantity": 50,
            "order_type": "market",
            "time_in_force": "ioc",
        }

        errors = service.validate_order(market_order)
        assert len(errors) == 0

    def test_validate_order_stop_order(self, service):
        """Test validation of stop order."""
        stop_order = {
            "symbol": "GOOGL",
            "side": "sell",
            "quantity": 25,
            "order_type": "stop",
            "time_in_force": "gtc",
        }

        # Note: The current implementation doesn't validate stop_price
        errors = service.validate_order(stop_order)
        assert len(errors) == 0

    def test_validate_order_stop_limit_order(self, service):
        """Test validation of stop-limit order."""
        stop_limit_order = {
            "symbol": "AMZN",
            "side": "buy",
            "quantity": 10,
            "order_type": "stop_limit",
            "price": Decimal("3010.00"),  # limit price
            "time_in_force": "day",
        }

        errors = service.validate_order(stop_limit_order)
        assert len(errors) == 0

    def test_validate_order_missing_required_fields(self, service):
        """Test validation of order with missing required fields."""
        incomplete_order = {
            "symbol": "AAPL",
            "side": "buy",
            # Missing quantity, order_type
        }

        errors = service.validate_order(incomplete_order)
        assert len(errors) > 0
        assert any("quantity" in error for error in errors)
        assert any("order_type" in error for error in errors)

    def test_validate_order_invalid_fields(self, service):
        """Test validation of order with invalid field values."""
        invalid_order = {
            "symbol": "AAPL123",  # Contains numbers
            "side": "invalid",  # Invalid side
            "quantity": -100,  # Negative
            "order_type": "unknown",  # Invalid type
            "price": "invalid",  # Invalid price
            "time_in_force": "forever",  # Invalid tif
        }

        errors = service.validate_order(invalid_order)
        assert len(errors) > 0
        # Check for specific error messages
        assert any("symbol" in error.lower() for error in errors)
        assert any("side" in error.lower() for error in errors)
        assert any("quantity" in error.lower() for error in errors)
        assert any("order type" in error.lower() for error in errors)

    def test_validate_order_limit_without_price(self, service):
        """Test validation of limit order without limit price."""
        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "order_type": "limit",
            "time_in_force": "day",
            # Missing price
        }

        errors = service.validate_order(order)
        assert len(errors) > 0
        assert any("price" in error.lower() for error in errors)

    def test_validate_order_market_with_price(self, service):
        """Test validation of market order with price (shouldn't have one)."""
        order = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 50,
            "order_type": "market",
            "price": Decimal("150.00"),  # Market orders shouldn't have price
            "time_in_force": "day",
        }

        errors = service.validate_order(order)
        assert len(errors) > 0
        assert any("market" in error.lower() and "price" in error.lower() for error in errors)

    # Portfolio Data Validation Tests

    def test_validate_portfolio_data_valid(self, service):
        """Test validation of valid portfolio data."""
        valid_portfolio = {
            "cash_balance": Decimal("10000.00"),
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "average_price": Decimal("150.00")},
                {"symbol": "MSFT", "quantity": 50, "average_price": Decimal("300.00")},
            ],
        }

        errors = service.validate_portfolio_data(valid_portfolio)
        assert len(errors) == 0

    def test_validate_portfolio_data_empty_positions(self, service):
        """Test validation of portfolio with no positions."""
        portfolio = {"cash_balance": Decimal("50000.00"), "positions": []}

        errors = service.validate_portfolio_data(portfolio)
        assert len(errors) == 0

    def test_validate_portfolio_data_invalid_cash(self, service):
        """Test validation of portfolio with invalid cash balance."""
        portfolio = {"cash_balance": Decimal("-1000.00"), "positions": []}  # Negative

        errors = service.validate_portfolio_data(portfolio)
        assert len(errors) > 0
        assert any("cash" in error.lower() for error in errors)

    def test_validate_portfolio_data_invalid_position(self, service):
        """Test validation of portfolio with invalid position."""
        portfolio = {
            "cash_balance": Decimal("10000.00"),
            "positions": [
                {
                    "symbol": "INVALID123",  # Invalid symbol (contains numbers)
                    "quantity": -10,  # Negative quantity
                    "average_price": Decimal("0"),  # Zero price
                }
            ],
        }

        errors = service.validate_portfolio_data(portfolio)
        assert len(errors) >= 3  # At least 3 errors

    def test_validate_portfolio_data_missing_fields(self, service):
        """Test validation of portfolio with missing fields."""
        portfolio = {
            "positions": [
                {
                    "symbol": "AAPL"
                    # Missing quantity and average_price
                }
            ]
            # Missing cash_balance is ok - it's optional
        }

        errors = service.validate_portfolio_data(portfolio)
        assert len(errors) > 0
        assert any("quantity" in error.lower() for error in errors)

    def test_validate_portfolio_data_mixed_valid_invalid(self, service):
        """Test validation of portfolio with mix of valid and invalid positions."""
        portfolio = {
            "cash_balance": Decimal("10000.00"),
            "positions": [
                {"symbol": "AAPL", "quantity": 100, "average_price": Decimal("150.00")},
                {
                    "symbol": "INVALID123",  # Invalid symbol
                    "quantity": -50,  # Invalid quantity
                    "average_price": Decimal("300.00"),
                },
            ],
        }

        errors = service.validate_portfolio_data(portfolio)
        assert len(errors) > 0

    # Get Order Schema Test

    def test_get_order_schema(self, service):
        """Test getting order schema definition."""
        schema = service.get_order_schema()

        assert "required" in schema
        assert "fields" in schema

        # Check required fields
        required = schema["required"]
        assert "symbol" in required
        assert "quantity" in required
        assert "order_type" in required
        assert "side" in required

        # Check field definitions
        fields = schema["fields"]
        assert "symbol" in fields
        assert "quantity" in fields
        assert "price" in fields
        assert "order_type" in fields
        assert "side" in fields
        assert "time_in_force" in fields

        # Check symbol field details
        symbol_field = fields["symbol"]
        assert symbol_field["type"] == "string"
        assert "pattern" in symbol_field
        assert symbol_field["max_length"] == service.MAX_SYMBOL_LENGTH

        # Check enums
        assert fields["order_type"]["enum"] == list(service.VALID_ORDER_TYPES)
        assert fields["side"]["enum"] == list(service.VALID_ORDER_SIDES)
        assert fields["time_in_force"]["enum"] == list(service.VALID_TIME_IN_FORCE)
