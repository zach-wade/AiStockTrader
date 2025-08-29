"""
Trading validation domain service.

This module contains all trading-specific validation business logic
that was previously in the infrastructure layer. It provides validation
for trading orders, portfolio data, and other trading-related inputs.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any


class TradingValidationService:
    """
    Domain service for trading-specific validation logic.

    This service encapsulates all business rules related to trading validation,
    ensuring that trading data conforms to domain requirements.
    """

    # Trading symbol pattern (1-10 uppercase letters)
    SYMBOL_PATTERN = r"^[A-Z]{1,10}$"

    # ISO currency code pattern (3 uppercase letters)
    CURRENCY_PATTERN = r"^[A-Z]{3}$"

    # Decimal pattern for price validation
    DECIMAL_PATTERN = r"^-?\d+(\.\d+)?$"

    # Trading limits (business rules)
    MAX_PRICE = Decimal("1000000")
    MIN_PRICE = Decimal("0")
    MAX_QUANTITY = 1000000
    MIN_QUANTITY = 0.01
    MAX_SYMBOL_LENGTH = 10

    # Valid order types and sides (business rules)
    VALID_ORDER_TYPES = {"market", "limit", "stop", "stop_limit"}
    VALID_ORDER_SIDES = {"buy", "sell"}
    VALID_TIME_IN_FORCE = {"day", "gtc", "ioc", "fok"}

    @classmethod
    def validate_trading_symbol(cls, symbol: str) -> bool:
        """
        Validate trading symbol format according to business rules.

        Args:
            symbol: Trading symbol to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not symbol or len(symbol) > cls.MAX_SYMBOL_LENGTH:
            return False
        return re.match(cls.SYMBOL_PATTERN, symbol.upper()) is not None

    @classmethod
    def validate_currency_code(cls, currency: str) -> bool:
        """
        Validate ISO currency code format.

        Args:
            currency: Currency code to validate

        Returns:
            bool: True if valid ISO currency code, False otherwise
        """
        if not currency or len(currency) != 3:
            return False
        return re.match(cls.CURRENCY_PATTERN, currency.upper()) is not None

    @classmethod
    def validate_price(cls, price: str | float | Decimal) -> bool:
        """
        Validate price according to trading business rules.

        Args:
            price: Price value to validate

        Returns:
            bool: True if valid price, False otherwise
        """
        try:
            if isinstance(price, Decimal):
                price_val = price
            elif isinstance(price, (int, float)):
                price_val = Decimal(str(price))
            else:  # str type
                if not re.match(cls.DECIMAL_PATTERN, price):
                    return False
                price_val = Decimal(price)

            # Business rule: Price must be positive and within reasonable bounds
            return cls.MIN_PRICE < price_val < cls.MAX_PRICE
        except (ValueError, InvalidOperation):
            return False

    @classmethod
    def validate_quantity(cls, quantity: str | int | float) -> bool:
        """
        Validate trading quantity according to business rules.

        Args:
            quantity: Quantity to validate

        Returns:
            bool: True if valid quantity, False otherwise
        """
        try:
            if isinstance(quantity, (int, float)):
                qty_val = float(quantity)
            else:  # str type
                if not re.match(r"^\d+(\.\d+)?$", quantity):
                    return False
                qty_val = float(quantity)

            # Business rule: Quantity must be within trading limits
            return cls.MIN_QUANTITY <= qty_val <= cls.MAX_QUANTITY
        except ValueError:
            return False

    @classmethod
    def validate_order_type(cls, order_type: str) -> bool:
        """
        Validate order type against allowed types.

        Args:
            order_type: Order type to validate

        Returns:
            bool: True if valid order type, False otherwise
        """
        return order_type.lower() in cls.VALID_ORDER_TYPES

    @classmethod
    def validate_order_side(cls, side: str) -> bool:
        """
        Validate order side (buy/sell).

        Args:
            side: Order side to validate

        Returns:
            bool: True if valid side, False otherwise
        """
        return side.lower() in cls.VALID_ORDER_SIDES

    @classmethod
    def validate_time_in_force(cls, tif: str) -> bool:
        """
        Validate time in force parameter.

        Args:
            tif: Time in force value to validate

        Returns:
            bool: True if valid time in force, False otherwise
        """
        return tif.lower() in cls.VALID_TIME_IN_FORCE

    @classmethod
    def validate_order(cls, order_data: dict[str, Any]) -> list[str]:
        """
        Validate complete trading order data according to business rules.

        Args:
            order_data: Dictionary containing order data

        Returns:
            List of validation error messages, empty if valid
        """
        errors = []

        # Validate required fields
        required_fields = ["symbol", "quantity", "order_type", "side"]
        for field in required_fields:
            if field not in order_data:
                errors.append(f"Required field '{field}' is missing")

        # Validate symbol
        if "symbol" in order_data:
            if not cls.validate_trading_symbol(order_data["symbol"]):
                errors.append("Invalid trading symbol format")

        # Validate quantity
        if "quantity" in order_data:
            if not cls.validate_quantity(order_data["quantity"]):
                errors.append(
                    f"Invalid quantity: must be between {cls.MIN_QUANTITY} and {cls.MAX_QUANTITY}"
                )

        # Validate order type
        if "order_type" in order_data:
            if not cls.validate_order_type(order_data["order_type"]):
                errors.append(f"Invalid order type: must be one of {cls.VALID_ORDER_TYPES}")

            # Business rule: Limit and stop limit orders require a price
            if order_data["order_type"].lower() in ["limit", "stop_limit"]:
                if "price" not in order_data:
                    errors.append("Price is required for limit and stop limit orders")
                elif not cls.validate_price(order_data["price"]):
                    errors.append("Invalid price format or value")

        # Validate side
        if "side" in order_data:
            if not cls.validate_order_side(order_data["side"]):
                errors.append(f"Invalid order side: must be one of {cls.VALID_ORDER_SIDES}")

        # Validate optional time in force
        if "time_in_force" in order_data:
            if not cls.validate_time_in_force(order_data["time_in_force"]):
                errors.append(f"Invalid time in force: must be one of {cls.VALID_TIME_IN_FORCE}")

        # Validate optional price (for all order types that might have it)
        if "price" in order_data and "order_type" in order_data:
            if order_data["order_type"].lower() in ["market"]:
                errors.append("Market orders should not have a price")

        return errors

    @classmethod
    def validate_portfolio_data(cls, portfolio_data: dict[str, Any]) -> list[str]:
        """
        Validate portfolio data structure according to business rules.

        Args:
            portfolio_data: Dictionary containing portfolio data

        Returns:
            List of validation error messages, empty if valid
        """
        errors = []

        # Validate positions structure
        if "positions" not in portfolio_data:
            errors.append("Portfolio data must contain 'positions' field")
            return errors

        positions = portfolio_data.get("positions", [])
        if not isinstance(positions, list):
            errors.append("Positions must be a list")
            return errors

        # Validate each position
        for i, position in enumerate(positions):
            if not isinstance(position, dict):
                errors.append(f"Position {i} must be a dictionary")
                continue

            # Validate required position fields
            if "symbol" not in position:
                errors.append(f"Position {i} missing required field 'symbol'")
            elif not cls.validate_trading_symbol(position["symbol"]):
                errors.append(f"Position {i} has invalid symbol format")

            if "quantity" not in position:
                errors.append(f"Position {i} missing required field 'quantity'")
            elif not cls.validate_quantity(position["quantity"]):
                errors.append(f"Position {i} has invalid quantity")

            # Validate optional fields if present
            if "average_price" in position:
                if not cls.validate_price(position["average_price"]):
                    errors.append(f"Position {i} has invalid average price")

            if "current_price" in position:
                if not cls.validate_price(position["current_price"]):
                    errors.append(f"Position {i} has invalid current price")

        # Validate optional portfolio-level fields
        if "cash_balance" in portfolio_data:
            if not cls.validate_price(portfolio_data["cash_balance"]):
                errors.append("Invalid cash balance format")

        if "total_value" in portfolio_data:
            if not cls.validate_price(portfolio_data["total_value"]):
                errors.append("Invalid total value format")

        return errors

    @classmethod
    def validate_trading_request(cls, request_data: dict[str, Any]) -> bool:
        """
        Validate trading request data according to business rules.
        This method is extracted from the original RequestValidationService.

        Args:
            request_data: Dictionary containing trading request data

        Returns:
            True if request is valid

        Raises:
            ValueError: If validation fails
        """
        # Get validation errors using existing comprehensive method
        errors = cls.validate_order(request_data)

        if errors:
            # Raise first error found for compatibility with original API
            raise ValueError(errors[0])

        return True

    @classmethod
    def get_order_schema(cls) -> dict[str, Any]:
        """
        Get the order validation schema definition.

        Returns:
            Dictionary describing the order schema with business rules
        """
        return {
            "required": ["symbol", "quantity", "order_type", "side"],
            "fields": {
                "symbol": {
                    "type": "string",
                    "pattern": cls.SYMBOL_PATTERN,
                    "min_length": 1,
                    "max_length": cls.MAX_SYMBOL_LENGTH,
                    "description": "Trading symbol (1-10 uppercase letters)",
                },
                "quantity": {
                    "type": "float",
                    "minimum": cls.MIN_QUANTITY,
                    "maximum": cls.MAX_QUANTITY,
                    "description": "Order quantity",
                },
                "price": {
                    "type": "float",
                    "minimum": float(cls.MIN_PRICE),
                    "maximum": float(cls.MAX_PRICE),
                    "description": "Order price (required for limit orders)",
                },
                "order_type": {
                    "type": "string",
                    "enum": list(cls.VALID_ORDER_TYPES),
                    "description": "Type of order",
                },
                "side": {
                    "type": "string",
                    "enum": list(cls.VALID_ORDER_SIDES),
                    "description": "Order side (buy or sell)",
                },
                "time_in_force": {
                    "type": "string",
                    "enum": list(cls.VALID_TIME_IN_FORCE),
                    "description": "Time in force for the order",
                },
            },
        }
