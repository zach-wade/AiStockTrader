"""
Market Data Validation Service - Domain service for market data validation business rules.

This service handles business logic for validating market data requests and parameters,
implementing the Single Responsibility Principle.
"""

import re
from typing import Any


class MarketDataValidationError(Exception):
    """Exception raised when market data validation fails."""

    pass


class MarketDataValidationService:
    """
    Domain service for market data validation business logic.

    This service contains business rules for validating market data requests,
    symbols, intervals, and market data-related parameters.
    """

    def validate_market_data_request(self, request_data: dict[str, Any]) -> bool:
        """
        Validate market data request according to business rules.

        Args:
            request_data: Dictionary containing market data request

        Returns:
            True if request is valid

        Raises:
            MarketDataValidationError: If validation fails
        """
        # Check for required fields
        required_fields = ["symbols", "interval"]
        for field in required_fields:
            if field not in request_data:
                raise MarketDataValidationError(f"{field} is required for market data request")

        # Validate symbols
        symbols = request_data.get("symbols", [])
        if not symbols or not isinstance(symbols, list):
            raise MarketDataValidationError("symbols must be a non-empty list")

        # Check for too many symbols
        if len(symbols) > 100:
            raise MarketDataValidationError(f"Too many symbols: {len(symbols)} (max 100)")

        for symbol in symbols:
            if not isinstance(symbol, str) or not re.match(r"^[A-Z0-9]{1,5}$", symbol):
                raise MarketDataValidationError(f"Invalid symbol: {symbol}")

        # Validate interval
        valid_intervals = ["1min", "5min", "15min", "30min", "1hour", "1day"]
        interval = request_data.get("interval", "")
        if interval not in valid_intervals:
            raise MarketDataValidationError(f"Invalid interval: {interval}")

        return True

    @classmethod
    def validate_market_symbol(cls, symbol: str) -> bool:
        """
        Validate market data symbol format according to business rules.

        Args:
            symbol: Market symbol string

        Returns:
            True if symbol format is valid

        Raises:
            MarketDataValidationError: If symbol format is invalid
        """
        if not symbol:
            raise MarketDataValidationError("Symbol cannot be empty")

        # Business rule: Market data symbols can include numbers (e.g., ETFs, indices)
        if not re.match(r"^[A-Z0-9]{1,5}$", symbol):
            raise MarketDataValidationError(f"Invalid market symbol format: {symbol}")

        return True
