"""Symbol value object for representing trading symbols."""

# Standard library imports
import re
from typing import ClassVar


class Symbol:
    """Immutable value object representing a trading symbol."""

    # Valid symbol patterns
    _PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "stock": re.compile(r"^[A-Z]{1,5}$"),  # AAPL, MSFT, etc.
        "stock_exchange": re.compile(r"^[A-Z]{1,5}\.[A-Z]{2,10}$"),  # AAPL.US
        "stock_venue": re.compile(r"^[A-Z]{1,5}:[A-Z]{2,10}$"),  # AAPL:NASDAQ
        "crypto": re.compile(r"^[A-Z]{2,10}-[A-Z]{2,10}$"),  # BTC-USD, ETH-USDT
        "option": re.compile(r"^[A-Z]{1,5}\d{6}[CP]\d{8}$"),  # AAPL240119C00150000
    }

    def __init__(self, value: str) -> None:
        """Initialize Symbol with validation.

        Args:
            value: The symbol string

        Raises:
            ValueError: If symbol format is invalid
        """
        if not value:
            raise ValueError("Symbol cannot be empty")

        # Normalize to uppercase and strip whitespace
        normalized = value.upper().strip()

        # Validate format
        if not self._is_valid_format(normalized):
            raise ValueError(f"Invalid symbol format: {value}")

        self._value = normalized
        self._parse_components()

    def _is_valid_format(self, symbol: str) -> bool:
        """Check if symbol matches any valid pattern."""
        return any(pattern.match(symbol) for pattern in self._PATTERNS.values())

    def _parse_components(self) -> None:
        """Parse symbol components like base symbol and exchange."""
        # Check for exchange suffix (e.g., AAPL.US)
        if "." in self._value:
            parts = self._value.split(".")
            self._base_symbol = parts[0]
            self._exchange = parts[1] if len(parts) > 1 else None
        # Check for venue prefix (e.g., AAPL:NASDAQ)
        elif ":" in self._value:
            parts = self._value.split(":")
            self._base_symbol = parts[0]
            self._exchange = parts[1] if len(parts) > 1 else None
        # Check for crypto pair (e.g., BTC-USD)
        elif "-" in self._value:
            parts = self._value.split("-")
            self._base_symbol = parts[0]
            self._quote_currency = parts[1] if len(parts) > 1 else None
            self._exchange = None
        else:
            self._base_symbol = self._value
            self._exchange = None
            self._quote_currency = None

    @property
    def value(self) -> str:
        """Get the full symbol value."""
        return self._value

    @property
    def base_symbol(self) -> str:
        """Get the base ticker symbol."""
        return self._base_symbol

    @property
    def exchange(self) -> str | None:
        """Get the exchange if specified."""
        return self._exchange

    @property
    def quote_currency(self) -> str | None:
        """Get quote currency for crypto pairs."""
        return getattr(self, "_quote_currency", None)

    def is_crypto(self) -> bool:
        """Check if this is a cryptocurrency symbol."""
        return "-" in self._value

    def is_option(self) -> bool:
        """Check if this is an options symbol."""
        return bool(self._PATTERNS["option"].match(self._value))

    def is_stock(self) -> bool:
        """Check if this is a stock symbol."""
        return not self.is_crypto() and not self.is_option()

    def with_exchange(self, exchange: str) -> "Symbol":
        """Create new symbol with specified exchange.

        Args:
            exchange: Exchange code

        Returns:
            New Symbol instance with exchange
        """
        if self.is_crypto():
            raise ValueError("Cannot add exchange to crypto symbol")

        return Symbol(f"{self._base_symbol}.{exchange.upper()}")

    def without_exchange(self) -> "Symbol":
        """Create new symbol without exchange.

        Returns:
            New Symbol instance without exchange
        """
        if self._base_symbol != self._value:
            return Symbol(self._base_symbol)
        return self

    def __eq__(self, other: object) -> bool:
        """Check equality with another Symbol."""
        if not isinstance(other, Symbol):
            return False
        return self._value == other._value

    def __hash__(self) -> int:
        """Get hash for use in sets/dicts."""
        return hash(self._value)

    def __repr__(self) -> str:
        """Get string representation for debugging."""
        return f"Symbol('{self._value}')"

    def __str__(self) -> str:
        """Get string representation."""
        return self._value

    def __lt__(self, other: "Symbol") -> bool:
        """Compare symbols alphabetically."""
        if not isinstance(other, Symbol):
            raise TypeError(f"Cannot compare Symbol and {type(other)}")
        return self._value < other._value

    @classmethod
    def validate(cls, value: str) -> bool:
        """Check if a string is a valid symbol without creating instance.

        Args:
            value: String to validate

        Returns:
            True if valid symbol format
        """
        try:
            cls(value)
            return True
        except ValueError:
            return False
