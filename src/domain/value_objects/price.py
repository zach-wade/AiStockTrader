"""Price value object for representing trading prices."""

from __future__ import annotations

# Standard library imports
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, ClassVar, Self

from .arithmetic_mixin import ArithmeticMixin
from .base import ComparableValueObject


class Price(ComparableValueObject, ArithmeticMixin):
    """Immutable value object representing a trading price."""

    # Common tick sizes for different markets
    _DEFAULT_TICK_SIZES: ClassVar[dict[str, Decimal]] = {
        "stock": Decimal("0.01"),  # US stocks: $0.01
        "forex": Decimal("0.0001"),  # Forex: 0.0001 (1 pip)
        "crypto": Decimal("0.00000001"),  # Crypto: satoshi level
        "futures": Decimal("0.25"),  # Some futures: 0.25
    }

    def __init__(
        self,
        value: Decimal | float | int | str,
        tick_size: Decimal | float | str | None = None,
        market_type: str = "stock",
    ) -> None:
        """Initialize Price with validation.

        Args:
            value: The price value
            tick_size: Minimum price increment (optional)
            market_type: Type of market for default tick size

        Raises:
            ValueError: If price is zero, negative, or invalid
        """
        if not isinstance(value, Decimal):
            value = Decimal(str(value))

        if value <= 0:
            raise ValueError(f"Price must be positive: {value}")

        self._value = value

        # Set tick size
        if tick_size is not None:
            if not isinstance(tick_size, Decimal):
                tick_size = Decimal(str(tick_size))
            if tick_size <= 0:
                raise ValueError(f"Tick size must be positive: {tick_size}")
            self._tick_size = tick_size
        else:
            self._tick_size = self._DEFAULT_TICK_SIZES.get(
                market_type, self._DEFAULT_TICK_SIZES["stock"]
            )

        self._market_type = market_type

    @property
    def value(self) -> Decimal:
        """Get the decimal value."""
        return self._value

    @property
    def amount(self) -> Decimal:
        """Get the decimal value (alias for ArithmeticMixin compatibility)."""
        return self._value

    def _create_new(self, amount: Decimal) -> Self:
        """Create a new Price instance with the given amount."""
        return type(self)(amount, self._tick_size, self._market_type)

    @property
    def tick_size(self) -> Decimal:
        """Get the tick size."""
        return self._tick_size

    @property
    def market_type(self) -> str:
        """Get the market type."""
        return self._market_type

    def round_to_tick(self) -> Self:
        """Round price to nearest valid tick.

        Returns:
            New Price rounded to tick size
        """
        if self._tick_size == 0:
            return self

        # Round to nearest tick
        num_ticks = (self._value / self._tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        rounded_value = num_ticks * self._tick_size

        return type(self)(rounded_value, self._tick_size, self._market_type)

    def is_valid(self) -> bool:
        """Check if price is valid for trading."""
        return self._value > 0

    def is_zero(self) -> bool:
        """Check if price is zero."""
        return self._value == 0

    def add(self, other: Price) -> Self:
        """Add another price.

        Args:
            other: Another Price instance

        Returns:
            New Price with sum
        """
        if not isinstance(other, Price):
            raise TypeError(f"Cannot add Price and {type(other)}")

        return type(self)(self._value + other._value, self._tick_size, self._market_type)

    def subtract(self, other: Price) -> Self:
        """Subtract another price.

        Args:
            other: Another Price instance

        Returns:
            New Price with difference

        Raises:
            ValueError: If result would be negative
        """
        if not isinstance(other, Price):
            raise TypeError(f"Cannot subtract {type(other)} from Price")

        result = self._value - other._value
        if result <= 0:
            raise ValueError("Price must be positive")

        return type(self)(result, self._tick_size, self._market_type)

    def multiply(self, factor: Decimal | float | int) -> Self:
        """Multiply price by a factor.

        Args:
            factor: Multiplication factor

        Returns:
            New Price with product
        """
        if not isinstance(factor, Decimal):
            factor = Decimal(str(factor))

        return type(self)(self._value * factor, self._tick_size, self._market_type)

    def divide(self, divisor: Decimal | float | int) -> Self:
        """Divide price by a divisor.

        Args:
            divisor: Division factor

        Returns:
            New Price with quotient

        Raises:
            ValueError: If divisor is zero
        """
        if not isinstance(divisor, Decimal):
            divisor = Decimal(str(divisor))

        if divisor == 0:
            raise ValueError("Cannot divide by zero")

        return type(self)(self._value / divisor, self._tick_size, self._market_type)

    def calculate_difference(self, other: Price) -> Decimal:
        """Calculate difference between two prices.

        Args:
            other: Another Price instance

        Returns:
            Difference as Decimal
        """
        if not isinstance(other, Price):
            raise TypeError(f"Cannot calculate difference with {type(other)}")

        return abs(self._value - other._value)

    def calculate_difference_percentage(self, other: Price) -> Decimal:
        """Calculate difference as percentage.

        Args:
            other: Another Price instance

        Returns:
            Difference percentage as Decimal
        """
        difference = self.calculate_difference(other)
        if self._value == 0 and other._value == 0:
            return Decimal(0)

        avg_price = (self._value + other._value) / 2
        if avg_price == 0:
            return Decimal(0)

        return (difference / avg_price) * 100

    def to_string(self, decimal_places: int | None = None) -> str:
        """Convert price to string representation.

        Args:
            decimal_places: Number of decimal places (auto if None)

        Returns:
            String representation of the price
        """
        if decimal_places is None:
            # Auto-detect based on tick size
            tick_str = str(self._tick_size)
            decimal_places = len(tick_str.split(".")[1]) if "." in tick_str else 0

        return f"{self._value:.{decimal_places}f}"

    def __eq__(self, other: object) -> bool:
        """Check equality with another Price."""
        if not isinstance(other, Price):
            return False
        return self._value == other._value

    def __lt__(self, other: object) -> bool:
        """Check if less than another Price or numeric value."""
        if isinstance(other, Price):
            return self._value < other._value
        if isinstance(other, (Decimal, int, float)):
            return self._value < Decimal(str(other))
        raise TypeError(f"Cannot compare Price and {type(other)}")

    def __hash__(self) -> int:
        """Get hash for use in sets/dicts."""
        return hash((self._value, self._tick_size))

    def __repr__(self) -> str:
        """Get string representation for debugging."""
        return f"Price({self._value}, tick_size={self._tick_size})"

    def __str__(self) -> str:
        """Get string representation."""
        return self.to_string()

    # Arithmetic operator overloads
    def __add__(self, other: Price | Decimal | int | float) -> Self:
        """Add another price or numeric value."""
        if isinstance(other, Price):
            return self.add(other)
        return type(self)(self._value + Decimal(str(other)), tick_size=self._tick_size)

    def __radd__(self, other: Decimal | int | float) -> Self:
        """Reverse add for numeric value + Price."""
        return type(self)(Decimal(str(other)) + self._value, tick_size=self._tick_size)

    def __sub__(self, other: Price | Decimal | int | float) -> Self:
        """Subtract another price or numeric value."""
        if isinstance(other, Price):
            return self.subtract(other)
        return type(self)(self._value - Decimal(str(other)), tick_size=self._tick_size)

    def __rsub__(self, other: Decimal | int | float) -> Self:
        """Reverse subtract for numeric value - Price."""
        return type(self)(Decimal(str(other)) - self._value, tick_size=self._tick_size)

    def __mul__(self, other: object) -> object:
        """Multiply by numeric value (returns Price) or Quantity (returns Money)."""
        from .money import Money
        from .quantity import Quantity

        if isinstance(other, Quantity):
            # Price * Quantity = Money
            return Money(self._value * other.value, "USD")
        elif isinstance(other, (Decimal, int, float)):
            # Price * numeric = Price
            return self.multiply(other)
        else:
            return NotImplemented

    def __rmul__(self, other: object) -> object:
        """Reverse multiply for Quantity * Price or numeric * Price."""
        return self.__mul__(other)

    def __truediv__(self, other: object) -> object:
        """Divide by a numeric value or another Price."""
        if isinstance(other, Price):
            return self._value / other._value
        if isinstance(other, (Decimal, int, float)):
            return self.divide(other)
        raise TypeError(f"Cannot divide Price by {type(other)}")

    @classmethod
    def from_bid_ask(cls, bid: Decimal | float, ask: Decimal | float, **kwargs: Any) -> Self:
        """Create Price from bid/ask midpoint.

        Args:
            bid: Bid price
            ask: Ask price
            **kwargs: Additional arguments for Price constructor

        Returns:
            New Price at midpoint
        """
        if not isinstance(bid, Decimal):
            bid = Decimal(str(bid))
        if not isinstance(ask, Decimal):
            ask = Decimal(str(ask))

        midpoint = (bid + ask) / 2
        return cls(midpoint, **kwargs)
