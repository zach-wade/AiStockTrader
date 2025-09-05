"""Quantity value object for representing trading quantities."""

from __future__ import annotations

# Standard library imports
from decimal import ROUND_DOWN, Decimal, getcontext
from typing import Self

# Import converter for DRY compliance
from .arithmetic_mixin import ArithmeticMixin
from .base import ComparableValueObject
from .converter import ValueObjectConverter

# Set high precision for Decimal operations to maintain precision in calculations
getcontext().prec = 50


class Quantity(ComparableValueObject, ArithmeticMixin):
    """Immutable value object representing a trading quantity."""

    def __init__(self, value: Decimal | float | int | str) -> None:
        """Initialize Quantity with validation.

        Args:
            value: The quantity value

        Raises:
            ValueError: If quantity is invalid
        """
        if not isinstance(value, Decimal):
            value = Decimal(str(value))

        # Allow zero for position closing
        # Negative values are allowed for short positions
        self._value = value

    @property
    def value(self) -> Decimal:
        """Get the decimal value."""
        return self._value

    @property
    def amount(self) -> Decimal:
        """Get the decimal value (alias for ArithmeticMixin compatibility)."""
        return self._value

    def _create_new(self, amount: Decimal) -> Self:
        """Create a new Quantity instance with the given amount."""
        return type(self)(amount)

    def is_valid(self) -> bool:
        """Check if quantity is valid for trading."""
        # Zero is valid for closing positions
        # Both positive and negative are valid
        return True

    def is_long(self) -> bool:
        """Check if quantity represents a long position."""
        return self._value > 0

    def is_short(self) -> bool:
        """Check if quantity represents a short position."""
        return self._value < 0

    def is_zero(self) -> bool:
        """Check if quantity is zero."""
        return self._value == 0

    def abs(self) -> Self:
        """Get absolute value of quantity.

        Returns:
            New Quantity with absolute value
        """
        return type(self)(abs(self._value))

    def add(self, other: Quantity) -> Self:
        """Add another quantity.

        Args:
            other: Another Quantity instance

        Returns:
            New Quantity with sum
        """
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot add Quantity and {type(other)}")

        return type(self)(self._value + other._value)

    def subtract(self, other: Quantity) -> Self:
        """Subtract another quantity.

        Args:
            other: Another Quantity instance

        Returns:
            New Quantity with difference
        """
        if not isinstance(other, Quantity):
            raise TypeError(f"Cannot subtract {type(other)} from Quantity")

        return type(self)(self._value - other._value)

    def multiply(self, factor: Decimal | float | int) -> Self:
        """Multiply quantity by a factor.

        Args:
            factor: Multiplication factor

        Returns:
            New Quantity with product
        """
        if not isinstance(factor, Decimal):
            factor = Decimal(str(factor))

        # Special case: multiplying by 1 should return the exact same value
        if factor == 1:
            return type(self)(self._value)

        return type(self)(self._value * factor)

    def divide(self, divisor: Decimal | float | int) -> Self:
        """Divide quantity by a divisor.

        Args:
            divisor: Division factor

        Returns:
            New Quantity with quotient

        Raises:
            ValueError: If divisor is zero
        """
        if not isinstance(divisor, Decimal):
            divisor = Decimal(str(divisor))

        if divisor == 0:
            raise ValueError("Cannot divide by zero")

        return type(self)(self._value / divisor)

    def split(self, num_parts: int) -> list[Self]:
        """Split quantity into equal parts.

        Args:
            num_parts: Number of parts to split into

        Returns:
            List of Quantity instances

        Raises:
            ValueError: If num_parts is less than 1
        """
        if num_parts < 1:
            raise ValueError("Number of parts must be at least 1")

        if num_parts == 1:
            return [self]

        # Calculate base amount per part (rounded down)
        base_amount = self._value / num_parts
        base_amount = base_amount.quantize(Decimal("1"), rounding=ROUND_DOWN)

        # Calculate remainder
        total_base = base_amount * num_parts
        remainder = self._value - total_base

        # Create parts
        parts = []
        for i in range(num_parts):
            if i == 0:
                # First part gets the remainder
                parts.append(type(self)(base_amount + remainder))
            else:
                parts.append(type(self)(base_amount))

        return parts

    def round(self, decimal_places: int = 0) -> Self:
        """Round to specified decimal places.

        Args:
            decimal_places: Number of decimal places

        Returns:
            New Quantity with rounded value
        """
        if decimal_places < 0:
            raise ValueError("Decimal places must be non-negative")

        quantizer = Decimal(10) ** -decimal_places
        rounded = self._value.quantize(quantizer, rounding=ROUND_DOWN)
        return type(self)(rounded)

    def __eq__(self, other: object) -> bool:
        """Check equality with another Quantity."""
        if not isinstance(other, Quantity):
            return False
        return self._value == other._value

    def __lt__(self, other: object) -> bool:
        """Check if less than another Quantity or numeric value."""
        if isinstance(other, Quantity):
            return self._value < other._value
        if isinstance(other, (Decimal, int, float)):
            return self._value < Decimal(str(other))
        raise TypeError(f"Cannot compare Quantity and {type(other)}")

    def __neg__(self) -> Self:
        """Negate the quantity."""
        return type(self)(-self._value)

    def __abs__(self) -> Self:
        """Get absolute value."""
        return type(self)(abs(self._value))

    def __hash__(self) -> int:
        """Get hash for use in sets/dicts."""
        return hash(self._value)

    def __repr__(self) -> str:
        """Get string representation for debugging."""
        return f"Quantity({self._value})"

    def __str__(self) -> str:
        """Get string representation."""
        # Format without unnecessary decimal places
        if self._value == self._value.to_integral_value():
            return str(int(self._value))
        # Normalize to remove trailing zeros
        normalized = self._value.normalize()
        return str(normalized)

    # Arithmetic operator overloads
    def __add__(self, other: Quantity | Decimal | int | float) -> Self:
        """Add another quantity or numeric value."""
        if isinstance(other, Quantity):
            return self.add(other)
        return type(self)(self._value + Decimal(str(other)))

    def __radd__(self, other: Decimal | int | float) -> Self:
        """Reverse add for numeric value + Quantity."""
        return type(self)(Decimal(str(other)) + self._value)

    def __sub__(self, other: Quantity | Decimal | int | float) -> Self:
        """Subtract another quantity or numeric value."""
        if isinstance(other, Quantity):
            return self.subtract(other)
        return type(self)(self._value - Decimal(str(other)))

    def __rsub__(self, other: Decimal | int | float) -> Self:
        """Reverse subtract for numeric value - Quantity."""
        return type(self)(Decimal(str(other)) - self._value)

    def __mul__(self, other: object) -> object:
        """Multiply by a numeric value or Price (returns Money)."""
        if hasattr(other, "_value") and hasattr(other, "_tick_size"):  # Duck typing for Price
            # Quantity * Price = Money
            from .money import Money

            return Money(self._value * ValueObjectConverter.extract_value(other), "USD")
        if isinstance(other, (Decimal, int, float)):
            return self.multiply(other)
        raise TypeError(f"Cannot multiply Quantity by {type(other)}")

    def __rmul__(self, other: Decimal | int | float) -> Self:
        """Reverse multiply for numeric value * Quantity."""
        return self.multiply(other)

    def __truediv__(self, other: object) -> object:
        """Divide by a numeric value or another Quantity."""
        if isinstance(other, Quantity):
            return self._value / other._value
        if isinstance(other, (Decimal, int, float)):
            return self.divide(other)
        raise TypeError(f"Cannot divide Quantity by {type(other)}")
