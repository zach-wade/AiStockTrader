"""
ArithmeticMixin for value objects to support mathematical operations.

This mixin provides common arithmetic operations for value objects like
Money, Price, and Quantity, ensuring type safety and proper decimal handling.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TypeVar

T = TypeVar("T", bound="ArithmeticMixin")


class ArithmeticMixin(ABC):
    """
    Mixin class that provides arithmetic operations for value objects.

    Classes using this mixin must have an 'amount' property that returns a Decimal.
    """

    @property
    @abstractmethod
    def amount(self) -> Decimal:
        """Get the numeric amount of this value object."""
        pass

    @abstractmethod
    def _create_new(self: T, amount: Decimal) -> T:
        """
        Create a new instance of the same type with the given amount.

        Args:
            amount: The amount for the new instance

        Returns:
            A new instance of the same type
        """
        pass

    def __add__(self: T, other: T | Decimal | int | float) -> T:
        """Add two values."""
        if isinstance(other, ArithmeticMixin):
            if type(other) != type(self):
                raise TypeError(f"Cannot add {type(self).__name__} and {type(other).__name__}")
            return self._create_new(self.amount + other.amount)
        elif isinstance(other, (Decimal, int, float)):
            return self._create_new(self.amount + Decimal(str(other)))
        else:
            raise TypeError(f"Cannot add {type(self).__name__} and {type(other).__name__}")

    def __sub__(self: T, other: T | Decimal | int | float) -> T:
        """Subtract two values."""
        if isinstance(other, ArithmeticMixin):
            if type(other) != type(self):
                raise TypeError(
                    f"Cannot subtract {type(other).__name__} from {type(self).__name__}"
                )
            return self._create_new(self.amount - other.amount)
        elif isinstance(other, (Decimal, int, float)):
            return self._create_new(self.amount - Decimal(str(other)))
        else:
            raise TypeError(f"Cannot subtract {type(other).__name__} from {type(self).__name__}")

    def __mul__(self: T, other: object) -> object:
        """Multiply by a scalar."""
        if isinstance(other, (Decimal, int, float)):
            return self._create_new(self.amount * Decimal(str(other)))
        else:
            raise TypeError(f"Cannot multiply {type(self).__name__} by {type(other).__name__}")

    def __truediv__(self: T, other: object) -> object:
        """Divide two values."""
        if isinstance(other, ArithmeticMixin):
            if type(other) == type(self):
                # Division of same types returns a scalar
                return self.amount / other.amount
            else:
                raise TypeError(f"Cannot divide {type(self).__name__} by {type(other).__name__}")
        elif isinstance(other, (Decimal, int, float)):
            return self._create_new(self.amount / Decimal(str(other)))
        else:
            raise TypeError(f"Cannot divide {type(self).__name__} by {type(other).__name__}")

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, ArithmeticMixin):
            return False
        if type(other) != type(self):
            return False
        return self.amount == other.amount

    def __lt__(self, other: object) -> bool:
        """Less than comparison."""
        if isinstance(other, ArithmeticMixin):
            if type(other) != type(self):
                raise TypeError(f"Cannot compare {type(self).__name__} and {type(other).__name__}")
            return self.amount < other.amount
        elif isinstance(other, (Decimal, int, float)):
            return self.amount < Decimal(str(other))
        else:
            raise TypeError(f"Cannot compare {type(self).__name__} and {type(other).__name__}")

    def __le__(self, other: T | Decimal | int | float) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, ArithmeticMixin):
            if type(other) != type(self):
                raise TypeError(f"Cannot compare {type(self).__name__} and {type(other).__name__}")
            return self.amount <= other.amount
        elif isinstance(other, (Decimal, int, float)):
            return self.amount <= Decimal(str(other))
        else:
            raise TypeError(f"Cannot compare {type(self).__name__} and {type(other).__name__}")

    def __gt__(self, other: T | Decimal | int | float) -> bool:
        """Greater than comparison."""
        if isinstance(other, ArithmeticMixin):
            if type(other) != type(self):
                raise TypeError(f"Cannot compare {type(self).__name__} and {type(other).__name__}")
            return self.amount > other.amount
        elif isinstance(other, (Decimal, int, float)):
            return self.amount > Decimal(str(other))
        else:
            raise TypeError(f"Cannot compare {type(self).__name__} and {type(other).__name__}")

    def __ge__(self, other: T | Decimal | int | float) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, ArithmeticMixin):
            if type(other) != type(self):
                raise TypeError(f"Cannot compare {type(self).__name__} and {type(other).__name__}")
            return self.amount >= other.amount
        elif isinstance(other, (Decimal, int, float)):
            return self.amount >= Decimal(str(other))
        else:
            raise TypeError(f"Cannot compare {type(self).__name__} and {type(other).__name__}")

    def __neg__(self: T) -> T:
        """Negate the value."""
        return self._create_new(-self.amount)

    def __abs__(self: T) -> T:
        """Get the absolute value."""
        return self._create_new(abs(self.amount))

    def __hash__(self) -> int:
        """Get hash for use in sets and dicts."""
        return hash((type(self).__name__, self.amount))
