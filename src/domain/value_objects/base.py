"""Base class for value objects."""

# Standard library imports
from abc import ABC, abstractmethod
from functools import total_ordering
from typing import Any, Self


class ValueObject(ABC):
    """Abstract base class for all value objects.

    Provides common functionality for value objects including:
    - Immutability enforcement
    - Equality comparison
    - Hashability
    """

    __slots__ = ()  # Subclasses should define their own __slots__

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check equality with another value object."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Get hash for use in sets/dicts."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """Get string representation for debugging."""
        pass

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification after initialization."""
        if hasattr(self, name):
            raise AttributeError(f"Cannot modify immutable value object attribute '{name}'")
        super().__setattr__(name, value)


@total_ordering
class ComparableValueObject(ValueObject):
    """Base class for value objects that support comparison operations.

    Subclasses need only implement __eq__ and __lt__ thanks to @total_ordering.
    """

    @abstractmethod
    def __lt__(self, other: Self) -> bool:
        """Check if less than another value object."""
        pass
