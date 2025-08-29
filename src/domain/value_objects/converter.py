"""
Value Object Converter Utility

Centralized utility for converting between value objects and their underlying decimal values.
Eliminates DRY violations from repetitive getattr patterns across the codebase.
"""

from decimal import Decimal
from typing import Any, TypeVar, cast

T = TypeVar("T")


class ValueObjectConverter:
    """
    Utility class for converting value objects to decimal values.

    This class centralizes the common pattern of extracting decimal amounts
    from value objects while handling both value objects and raw Decimal types.
    """

    @staticmethod
    def to_decimal(value: Decimal | Any) -> Decimal:
        """
        Convert a value object or decimal to a Decimal.

        This method handles the common pattern where we need to extract
        the underlying decimal value from value objects like Money, Price,
        or Quantity, while also handling raw Decimal values.

        Args:
            value: Value object (Money, Price, Quantity) or Decimal value

        Returns:
            The underlying Decimal value

        Examples:
            >>> money = Money(Decimal("100.00"))
            >>> ValueObjectConverter.to_decimal(money)
            Decimal('100.00')

            >>> ValueObjectConverter.to_decimal(Decimal("50.00"))
            Decimal('50.00')
        """
        # Check for Money objects with 'amount' attribute
        if hasattr(value, "amount"):
            return cast(Decimal, value.amount)
        # Check for Price/Quantity objects with 'value' attribute
        elif hasattr(value, "value"):
            return cast(Decimal, value.value)
        elif isinstance(value, Decimal):
            return value
        else:
            # Fallback for other numeric types
            return Decimal(str(value))

    @staticmethod
    def to_decimal_safe(value: Decimal | Any, default: Decimal = Decimal("0")) -> Decimal:
        """
        Safely convert a value to Decimal with error handling.

        Args:
            value: Value to convert
            default: Default value to return if conversion fails

        Returns:
            The converted Decimal value or default if conversion fails
        """
        try:
            return ValueObjectConverter.to_decimal(value)
        except (ValueError, TypeError, AttributeError):
            return default

    @staticmethod
    def compare_values(left: Decimal | Any, right: Decimal | Any) -> int:
        """
        Compare two values after converting to Decimal.

        Args:
            left: First value to compare
            right: Second value to compare

        Returns:
            -1 if left < right, 0 if equal, 1 if left > right
        """
        left_decimal = ValueObjectConverter.to_decimal(left)
        right_decimal = ValueObjectConverter.to_decimal(right)

        if left_decimal < right_decimal:
            return -1
        elif left_decimal > right_decimal:
            return 1
        else:
            return 0

    @staticmethod
    def are_equal(left: Decimal | Any, right: Decimal | Any) -> bool:
        """
        Check if two values are equal after converting to Decimal.

        Args:
            left: First value to compare
            right: Second value to compare

        Returns:
            True if values are equal, False otherwise
        """
        return ValueObjectConverter.compare_values(left, right) == 0

    @staticmethod
    def extract_value(obj: Any) -> Decimal:
        """
        Extract decimal value from value objects (Price/Quantity).

        This is a specialized method for Price and Quantity objects that use
        the 'value' attribute. Provides better type safety than to_decimal.

        Args:
            obj: Price or Quantity object, or raw Decimal

        Returns:
            The underlying Decimal value
        """
        if hasattr(obj, "value"):
            return cast(Decimal, obj.value)
        elif isinstance(obj, Decimal):
            return obj
        else:
            return Decimal(str(obj))

    @staticmethod
    def extract_amount(obj: Any) -> Decimal:
        """
        Extract decimal amount from Money objects.

        This is a specialized method for Money objects that use
        the 'amount' attribute. Provides better type safety than to_decimal.

        Args:
            obj: Money object or raw Decimal

        Returns:
            The underlying Decimal amount
        """
        if hasattr(obj, "amount"):
            return cast(Decimal, obj.amount)
        elif isinstance(obj, Decimal):
            return obj
        else:
            return Decimal(str(obj))
