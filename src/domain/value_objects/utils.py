"""Utility functions for value objects."""

# Standard library imports
from decimal import Decimal


def ensure_decimal(value: Decimal | float | int | str) -> Decimal:
    """Convert value to Decimal if not already.

    Args:
        value: Value to convert to Decimal

    Returns:
        Decimal representation of the value
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))
