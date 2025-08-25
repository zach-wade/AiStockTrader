"""
Domain protocol for datetime adaptation.

This protocol defines the interface for datetime objects that can be adapted
between different timezone libraries. It ensures the domain layer remains
independent of specific timezone implementations.
"""

from datetime import datetime
from typing import Protocol


class DatetimeAdapter(Protocol):
    """
    Protocol for datetime adaptation between different timezone libraries.

    This allows the domain layer to work with timezone-aware datetimes without
    depending on specific implementations (pytz, zoneinfo, etc.).
    """

    @property
    def native_datetime(self) -> datetime:
        """
        Get the underlying native datetime object.

        Returns:
            The native datetime object
        """
        ...

    def as_datetime(self) -> datetime:
        """
        Convert to standard datetime object.

        Returns:
            Standard datetime object with timezone information
        """
        ...

    def weekday(self) -> int:
        """
        Get the day of week.

        Returns:
            Day of week (0=Monday, 6=Sunday)
        """
        ...

    def time(self) -> datetime:
        """
        Get the time component.

        Returns:
            Time component of the datetime
        """
        ...
