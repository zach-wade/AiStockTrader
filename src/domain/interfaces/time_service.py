"""
Domain time service interface for timezone and datetime operations.

This interface abstracts all time-related operations to maintain domain layer purity
according to Domain-Driven Design principles. The domain layer defines what time
operations it needs, and infrastructure provides the implementation.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime, time
from typing import Any, Protocol


class TimezoneInfo(Protocol):
    """Protocol for timezone information objects."""

    def __str__(self) -> str:
        """String representation of the timezone."""
        ...


class LocalizedDatetime(Protocol):
    """Protocol for timezone-aware datetime objects."""

    def date(self) -> date:
        """Get the date component."""
        ...

    def time(self) -> time:
        """Get the time component."""
        ...

    def weekday(self) -> int:
        """Get the weekday (0=Monday, 6=Sunday)."""
        ...

    def strftime(self, format_string: str) -> str:
        """Format datetime as string."""
        ...

    def replace(self, **kwargs: Any) -> "LocalizedDatetime":
        """Return datetime with specified components replaced."""
        ...

    def __add__(self, other: Any) -> "LocalizedDatetime":
        """Add timedelta to datetime."""
        ...

    def __sub__(self, other: Any) -> Any:
        """Subtract datetime or timedelta."""
        ...

    def __lt__(self, other: "LocalizedDatetime") -> bool:
        """Less than comparison."""
        ...

    def __le__(self, other: "LocalizedDatetime") -> bool:
        """Less than or equal comparison."""
        ...

    def __gt__(self, other: "LocalizedDatetime") -> bool:
        """Greater than comparison."""
        ...

    def __ge__(self, other: "LocalizedDatetime") -> bool:
        """Greater than or equal comparison."""
        ...

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        ...

    def as_datetime(self) -> "datetime":
        """Convert to standard datetime object."""
        ...


class TimeService(ABC):
    """
    Abstract interface for time and timezone operations.

    This service abstracts all timezone-related functionality that the domain
    layer needs, allowing for proper dependency inversion. Infrastructure
    implementations can use any timezone library (pytz, zoneinfo, etc.)
    without affecting the domain logic.

    The service provides domain-friendly operations while hiding the complexity
    of timezone handling, localization, and datetime manipulation.
    """

    @abstractmethod
    def get_timezone(self, timezone_name: str) -> TimezoneInfo:
        """
        Get timezone information for the given timezone name.

        Args:
            timezone_name: Standard timezone name (e.g., "America/New_York")

        Returns:
            TimezoneInfo: Timezone information object

        Raises:
            ValueError: If timezone name is invalid
        """

    @abstractmethod
    def get_current_time(self, timezone: TimezoneInfo) -> LocalizedDatetime:
        """
        Get current time in the specified timezone.

        Args:
            timezone: Timezone to get current time for

        Returns:
            LocalizedDatetime: Current time localized to the timezone
        """

    @abstractmethod
    def localize_naive_datetime(
        self, naive_datetime: datetime, timezone: TimezoneInfo
    ) -> LocalizedDatetime:
        """
        Convert a naive datetime to timezone-aware datetime.

        Args:
            naive_datetime: Datetime without timezone information
            timezone: Target timezone for localization

        Returns:
            LocalizedDatetime: Timezone-aware datetime

        Raises:
            ValueError: If datetime is already timezone-aware
        """

    @abstractmethod
    def convert_timezone(
        self, source_datetime: LocalizedDatetime, target_timezone: TimezoneInfo
    ) -> LocalizedDatetime:
        """
        Convert datetime from one timezone to another.

        Args:
            source_datetime: Timezone-aware datetime to convert
            target_timezone: Target timezone for conversion

        Returns:
            LocalizedDatetime: Datetime converted to target timezone
        """

    @abstractmethod
    def combine_date_time_timezone(
        self, date_part: date, time_part: time, timezone: TimezoneInfo
    ) -> LocalizedDatetime:
        """
        Combine date and time with timezone information.

        Args:
            date_part: Date component
            time_part: Time component
            timezone: Timezone to apply

        Returns:
            LocalizedDatetime: Combined timezone-aware datetime
        """

    @abstractmethod
    def is_timezone_aware(self, dt: datetime) -> bool:
        """
        Check if datetime has timezone information.

        Args:
            dt: Datetime to check

        Returns:
            bool: True if datetime is timezone-aware, False otherwise
        """

    @abstractmethod
    def format_datetime(self, dt: LocalizedDatetime, format_string: str) -> str:
        """
        Format datetime as string using the specified format.

        Args:
            dt: Datetime to format
            format_string: Format specification (e.g., "%Y-%m-%d")

        Returns:
            str: Formatted datetime string
        """

    @abstractmethod
    def get_utc_now(self) -> LocalizedDatetime:
        """
        Get current time in UTC.

        Returns:
            LocalizedDatetime: Current UTC time
        """

    @abstractmethod
    def to_utc(self, dt: LocalizedDatetime) -> LocalizedDatetime:
        """
        Convert datetime to UTC.

        Args:
            dt: Timezone-aware datetime to convert

        Returns:
            LocalizedDatetime: Datetime in UTC
        """
