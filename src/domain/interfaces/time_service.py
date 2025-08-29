"""
Domain time service interface for timezone and datetime operations.

This interface abstracts all time-related operations to maintain domain layer purity
according to Domain-Driven Design principles. The domain layer defines what time
operations it needs, and infrastructure provides the implementation.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime, time, timedelta
from datetime import tzinfo as timezone_info
from typing import Protocol, Union, overload, runtime_checkable


@runtime_checkable
class TimezoneInfo(Protocol):
    """Protocol for timezone information objects compatible with Python's tzinfo."""

    # Required tzinfo methods
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        """Return offset of local time from UTC."""
        ...

    def tzname(self, dt: datetime | None) -> str | None:
        """Return name of timezone."""
        ...

    def dst(self, dt: datetime | None) -> timedelta | None:
        """Return DST offset."""
        ...

    def __str__(self) -> str:
        """String representation of the timezone."""
        ...


@runtime_checkable
class LocalizedDatetime(Protocol):
    """Protocol for timezone-aware datetime objects compatible with datetime.

    This protocol defines the interface that timezone-aware datetime objects
    must implement to be compatible with the domain's time operations.
    It includes all essential datetime properties and methods while maintaining
    compatibility with Python's standard datetime interface.
    """

    # Essential datetime properties
    @property
    def year(self) -> int:
        """Get the year."""
        ...

    @property
    def month(self) -> int:
        """Get the month."""
        ...

    @property
    def day(self) -> int:
        """Get the day."""
        ...

    @property
    def hour(self) -> int:
        """Get the hour."""
        ...

    @property
    def minute(self) -> int:
        """Get the minute."""
        ...

    @property
    def second(self) -> int:
        """Get the second."""
        ...

    @property
    def microsecond(self) -> int:
        """Get the microsecond."""
        ...

    @property
    def tzinfo(self) -> timezone_info | None:
        """Get timezone info."""
        ...

    # Essential datetime methods
    def date(self) -> date:
        """Get the date component."""
        ...

    def time(self) -> time:
        """Get the time component."""
        ...

    def weekday(self) -> int:
        """Get the weekday (0=Monday, 6=Sunday)."""
        ...

    def isoweekday(self) -> int:
        """Get the ISO weekday (1=Monday, 7=Sunday)."""
        ...

    def strftime(self, format_string: str) -> str:
        """Format datetime as string."""
        ...

    def replace(
        self,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        hour: int | None = None,
        minute: int | None = None,
        second: int | None = None,
        microsecond: int | None = None,
        tzinfo: timezone_info | None = ...,
        fold: int = 0,
    ) -> "LocalizedDatetime":
        """Return datetime with specified components replaced."""
        ...

    def __add__(self, other: timedelta) -> "LocalizedDatetime":
        """Add timedelta to datetime."""
        ...

    @overload
    def __sub__(self, other: "LocalizedDatetime") -> timedelta:
        """Subtract datetime to get timedelta."""
        ...

    @overload
    def __sub__(self, other: timedelta) -> "LocalizedDatetime":
        """Subtract timedelta from datetime."""
        ...

    def __sub__(
        self, other: Union["LocalizedDatetime", timedelta]
    ) -> Union[timedelta, "LocalizedDatetime"]:
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

    def __ne__(self, other: object) -> bool:
        """Not equal comparison."""
        ...

    def timestamp(self) -> float:
        """Return POSIX timestamp as float."""
        ...

    def as_datetime(self) -> datetime:
        """Convert to standard datetime object.

        This method provides a way to get the underlying datetime
        for compatibility with code that requires standard datetime objects.
        """
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

    @abstractmethod
    def create_adapter(self, dt: datetime) -> LocalizedDatetime:
        """
        Create a LocalizedDatetime adapter from a standard datetime.

        Args:
            dt: Datetime object (must be timezone-aware)

        Returns:
            LocalizedDatetime: Adapted datetime object

        Raises:
            ValueError: If datetime is not timezone-aware
        """
