"""
Infrastructure implementation of domain time service interface.

This implementation uses Python's timezone libraries (zoneinfo with pytz fallback)
to provide concrete timezone operations for the domain layer. It handles all the
complexity of timezone conversions, localization, and datetime manipulation.
"""

import sys
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any

# Use zoneinfo for Python 3.9+ with pytz fallback for older versions
if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo

    HAS_ZONEINFO = True
    try:
        import pytz

        HAS_PYTZ = True
    except ImportError:
        HAS_PYTZ = False
else:
    HAS_ZONEINFO = False
    import pytz

    HAS_PYTZ = True

from src.domain.interfaces.time_service import LocalizedDatetime, TimeService, TimezoneInfo


def to_datetime(localized_dt: LocalizedDatetime) -> datetime:
    """Convert LocalizedDatetime to standard datetime."""
    if hasattr(localized_dt, "as_datetime"):
        return localized_dt.as_datetime()
    elif hasattr(localized_dt, "_dt"):
        return localized_dt._dt  # type: ignore
    else:
        # Assume it's a datetime or compatible type
        return localized_dt  # type: ignore


class TimezoneInfoAdapter:
    """Adapter for timezone information from different libraries."""

    def __init__(self, tz: Any, name: str):
        self._tz = tz
        self._name = name

    def __str__(self) -> str:
        return self._name

    @property
    def native_tz(self) -> Any:
        """Get the underlying timezone object."""
        return self._tz

    # TimezoneInfo protocol methods
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        """Return offset of local time from UTC."""
        result = self._tz.utcoffset(dt)
        return result if isinstance(result, (timedelta, type(None))) else None

    def tzname(self, dt: datetime | None) -> str | None:
        """Return name of timezone."""
        result = self._tz.tzname(dt)
        return str(result) if result is not None else None

    def dst(self, dt: datetime | None) -> timedelta | None:
        """Return DST offset."""
        result = self._tz.dst(dt)
        return result if isinstance(result, (timedelta, type(None))) else None


class LocalizedDatetimeAdapter:
    """Adapter for localized datetime from different libraries.

    This adapter implements the LocalizedDatetime protocol by wrapping
    a standard datetime object and providing all required properties and methods.
    """

    def __init__(self, dt: datetime):
        if dt.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        self._dt = dt

    # Properties required by LocalizedDatetime protocol
    @property
    def year(self) -> int:
        return self._dt.year

    @property
    def month(self) -> int:
        return self._dt.month

    @property
    def day(self) -> int:
        return self._dt.day

    @property
    def hour(self) -> int:
        return self._dt.hour

    @property
    def minute(self) -> int:
        return self._dt.minute

    @property
    def second(self) -> int:
        return self._dt.second

    @property
    def microsecond(self) -> int:
        return self._dt.microsecond

    @property
    def tzinfo(self) -> tzinfo | None:
        return self._dt.tzinfo

    # Methods required by LocalizedDatetime protocol
    def date(self) -> date:
        return self._dt.date()

    def time(self) -> time:
        return self._dt.time()

    def weekday(self) -> int:
        return self._dt.weekday()

    def isoweekday(self) -> int:
        return self._dt.isoweekday()

    def strftime(self, format_string: str) -> str:
        return self._dt.strftime(format_string)

    def timestamp(self) -> float:
        return self._dt.timestamp()

    def replace(
        self,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        hour: int | None = None,
        minute: int | None = None,
        second: int | None = None,
        microsecond: int | None = None,
        tzinfo: "tzinfo | None" = ...,  # type: ignore
        fold: int = 0,
    ) -> "LocalizedDatetimeAdapter":
        """Return datetime with specified components replaced."""
        # Build kwargs dictionary, filtering out None values
        replace_kwargs: dict[str, Any] = {}
        if year is not None:
            replace_kwargs["year"] = year
        if month is not None:
            replace_kwargs["month"] = month
        if day is not None:
            replace_kwargs["day"] = day
        if hour is not None:
            replace_kwargs["hour"] = hour
        if minute is not None:
            replace_kwargs["minute"] = minute
        if second is not None:
            replace_kwargs["second"] = second
        if microsecond is not None:
            replace_kwargs["microsecond"] = microsecond
        if tzinfo is not ...:
            replace_kwargs["tzinfo"] = tzinfo
        if fold != 0:
            replace_kwargs["fold"] = fold

        new_dt = self._dt.replace(**replace_kwargs)
        return LocalizedDatetimeAdapter(new_dt)

    def __add__(self, other: timedelta) -> "LocalizedDatetimeAdapter":
        """Add timedelta to datetime."""
        return LocalizedDatetimeAdapter(self._dt + other)

    def __sub__(
        self, other: "LocalizedDatetimeAdapter | timedelta"
    ) -> "LocalizedDatetimeAdapter | timedelta":
        """Subtract datetime or timedelta."""
        if isinstance(other, LocalizedDatetimeAdapter):
            result = self._dt - other._dt
            return result
        elif hasattr(other, "as_datetime"):
            result = self._dt - other.as_datetime()
            return result
        elif isinstance(other, timedelta):
            return LocalizedDatetimeAdapter(self._dt - other)
        else:
            # This should never happen due to type annotation, but kept for safety
            return NotImplemented  # type: ignore[unreachable]

    def __lt__(self, other: "LocalizedDatetime") -> bool:
        """Less than comparison."""
        if hasattr(other, "as_datetime"):
            return self._dt < other.as_datetime()
        elif hasattr(other, "_dt"):
            return self._dt < other._dt  # type: ignore
        else:
            return NotImplemented

    def __le__(self, other: "LocalizedDatetime") -> bool:
        """Less than or equal comparison."""
        if hasattr(other, "as_datetime"):
            return self._dt <= other.as_datetime()
        elif hasattr(other, "_dt"):
            return self._dt <= other._dt  # type: ignore
        else:
            return NotImplemented

    def __gt__(self, other: "LocalizedDatetime") -> bool:
        """Greater than comparison."""
        if hasattr(other, "as_datetime"):
            return self._dt > other.as_datetime()
        elif hasattr(other, "_dt"):
            return self._dt > other._dt  # type: ignore
        else:
            return NotImplemented

    def __ge__(self, other: "LocalizedDatetime") -> bool:
        """Greater than or equal comparison."""
        if hasattr(other, "as_datetime"):
            return self._dt >= other.as_datetime()
        elif hasattr(other, "_dt"):
            return self._dt >= other._dt  # type: ignore
        else:
            return NotImplemented

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if isinstance(other, LocalizedDatetimeAdapter):
            return self._dt == other._dt
        elif hasattr(other, "as_datetime"):
            return self._dt == other.as_datetime()  # type: ignore
        elif isinstance(other, datetime):
            return self._dt == other
        return False

    def __ne__(self, other: object) -> bool:
        """Not equal comparison."""
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def as_datetime(self) -> datetime:
        """Convert to standard datetime object.

        This method provides compatibility with code that requires
        standard datetime objects.
        """
        return self._dt

    @property
    def native_datetime(self) -> datetime:
        """Get the underlying datetime object.

        This property is deprecated. Use as_datetime() instead.
        """
        return self._dt


class PythonTimeService(TimeService):
    """
    Python implementation of the TimeService interface.

    Uses zoneinfo (Python 3.9+) with pytz fallback for timezone operations.
    This implementation handles all timezone complexity while providing a
    clean interface for the domain layer.

    Features:
    - Automatic library selection (zoneinfo preferred, pytz fallback)
    - Robust error handling for invalid timezones
    - Efficient timezone conversions
    - Support for both naive and aware datetime objects
    """

    def __init__(self, prefer_zoneinfo: bool = True):
        """
        Initialize the time service.

        Args:
            prefer_zoneinfo: If True, prefer zoneinfo over pytz when available
        """
        self._prefer_zoneinfo = prefer_zoneinfo and HAS_ZONEINFO

    def get_timezone(self, timezone_name: str) -> TimezoneInfo:
        """Get timezone information for the given timezone name."""
        try:
            if self._prefer_zoneinfo and HAS_ZONEINFO:
                tz: Any | tzinfo = ZoneInfo(timezone_name)
            elif HAS_PYTZ:
                tz = pytz.timezone(timezone_name)
            else:
                raise RuntimeError("No timezone library available")

            return TimezoneInfoAdapter(tz, timezone_name)

        except Exception as e:
            raise ValueError(f"Invalid timezone name: {timezone_name}") from e

    def get_current_time(self, timezone: TimezoneInfo) -> LocalizedDatetime:
        """Get current time in the specified timezone."""
        # Cast to our adapter since we control the implementation
        tz_adapter = (
            timezone
            if isinstance(timezone, TimezoneInfoAdapter)
            else TimezoneInfoAdapter(timezone, str(timezone))
        )

        if self._prefer_zoneinfo and HAS_ZONEINFO:
            # zoneinfo approach
            utc_now = datetime.now(tz_adapter._tz)
        elif HAS_PYTZ:
            # pytz approach
            utc_now = datetime.now(tz_adapter._tz)
        else:
            raise RuntimeError("No timezone library available")

        return LocalizedDatetimeAdapter(utc_now)  # type: ignore[return-value]  # type: ignore[return-value]

    def localize_naive_datetime(
        self, naive_datetime: datetime, timezone: TimezoneInfo
    ) -> LocalizedDatetime:
        """Convert a naive datetime to timezone-aware datetime."""
        if naive_datetime.tzinfo is not None:
            raise ValueError("Datetime is already timezone-aware")

        # Cast to our adapter since we control the implementation
        tz_adapter = (
            timezone
            if isinstance(timezone, TimezoneInfoAdapter)
            else TimezoneInfoAdapter(timezone, str(timezone))
        )

        if self._prefer_zoneinfo and HAS_ZONEINFO:
            # zoneinfo approach - replace tzinfo
            localized = naive_datetime.replace(tzinfo=tz_adapter._tz)
        elif HAS_PYTZ:
            # pytz approach - use localize method to handle DST correctly
            localized = tz_adapter._tz.localize(naive_datetime)
        else:
            raise RuntimeError("No timezone library available")

        return LocalizedDatetimeAdapter(localized)  # type: ignore[return-value]

    def convert_timezone(
        self, source_datetime: LocalizedDatetime, target_timezone: TimezoneInfo
    ) -> LocalizedDatetime:
        """Convert datetime from one timezone to another."""
        if hasattr(source_datetime, "native_datetime"):
            source_dt = to_datetime(source_datetime)
        elif hasattr(source_datetime, "_dt"):
            source_dt = source_datetime._dt
        else:
            source_dt = source_datetime  # type: ignore[assignment]

        if source_dt.tzinfo is None:
            raise ValueError("Source datetime must be timezone-aware")

        # Convert to target timezone
        tz_adapter = (
            target_timezone
            if isinstance(target_timezone, TimezoneInfoAdapter)
            else TimezoneInfoAdapter(target_timezone, str(target_timezone))
        )
        target_dt = source_dt.astimezone(tz_adapter._tz)

        return LocalizedDatetimeAdapter(target_dt)  # type: ignore[return-value]

    def combine_date_time_timezone(
        self, date_part: date, time_part: time, timezone: TimezoneInfo
    ) -> LocalizedDatetime:
        """Combine date and time with timezone information."""
        # First create naive datetime
        naive_dt = datetime.combine(date_part, time_part)

        # Then localize it
        return self.localize_naive_datetime(naive_dt, timezone)

    def is_timezone_aware(self, dt: datetime) -> bool:
        """Check if datetime has timezone information."""
        return dt.tzinfo is not None

    def format_datetime(self, dt: LocalizedDatetime, format_string: str) -> str:
        """Format datetime as string using the specified format."""
        return to_datetime(dt).strftime(format_string)

    def get_utc_now(self) -> LocalizedDatetime:
        """Get current time in UTC."""
        if self._prefer_zoneinfo and HAS_ZONEINFO:
            from datetime import UTC

            utc_now = datetime.now(UTC)
        elif HAS_PYTZ:
            utc_now = datetime.now(pytz.UTC)
        else:
            raise RuntimeError("No timezone library available")

        return LocalizedDatetimeAdapter(utc_now)  # type: ignore[return-value]

    def to_utc(self, dt: LocalizedDatetime) -> LocalizedDatetime:
        """Convert datetime to UTC."""
        source_dt = to_datetime(dt)

        if source_dt.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")

        if self._prefer_zoneinfo and HAS_ZONEINFO:
            from datetime import UTC

            utc_dt = source_dt.astimezone(UTC)
        elif HAS_PYTZ:
            utc_dt = source_dt.astimezone(pytz.UTC)
        else:
            raise RuntimeError("No timezone library available")

        return LocalizedDatetimeAdapter(utc_dt)  # type: ignore[return-value]

    def create_adapter(self, dt: datetime) -> LocalizedDatetime:
        """Create a LocalizedDatetime adapter from a standard datetime."""
        if dt.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return LocalizedDatetimeAdapter(dt)  # type: ignore[return-value]
