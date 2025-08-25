"""
Infrastructure implementation of domain time service interface.

This implementation uses Python's timezone libraries (zoneinfo with pytz fallback)
to provide concrete timezone operations for the domain layer. It handles all the
complexity of timezone conversions, localization, and datetime manipulation.
"""

import sys
from datetime import date, datetime, time
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
    elif isinstance(localized_dt, datetime):
        return localized_dt  # type: ignore
    else:
        raise TypeError(f"Cannot convert {type(localized_dt)} to datetime")


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


class LocalizedDatetimeAdapter:
    """Adapter for localized datetime from different libraries."""

    def __init__(self, dt: datetime):
        if dt.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        self._dt = dt

    def date(self) -> date:
        return self._dt.date()

    def time(self) -> time:
        return self._dt.time()

    def weekday(self) -> int:
        return self._dt.weekday()

    def strftime(self, format_string: str) -> str:
        return self._dt.strftime(format_string)

    def replace(self, **kwargs: Any) -> "LocalizedDatetime":
        new_dt = self._dt.replace(**kwargs)
        return LocalizedDatetimeAdapter(new_dt)

    def __add__(self, other: Any) -> "LocalizedDatetime":
        return LocalizedDatetimeAdapter(self._dt + other)

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, LocalizedDatetimeAdapter):
            return self._dt - other._dt
        elif isinstance(other, datetime):
            return self._dt - other
        else:  # timedelta
            return LocalizedDatetimeAdapter(self._dt - other)

    def __lt__(self, other: "LocalizedDatetime") -> bool:
        if hasattr(other, "_dt"):
            return self._dt < other._dt  # type: ignore
        elif isinstance(other, datetime):
            return self._dt < other  # type: ignore
        else:
            raise TypeError(
                f"'<' not supported between instances of 'LocalizedDatetimeAdapter' and '{type(other).__name__}'"
            )

    def __le__(self, other: "LocalizedDatetime") -> bool:
        if hasattr(other, "_dt"):
            return self._dt <= other._dt  # type: ignore
        elif isinstance(other, datetime):
            return self._dt <= other  # type: ignore
        else:
            raise TypeError(
                f"'<=' not supported between instances of 'LocalizedDatetimeAdapter' and '{type(other).__name__}'"
            )

    def __gt__(self, other: "LocalizedDatetime") -> bool:
        if hasattr(other, "_dt"):
            return self._dt > other._dt  # type: ignore
        elif isinstance(other, datetime):
            return self._dt > other  # type: ignore
        else:
            raise TypeError(
                f"'>' not supported between instances of 'LocalizedDatetimeAdapter' and '{type(other).__name__}'"
            )

    def __ge__(self, other: "LocalizedDatetime") -> bool:
        if hasattr(other, "_dt"):
            return self._dt >= other._dt  # type: ignore
        elif isinstance(other, datetime):
            return self._dt >= other  # type: ignore
        else:
            raise TypeError(
                f"'>=' not supported between instances of 'LocalizedDatetimeAdapter' and '{type(other).__name__}'"
            )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LocalizedDatetimeAdapter):
            return self._dt == other._dt
        elif isinstance(other, datetime):
            return self._dt == other
        return False

    def as_datetime(self) -> datetime:
        """Convert to standard datetime object."""
        return self._dt

    @property
    def native_datetime(self) -> datetime:
        """Get the underlying datetime object."""
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
                tz = ZoneInfo(timezone_name)
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

        return LocalizedDatetimeAdapter(utc_now)

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

        return LocalizedDatetimeAdapter(localized)

    def convert_timezone(
        self, source_datetime: LocalizedDatetime, target_timezone: TimezoneInfo
    ) -> LocalizedDatetime:
        """Convert datetime from one timezone to another."""
        if hasattr(source_datetime, "native_datetime"):
            source_dt = to_datetime(source_datetime)
        elif hasattr(source_datetime, "_dt"):
            source_dt = source_datetime._dt
        else:
            source_dt = source_datetime

        if source_dt.tzinfo is None:
            raise ValueError("Source datetime must be timezone-aware")

        # Convert to target timezone
        tz_adapter = (
            target_timezone
            if isinstance(target_timezone, TimezoneInfoAdapter)
            else TimezoneInfoAdapter(target_timezone, str(target_timezone))
        )
        target_dt = source_dt.astimezone(tz_adapter._tz)

        return LocalizedDatetimeAdapter(target_dt)

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

        return LocalizedDatetimeAdapter(utc_now)

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

        return LocalizedDatetimeAdapter(utc_dt)
