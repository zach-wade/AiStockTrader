"""
Market hours domain service.

This module contains all market hours and market status business logic
that was previously in the infrastructure layer. It determines market
status based on current time and market calendar rules.
"""

from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any, ClassVar

import pytz


class MarketStatus(Enum):
    """Market status enumeration."""

    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    OPEN = "open"
    AFTER_MARKET = "after_market"
    HOLIDAY = "holiday"


class MarketHoursService:
    """
    Domain service for market hours and status determination.

    This service encapsulates all business logic related to market hours,
    trading sessions, and market status determination.
    """

    # Default NYSE market hours (business rules)
    DEFAULT_TIMEZONE = "America/New_York"

    # Pre-market hours (4:00 AM - 9:30 AM ET)
    PRE_MARKET_OPEN_HOUR = 4
    PRE_MARKET_OPEN_MINUTE = 0

    # Regular market hours (9:30 AM - 4:00 PM ET)
    REGULAR_OPEN_HOUR = 9
    REGULAR_OPEN_MINUTE = 30
    REGULAR_CLOSE_HOUR = 16
    REGULAR_CLOSE_MINUTE = 0

    # After-market hours (4:00 PM - 8:00 PM ET)
    AFTER_MARKET_CLOSE_HOUR = 20
    AFTER_MARKET_CLOSE_MINUTE = 0

    # Major US market holidays for 2024-2025 (business rules)
    DEFAULT_HOLIDAYS: ClassVar[set[str]] = {
        # 2024 Holidays
        "2024-01-01",  # New Year's Day
        "2024-01-15",  # Martin Luther King Jr. Day
        "2024-02-19",  # Presidents' Day
        "2024-03-29",  # Good Friday
        "2024-05-27",  # Memorial Day
        "2024-06-19",  # Juneteenth
        "2024-07-04",  # Independence Day
        "2024-09-02",  # Labor Day
        "2024-11-28",  # Thanksgiving Day
        "2024-12-25",  # Christmas Day
        # 2025 Holidays
        "2025-01-01",  # New Year's Day
        "2025-01-20",  # Martin Luther King Jr. Day
        "2025-02-17",  # Presidents' Day
        "2025-04-18",  # Good Friday
        "2025-05-26",  # Memorial Day
        "2025-06-19",  # Juneteenth
        "2025-07-04",  # Independence Day
        "2025-09-01",  # Labor Day
        "2025-11-27",  # Thanksgiving Day
        "2025-12-25",  # Christmas Day
    }

    def __init__(
        self,
        timezone: str | None = None,
        holidays: set[str] | None = None,
        pre_market_open: time | None = None,
        regular_open: time | None = None,
        regular_close: time | None = None,
        after_market_close: time | None = None,
    ):
        """
        Initialize market hours service with configurable parameters.

        Args:
            timezone: Market timezone (defaults to NYSE timezone)
            holidays: Set of holiday dates in YYYY-MM-DD format
            pre_market_open: Pre-market session open time
            regular_open: Regular market session open time
            regular_close: Regular market session close time
            after_market_close: After-market session close time
        """
        self.timezone_str = timezone or self.DEFAULT_TIMEZONE
        self.timezone = pytz.timezone(self.timezone_str)
        self.holidays = holidays or self.DEFAULT_HOLIDAYS.copy()

        # Market hours configuration
        self.pre_market_open = pre_market_open or time(
            self.PRE_MARKET_OPEN_HOUR, self.PRE_MARKET_OPEN_MINUTE
        )
        self.regular_open = regular_open or time(self.REGULAR_OPEN_HOUR, self.REGULAR_OPEN_MINUTE)
        self.regular_close = regular_close or time(
            self.REGULAR_CLOSE_HOUR, self.REGULAR_CLOSE_MINUTE
        )
        self.after_market_close = after_market_close or time(
            self.AFTER_MARKET_CLOSE_HOUR, self.AFTER_MARKET_CLOSE_MINUTE
        )

    def get_current_market_status(self, current_time: datetime | None = None) -> MarketStatus:
        """
        Determine current market status based on business rules.

        Args:
            current_time: Optional datetime to check (defaults to current time)

        Returns:
            MarketStatus enum value
        """
        # Use provided time or current time
        if current_time is None:
            now = datetime.now(self.timezone)
        elif current_time.tzinfo is None:
            now = self.timezone.localize(current_time)
        else:
            now = current_time.astimezone(self.timezone)

        # Business rule: Check if today is a holiday
        if self.is_holiday(now):
            return MarketStatus.HOLIDAY

        # Business rule: Markets are closed on weekends
        if self.is_weekend(now):
            return MarketStatus.CLOSED

        # Determine status based on time of day
        current_time_only = now.time()

        # Business rule: Pre-market hours
        if self.pre_market_open <= current_time_only < self.regular_open:
            return MarketStatus.PRE_MARKET

        # Business rule: Regular market hours
        elif self.regular_open <= current_time_only < self.regular_close:
            return MarketStatus.OPEN

        # Business rule: After-market hours
        elif self.regular_close <= current_time_only < self.after_market_close:
            return MarketStatus.AFTER_MARKET

        # Business rule: Outside trading hours
        else:
            return MarketStatus.CLOSED

    def is_holiday(self, date: datetime) -> bool:
        """
        Check if a given date is a market holiday.

        Args:
            date: Date to check

        Returns:
            True if the date is a holiday, False otherwise
        """
        date_str = date.strftime("%Y-%m-%d")
        return date_str in self.holidays

    def is_weekend(self, date: datetime) -> bool:
        """
        Check if a given date is a weekend.

        Args:
            date: Date to check

        Returns:
            True if the date is Saturday or Sunday, False otherwise
        """
        # Business rule: Saturday = 5, Sunday = 6
        return date.weekday() >= 5

    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if a given date is a trading day.

        Args:
            date: Date to check

        Returns:
            True if markets are open on this date, False otherwise
        """
        return not (self.is_weekend(date) or self.is_holiday(date))

    def is_market_open(self, current_time: datetime | None = None) -> bool:
        """
        Check if the market is currently open for regular trading.

        Args:
            current_time: Optional datetime to check

        Returns:
            True if market is open for regular trading, False otherwise
        """
        status = self.get_current_market_status(current_time)
        return status == MarketStatus.OPEN

    def is_extended_hours(self, current_time: datetime | None = None) -> bool:
        """
        Check if we're in extended trading hours (pre-market or after-market).

        Args:
            current_time: Optional datetime to check

        Returns:
            True if in extended hours, False otherwise
        """
        status = self.get_current_market_status(current_time)
        return status in [MarketStatus.PRE_MARKET, MarketStatus.AFTER_MARKET]

    def get_next_market_open(self, from_time: datetime | None = None) -> datetime | None:
        """
        Get the next market open time from a given datetime.

        Args:
            from_time: Starting datetime (defaults to current time)

        Returns:
            Next market open datetime, or None if unable to determine
        """
        if from_time is None:
            current = datetime.now(self.timezone)
        elif from_time.tzinfo is None:
            current = self.timezone.localize(from_time)
        else:
            current = from_time.astimezone(self.timezone)

        # Start checking from current time
        check_date = current

        # Look up to 10 days ahead (handles long weekends and holidays)
        for _ in range(10):
            # If it's a trading day
            if self.is_trading_day(check_date):
                # Create market open time for this date
                market_open = check_date.replace(
                    hour=self.regular_open.hour,
                    minute=self.regular_open.minute,
                    second=0,
                    microsecond=0,
                )

                # If market open is in the future, return it
                if market_open > current:
                    return market_open

            # Move to next day
            check_date = check_date.replace(hour=0, minute=0, second=0, microsecond=0)
            check_date += timedelta(days=1)

        return None

    def get_next_market_close(self, from_time: datetime | None = None) -> datetime | None:
        """
        Get the next market close time from a given datetime.

        Args:
            from_time: Starting datetime (defaults to current time)

        Returns:
            Next market close datetime, or None if unable to determine
        """
        if from_time is None:
            current = datetime.now(self.timezone)
        elif from_time.tzinfo is None:
            current = self.timezone.localize(from_time)
        else:
            current = from_time.astimezone(self.timezone)

        # If market is currently open, return today's close
        if self.is_market_open(current):
            return current.replace(
                hour=self.regular_close.hour,
                minute=self.regular_close.minute,
                second=0,
                microsecond=0,
            )

        # Otherwise, find next open and then close
        next_open = self.get_next_market_open(current)
        if next_open:
            return next_open.replace(
                hour=self.regular_close.hour,
                minute=self.regular_close.minute,
                second=0,
                microsecond=0,
            )

        return None

    def add_holiday(self, date: str) -> None:
        """
        Add a holiday to the market calendar.

        Args:
            date: Holiday date in YYYY-MM-DD format
        """
        self.holidays.add(date)

    def remove_holiday(self, date: str) -> None:
        """
        Remove a holiday from the market calendar.

        Args:
            date: Holiday date in YYYY-MM-DD format
        """
        self.holidays.discard(date)

    def get_market_hours_info(self) -> dict[str, Any]:
        """
        Get comprehensive market hours information.

        Returns:
            Dictionary containing market hours configuration
        """
        return {
            "timezone": self.timezone_str,
            "pre_market_open": self.pre_market_open.strftime("%H:%M"),
            "regular_open": self.regular_open.strftime("%H:%M"),
            "regular_close": self.regular_close.strftime("%H:%M"),
            "after_market_close": self.after_market_close.strftime("%H:%M"),
            "holidays_count": len(self.holidays),
        }

    def get_regular_market_hours(self, trading_day: date) -> dict[str, Any] | None:
        """
        Get regular market hours for a specific trading day.

        Args:
            trading_day: The date to get market hours for

        Returns:
            Dictionary with open and close times, or None if not a trading day
        """
        # Convert date to datetime for checking
        dt = datetime.combine(trading_day, time(12, 0))

        if not self.is_trading_day(dt):
            return None

        # Localize the times for the given day
        market_open = self.timezone.localize(datetime.combine(trading_day, self.regular_open))
        market_close = self.timezone.localize(datetime.combine(trading_day, self.regular_close))

        return {
            "open": market_open,
            "close": market_close,
            "is_holiday": self.is_holiday(dt),
            "is_weekend": self.is_weekend(dt),
        }

    def get_extended_market_hours(self, trading_day: date) -> dict[str, Any] | None:
        """
        Get extended market hours for a specific trading day.

        Args:
            trading_day: The date to get extended hours for

        Returns:
            Dictionary with pre-market and after-hours times, or None if not a trading day
        """
        # Convert date to datetime for checking
        dt = datetime.combine(trading_day, time(12, 0))

        if not self.is_trading_day(dt):
            return None

        # Localize the times for the given day
        pre_market_open = self.timezone.localize(
            datetime.combine(trading_day, self.pre_market_open)
        )
        after_market_close = self.timezone.localize(
            datetime.combine(trading_day, self.after_market_close)
        )

        return {
            "pre_market_open": pre_market_open,
            "pre_market_close": self.timezone.localize(
                datetime.combine(trading_day, self.regular_open)
            ),
            "regular_open": self.timezone.localize(
                datetime.combine(trading_day, self.regular_open)
            ),
            "regular_close": self.timezone.localize(
                datetime.combine(trading_day, self.regular_close)
            ),
            "after_market_open": self.timezone.localize(
                datetime.combine(trading_day, self.regular_close)
            ),
            "after_market_close": after_market_close,
        }

    def get_next_trading_day(self, from_date: date | None = None) -> date:
        """
        Get the next trading day from a given date.

        Args:
            from_date: Date to start from (defaults to today)

        Returns:
            The next trading day
        """
        if from_date is None:
            from_date = datetime.now(self.timezone).date()

        # Start checking from the next day
        next_day = from_date + timedelta(days=1)

        # Keep incrementing until we find a trading day
        while not self.is_trading_day(datetime.combine(next_day, time(12, 0))):
            next_day += timedelta(days=1)

        return next_day

    def get_previous_trading_day(self, from_date: date | None = None) -> date:
        """
        Get the previous trading day from a given date.

        Args:
            from_date: Date to start from (defaults to today)

        Returns:
            The previous trading day
        """
        if from_date is None:
            from_date = datetime.now(self.timezone).date()

        # Start checking from the previous day
        prev_day = from_date - timedelta(days=1)

        # Keep decrementing until we find a trading day
        while not self.is_trading_day(datetime.combine(prev_day, time(12, 0))):
            prev_day -= timedelta(days=1)

        return prev_day

    def time_until_market_open(self, from_time: datetime | None = None) -> timedelta | None:
        """
        Calculate time until next market open.

        Args:
            from_time: Time to calculate from (defaults to now)

        Returns:
            Timedelta until market opens, or None if market is currently open
        """
        if from_time is None:
            from_time = datetime.now(self.timezone)
        elif from_time.tzinfo is None:
            from_time = self.timezone.localize(from_time)

        # If market is currently open, return None
        if self.is_market_open(from_time):
            return None

        # Get next market open
        next_open = self.get_next_market_open(from_time)
        if next_open:
            return next_open - from_time

        return None

    def get_time_until_market_open(self, from_time: datetime | None = None) -> timedelta | None:
        """Alias for time_until_market_open for backward compatibility."""
        return self.time_until_market_open(from_time)

    def get_time_until_market_close(self, from_time: datetime | None = None) -> timedelta | None:
        """
        Calculate time until market close.

        Args:
            from_time: Time to calculate from (defaults to now)

        Returns:
            Timedelta until market closes, or None if market is closed
        """
        if from_time is None:
            from_time = datetime.now(self.timezone)
        elif from_time.tzinfo is None:
            from_time = self.timezone.localize(from_time)

        # Get next market close
        next_close = self.get_next_market_close(from_time)
        if next_close:
            return next_close - from_time

        return None
