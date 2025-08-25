"""
Trading Calendar - Domain service for market hours and trading days.

This module provides the TradingCalendar service which manages market hours,
trading days, and holiday schedules for various exchanges. It's essential for
determining when orders can be placed and when markets are open for trading.

The TradingCalendar encapsulates complex market scheduling logic including regular
hours, extended hours, holidays, and special cases like Forex market hours that
span multiple days.

Key Responsibilities:
    - Determining if markets are open at a given time
    - Managing trading holidays for different exchanges
    - Calculating next market open/close times
    - Supporting multiple exchanges (NYSE, NASDAQ, Forex, Crypto)
    - Handling extended trading hours (pre-market, after-hours)

Supported Exchanges:
    - NYSE/NASDAQ: US equity markets with standard and extended hours
    - CME: Futures markets (placeholder for future implementation)
    - FOREX: 24/5 foreign exchange markets
    - CRYPTO: 24/7 cryptocurrency markets

Design Patterns:
    - Enum Pattern: Exchange types for type safety
    - Value Object: MarketHours and TradingSession encapsulate schedule data
    - Factory Pattern: Could be extended to create exchange-specific calendars

Architectural Decisions:
    - Uses ZoneInfo for proper timezone handling (Python 3.9+)
    - Holidays hardcoded for simplicity (production would use external data)
    - Separate handling for each exchange type's unique characteristics
    - Returns UTC times for consistency across timezones

Example:
    >>> from datetime import datetime
    >>> from domain.services import TradingCalendar
    >>> from domain.services.trading_calendar import Exchange
    >>>
    >>> calendar = TradingCalendar(Exchange.NYSE)
    >>> if calendar.is_market_open():
    ...     print("NYSE is open for trading")
    >>> next_open = calendar.next_market_open()
    >>> print(f"Market opens at: {next_open}")

Note:
    Holiday schedules are simplified and would need regular updates in production.
    Consider using a market data provider's calendar API for accurate holidays.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum

from src.domain.interfaces.time_service import TimeService, TimezoneInfo


class Exchange(Enum):
    """Supported exchanges.

    Enumeration of supported exchange types, each with different trading hours
    and holiday schedules.

    Values:
        NYSE: New York Stock Exchange
        NASDAQ: NASDAQ Stock Market
        CME: Chicago Mercantile Exchange (futures)
        FOREX: Foreign Exchange Market
        CRYPTO: Cryptocurrency exchanges
    """

    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    CME = "CME"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"


@dataclass
class MarketHours:
    """Market hours for a trading session.

    Represents the operating hours for a specific trading session (regular,
    pre-market, or after-hours). Handles timezone conversions and time comparisons.

    Attributes:
        open_time: Time when the session opens (in market timezone).
        close_time: Time when the session closes (in market timezone).
        timezone: Timezone for the market (e.g., America/New_York for NYSE).
        time_service: Service for timezone operations.
        is_open: Whether this session is active (default True).

    Note:
        For overnight sessions (e.g., some futures), close_time may be before
        open_time, indicating the session spans midnight.
    """

    open_time: time
    close_time: time
    timezone: TimezoneInfo
    time_service: TimeService
    is_open: bool = True

    def is_time_in_session(self, check_time: datetime) -> bool:
        """Check if given time is within market hours.

        Determines if a specific time falls within this trading session,
        handling timezone conversions and overnight sessions.

        Args:
            check_time: Time to check (any timezone, will be converted).

        Returns:
            bool: True if the time is within session hours, False otherwise.

        Example:
            >>> from src.infrastructure.time import PythonTimeService
            >>> time_service = PythonTimeService()
            >>> tz = time_service.get_timezone("America/New_York")
            >>> market_hours = MarketHours(
            ...     open_time=time(9, 30),
            ...     close_time=time(16, 0),
            ...     timezone=tz,
            ...     time_service=time_service
            ... )
            >>> check_time = time_service.get_utc_now()
            >>> if market_hours.is_time_in_session(check_time.as_datetime()):
            ...     print("Market is in session")
        """
        if not self.is_open:
            return False

        # Convert to market timezone
        if self.time_service.is_timezone_aware(check_time):
            # Use time service to create adapter instead of importing from infrastructure
            adapter = self.time_service.create_adapter(check_time)
            market_time = self.time_service.convert_timezone(adapter, self.timezone)
        else:
            market_time = self.time_service.localize_naive_datetime(check_time, self.timezone)

        current_time = market_time.time()

        # Handle regular hours
        if self.open_time <= self.close_time:
            return self.open_time <= current_time <= self.close_time
        else:
            # Handle overnight sessions (e.g., futures)
            return current_time >= self.open_time or current_time <= self.close_time


@dataclass
class TradingSession:
    """Complete trading session information.

    Encapsulates all trading sessions for an exchange including regular hours
    and extended trading sessions.

    Attributes:
        regular_hours: Primary trading session hours.
        pre_market: Optional pre-market trading session.
        after_hours: Optional after-hours trading session.

    Note:
        Not all exchanges have extended hours. Forex and crypto markets
        typically only have regular hours that run continuously.
    """

    regular_hours: MarketHours
    pre_market: MarketHours | None = None
    after_hours: MarketHours | None = None

    def is_market_open(self, check_time: datetime, extended_hours: bool = False) -> bool:
        """Check if market is open at given time.

        Determines if any trading session is active at the specified time.

        Args:
            check_time: Time to check for market availability.
            extended_hours: If True, includes pre-market and after-hours sessions.
                If False, only checks regular trading hours.

        Returns:
            bool: True if market is open in any applicable session.

        Example:
            >>> session = TradingSession(regular_hours=...)
            >>> # Check if market is open including extended hours
            >>> if session.is_market_open(datetime.now(UTC), extended_hours=True):
            ...     print("Market is open for extended hours trading")
        """
        # Check regular hours
        if self.regular_hours.is_time_in_session(check_time):
            return True

        # Check extended hours if requested
        if extended_hours:
            if self.pre_market and self.pre_market.is_time_in_session(check_time):
                return True
            if self.after_hours and self.after_hours.is_time_in_session(check_time):
                return True

        return False


class TradingCalendar:
    """Service for managing trading calendars and market hours.

    Provides comprehensive market schedule management including trading days,
    market hours, and holiday handling for various exchanges. This service is
    essential for determining when orders can be placed and executed.

    Attributes:
        exchange: The exchange this calendar is configured for.
        session: Trading session configuration for the exchange.

    Class Attributes:
        US_MARKET_HOLIDAYS: Set of US market holidays (2024-2025).
        EXCHANGE_SESSIONS: Trading session configurations per exchange.

    Note:
        Holiday lists are simplified and would need regular updates in production.
        Consider using a market data provider's calendar API for accuracy.
    """

    # US market holidays for 2024-2025 (simplified)
    US_MARKET_HOLIDAYS = {
        date(2024, 1, 1),  # New Year's Day
        date(2024, 1, 15),  # MLK Day
        date(2024, 2, 19),  # Presidents Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day
        date(2024, 9, 2),  # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
        date(2025, 1, 1),  # New Year's Day
        date(2025, 1, 20),  # MLK Day
        date(2025, 2, 17),  # Presidents Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),  # Independence Day
        date(2025, 9, 1),  # Labor Day
        date(2025, 11, 27),  # Thanksgiving
        date(2025, 12, 25),  # Christmas
    }

    # Note: Exchange sessions are now created dynamically in _create_exchange_session()
    # to properly inject the time service dependency

    def __init__(self, time_service: TimeService, exchange: Exchange = Exchange.NYSE) -> None:
        """Initialize calendar for specific exchange.

        Args:
            time_service: Service for timezone and datetime operations
            exchange: Exchange to create calendar for (default NYSE).

        Example:
            >>> from src.infrastructure.time import PythonTimeService
            >>> time_service = PythonTimeService()
            >>> nyse_calendar = TradingCalendar(time_service, Exchange.NYSE)
            >>> forex_calendar = TradingCalendar(time_service, Exchange.FOREX)
        """
        self._time_service = time_service
        self.exchange = exchange
        self.session = self._create_exchange_session(exchange)

    def _create_exchange_session(self, exchange: Exchange) -> TradingSession:
        """Create a trading session with proper time service injection."""
        ny_tz = self._time_service.get_timezone("America/New_York")
        utc_tz = self._time_service.get_timezone("UTC")

        if exchange == Exchange.NYSE or exchange == Exchange.NASDAQ:
            return TradingSession(
                regular_hours=MarketHours(
                    open_time=time(9, 30),
                    close_time=time(16, 0),
                    timezone=ny_tz,
                    time_service=self._time_service,
                ),
                pre_market=MarketHours(
                    open_time=time(4, 0),
                    close_time=time(9, 30),
                    timezone=ny_tz,
                    time_service=self._time_service,
                ),
                after_hours=MarketHours(
                    open_time=time(16, 0),
                    close_time=time(20, 0),
                    timezone=ny_tz,
                    time_service=self._time_service,
                ),
            )
        elif exchange == Exchange.FOREX:
            return TradingSession(
                regular_hours=MarketHours(
                    open_time=time(17, 0),
                    close_time=time(17, 0),
                    timezone=ny_tz,
                    time_service=self._time_service,
                )
            )
        elif exchange == Exchange.CRYPTO:
            return TradingSession(
                regular_hours=MarketHours(
                    open_time=time(0, 0),
                    close_time=time(23, 59, 59),
                    timezone=utc_tz,
                    time_service=self._time_service,
                )
            )
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")

    def is_trading_day(self, check_date: date) -> bool:
        """Check if given date is a trading day.

        Determines if markets are open on a specific date, considering weekends
        and holidays for the exchange.

        Args:
            check_date: Date to check.

        Returns:
            bool: True if markets trade on this date, False otherwise.

        Exchange Rules:
            - CRYPTO: Every day (24/7)
            - FOREX: Sunday evening through Friday evening (closed Saturday)
            - NYSE/NASDAQ: Weekdays except US market holidays

        Example:
            >>> calendar = TradingCalendar(Exchange.NYSE)
            >>> from datetime import date
            >>> if calendar.is_trading_day(date.today()):
            ...     print("Markets are open today")
        """
        # Crypto trades every day
        if self.exchange == Exchange.CRYPTO:
            return True

        # Forex trades Sunday-Friday
        if self.exchange == Exchange.FOREX:
            return check_date.weekday() != 5  # Not Saturday

        # US exchanges - weekdays except holidays
        if check_date.weekday() >= 5:  # Weekend
            return False

        # Check holidays
        if check_date in self.US_MARKET_HOLIDAYS:
            return False

        return True

    def is_market_open(
        self, check_time: datetime | None = None, extended_hours: bool = False
    ) -> bool:
        """Check if market is currently open.

        Determines if the market is open for trading at the specified time,
        considering trading days, market hours, and extended sessions.

        Args:
            check_time: Time to check (default: current time in UTC).
            extended_hours: If True, includes pre-market and after-hours.

        Returns:
            bool: True if market is open for trading.

        Example:
            >>> calendar = TradingCalendar(Exchange.NYSE)
            >>> if calendar.is_market_open(extended_hours=True):
            ...     print("Market is open (including extended hours)")
            >>> elif calendar.is_market_open():
            ...     print("Regular trading hours are active")

        Note:
            Forex markets have special handling for their Sunday-Friday schedule.
        """
        if check_time is None:
            check_time = self._time_service.get_utc_now().as_datetime()

        # Check if it's a trading day
        if not self.is_trading_day(check_time.date()):
            return False

        # Special handling for Forex (Sunday 5PM - Friday 5PM)
        if self.exchange == Exchange.FOREX:
            return self._is_forex_open(check_time)

        # Check session hours
        return self.session.is_market_open(check_time, extended_hours)

    def _is_forex_open(self, check_time: datetime) -> bool:
        """Special handling for Forex market hours.

        Forex markets operate Sunday 5PM ET through Friday 5PM ET, requiring
        special logic to handle the weekly schedule.

        Args:
            check_time: Time to check for Forex market status.

        Returns:
            bool: True if Forex markets are open.

        Market Schedule:
            - Sunday: Opens at 5PM ET
            - Monday-Thursday: Open 24 hours
            - Friday: Closes at 5PM ET
            - Saturday: Closed all day
        """
        ny_tz = self._time_service.get_timezone("America/New_York")

        if self._time_service.is_timezone_aware(check_time):
            # Use time service to create adapter instead of importing from infrastructure
            adapter = self._time_service.create_adapter(check_time)
            ny_time = self._time_service.convert_timezone(adapter, ny_tz)
        else:
            ny_time = self._time_service.localize_naive_datetime(check_time, ny_tz)

        weekday = ny_time.weekday()
        current_time = ny_time.time()

        # Closed on Saturday
        if weekday == 5:
            return False

        # Sunday - opens at 5PM
        if weekday == 6:
            return current_time >= time(17, 0)

        # Friday - closes at 5PM
        if weekday == 4:
            return current_time < time(17, 0)

        # Monday-Thursday - open all day
        return True

    def next_market_open(self, from_time: datetime | None = None) -> datetime:
        """Get next market open time.

        Calculates when the market will next be open for trading. If the market
        is currently open, returns the current time.

        Args:
            from_time: Starting time for calculation (default: current time).

        Returns:
            datetime: Next market open time in UTC. For 24/7 markets (crypto),
                returns the current/provided time.

        Example:
            >>> calendar = TradingCalendar(Exchange.NYSE)
            >>> next_open = calendar.next_market_open()
            >>> print(f"Market opens: {next_open.isoformat()}")

        Note:
            Returns UTC time for consistency. Convert to local timezone as needed.
        """
        if from_time is None:
            from_time = self._time_service.get_utc_now().as_datetime()

        # If currently open, return current time
        if self.is_market_open(from_time):
            return from_time

        # Crypto is always open
        if self.exchange == Exchange.CRYPTO:
            return from_time

        # Find next trading day
        next_day = from_time.date() + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)

        # Get market open time for that day
        market_tz = self.session.regular_hours.timezone
        open_time = self._time_service.combine_date_time_timezone(
            next_day, self.session.regular_hours.open_time, market_tz
        )

        return self._time_service.to_utc(open_time).as_datetime()

    def next_market_close(self, from_time: datetime | None = None) -> datetime:
        """Get next market close time.

        Calculates when the market will next close. If market is currently open,
        returns today's close time. If closed, returns the close time after the
        next open.

        Args:
            from_time: Starting time for calculation (default: current time).

        Returns:
            datetime: Next market close time in UTC. For 24/7 markets (crypto),
                returns a far future date.

        Special Cases:
            - Crypto: Returns far future date (market never closes)
            - Forex: Special handling for Friday 5PM ET close

        Example:
            >>> calendar = TradingCalendar(Exchange.NYSE)
            >>> next_close = calendar.next_market_close()
            >>> print(f"Market closes: {next_close.isoformat()}")
        """
        if from_time is None:
            from_time = self._time_service.get_utc_now().as_datetime()

        # Crypto never closes
        if self.exchange == Exchange.CRYPTO:
            return from_time + timedelta(days=365)  # Far future

        # If market is open, return today's close
        if self.is_market_open(from_time):
            market_tz = self.session.regular_hours.timezone
            close_time = self._time_service.combine_date_time_timezone(
                from_time.date(), self.session.regular_hours.close_time, market_tz
            )

            # Handle Forex Friday close
            if self.exchange == Exchange.FOREX:
                ny_tz = self._time_service.get_timezone("America/New_York")

                if self._time_service.is_timezone_aware(from_time):
                    # Use time service to create adapter instead of importing from infrastructure
                    adapter = self._time_service.create_adapter(from_time)
                else:
                    adapter = self._time_service.localize_naive_datetime(from_time, ny_tz)

                ny_time = self._time_service.convert_timezone(adapter, ny_tz)
                if ny_time.weekday() == 4:  # Friday
                    close_time = self._time_service.combine_date_time_timezone(
                        ny_time.date(), time(17, 0), ny_tz
                    )

            return self._time_service.to_utc(close_time).as_datetime()

        # Market is closed, find next open then close
        next_open = self.next_market_open(from_time)
        return self.next_market_close(next_open)

    def get_trading_days(
        self, start_date: date, end_date: date, include_partial: bool = False
    ) -> list[date]:
        """Get list of trading days in date range.

        Returns all dates when the market is open within the specified range.

        Args:
            start_date: Beginning of date range (inclusive).
            end_date: End of date range (inclusive).
            include_partial: Currently unused, reserved for handling partial
                trading days (e.g., early closes).

        Returns:
            list[date]: List of trading days in chronological order.

        Example:
            >>> calendar = TradingCalendar(Exchange.NYSE)
            >>> from datetime import date
            >>> trading_days = calendar.get_trading_days(
            ...     date(2024, 1, 1),
            ...     date(2024, 1, 31)
            ... )
            >>> print(f"January 2024 has {len(trading_days)} trading days")
        """
        trading_days = []
        current = start_date

        while current <= end_date:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days

    def get_previous_trading_day(self, from_date: date | None = None) -> date:
        """Get previous trading day.

        Finds the most recent trading day before the specified date.

        Args:
            from_date: Starting date (default: today).

        Returns:
            date: Previous trading day.

        Example:
            >>> calendar = TradingCalendar(Exchange.NYSE)
            >>> prev_day = calendar.get_previous_trading_day()
            >>> print(f"Previous trading day: {prev_day}")
        """
        if from_date is None:
            from_date = date.today()

        previous = from_date - timedelta(days=1)
        while not self.is_trading_day(previous):
            previous -= timedelta(days=1)

        return previous

    def get_next_trading_day(self, from_date: date | None = None) -> date:
        """Get next trading day.

        Finds the next trading day after the specified date.

        Args:
            from_date: Starting date (default: today).

        Returns:
            date: Next trading day.

        Example:
            >>> calendar = TradingCalendar(Exchange.NYSE)
            >>> next_day = calendar.get_next_trading_day()
            >>> print(f"Next trading day: {next_day}")
        """
        if from_date is None:
            from_date = date.today()

        next_day = from_date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)

        return next_day

    def trading_days_between(self, start_date: date, end_date: date) -> int:
        """Count trading days between two dates.

        Calculates the number of trading days in a date range, inclusive of
        both start and end dates.

        Args:
            start_date: Beginning of range.
            end_date: End of range.

        Returns:
            int: Number of trading days in the range.

        Example:
            >>> calendar = TradingCalendar(Exchange.NYSE)
            >>> from datetime import date
            >>> days = calendar.trading_days_between(
            ...     date(2024, 1, 1),
            ...     date(2024, 12, 31)
            ... )
            >>> print(f"2024 has {days} trading days")

        Note:
            The dates can be provided in any order; the method will handle
            the ordering automatically.
        """
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        count = 0
        current = start_date
        while current <= end_date:
            if self.is_trading_day(current):
                count += 1
            current += timedelta(days=1)

        return count
