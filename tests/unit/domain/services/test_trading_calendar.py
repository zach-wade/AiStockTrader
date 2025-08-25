"""
Comprehensive unit tests for Trading Calendar service
"""

# Standard library imports
from datetime import UTC, date, datetime, time, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

# Third-party imports
# Local imports
from src.domain.services.trading_calendar import (
    Exchange,
    MarketHours,
    TradingCalendar,
    TradingSession,
)


class TestMarketHours:
    """Test MarketHours functionality"""

    def test_create_market_hours(self):
        """Test creating market hours"""
        hours = MarketHours(
            open_time=time(9, 30),
            close_time=time(16, 0),
            timezone=ZoneInfo("America/New_York"),
            is_open=True,
        )

        assert hours.open_time == time(9, 30)
        assert hours.close_time == time(16, 0)
        assert hours.timezone == ZoneInfo("America/New_York")
        assert hours.is_open is True

    def test_closed_market_hours(self):
        """Test market hours marked as closed"""
        hours = MarketHours(
            open_time=time(9, 30),
            close_time=time(16, 0),
            timezone=ZoneInfo("America/New_York"),
            is_open=False,
        )

        # Should always return False when market is closed
        test_time = datetime(2024, 1, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert hours.is_time_in_session(test_time) is False

    def test_regular_hours_in_session(self):
        """Test time within regular market hours"""
        hours = MarketHours(
            open_time=time(9, 30), close_time=time(16, 0), timezone=ZoneInfo("America/New_York")
        )

        # Test middle of day
        test_time = datetime(2024, 1, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert hours.is_time_in_session(test_time) is True

        # Test market open
        test_time = datetime(2024, 1, 15, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        assert hours.is_time_in_session(test_time) is True

        # Test market close
        test_time = datetime(2024, 1, 15, 16, 0, tzinfo=ZoneInfo("America/New_York"))
        assert hours.is_time_in_session(test_time) is True

    def test_regular_hours_out_of_session(self):
        """Test time outside regular market hours"""
        hours = MarketHours(
            open_time=time(9, 30), close_time=time(16, 0), timezone=ZoneInfo("America/New_York")
        )

        # Before market open
        test_time = datetime(2024, 1, 15, 9, 0, tzinfo=ZoneInfo("America/New_York"))
        assert hours.is_time_in_session(test_time) is False

        # After market close
        test_time = datetime(2024, 1, 15, 16, 1, tzinfo=ZoneInfo("America/New_York"))
        assert hours.is_time_in_session(test_time) is False

    def test_overnight_session(self):
        """Test overnight trading session (e.g., futures)"""
        hours = MarketHours(
            open_time=time(18, 0),  # 6 PM
            close_time=time(5, 0),  # 5 AM next day
            timezone=ZoneInfo("America/Chicago"),
        )

        # Evening - should be open
        test_time = datetime(2024, 1, 15, 20, 0, tzinfo=ZoneInfo("America/Chicago"))
        assert hours.is_time_in_session(test_time) is True

        # Midnight - should be open
        test_time = datetime(2024, 1, 15, 0, 0, tzinfo=ZoneInfo("America/Chicago"))
        assert hours.is_time_in_session(test_time) is True

        # Early morning - should be open
        test_time = datetime(2024, 1, 15, 4, 0, tzinfo=ZoneInfo("America/Chicago"))
        assert hours.is_time_in_session(test_time) is True

        # After close - should be closed
        test_time = datetime(2024, 1, 15, 6, 0, tzinfo=ZoneInfo("America/Chicago"))
        assert hours.is_time_in_session(test_time) is False

    def test_timezone_conversion(self):
        """Test timezone conversion in session check"""
        hours = MarketHours(
            open_time=time(9, 30), close_time=time(16, 0), timezone=ZoneInfo("America/New_York")
        )

        # Test with UTC time (12:00 UTC = 7:00 AM EST in winter)
        test_time = datetime(2024, 1, 15, 17, 0, tzinfo=UTC)  # 12:00 PM EST
        assert hours.is_time_in_session(test_time) is True

        # Test with Pacific time
        test_time = datetime(
            2024, 1, 15, 9, 0, tzinfo=ZoneInfo("America/Los_Angeles")
        )  # 12:00 PM EST
        assert hours.is_time_in_session(test_time) is True


class TestTradingSession:
    """Test TradingSession functionality"""

    def test_regular_hours_only(self):
        """Test session with only regular hours"""
        session = TradingSession(
            regular_hours=MarketHours(
                open_time=time(9, 30), close_time=time(16, 0), timezone=ZoneInfo("America/New_York")
            )
        )

        # During regular hours
        test_time = datetime(2024, 1, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert session.is_market_open(test_time) is True

        # Outside regular hours
        test_time = datetime(2024, 1, 15, 8, 0, tzinfo=ZoneInfo("America/New_York"))
        assert session.is_market_open(test_time) is False

    def test_extended_hours(self):
        """Test session with pre-market and after-hours"""
        session = TradingSession(
            regular_hours=MarketHours(
                open_time=time(9, 30), close_time=time(16, 0), timezone=ZoneInfo("America/New_York")
            ),
            pre_market=MarketHours(
                open_time=time(4, 0), close_time=time(9, 30), timezone=ZoneInfo("America/New_York")
            ),
            after_hours=MarketHours(
                open_time=time(16, 0), close_time=time(20, 0), timezone=ZoneInfo("America/New_York")
            ),
        )

        # Pre-market without extended hours flag
        test_time = datetime(2024, 1, 15, 5, 0, tzinfo=ZoneInfo("America/New_York"))
        assert session.is_market_open(test_time, extended_hours=False) is False
        assert session.is_market_open(test_time, extended_hours=True) is True

        # After-hours without extended hours flag
        test_time = datetime(2024, 1, 15, 17, 0, tzinfo=ZoneInfo("America/New_York"))
        assert session.is_market_open(test_time, extended_hours=False) is False
        assert session.is_market_open(test_time, extended_hours=True) is True

        # Regular hours (both should be True)
        test_time = datetime(2024, 1, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert session.is_market_open(test_time, extended_hours=False) is True
        assert session.is_market_open(test_time, extended_hours=True) is True


class TestTradingCalendar:
    """Test TradingCalendar main functionality"""

    def test_exchange_initialization(self):
        """Test calendar initialization for different exchanges"""
        nyse_cal = TradingCalendar(Exchange.NYSE)
        assert nyse_cal.exchange == Exchange.NYSE

        nasdaq_cal = TradingCalendar(Exchange.NASDAQ)
        assert nasdaq_cal.exchange == Exchange.NASDAQ

        forex_cal = TradingCalendar(Exchange.FOREX)
        assert forex_cal.exchange == Exchange.FOREX

        crypto_cal = TradingCalendar(Exchange.CRYPTO)
        assert crypto_cal.exchange == Exchange.CRYPTO

    def test_is_trading_day_weekday(self):
        """Test weekday trading day detection"""
        cal = TradingCalendar(Exchange.NYSE)

        # Monday
        assert cal.is_trading_day(date(2024, 1, 8)) is True
        # Tuesday
        assert cal.is_trading_day(date(2024, 1, 9)) is True
        # Wednesday
        assert cal.is_trading_day(date(2024, 1, 10)) is True
        # Thursday
        assert cal.is_trading_day(date(2024, 1, 11)) is True
        # Friday
        assert cal.is_trading_day(date(2024, 1, 12)) is True

    def test_is_trading_day_weekend(self):
        """Test weekend non-trading days"""
        cal = TradingCalendar(Exchange.NYSE)

        # Saturday
        assert cal.is_trading_day(date(2024, 1, 6)) is False
        # Sunday
        assert cal.is_trading_day(date(2024, 1, 7)) is False

    def test_is_trading_day_holidays(self):
        """Test US market holidays"""
        cal = TradingCalendar(Exchange.NYSE)

        # New Year's Day 2024
        assert cal.is_trading_day(date(2024, 1, 1)) is False
        # MLK Day 2024
        assert cal.is_trading_day(date(2024, 1, 15)) is False
        # Good Friday 2024
        assert cal.is_trading_day(date(2024, 3, 29)) is False
        # Independence Day 2024
        assert cal.is_trading_day(date(2024, 7, 4)) is False
        # Christmas 2024
        assert cal.is_trading_day(date(2024, 12, 25)) is False

    def test_crypto_always_trading(self):
        """Test crypto trades every day"""
        cal = TradingCalendar(Exchange.CRYPTO)

        # Weekday
        assert cal.is_trading_day(date(2024, 1, 8)) is True
        # Weekend
        assert cal.is_trading_day(date(2024, 1, 6)) is True
        assert cal.is_trading_day(date(2024, 1, 7)) is True
        # Holiday
        assert cal.is_trading_day(date(2024, 12, 25)) is True

    def test_forex_trading_days(self):
        """Test forex trading days (Sunday-Friday)"""
        cal = TradingCalendar(Exchange.FOREX)

        # Sunday - trades
        assert cal.is_trading_day(date(2024, 1, 7)) is True
        # Monday-Friday - trades
        assert cal.is_trading_day(date(2024, 1, 8)) is True
        assert cal.is_trading_day(date(2024, 1, 12)) is True
        # Saturday - no trading
        assert cal.is_trading_day(date(2024, 1, 6)) is False

    def test_is_market_open_during_hours(self):
        """Test market open during regular hours"""
        cal = TradingCalendar(Exchange.NYSE)

        # Monday at noon EST
        test_time = datetime(2024, 1, 8, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is True

    def test_is_market_open_outside_hours(self):
        """Test market closed outside hours"""
        cal = TradingCalendar(Exchange.NYSE)

        # Monday at 5 PM EST (after close)
        test_time = datetime(2024, 1, 8, 17, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is False

        # Monday at 8 AM EST (before open)
        test_time = datetime(2024, 1, 8, 8, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is False

    def test_is_market_open_on_holiday(self):
        """Test market closed on holidays"""
        cal = TradingCalendar(Exchange.NYSE)

        # Christmas Day at noon
        test_time = datetime(2024, 12, 25, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is False

    def test_is_market_open_extended_hours(self):
        """Test extended hours trading"""
        cal = TradingCalendar(Exchange.NYSE)

        # Pre-market at 5 AM EST
        test_time = datetime(2024, 1, 8, 5, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time, extended_hours=False) is False
        assert cal.is_market_open(test_time, extended_hours=True) is True

        # After-hours at 6 PM EST
        test_time = datetime(2024, 1, 8, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time, extended_hours=False) is False
        assert cal.is_market_open(test_time, extended_hours=True) is True

    def test_forex_market_hours(self):
        """Test forex special market hours"""
        cal = TradingCalendar(Exchange.FOREX)

        # Sunday 6 PM EST - should be open
        test_time = datetime(2024, 1, 7, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is True

        # Sunday 4 PM EST - should be closed
        test_time = datetime(2024, 1, 7, 16, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is False

        # Wednesday noon - should be open
        test_time = datetime(2024, 1, 10, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is True

        # Friday 4 PM EST - should be open
        test_time = datetime(2024, 1, 12, 16, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is True

        # Friday 6 PM EST - should be closed
        test_time = datetime(2024, 1, 12, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is False

        # Saturday - should be closed
        test_time = datetime(2024, 1, 13, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is False

    def test_crypto_always_open(self):
        """Test crypto market always open"""
        cal = TradingCalendar(Exchange.CRYPTO)

        # Any time should be open
        test_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        assert cal.is_market_open(test_time) is True

        test_time = datetime(2024, 12, 25, 12, 0, tzinfo=UTC)
        assert cal.is_market_open(test_time) is True

    def test_next_market_open_currently_open(self):
        """Test next market open when currently open"""
        cal = TradingCalendar(Exchange.NYSE)

        # Monday noon - market is open
        current_time = datetime(2024, 1, 8, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        next_open = cal.next_market_open(current_time)

        # Should return current time since market is open
        assert next_open == current_time

    def test_next_market_open_after_close(self):
        """Test next market open after market close"""
        cal = TradingCalendar(Exchange.NYSE)

        # Monday 5 PM - after close
        current_time = datetime(2024, 1, 8, 17, 0, tzinfo=ZoneInfo("America/New_York"))
        next_open = cal.next_market_open(current_time)

        # Should be Tuesday 9:30 AM
        expected = datetime(2024, 1, 9, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        assert next_open.astimezone(ZoneInfo("America/New_York")) == expected

    def test_next_market_open_weekend(self):
        """Test next market open from weekend"""
        cal = TradingCalendar(Exchange.NYSE)

        # Saturday
        current_time = datetime(2024, 1, 6, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        next_open = cal.next_market_open(current_time)

        # Should be Monday 9:30 AM
        expected = datetime(2024, 1, 8, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        assert next_open.astimezone(ZoneInfo("America/New_York")) == expected

    def test_next_market_open_before_holiday(self):
        """Test next market open before holiday"""
        cal = TradingCalendar(Exchange.NYSE)

        # December 24, 2024 (day before Christmas)
        current_time = datetime(2024, 12, 24, 17, 0, tzinfo=ZoneInfo("America/New_York"))
        next_open = cal.next_market_open(current_time)

        # Should skip Christmas and be December 26
        expected = datetime(2024, 12, 26, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        assert next_open.astimezone(ZoneInfo("America/New_York")) == expected

    def test_next_market_close_during_market(self):
        """Test next market close during market hours"""
        cal = TradingCalendar(Exchange.NYSE)

        # Monday noon
        current_time = datetime(2024, 1, 8, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        next_close = cal.next_market_close(current_time)

        # Should be same day 4 PM
        expected = datetime(2024, 1, 8, 16, 0, tzinfo=ZoneInfo("America/New_York"))
        assert next_close.astimezone(ZoneInfo("America/New_York")) == expected

    def test_next_market_close_after_hours(self):
        """Test next market close after market hours"""
        cal = TradingCalendar(Exchange.NYSE)

        # Monday 5 PM (after close)
        current_time = datetime(2024, 1, 8, 17, 0, tzinfo=ZoneInfo("America/New_York"))
        next_close = cal.next_market_close(current_time)

        # Should be Tuesday 4 PM
        expected = datetime(2024, 1, 9, 16, 0, tzinfo=ZoneInfo("America/New_York"))
        assert next_close.astimezone(ZoneInfo("America/New_York")) == expected

    def test_crypto_next_market_times(self):
        """Test crypto market open/close times"""
        cal = TradingCalendar(Exchange.CRYPTO)

        current_time = datetime(2024, 1, 8, 12, 0, tzinfo=UTC)

        # Crypto is always open
        next_open = cal.next_market_open(current_time)
        assert next_open == current_time

        # Close is far in the future
        next_close = cal.next_market_close(current_time)
        assert next_close > current_time + timedelta(days=300)

    def test_get_trading_days_range(self):
        """Test getting trading days in date range"""
        cal = TradingCalendar(Exchange.NYSE)

        start = date(2024, 1, 8)  # Monday
        end = date(2024, 1, 12)  # Friday

        trading_days = cal.get_trading_days(start, end)

        # Should be Mon-Fri
        assert len(trading_days) == 5
        assert trading_days[0] == date(2024, 1, 8)
        assert trading_days[-1] == date(2024, 1, 12)

    def test_get_trading_days_with_weekend(self):
        """Test trading days range including weekend"""
        cal = TradingCalendar(Exchange.NYSE)

        start = date(2024, 1, 5)  # Friday
        end = date(2024, 1, 8)  # Monday

        trading_days = cal.get_trading_days(start, end)

        # Should be Friday and Monday only
        assert len(trading_days) == 2
        assert trading_days[0] == date(2024, 1, 5)
        assert trading_days[1] == date(2024, 1, 8)

    def test_get_trading_days_with_holiday(self):
        """Test trading days range including holiday"""
        cal = TradingCalendar(Exchange.NYSE)

        start = date(2024, 1, 12)  # Friday before MLK Day
        end = date(2024, 1, 16)  # Tuesday after MLK Day

        trading_days = cal.get_trading_days(start, end)

        # Should skip MLK Day (Jan 15)
        assert len(trading_days) == 2
        assert date(2024, 1, 15) not in trading_days

    def test_get_previous_trading_day(self):
        """Test getting previous trading day"""
        cal = TradingCalendar(Exchange.NYSE)

        # From Tuesday
        assert cal.get_previous_trading_day(date(2024, 1, 9)) == date(2024, 1, 8)

        # From Monday (should be previous Friday)
        assert cal.get_previous_trading_day(date(2024, 1, 8)) == date(2024, 1, 5)

        # From day after holiday
        assert cal.get_previous_trading_day(date(2024, 1, 16)) == date(2024, 1, 12)

    def test_get_next_trading_day(self):
        """Test getting next trading day"""
        cal = TradingCalendar(Exchange.NYSE)

        # From Monday
        assert cal.get_next_trading_day(date(2024, 1, 8)) == date(2024, 1, 9)

        # From Friday (should be next Monday)
        assert cal.get_next_trading_day(date(2024, 1, 12)) == date(2024, 1, 16)

        # From day before holiday
        assert cal.get_next_trading_day(date(2024, 1, 12)) == date(2024, 1, 16)

    def test_trading_days_between(self):
        """Test counting trading days between dates"""
        cal = TradingCalendar(Exchange.NYSE)

        # Same day
        assert cal.trading_days_between(date(2024, 1, 8), date(2024, 1, 8)) == 1

        # One week (Mon-Fri)
        assert cal.trading_days_between(date(2024, 1, 8), date(2024, 1, 12)) == 5

        # Two weeks with weekend
        assert cal.trading_days_between(date(2024, 1, 8), date(2024, 1, 19)) == 10

        # Reverse order (should handle gracefully)
        assert cal.trading_days_between(date(2024, 1, 12), date(2024, 1, 8)) == 5

    def test_trading_days_between_with_holidays(self):
        """Test counting trading days with holidays"""
        cal = TradingCalendar(Exchange.NYSE)

        # Range including MLK Day
        start = date(2024, 1, 12)
        end = date(2024, 1, 19)

        count = cal.trading_days_between(start, end)
        # Friday, Tuesday, Wednesday, Thursday, Friday = 5 days (skip weekend + MLK)
        assert count == 5

    def test_default_from_time(self):
        """Test methods with default from_time parameter"""
        cal = TradingCalendar(Exchange.NYSE)

        with patch("src.domain.services.trading_calendar.datetime") as mock_datetime:
            mock_now = datetime(2024, 1, 8, 12, 0, tzinfo=UTC)
            mock_datetime.now.return_value = mock_now
            mock_datetime.combine = datetime.combine

            # These should use current time
            cal.is_market_open()  # Should not raise
            cal.next_market_open()  # Should not raise
            cal.next_market_close()  # Should not raise

    def test_default_from_date(self):
        """Test methods with default from_date parameter"""
        cal = TradingCalendar(Exchange.NYSE)

        with patch("src.domain.services.trading_calendar.date") as mock_date:
            mock_today = date(2024, 1, 8)
            mock_date.today.return_value = mock_today

            # These should use today's date
            cal.get_previous_trading_day()  # Should not raise
            cal.get_next_trading_day()  # Should not raise


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_market_hours_at_exact_open(self):
        """Test market status at exact open time"""
        hours = MarketHours(
            open_time=time(9, 30), close_time=time(16, 0), timezone=ZoneInfo("America/New_York")
        )

        test_time = datetime(2024, 1, 8, 9, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        assert hours.is_time_in_session(test_time) is True

    def test_market_hours_at_exact_close(self):
        """Test market status at exact close time"""
        hours = MarketHours(
            open_time=time(9, 30), close_time=time(16, 0), timezone=ZoneInfo("America/New_York")
        )

        test_time = datetime(2024, 1, 8, 16, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        assert hours.is_time_in_session(test_time) is True

        # One second after should be closed
        test_time = datetime(2024, 1, 8, 16, 0, 1, tzinfo=ZoneInfo("America/New_York"))
        assert hours.is_time_in_session(test_time) is False

    def test_midnight_crossing(self):
        """Test sessions crossing midnight"""
        hours = MarketHours(open_time=time(23, 0), close_time=time(1, 0), timezone=ZoneInfo("UTC"))

        # Before midnight - open
        test_time = datetime(2024, 1, 8, 23, 30, tzinfo=UTC)
        assert hours.is_time_in_session(test_time) is True

        # After midnight - still open
        test_time = datetime(2024, 1, 9, 0, 30, tzinfo=UTC)
        assert hours.is_time_in_session(test_time) is True

        # After close - closed
        test_time = datetime(2024, 1, 9, 1, 30, tzinfo=UTC)
        assert hours.is_time_in_session(test_time) is False

    def test_24_hour_session(self):
        """Test 24-hour trading session"""
        hours = MarketHours(
            open_time=time(0, 0), close_time=time(23, 59, 59), timezone=ZoneInfo("UTC")
        )

        # Should be open all day
        for hour in range(24):
            test_time = datetime(2024, 1, 8, hour, 0, tzinfo=UTC)
            assert hours.is_time_in_session(test_time) is True

    def test_year_boundary(self):
        """Test trading days across year boundary"""
        cal = TradingCalendar(Exchange.NYSE)

        # Dec 29, 2023 to Jan 2, 2024
        start = date(2023, 12, 29)
        end = date(2024, 1, 2)

        trading_days = cal.get_trading_days(start, end)

        # Dec 29 (Fri), Jan 2 (Tue) - New Year's Day is holiday
        assert date(2023, 12, 29) in trading_days
        assert date(2024, 1, 1) not in trading_days
        assert date(2024, 1, 2) in trading_days

    def test_leap_year(self):
        """Test leap year handling"""
        cal = TradingCalendar(Exchange.NYSE)

        # Feb 29, 2024 (leap year)
        leap_day = date(2024, 2, 29)

        # Thursday - should be trading day
        assert cal.is_trading_day(leap_day) is True

    def test_daylight_savings_transition(self):
        """Test daylight savings time transitions"""
        cal = TradingCalendar(Exchange.NYSE)

        # Spring forward - March 10, 2024
        # Market should still open at 9:30 AM local time
        spring_forward = datetime(2024, 3, 11, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(spring_forward) is True

        # Fall back - November 3, 2024
        # Market should still open at 9:30 AM local time
        fall_back = datetime(2024, 11, 4, 9, 30, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(fall_back) is True

    def test_empty_date_range(self):
        """Test empty date range"""
        cal = TradingCalendar(Exchange.NYSE)

        # End before start
        start = date(2024, 1, 10)
        end = date(2024, 1, 8)

        # Should handle gracefully
        trading_days = cal.get_trading_days(end, start)
        assert len(trading_days) == 3  # Should swap and return valid range

    def test_single_day_range(self):
        """Test single day range"""
        cal = TradingCalendar(Exchange.NYSE)

        # Same start and end
        single_day = date(2024, 1, 8)
        trading_days = cal.get_trading_days(single_day, single_day)

        assert len(trading_days) == 1
        assert trading_days[0] == single_day

    def test_very_long_date_range(self):
        """Test very long date range"""
        cal = TradingCalendar(Exchange.NYSE)

        # Full year
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)

        trading_days = cal.get_trading_days(start, end)

        # Approximately 252 trading days in a year
        assert 250 <= len(trading_days) <= 254

    def test_forex_sunday_edge_case(self):
        """Test forex Sunday opening edge case"""
        cal = TradingCalendar(Exchange.FOREX)

        # Sunday 4:59 PM EST - should be closed
        test_time = datetime(2024, 1, 7, 16, 59, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is False

        # Sunday 5:00 PM EST - should be open
        test_time = datetime(2024, 1, 7, 17, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is True

    def test_forex_friday_edge_case(self):
        """Test forex Friday closing edge case"""
        cal = TradingCalendar(Exchange.FOREX)

        # Friday 4:59 PM EST - should be open
        test_time = datetime(2024, 1, 12, 16, 59, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is True

        # Friday 5:00 PM EST - should be closed
        test_time = datetime(2024, 1, 12, 17, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal.is_market_open(test_time) is False
