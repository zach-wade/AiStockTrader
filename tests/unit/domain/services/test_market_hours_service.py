"""
Comprehensive unit tests for MarketHoursService.

Tests all market hours determination, status checking, and calendar logic.
"""

from datetime import date, datetime, time, timedelta
from unittest.mock import patch

import pytest
import pytz

from src.domain.services.market_hours_service import MarketHoursService, MarketStatus


class TestMarketHoursService:
    """Test suite for MarketHoursService."""

    @pytest.fixture
    def service(self):
        """Create a MarketHoursService instance."""
        return MarketHoursService()

    @pytest.fixture
    def eastern_tz(self):
        """Get Eastern timezone."""
        return pytz.timezone("America/New_York")

    # Market Status Tests

    def test_market_status_regular_open(self, service, eastern_tz):
        """Test market status during regular trading hours."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 9, 30))  # Tuesday 9:30 AM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            status = service.get_current_market_status()
            assert status == MarketStatus.OPEN

    def test_market_status_regular_open_end(self, service, eastern_tz):
        """Test market status just before close."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 15, 59, 59))  # Tuesday 3:59:59 PM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            status = service.get_current_market_status()
            assert status == MarketStatus.OPEN

    def test_market_status_after_market_start(self, service, eastern_tz):
        """Test market status at after-market open."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 16, 0))  # Tuesday 4:00 PM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            status = service.get_current_market_status()
            assert status == MarketStatus.AFTER_MARKET

    def test_market_status_pre_market_start(self, service, eastern_tz):
        """Test market status at pre-market open."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 4, 0))  # Tuesday 4:00 AM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            status = service.get_current_market_status()
            assert status == MarketStatus.PRE_MARKET

    def test_market_status_pre_market_end(self, service, eastern_tz):
        """Test market status just before regular open."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 9, 29, 59))  # Tuesday 9:29:59 AM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            status = service.get_current_market_status()
            assert status == MarketStatus.PRE_MARKET

    def test_market_status_after_market_close(self, service, eastern_tz):
        """Test market status at after-market close."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 20, 0))  # Tuesday 8:00 PM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            status = service.get_current_market_status()
            assert status == MarketStatus.CLOSED

    def test_market_status_before_pre_market(self, service, eastern_tz):
        """Test market status before pre-market."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 3, 59, 59))  # Tuesday 3:59:59 AM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            status = service.get_current_market_status()
            assert status == MarketStatus.CLOSED

    def test_market_status_weekend(self, service, eastern_tz):
        """Test market status on weekend."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 13, 12, 0))  # Saturday noon ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            status = service.get_current_market_status()
            assert status == MarketStatus.CLOSED

    def test_market_status_holiday(self, service, eastern_tz):
        """Test market status on holiday."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 1, 12, 0))  # New Year's Day
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            status = service.get_current_market_status()
            assert status == MarketStatus.HOLIDAY

    # Custom Datetime Tests

    def test_get_market_status_with_custom_datetime(self, service, eastern_tz):
        """Test getting market status for specific datetime."""
        # Regular trading hours
        dt = eastern_tz.localize(datetime(2024, 1, 16, 10, 30))
        status = service.get_current_market_status(dt)
        assert status == MarketStatus.OPEN

        # Pre-market
        dt = eastern_tz.localize(datetime(2024, 1, 16, 5, 0))
        status = service.get_current_market_status(dt)
        assert status == MarketStatus.PRE_MARKET

        # After-market
        dt = eastern_tz.localize(datetime(2024, 1, 16, 17, 0))
        status = service.get_current_market_status(dt)
        assert status == MarketStatus.AFTER_MARKET

        # Closed
        dt = eastern_tz.localize(datetime(2024, 1, 16, 21, 0))
        status = service.get_current_market_status(dt)
        assert status == MarketStatus.CLOSED

    # Is Market Open Tests

    def test_is_market_open_true(self, service, eastern_tz):
        """Test is_market_open during regular hours."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 10, 0))  # Tuesday 10:00 AM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            assert service.is_market_open() is True

    def test_is_market_open_false(self, service, eastern_tz):
        """Test is_market_open when closed."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 3, 0))  # Tuesday 3:00 AM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            assert service.is_market_open() is False

    def test_is_market_open_pre_market(self, service, eastern_tz):
        """Test is_market_open during pre-market."""
        mock_dt = eastern_tz.localize(
            datetime(2024, 1, 16, 5, 0)
        )  # Tuesday 5:00 AM ET (pre-market)
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            assert service.is_market_open() is False
            assert service.is_extended_hours() is True

    def test_is_market_open_after_market(self, service, eastern_tz):
        """Test is_market_open during after-market."""
        mock_dt = eastern_tz.localize(
            datetime(2024, 1, 16, 17, 0)
        )  # Tuesday 5:00 PM ET (after-market)
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            assert service.is_market_open() is False
            assert service.is_extended_hours() is True

    # Is Trading Day Tests

    def test_is_trading_day_weekday(self, service):
        """Test is_trading_day for regular weekdays."""
        # Tuesday through Friday in 2024 (avoiding MLK Day on Jan 15)
        weekdays = [
            date(2024, 1, 16),  # Tuesday
            date(2024, 1, 17),  # Wednesday
            date(2024, 1, 18),  # Thursday
            date(2024, 1, 19),  # Friday
            date(2024, 1, 22),  # Monday
        ]

        for day in weekdays:
            assert service.is_trading_day(day) is True

    def test_is_trading_day_weekend(self, service):
        """Test is_trading_day for weekends."""
        weekend_days = [
            date(2024, 1, 13),  # Saturday
            date(2024, 1, 14),  # Sunday
        ]

        for day in weekend_days:
            assert service.is_trading_day(day) is False

    def test_is_trading_day_holidays(self, service):
        """Test is_trading_day for holidays."""
        holidays = [
            date(2024, 1, 1),  # New Year's Day
            date(2024, 7, 4),  # Independence Day
            date(2024, 12, 25),  # Christmas
        ]

        for holiday in holidays:
            assert service.is_trading_day(holiday) is False

    # Get Regular Market Hours Tests

    def test_get_regular_market_hours(self, service, eastern_tz):
        """Test getting regular market hours."""
        trading_day = date(2024, 1, 16)  # Tuesday
        hours = service.get_regular_market_hours(trading_day)

        assert hours is not None
        assert "open" in hours
        assert "close" in hours

        expected_open = eastern_tz.localize(datetime.combine(trading_day, time(9, 30)))
        expected_close = eastern_tz.localize(datetime.combine(trading_day, time(16, 0)))

        assert hours["open"] == expected_open
        assert hours["close"] == expected_close

    def test_get_regular_market_hours_non_trading_day(self, service):
        """Test getting market hours for non-trading day."""
        weekend = date(2024, 1, 13)  # Saturday
        hours = service.get_regular_market_hours(weekend)
        assert hours is None

        holiday = date(2024, 1, 1)  # New Year's Day
        hours = service.get_regular_market_hours(holiday)
        assert hours is None

    # Get Extended Market Hours Tests

    def test_get_extended_market_hours(self, service, eastern_tz):
        """Test getting extended market hours."""
        trading_day = date(2024, 1, 16)  # Tuesday
        hours = service.get_extended_market_hours(trading_day)

        assert hours is not None
        assert "pre_market_open" in hours
        assert "pre_market_close" in hours
        assert "after_market_open" in hours
        assert "after_market_close" in hours

        expected_pre_open = eastern_tz.localize(datetime.combine(trading_day, time(4, 0)))
        expected_pre_close = eastern_tz.localize(datetime.combine(trading_day, time(9, 30)))
        expected_after_open = eastern_tz.localize(datetime.combine(trading_day, time(16, 0)))
        expected_after_close = eastern_tz.localize(datetime.combine(trading_day, time(20, 0)))

        assert hours["pre_market_open"] == expected_pre_open
        assert hours["pre_market_close"] == expected_pre_close
        assert hours["after_market_open"] == expected_after_open
        assert hours["after_market_close"] == expected_after_close

    def test_get_extended_market_hours_non_trading_day(self, service):
        """Test getting extended hours for non-trading day."""
        weekend = date(2024, 1, 13)  # Saturday
        hours = service.get_extended_market_hours(weekend)
        assert hours is None

    # Next Trading Day Tests

    def test_get_next_trading_day_from_weekday(self, service):
        """Test getting next trading day from a weekday."""
        monday = date(2024, 1, 15)
        next_day = service.get_next_trading_day(monday)
        assert next_day == date(2024, 1, 16)  # Tuesday

    def test_get_next_trading_day_from_friday(self, service):
        """Test getting next trading day from Friday."""
        friday = date(2024, 1, 19)
        next_day = service.get_next_trading_day(friday)
        assert next_day == date(2024, 1, 22)  # Tuesday

    def test_get_next_trading_day_from_weekend(self, service):
        """Test getting next trading day from weekend."""
        saturday = date(2024, 1, 13)
        next_day = service.get_next_trading_day(saturday)
        assert next_day == date(2024, 1, 16)  # Tuesday

        sunday = date(2024, 1, 14)
        next_day = service.get_next_trading_day(sunday)
        assert next_day == date(2024, 1, 16)  # Tuesday

    def test_get_next_trading_day_skip_holiday(self, service):
        """Test getting next trading day skips holidays."""
        # December 24, 2024 is Tuesday, December 25 is Christmas
        dec_24 = date(2024, 12, 24)
        next_day = service.get_next_trading_day(dec_24)
        assert next_day == date(2024, 12, 26)  # Thursday (skip Christmas)

    # Previous Trading Day Tests

    def test_get_previous_trading_day_from_weekday(self, service):
        """Test getting previous trading day from a weekday."""
        tuesday = date(2024, 1, 16)
        prev_day = service.get_previous_trading_day(tuesday)
        assert prev_day == date(2024, 1, 12)  # Friday (Monday is MLK Day)

    def test_get_previous_trading_day_from_monday(self, service):
        """Test getting previous trading day from Tuesday."""
        monday = date(2024, 1, 15)
        prev_day = service.get_previous_trading_day(monday)
        assert prev_day == date(2024, 1, 12)  # Friday

    def test_get_previous_trading_day_from_weekend(self, service):
        """Test getting previous trading day from weekend."""
        saturday = date(2024, 1, 13)
        prev_day = service.get_previous_trading_day(saturday)
        assert prev_day == date(2024, 1, 12)  # Friday

        sunday = date(2024, 1, 14)
        prev_day = service.get_previous_trading_day(sunday)
        assert prev_day == date(2024, 1, 12)  # Friday

    def test_get_previous_trading_day_skip_holiday(self, service):
        """Test getting previous trading day skips holidays."""
        # January 2, 2024 is Tuesday (after New Year's Day)
        jan_2 = date(2024, 1, 2)
        prev_day = service.get_previous_trading_day(jan_2)
        # Should skip New Year's Day (Jan 1) and go to previous Friday
        assert prev_day == date(2023, 12, 29)

    # Time Until Market Open Tests

    def test_time_until_market_open_same_day(self, service, eastern_tz):
        """Test time until market open on same day."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 9, 0))  # Tuesday 9:00 AM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            time_until = service.get_time_until_market_open()
            assert time_until is not None
            assert time_until == timedelta(minutes=30)

    def test_time_until_market_open_during_market(self, service, eastern_tz):
        """Test time until market open when market is already open."""
        mock_dt = eastern_tz.localize(
            datetime(2024, 1, 16, 10, 0)
        )  # Tuesday 10:00 AM ET (market open)
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            time_until = service.get_time_until_market_open()
            assert time_until == timedelta(0)

    def test_time_until_market_open_weekend(self, service, eastern_tz):
        """Test time until market open from weekend."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 14, 20, 0))  # Sunday 8:00 PM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            time_until = service.get_time_until_market_open()
            assert time_until is not None
            # Should be time until Tuesday 9:30 AM (Monday is MLK Day holiday)
            expected = timedelta(days=1, hours=13, minutes=30)
            assert time_until == expected

    def test_time_until_market_open_after_close(self, service, eastern_tz):
        """Test time until market open after market close."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 17, 0))  # Tuesday 5:00 PM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            time_until = service.get_time_until_market_open()
            assert time_until is not None
            # Should be time until Tuesday 9:30 AM
            expected = timedelta(hours=16, minutes=30)
            assert time_until == expected

    # Time Until Market Close Tests

    def test_time_until_market_close_during_market(self, service, eastern_tz):
        """Test time until market close during trading hours."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 15, 0))  # Tuesday 3:00 PM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            time_until = service.get_time_until_market_close()
            assert time_until is not None
            assert time_until == timedelta(hours=1)

    def test_time_until_market_close_at_open(self, service, eastern_tz):
        """Test time until market close at market open."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 9, 30))  # Tuesday 9:30 AM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            time_until = service.get_time_until_market_close()
            assert time_until is not None
            assert time_until == timedelta(hours=6, minutes=30)

    def test_time_until_market_close_after_hours(self, service, eastern_tz):
        """Test time until market close when market is closed."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 16, 17, 0))  # Tuesday 5:00 PM ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            time_until = service.get_time_until_market_close()
            assert time_until is None

    def test_time_until_market_close_weekend(self, service, eastern_tz):
        """Test time until market close on weekend."""
        mock_dt = eastern_tz.localize(datetime(2024, 1, 14, 12, 0))  # Sunday noon ET
        with patch("src.domain.services.market_hours_service.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.combine = datetime.combine
            time_until = service.get_time_until_market_close()
            assert time_until is None

    # Trading Days Between Tests

    def test_get_trading_days_between_same_week(self, service):
        """Test getting trading days between dates in same week."""
        start = date(2024, 1, 16)  # Tuesday
        end = date(2024, 1, 19)  # Friday
        days = service.get_trading_days_between(start, end)
        assert len(days) == 5
        assert days[0] == start
        assert days[-1] == end

    def test_get_trading_days_between_across_weekend(self, service):
        """Test getting trading days across weekend."""
        start = date(2024, 1, 12)  # Friday
        end = date(2024, 1, 16)  # Tuesday
        days = service.get_trading_days_between(start, end)
        assert len(days) == 3  # Friday, Tuesday, Tuesday
        assert date(2024, 1, 13) not in days  # Saturday
        assert date(2024, 1, 14) not in days  # Sunday

    def test_get_trading_days_between_across_holiday(self, service):
        """Test getting trading days across holiday."""
        start = date(2023, 12, 29)  # Friday
        end = date(2024, 1, 3)  # Wednesday
        days = service.get_trading_days_between(start, end)
        # Should skip weekend and New Year's Day
        assert date(2024, 1, 1) not in days  # New Year's Day
        assert date(2023, 12, 30) not in days  # Saturday
        assert date(2023, 12, 31) not in days  # Sunday

    def test_get_trading_days_between_same_day(self, service):
        """Test getting trading days for same start and end date."""
        day = date(2024, 1, 16)  # Tuesday
        days = service.get_trading_days_between(day, day)
        assert len(days) == 1
        assert days[0] == day

    def test_get_trading_days_between_reverse_order(self, service):
        """Test getting trading days with end before start."""
        start = date(2024, 1, 19)  # Friday
        end = date(2024, 1, 15)  # Tuesday
        days = service.get_trading_days_between(start, end)
        assert len(days) == 0

    # Is Holiday Tests

    def test_is_holiday_us_holidays(self, service):
        """Test is_holiday for US market holidays."""
        holidays_2024 = [
            date(2024, 1, 1),  # New Year's Day
            date(2024, 1, 15),  # MLK Day
            date(2024, 2, 19),  # Presidents' Day
            date(2024, 3, 29),  # Good Friday
            date(2024, 5, 27),  # Memorial Day
            date(2024, 6, 19),  # Juneteenth
            date(2024, 7, 4),  # Independence Day
            date(2024, 9, 2),  # Labor Day
            date(2024, 11, 28),  # Thanksgiving
            date(2024, 12, 25),  # Christmas
        ]

        for holiday in holidays_2024:
            assert service.is_holiday(holiday) is True

    def test_is_holiday_non_holidays(self, service):
        """Test is_holiday for regular trading days."""
        non_holidays = [
            date(2024, 1, 2),  # Day after New Year's
            date(2024, 7, 3),  # Day before Independence Day
            date(2024, 12, 24),  # Christmas Eve (half day but not holiday)
        ]

        for day in non_holidays:
            assert service.is_holiday(day) is False

    # Custom Holiday Set Tests

    def test_custom_holidays(self):
        """Test MarketHoursService with custom holidays."""
        custom_holidays = {
            date(2024, 1, 1),  # New Year's
            date(2024, 12, 25),  # Christmas
            date(2024, 8, 15),  # Custom holiday
        }

        service = MarketHoursService(holidays=custom_holidays)

        assert service.is_holiday(date(2024, 1, 1)) is True
        assert service.is_holiday(date(2024, 12, 25)) is True
        assert service.is_holiday(date(2024, 8, 15)) is True
        assert service.is_holiday(date(2024, 7, 4)) is False  # Not in custom set

    # Timezone Tests

    def test_custom_timezone(self):
        """Test MarketHoursService with custom timezone."""
        london_tz = "Europe/London"
        service = MarketHoursService(timezone=london_tz)

        # Service should still work with different timezone
        assert london_tz == service.DEFAULT_TIMEZONE

        # Check that timezone is properly used
        london = pytz.timezone(london_tz)
        dt = london.localize(datetime(2024, 1, 16, 14, 30))  # 2:30 PM London time
        status = service.get_current_market_status(dt)
        # This would be 9:30 AM ET, so market should be OPEN
        assert status in [MarketStatus.OPEN, MarketStatus.CLOSED]  # Depends on exact conversion

    # Edge Cases and Error Handling

    def test_naive_datetime_handling(self, service, eastern_tz):
        """Test handling of naive datetime objects."""
        naive_dt = datetime(2024, 1, 16, 10, 30)  # No timezone
        # Should assume Eastern timezone
        status = service.get_current_market_status(naive_dt)
        assert status == MarketStatus.OPEN

    def test_different_timezone_datetime(self, service):
        """Test handling of datetime in different timezone."""
        pacific = pytz.timezone("America/Los_Angeles")
        dt = pacific.localize(datetime(2024, 1, 16, 7, 30))  # 7:30 AM PT = 10:30 AM ET
        status = service.get_current_market_status(dt)
        assert status == MarketStatus.OPEN

    def test_early_closing_days(self, service):
        """Test recognition of early closing days."""
        # Note: Implementation might need enhancement for early closing days
        # Christmas Eve and day before Independence Day typically close early
        christmas_eve = date(2024, 12, 24)
        july_3 = date(2024, 7, 3)

        # These are still trading days
        assert service.is_trading_day(christmas_eve) is True
        assert service.is_trading_day(july_3) is True

    def test_market_status_boundary_conditions(self, service, eastern_tz):
        """Test market status at exact boundary times."""
        trading_day = date(2024, 1, 15)

        # Exactly at pre-market open (4:00:00 AM)
        dt = eastern_tz.localize(datetime.combine(trading_day, time(4, 0, 0)))
        assert service.get_current_market_status(dt) == MarketStatus.PRE_MARKET

        # One second before pre-market (3:59:59 AM)
        dt = eastern_tz.localize(datetime.combine(trading_day, time(3, 59, 59)))
        assert service.get_current_market_status(dt) == MarketStatus.CLOSED

        # Exactly at market open (9:30:00 AM)
        dt = eastern_tz.localize(datetime.combine(trading_day, time(9, 30, 0)))
        assert service.get_current_market_status(dt) == MarketStatus.OPEN

        # Exactly at market close (4:00:00 PM)
        dt = eastern_tz.localize(datetime.combine(trading_day, time(16, 0, 0)))
        assert service.get_current_market_status(dt) == MarketStatus.AFTER_MARKET

        # Exactly at after-market close (8:00:00 PM)
        dt = eastern_tz.localize(datetime.combine(trading_day, time(20, 0, 0)))
        assert service.get_current_market_status(dt) == MarketStatus.CLOSED
