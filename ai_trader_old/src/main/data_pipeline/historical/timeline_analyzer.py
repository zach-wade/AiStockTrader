"""
Timeline Analyzer Service

Generates expected data timelines for gap detection.
Handles market hours, trading days, and data frequency calculations.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

# Local imports
from main.data_pipeline.types import DataType, TimeInterval
from main.utils.core import ensure_utc, get_logger, get_trading_days_between
from main.utils.core.time_helpers import get_market_hours, is_market_open


@dataclass
class DataPointInfo:
    """Information about an expected data point."""

    timestamp: datetime
    is_trading_time: bool
    interval: str
    data_type: str


class TimelineAnalyzer:
    """
    Service for analyzing and generating expected data timelines.

    Determines when data should exist based on:
    - Market hours and trading days
    - Data type and frequency
    - Historical patterns
    """

    def __init__(self, use_market_hours: bool = True):
        """
        Initialize the timeline analyzer.

        Args:
            use_market_hours: Whether to consider market hours for intraday data
        """
        self.use_market_hours = use_market_hours
        self.logger = get_logger(__name__)

        # Cache for market calendar
        self._market_calendar = {}

    def get_expected_data_points(
        self,
        symbol: str,
        data_type: DataType,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
    ) -> list[DataPointInfo]:
        """
        Generate expected data points for a symbol and time range.

        Args:
            symbol: Symbol to analyze
            data_type: Type of data (market_data, news, etc.)
            interval: Time interval
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            List of expected data points
        """
        start_date = ensure_utc(start_date)
        end_date = ensure_utc(end_date)

        self.logger.debug(
            f"Generating timeline for {symbol} {data_type.value} {interval.value} "
            f"from {start_date} to {end_date}"
        )

        if data_type == DataType.MARKET_DATA:
            return self._get_market_data_timeline(symbol, interval, start_date, end_date)
        elif data_type == DataType.NEWS:
            return self._get_news_timeline(symbol, start_date, end_date)
        else:
            return self._get_generic_timeline(symbol, data_type, interval, start_date, end_date)

    def _get_market_data_timeline(
        self, symbol: str, interval: TimeInterval, start_date: datetime, end_date: datetime
    ) -> list[DataPointInfo]:
        """Generate expected timeline for market data."""
        timeline = []

        if interval == TimeInterval.ONE_DAY:
            # Daily data - one point per trading day
            trading_days = self._get_trading_days(start_date, end_date)

            for day in trading_days:
                # Market close time for daily data
                close_time = day.replace(hour=16, minute=0, second=0, microsecond=0)

                timeline.append(
                    DataPointInfo(
                        timestamp=close_time,
                        is_trading_time=True,
                        interval=interval.value,
                        data_type=DataType.MARKET_DATA.value,
                    )
                )

        elif interval in [TimeInterval.ONE_HOUR, TimeInterval.FIFTEEN_MIN, TimeInterval.FIVE_MIN]:
            # Intraday data - multiple points per trading day
            trading_days = self._get_trading_days(start_date, end_date)
            interval_minutes = self._get_interval_minutes(interval)

            for day in trading_days:
                day_timeline = self._get_trading_day_timeline(day, interval_minutes, interval.value)
                timeline.extend(day_timeline)

        return timeline

    def _get_news_timeline(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> list[DataPointInfo]:
        """
        Generate expected timeline for news data.

        News can occur any time, but we expect at least some news
        on major trading days for active symbols.
        """
        timeline = []
        trading_days = self._get_trading_days(start_date, end_date)

        # For news, we expect at least daily checks
        for day in trading_days:
            # Check at market open for daily news
            open_time = day.replace(hour=9, minute=30, second=0, microsecond=0)

            timeline.append(
                DataPointInfo(
                    timestamp=open_time,
                    is_trading_time=True,
                    interval="daily",
                    data_type=DataType.NEWS.value,
                )
            )

        return timeline

    def _get_generic_timeline(
        self,
        symbol: str,
        data_type: DataType,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
    ) -> list[DataPointInfo]:
        """Generate timeline for other data types."""
        timeline = []

        # For other data types, generate based on interval
        current = start_date
        interval_minutes = self._get_interval_minutes(interval)

        while current <= end_date:
            timeline.append(
                DataPointInfo(
                    timestamp=current,
                    is_trading_time=self._is_trading_time(current),
                    interval=interval.value,
                    data_type=data_type.value,
                )
            )

            current += timedelta(minutes=interval_minutes)

        return timeline

    def _get_trading_days(self, start_date: datetime, end_date: datetime) -> list[datetime]:
        """Get list of trading days between dates."""
        # Use cached calendar if available
        cache_key = f"{start_date.date()}_{end_date.date()}"
        if cache_key in self._market_calendar:
            return self._market_calendar[cache_key]

        # Generate trading days
        trading_days = get_trading_days_between(start_date, end_date)

        # Cache the result
        self._market_calendar[cache_key] = trading_days

        return trading_days

    def _get_trading_day_timeline(
        self, trading_day: datetime, interval_minutes: int, interval_name: str
    ) -> list[DataPointInfo]:
        """Generate timeline for a single trading day."""
        timeline = []

        if not self.use_market_hours:
            # 24-hour timeline
            current = trading_day.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = trading_day.replace(hour=23, minute=59, second=59, microsecond=0)
        else:
            # Market hours only
            market_open, market_close = get_market_hours(trading_day)
            current = market_open
            day_end = market_close

        while current <= day_end:
            timeline.append(
                DataPointInfo(
                    timestamp=current,
                    is_trading_time=self._is_trading_time(current),
                    interval=interval_name,
                    data_type=DataType.MARKET_DATA.value,
                )
            )

            current += timedelta(minutes=interval_minutes)

        return timeline

    def _get_interval_minutes(self, interval: TimeInterval) -> int:
        """Convert interval to minutes."""
        interval_map = {
            TimeInterval.ONE_MIN: 1,
            TimeInterval.FIVE_MIN: 5,
            TimeInterval.FIFTEEN_MIN: 15,
            TimeInterval.THIRTY_MIN: 30,
            TimeInterval.ONE_HOUR: 60,
            TimeInterval.ONE_DAY: 1440,  # 24 hours
        }
        return interval_map.get(interval, 60)

    def _is_trading_time(self, timestamp: datetime) -> bool:
        """Check if timestamp is during trading hours."""
        if not self.use_market_hours:
            return True

        return is_market_open(timestamp)

    def get_timeline_stats(self, timeline: list[DataPointInfo]) -> dict[str, Any]:
        """
        Get statistics about a timeline.

        Args:
            timeline: Timeline to analyze

        Returns:
            Dictionary of timeline statistics
        """
        if not timeline:
            return {"total_points": 0, "trading_points": 0, "date_range": None}

        trading_points = sum(1 for point in timeline if point.is_trading_time)

        return {
            "total_points": len(timeline),
            "trading_points": trading_points,
            "non_trading_points": len(timeline) - trading_points,
            "date_range": {"start": timeline[0].timestamp, "end": timeline[-1].timestamp},
            "intervals": list(set(point.interval for point in timeline)),
            "data_types": list(set(point.data_type for point in timeline)),
        }
