"""
Gap Detection Service

Service for identifying missing data gaps in time-series data.
Part of the service-oriented architecture for historical data processing.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Any

# Local imports
from main.data_pipeline.historical.gap_priority_calculator import (
    GapPriorityCalculator,
    PriorityConfig,
    PriorityStrategy,
)

# Repository imports removed - initialized via factory in constructor
from main.data_pipeline.services.storage import TableRoutingService
from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.types import DataType, GapInfo, TimeInterval
from main.interfaces.database import IAsyncDatabase
from main.utils.core import ensure_utc, get_logger, get_trading_days_between, timer
from main.utils.core.time_helpers import is_market_open
from main.utils.time.interval_utils import TimeIntervalUtils

logger = get_logger(__name__)


@dataclass
class GapDetectionConfig:
    """Configuration for gap detection."""

    max_lookback_days: int = 365  # Maximum days to look back for gaps
    min_gap_size: int = 1  # Minimum gap size to report
    check_archive: bool = True  # Whether to check cold storage
    use_market_hours: bool = True  # Consider market hours for intraday data
    priority_strategy: PriorityStrategy = PriorityStrategy.STANDARD


@dataclass
class DataPointInfo:
    """Information about a data point."""

    timestamp: datetime
    exists: bool
    source: str = "unknown"  # 'hot', 'cold', or 'missing'


class GapDetectionService:
    """
    Service for detecting gaps in time-series data.

    This service identifies missing data periods across different data types
    and storage layers, prioritizing gaps based on layer and recency.
    """

    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        archive: DataArchive | None = None,
        config: GapDetectionConfig | None = None,
        market_data_repo: Any | None = None,
        news_repo: Any | None = None,
        table_routing_service: TableRoutingService | None = None,
        priority_calculator: GapPriorityCalculator | None = None,
    ):
        """
        Initialize the gap detection service.

        Args:
            db_adapter: Database adapter for hot storage queries
            archive: Optional archive for cold storage queries
            config: Service configuration
        """
        self.logger = get_logger(__name__)
        self.db_adapter = db_adapter
        self.archive = archive
        self.config = config or GapDetectionConfig()

        # Initialize repositories and services using factory if not provided
        if market_data_repo is None or news_repo is None:
            # Local imports
            from main.data_pipeline.storage.repositories import get_repository_factory

            factory = get_repository_factory()
            self.market_data_repo = market_data_repo or factory.create_market_data_repository(
                db_adapter
            )
            self.news_repo = news_repo or factory.create_news_repository(db_adapter)
        else:
            self.market_data_repo = market_data_repo
            self.news_repo = news_repo

        self.table_routing = table_routing_service or TableRoutingService()

        # Initialize priority calculator
        priority_config = PriorityConfig()
        self.priority_calculator = priority_calculator or GapPriorityCalculator(
            config=priority_config, strategy=self.config.priority_strategy
        )

        # Cache for market calendar
        self._market_calendar = None
        self._calendar_date = None

        self.logger.info("GapDetectionService initialized")

    @timer
    async def detect_gaps(
        self,
        symbol: str,
        data_type: DataType,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
        layer: int = 1,
    ) -> list[GapInfo]:
        """
        Detect gaps in data for a symbol.

        Args:
            symbol: Stock symbol
            data_type: Type of data to check
            interval: Time interval
            start_date: Start of period to check
            end_date: End of period to check
            layer: Symbol layer for prioritization

        Returns:
            List of GapInfo objects describing gaps
        """
        gaps = []

        try:
            # Ensure dates are UTC
            start_date = ensure_utc(start_date)
            end_date = ensure_utc(end_date)

            # Get expected and actual data points
            expected_points = await self._get_expected_data_points(
                symbol, data_type, interval, start_date, end_date
            )

            actual_points = await self._get_actual_data_points(
                symbol, data_type, interval, start_date, end_date
            )

            # Find gaps
            raw_gaps = self._identify_gaps(expected_points, actual_points)

            # Convert to GapInfo objects with priority
            for gap_start, gap_end in raw_gaps:
                gap_size = self._calculate_gap_size(gap_start, gap_end, interval)

                if gap_size >= self.config.min_gap_size:
                    priority = self.priority_calculator.calculate_priority(
                        data_type=data_type,
                        layer=layer,
                        start_date=gap_start,
                        end_date=gap_end,
                        gap_size=gap_size,
                    )

                    gap = GapInfo(
                        symbol=symbol,
                        data_type=data_type,
                        interval=interval,
                        start_date=gap_start,
                        end_date=gap_end,
                        records_missing=gap_size,
                        priority=priority,
                    )
                    gaps.append(gap)

            # Sort by priority (highest first)
            gaps.sort(key=lambda g: g.priority)

            self.logger.debug(
                f"Found {len(gaps)} gaps for {symbol}/{data_type.value}/{interval.value}"
            )

        except Exception as e:
            self.logger.error(f"Error detecting gaps for {symbol}: {e}")

        return gaps

    async def detect_market_data_gaps(
        self,
        symbol: str,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
        layer: int = 1,
    ) -> list[GapInfo]:
        """
        Detect gaps specifically in market data.

        Args:
            symbol: Stock symbol
            interval: Time interval (1min, 5min, 1hour, 1day)
            start_date: Start date
            end_date: End date
            layer: Symbol layer

        Returns:
            List of gaps in market data
        """
        return await self.detect_gaps(
            symbol=symbol,
            data_type=DataType.MARKET_DATA,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            layer=layer,
        )

    async def detect_news_gaps(
        self, symbol: str, start_date: datetime, end_date: datetime, layer: int = 1
    ) -> list[GapInfo]:
        """
        Detect gaps in news data.

        News doesn't have regular intervals, so we check for days without news.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            layer: Symbol layer

        Returns:
            List of gaps in news coverage
        """
        gaps = []

        try:
            # For news, we check daily coverage
            current_date = start_date.date()
            end = end_date.date()

            while current_date <= end:
                check_date = datetime.combine(current_date, time(0, 0), UTC)

                # Check if we have news for this day
                has_news = await self._check_news_for_date(symbol, check_date)

                if not has_news:
                    # Create a gap for this day
                    gap_end = check_date + timedelta(days=1)
                    priority = self.priority_calculator.calculate_priority(
                        data_type=DataType.NEWS,
                        layer=layer,
                        start_date=check_date,
                        end_date=gap_end,
                        gap_size=1,
                    )

                    gap = GapInfo(
                        symbol=symbol,
                        data_type=DataType.NEWS,
                        interval=TimeInterval.ONE_DAY,
                        start_date=check_date,
                        end_date=gap_end,
                        records_missing=1,  # One day of news
                        priority=priority,
                    )
                    gaps.append(gap)

                current_date += timedelta(days=1)

            # Merge consecutive gaps
            gaps = self._merge_consecutive_gaps(gaps)

        except Exception as e:
            self.logger.error(f"Error detecting news gaps for {symbol}: {e}")

        return gaps

    async def _get_expected_data_points(
        self,
        symbol: str,
        data_type: DataType,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
    ) -> set[datetime]:
        """
        Calculate expected data points based on interval and market hours.

        Args:
            symbol: Stock symbol
            data_type: Type of data
            interval: Time interval
            start_date: Start date
            end_date: End date

        Returns:
            Set of expected timestamps
        """
        expected = set()

        if data_type != DataType.MARKET_DATA:
            # For non-market data, we don't have strict expectations
            return expected

        # Get interval duration
        interval_delta = TimeIntervalUtils.get_interval_timedelta_from_enum(interval)

        if interval == TimeInterval.ONE_DAY:
            # For daily data, use trading days
            trading_days = get_trading_days_between(start_date, end_date)
            for day in trading_days:
                expected.add(ensure_utc(day))

        elif interval in [
            TimeInterval.ONE_MINUTE,
            TimeInterval.FIVE_MINUTES,
            TimeInterval.FIFTEEN_MINUTES,
            TimeInterval.THIRTY_MINUTES,
            TimeInterval.ONE_HOUR,
        ]:
            # For intraday data, consider market hours
            current = start_date

            while current <= end_date:
                if self.config.use_market_hours:
                    if is_market_open(current):
                        expected.add(current)
                elif current.weekday() < 5:  # Monday-Friday
                    expected.add(current)

                current += interval_delta

        else:
            # For weekly/monthly, use regular intervals
            current = start_date
            while current <= end_date:
                expected.add(current)
                current += interval_delta

        return expected

    async def _get_actual_data_points(
        self,
        symbol: str,
        data_type: DataType,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
    ) -> set[datetime]:
        """
        Get actual data points from hot and cold storage.

        Args:
            symbol: Stock symbol
            data_type: Type of data
            interval: Time interval
            start_date: Start date
            end_date: End date

        Returns:
            Set of actual timestamps where data exists
        """
        actual = set()

        # Query hot storage (PostgreSQL)
        hot_points = await self._query_hot_storage(
            symbol, data_type, interval, start_date, end_date
        )
        actual.update(hot_points)

        # Query cold storage (Archive) if configured
        if self.config.check_archive and self.archive:
            cold_points = await self._query_cold_storage(
                symbol, data_type, interval, start_date, end_date
            )
            actual.update(cold_points)

        return actual

    async def _query_hot_storage(
        self,
        symbol: str,
        data_type: DataType,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
    ) -> set[datetime]:
        """Query hot storage for existing data points."""
        points = set()

        try:
            if data_type == DataType.MARKET_DATA:
                # Use market data repository
                data = await self.market_data_repo.get_ohlcv(
                    symbol=symbol, start_date=start_date, end_date=end_date, interval=interval.value
                )

                if data is not None and not data.empty:
                    for timestamp in data.index:
                        points.add(ensure_utc(timestamp))

            elif data_type == DataType.NEWS:
                # Use news repository
                news_items = await self.news_repo.get_news_by_symbol(
                    symbol=symbol, start_date=start_date, end_date=end_date
                )

                if news_items:
                    for item in news_items:
                        if "published_at" in item:
                            points.add(
                                ensure_utc(item["published_at"]).replace(
                                    hour=0, minute=0, second=0, microsecond=0
                                )
                            )

        except Exception as e:
            self.logger.error(f"Error querying hot storage: {e}")

        return points

    async def _query_cold_storage(
        self,
        symbol: str,
        data_type: DataType,
        interval: TimeInterval,
        start_date: datetime,
        end_date: datetime,
    ) -> set[datetime]:
        """Query cold storage for existing data points."""
        points = set()

        try:
            # Query archive
            records = await self.archive.query_raw_records(
                source="polygon",  # Primary source
                data_type=data_type.value,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

            for record in records:
                # Extract timestamps from records
                if hasattr(record, "timestamp"):
                    points.add(ensure_utc(record.timestamp))
                elif hasattr(record, "data") and isinstance(record.data, list):
                    # For batch records, extract individual timestamps
                    for item in record.data:
                        if isinstance(item, dict) and "timestamp" in item:
                            points.add(ensure_utc(item["timestamp"]))

        except Exception as e:
            self.logger.error(f"Error querying cold storage: {e}")

        return points

    async def _check_news_for_date(self, symbol: str, date: datetime) -> bool:
        """Check if news exists for a specific date."""
        try:
            # Use news repository to check for news on specific date
            end_date = date + timedelta(days=1)
            news_items = await self.news_repo.get_news_by_symbol(
                symbol=symbol, start_date=date, end_date=end_date
            )

            return len(news_items) > 0 if news_items else False

        except Exception as e:
            self.logger.error(f"Error checking news for {symbol} on {date}: {e}")
            return False

    def _identify_gaps(
        self, expected: set[datetime], actual: set[datetime]
    ) -> list[tuple[datetime, datetime]]:
        """
        Identify continuous gaps between expected and actual points.

        Args:
            expected: Set of expected timestamps
            actual: Set of actual timestamps

        Returns:
            List of (start, end) tuples for gaps
        """
        if not expected:
            return []

        # Find missing points
        missing = sorted(expected - actual)

        if not missing:
            return []

        # Group consecutive missing points into gaps
        gaps = []
        gap_start = missing[0]
        gap_end = missing[0]

        for i in range(1, len(missing)):
            # Check if consecutive (within reasonable range)
            time_diff = (missing[i] - gap_end).total_seconds()

            # Allow up to 1 day gap for continuity
            if time_diff <= 86400:  # 24 hours
                gap_end = missing[i]
            else:
                # Save current gap and start new one
                gaps.append((gap_start, gap_end))
                gap_start = missing[i]
                gap_end = missing[i]

        # Add final gap
        gaps.append((gap_start, gap_end))

        return gaps

    def _merge_consecutive_gaps(self, gaps: list[GapInfo]) -> list[GapInfo]:
        """Merge consecutive gaps into larger gaps."""
        if not gaps:
            return []

        # Sort by start date
        gaps.sort(key=lambda g: g.start_date)

        merged = []
        current = gaps[0]

        for gap in gaps[1:]:
            # Check if consecutive
            time_diff = (gap.start_date - current.end_date).total_seconds()

            if time_diff <= 86400:  # Within 1 day
                # Merge gaps
                current = GapInfo(
                    symbol=current.symbol,
                    data_type=current.data_type,
                    interval=current.interval,
                    start_date=current.start_date,
                    end_date=gap.end_date,
                    records_missing=current.records_missing + gap.records_missing,
                    priority=max(current.priority, gap.priority),
                )
            else:
                # Save current and start new
                merged.append(current)
                current = gap

        # Add final gap
        merged.append(current)

        return merged

    def _calculate_gap_size(
        self, start_date: datetime, end_date: datetime, interval: TimeInterval
    ) -> int:
        """Calculate the number of missing data points in a gap."""
        delta = TimeIntervalUtils.get_interval_timedelta_from_enum(interval)
        total_seconds = (end_date - start_date).total_seconds()
        interval_seconds = delta.total_seconds()

        return max(1, int(total_seconds / interval_seconds))

    def get_metrics(self) -> dict[str, Any]:
        """Get service metrics."""
        return {
            "config": {
                "max_lookback_days": self.config.max_lookback_days,
                "min_gap_size": self.config.min_gap_size,
                "check_archive": self.config.check_archive,
            }
        }

    async def prioritize_gaps(
        self, gaps: list[GapInfo], max_gaps: int | None = None
    ) -> list[GapInfo]:
        """
        Prioritize and limit gaps for processing.

        Args:
            gaps: List of gaps to prioritize
            max_gaps: Maximum number of gaps to return

        Returns:
            Prioritized list of gaps
        """
        # Sort by priority (lower value = higher priority)
        sorted_gaps = sorted(gaps, key=lambda g: g.priority)

        if max_gaps:
            return sorted_gaps[:max_gaps]

        return sorted_gaps
