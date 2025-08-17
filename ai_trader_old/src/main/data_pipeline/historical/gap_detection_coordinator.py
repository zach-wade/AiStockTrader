"""
Gap Detection Coordinator Service

Coordinates gap detection operations by composing specialized services.
Provides the main entry point for gap detection functionality.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

# Local imports
from main.data_pipeline.services.storage import TableRoutingService
from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.storage.repositories import MarketDataRepository, NewsRepository
from main.data_pipeline.types import DataType, GapInfo, TimeInterval
from main.interfaces.database import IAsyncDatabase
from main.utils.core import ensure_utc, get_logger

from .data_existence_checker import DataExistenceChecker
from .gap_analyzer import GapAnalyzer
from .timeline_analyzer import TimelineAnalyzer


@dataclass
class GapDetectionRequest:
    """Request for gap detection operation."""

    symbol: str
    data_type: DataType
    interval: TimeInterval
    start_date: datetime
    end_date: datetime
    check_archive: bool = True
    use_market_hours: bool = True


@dataclass
class GapDetectionResult:
    """Result of gap detection operation."""

    symbol: str
    data_type: DataType
    interval: TimeInterval
    gaps: list[GapInfo]
    statistics: dict[str, Any]
    timeline_stats: dict[str, Any]
    existence_stats: dict[str, Any]


class GapDetectionCoordinator:
    """
    Coordinator for gap detection operations.

    Composes timeline analysis, data existence checking, and gap analysis
    to provide comprehensive gap detection functionality.
    """

    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        archive: DataArchive | None = None,
        market_data_repo: MarketDataRepository | None = None,
        news_repo: NewsRepository | None = None,
        table_routing: TableRoutingService | None = None,
    ):
        """
        Initialize the gap detection coordinator.

        Args:
            db_adapter: Database adapter for hot storage
            archive: Optional archive for cold storage
            market_data_repo: Market data repository
            news_repo: News repository
            table_routing: Table routing service
        """
        self.logger = get_logger(__name__)

        # Initialize composed services
        self.timeline_analyzer = TimelineAnalyzer(use_market_hours=True)
        self.existence_checker = DataExistenceChecker(
            db_adapter=db_adapter,
            archive=archive,
            market_data_repo=market_data_repo,
            news_repo=news_repo,
            table_routing=table_routing,
        )
        self.gap_analyzer = GapAnalyzer()

        self.logger.info("GapDetectionCoordinator initialized with composed services")

    async def detect_gaps(self, request: GapDetectionRequest) -> GapDetectionResult:
        """
        Detect gaps for a symbol and time range.

        Args:
            request: Gap detection request

        Returns:
            Gap detection result with gaps and statistics
        """
        self.logger.info(
            f"Starting gap detection for {request.symbol} "
            f"{request.data_type.value} {request.interval.value}"
        )

        # Normalize dates
        start_date = ensure_utc(request.start_date)
        end_date = ensure_utc(request.end_date)

        try:
            # Step 1: Generate expected timeline
            expected_timeline = self.timeline_analyzer.get_expected_data_points(
                symbol=request.symbol,
                data_type=request.data_type,
                interval=request.interval,
                start_date=start_date,
                end_date=end_date,
            )

            self.logger.debug(f"Generated {len(expected_timeline)} expected data points")

            # Step 2: Check actual data existence
            actual_data = await self.existence_checker.get_actual_data_points(
                symbol=request.symbol,
                data_type=request.data_type,
                interval=request.interval,
                start_date=start_date,
                end_date=end_date,
                check_archive=request.check_archive,
            )

            self.logger.debug(f"Found {len(actual_data)} actual data points")

            # Step 3: Analyze gaps
            gap_segments = self.gap_analyzer.identify_gaps(
                expected_timeline=expected_timeline,
                actual_data=actual_data,
                data_type=request.data_type,
                interval=request.interval,
            )

            self.logger.debug(f"Identified {len(gap_segments)} gap segments")

            # Step 4: Convert to GapInfo objects
            gaps = self.gap_analyzer.convert_to_gap_info(
                segments=gap_segments,
                symbol=request.symbol,
                data_type=request.data_type,
                interval=request.interval,
            )

            # Step 5: Generate statistics
            gap_statistics = self.gap_analyzer.get_gap_statistics(gap_segments)
            timeline_stats = self.timeline_analyzer.get_timeline_stats(expected_timeline)
            existence_stats = self.existence_checker.get_existence_summary(actual_data)

            result = GapDetectionResult(
                symbol=request.symbol,
                data_type=request.data_type,
                interval=request.interval,
                gaps=gaps,
                statistics=gap_statistics,
                timeline_stats=timeline_stats,
                existence_stats=existence_stats,
            )

            self.logger.info(
                f"Gap detection completed: {len(gaps)} gaps found "
                f"({gap_statistics.get('total_missing_points', 0)} missing points)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Gap detection failed: {e}")
            raise

    async def detect_gaps_batch(
        self, requests: list[GapDetectionRequest]
    ) -> list[GapDetectionResult]:
        """
        Detect gaps for multiple requests in batch.

        Args:
            requests: List of gap detection requests

        Returns:
            List of gap detection results
        """
        results = []

        self.logger.info(f"Starting batch gap detection for {len(requests)} requests")

        for i, request in enumerate(requests):
            try:
                result = await self.detect_gaps(request)
                results.append(result)

                self.logger.debug(f"Completed request {i+1}/{len(requests)}")

            except Exception as e:
                self.logger.error(f"Failed request {i+1}/{len(requests)}: {e}")
                # Create empty result for failed request
                results.append(
                    GapDetectionResult(
                        symbol=request.symbol,
                        data_type=request.data_type,
                        interval=request.interval,
                        gaps=[],
                        statistics={},
                        timeline_stats={},
                        existence_stats={},
                    )
                )

        self.logger.info(f"Batch gap detection completed: {len(results)} results")

        return results

    async def get_gap_summary(
        self,
        symbol: str,
        data_types: list[DataType] | None = None,
        intervals: list[TimeInterval] | None = None,
        days_back: int = 30,
    ) -> dict[str, Any]:
        """
        Get gap summary for a symbol across data types and intervals.

        Args:
            symbol: Symbol to analyze
            data_types: Data types to check (default: all)
            intervals: Intervals to check (default: common ones)
            days_back: How many days back to analyze

        Returns:
            Summary of gaps across all requested combinations
        """
        if not data_types:
            data_types = [DataType.MARKET_DATA, DataType.NEWS]

        if not intervals:
            intervals = [TimeInterval.ONE_DAY, TimeInterval.ONE_HOUR]

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Generate requests for all combinations
        requests = []
        for data_type in data_types:
            for interval in intervals:
                # Skip irrelevant combinations
                if data_type == DataType.NEWS and interval != TimeInterval.ONE_DAY:
                    continue

                requests.append(
                    GapDetectionRequest(
                        symbol=symbol,
                        data_type=data_type,
                        interval=interval,
                        start_date=start_date,
                        end_date=end_date,
                    )
                )

        # Execute batch detection
        results = await self.detect_gaps_batch(requests)

        # Aggregate results
        summary = {
            "symbol": symbol,
            "analysis_period": {"start": start_date, "end": end_date, "days": days_back},
            "by_data_type": {},
            "overall": {
                "total_gaps": 0,
                "total_missing_points": 0,
                "critical_gaps": 0,
                "high_priority_gaps": 0,
            },
        }

        for result in results:
            key = f"{result.data_type.value}_{result.interval.value}"

            summary["by_data_type"][key] = {
                "gaps": len(result.gaps),
                "missing_points": result.statistics.get("total_missing_points", 0),
                "severity_distribution": result.statistics.get("severity_distribution", {}),
                "coverage_percentage": result.existence_stats.get("coverage_percentage", 0),
            }

            # Update overall stats
            summary["overall"]["total_gaps"] += len(result.gaps)
            summary["overall"]["total_missing_points"] += result.statistics.get(
                "total_missing_points", 0
            )

            # Count critical gaps
            for gap in result.gaps:
                if gap.priority_score >= 4.0:
                    summary["overall"]["critical_gaps"] += 1
                elif gap.priority_score >= 3.0:
                    summary["overall"]["high_priority_gaps"] += 1

        return summary

    def get_coordinator_stats(self) -> dict[str, Any]:
        """Get statistics about the coordinator and its services."""
        return {
            "coordinator": {
                "timeline_analyzer": {
                    "use_market_hours": self.timeline_analyzer.use_market_hours,
                    "market_calendar_cache_size": len(self.timeline_analyzer._market_calendar),
                },
                "existence_checker": {
                    "has_archive": self.existence_checker.archive is not None,
                    "existence_cache_size": len(self.existence_checker._existence_cache),
                },
                "gap_analyzer": {"initialized": True},
            }
        }
