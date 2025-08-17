"""
Data Coverage Analyzer - Interface Implementation

Analyzes data coverage across multiple storage layers and time dimensions.
Implements ICoverageAnalyzer interface for comprehensive coverage analysis.
"""

# Standard library imports
from datetime import datetime
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
# Interface imports
from main.interfaces.validation import IValidationContext

# Core imports
from main.utils.core import get_logger

logger = get_logger(__name__)


class DataCoverageAnalyzer:
    """
    Data coverage analyzer implementation.

    Implements ICoverageAnalyzer interface for comprehensive
    data coverage analysis across storage layers.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the data coverage analyzer.

        Args:
            config: Configuration dictionary with coverage settings
        """
        self.config = config

        # Coverage settings
        self.cache_freshness_hours = config.get("cache_freshness_hours", 24)
        self.expected_intervals = config.get("expected_intervals", ["1day", "1hour"])
        self.coverage_threshold = config.get("coverage_threshold", 0.8)
        self.gap_tolerance_hours = config.get("gap_tolerance_hours", 2)

        # Database and storage references (would be injected in real implementation)
        self.db_adapter = config.get("db_adapter")
        self.archive = config.get("archive")

        logger.info("Initialized DataCoverageAnalyzer with interface-based architecture")

    # ICoverageAnalyzer interface methods
    async def analyze_temporal_coverage(
        self, data: Any, context: IValidationContext, expected_timeframe: tuple | None = None
    ) -> dict[str, Any]:
        """Analyze temporal data coverage."""
        if not isinstance(data, pd.DataFrame):
            return {"error": "Data must be a pandas DataFrame for temporal analysis"}

        if not isinstance(data.index, pd.DatetimeIndex):
            return {"error": "DataFrame must have DatetimeIndex for temporal analysis"}

        # Determine expected timeframe
        if expected_timeframe:
            start_time, end_time = expected_timeframe
        else:
            # Use data range with some buffer
            start_time = data.index.min()
            end_time = data.index.max()

        # Analyze coverage
        coverage_stats = await self._analyze_time_coverage(data, start_time, end_time, context)

        return {
            "coverage_score": coverage_stats["coverage_score"],
            "expected_points": coverage_stats["expected_points"],
            "actual_points": coverage_stats["actual_points"],
            "missing_points": coverage_stats["missing_points"],
            "gaps": coverage_stats["gaps"],
            "largest_gap": coverage_stats["largest_gap"],
            "gap_count": coverage_stats["gap_count"],
            "timeframe": {
                "start": (
                    start_time.isoformat() if hasattr(start_time, "isoformat") else str(start_time)
                ),
                "end": end_time.isoformat() if hasattr(end_time, "isoformat") else str(end_time),
            },
        }

    async def analyze_symbol_coverage(
        self, data: Any, context: IValidationContext, expected_symbols: list[str] | None = None
    ) -> dict[str, Any]:
        """Analyze symbol coverage."""
        if isinstance(data, pd.DataFrame):
            # Extract symbols from DataFrame
            if "symbol" in data.columns:
                actual_symbols = set(data["symbol"].unique())
            elif hasattr(data, "symbol") and data.symbol is not None:
                actual_symbols = {data.symbol}
            else:
                actual_symbols = {context.symbol} if context.symbol else set()
        elif isinstance(data, list):
            # Extract symbols from list of records
            actual_symbols = set()
            for record in data:
                if isinstance(record, dict) and "symbol" in record:
                    actual_symbols.add(record["symbol"])
        else:
            actual_symbols = {context.symbol} if context.symbol else set()

        # Compare with expected symbols
        if expected_symbols:
            expected_set = set(expected_symbols)
            missing_symbols = expected_set - actual_symbols
            extra_symbols = actual_symbols - expected_set
            coverage_score = (
                len(actual_symbols & expected_set) / len(expected_set) if expected_set else 1.0
            )
        else:
            missing_symbols = set()
            extra_symbols = set()
            coverage_score = 1.0 if actual_symbols else 0.0

        return {
            "coverage_score": coverage_score,
            "expected_symbols": list(expected_symbols) if expected_symbols else [],
            "actual_symbols": list(actual_symbols),
            "missing_symbols": list(missing_symbols),
            "extra_symbols": list(extra_symbols),
            "symbol_count": len(actual_symbols),
        }

    async def analyze_field_coverage(
        self, data: Any, context: IValidationContext, required_fields: list[str] | None = None
    ) -> dict[str, Any]:
        """Analyze field/column coverage."""
        if isinstance(data, pd.DataFrame):
            actual_fields = set(data.columns)
            # Calculate field completeness
            field_completeness = {}
            for field in actual_fields:
                non_null_count = data[field].count()
                total_count = len(data)
                field_completeness[field] = (
                    (non_null_count / total_count) if total_count > 0 else 0.0
                )
        elif isinstance(data, list) and data:
            # Analyze fields from list of records
            all_fields = set()
            field_counts = {}

            for record in data:
                if isinstance(record, dict):
                    for field in record.keys():
                        all_fields.add(field)
                        if field not in field_counts:
                            field_counts[field] = 0
                        if record[field] is not None:
                            field_counts[field] += 1

            actual_fields = all_fields
            field_completeness = {
                field: (field_counts.get(field, 0) / len(data)) for field in actual_fields
            }
        else:
            actual_fields = set()
            field_completeness = {}

        # Compare with required fields
        if required_fields:
            required_set = set(required_fields)
            missing_fields = required_set - actual_fields
            extra_fields = actual_fields - required_set
            coverage_score = (
                len(actual_fields & required_set) / len(required_set) if required_set else 1.0
            )
        else:
            missing_fields = set()
            extra_fields = set()
            coverage_score = 1.0 if actual_fields else 0.0

        # Calculate average field completeness
        avg_completeness = (
            sum(field_completeness.values()) / len(field_completeness)
            if field_completeness
            else 0.0
        )

        return {
            "coverage_score": coverage_score,
            "field_completeness_score": avg_completeness,
            "required_fields": list(required_fields) if required_fields else [],
            "actual_fields": list(actual_fields),
            "missing_fields": list(missing_fields),
            "extra_fields": list(extra_fields),
            "field_completeness": field_completeness,
            "total_fields": len(actual_fields),
        }

    async def calculate_coverage_score(self, data: Any, context: IValidationContext) -> float:
        """Calculate overall coverage score."""
        # Get individual coverage scores
        temporal_result = await self.analyze_temporal_coverage(data, context)
        symbol_result = await self.analyze_symbol_coverage(data, context)
        field_result = await self.analyze_field_coverage(data, context)

        # Weight the different coverage dimensions
        temporal_weight = 0.4
        symbol_weight = 0.3
        field_weight = 0.3

        temporal_score = temporal_result.get("coverage_score", 0.0)
        symbol_score = symbol_result.get("coverage_score", 0.0)
        field_score = field_result.get("coverage_score", 0.0)

        overall_score = (
            temporal_score * temporal_weight
            + symbol_score * symbol_weight
            + field_score * field_weight
        )

        return min(1.0, max(0.0, overall_score))

    # Additional methods for comprehensive coverage analysis
    async def analyze_multi_symbol_coverage(
        self, symbols: list[str], intervals: list[str], force_refresh: bool = False
    ) -> dict[str, Any]:
        """Analyze coverage for multiple symbols and intervals."""
        logger.info(f"Analyzing coverage for {len(symbols)} symbols and {len(intervals)} intervals")

        # This would integrate with database and archive in real implementation
        coverage_results = {}

        for symbol in symbols:
            symbol_coverage = {"intervals": {}}

            for interval in intervals:
                # Simulate coverage analysis
                # In real implementation, this would query DB and archive
                interval_coverage = await self._analyze_symbol_interval_coverage(symbol, interval)
                symbol_coverage["intervals"][interval] = interval_coverage

            coverage_results[symbol] = symbol_coverage

        # Generate summary
        summary = self._generate_coverage_summary(coverage_results)

        return {
            "coverage_by_symbol": coverage_results,
            "summary": summary,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    async def check_coverage_gaps(
        self, data: pd.DataFrame, expected_frequency: str = "D"
    ) -> dict[str, Any]:
        """Check for gaps in time series data."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return {"error": "DataFrame must have DatetimeIndex"}

        if data.empty:
            return {"gaps": [], "gap_count": 0, "largest_gap": None}

        # Generate expected date range
        expected_range = pd.date_range(
            start=data.index.min(), end=data.index.max(), freq=expected_frequency
        )

        # Find missing dates
        missing_dates = expected_range.difference(data.index)

        # Group consecutive missing dates into gaps
        gaps = []
        if len(missing_dates) > 0:
            gaps = self._group_consecutive_dates(missing_dates)

        # Calculate gap statistics
        gap_durations = [gap["duration_hours"] for gap in gaps]
        largest_gap = max(gap_durations) if gap_durations else 0

        return {
            "gaps": gaps,
            "gap_count": len(gaps),
            "largest_gap": largest_gap,
            "missing_points": len(missing_dates),
            "expected_points": len(expected_range),
            "coverage_ratio": (len(expected_range) - len(missing_dates)) / len(expected_range),
        }

    # Private helper methods
    async def _analyze_time_coverage(
        self,
        data: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
        context: IValidationContext,
    ) -> dict[str, Any]:
        """Analyze temporal coverage for a specific timeframe."""
        # Infer expected frequency from data
        if len(data) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            median_diff = time_diffs.median()

            # Map to pandas frequency
            if median_diff <= pd.Timedelta(minutes=5):
                freq = "5T"
            elif median_diff <= pd.Timedelta(hours=1):
                freq = "1H"
            elif median_diff <= pd.Timedelta(days=1):
                freq = "D"
            else:
                freq = "W"
        else:
            freq = "D"  # Default

        # Generate expected range
        expected_range = pd.date_range(start=start_time, end=end_time, freq=freq)
        actual_points = len(data)
        expected_points = len(expected_range)

        # Find gaps
        missing_dates = expected_range.difference(data.index)
        gaps = self._group_consecutive_dates(missing_dates) if len(missing_dates) > 0 else []

        # Calculate metrics
        coverage_score = (
            (expected_points - len(missing_dates)) / expected_points if expected_points > 0 else 0.0
        )
        largest_gap = max([gap["duration_hours"] for gap in gaps]) if gaps else 0

        return {
            "coverage_score": coverage_score,
            "expected_points": expected_points,
            "actual_points": actual_points,
            "missing_points": len(missing_dates),
            "gaps": gaps,
            "largest_gap": largest_gap,
            "gap_count": len(gaps),
        }

    async def _analyze_symbol_interval_coverage(self, symbol: str, interval: str) -> dict[str, Any]:
        """Analyze coverage for a specific symbol and interval."""
        # This is a placeholder - in real implementation would query actual storage

        # Simulate database coverage check
        db_coverage = {
            "has_data": True,
            "record_count": 1000,
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "gaps": [],
        }

        # Simulate data lake coverage check
        lake_coverage = {
            "has_data": True,
            "file_count": 50,
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "gaps": [],
        }

        # Merge coverage metrics
        return self._merge_coverage_metrics(db_coverage, lake_coverage, interval)

    def _merge_coverage_metrics(
        self, db_info: dict[str, Any], lake_info: dict[str, Any], interval: str
    ) -> dict[str, Any]:
        """Merge coverage metrics from different sources."""
        return {
            "interval": interval,
            "database": db_info,
            "data_lake": lake_info,
            "overall_coverage": min(
                1.0 if db_info.get("has_data", False) else 0.0,
                1.0 if lake_info.get("has_data", False) else 0.0,
            ),
            "total_records": db_info.get("record_count", 0),
            "total_files": lake_info.get("file_count", 0),
        }

    def _generate_coverage_summary(self, coverage_results: dict[str, Any]) -> dict[str, Any]:
        """Generate summary statistics from coverage results."""
        if not coverage_results:
            return {"avg_coverage_pct": 0.0, "symbols_analyzed": 0}

        total_coverage = 0.0
        symbol_count = len(coverage_results)

        for symbol_data in coverage_results.values():
            intervals = symbol_data.get("intervals", {})
            if intervals:
                symbol_coverage = sum(
                    interval_data.get("overall_coverage", 0.0)
                    for interval_data in intervals.values()
                ) / len(intervals)
                total_coverage += symbol_coverage

        avg_coverage = (total_coverage / symbol_count) if symbol_count > 0 else 0.0

        return {
            "avg_coverage_pct": avg_coverage * 100,
            "symbols_analyzed": symbol_count,
            "intervals_per_symbol": len(self.expected_intervals),
        }

    def _group_consecutive_dates(self, missing_dates: pd.DatetimeIndex) -> list[dict[str, Any]]:
        """Group consecutive missing dates into gap periods."""
        if len(missing_dates) == 0:
            return []

        gaps = []
        current_gap_start = missing_dates[0]
        current_gap_end = missing_dates[0]

        for i in range(1, len(missing_dates)):
            if missing_dates[i] - current_gap_end <= pd.Timedelta(days=1):
                # Consecutive date, extend current gap
                current_gap_end = missing_dates[i]
            else:
                # Non-consecutive, close current gap and start new one
                gaps.append(
                    {
                        "start": current_gap_start.isoformat(),
                        "end": current_gap_end.isoformat(),
                        "duration_hours": (current_gap_end - current_gap_start).total_seconds()
                        / 3600,
                    }
                )
                current_gap_start = missing_dates[i]
                current_gap_end = missing_dates[i]

        # Close the last gap
        gaps.append(
            {
                "start": current_gap_start.isoformat(),
                "end": current_gap_end.isoformat(),
                "duration_hours": (current_gap_end - current_gap_start).total_seconds() / 3600,
            }
        )

        return gaps
