"""Utility functions specific to historical data management."""

# Standard library imports
from datetime import UTC, datetime, timedelta  # FIXED: Added timezone import
from typing import Any

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.utils.core import get_market_hours, get_trading_days_between

# Note: DataSource and TimeInterval are not needed for this implementation


# Define constants inline to avoid circular imports
class TimeConstants:
    MAX_LOOKBACK_YEARS = 5
    INTERVAL_GAPS = {
        "1minute": timedelta(minutes=1),
        "5minute": timedelta(minutes=5),
        "15minute": timedelta(minutes=15),
        "30minute": timedelta(minutes=30),
        "1hour": timedelta(hours=1),
        "1day": timedelta(days=1),
        "1week": timedelta(weeks=1),
        "1month": timedelta(days=30),
    }


class ProcessingUtils:
    """Utilities for processing configuration and strategy"""

    @staticmethod
    def needs_sequential_processing(sources: list[str]) -> bool:
        """
        Determine if sources need sequential processing.

        Some data sources have strict rate limits that require sequential processing
        to avoid hitting API limits.

        Args:
            sources: List of data sources

        Returns:
            True if sequential processing is required
        """
        # Don't force sequential processing - let rate limiting handle it
        # This allows for parallel processing with proper rate limiting
        return False

    @staticmethod
    def calculate_optimal_parallel(sources: list[str], config: DictConfig | None = None) -> int:
        """
        Calculate optimal parallel processing based on source rate limits.

        Args:
            sources: List of data sources
            config: Optional configuration object

        Returns:
            Optimal number of parallel workers
        """
        # Get default from config if available
        default_parallel = 10  # Increased from 5
        if config:
            default_parallel = config.get("data.backfill.max_parallel", 10)

        # Source-specific limits - adjusted for connection pool constraints
        if "polygon" in sources:
            # Polygon has limited connection pools, be conservative
            # Connection pool warnings occur when parallel > num_pools/2
            return min(3, default_parallel)  # Conservative limit to avoid pool exhaustion
        elif "alpaca" in sources:
            return min(5, default_parallel)  # Alpaca handles parallel well
        elif "yahoo" in sources:
            return min(20, default_parallel)  # Yahoo can handle many parallel requests
        else:
            return min(10, default_parallel)  # Default to 10 parallel workers

    @staticmethod
    def get_source_priority(sources: list[str]) -> list[str]:
        """
        Order sources by data quality and reliability.

        Args:
            sources: List of data sources

        Returns:
            Sources ordered by priority
        """
        priority_order = {
            "polygon": 1,  # Most reliable, professional data
            "alpaca": 2,  # Good quality, free tier available
            "yahoo": 3,  # Good free alternative, some limitations
        }

        # Sort by priority, unknown sources go last
        return sorted(sources, key=lambda x: priority_order.get(x, 99))

    @staticmethod
    def should_retry_source(source: str, error: Exception) -> bool:
        """
        Determine if we should retry a failed source.

        Args:
            source: Data source name
            error: Exception that occurred

        Returns:
            True if retry is recommended
        """
        # Don't retry on authentication errors
        auth_errors = ["401", "forbidden", "unauthorized", "invalid api key"]
        error_str = str(error).lower()

        if any(auth_error in error_str for auth_error in auth_errors):
            return False

        # Don't retry on invalid symbol errors
        if "invalid symbol" in error_str or "symbol not found" in error_str:
            return False

        # Retry on temporary errors
        temp_errors = ["timeout", "connection", "temporary", "503", "502", "429"]
        return any(temp_error in error_str for temp_error in temp_errors)


class DataCalculationUtils:
    """Utilities for data-related calculations"""

    @staticmethod
    def calculate_expected_data_points(
        start: datetime, end: datetime, interval: str, market_hours_only: bool = True
    ) -> int:
        """
        Calculate expected number of data points for validation.

        Args:
            start: Start datetime
            end: End datetime
            interval: Time interval
            market_hours_only: Whether to consider only market hours

        Returns:
            Expected number of data points
        """
        if interval == "1day":  # FIXED: Use string literals
            return len(get_trading_days_between(start, end))

        elif interval == "1week":
            # Approximate - one point per week that contains trading days
            weeks = int((end - start).days / 7)
            return max(1, weeks)

        elif interval == "1month":
            # Approximate - one point per month
            months = (end.year - start.year) * 12 + (end.month - start.month)
            return max(1, months)

        elif DataCalculationUtils._is_intraday_interval(interval):
            if not market_hours_only:
                # 24/7 data (like crypto)
                total_minutes = (end - start).total_seconds() / 60
                interval_minutes = TimeConstants.INTERVAL_GAPS[interval].total_seconds() / 60
                return int(total_minutes / interval_minutes)
            else:
                # Market hours only
                return DataCalculationUtils._calculate_market_hours_points(start, end, interval)

        return 0

    @staticmethod
    def _is_intraday_interval(interval: str) -> bool:
        """Check if interval is intraday (less than 1 day)."""
        intraday_intervals = ["1minute", "5minute", "15minute", "30minute", "1hour"]
        return interval in intraday_intervals

    @staticmethod
    def _calculate_market_hours_points(start: datetime, end: datetime, interval: str) -> int:
        """Calculate data points during market hours only"""
        total_points = 0
        interval_minutes = TimeConstants.INTERVAL_GAPS[interval].total_seconds() / 60

        for day in get_trading_days_between(start, end):
            market_open, market_close = get_market_hours(day)

            # Handle None values from get_market_hours - FIXED
            if market_open is None or market_close is None:
                continue

            # Adjust for the specific day's boundaries
            day_start = max(start, market_open)
            day_end = min(end, market_close)

            if day_start < day_end:
                market_minutes = (day_end - day_start).total_seconds() / 60
                total_points += int(market_minutes / interval_minutes)

        return total_points

    @staticmethod
    def estimate_download_size(
        symbols: list[str], intervals: list[str], days: int
    ) -> dict[str, Any]:
        """
        Estimate the download size and time for a backfill operation.

        Args:
            symbols: List of symbols
            intervals: List of intervals
            days: Number of days to download

        Returns:
            Dictionary with estimates
        """
        # Rough estimates based on experience
        bytes_per_record = {
            "1minute": 100,
            "5minute": 100,
            "15minute": 100,
            "30minute": 100,
            "1hour": 100,
            "1day": 120,
            "1week": 120,
            "1month": 120,
        }

        # Points per day estimates (during market hours)
        points_per_day = {
            "1minute": 390,  # 6.5 hours * 60
            "5minute": 78,  # 6.5 hours * 12
            "15minute": 26,  # 6.5 hours * 4
            "30minute": 13,  # 6.5 hours * 2
            "1hour": 7,  # ~7 hourly bars
            "1day": 1,
            "1week": 0.2,  # ~1 per 5 days
            "1month": 0.05,  # ~1 per 20 days
        }

        total_records = 0
        total_bytes = 0

        for symbol in symbols:
            for interval in intervals:
                records = int(days * points_per_day.get(interval, 1))
                total_records += records
                total_bytes += records * bytes_per_record.get(interval, 100)

        # Convert to readable formats
        mb = total_bytes / (1024 * 1024)

        # Estimate download time (assume 1MB/s average speed)
        estimated_seconds = mb  # 1 second per MB

        return {
            "total_symbols": len(symbols),
            "total_intervals": len(intervals),
            "total_days": days,
            "estimated_records": total_records,
            "estimated_size_mb": round(mb, 2),
            "estimated_time_seconds": round(estimated_seconds, 2),
            "estimated_time_minutes": round(estimated_seconds / 60, 2),
            "warnings": DataCalculationUtils._get_estimate_warnings(
                total_records, mb, estimated_seconds
            ),
        }

    @staticmethod
    def _get_estimate_warnings(records: int, size_mb: float, time_seconds: float) -> list[str]:
        """Generate warnings based on estimates"""
        warnings = []

        if records > 1_000_000:
            warnings.append(f"Large dataset: {records:,} records expected")

        if size_mb > 100:
            warnings.append(f"Large download: {size_mb:.1f} MB expected")

        if time_seconds > 600:  # 10 minutes
            warnings.append(f"Long operation: {time_seconds/60:.1f} minutes expected")

        return warnings


class ValidationUtils:
    """Utilities for data validation"""

    @staticmethod
    def validate_symbol_format(symbol: str) -> tuple[bool, str | None]:
        """
        Validate stock symbol format.

        Args:
            symbol: Stock symbol to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not symbol:
            return False, "Symbol cannot be empty"

        # Check length
        if len(symbol) > 10:
            return False, "Symbol too long (max 10 characters)"

        # Check characters (alphanumeric plus some special chars)
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
        if not all(c in valid_chars for c in symbol.upper()):
            return False, "Symbol contains invalid characters"

        # Check for common invalid patterns
        if symbol.startswith(".") or symbol.endswith("."):
            return False, "Symbol cannot start or end with period"

        return True, None

    @staticmethod
    def validate_date_range(
        start: datetime, end: datetime, max_days: int | None = None
    ) -> tuple[bool, str | None]:
        """
        Validate date range for backfill.

        Args:
            start: Start date
            end: End date
            max_days: Optional maximum days allowed

        Returns:
            Tuple of (is_valid, error_message)
        """
        if start >= end:
            return False, "Start date must be before end date"

        # Check if dates are in the future
        now = datetime.now(start.tzinfo or UTC)
        if start > now:
            return False, "Start date cannot be in the future"

        # Check range size
        days = (end - start).days
        if max_days and days > max_days:
            return False, f"Date range too large ({days} days, max {max_days})"

        # Check if too old
        max_history = now - timedelta(days=365 * TimeConstants.MAX_LOOKBACK_YEARS)
        if start < max_history:
            return False, f"Start date too old (max {TimeConstants.MAX_LOOKBACK_YEARS} years)"

        return True, None

    @staticmethod
    def suggest_optimal_intervals(symbol_type: str = "stock") -> list[str]:
        """
        Suggest optimal intervals based on symbol type.

        Args:
            symbol_type: Type of symbol ('stock', 'crypto', 'forex')

        Returns:
            List of recommended intervals
        """
        if symbol_type == "crypto":
            # Crypto trades 24/7, all intervals useful
            return ["1minute", "5minute", "15minute", "1hour", "1day"]
        elif symbol_type == "forex":
            # Forex has different sessions
            return ["5minute", "15minute", "1hour", "1day"]
        else:  # stocks
            # For stocks, minute data might be overkill for most users
            return ["5minute", "15minute", "1hour", "1day"]


class DebugUtils:
    """Utilities for debugging and diagnostics"""

    @staticmethod
    def format_gap_summary(gaps: list[tuple[datetime, datetime]], max_gaps: int = 10) -> str:
        """
        Format gap list for readable logging.

        Args:
            gaps: List of gap tuples
            max_gaps: Maximum gaps to show

        Returns:
            Formatted string
        """
        if not gaps:
            return "No gaps found"

        lines = [f"Found {len(gaps)} gaps:"]

        for i, (start, end) in enumerate(gaps[:max_gaps]):
            days = (end - start).days
            lines.append(
                f"  Gap {i+1}: {start.strftime('%Y-%m-%d')} to "
                f"{end.strftime('%Y-%m-%d')} ({days} days)"
            )

        if len(gaps) > max_gaps:
            lines.append(f"  ... and {len(gaps) - max_gaps} more gaps")

        # Summary statistics
        total_days = sum((end - start).days for start, end in gaps)
        avg_days = total_days / len(gaps) if gaps else 0

        lines.append(f"\nTotal gap days: {total_days}")
        lines.append(f"Average gap size: {avg_days:.1f} days")

        return "\n".join(lines)

    @staticmethod
    def generate_test_params(quick: bool = True) -> dict[str, Any]:
        """
        Generate test parameters for debugging.

        Args:
            quick: If True, generate minimal test params

        Returns:
            Dictionary of test parameters
        """
        if quick:
            return {
                "symbols": ["AAPL", "MSFT"],
                "sources": ["yahoo"],
                "intervals": ["1day"],
                "start_date": datetime.now(UTC) - timedelta(days=7),
                "end_date": datetime.now(UTC),
                "force_full": False,
            }
        else:
            return {
                "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"],
                "sources": ["yahoo", "polygon"],
                "intervals": ["15minute", "1hour", "1day"],
                "start_date": datetime.now(UTC) - timedelta(days=30),
                "end_date": datetime.now(UTC),
                "force_full": False,
            }
