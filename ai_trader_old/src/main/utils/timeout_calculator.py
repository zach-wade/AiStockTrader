"""
Dynamic timeout calculator for API requests based on expected data volume.

This module provides intelligent timeout calculation based on:
- Time interval (1min, 5min, 1hour, 1day)
- Date range (days of data)
- Symbol characteristics (high/low volume)
"""

# Standard library imports
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TimeoutCalculator:
    """Calculate dynamic timeouts based on expected data volume."""

    # Base timeout per interval type (seconds per year of data)
    TIMEOUT_PER_YEAR = {
        "1min": 60,  # 1 minute per year for minute data
        "1minute": 60,  # Alias
        "5min": 20,  # 20 seconds per year for 5-minute data
        "5minute": 20,  # Alias
        "15min": 10,  # 10 seconds per year for 15-minute data
        "15minute": 10,  # Alias
        "30min": 8,  # 8 seconds per year for 30-minute data
        "30minute": 8,  # Alias
        "1hour": 5,  # 5 seconds per year for hourly data
        "60min": 5,  # Alias
        "1day": 2,  # 2 seconds per year for daily data
        "day": 2,  # Alias
        "1week": 1,  # 1 second per year for weekly data
        "week": 1,  # Alias
    }

    # Minimum and maximum timeouts
    MIN_TIMEOUT = 30  # Never less than 30 seconds
    MAX_TIMEOUT = 600  # Never more than 10 minutes

    # High-volume symbols that need extra time
    HIGH_VOLUME_SYMBOLS = {"AAPL", "TSLA", "SPY", "QQQ", "NVDA", "AMZN", "MSFT", "META", "GOOGL"}
    HIGH_VOLUME_MULTIPLIER = 1.5

    @classmethod
    def calculate_timeout(
        cls,
        interval: str,
        start_date: datetime | str,
        end_date: datetime | str,
        symbol: str | None = None,
        base_timeout: int = 30,
    ) -> int:
        """
        Calculate dynamic timeout based on data volume.

        Args:
            interval: Time interval (e.g., '1min', '5min', '1hour', '1day' or TimeInterval enum)
            start_date: Start date for data range
            end_date: End date for data range
            symbol: Optional symbol to check for high-volume adjustments
            base_timeout: Base timeout to use as minimum

        Returns:
            Calculated timeout in seconds
        """
        try:
            # Convert string dates to datetime if needed
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

            # Calculate date range in years
            date_range_days = (end_date - start_date).days
            date_range_years = date_range_days / 365.25

            # Handle TimeInterval enum objects
            if hasattr(interval, "value"):
                interval = interval.value

            # Get timeout per year for this interval
            interval_lower = interval.lower()
            timeout_per_year = cls.TIMEOUT_PER_YEAR.get(interval_lower, 10)

            # Calculate base timeout
            calculated_timeout = int(base_timeout + (timeout_per_year * date_range_years))

            # Apply high-volume multiplier if needed
            if symbol and symbol.upper() in cls.HIGH_VOLUME_SYMBOLS:
                calculated_timeout = int(calculated_timeout * cls.HIGH_VOLUME_MULTIPLIER)
                logger.debug(f"Applied high-volume multiplier for {symbol}")

            # Apply bounds
            final_timeout = max(cls.MIN_TIMEOUT, min(calculated_timeout, cls.MAX_TIMEOUT))

            logger.debug(
                f"Calculated timeout for {symbol}/{interval} "
                f"({date_range_days} days): {final_timeout}s"
            )

            return final_timeout

        except Exception as e:
            logger.warning(f"Error calculating dynamic timeout: {e}. Using base timeout.")
            return max(base_timeout, cls.MIN_TIMEOUT)

    @classmethod
    def calculate_chunked_timeout(
        cls, interval: str, chunk_size_days: int, symbol: str | None = None
    ) -> int:
        """
        Calculate timeout for a single chunk of data.

        Args:
            interval: Time interval (string or TimeInterval enum)
            chunk_size_days: Size of chunk in days
            symbol: Optional symbol for high-volume adjustments

        Returns:
            Timeout in seconds for the chunk
        """
        # Create dummy dates for calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=chunk_size_days)

        return cls.calculate_timeout(interval, start_date, end_date, symbol)

    @classmethod
    def get_recommended_chunk_size(cls, interval: str) -> int:
        """
        Get recommended chunk size in days for an interval.

        Args:
            interval: Time interval (string or TimeInterval enum)

        Returns:
            Recommended chunk size in days
        """
        # Handle TimeInterval enum objects
        if hasattr(interval, "value"):
            interval = interval.value

        interval_lower = interval.lower()

        # Chunk sizes optimized for ~30-60 second requests
        chunk_sizes = {
            "1min": 30,  # 1 month chunks for minute data
            "1minute": 30,
            "5min": 90,  # 3 month chunks for 5-minute data
            "5minute": 90,
            "15min": 180,  # 6 month chunks for 15-minute data
            "15minute": 180,
            "30min": 365,  # 1 year chunks for 30-minute data
            "30minute": 365,
            "1hour": 730,  # 2 year chunks for hourly data
            "60min": 730,
            "1day": 1825,  # 5 year chunks for daily data
            "day": 1825,
            "1week": 3650,  # 10 year chunks for weekly data
            "week": 3650,
        }

        return chunk_sizes.get(interval_lower, 365)  # Default to 1 year
