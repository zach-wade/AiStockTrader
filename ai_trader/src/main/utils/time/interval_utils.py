"""
Time Interval Utilities

Provides utilities for working with time intervals, including conversions
between interval types and timedeltas, normalization, and validation.
"""

from datetime import timedelta
from typing import Optional, Dict
from enum import Enum


class TimeIntervalUtils:
    """Utilities for working with time intervals."""
    
    # Standard interval mappings
    INTERVAL_MAP: Dict[str, timedelta] = {
        # Minute intervals
        "1minute": timedelta(minutes=1),
        "1min": timedelta(minutes=1),
        "5minute": timedelta(minutes=5),
        "5min": timedelta(minutes=5),
        "15minute": timedelta(minutes=15),
        "15min": timedelta(minutes=15),
        "30minute": timedelta(minutes=30),
        "30min": timedelta(minutes=30),
        
        # Hour intervals
        "1hour": timedelta(hours=1),
        "60minute": timedelta(hours=1),
        "60min": timedelta(hours=1),
        
        # Day/Week/Month intervals
        "1day": timedelta(days=1),
        "daily": timedelta(days=1),
        "1week": timedelta(weeks=1),
        "weekly": timedelta(weeks=1),
        "1month": timedelta(days=30),  # Approximate
        "monthly": timedelta(days=30)
    }
    
    # Canonical interval names
    CANONICAL_INTERVALS = {
        "1minute": ["1min", "1minute"],
        "5minute": ["5min", "5minute"],
        "15minute": ["15min", "15minute"],
        "30minute": ["30min", "30minute"],
        "1hour": ["60min", "60minute", "1hour"],
        "1day": ["daily", "1day"],
        "1week": ["weekly", "1week"],
        "1month": ["monthly", "1month"]
    }
    
    @classmethod
    def get_interval_timedelta(cls, interval: str) -> timedelta:
        """
        Get timedelta for an interval string.
        
        Args:
            interval: Interval string (e.g., '1minute', '5min', '1hour')
            
        Returns:
            Corresponding timedelta
            
        Raises:
            ValueError: If interval is not recognized
        """
        normalized = cls.normalize_interval_string(interval)
        delta = cls.INTERVAL_MAP.get(normalized)
        
        if delta is None:
            raise ValueError(f"Unknown interval: {interval}")
        
        return delta
    
    @classmethod
    def get_interval_timedelta_from_enum(cls, interval_enum) -> timedelta:
        """
        Get timedelta for a TimeInterval enum.
        
        Args:
            interval_enum: TimeInterval enum value
            
        Returns:
            Corresponding timedelta
        """
        # Map enum values to interval strings
        enum_to_string = {
            "ONE_MINUTE": "1minute",
            "FIVE_MINUTES": "5minute",
            "FIFTEEN_MINUTES": "15minute",
            "THIRTY_MINUTES": "30minute",
            "ONE_HOUR": "1hour",
            "ONE_DAY": "1day",
            "ONE_WEEK": "1week",
            "ONE_MONTH": "1month"
        }
        
        # Handle both enum name and value
        if hasattr(interval_enum, 'name'):
            interval_str = enum_to_string.get(interval_enum.name, interval_enum.value)
        else:
            interval_str = str(interval_enum)
        
        return cls.get_interval_timedelta(interval_str)
    
    @classmethod
    def get_interval_seconds(cls, interval: str) -> int:
        """
        Get the number of seconds in an interval.
        
        Args:
            interval: Interval string
            
        Returns:
            Number of seconds
        """
        delta = cls.get_interval_timedelta(interval)
        return int(delta.total_seconds())
    
    @classmethod
    def normalize_interval_string(cls, interval: str) -> str:
        """
        Normalize an interval string to its canonical form.
        
        Args:
            interval: Interval string in any supported format
            
        Returns:
            Normalized interval string
        """
        interval_lower = interval.lower().strip()
        
        # Direct lookup first
        if interval_lower in cls.INTERVAL_MAP:
            # Find canonical form
            for canonical, variants in cls.CANONICAL_INTERVALS.items():
                if interval_lower in variants:
                    return variants[0]  # Return first variant as canonical
            return interval_lower
        
        # Handle special cases
        if interval_lower in ["1m", "1"]:
            return "1min"
        elif interval_lower in ["5m", "5"]:
            return "5min"
        elif interval_lower in ["15m", "15"]:
            return "15min"
        elif interval_lower in ["30m", "30"]:
            return "30min"
        elif interval_lower in ["60m", "60", "1h"]:
            return "1hour"
        elif interval_lower in ["1d", "d"]:
            return "1day"
        elif interval_lower in ["1w", "w"]:
            return "1week"
        elif interval_lower in ["1mo", "mo"]:
            return "1month"
        
        return interval_lower
    
    @classmethod
    def is_intraday_interval(cls, interval: str) -> bool:
        """
        Check if an interval is intraday (less than 1 day).
        
        Args:
            interval: Interval string
            
        Returns:
            True if intraday, False otherwise
        """
        delta = cls.get_interval_timedelta(interval)
        return delta < timedelta(days=1)
    
    @classmethod
    def is_daily_or_higher(cls, interval: str) -> bool:
        """
        Check if an interval is daily or higher frequency.
        
        Args:
            interval: Interval string
            
        Returns:
            True if daily or higher, False otherwise
        """
        delta = cls.get_interval_timedelta(interval)
        return delta >= timedelta(days=1)
    
    @classmethod
    def get_periods_in_day(cls, interval: str) -> int:
        """
        Calculate how many periods of the given interval fit in a trading day.
        
        Args:
            interval: Interval string
            
        Returns:
            Number of periods (assumes 6.5 hour trading day for intraday)
        """
        if cls.is_daily_or_higher(interval):
            return 1
        
        delta = cls.get_interval_timedelta(interval)
        trading_hours = 6.5  # Standard US market hours
        trading_seconds = trading_hours * 3600
        interval_seconds = delta.total_seconds()
        
        return int(trading_seconds / interval_seconds)
    
    @classmethod
    def validate_interval(cls, interval: str) -> bool:
        """
        Validate if an interval string is supported.
        
        Args:
            interval: Interval string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            cls.get_interval_timedelta(interval)
            return True
        except (ValueError, KeyError):
            return False


# Convenience functions for direct import
def get_interval_timedelta(interval: str) -> timedelta:
    """Get timedelta for an interval string."""
    return TimeIntervalUtils.get_interval_timedelta(interval)


def get_interval_seconds(interval: str) -> int:
    """Get seconds for an interval string."""
    return TimeIntervalUtils.get_interval_seconds(interval)


def normalize_interval(interval: str) -> str:
    """Normalize an interval string."""
    return TimeIntervalUtils.normalize_interval_string(interval)


def is_intraday(interval: str) -> bool:
    """Check if interval is intraday."""
    return TimeIntervalUtils.is_intraday_interval(interval)