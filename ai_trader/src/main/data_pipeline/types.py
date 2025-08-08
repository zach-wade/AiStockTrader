"""
Data Pipeline Types

Common types used across the data pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

# Re-export from archive for compatibility
from main.data_pipeline.storage.archive import RawDataRecord


class DataType(Enum):
    """Types of data in the pipeline."""
    MARKET_DATA = "market_data"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    CORPORATE_ACTIONS = "corporate_actions"
    SOCIAL_SENTIMENT = "social_sentiment"
    RATINGS = "ratings"


class TimeInterval(Enum):
    """Time intervals for data."""
    ONE_MINUTE = "1minute"
    FIVE_MINUTES = "5minute"
    FIFTEEN_MINUTES = "15minute"
    THIRTY_MINUTES = "30minute"
    ONE_HOUR = "1hour"
    ONE_DAY = "1day"
    ONE_WEEK = "1week"
    ONE_MONTH = "1month"


@dataclass
class BackfillParams:
    """Parameters for backfill operations."""
    symbols: List[str]
    data_types: List[DataType]
    start_date: datetime
    end_date: datetime
    intervals: Optional[List[TimeInterval]] = None
    force_refresh: bool = False
    use_bulk_loader: bool = True
    layer: int = 1
    user_requested_days: Optional[int] = None
    
    def __post_init__(self):
        """Ensure intervals has a default."""
        if self.intervals is None:
            self.intervals = [TimeInterval.ONE_DAY]


@dataclass
class GapInfo:
    """Information about a data gap."""
    symbol: str
    data_type: DataType
    interval: TimeInterval
    start_date: datetime
    end_date: datetime
    records_missing: int = 0
    priority: int = 1


__all__ = [
    'RawDataRecord',
    'DataType',
    'TimeInterval',
    'BackfillParams',
    'GapInfo'
]