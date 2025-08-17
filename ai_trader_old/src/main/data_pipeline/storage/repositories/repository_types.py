"""
Repository Type Definitions

Core types and configurations for the repository system.
"""

# Standard library imports
import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any


class ValidationLevel(Enum):
    """Validation levels for repository operations."""

    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


class TransactionStrategy(Enum):
    """Transaction handling strategies."""

    NONE = "none"
    SINGLE = "single"
    BATCH = "batch"
    SAVEPOINT = "savepoint"


@dataclass
class RepositoryConfig:
    """Configuration for repository behavior."""

    # Caching settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    cache_size_limit: int = 1000

    # Metrics settings
    enable_metrics: bool = True
    log_operations: bool = False
    metrics_sample_rate: float = 1.0

    # Validation settings
    validation_level: ValidationLevel = ValidationLevel.BASIC
    validate_on_read: bool = False
    validate_on_write: bool = True

    # Batch processing
    batch_size: int = 1000
    max_batch_size: int = 10000
    batch_timeout_seconds: float = 30.0

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0

    # Storage settings
    enable_dual_storage: bool = False
    hot_storage_days: int = 30
    prefer_hot_storage: bool = True

    # Transaction settings
    transaction_strategy: TransactionStrategy = TransactionStrategy.BATCH
    transaction_timeout_seconds: float = 60.0

    # Performance settings
    enable_query_optimization: bool = True
    enable_parallel_processing: bool = True
    max_parallel_workers: int = 4


@dataclass
class QueryFilter:
    """Filter criteria for repository queries."""

    # Symbol filters
    symbol: str | None = None
    symbols: list[str] | None = None

    # Date filters
    start_date: datetime | None = None
    end_date: datetime | None = None

    # Pagination
    limit: int | None = None
    offset: int | None = None

    # Ordering
    order_by: list[str] | None = None
    ascending: bool = True

    # Additional filters
    filters: dict[str, Any] = field(default_factory=dict)

    # Query hints
    use_hot_storage: bool | None = None
    bypass_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for query building."""
        result = {}

        if self.symbol:
            result["symbol"] = self.symbol
        if self.symbols:
            result["symbols"] = self.symbols
        if self.start_date:
            result["start_date"] = self.start_date
        if self.end_date:
            result["end_date"] = self.end_date
        if self.limit is not None:
            result["limit"] = self.limit
        if self.offset is not None:
            result["offset"] = self.offset
        if self.order_by:
            result["order_by"] = self.order_by

        result["ascending"] = self.ascending
        result.update(self.filters)

        return result

    def with_pagination(self, page: int, page_size: int) -> "QueryFilter":
        """Create a new filter with pagination."""
        new_filter = copy.deepcopy(self)
        new_filter.limit = page_size
        new_filter.offset = (page - 1) * page_size
        return new_filter


@dataclass
class OperationResult:
    """Result of a repository operation."""

    success: bool
    data: Any | None = None
    error: str | None = None
    records_affected: int = 0
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Additional tracking
    records_created: int = 0
    records_updated: int = 0
    records_deleted: int = 0
    records_skipped: int = 0

    @property
    def total_records(self) -> int:
        """Total records processed."""
        return self.records_created + self.records_updated + self.records_deleted

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "records_affected": self.records_affected,
            "records_created": self.records_created,
            "records_updated": self.records_updated,
            "records_deleted": self.records_deleted,
            "records_skipped": self.records_skipped,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class TimeRange:
    """Time range for queries and operations."""

    start: datetime
    end: datetime

    def __post_init__(self):
        """Ensure times are timezone-aware."""
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=UTC)
        if self.end.tzinfo is None:
            self.end = self.end.replace(tzinfo=UTC)

    @property
    def duration(self) -> timedelta:
        """Duration of the time range."""
        return self.end - self.start

    @property
    def days(self) -> int:
        """Number of days in range."""
        return self.duration.days

    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another."""
        return self.start < other.end and other.start < self.end

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within range."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return self.start <= timestamp <= self.end


def create_time_range(
    start: datetime | None = None, end: datetime | None = None, days_back: int | None = None
) -> TimeRange:
    """
    Create a TimeRange with convenient defaults.

    Args:
        start: Start datetime
        end: End datetime (defaults to now)
        days_back: Alternative to start, days before end

    Returns:
        TimeRange instance
    """
    if end is None:
        end = datetime.now(UTC)

    if start is None:
        if days_back is not None:
            start = end - timedelta(days=days_back)
        else:
            raise ValueError("Must provide either start or days_back")

    return TimeRange(start=start, end=end)
