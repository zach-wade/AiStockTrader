"""
Base Repository Interface

Core interfaces and types for all repository implementations.
"""

# Standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Third-party imports
import pandas as pd


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

    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_metrics: bool = True
    log_operations: bool = False
    validation_level: ValidationLevel = ValidationLevel.BASIC
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_dual_storage: bool = False
    transaction_strategy: TransactionStrategy = TransactionStrategy.BATCH
    max_parallel_workers: int = 4  # Added for parallel processing support
    connection_pool_size: int = 10  # Added for connection pooling


@dataclass
class QueryFilter:
    """Filter criteria for repository queries."""

    symbol: str | None = None
    symbols: list[str] | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    limit: int | None = None
    offset: int | None = None
    order_by: list[str] | None = None
    ascending: bool = True
    filters: dict[str, Any] = field(default_factory=dict)

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
        if self.limit:
            result["limit"] = self.limit
        if self.offset:
            result["offset"] = self.offset
        if self.order_by:
            result["order_by"] = self.order_by
        result["ascending"] = self.ascending
        result.update(self.filters)
        return result


@dataclass
class OperationResult:
    """Result of a repository operation."""

    success: bool
    data: Any | None = None
    error: str | None = None
    records_affected: int = 0
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class IRepository(ABC):
    """Base interface for all data repositories."""

    @abstractmethod
    async def get_by_id(self, record_id: Any) -> dict[str, Any] | None:
        """
        Get a single record by ID.

        Args:
            record_id: Primary key or unique identifier

        Returns:
            Record as dictionary or None if not found
        """
        pass

    @abstractmethod
    async def get_by_filter(self, query_filter: QueryFilter) -> pd.DataFrame:
        """
        Get records matching filter criteria.

        Args:
            query_filter: Filter criteria

        Returns:
            DataFrame with matching records
        """
        pass

    @abstractmethod
    async def create(self, data: dict[str, Any] | pd.DataFrame) -> OperationResult:
        """
        Create new records.

        Args:
            data: Data to create (single record or DataFrame)

        Returns:
            Operation result with created record IDs
        """
        pass

    @abstractmethod
    async def update(self, record_id: Any, data: dict[str, Any]) -> OperationResult:
        """
        Update an existing record.

        Args:
            record_id: Record identifier
            data: Fields to update

        Returns:
            Operation result
        """
        pass

    @abstractmethod
    async def upsert(self, data: dict[str, Any] | pd.DataFrame) -> OperationResult:
        """
        Insert or update records.

        Args:
            data: Data to upsert

        Returns:
            Operation result with upsert statistics
        """
        pass

    @abstractmethod
    async def delete(self, record_id: Any) -> OperationResult:
        """
        Delete a record.

        Args:
            record_id: Record identifier

        Returns:
            Operation result
        """
        pass

    @abstractmethod
    async def delete_by_filter(self, query_filter: QueryFilter) -> OperationResult:
        """
        Delete records matching filter criteria.

        Args:
            query_filter: Filter criteria

        Returns:
            Operation result with deletion count
        """
        pass

    @abstractmethod
    async def count(self, query_filter: QueryFilter | None = None) -> int:
        """
        Count records matching filter criteria.

        Args:
            query_filter: Optional filter criteria

        Returns:
            Record count
        """
        pass

    @abstractmethod
    async def exists(self, record_id: Any) -> bool:
        """
        Check if a record exists.

        Args:
            record_id: Record identifier

        Returns:
            True if record exists
        """
        pass


class IRepositoryFactory(ABC):
    """Factory interface for creating repository instances."""

    @abstractmethod
    def create_repository(
        self, repo_type: str, config: RepositoryConfig | None = None
    ) -> IRepository:
        """
        Create a repository instance.

        Args:
            repo_type: Type of repository to create
            config: Optional repository configuration

        Returns:
            Repository instance

        Raises:
            ValueError: If repo_type is unknown
        """
        pass

    @abstractmethod
    def get_available_repositories(self) -> list[str]:
        """
        Get list of available repository types.

        Returns:
            List of repository type names
        """
        pass

    @abstractmethod
    def register_repository(
        self, repo_type: str, repo_class: type[IRepository], override: bool = False
    ) -> None:
        """
        Register a new repository type.

        Args:
            repo_type: Type name for the repository
            repo_class: Repository class (must implement IRepository)
            override: Whether to override existing registration

        Raises:
            ValueError: If repo_type exists and override is False
        """
        pass


class IRepositoryProvider(ABC):
    """Provider interface to break circular dependencies."""

    @abstractmethod
    def get_repository(self, repo_type: str) -> IRepository:
        """
        Get a repository instance.

        Args:
            repo_type: Type of repository

        Returns:
            Repository instance

        Raises:
            ValueError: If repository type not found
        """
        pass

    @abstractmethod
    def register_repository_instance(self, repo_type: str, instance: IRepository) -> None:
        """
        Register a repository instance.

        Args:
            repo_type: Type name
            instance: Repository instance
        """
        pass
