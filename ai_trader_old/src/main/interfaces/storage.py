"""
Storage System Interfaces

Defines interfaces for the storage routing and execution system,
enabling clean separation of concerns and dependency injection.
"""

# Standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .repositories import IRepository


class StorageTier(Enum):
    """Storage tier options."""

    HOT = "hot"  # PostgreSQL - recent, frequently accessed data
    COLD = "cold"  # Data Lake - historical, archival data
    BOTH = "both"  # Query both tiers


class QueryType(Enum):
    """Types of queries for routing decisions."""

    REAL_TIME = "real_time"  # Real-time data access
    ANALYSIS = "analysis"  # Historical analysis
    FEATURE_CALC = "feature_calc"  # Feature calculation
    BULK_EXPORT = "bulk_export"  # Bulk data export
    ADMIN = "admin"  # Administrative queries


@dataclass
class QueryFilter:
    """Query filter parameters."""

    start_date: Any | None = None
    end_date: Any | None = None
    symbols: list[str] | None = None
    additional_filters: dict[str, Any] | None = None


@dataclass
class RoutingDecision:
    """Represents a routing decision made by IStorageRouter."""

    primary_tier: StorageTier
    fallback_tier: StorageTier | None = None
    reason: str = ""
    estimated_performance: str = ""  # "fast", "medium", "slow"
    cache_ttl_seconds: int | None = None


@runtime_checkable
class IStorageRouter(Protocol):
    """
    Interface for storage routing decisions.

    This component decides which storage tier to use based on
    query characteristics, but does NOT execute queries.
    """

    def route_query(
        self,
        query_filter: QueryFilter,
        query_type: QueryType = QueryType.ANALYSIS,
        prefer_performance: bool = False,
    ) -> RoutingDecision:
        """
        Determine the optimal storage tier for a query.

        Args:
            query_filter: Query parameters including time range
            query_type: Type of query being performed
            prefer_performance: Whether to prioritize speed over cost

        Returns:
            RoutingDecision with tier selection and metadata
        """
        ...

    def get_routing_stats(self) -> dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dictionary with routing metrics
        """
        ...


@runtime_checkable
class IStorageExecutor(Protocol):
    """
    Interface for executing queries against storage tiers.

    This component executes queries but does NOT make routing decisions.
    """

    async def execute_hot_query(
        self, repository_name: str, method_name: str, query_filter: QueryFilter, **kwargs
    ) -> Any:
        """
        Execute a query against hot storage (PostgreSQL).

        Args:
            repository_name: Name of the repository to use
            method_name: Method to call on the repository
            query_filter: Query parameters
            **kwargs: Additional arguments for the method

        Returns:
            Query results
        """
        ...

    async def execute_cold_query(
        self, repository_name: str, method_name: str, query_filter: QueryFilter, **kwargs
    ) -> Any:
        """
        Execute a query against cold storage (Data Lake).

        Args:
            repository_name: Name of the repository type
            method_name: Method to emulate in cold storage
            query_filter: Query parameters
            **kwargs: Additional arguments

        Returns:
            Query results (typically DataFrame)
        """
        ...

    async def execute_query(
        self,
        routing_decision: RoutingDecision,
        repository_name: str,
        method_name: str,
        query_filter: QueryFilter,
        **kwargs,
    ) -> Any:
        """
        Execute a query based on routing decision.

        Args:
            routing_decision: Pre-computed routing decision
            repository_name: Repository to use
            method_name: Method to call
            query_filter: Query parameters
            **kwargs: Additional arguments

        Returns:
            Query results with automatic fallback handling
        """
        ...


@runtime_checkable
class IRepositoryProvider(Protocol):
    """
    Interface for providing repository instances.

    This allows components to get repositories without
    depending on the concrete RepositoryFactory.
    """

    def get_repository(self, repository_name: str) -> IRepository:
        """
        Get a repository instance by name.

        Args:
            repository_name: Name of the repository

        Returns:
            Repository instance

        Raises:
            ValueError: If repository not found
        """
        ...

    def has_repository(self, repository_name: str) -> bool:
        """
        Check if a repository is available.

        Args:
            repository_name: Name to check

        Returns:
            True if repository exists
        """
        ...

    def list_repositories(self) -> list[str]:
        """
        Get list of available repository names.

        Returns:
            List of repository names
        """
        ...


class IStorageSystem(ABC):
    """
    Complete storage system interface combining routing and execution.

    This is for components that need the full storage system functionality.
    """

    @abstractmethod
    def get_router(self) -> IStorageRouter:
        """Get the storage router."""
        pass

    @abstractmethod
    def get_executor(self) -> IStorageExecutor:
        """Get the storage executor."""
        pass

    @abstractmethod
    async def query(
        self,
        repository_name: str,
        method_name: str,
        query_filter: QueryFilter,
        query_type: QueryType = QueryType.ANALYSIS,
        **kwargs,
    ) -> Any:
        """
        High-level query interface that combines routing and execution.

        Args:
            repository_name: Repository to query
            method_name: Method to call
            query_filter: Query parameters
            query_type: Type of query
            **kwargs: Additional arguments

        Returns:
            Query results
        """
        pass
