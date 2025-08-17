"""
Storage Router

Intelligent routing between hot (PostgreSQL) and cold (Data Archive) storage
based on query characteristics and data age.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.data_pipeline.storage.archive import get_archive
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.interfaces.storage import (
    IStorageRouter,
    QueryFilter,
    QueryType,
    RoutingDecision,
    StorageTier,
)
from main.utils.core import ensure_utc, get_logger

logger = get_logger(__name__)


class StorageRouter(IStorageRouter):
    """
    Routes queries between hot and cold storage tiers based on data characteristics.

    Hot Storage (PostgreSQL):
    - Recent data (< 30 days by default)
    - Frequently accessed data
    - Real-time queries
    - Small result sets

    Cold Storage (Data Archive):
    - Historical data (> 30 days)
    - Bulk exports
    - Analysis queries
    - Large result sets
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the storage router.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logger

        # Configure thresholds
        self.hot_storage_days = self.config.get("hot_storage_days", 30)
        self.cache_enabled = self.config.get("cache_enabled", True)

        # Initialize storage backends
        self._initialize_storage_backends()

        # Track routing statistics
        self.routing_stats = {
            "hot_queries": 0,
            "cold_queries": 0,
            "both_queries": 0,
            "cache_hits": 0,
            "fallback_count": 0,
        }

        self.logger.info(f"StorageRouter initialized (hot_storage_days={self.hot_storage_days})")

    def _initialize_storage_backends(self):
        """Initialize hot and cold storage backends."""
        # Initialize hot storage (PostgreSQL)
        db_factory = DatabaseFactory()
        db_adapter = db_factory.create_async_database(self.config)
        # Local imports
        from main.data_pipeline.storage.repositories import get_repository_factory

        repo_factory = get_repository_factory()
        self.market_data_repo = repo_factory.create_market_data_repository(db_adapter)

        # Initialize cold storage (Archive)
        self.archive = get_archive()

        self.logger.info("Storage backends initialized successfully")

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
        # Calculate data age
        end_date = query_filter.end_date or datetime.now(UTC)
        start_date = query_filter.start_date

        if start_date:
            data_age_days = (datetime.now(UTC) - ensure_utc(start_date)).days
            data_range_days = (ensure_utc(end_date) - ensure_utc(start_date)).days
        else:
            data_age_days = 0
            data_range_days = 0

        # Routing logic based on query characteristics
        if query_type == QueryType.REAL_TIME:
            # Real-time queries always use hot storage
            return RoutingDecision(
                primary_tier=StorageTier.HOT,
                reason="Real-time queries require hot storage",
                estimated_performance="fast",
                cache_ttl_seconds=60 if self.cache_enabled else None,
            )

        elif data_age_days <= self.hot_storage_days:
            # Recent data - use hot storage
            return RoutingDecision(
                primary_tier=StorageTier.HOT,
                fallback_tier=StorageTier.COLD,
                reason=f"Data within {self.hot_storage_days} days - using hot storage",
                estimated_performance="fast",
                cache_ttl_seconds=300 if self.cache_enabled else None,
            )

        elif query_type == QueryType.BULK_EXPORT or data_range_days > 365:
            # Large historical queries - use cold storage
            return RoutingDecision(
                primary_tier=StorageTier.COLD,
                reason="Large historical query - using cold storage",
                estimated_performance="slow",
                cache_ttl_seconds=3600 if self.cache_enabled else None,
            )

        elif query_type == QueryType.FEATURE_CALC:
            # Feature calculation - check both tiers
            if data_age_days <= self.hot_storage_days * 2:
                # Somewhat recent - try hot first
                return RoutingDecision(
                    primary_tier=StorageTier.HOT,
                    fallback_tier=StorageTier.COLD,
                    reason="Feature calculation on semi-recent data",
                    estimated_performance="medium",
                    cache_ttl_seconds=1800 if self.cache_enabled else None,
                )
            else:
                # Historical - use cold
                return RoutingDecision(
                    primary_tier=StorageTier.COLD,
                    reason="Feature calculation on historical data",
                    estimated_performance="slow",
                    cache_ttl_seconds=3600 if self.cache_enabled else None,
                )

        else:
            # Default: route based on data age
            if data_age_days <= self.hot_storage_days:
                tier = StorageTier.HOT
                performance = "fast"
            else:
                tier = StorageTier.COLD
                performance = "medium"

            return RoutingDecision(
                primary_tier=tier,
                fallback_tier=StorageTier.COLD if tier == StorageTier.HOT else None,
                reason=f"Default routing based on data age ({data_age_days} days)",
                estimated_performance=performance,
                cache_ttl_seconds=600 if self.cache_enabled else None,
            )

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
        self.routing_stats["hot_queries"] += 1

        try:
            # For now, we only support market_data repository
            if repository_name != "market_data":
                raise ValueError(f"Unsupported repository: {repository_name}")

            # Build query parameters
            params = {
                "symbols": query_filter.symbols,
                "start_date": query_filter.start_date,
                "end_date": query_filter.end_date,
            }
            params.update(kwargs)

            # Execute query based on method name
            if method_name == "get_by_filter":
                result = await self.market_data_repo.get_with_filters(**params)
                return result.data if result.success else []
            elif method_name == "get_latest":
                result = await self.market_data_repo.get_latest(
                    symbol=query_filter.symbols[0] if query_filter.symbols else None
                )
                return result
            else:
                # Generic method call
                method = getattr(self.market_data_repo, method_name)
                return await method(**params)

        except Exception as e:
            self.logger.error(f"Hot storage query failed: {e}")
            raise

    async def execute_cold_query(
        self, repository_name: str, method_name: str, query_filter: QueryFilter, **kwargs
    ) -> Any:
        """
        Execute a query against cold storage (Data Archive).

        Args:
            repository_name: Name of the repository type
            method_name: Method to emulate in cold storage
            query_filter: Query parameters
            **kwargs: Additional arguments

        Returns:
            Query results (typically DataFrame)
        """
        self.routing_stats["cold_queries"] += 1

        try:
            # Map repository name to data type
            data_type_map = {
                "market_data": "market_data",
                "news": "news",
                "fundamentals": "fundamentals",
                "corporate_actions": "corporate_actions",
            }

            data_type = data_type_map.get(repository_name, "market_data")

            # Query archive for raw records
            records = await self.archive.query_raw_records(
                source=kwargs.get("source", "polygon"),
                data_type=data_type,
                symbol=query_filter.symbols[0] if query_filter.symbols else None,
                start_date=query_filter.start_date,
                end_date=query_filter.end_date,
            )

            # Convert records to appropriate format
            if method_name == "get_by_filter" or method_name == "get_latest":
                # Extract data from records and combine
                all_data = []
                for record in records:
                    if record.data:
                        data = record.data.get(data_type, record.data)
                        if isinstance(data, list):
                            all_data.extend(data)
                        elif isinstance(data, dict) or isinstance(data, pd.DataFrame):
                            all_data.append(data)

                # Return as DataFrame if we have data
                if all_data:
                    if isinstance(all_data[0], pd.DataFrame):
                        return pd.concat(all_data, ignore_index=True)
                    else:
                        return pd.DataFrame(all_data)
                else:
                    return pd.DataFrame()
            else:
                # Return raw records for other methods
                return records

        except Exception as e:
            self.logger.error(f"Cold storage query failed: {e}")
            raise

    async def execute_query(
        self,
        repository_name: str,
        method_name: str,
        query_filter: QueryFilter,
        query_type: QueryType = QueryType.ANALYSIS,
        prefer_performance: bool = False,
        **kwargs,
    ) -> Any:
        """
        Execute a query with automatic routing and fallback.

        This is the main entry point that combines routing decision with execution.

        Args:
            repository_name: Repository to query
            method_name: Method to call
            query_filter: Query parameters
            query_type: Type of query
            prefer_performance: Whether to prioritize speed
            **kwargs: Additional arguments

        Returns:
            Query results with automatic fallback handling
        """
        # Get routing decision
        routing_decision = self.route_query(query_filter, query_type, prefer_performance)

        self.logger.debug(
            f"Routing query to {routing_decision.primary_tier.value} storage: "
            f"{routing_decision.reason}"
        )

        # Try primary tier
        try:
            if routing_decision.primary_tier == StorageTier.HOT:
                result = await self.execute_hot_query(
                    repository_name, method_name, query_filter, **kwargs
                )
            elif routing_decision.primary_tier == StorageTier.COLD:
                result = await self.execute_cold_query(
                    repository_name, method_name, query_filter, **kwargs
                )
            else:  # BOTH
                # Execute on both tiers and combine
                self.routing_stats["both_queries"] += 1
                hot_task = self.execute_hot_query(
                    repository_name, method_name, query_filter, **kwargs
                )
                cold_task = self.execute_cold_query(
                    repository_name, method_name, query_filter, **kwargs
                )
                hot_result, cold_result = await asyncio.gather(
                    hot_task, cold_task, return_exceptions=True
                )

                # Combine results
                results = []
                if not isinstance(hot_result, Exception):
                    results.append(hot_result)
                if not isinstance(cold_result, Exception):
                    results.append(cold_result)

                if results:
                    # Combine DataFrames or lists
                    if all(isinstance(r, pd.DataFrame) for r in results):
                        result = pd.concat(results, ignore_index=True)
                    elif all(isinstance(r, list) for r in results):
                        result = sum(results, [])
                    else:
                        result = results[0]  # Use first valid result
                else:
                    result = pd.DataFrame()  # Empty result

            return result

        except Exception as e:
            # Try fallback tier if available
            if routing_decision.fallback_tier:
                self.routing_stats["fallback_count"] += 1
                self.logger.warning(
                    f"Primary tier failed ({e}), trying fallback tier: "
                    f"{routing_decision.fallback_tier.value}"
                )

                try:
                    if routing_decision.fallback_tier == StorageTier.HOT:
                        return await self.execute_hot_query(
                            repository_name, method_name, query_filter, **kwargs
                        )
                    else:
                        return await self.execute_cold_query(
                            repository_name, method_name, query_filter, **kwargs
                        )
                except Exception as fallback_error:
                    self.logger.error(f"Fallback tier also failed: {fallback_error}")
                    raise
            else:
                raise

    def get_routing_stats(self) -> dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dictionary with routing metrics
        """
        total_queries = (
            self.routing_stats["hot_queries"]
            + self.routing_stats["cold_queries"]
            + self.routing_stats["both_queries"]
        )

        return {
            "total_queries": total_queries,
            "hot_queries": self.routing_stats["hot_queries"],
            "cold_queries": self.routing_stats["cold_queries"],
            "both_queries": self.routing_stats["both_queries"],
            "cache_hits": self.routing_stats["cache_hits"],
            "fallback_count": self.routing_stats["fallback_count"],
            "hot_percentage": (
                self.routing_stats["hot_queries"] / total_queries * 100 if total_queries > 0 else 0
            ),
            "fallback_rate": (
                self.routing_stats["fallback_count"] / total_queries * 100
                if total_queries > 0
                else 0
            ),
        }


# Export QueryType for convenience
__all__ = ["StorageRouter", "QueryType"]
