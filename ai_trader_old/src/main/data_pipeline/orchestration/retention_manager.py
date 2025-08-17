"""
Retention Manager

Manages data retention policies based on the layer system, including
hot/cold storage transitions and data archival according to layer-specific
retention rules.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from typing import Any

# Local imports
from main.config import get_config_manager
from main.data_pipeline.core.enums import DataLayer, DataType
from main.data_pipeline.core.exceptions import StorageError
from main.interfaces.data_pipeline.orchestration import IRetentionManager
from main.interfaces.database import IAsyncDatabase
from main.utils.core import (
    AsyncCircuitBreaker,
    ensure_utc,
    get_logger,
    get_trading_days_between,
    process_in_batches,
)
from main.utils.data import chunk_list


class RetentionManager(IRetentionManager):
    """
    Manages data retention policies with layer-based hot/cold storage.

    Retention Policies by Layer:
    - Layer 0: Hot 7 days → Cold 30 days → Archive
    - Layer 1: Hot 30 days → Cold 365 days → Archive
    - Layer 2: Hot 60 days → Cold 730 days → Archive
    - Layer 3: Hot 90 days → Cold 1825 days → Archive
    """

    def __init__(self, db_adapter: IAsyncDatabase, config: dict[str, Any] | None = None):
        self.db_adapter = db_adapter
        self.config = config or {}
        self.logger = get_logger(__name__)

        # Initialize config manager
        self.config_manager = get_config_manager()

        # Get storage configuration - use defaults/storage.yaml
        try:
            storage_config = self.config_manager.load_config("defaults/storage")
        except (FileNotFoundError, KeyError, Exception) as e:
            # Fallback to empty config if not found
            logger.debug(f"Could not load storage config, using defaults: {e}")
            storage_config = {}

        # Storage configuration
        hot_config = storage_config.get("hot", {})
        cold_config = storage_config.get("cold", {})
        archive_config = storage_config.get("archive", {})

        self.hot_storage_table_prefix = self.config.get("hot_storage_table_prefix", "market_data")
        self.cold_storage_path = cold_config.get("base_path", "data_lake")
        self.archive_storage_path = archive_config.get("local_path", "./data_lake/archive")

        # Processing configuration
        optimization_config = storage_config.get("optimization", {}).get("bulk_operations", {})
        self.batch_size = optimization_config.get("accumulation_size", 10000)
        self.max_concurrent_operations = optimization_config.get("parallel_loaders", 3)

        # Circuit breaker for archive operations
        self.archive_circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=5, recovery_timeout=60, expected_exception=StorageError
        )

        self.logger.info("RetentionManager initialized with unified configuration")

    async def get_retention_policy(self, layer: DataLayer, data_type: DataType) -> dict[str, Any]:
        """Get retention policy for layer and data type."""
        try:
            # Get layer configuration
            # Local imports
            from main.utils.layer_utils import get_layer_config

            layer_config = get_layer_config(layer, self.config_manager)
            retention_config = layer_config.get("retention", {})

            base_policy = {
                "layer": layer.value,
                "data_type": data_type.value,
                "hot_storage_days": retention_config.get(
                    "hot_storage_days", layer.hot_storage_days
                ),
                "total_retention_days": retention_config.get(
                    "total_retention_days", layer.retention_days
                ),
                "cold_storage_days": retention_config.get(
                    "total_retention_days", layer.retention_days
                )
                - retention_config.get("hot_storage_days", layer.hot_storage_days),
                "retention_policy": retention_config.get("retention_policy", "hot_cold"),
                "archive_after_days": retention_config.get(
                    "total_retention_days", layer.retention_days
                ),
                "cleanup_after_days": self._get_cleanup_days(layer, data_type),
            }

            # Data type specific adjustments with trading days consideration
            if data_type == DataType.NEWS:
                # News data has different retention patterns
                base_policy["hot_storage_days"] = min(layer.hot_storage_days, 14)  # Max 2 weeks hot
                base_policy["total_retention_days"] = max(
                    layer.retention_days, 90
                )  # Min 3 months total
                # Calculate actual trading days
                end_date = ensure_utc(datetime.now(UTC))
                start_date = end_date - timedelta(days=base_policy["total_retention_days"])
                base_policy["trading_days_retained"] = len(
                    get_trading_days_between(start_date, end_date)
                )

            elif data_type == DataType.CORPORATE_ACTIONS:
                # Corporate actions kept longer
                base_policy["total_retention_days"] = max(layer.retention_days, 1095)  # Min 3 years
                base_policy["cleanup_after_days"] = None  # Never auto-cleanup

            elif data_type == DataType.REALTIME:
                # Realtime data shorter retention
                base_policy["hot_storage_days"] = min(layer.hot_storage_days, 3)  # Max 3 days hot
                base_policy["total_retention_days"] = min(
                    layer.retention_days, 30
                )  # Max 30 days total

            # Add policy metadata
            base_policy.update(
                {
                    "policy_version": "2.0",
                    "supports_hot_cold": True,
                    "supports_compression": True,
                    "auto_cleanup_enabled": base_policy["cleanup_after_days"] is not None,
                    "created_at": ensure_utc(datetime.now(UTC)).isoformat(),
                }
            )

            self.logger.debug(
                f"Generated retention policy for layer {layer.value}, "
                f"data_type {data_type.value}: {base_policy['total_retention_days']} days"
            )

            return base_policy

        except Exception as e:
            error = StorageError(
                f"Failed to get retention policy for layer {layer.value}, data_type {data_type.value}",
                operation="get_retention_policy",
                original_error=e,
                context={"layer": layer.value, "data_type": data_type.value},
            )
            self.logger.error(f"Retention policy error: {error}")
            raise error

    async def apply_retention_policy(self, symbol: str, layer: DataLayer) -> dict[str, Any]:
        """Apply retention policy to a symbol's data."""
        try:
            results = {
                "symbol": symbol,
                "layer": layer.value,
                "operations_performed": [],
                "data_moved": 0,
                "data_archived": 0,
                "data_deleted": 0,
                "start_time": ensure_utc(datetime.now(UTC)),
            }

            # Process each data type for the symbol
            for data_type in DataType:
                if await self._has_data_for_type(symbol, data_type):
                    type_results = await self._apply_retention_for_data_type(
                        symbol, layer, data_type
                    )
                    results["operations_performed"].append(type_results)
                    results["data_moved"] += type_results.get("moved_records", 0)
                    results["data_archived"] += type_results.get("archived_records", 0)
                    results["data_deleted"] += type_results.get("deleted_records", 0)

            results["end_time"] = ensure_utc(datetime.now(UTC))
            results["duration_seconds"] = (
                results["end_time"] - results["start_time"]
            ).total_seconds()

            self.logger.info(
                f"Applied retention policy for {symbol} (layer {layer.value}): "
                f"moved {results['data_moved']}, archived {results['data_archived']}, "
                f"deleted {results['data_deleted']} records"
            )

            return results

        except Exception as e:
            error = StorageError(
                f"Failed to apply retention policy for symbol {symbol}",
                operation="apply_retention_policy",
                original_error=e,
                context={"symbol": symbol, "layer": layer.value},
            )
            self.logger.error(f"Retention application error: {error}")
            raise error

    async def cleanup_expired_data(self, layer: DataLayer) -> dict[str, Any]:
        """Clean up expired data for a layer."""
        try:
            cleanup_results = {
                "layer": layer.value,
                "cleanup_operations": [],
                "total_records_deleted": 0,
                "total_space_freed_mb": 0,
                "start_time": ensure_utc(datetime.now(UTC)),
            }

            # Get all symbols in the layer
            symbols_query = """
                SELECT symbol FROM companies
                WHERE layer = $1 AND is_active = true
            """
            symbol_rows = await self.db_adapter.fetch_all(symbols_query, layer.value)
            symbols = [row["symbol"] for row in symbol_rows]

            # Process symbols in batches
            symbol_batches = chunk_list(symbols, self.batch_size)

            async def process_symbol_batch(batch):
                batch_results = []
                for symbol in batch:
                    for data_type in DataType:
                        cleanup_days = self._get_cleanup_days(layer, data_type)
                        if cleanup_days is None:
                            continue  # No auto-cleanup for this data type

                        cutoff_date = ensure_utc(datetime.now(UTC)) - timedelta(days=cleanup_days)
                        deleted_count = await self._cleanup_expired_records(
                            symbol, data_type, cutoff_date
                        )

                        if deleted_count > 0:
                            batch_results.append(
                                {
                                    "symbol": symbol,
                                    "data_type": data_type.value,
                                    "deleted_records": deleted_count,
                                    "cutoff_date": cutoff_date.isoformat(),
                                }
                            )
                return batch_results

            # Process all batches with concurrency control
            batch_results = await process_in_batches(
                symbol_batches, process_symbol_batch, max_concurrent=self.max_concurrent_operations
            )

            # Flatten results
            for batch_result in batch_results:
                for operation in batch_result:
                    cleanup_results["cleanup_operations"].append(operation)
                    cleanup_results["total_records_deleted"] += operation["deleted_records"]

            cleanup_results["end_time"] = ensure_utc(datetime.now(UTC))
            cleanup_results["duration_seconds"] = (
                cleanup_results["end_time"] - cleanup_results["start_time"]
            ).total_seconds()

            self.logger.info(
                f"Cleaned up expired data for layer {layer.value}: "
                f"deleted {cleanup_results['total_records_deleted']} records"
            )

            return cleanup_results

        except Exception as e:
            error = StorageError(
                f"Failed to cleanup expired data for layer {layer.value}",
                operation="cleanup_expired_data",
                original_error=e,
                context={"layer": layer.value},
            )
            self.logger.error(f"Cleanup error: {error}")
            raise error

    async def move_to_cold_storage(self, symbol: str, layer: DataLayer) -> bool:
        """Move data from hot to cold storage."""
        try:
            cutoff_date = ensure_utc(datetime.now(UTC)) - timedelta(days=layer.hot_storage_days)

            # Query hot storage data that needs to be moved
            hot_data_query = """
                SELECT COUNT(*) as record_count
                FROM {table_name}
                WHERE symbol = $1 AND timestamp < $2
            """.format(
                table_name=f"{self.hot_storage_table_prefix}_1h"
            )

            count_result = await self.db_adapter.fetch_one(hot_data_query, symbol, cutoff_date)
            records_to_move = count_result["record_count"] if count_result else 0

            if records_to_move == 0:
                self.logger.debug(f"No data to move to cold storage for {symbol}")
                return True

            # Move data to cold storage (implementation would depend on your cold storage system)
            # For now, we'll simulate the move by marking records
            # Note: storage_tier refers to data temperature (hot/cold), NOT API layers (0-3)
            # Hot storage = recent, frequently accessed data in PostgreSQL
            # Cold storage = older, archived data (e.g., S3, archive tables)
            move_query = """
                UPDATE {table_name}
                SET storage_tier = 'cold', moved_to_cold_at = $3
                WHERE symbol = $1 AND timestamp < $2 AND storage_tier = 'hot'
            """.format(
                table_name=f"{self.hot_storage_table_prefix}_1h"
            )

            result = await self.db_adapter.execute_query(
                move_query, symbol, cutoff_date, ensure_utc(datetime.now(UTC))
            )

            moved_count = result.rowcount

            self.logger.info(
                f"Moved {moved_count} records to cold storage for {symbol} (layer {layer.value})"
            )

            return moved_count > 0

        except Exception as e:
            error = StorageError(
                f"Failed to move data to cold storage for symbol {symbol}",
                storage_type="cold",
                operation="move_to_cold",
                original_error=e,
                context={"symbol": symbol, "layer": layer.value},
            )
            self.logger.error(f"Cold storage move error: {error}")
            raise error

    async def archive_old_data(self, symbol: str, layer: DataLayer) -> bool:
        """Archive old data according to policy."""

        async def _archive_with_breaker():
            archive_cutoff = ensure_utc(datetime.now(UTC)) - timedelta(days=layer.retention_days)

            archived_count = 0

            for data_type in DataType:
                table_name = data_type.table_name

                # Check if there's data to archive
                count_query = f"""
                    SELECT COUNT(*) as record_count
                    FROM {table_name}
                    WHERE symbol = $1 AND timestamp < $2 AND archived_at IS NULL
                """

                count_result = await self.db_adapter.fetch_one(count_query, symbol, archive_cutoff)
                records_to_archive = count_result["record_count"] if count_result else 0

                if records_to_archive > 0:
                    # Mark records as archived (actual archiving would be done by separate process)
                    archive_query = f"""
                        UPDATE {table_name}
                        SET archived_at = $3, archive_path = $4
                        WHERE symbol = $1 AND timestamp < $2 AND archived_at IS NULL
                    """

                    archive_path = f"{self.archive_storage_path}/{symbol}/{data_type.value}"

                    result = await self.db_adapter.execute_query(
                        archive_query,
                        symbol,
                        archive_cutoff,
                        ensure_utc(datetime.now(UTC)),
                        archive_path,
                    )

                    archived_count += result.rowcount

            if archived_count > 0:
                self.logger.info(
                    f"Archived {archived_count} records for {symbol} (layer {layer.value})"
                )

            return archived_count > 0

        try:
            # Use circuit breaker for resilient archive operations
            return await self.archive_circuit_breaker.call(_archive_with_breaker)

        except Exception as e:
            error = StorageError(
                f"Failed to archive data for symbol {symbol}",
                storage_type="archive",
                operation="archive_data",
                original_error=e,
                context={"symbol": symbol, "layer": layer.value},
            )
            self.logger.error(f"Archive error: {error}")
            raise error

    async def _has_data_for_type(self, symbol: str, data_type: DataType) -> bool:
        """Check if symbol has data for a specific data type."""
        try:
            table_name = data_type.table_name

            count_query = f"""
                SELECT 1 FROM {table_name}
                WHERE symbol = $1
                LIMIT 1
            """

            result = await self.db_adapter.fetch_one(count_query, symbol)
            return result is not None

        except Exception:
            # If table doesn't exist or query fails, assume no data
            return False

    async def _apply_retention_for_data_type(
        self, symbol: str, layer: DataLayer, data_type: DataType
    ) -> dict[str, Any]:
        """Apply retention policy for a specific data type."""
        policy = await self.get_retention_policy(layer, data_type)

        results = {
            "data_type": data_type.value,
            "moved_records": 0,
            "archived_records": 0,
            "deleted_records": 0,
        }

        # Move to cold storage
        if policy["hot_storage_days"] > 0:
            moved = await self.move_to_cold_storage(symbol, layer)
            if moved:
                results["moved_records"] = 1  # Simplified for now

        # Archive old data
        if policy["archive_after_days"] > 0:
            archived = await self.archive_old_data(symbol, layer)
            if archived:
                results["archived_records"] = 1  # Simplified for now

        return results

    async def _cleanup_expired_records(
        self, symbol: str, data_type: DataType, cutoff_date: datetime
    ) -> int:
        """Delete expired records for a symbol and data type."""
        try:
            table_name = data_type.table_name

            delete_query = f"""
                DELETE FROM {table_name}
                WHERE symbol = $1 AND timestamp < $2 AND archived_at IS NOT NULL
            """

            result = await self.db_adapter.execute_query(delete_query, symbol, cutoff_date)
            return result.rowcount

        except Exception:
            # If deletion fails, return 0
            return 0

    def _get_cleanup_days(self, layer: DataLayer, data_type: DataType) -> int | None:
        """Get cleanup days for layer and data type (None means no auto-cleanup)."""
        # Corporate actions are never auto-cleaned up
        if data_type == DataType.CORPORATE_ACTIONS:
            return None

        # Other data types get cleaned up after retention period + buffer
        buffer_days = {
            DataLayer.BASIC: 30,  # 30 day buffer
            DataLayer.LIQUID: 90,  # 90 day buffer
            DataLayer.CATALYST: 180,  # 180 day buffer
            DataLayer.ACTIVE: 365,  # 365 day buffer
        }

        return layer.retention_days + buffer_days[layer]
