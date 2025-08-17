"""
ETL Manager

Manages Extract, Transform, Load operations from archive to database.
Extracted and refactored from symbol_data_processor for reusability.
"""

# Standard library imports
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.data_pipeline.core.enums import DataLayer, DataType
from main.data_pipeline.storage.archive import DataArchive
from main.interfaces.database import IAsyncDatabase
from main.utils.core import (
    CircuitBreaker,
    CircuitBreakerConfig,
    ErrorHandlingMixin,
    async_retry,
    get_logger,
)
from main.utils.monitoring import MetricType, record_metric, timer

from .loader_coordinator import LoaderCoordinator


@dataclass
class ETLResult:
    """Result of an ETL operation."""

    success: bool
    records_loaded: int = 0
    records_failed: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ETLManager(ErrorHandlingMixin):
    """
    Manages ETL operations from archive to database.

    Handles extraction from archives, data transformation,
    and loading to database using appropriate bulk loaders.
    """

    # Source mappings for different data types
    SOURCE_MAPPINGS = {
        DataType.MARKET_DATA: "polygon",
        DataType.NEWS: "polygon",
        DataType.FINANCIALS: "polygon_financials",
        DataType.CORPORATE_ACTIONS: "polygon",
    }

    def __init__(
        self, db_adapter: IAsyncDatabase, archive: DataArchive, config: dict[str, Any] | None = None
    ):
        """
        Initialize ETL manager.

        Args:
            db_adapter: Database adapter
            archive: Data archive instance
            config: Optional configuration
        """
        self.db_adapter = db_adapter
        self.archive = archive
        self.config = config or {}
        self.logger = get_logger(__name__)

        # Initialize circuit breaker for resilience
        cb_config = CircuitBreakerConfig(
            failure_threshold=5, recovery_timeout=60.0, success_threshold=3
        )
        self.circuit_breaker = CircuitBreaker(cb_config)

        # Initialize loader coordinator
        self.loader_coordinator = LoaderCoordinator(db_adapter, archive, config)

        # ETL statistics
        self._etl_stats = {"total_operations": 0, "successful": 0, "failed": 0, "total_records": 0}

    @async_retry(max_attempts=3, delay=1.0)
    async def load_data(
        self,
        symbol: str,
        data_type: DataType,
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer,
        source: str | None = None,
        interval: str | None = None,
    ) -> ETLResult:
        """
        Main ETL entry point for loading data from archive to database.

        Args:
            symbol: Stock symbol
            data_type: Type of data to load
            start_date: Start date for data
            end_date: End date for data
            layer: Data layer for processing rules
            source: Optional data source override
            interval: Optional interval for market data

        Returns:
            ETLResult with load statistics
        """
        with timer(
            "etl.load",
            tags={
                "symbol": symbol,
                "data_type": data_type.value if hasattr(data_type, "value") else str(data_type),
                "layer": layer.name,
            },
        ):
            try:
                self.logger.info(
                    f"Starting ETL for {symbol} {data_type} from {start_date} to {end_date}"
                )

                # Use circuit breaker for protection
                result = await self.circuit_breaker.call(
                    self._perform_etl,
                    symbol,
                    data_type,
                    start_date,
                    end_date,
                    layer,
                    source,
                    interval,
                )

                # Update statistics
                self._etl_stats["total_operations"] += 1
                if result.success:
                    self._etl_stats["successful"] += 1
                    self._etl_stats["total_records"] += result.records_loaded
                else:
                    self._etl_stats["failed"] += 1

                # Record metrics
                record_metric(
                    "etl.records_loaded",
                    result.records_loaded,
                    MetricType.COUNTER,
                    tags={"symbol": symbol, "data_type": str(data_type)},
                )

                if result.records_failed > 0:
                    record_metric(
                        "etl.records_failed",
                        result.records_failed,
                        MetricType.COUNTER,
                        tags={"symbol": symbol, "data_type": str(data_type)},
                    )

                return result

            except Exception as e:
                self.logger.error(f"ETL failed for {symbol} {data_type}: {e}")
                self._etl_stats["failed"] += 1
                return ETLResult(success=False, errors=[str(e)])

    async def _perform_etl(
        self,
        symbol: str,
        data_type: DataType,
        start_date: datetime,
        end_date: datetime,
        layer: DataLayer,
        source: str | None,
        interval: str | None,
    ) -> ETLResult:
        """Perform the actual ETL operation."""
        # Extract from archive
        raw_records = await self._extract_from_archive(
            symbol, data_type, start_date, end_date, source
        )

        if not raw_records:
            self.logger.info(f"No data found in archive for {symbol} {data_type}")
            return ETLResult(success=True, records_loaded=0)

        self.logger.info(f"Extracted {len(raw_records)} raw records from archive")
        # gauge("etl.raw_records", len(raw_records), tags={"symbol": symbol})

        # Transform data
        transformed_data = await self._transform_data(raw_records, data_type, interval)

        if not transformed_data:
            self.logger.warning(f"No data after transformation for {symbol} {data_type}")
            return ETLResult(
                success=True, records_loaded=0, warnings=["No data after transformation"]
            )

        self.logger.info(f"Transformed {len(transformed_data)} records")

        # Load to database
        load_result = await self._load_to_database(
            transformed_data, symbol, data_type, layer, interval, source
        )

        return load_result

    async def _extract_from_archive(
        self,
        symbol: str,
        data_type: DataType,
        start_date: datetime,
        end_date: datetime,
        source: str | None,
    ) -> list[Any]:
        """Extract data from archive."""
        # Determine source
        if source is None:
            source = self.SOURCE_MAPPINGS.get(data_type, "polygon")

        # Map DataType enum to string for archive query
        data_type_str = (
            data_type.value.lower() if hasattr(data_type, "value") else str(data_type).lower()
        )

        # Query archive
        with timer("etl.extract", tags={"source": source, "data_type": data_type_str}):
            raw_records = await self.archive.query_raw_records(
                source=source,
                data_type=data_type_str,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )

        return raw_records

    async def _transform_data(
        self, raw_records: list[Any], data_type: DataType, interval: str | None
    ) -> list[dict[str, Any]]:
        """Transform raw records to loadable format."""
        transformed = []

        with timer("etl.transform", tags={"data_type": str(data_type)}):
            for i, record in enumerate(raw_records):
                if record.data is None:
                    continue

                try:
                    # Handle different data formats
                    if isinstance(record.data, pd.DataFrame):
                        # Convert DataFrame to records
                        records = record.data.to_dict("records")

                        # Add metadata to each record if needed
                        if data_type == DataType.MARKET_DATA and interval:
                            for r in records:
                                r["interval"] = interval or record.metadata.get("interval", "1day")

                        transformed.extend(records)

                    elif isinstance(record.data, dict):
                        # Handle nested data structure
                        if "data" in record.data:
                            inner_data = record.data["data"]
                            if isinstance(inner_data, list):
                                transformed.extend(inner_data)
                            else:
                                transformed.append(inner_data)
                        else:
                            transformed.append(record.data)

                    elif isinstance(record.data, list):
                        transformed.extend(record.data)

                    else:
                        # Try to append as-is
                        transformed.append(record.data)

                except Exception as e:
                    self.logger.warning(f"Failed to transform record {i}: {e}")
                    continue

        self.logger.debug(
            f"Transformed {len(raw_records)} raw records into {len(transformed)} data records"
        )
        return transformed

    async def _load_to_database(
        self,
        data: list[dict[str, Any]],
        symbol: str,
        data_type: DataType,
        layer: DataLayer,
        interval: str | None,
        source: str | None,
    ) -> ETLResult:
        """Load data to database using appropriate bulk loader."""
        try:
            # Get appropriate loader
            loader = await self.loader_coordinator.get_loader(data_type)

            if loader is None:
                return ETLResult(success=False, errors=[f"No loader available for {data_type}"])

            # Prepare load parameters based on data type
            load_params = {"data": data, "source": source or "polygon"}

            if data_type == DataType.MARKET_DATA:
                load_params["symbol"] = symbol
                load_params["interval"] = interval or "1day"
            elif data_type == DataType.NEWS:
                load_params["symbols"] = [symbol]
            elif data_type == DataType.FINANCIALS or data_type == DataType.CORPORATE_ACTIONS:
                load_params["symbol"] = symbol

            # Load data
            with timer("etl.database_load", tags={"data_type": str(data_type)}):
                load_result = await loader.load(**load_params)

            # Flush any remaining data
            flush_result = await loader.flush_all()

            # Combine results
            total_loaded = load_result.records_loaded + (
                flush_result.records_loaded if hasattr(flush_result, "records_loaded") else 0
            )

            total_failed = load_result.records_failed + (
                flush_result.records_failed if hasattr(flush_result, "records_failed") else 0
            )

            self.logger.info(f"Loaded {total_loaded} records to database ({total_failed} failed)")

            return ETLResult(
                success=True,
                records_loaded=total_loaded,
                records_failed=total_failed,
                metadata={"layer": layer.name},
            )

        except Exception as e:
            self.logger.error(f"Database load failed: {e}")
            return ETLResult(success=False, errors=[str(e)])

    def _get_batch_size(self, layer: DataLayer) -> int:
        """Get batch size based on layer."""
        batch_sizes = {
            DataLayer.BASIC: 500,
            DataLayer.LIQUID: 1000,
            DataLayer.CATALYST: 2000,
            DataLayer.ACTIVE: 5000,
        }
        return batch_sizes.get(layer, 1000)

    async def get_etl_stats(self) -> dict[str, Any]:
        """Get ETL statistics."""
        stats = self._etl_stats.copy()
        if stats["total_operations"] > 0:
            stats["success_rate"] = stats["successful"] / stats["total_operations"]
            stats["avg_records_per_operation"] = stats["total_records"] / stats["total_operations"]
        return stats
