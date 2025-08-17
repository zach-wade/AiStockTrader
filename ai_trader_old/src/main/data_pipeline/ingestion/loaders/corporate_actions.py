"""
Corporate actions bulk loader for efficient backfill operations.

This module provides optimized bulk loading for corporate actions (dividends, splits),
using PostgreSQL COPY command and efficient batching strategies.
Uses Strategy pattern with action processors for different action types.
"""

# Standard library imports
from datetime import UTC, datetime
from typing import Any

# Local imports
from main.data_pipeline.services.processing import CorporateActionsConfig, CorporateActionsService
from main.data_pipeline.services.processing.action_processors import (
    ActionProcessorConfig,
    DividendProcessor,
    SplitProcessor,
)
from main.interfaces.database import IAsyncDatabase
from main.interfaces.ingestion import BulkLoadConfig, BulkLoadResult
from main.utils.core import get_logger

from .base import BaseBulkLoader

logger = get_logger(__name__)


class CorporateActionsBulkLoader(BaseBulkLoader[dict[str, Any]]):
    """
    Optimized bulk loader for corporate actions data.

    Uses action processors to handle different types (dividends, splits)
    and loads efficiently using PostgreSQL COPY operations.
    """

    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        actions_service: CorporateActionsService | None = None,
        archive: Any | None = None,
        config: BulkLoadConfig | None = None,
    ):
        """
        Initialize corporate actions bulk loader.

        Args:
            db_adapter: Database adapter for operations
            actions_service: Service for corporate actions operations
            archive: Optional archive for cold storage
            config: Bulk loading configuration
        """
        super().__init__(
            db_adapter=db_adapter, archive=archive, config=config, data_type="corporate_actions"
        )

        # Initialize corporate actions service if not provided
        if not actions_service:
            actions_config = CorporateActionsConfig()
            actions_service = CorporateActionsService(actions_config)

        self.actions_service = actions_service

        # Initialize action processors
        processor_config = ActionProcessorConfig()
        self.processors = [
            DividendProcessor(self.actions_service, processor_config),
            SplitProcessor(self.actions_service, processor_config),
        ]

        logger.info(
            f"CorporateActionsBulkLoader initialized with {len(self.processors)} processors"
        )

    async def load(
        self, data: list[dict[str, Any]], symbols: list[str], source: str = "polygon", **kwargs
    ) -> BulkLoadResult:
        """
        Load corporate actions efficiently using bulk operations.

        Args:
            data: List of corporate action records
            symbols: List of symbols this data relates to
            source: Data source name ('polygon', 'alpaca', etc.)
            **kwargs: Additional parameters

        Returns:
            BulkLoadResult with operation details
        """
        result = BulkLoadResult(success=False, data_type=self.data_type)

        if not data:
            result.success = True
            result.skip_reason = "No data provided"
            return result

        try:
            # Process each action with appropriate processor
            prepared_records = []
            unprocessed = 0

            for action in data:
                # Find appropriate processor
                processor = None
                for p in self.processors:
                    if p.can_process(action):
                        processor = p
                        break

                if not processor:
                    logger.debug(f"No processor found for action: {action}")
                    unprocessed += 1
                    continue

                # Process the action
                record = processor.process(action, source)
                if record:
                    prepared_records.append(record)
                else:
                    unprocessed += 1

            if not prepared_records:
                logger.info(f"No valid corporate actions to load ({unprocessed} unprocessed)")
                result.success = True
                result.skip_reason = (
                    f"No valid records after processing ({unprocessed} unprocessed)"
                )
                return result

            logger.debug(f"Prepared {len(prepared_records)} corporate action records")

            # Add to buffer
            self._add_to_buffer(prepared_records)
            for symbol in symbols:
                self._symbols_in_buffer.add(symbol.upper())

            # Check if we should flush
            if self._should_flush():
                flush_result = await self._flush_buffer()
                result.records_loaded = flush_result.records_loaded
                result.records_failed = flush_result.records_failed
                result.symbols_processed = flush_result.symbols_processed
                result.load_time_seconds = flush_result.load_time_seconds
                result.archive_time_seconds = flush_result.archive_time_seconds
                result.errors = flush_result.errors
                result.success = flush_result.success
            else:
                # Data is buffered, will be written later
                result.success = True
                result.records_loaded = len(prepared_records)
                result.metadata["buffered"] = True
                result.metadata["unprocessed"] = unprocessed

            return result

        except Exception as e:
            error_msg = f"Failed to load corporate actions: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            return result

    def _estimate_record_size(self, record: dict[str, Any]) -> int:
        """Estimate size of a corporate action record."""
        # Corporate action records are relatively small
        return 100

    async def _load_to_database(self, records: list[dict[str, Any]]) -> int:
        """
        Load corporate action records to database using COPY.

        Args:
            records: Corporate action records to load

        Returns:
            Number of records loaded
        """
        if not records:
            return 0

        # Columns for corporate_actions table
        columns = [
            "ticker",
            "action_type",
            "ex_date",
            "cash_amount",
            "currency",
            "dividend_type",
            "frequency",
            "pay_date",
            "record_date",
            "declaration_date",
            "split_from",
            "split_to",
            "polygon_id",
            "created_at",
            "updated_at",
        ]

        # Convert records to tuples for COPY
        copy_records = []
        for record in records:
            copy_record = (
                record.get("ticker", ""),
                record.get("action_type", record.get("type")),
                record.get("ex_date"),
                record.get("cash_amount"),
                record.get("currency"),
                record.get("dividend_type"),
                record.get("frequency"),
                record.get("payment_date"),  # Maps to pay_date in DB
                record.get("record_date"),
                record.get("declaration_date"),
                self._convert_to_int(record.get("split_from")),
                self._convert_to_int(record.get("split_to")),
                record.get("polygon_id"),
                record.get("created_at"),
                record.get("updated_at"),
            )
            copy_records.append(copy_record)

        async with self.db_adapter.acquire() as conn:
            try:
                # Create temp table
                await conn.execute("DROP TABLE IF EXISTS temp_corporate_actions")
                await conn.execute(
                    "CREATE TEMP TABLE temp_corporate_actions (LIKE corporate_actions INCLUDING ALL)"
                )

                # Use COPY to load data
                await conn.copy_records_to_table(
                    "temp_corporate_actions", records=copy_records, columns=columns
                )

                # UPSERT from temp table
                upsert_sql = """
                INSERT INTO corporate_actions
                SELECT * FROM temp_corporate_actions
                ON CONFLICT (ticker, action_type, ex_date)
                DO UPDATE SET
                    cash_amount = EXCLUDED.cash_amount,
                    currency = EXCLUDED.currency,
                    dividend_type = EXCLUDED.dividend_type,
                    frequency = EXCLUDED.frequency,
                    pay_date = EXCLUDED.pay_date,
                    record_date = EXCLUDED.record_date,
                    declaration_date = EXCLUDED.declaration_date,
                    split_from = EXCLUDED.split_from,
                    split_to = EXCLUDED.split_to,
                    polygon_id = EXCLUDED.polygon_id,
                    updated_at = EXCLUDED.updated_at
                """

                result = await conn.execute(upsert_sql)

                # Clean up
                await conn.execute("DROP TABLE temp_corporate_actions")

                # Extract count
                if result and result.startswith("INSERT"):
                    parts = result.split()
                    if len(parts) >= 3:
                        return int(parts[2])

                return len(records)

            except Exception as e:
                logger.warning(f"COPY failed for corporate actions: {e}, falling back to INSERT")
                # Fall back to INSERT method
                return await self._load_with_insert(records)

    async def _load_with_insert(self, records: list[dict[str, Any]]) -> int:
        """Fallback INSERT method for loading corporate actions."""
        batch_size = 100
        total_loaded = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            # Build parameterized insert
            placeholders = []
            values = []
            param_count = 1

            for record in batch:
                # 15 parameters per record
                params = []
                for j in range(15):
                    params.append(f"${param_count + j}")
                placeholder = f"({','.join(params)})"
                placeholders.append(placeholder)

                values.extend(
                    [
                        record.get("ticker", ""),
                        record.get("action_type", record.get("type")),
                        record.get("ex_date"),
                        record.get("cash_amount"),
                        record.get("currency"),
                        record.get("dividend_type"),
                        record.get("frequency"),
                        record.get("payment_date"),
                        record.get("record_date"),
                        record.get("declaration_date"),
                        self._convert_to_int(record.get("split_from")),
                        self._convert_to_int(record.get("split_to")),
                        record.get("polygon_id"),
                        record.get("created_at"),
                        record.get("updated_at"),
                    ]
                )

                param_count += 15

            sql = f"""
            INSERT INTO corporate_actions (
                ticker, action_type, ex_date, cash_amount, currency,
                dividend_type, frequency, pay_date, record_date,
                declaration_date, split_from, split_to, polygon_id,
                created_at, updated_at
            )
            VALUES {','.join(placeholders)}
            ON CONFLICT (ticker, action_type, ex_date)
            DO UPDATE SET
                cash_amount = EXCLUDED.cash_amount,
                currency = EXCLUDED.currency,
                dividend_type = EXCLUDED.dividend_type,
                frequency = EXCLUDED.frequency,
                pay_date = EXCLUDED.pay_date,
                record_date = EXCLUDED.record_date,
                declaration_date = EXCLUDED.declaration_date,
                split_from = EXCLUDED.split_from,
                split_to = EXCLUDED.split_to,
                polygon_id = EXCLUDED.polygon_id,
                updated_at = EXCLUDED.updated_at
            """

            async with self.db_adapter.acquire() as conn:
                result = await conn.execute(sql, *values)
                if result.startswith("INSERT"):
                    parts = result.split()
                    if len(parts) >= 3:
                        total_loaded += int(parts[2])

        return total_loaded

    def _convert_to_int(self, value: Any) -> int | None:
        """Convert float split values to int for database."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    async def _archive_records(self, records: list[dict[str, Any]]) -> None:
        """Archive corporate action records to cold storage."""
        if not self.archive or not records:
            return

        # Group by action type and date for archiving
        # Standard library imports
        from collections import defaultdict

        type_date_groups = defaultdict(lambda: defaultdict(list))

        for record in records:
            action_type = record.get("type", "unknown")
            if record.get("ex_date"):
                date = (
                    record["ex_date"].date()
                    if isinstance(record["ex_date"], datetime)
                    else record["ex_date"]
                )
                type_date_groups[action_type][date].append(record)

        # Archive each group
        for action_type, date_groups in type_date_groups.items():
            for date, group_records in date_groups.items():
                try:
                    # Create archive metadata
                    metadata = {
                        "data_type": "corporate_actions",
                        "action_type": action_type,
                        "record_count": len(group_records),
                        "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                        "symbols": list(
                            set(
                                record.get("symbol", "")
                                for record in group_records
                                if record.get("symbol")
                            )
                        ),
                    }

                    # Create RawDataRecord for archive
                    # Local imports
                    from main.data_pipeline.storage.archive import RawDataRecord

                    record = RawDataRecord(
                        source=group_records[0].get("source", "polygon"),
                        data_type="corporate_actions",
                        symbol=f"{action_type}_{date}",  # Use type+date as identifier
                        timestamp=datetime.now(UTC),
                        data={"actions": group_records},
                        metadata=metadata,
                    )

                    # Use archive's async save method
                    await self.archive.save_raw_record_async(record)

                    logger.debug(
                        f"Archived {len(group_records)} {action_type} actions for {date} "
                        f"({len(metadata['symbols'])} unique symbols)"
                    )
                except Exception as e:
                    logger.error(f"Failed to archive {action_type} for {date}: {e}")
