"""
Common database operations and patterns.

This module provides reusable database operations including:
- Transaction strategies
- Batch operations (upsert, delete)
- Common query patterns
- Database utilities
"""

# Standard library imports
import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
import logging
from typing import Any

# Third-party imports
from sqlalchemy import and_, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class TransactionStrategy(Enum):
    """Strategy for handling transactions in batch operations."""

    ALL_OR_NOTHING = "all_or_nothing"  # Single transaction, rollback on any failure
    INDIVIDUAL_SAVEPOINTS = "individual_savepoints"  # Savepoint per record
    BATCH_WITH_FALLBACK = "batch_with_fallback"  # Try batch, fallback to individual


class BatchOperationResult:
    """Result of a batch database operation."""

    def __init__(self):
        self.total_records = 0
        self.successful_records = 0
        self.failed_records = 0
        self.errors = []
        self.duration_ms = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_records == 0:
            return 0.0
        return self.successful_records / self.total_records

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_records": self.total_records,
            "successful_records": self.successful_records,
            "failed_records": self.failed_records,
            "success_rate": self.success_rate,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


async def batch_upsert(
    session: Session,
    model_class: type,
    records: list[dict[str, Any]],
    constraint_name: str,
    update_fields: list[str] | None = None,
    strategy: TransactionStrategy = TransactionStrategy.BATCH_WITH_FALLBACK,
    batch_size: int = 1000,
) -> BatchOperationResult:
    """
    Perform batch upsert operation with configurable strategy.

    Args:
        session: Database session
        model_class: SQLAlchemy model class
        records: List of records to upsert
        constraint_name: Unique constraint name for conflict resolution
        update_fields: Fields to update on conflict
        strategy: Transaction strategy to use
        batch_size: Maximum records per batch

    Returns:
        BatchOperationResult with operation statistics
    """
    result = BatchOperationResult()
    result.total_records = len(records)
    start_time = datetime.now()

    if not records:
        return result

    # Deduplicate records based on primary key
    table = model_class.__table__
    pk_columns = [col.name for col in table.primary_key.columns]

    if pk_columns:
        seen = {}
        unique_records = []
        for record in records:
            key_parts = tuple(record.get(col_name) for col_name in pk_columns)
            if all(k is not None for k in key_parts):
                if key_parts not in seen:
                    seen[key_parts] = record
                    unique_records.append(record)
            else:
                logger.warning(f"Record missing primary key {pk_columns}: {record}")
                unique_records.append(record)

        if len(unique_records) < len(records):
            logger.debug(f"Deduplicated {len(records) - len(unique_records)} records")
        records = unique_records

    # Process based on strategy
    if strategy == TransactionStrategy.ALL_OR_NOTHING:
        result = await _batch_upsert_all_or_nothing(
            session, model_class, records, constraint_name, update_fields, batch_size
        )
    elif strategy == TransactionStrategy.INDIVIDUAL_SAVEPOINTS:
        result = await _batch_upsert_with_savepoints(
            session, model_class, records, constraint_name, update_fields
        )
    else:  # BATCH_WITH_FALLBACK
        try:
            result = await _batch_upsert_all_or_nothing(
                session, model_class, records, constraint_name, update_fields, batch_size
            )
        except Exception as e:
            logger.warning(f"Batch upsert failed, falling back to savepoints: {e}")
            result = await _batch_upsert_with_savepoints(
                session, model_class, records, constraint_name, update_fields
            )

    result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
    return result


async def _batch_upsert_all_or_nothing(
    session: Session,
    model_class: type,
    records: list[dict[str, Any]],
    constraint_name: str,
    update_fields: list[str] | None,
    batch_size: int,
) -> BatchOperationResult:
    """Execute batch upsert in a single transaction."""
    result = BatchOperationResult()
    result.total_records = len(records)

    try:
        # Process in batches to avoid memory issues
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            stmt = pg_insert(model_class).values(batch)

            if update_fields:
                # Update specified fields on conflict
                update_dict = {field: getattr(stmt.excluded, field) for field in update_fields}
                stmt = stmt.on_conflict_do_update(constraint=constraint_name, set_=update_dict)
            else:
                # Do nothing on conflict
                stmt = stmt.on_conflict_do_nothing(constraint=constraint_name)

            await session.execute(stmt)

        await session.commit()
        result.successful_records = len(records)

    except Exception as e:
        await session.rollback()
        logger.error(f"Batch upsert failed: {e}")
        result.failed_records = len(records)
        result.errors.append(str(e))
        raise

    return result


async def _batch_upsert_with_savepoints(
    session: Session,
    model_class: type,
    records: list[dict[str, Any]],
    constraint_name: str,
    update_fields: list[str] | None,
) -> BatchOperationResult:
    """Execute upsert with savepoint per record."""
    result = BatchOperationResult()
    result.total_records = len(records)

    for idx, record in enumerate(records):
        try:
            # Create savepoint
            savepoint = await session.begin_nested()

            stmt = pg_insert(model_class).values(record)

            if update_fields:
                update_dict = {field: getattr(stmt.excluded, field) for field in update_fields}
                stmt = stmt.on_conflict_do_update(constraint=constraint_name, set_=update_dict)
            else:
                stmt = stmt.on_conflict_do_nothing(constraint=constraint_name)

            await session.execute(stmt)
            await savepoint.commit()
            result.successful_records += 1

        except Exception as e:
            await savepoint.rollback()
            result.failed_records += 1
            error_msg = f"Record {idx} failed: {e!s}"
            result.errors.append(error_msg)
            logger.debug(error_msg)

    # Commit all successful records
    await session.commit()
    return result


async def batch_delete(
    session: Session,
    model_class: type,
    filters: list[tuple[str, Any]],
    strategy: TransactionStrategy = TransactionStrategy.ALL_OR_NOTHING,
) -> BatchOperationResult:
    """
    Perform batch delete operation.

    Args:
        session: Database session
        model_class: SQLAlchemy model class
        filters: List of (field, value) tuples for WHERE clause
        strategy: Transaction strategy to use

    Returns:
        BatchOperationResult with operation statistics
    """
    result = BatchOperationResult()
    start_time = datetime.now()

    # Build WHERE clause
    conditions = []
    for field, value in filters:
        column = getattr(model_class, field)
        if isinstance(value, list):
            conditions.append(column.in_(value))
        else:
            conditions.append(column == value)

    if not conditions:
        return result

    where_clause = and_(*conditions) if len(conditions) > 1 else conditions[0]

    try:
        # Count records to be deleted
        count_stmt = select(func.count()).select_from(model_class).where(where_clause)
        count_result = await session.execute(count_stmt)
        result.total_records = count_result.scalar()

        # Execute delete
        delete_stmt = delete(model_class).where(where_clause)
        delete_result = await session.execute(delete_stmt)

        await session.commit()
        result.successful_records = delete_result.rowcount

    except Exception as e:
        await session.rollback()
        logger.error(f"Batch delete failed: {e}")
        result.failed_records = result.total_records
        result.errors.append(str(e))

        if strategy == TransactionStrategy.BATCH_WITH_FALLBACK:
            # Could implement row-by-row deletion here if needed
            pass

    result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000
    return result


async def execute_with_retry(
    operation: Callable,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
) -> Any:
    """
    Execute a database operation with retry logic.

    Args:
        operation: Callable that performs the database operation
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff

    Returns:
        Result of the operation
    """
    last_error = None
    delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except (IntegrityError, SQLAlchemyError) as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                await asyncio.sleep(delay)
                if exponential_backoff:
                    delay *= 2
            else:
                logger.error(f"Operation failed after {max_retries + 1} attempts")

    raise last_error


@asynccontextmanager
async def transaction_context(session: Session, nested: bool = False):
    """
    Context manager for database transactions with automatic rollback.

    Args:
        session: Database session
        nested: Whether to use savepoint (nested transaction)

    Yields:
        Transaction object
    """
    if nested:
        transaction = await session.begin_nested()
    else:
        transaction = await session.begin()

    try:
        yield transaction
        await transaction.commit()
    except Exception:
        await transaction.rollback()
        raise
