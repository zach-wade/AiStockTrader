"""
CRUD Executor Helper

Executes CRUD operations with retry logic, circuit breaker, and transaction support.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import time

from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import (
    OperationResult,
    TransactionStrategy
)
from main.utils.core import get_logger, AsyncCircuitBreaker
from ..constants import DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT, DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD

logger = get_logger(__name__)


async def _retry_operation(operation, max_retries: int = 3, initial_delay: float = 1.0):
    """Simple retry helper for database operations."""
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.warning(f"Operation failed after {max_retries} attempts: {e}")
    
    raise last_exception


class CrudExecutor:
    """
    Executes CRUD operations with resilience patterns.
    
    Provides retry logic, circuit breaker protection, and
    transaction management for database operations.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        table_name: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_circuit_breaker: bool = True,
        transaction_strategy: TransactionStrategy = TransactionStrategy.BATCH
    ):
        """
        Initialize the CrudExecutor.
        
        Args:
            db_adapter: Database adapter for operations
            table_name: Target table name
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay in seconds
            enable_circuit_breaker: Enable circuit breaker pattern
            transaction_strategy: Transaction handling strategy
        """
        self.db_adapter = db_adapter
        self.table_name = table_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.transaction_strategy = transaction_strategy
        
        # Initialize circuit breaker if enabled
        self.circuit_breaker = None
        if enable_circuit_breaker:
            self.circuit_breaker = AsyncCircuitBreaker(
                failure_threshold=DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                recovery_timeout=DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
                expected_exception=Exception
            )
        
        logger.debug(f"CrudExecutor initialized for table: {table_name}")
    
    async def execute_insert(
        self,
        query: str,
        params: List[Any]
    ) -> OperationResult:
        """
        Execute an INSERT query with retry logic.
        
        Args:
            query: SQL INSERT query
            params: Query parameters
            
        Returns:
            Operation result
        """
        start_time = time.time()
        
        async def _insert():
            if self.circuit_breaker:
                return await self.circuit_breaker.call(
                    self.db_adapter.execute_query,
                    query,
                    *params
                )
            else:
                return await self.db_adapter.execute_query(query, *params)
        
        try:
            result = await _retry_operation(
                _insert,
                max_retries=self.max_retries,
                initial_delay=self.retry_delay
            )
            
            return OperationResult(
                success=True,
                data=result,
                records_affected=1,
                records_created=1,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Insert failed after {self.max_retries} retries: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time
            )
    
    async def execute_bulk_insert(
        self,
        records: List[Dict[str, Any]]
    ) -> OperationResult:
        """
        Execute bulk INSERT with transaction support.
        
        Args:
            records: List of records to insert
            
        Returns:
            Operation result with statistics
        """
        start_time = time.time()
        created = 0
        failed = 0
        errors = []
        
        if self.transaction_strategy == TransactionStrategy.BATCH:
            # Process in a single transaction
            try:
                async with self.db_adapter.transaction():
                    for record in records:
                        query = self._build_insert_query(record)
                        await self.db_adapter.execute_query(query['sql'], *query['params'])
                        created += 1
                
                return OperationResult(
                    success=True,
                    records_affected=created,
                    records_created=created,
                    duration_seconds=time.time() - start_time
                )
                
            except Exception as e:
                logger.error(f"Bulk insert transaction failed: {e}")
                return OperationResult(
                    success=False,
                    error=str(e),
                    duration_seconds=time.time() - start_time
                )
        
        elif self.transaction_strategy == TransactionStrategy.SAVEPOINT:
            # Use savepoints for partial rollback
            try:
                async with self.db_adapter.transaction():
                    for i, record in enumerate(records):
                        savepoint_name = f"sp_{i}"
                        try:
                            # Create savepoint
                            await self.db_adapter.execute_query(f"SAVEPOINT {savepoint_name}")
                            
                            # Insert record
                            query = self._build_insert_query(record)
                            await self.db_adapter.execute_query(query['sql'], *query['params'])
                            created += 1
                            
                        except Exception as e:
                            # Safely rollback to savepoint
                            try:
                                await self.db_adapter.execute_query(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                            except Exception as rollback_error:
                                logger.warning(f"Failed to rollback savepoint {savepoint_name}: {rollback_error}")
                            
                            failed += 1
                            errors.append(f"Record {i}: {str(e)}")
            except Exception as transaction_error:
                logger.error(f"Transaction failed in SAVEPOINT strategy: {transaction_error}")
                return OperationResult(
                    success=False,
                    error=f"Transaction failed: {str(transaction_error)}",
                    duration_seconds=time.time() - start_time
                )
            
            return OperationResult(
                success=failed == 0,
                records_affected=created,
                records_created=created,
                records_skipped=failed,
                error="; ".join(errors) if errors else None,
                duration_seconds=time.time() - start_time
            )
        
        else:  # NONE or SINGLE
            # Process individually without transactions
            for record in records:
                try:
                    query = self._build_insert_query(record)
                    await self.db_adapter.execute_query(query['sql'], *query['params'])
                    created += 1
                except Exception as e:
                    failed += 1
                    errors.append(str(e))
            
            return OperationResult(
                success=failed == 0,
                records_affected=created,
                records_created=created,
                records_skipped=failed,
                error="; ".join(errors[:5]) if errors else None,  # Limit error messages
                duration_seconds=time.time() - start_time
            )
    
    async def execute_update(
        self,
        query: str,
        params: List[Any]
    ) -> OperationResult:
        """
        Execute an UPDATE query with retry logic.
        
        Args:
            query: SQL UPDATE query
            params: Query parameters
            
        Returns:
            Operation result
        """
        start_time = time.time()
        
        async def _update():
            if self.circuit_breaker:
                return await self.circuit_breaker.call(
                    self.db_adapter.execute_query,
                    query,
                    *params
                )
            else:
                return await self.db_adapter.execute_query(query, *params)
        
        try:
            affected = await _retry_operation(
                _update,
                max_retries=self.max_retries,
                initial_delay=self.retry_delay
            )
            
            return OperationResult(
                success=True,
                records_affected=affected or 0,
                records_updated=affected or 0,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Update failed after {self.max_retries} retries: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time
            )
    
    async def execute_delete(
        self,
        query: str,
        params: List[Any]
    ) -> OperationResult:
        """
        Execute a DELETE query with retry logic.
        
        Args:
            query: SQL DELETE query
            params: Query parameters
            
        Returns:
            Operation result
        """
        start_time = time.time()
        
        async def _delete():
            if self.circuit_breaker:
                return await self.circuit_breaker.call(
                    self.db_adapter.execute_query,
                    query,
                    *params
                )
            else:
                return await self.db_adapter.execute_query(query, *params)
        
        try:
            affected = await _retry_operation(
                _delete,
                max_retries=self.max_retries,
                initial_delay=self.retry_delay
            )
            
            return OperationResult(
                success=True,
                records_affected=affected or 0,
                records_deleted=affected or 0,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Delete failed after {self.max_retries} retries: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time
            )
    
    async def execute_upsert(
        self,
        query: str,
        params: List[Any]
    ) -> OperationResult:
        """
        Execute an UPSERT query.
        
        Args:
            query: SQL UPSERT query (INSERT ... ON CONFLICT)
            params: Query parameters
            
        Returns:
            Operation result
        """
        start_time = time.time()
        
        try:
            result = await self.db_adapter.execute_query(query, *params)
            
            # Determine if it was insert or update based on returned ID
            if result:
                return OperationResult(
                    success=True,
                    data=result,
                    records_affected=1,
                    records_created=1,  # Assume insert if ID returned
                    duration_seconds=time.time() - start_time
                )
            else:
                return OperationResult(
                    success=True,
                    records_affected=1,
                    records_updated=1,  # Assume update if no ID
                    duration_seconds=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def _build_insert_query(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Build INSERT query from record."""
        columns = list(record.keys())
        values = list(record.values())
        placeholders = [f"${i+1}" for i in range(len(values))]
        
        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
        """
        
        return {'sql': query.strip(), 'params': values}