"""
Repository Query Processor Service

Handles complex query operations including filtering, bulk operations, and aggregations.
Processes queries that involve multiple records or complex conditions.
"""

import time
from typing import Any, Dict, List, Optional
import pandas as pd

from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import QueryFilter, OperationResult
from main.utils.core import get_logger
from .repository_query_builder import RepositoryQueryBuilder

logger = get_logger(__name__)


class RepositoryQueryProcessor:
    """
    Service for complex repository query operations.
    
    Handles bulk operations, filtered queries, and complex aggregations.
    Uses query builder for SQL construction and database adapter for execution.
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        query_builder: RepositoryQueryBuilder,
        table_name: str,
        metrics_mixin: Any = None
    ):
        """
        Initialize query processor service.
        
        Args:
            db_adapter: Database adapter for async operations
            query_builder: Query builder for SQL construction
            table_name: Database table name
            metrics_mixin: Optional metrics mixin for recording metrics
        """
        self.db_adapter = db_adapter
        self.query_builder = query_builder
        self.table_name = table_name
        self.metrics_mixin = metrics_mixin
        self.logger = get_logger(__name__)
    
    async def get_by_filter(self, query_filter: QueryFilter) -> pd.DataFrame:
        """
        Get records matching filter criteria.
        
        Args:
            query_filter: Filter criteria for the query
            
        Returns:
            DataFrame with matching records
        """
        start_time = time.time()
        
        try:
            # Build query
            query, params = self.query_builder.build_select_query(query_filter)
            
            # Execute query
            results = await self.db_adapter.fetch_all(query, *params)
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()
            
            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                'get_by_filter', duration, success=True, records=len(df)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting records by filter: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                'get_by_filter', duration, success=False
            )
            raise
    
    async def count_by_filter(self, query_filter: QueryFilter) -> int:
        """
        Count records matching filter criteria.
        
        Args:
            query_filter: Filter criteria for counting
            
        Returns:
            Number of matching records
        """
        try:
            query, params = self.query_builder.build_count_query(query_filter)
            result = await self.db_adapter.fetch_one(query, *params)
            return result.get('count', 0) if result else 0
            
        except Exception as e:
            logger.error(f"Error counting records by filter: {e}")
            return 0
    
    async def delete_by_filter(self, query_filter: QueryFilter) -> OperationResult:
        """
        Delete records matching filter criteria.
        
        Args:
            query_filter: Filter criteria for deletion
            
        Returns:
            OperationResult with deletion details
        """
        start_time = time.time()
        
        try:
            # Build delete query
            query, params = self.query_builder.build_delete_by_filter_query(query_filter)
            
            # Execute deletion
            affected = await self.db_adapter.execute_query(query, *params)
            
            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                'delete_by_filter', duration, success=True, records=affected or 0
            )
            
            return OperationResult(
                success=True,
                records_affected=affected or 0,
                records_deleted=affected or 0,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Error deleting by filter: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                'delete_by_filter', duration, success=False
            )
            return OperationResult(
                success=False,
                error=str(e),
                duration_seconds=duration
            )
    
    async def get_aggregated_data(
        self,
        query_filter: QueryFilter,
        aggregation: str,
        group_by: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get aggregated data with grouping.
        
        Args:
            query_filter: Filter criteria for the query
            aggregation: Aggregation function (COUNT, SUM, AVG, etc.)
            group_by: Optional columns to group by
            
        Returns:
            DataFrame with aggregated results
        """
        start_time = time.time()
        
        try:
            # Build base query conditions
            base_query, params = self.query_builder.build_select_query(query_filter)
            
            # Extract WHERE clause from base query
            where_start = base_query.find(" WHERE ")
            where_clause = base_query[where_start:] if where_start != -1 else ""
            
            # Remove ORDER BY and LIMIT from where clause for aggregation
            for clause in [" ORDER BY ", " LIMIT "]:
                if clause in where_clause:
                    where_clause = where_clause[:where_clause.find(clause)]
            
            # Build aggregation query
            if group_by:
                select_clause = f"{', '.join(group_by)}, {aggregation}(*) as result"
                group_clause = f" GROUP BY {', '.join(group_by)}"
            else:
                select_clause = f"{aggregation}(*) as result"
                group_clause = ""
            
            query = f"SELECT {select_clause} FROM {self.table_name}{where_clause}{group_clause}"
            
            # Execute query
            results = await self.db_adapter.fetch_all(query, *params)
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()
            
            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                'get_aggregated_data', duration, success=True, records=len(df)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting aggregated data: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                'get_aggregated_data', duration, success=False
            )
            raise
    
    async def bulk_upsert(
        self,
        records: List[Dict[str, Any]],
        conflict_columns: List[str]
    ) -> OperationResult:
        """
        Perform bulk upsert operation.
        
        Args:
            records: List of records to upsert
            conflict_columns: Columns to use for conflict resolution
            
        Returns:
            OperationResult with upsert details
        """
        start_time = time.time()
        
        try:
            if not records:
                return OperationResult(
                    success=True,
                    records_affected=0,
                    duration_seconds=0
                )
            
            # Build bulk upsert query
            columns = list(records[0].keys())
            values_placeholders = []
            all_values = []
            
            for i, record in enumerate(records):
                record_placeholders = []
                for j, column in enumerate(columns):
                    placeholder_num = i * len(columns) + j + 1
                    record_placeholders.append(f"${placeholder_num}")
                    all_values.append(record.get(column))
                
                values_placeholders.append(f"({', '.join(record_placeholders)})")
            
            # Build update clause for ON CONFLICT
            update_clauses = [f"{col} = EXCLUDED.{col}" for col in columns if col not in conflict_columns]
            
            query = f"""
                INSERT INTO {self.table_name} ({', '.join(columns)})
                VALUES {', '.join(values_placeholders)}
                ON CONFLICT ({', '.join(conflict_columns)}) 
                DO UPDATE SET {', '.join(update_clauses)}
            """
            
            # Execute bulk upsert
            affected = await self.db_adapter.execute_query(query, *all_values)
            
            # Record metrics
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                'bulk_upsert', duration, success=True, records=len(records)
            )
            
            return OperationResult(
                success=True,
                records_affected=len(records),
                records_upserted=len(records),
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Error in bulk upsert: {e}")
            duration = time.time() - start_time
            await self._record_operation_metric_if_enabled(
                'bulk_upsert', duration, success=False
            )
            return OperationResult(
                success=False,
                error=str(e),
                duration_seconds=duration
            )
    
    async def get_distinct_values(
        self,
        column: str,
        query_filter: Optional[QueryFilter] = None
    ) -> List[Any]:
        """
        Get distinct values from a column.
        
        Args:
            column: Column name to get distinct values from
            query_filter: Optional filter criteria
            
        Returns:
            List of distinct values
        """
        try:
            if query_filter:
                base_query, params = self.query_builder.build_select_query(query_filter)
                # Extract WHERE clause
                where_start = base_query.find(" WHERE ")
                where_clause = base_query[where_start:] if where_start != -1 else ""
                # Remove ORDER BY and LIMIT
                for clause in [" ORDER BY ", " LIMIT "]:
                    if clause in where_clause:
                        where_clause = where_clause[:where_clause.find(clause)]
            else:
                where_clause = ""
                params = []
            
            query = f"SELECT DISTINCT {column} FROM {self.table_name}{where_clause} ORDER BY {column}"
            
            results = await self.db_adapter.fetch_all(query, *params)
            return [dict(r)[column] for r in results if dict(r)[column] is not None]
            
        except Exception as e:
            logger.error(f"Error getting distinct values for {column}: {e}")
            return []
    
    # Metrics helper method
    async def _record_operation_metric_if_enabled(
        self, operation: str, duration: float, success: bool, records: int = 0
    ) -> None:
        """Record operation metric if metrics are enabled."""
        if not self.metrics_mixin:
            return
        
        await self.metrics_mixin._record_operation_metric(
            operation, duration, success, records
        )