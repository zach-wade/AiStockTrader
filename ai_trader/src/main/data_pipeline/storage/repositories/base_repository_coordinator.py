"""
Base Repository Coordinator

Main entry point for repository operations, implementing IRepository interface.
Composes specialized services for different types of operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
import pandas as pd

from main.interfaces.repositories import IRepository
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import (
    RepositoryConfig,
    QueryFilter,
    OperationResult
)
from .repository_patterns import (
    RepositoryMixin,
    DualStorageMixin,
    CacheMixin,
    MetricsMixin
)
from .repository_query_builder import RepositoryQueryBuilder
from .repository_core_operations import RepositoryCoreOperations
from .repository_query_processor import RepositoryQueryProcessor
from main.utils.core import get_logger

logger = get_logger(__name__)


class BaseRepositoryCoordinator(
    IRepository,
    RepositoryMixin,
    DualStorageMixin,
    CacheMixin,
    MetricsMixin,
    ABC
):
    """
    Enhanced base repository coordinator providing common database operations.
    
    This class orchestrates interactions between specialized services:
    - RepositoryQueryBuilder: SQL query construction
    - RepositoryCoreOperations: Basic CRUD operations
    - RepositoryQueryProcessor: Complex filtered operations
    """
    
    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        model_class: Type,
        config: Optional[RepositoryConfig] = None
    ):
        """
        Initialize the base repository coordinator.
        
        Args:
            db_adapter: Database adapter for async operations
            model_class: SQLAlchemy model class
            config: Optional repository configuration
        """
        self.db_adapter = db_adapter
        self.model_class = model_class
        self.config = config or RepositoryConfig()
        
        # Initialize mixins
        super().__init__()
        
        # Table name from model
        self.table_name = getattr(model_class, '__tablename__', model_class.__name__.lower())
        
        # Initialize composed services
        self.query_builder = RepositoryQueryBuilder(self.table_name)
        
        self.core_operations = RepositoryCoreOperations(
            db_adapter=db_adapter,
            query_builder=self.query_builder,
            table_name=self.table_name,
            validate_record_fn=self.validate_record,
            cache_mixin=self,
            metrics_mixin=self
        )
        
        self.query_processor = RepositoryQueryProcessor(
            db_adapter=db_adapter,
            query_builder=self.query_builder,
            table_name=self.table_name,
            metrics_mixin=self
        )
        
        logger.info(f"Initialized {self.__class__.__name__} for table {self.table_name}")
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get list of required fields for this repository."""
        pass
    
    @abstractmethod
    def validate_record(self, record: Dict[str, Any]) -> List[str]:
        """
        Validate a record.
        
        Args:
            record: Record to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    # IRepository interface implementation - delegate to composed services
    async def get_by_id(self, record_id: Any) -> Optional[Dict[str, Any]]:
        """Get a single record by ID."""
        return await self.core_operations.get_by_id(record_id)
    
    async def get_by_filter(self, query_filter: QueryFilter) -> pd.DataFrame:
        """Get records matching filter criteria."""
        return await self.query_processor.get_by_filter(query_filter)
    
    async def create(self, data: Union[Dict[str, Any], pd.DataFrame]) -> OperationResult:
        """Create new records."""
        return await self.core_operations.create(data)
    
    async def update(self, record_id: Any, data: Dict[str, Any]) -> OperationResult:
        """Update an existing record."""
        return await self.core_operations.update(record_id, data)
    
    async def upsert(self, data: Union[Dict[str, Any], pd.DataFrame]) -> OperationResult:
        """Insert or update records."""
        # Use core operations for single record, query processor for multiple records
        if isinstance(data, pd.DataFrame) or (isinstance(data, list) and len(data) > 1):
            # Handle multiple records - need to implement bulk upsert logic
            return await self._handle_bulk_upsert(data)
        else:
            # Single record upsert using core operations
            return await self._handle_single_upsert(data)
    
    async def delete(self, record_id: Any) -> OperationResult:
        """Delete a record."""
        return await self.core_operations.delete(record_id)
    
    async def delete_by_filter(self, query_filter: QueryFilter) -> OperationResult:
        """Delete records matching filter criteria."""
        return await self.query_processor.delete_by_filter(query_filter)
    
    async def count(self, query_filter: Optional[QueryFilter] = None) -> int:
        """Count records matching filter criteria."""
        if query_filter:
            return await self.query_processor.count_by_filter(query_filter)
        else:
            return await self.core_operations.count_all()
    
    async def exists(self, record_id: Any) -> bool:
        """Check if a record exists."""
        return await self.core_operations.exists(record_id)
    
    # Additional methods for enhanced functionality
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
        return await self.query_processor.get_aggregated_data(
            query_filter, aggregation, group_by
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
        return await self.query_processor.get_distinct_values(column, query_filter)
    
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
        return await self.query_processor.bulk_upsert(records, conflict_columns)
    
    # Private helper methods
    async def _handle_single_upsert(self, data: Dict[str, Any]) -> OperationResult:
        """Handle single record upsert logic."""
        # Check if record exists
        record_id = data.get('id') or data.get(f'{self.table_name}_id')
        
        if record_id and await self.core_operations.exists(record_id):
            # Update existing
            return await self.core_operations.update(record_id, data)
        else:
            # Create new
            return await self.core_operations.create(data)
    
    async def _handle_bulk_upsert(
        self, 
        data: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> OperationResult:
        """Handle bulk upsert logic."""
        # Convert DataFrame to list of dicts if needed
        if isinstance(data, pd.DataFrame):
            records = data.to_dict('records')
        else:
            records = data if isinstance(data, list) else [data]
        
        # For bulk operations, determine conflict columns from required fields
        required_fields = self.get_required_fields()
        
        # Use common conflict columns (id, symbol+timestamp, etc.)
        conflict_columns = []
        if 'id' in required_fields:
            conflict_columns = ['id']
        elif 'symbol' in required_fields and 'timestamp' in required_fields:
            conflict_columns = ['symbol', 'timestamp']
        else:
            # Fall back to individual upserts if we can't determine conflict columns
            return await self._fallback_individual_upserts(records)
        
        return await self.query_processor.bulk_upsert(records, conflict_columns)
    
    async def _fallback_individual_upserts(
        self, 
        records: List[Dict[str, Any]]
    ) -> OperationResult:
        """Fallback to individual upserts when bulk upsert isn't suitable."""
        created = 0
        updated = 0
        errors = []
        
        for record in records:
            try:
                result = await self._handle_single_upsert(record)
                if result.success:
                    created += result.records_created or 0
                    updated += result.records_updated or 0
                else:
                    errors.append(result.error)
            except Exception as e:
                errors.append(str(e))
        
        return OperationResult(
            success=len(errors) == 0,
            records_affected=created + updated,
            records_created=created,
            records_updated=updated,
            error="; ".join(errors) if errors else None
        )