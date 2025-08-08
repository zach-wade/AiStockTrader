"""
Unit tests for BaseRepository class.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Optional

from main.data_pipeline.storage.repositories.base_repository import BaseRepository
from main.data_pipeline.storage.repositories.repository_types import (
    RepositoryConfig,
    OperationResult,
    QueryFilter,
    ValidationLevel
)
from main.utils.database import TransactionStrategy


class ConcreteRepository(BaseRepository):
    """Concrete implementation for testing."""
    
    def get_required_fields(self) -> List[str]:
        return ['id', 'symbol', 'timestamp', 'value']
    
    def prepare_record(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Simple preparation - just ensure timestamp is datetime
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return data
    
    async def validate_specific_fields(self, record: Dict[str, Any]) -> tuple[bool, List[str]]:
        errors = []
        if 'value' in record and record['value'] < 0:
            errors.append("Value cannot be negative")
        return len(errors) == 0, errors
    
    def get_table_name(self) -> str:
        """Returns the database table name for testing."""
        return "test_table"
    
    def get_unique_constraint_name(self) -> Optional[str]:
        """Returns the unique constraint name for testing."""
        return "test_pkey"


class TestModel:
    """Mock database model."""
    __tablename__ = 'test_table'
    
    # Create a proper mock for __table__ with column support
    __table__ = Mock()
    
    # Mock the columns collection to support 'in' operator
    mock_columns = Mock()
    mock_columns.__contains__ = Mock(side_effect=lambda x: x in ['id', 'symbol', 'timestamp', 'value'])
    
    # Create mock column objects
    mock_columns.id = Mock(name='id_column')
    mock_columns.symbol = Mock(name='symbol_column')
    mock_columns.timestamp = Mock(name='timestamp_column')
    mock_columns.value = Mock(name='value_column')
    
    __table__.c = mock_columns
    __table__.columns = ['id', 'symbol', 'timestamp', 'value']


class TestBaseRepository:
    """Test BaseRepository functionality."""
    
    @pytest.fixture
    def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = Mock()
        adapter.run_sync = AsyncMock()
        return adapter
    
    @pytest.fixture
    def mock_config(self):
        """Create mock repository configuration."""
        return RepositoryConfig(
            batch_size=100,
            max_retries=2,
            retry_delay=0.1,
            transaction_strategy=TransactionStrategy.BATCH_WITH_FALLBACK,
            validation_level=ValidationLevel.LENIENT,
            enable_caching=False,
            enable_metrics=False,
            log_operations=False
        )
    
    @pytest_asyncio.fixture
    async def repository(self, mock_db_adapter, mock_config):
        """Create test repository instance."""
        with patch('main.data_pipeline.storage.repositories.base_repository.get_global_cache', return_value=None):
            repo = ConcreteRepository(
                db_adapter=mock_db_adapter,
                model_class=TestModel,
                config=mock_config
            )
            
            # Mock the query builder to avoid SQLAlchemy table issues
            repo._query_builder = Mock()
            repo._query_builder.build_filtered_select_statement = Mock(return_value=Mock())
            
            return repo
    
    @pytest.mark.asyncio

    async def test_repository_initialization(self, mock_db_adapter, mock_config):
        """Test repository initialization."""
        with patch('main.data_pipeline.storage.repositories.base_repository.get_global_cache', return_value=None):
            repo = ConcreteRepository(
                db_adapter=mock_db_adapter,
                model_class=TestModel,
                config=mock_config
            )
            
            assert repo.db_adapter == mock_db_adapter
            assert repo.model_class == TestModel
            assert repo.config == mock_config
            assert repo._repo_name == 'ConcreteRepository'
    
    @pytest.mark.asyncio

    async def test_bulk_upsert_success(self, repository, mock_db_adapter):
        """Test successful bulk upsert operation."""
        # Prepare test data
        records = [
            {'id': 1, 'symbol': 'AAPL', 'timestamp': datetime.now(timezone.utc), 'value': 150.0},
            {'id': 2, 'symbol': 'MSFT', 'timestamp': datetime.now(timezone.utc), 'value': 300.0}
        ]
        
        # Mock crud executor behavior
        mock_db_adapter.run_sync.return_value = 2  # 2 records processed
        
        # Execute
        result = await repository.bulk_upsert(records, constraint_name='test_pkey')
        
        # Verify
        assert result.success is True
        assert result.total_records == 2
        assert result.processed_records == 2
        assert result.failed_records == 0
        assert result.operation_type == 'bulk_upsert'
    
    @pytest.mark.asyncio

    async def test_bulk_upsert_with_validation_errors(self, mock_db_adapter):
        """Test bulk upsert with validation errors."""
        # Create repository with STRICT validation for this test
        strict_config = RepositoryConfig(
            batch_size=100,
            max_retries=2,
            retry_delay=0.1,
            transaction_strategy=TransactionStrategy.BATCH_WITH_FALLBACK,
            validation_level=ValidationLevel.STRICT,  # Use STRICT to filter invalid records
            enable_caching=False,
            enable_metrics=False,
            log_operations=False
        )
        
        with patch('main.data_pipeline.storage.repositories.base_repository.get_global_cache', return_value=None):
            repository = ConcreteRepository(
                db_adapter=mock_db_adapter,
                model_class=TestModel,
                config=strict_config
            )
        
        # Prepare test data with invalid record
        records = [
            {'id': 1, 'symbol': 'AAPL', 'timestamp': datetime.now(timezone.utc), 'value': 150.0},
            {'id': 2, 'symbol': 'MSFT', 'timestamp': datetime.now(timezone.utc), 'value': -50.0}  # Invalid
        ]
        
        # Mock crud executor behavior - 1 record processed (invalid one filtered out)
        mock_db_adapter.run_sync.return_value = 1
        
        # Execute
        result = await repository.bulk_upsert(records, constraint_name='test_pkey')
        
        # Verify - validation should filter out invalid record
        assert result.total_records == 2
        assert result.failed_records == 1
        assert result.processed_records == 1  # Only valid record processed
    
    @pytest.mark.asyncio

    async def test_get_by_symbol(self, repository, mock_db_adapter):
        """Test get_by_symbol method."""
        symbol = 'AAPL'
        mock_records = [
            {'id': 1, 'symbol': 'AAPL', 'timestamp': datetime.now(timezone.utc), 'value': 150.0},
            {'id': 2, 'symbol': 'AAPL', 'timestamp': datetime.now(timezone.utc), 'value': 151.0}
        ]
        
        # Mock database response
        mock_db_adapter.run_sync.return_value = mock_records
        
        # Execute
        results = await repository.get_by_symbol(symbol)
        
        # Verify
        assert len(results) == 2
        assert all(r['symbol'] == 'AAPL' for r in results)
        mock_db_adapter.run_sync.assert_called_once()
    
    @pytest.mark.asyncio

    async def test_get_by_symbols(self, repository, mock_db_adapter):
        """Test get_by_symbols method."""
        symbols = ['AAPL', 'MSFT']
        mock_records = [
            {'id': 1, 'symbol': 'AAPL', 'timestamp': datetime.now(timezone.utc), 'value': 150.0},
            {'id': 2, 'symbol': 'MSFT', 'timestamp': datetime.now(timezone.utc), 'value': 300.0}
        ]
        
        # Mock database response
        mock_db_adapter.run_sync.return_value = mock_records
        
        # Execute
        results = await repository.get_by_symbols(symbols)
        
        # Verify
        assert len(results) == 2
        assert {r['symbol'] for r in results} == {'AAPL', 'MSFT'}
    
    @pytest.mark.asyncio

    async def test_get_by_date_range(self, repository, mock_db_adapter):
        """Test get_by_date_range method."""
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)
        
        mock_records = [
            {'id': 1, 'symbol': 'AAPL', 'timestamp': start_date + timedelta(days=1), 'value': 150.0},
            {'id': 2, 'symbol': 'AAPL', 'timestamp': start_date + timedelta(days=3), 'value': 151.0}
        ]
        
        # Mock database response
        mock_db_adapter.run_sync.return_value = mock_records
        
        # Execute
        results = await repository.get_by_date_range('AAPL', start_date, end_date)
        
        # Verify
        assert len(results) == 2
        mock_db_adapter.run_sync.assert_called_once()
    
    @pytest.mark.asyncio

    async def test_query_with_filters(self, repository, mock_db_adapter):
        """Test query method with various filters."""
        query_filter = QueryFilter(
            symbols=['AAPL', 'MSFT'],
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc),
            limit=10,
            order_by='timestamp',
            order_desc=True
        )
        
        mock_records = [
            {'id': 1, 'symbol': 'AAPL', 'timestamp': datetime.now(timezone.utc), 'value': 150.0}
        ]
        
        # Mock database response
        mock_db_adapter.run_sync.return_value = mock_records
        
        # Execute
        results = await repository.query(query_filter)
        
        # Verify
        assert len(results) == 1
        mock_db_adapter.run_sync.assert_called_once()
    
    @pytest.mark.asyncio

    async def test_delete_by_symbol(self, repository, mock_db_adapter):
        """Test delete_by_symbol method."""
        symbol = 'AAPL'
        
        # Mock delete operation returning row count
        mock_db_adapter.run_sync.return_value = 5
        
        # Execute
        count = await repository.delete_by_symbol(symbol)
        
        # Verify
        assert count == 5
        mock_db_adapter.run_sync.assert_called_once()
    
    @pytest.mark.asyncio

    async def test_delete_old_records(self, repository, mock_db_adapter):
        """Test delete_old_records method."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Mock delete operation returning row count
        mock_db_adapter.run_sync.return_value = 100
        
        # Execute
        count = await repository.delete_old_records(cutoff_date)
        
        # Verify
        assert count == 100
        mock_db_adapter.run_sync.assert_called_once()
    
    @pytest.mark.asyncio

    async def test_get_record_count(self, repository, mock_db_adapter):
        """Test get_record_count method."""
        # Mock count query
        mock_db_adapter.run_sync.return_value = 1500
        
        # Execute
        count = await repository.get_record_count()
        
        # Verify
        assert count == 1500
        mock_db_adapter.run_sync.assert_called_once()
    
    @pytest.mark.asyncio

    async def test_get_latest_timestamp(self, repository, mock_db_adapter):
        """Test get_latest_timestamp method."""
        latest_time = datetime.now(timezone.utc)
        
        # Mock query result
        mock_db_adapter.run_sync.return_value = latest_time
        
        # Execute
        result = await repository.get_latest_timestamp('AAPL')
        
        # Verify
        assert result == latest_time
        mock_db_adapter.run_sync.assert_called_once()
    
    @pytest.mark.asyncio

    async def test_health_check(self, repository, mock_db_adapter):
        """Test health_check method."""
        # Mock the repository methods that health_check calls
        repository.get_record_count = AsyncMock(return_value=1000)
        repository.get_symbol_count = AsyncMock(return_value=5)
        repository.get_latest = AsyncMock(return_value=[
            {'id': 1, 'symbol': 'AAPL', 'timestamp': datetime.now(timezone.utc), 'value': 150.0}
        ])
        repository.get_data_range = AsyncMock(return_value=(
            datetime.now(timezone.utc) - timedelta(days=30),
            datetime.now(timezone.utc)
        ))
        
        # Execute
        health = await repository.health_check()
        
        # Verify
        assert health['status'] == 'healthy'
        assert health['total_records'] == 1000
        assert health['unique_symbols'] == 5
        assert health['oldest_record'] is not None
        assert health['newest_record'] is not None
        assert len(health['errors']) == 0
    
    @pytest.mark.asyncio

    async def test_health_check_failure(self, repository, mock_db_adapter):
        """Test health_check with database error."""
        # Mock database error
        mock_db_adapter.run_sync.side_effect = Exception("Database connection failed")
        
        # Execute
        health = await repository.health_check()
        
        # Verify
        assert health['status'] == 'unhealthy'
        assert 'errors' in health
        assert len(health['errors']) > 0
        assert 'Database connection failed' in health['errors'][0]
        
        # TODO: Standardize error reporting across the codebase
        # Currently using 'errors' (list) but should consider consistent convention
    
    @pytest.mark.asyncio

    async def test_get_metrics(self, repository):
        """Test get_metrics method."""
        # Execute some operations to generate metrics
        repository._metrics_data['operations_count'] = 10
        repository._metrics_data['success_count'] = 8
        repository._metrics_data['error_count'] = 2
        
        # Get metrics
        metrics = repository.get_metrics()
        
        # Verify
        assert metrics['operations_count'] == 10
        assert metrics['success_count'] == 8
        assert metrics['error_count'] == 2
        assert 'config' in metrics
    
    @pytest.mark.asyncio

    async def test_record_validation(self, repository):
        """Test record validation logic."""
        # Valid record
        valid_record = {
            'id': 1,
            'symbol': 'AAPL',
            'timestamp': datetime.now(timezone.utc),
            'value': 150.0
        }
        
        # Invalid record (negative value)
        invalid_record = {
            'id': 2,
            'symbol': 'MSFT',
            'timestamp': datetime.now(timezone.utc),
            'value': -50.0
        }
        
        # Test valid record
        is_valid, errors = await repository.validate_specific_fields(valid_record)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test invalid record
        is_valid, errors = await repository.validate_specific_fields(invalid_record)
        assert is_valid is False
        assert len(errors) > 0
        assert "Value cannot be negative" in errors[0]
    
    @pytest.mark.asyncio

    async def test_update_repository_metrics(self, repository):
        """Test metrics update functionality."""
        # Create operation result
        result = OperationResult(
            success=True,
            total_records=100,
            processed_records=95,
            failed_records=5,
            duration_seconds=2.5,
            operation_type='bulk_upsert'
        )
        
        # Update metrics
        repository._update_repository_metrics(result)
        
        # Verify metrics were updated
        assert repository._metrics_data['operations_count'] == 1
        assert repository._metrics_data['success_count'] == 1
        assert repository._metrics_data['total_duration'] == 2.5
        assert repository._metrics_data['average_duration'] == 2.5