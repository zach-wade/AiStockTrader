# File: tests/integration/test_dual_storage.py

"""
Integration tests for the V3 Dual Storage system.

Tests the complete flow of:
1. Writing to both hot and cold storage
2. Routing queries intelligently
3. Data lifecycle management
4. Historical data migration
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch

from main.data_pipeline.storage.dual_storage_writer import DualStorageWriter, DualStorageConfig, WriteMode
from main.data_pipeline.storage.storage_router import StorageRouter, QueryType
from main.data_pipeline.storage.cold_storage_query_engine import ColdStorageQueryEngine
from main.data_pipeline.storage.data_lifecycle_manager import DataLifecycleManager
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.feature_pipeline.dataloader import DataLoader, DataRequest
from main.interfaces.events import IEventBus, Event, EventType


class TestDualStorageIntegration:
    """Integration tests for dual storage system."""
    
    @pytest.fixture
    async def mock_components(self):
        """Create mock components for testing."""
        # Mock database
        mock_db = AsyncMock()
        mock_db.execute.return_value = AsyncMock()
        mock_db.execute_many.return_value = AsyncMock()
        
        # Mock archive
        mock_archive = Mock()
        mock_archive.save = Mock()
        mock_archive.load = Mock(return_value=pd.DataFrame())
        
        # Mock event bus
        mock_event_bus = AsyncMock(spec=IEventBus)
        
        # Mock repositories
        mock_repo_factory = Mock()
        mock_market_repo = AsyncMock()
        mock_market_repo.bulk_insert.return_value = 100
        mock_market_repo.get_by_filter = AsyncMock(return_value=[])
        mock_repo_factory.get_repository.return_value = mock_market_repo
        
        return {
            'db': mock_db,
            'archive': mock_archive,
            'event_bus': mock_event_bus,
            'repo_factory': mock_repo_factory,
            'market_repo': mock_market_repo
        }
    
    @pytest.mark.asyncio
    async def test_dual_storage_write_flow(self, mock_components):
        """Test writing data to both hot and cold storage."""
        # Create test configuration
        config = DualStorageConfig(
            mode=WriteMode.ASYNC,
            enable_compression=True,
            enable_deduplication=True
        )
        
        # Create writer
        writer = DualStorageWriter(
            config=config,
            event_bus=mock_components['event_bus']
        )
        
        # Inject mocks
        writer.repository_factory = mock_components['repo_factory']
        writer.archive = mock_components['archive']
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'symbol': ['AAPL'] * 10,
            'open': [100.0 + i for i in range(10)],
            'high': [101.0 + i for i in range(10)],
            'low': [99.0 + i for i in range(10)],
            'close': [100.5 + i for i in range(10)],
            'volume': [1000000 + i * 10000 for i in range(10)]
        })
        
        # Write data
        result = await writer.write(
            repository_name='market_data',
            data=test_data
        )
        
        # Verify write result
        assert result.success
        assert result.hot_storage_count == 100  # Mocked return value
        assert result.mode == WriteMode.ASYNC
        
        # Verify hot storage write
        mock_components['market_repo'].bulk_insert.assert_called_once()
        
        # Verify event was published for cold storage
        mock_components['event_bus'].publish.assert_called()
        published_event = mock_components['event_bus'].publish.call_args[0][0]
        assert published_event.event_type == EventType.DATA_WRITTEN
        assert published_event.data['repository'] == 'market_data'
        assert published_event.data['record_count'] == 10
    
    @pytest.mark.asyncio
    async def test_storage_router_query_routing(self, mock_components):
        """Test intelligent query routing based on data age."""
        # Create router
        router = StorageRouter()
        
        # Test 1: Recent data should route to hot storage
        recent_filter = QueryFilter(
            symbols=['AAPL'],
            start_date=datetime.now(timezone.utc) - timedelta(days=7),
            end_date=datetime.now(timezone.utc)
        )
        
        decision = router.route_query(recent_filter, QueryType.REAL_TIME)
        assert decision.primary_tier.value == 'hot'
        assert 'within hot threshold' in decision.reason
        
        # Test 2: Old data should route to cold storage
        old_filter = QueryFilter(
            symbols=['AAPL'],
            start_date=datetime.now(timezone.utc) - timedelta(days=90),
            end_date=datetime.now(timezone.utc) - timedelta(days=60)
        )
        
        decision = router.route_query(old_filter, QueryType.ANALYSIS)
        assert decision.primary_tier.value == 'cold'
        assert 'older than hot threshold' in decision.reason
        
        # Test 3: Mixed range should route to both
        mixed_filter = QueryFilter(
            symbols=['AAPL'],
            start_date=datetime.now(timezone.utc) - timedelta(days=45),
            end_date=datetime.now(timezone.utc)
        )
        
        decision = router.route_query(mixed_filter, QueryType.FEATURE_CALC)
        assert decision.primary_tier.value in ['hot', 'both']
    
    @pytest.mark.asyncio
    async def test_cold_storage_query_engine(self, mock_components):
        """Test querying data from cold storage."""
        # Create test data
        test_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1D'),
            'symbol': ['AAPL'] * 50 + ['GOOGL'] * 50,
            'close': [100.0 + i for i in range(100)],
            'volume': [1000000 + i * 10000 for i in range(100)]
        })
        
        # Mock archive to return test data
        mock_components['archive'].load = Mock(return_value=test_df)
        
        # Create query engine
        engine = ColdStorageQueryEngine()
        engine.archive = mock_components['archive']
        
        # Mock file finding
        with patch.object(engine, '_find_files', return_value=['/path/to/file.parquet']):
            with patch('pandas.read_parquet', return_value=test_df):
                # Execute query
                query_filter = QueryFilter(
                    symbols=['AAPL'],
                    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end_date=datetime(2024, 2, 1, tzinfo=timezone.utc)
                )
                
                result = await engine.query(
                    table_name='market_data',
                    query_filter=query_filter
                )
                
                # Verify results
                assert len(result) == 50  # Only AAPL records
                assert all(result['symbol'] == 'AAPL')
    
    @pytest.mark.asyncio
    async def test_data_lifecycle_archival(self, mock_components):
        """Test automatic archival of old data."""
        # Create lifecycle manager
        config = {'storage': {'lifecycle': {'hot_days': 30}}}
        manager = DataLifecycleManager(
            config=config,
            db_adapter=mock_components['db'],
            archive=mock_components['archive']
        )
        
        # Mock identify candidates
        with patch.object(manager, '_identify_archive_candidates') as mock_identify:
            mock_identify.return_value = [
                {'symbol': 'AAPL', 'archive_date': datetime(2024, 1, 1).date(), 'record_count': 100},
                {'symbol': 'GOOGL', 'archive_date': datetime(2024, 1, 1).date(), 'record_count': 150}
            ]
            
            # Mock archive data
            with patch.object(manager, '_archive_data') as mock_archive:
                mock_archive.return_value = 250
                
                # Mock cleanup
                with patch.object(manager, '_cleanup_hot_storage') as mock_cleanup:
                    # Run archival cycle
                    result = await manager.run_archival_cycle(dry_run=False)
                    
                    # Verify results
                    assert result['status'] == 'success'
                    assert result['records_archived'] == 250
                    
                    # Verify methods were called
                    mock_identify.assert_called_once()
                    mock_archive.assert_called_once()
                    mock_cleanup.assert_called_once_with(250)
    
    @pytest.mark.asyncio
    async def test_dataloader_with_storage_router(self):
        """Test DataLoader using StorageRouter for intelligent data loading."""
        # Create DataLoader
        loader = DataLoader()
        
        # Mock storage router
        mock_router = AsyncMock()
        mock_router.execute_query.return_value = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1D'),
            'symbol': ['AAPL'] * 10,
            'close': [100.0 + i for i in range(10)]
        })
        
        # Inject mock router
        loader.sources['market_data'].storage_router = mock_router
        
        # Create data request
        request = DataRequest(
            symbols=['AAPL'],
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 10, tzinfo=timezone.utc),
            data_types=['market_data']
        )
        
        # Load data
        result = await loader.load_data(request)
        
        # Verify results
        assert 'market_data' in result
        assert len(result['market_data']) == 10
        
        # Verify router was called
        mock_router.execute_query.assert_called_once()
        call_args = mock_router.execute_query.call_args
        assert call_args[1]['repository_name'] == 'market_data'
        assert call_args[1]['query_type'] == QueryType.FEATURE_CALC
    
    @pytest.mark.asyncio
    async def test_end_to_end_dual_storage_flow(self, mock_components):
        """Test complete end-to-end flow of dual storage system."""
        # Step 1: Write data to dual storage
        writer = DualStorageWriter(
            config=DualStorageConfig(mode=WriteMode.ASYNC),
            event_bus=mock_components['event_bus']
        )
        writer.repository_factory = mock_components['repo_factory']
        writer.archive = mock_components['archive']
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': [datetime.now(timezone.utc) - timedelta(days=i) for i in range(100)],
            'symbol': ['AAPL'] * 100,
            'close': [100.0 + i for i in range(100)]
        })
        
        # Write data
        write_result = await writer.write('market_data', test_data)
        assert write_result.success
        
        # Step 2: Query data using router
        router = StorageRouter()
        
        # Recent data query
        recent_filter = QueryFilter(
            symbols=['AAPL'],
            start_date=datetime.now(timezone.utc) - timedelta(days=10),
            end_date=datetime.now(timezone.utc)
        )
        
        recent_decision = router.route_query(recent_filter, QueryType.REAL_TIME)
        assert recent_decision.primary_tier.value == 'hot'
        
        # Historical data query
        historical_filter = QueryFilter(
            symbols=['AAPL'],
            start_date=datetime.now(timezone.utc) - timedelta(days=90),
            end_date=datetime.now(timezone.utc) - timedelta(days=60)
        )
        
        historical_decision = router.route_query(historical_filter, QueryType.ANALYSIS)
        assert historical_decision.primary_tier.value == 'cold'
        
        # Step 3: Verify lifecycle management would archive old data
        manager = DataLifecycleManager(
            config={'storage': {'lifecycle': {'hot_days': 30}}},
            db_adapter=mock_components['db'],
            archive=mock_components['archive']
        )
        
        # Old records should be identified for archival
        old_records = [row for row in test_data.to_dict('records') 
                      if (datetime.now(timezone.utc) - row['timestamp']).days > 30]
        assert len(old_records) > 0
    
    @pytest.mark.asyncio
    async def test_query_performance_metrics(self):
        """Test that query performance is tracked correctly."""
        router = StorageRouter()
        
        # Reset stats
        router.reset_stats()
        
        # Execute various queries
        filters = [
            (QueryFilter(symbols=['AAPL'], 
                        start_date=datetime.now(timezone.utc) - timedelta(days=5)), 
             QueryType.REAL_TIME),
            (QueryFilter(symbols=['GOOGL'], 
                        start_date=datetime.now(timezone.utc) - timedelta(days=90)), 
             QueryType.ANALYSIS),
            (QueryFilter(symbols=['MSFT'], 
                        start_date=datetime.now(timezone.utc) - timedelta(days=45),
                        end_date=datetime.now(timezone.utc)), 
             QueryType.FEATURE_CALC)
        ]
        
        for query_filter, query_type in filters:
            router.route_query(query_filter, query_type)
        
        # Get statistics
        stats = router.get_routing_stats()
        
        # Verify statistics
        assert stats['total_queries'] == 3
        assert stats['hot_queries'] >= 1
        assert stats['cold_queries'] >= 0
        assert stats['hot_percentage'] + stats['cold_percentage'] + stats['both_percentage'] == 100