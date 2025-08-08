"""
Integration tests for complete data pipeline flow.

Tests the entire pipeline from ingestion through validation, processing,
and storage to ensure all components work together correctly.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from main.data_pipeline.orchestrator import DataPipelineOrchestrator
from main.data_pipeline.ingestion.orchestrator import IngestionOrchestrator
from main.data_pipeline.processing.manager import ProcessingManager
from main.data_pipeline.validation.validation_pipeline import ValidationPipeline, ValidationStage
from main.data_pipeline.types import DataPipelineStatus, DataType
from main.data_pipeline.config_adapter import DataPipelineConfig

from tests.fixtures.data_pipeline.mock_clients import create_mock_clients_dict
from tests.fixtures.data_pipeline.test_configs import get_test_config
from tests.fixtures.data_pipeline.database_fixtures import MockAsyncDatabase, MockArchive


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompletePipelineFlow:
    """Test complete data pipeline flow scenarios."""
    
    @pytest.fixture
    def test_config(self):
        """Get test configuration."""
        return get_test_config()
    
    @pytest.fixture
    def mock_clients(self, test_config):
        """Create mock data source clients."""
        return create_mock_clients_dict(test_config)
    
    @pytest.fixture
    async def mock_db_adapter(self):
        """Create mock database adapter."""
        adapter = MockAsyncDatabase()
        yield adapter
        await adapter.close()
    
    @pytest.fixture
    def mock_archive(self):
        """Create mock archive."""
        return MockArchive()
    
    @pytest.fixture
    async def ingestion_orchestrator(self, mock_clients, mock_archive):
        """Create ingestion orchestrator with mocks."""
        with patch('main.data_pipeline.ingestion.orchestrator.get_archive', return_value=mock_archive):
            orchestrator = IngestionOrchestrator(mock_clients)
            yield orchestrator
    
    @pytest.fixture
    async def processing_manager(self, test_config):
        """Create processing manager."""
        manager = ProcessingManager(test_config)
        return manager
    
    @pytest.fixture
    async def validation_pipeline(self):
        """Create validation pipeline."""
        pipeline = ValidationPipeline()
        return pipeline
    
    async def test_market_data_full_flow(
        self,
        test_config,
        ingestion_orchestrator,
        processing_manager,
        validation_pipeline,
        mock_db_adapter,
        mock_archive
    ):
        """Test complete flow for market data processing."""
        # Setup
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        start_date = datetime.now(timezone.utc) - timedelta(days=7)
        end_date = datetime.now(timezone.utc)
        
        # Step 1: Ingest data
        ingestion_result = await ingestion_orchestrator.run_ingestion(
            data_type='market_data',
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe='1day'
        )
        
        # Verify ingestion
        assert ingestion_result.status == DataPipelineStatus.SUCCESS
        assert ingestion_result.records_processed > 0
        assert len(ingestion_result.errors) == 0
        
        # Verify data was archived
        assert mock_archive.archive_count > 0
        
        # Step 2: Validate ingested data
        # Get the ingested data from the result
        ingested_data = ingestion_result.metadata.get('data', [])
        if not ingested_data and hasattr(ingestion_result, 'data'):
            ingested_data = ingestion_result.data
        
        if ingested_data:
            validation_result = await validation_pipeline.validate_ingest(
                data=ingested_data,
                source='alpaca',
                data_type='market_data'
            )
            
            # Verify validation
            assert validation_result.passed is True
            assert validation_result.total_records == len(ingested_data)
            assert validation_result.valid_records == validation_result.total_records
        
        # Step 3: Process data (transform and extract features)
        # Mock the retrieval of data from archive
        mock_raw_records = []
        for symbol in symbols:
            mock_raw_records.extend([
                {
                    'symbol': symbol,
                    'timestamp': start_date + timedelta(days=i),
                    'open': 150.0 + i,
                    'high': 152.0 + i,
                    'low': 149.0 + i,
                    'close': 151.0 + i,
                    'volume': 1000000 + i * 10000
                }
                for i in range(7)
            ])
        
        # Process the data
        processing_result = await processing_manager.process_data(
            data_type='market_data',
            raw_records=mock_raw_records
        )
        
        # Verify processing
        assert processing_result.status == DataPipelineStatus.SUCCESS
        assert processing_result.records_processed > 0
        
        # Step 4: Validate processed data
        if hasattr(processing_result, 'data'):
            processed_data = processing_result.data
            post_validation_result = await validation_pipeline.validate_post_etl(
                data=processed_data,
                data_type='market_data'
            )
            
            assert post_validation_result.passed is True
    
    async def test_multi_source_ingestion(
        self,
        test_config,
        mock_clients,
        mock_archive
    ):
        """Test ingesting from multiple sources with prioritization."""
        # Create orchestrator with multiple sources
        with patch('main.data_pipeline.ingestion.orchestrator.get_archive', return_value=mock_archive):
            orchestrator = IngestionOrchestrator(mock_clients)
            
            # Configure one client to fail
            mock_clients['alpaca_market'].fail_on_fetch = True
            
            # Run ingestion - should fallback to polygon
            result = await orchestrator.run_ingestion(
                data_type='market_data',
                symbols=['AAPL'],
                start_date=datetime.now(timezone.utc) - timedelta(days=1),
                end_date=datetime.now(timezone.utc),
                timeframe='1hour'
            )
            
            # Verify fallback worked
            assert result.status in [DataPipelineStatus.SUCCESS, DataPipelineStatus.PARTIAL]
            assert result.records_processed > 0
            assert len(result.errors) >= 1  # Should have error from alpaca
    
    async def test_validation_failure_handling(
        self,
        validation_pipeline
    ):
        """Test handling of validation failures."""
        # Create data with validation issues
        invalid_data = [
            {
                'symbol': 'AAPL',
                'timestamp': datetime.now(timezone.utc),
                'open': 150.0,
                'high': 140.0,  # Invalid: high < open
                'low': 160.0,   # Invalid: low > high
                'close': 155.0,
                'volume': -1000  # Invalid: negative volume
            }
        ]
        
        # Run validation
        result = await validation_pipeline.validate_ingest(
            data=invalid_data,
            source='test',
            data_type='market_data'
        )
        
        # Verify validation caught the issues
        assert result.passed is False
        assert result.invalid_records > 0
        assert len(result.errors) > 0
    
    async def test_news_data_flow(
        self,
        ingestion_orchestrator,
        validation_pipeline,
        mock_archive
    ):
        """Test news data processing flow."""
        # Ingest news data
        symbols = ['AAPL', 'TSLA']
        result = await ingestion_orchestrator.run_ingestion(
            data_type='news',
            symbols=symbols,
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc)
        )
        
        # Verify ingestion
        assert result.status == DataPipelineStatus.SUCCESS
        assert result.records_processed > 0
        
        # Verify news items were created for each symbol
        archived_keys = mock_archive.list_raw_data_keys()
        news_keys = [k for k in archived_keys if 'news' in k]
        assert len(news_keys) > 0
    
    async def test_concurrent_data_type_processing(
        self,
        ingestion_orchestrator,
        mock_archive
    ):
        """Test processing multiple data types concurrently."""
        symbols = ['AAPL', 'MSFT']
        start_date = datetime.now(timezone.utc) - timedelta(days=1)
        end_date = datetime.now(timezone.utc)
        
        # Run multiple data types concurrently
        tasks = [
            ingestion_orchestrator.run_ingestion('market_data', symbols, start_date, end_date),
            ingestion_orchestrator.run_ingestion('news', symbols, start_date, end_date),
            ingestion_orchestrator.run_ingestion('social_sentiment', symbols, start_date, end_date)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        for result in results:
            assert result.status == DataPipelineStatus.SUCCESS
            assert result.records_processed > 0
        
        # Verify different data types were archived
        archived_keys = mock_archive.list_raw_data_keys()
        assert any('market_data' in k for k in archived_keys)
        assert any('news' in k for k in archived_keys)
        assert any('social_sentiment' in k for k in archived_keys)
    
    async def test_error_recovery_flow(
        self,
        test_config,
        mock_clients,
        mock_archive
    ):
        """Test error recovery and resilience."""
        # Create orchestrator with resilience config
        with patch('main.data_pipeline.ingestion.orchestrator.get_archive', return_value=mock_archive):
            orchestrator = IngestionOrchestrator(mock_clients)
            
            # Make client fail intermittently
            call_count = 0
            original_fetch = mock_clients['alpaca_market'].fetch_data
            
            async def intermittent_fetch(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Temporary failure")
                return await original_fetch(*args, **kwargs)
            
            mock_clients['alpaca_market'].fetch_data = intermittent_fetch
            
            # Run ingestion - should retry and succeed
            result = await orchestrator.run_ingestion(
                data_type='market_data',
                symbols=['AAPL'],
                start_date=datetime.now(timezone.utc) - timedelta(hours=1),
                end_date=datetime.now(timezone.utc)
            )
            
            # Verify recovery worked
            assert result.status == DataPipelineStatus.SUCCESS
            assert call_count > 1  # Should have retried
    
    async def test_batch_processing_performance(
        self,
        ingestion_orchestrator,
        mock_archive
    ):
        """Test batch processing with large symbol sets."""
        # Create large symbol list
        symbols = [f'SYM{i:04d}' for i in range(100)]
        
        # Run ingestion with batching
        result = await ingestion_orchestrator.run_ingestion(
            data_type='market_data',
            symbols=symbols,
            start_date=datetime.now(timezone.utc) - timedelta(hours=1),
            end_date=datetime.now(timezone.utc),
            timeframe='1hour'
        )
        
        # Verify batch processing worked
        assert result.status == DataPipelineStatus.SUCCESS
        assert result.records_processed > 0
        assert 'symbols_count' in result.metadata
        assert result.metadata['symbols_count'] == 100
    
    async def test_end_to_end_with_storage(
        self,
        test_config,
        mock_clients,
        mock_db_adapter,
        mock_archive
    ):
        """Test complete end-to-end flow including database storage."""
        # Mock repository
        mock_repo = Mock()
        mock_repo.bulk_upsert = AsyncMock(return_value=Mock(
            success=True,
            processed_records=10,
            failed_records=0
        ))
        
        # Create data pipeline orchestrator
        with patch('main.data_pipeline.orchestrator.get_archive', return_value=mock_archive):
            with patch('main.data_pipeline.orchestrator.MarketDataRepository', return_value=mock_repo):
                pipeline = DataPipelineOrchestrator(test_config)
                
                # Run full pipeline
                result = await pipeline.run_full_pipeline(
                    symbols=['AAPL', 'MSFT'],
                    start_date=datetime.now(timezone.utc) - timedelta(days=1),
                    end_date=datetime.now(timezone.utc)
                )
                
                # Verify complete flow
                assert 'market_data' in result
                assert result['market_data']['status'] == 'success'
                
                # Verify repository was called
                assert mock_repo.bulk_upsert.called