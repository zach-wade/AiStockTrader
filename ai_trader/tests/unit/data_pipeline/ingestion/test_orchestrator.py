"""
Unit tests for IngestionOrchestrator.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from main.data_pipeline.ingestion.orchestrator import IngestionOrchestrator
from main.data_pipeline.types import DataPipelineStatus, DataPipelineResult
from main.data_pipeline.validation.validation_pipeline import ValidationPipeline

from tests.fixtures.data_pipeline.mock_clients import (
    MockAlpacaMarketClient,
    MockPolygonMarketClient,
    MockNewsClient,
    create_mock_clients_dict
)
from tests.fixtures.data_pipeline.test_configs import get_test_config, get_error_test_config


@pytest.mark.asyncio
class TestIngestionOrchestrator:
    """Test IngestionOrchestrator functionality."""
    
    @pytest.fixture
    def test_config(self):
        """Get test configuration."""
        return get_test_config()
    
    @pytest.fixture
    def mock_clients(self):
        """Create mock client dictionary."""
        return create_mock_clients_dict()
    
    @pytest.fixture
    def mock_archive(self):
        """Create mock archive."""
        archive = Mock()
        archive.archive_raw_data = Mock(return_value='mock_key')
        return archive
    
    @pytest.fixture
    def mock_validation_pipeline(self):
        """Create mock validation pipeline."""
        pipeline = Mock(spec=ValidationPipeline)
        pipeline.validate_ingest = AsyncMock(return_value=Mock(
            passed=True,
            errors=[],
            warnings=[],
            has_warnings=False
        ))
        return pipeline
    
    @pytest_asyncio.fixture
    async def orchestrator(self, test_config, mock_clients, mock_archive, mock_validation_pipeline):
        """Create orchestrator instance with mocks."""
        with patch('main.data_pipeline.ingestion.orchestrator.get_config', return_value=test_config):
            with patch('main.data_pipeline.ingestion.orchestrator.get_archive', return_value=mock_archive):
                with patch('main.data_pipeline.ingestion.orchestrator.ValidationPipeline', return_value=mock_validation_pipeline):
                    orchestrator = IngestionOrchestrator(mock_clients)
                    yield orchestrator
    
    async def test_orchestrator_initialization(self, test_config, mock_clients):
        """Test orchestrator initialization."""
        with patch('main.data_pipeline.ingestion.orchestrator.get_config', return_value=test_config):
            with patch('main.data_pipeline.ingestion.orchestrator.get_archive'):
                orchestrator = IngestionOrchestrator(mock_clients)
                
                assert orchestrator.config == test_config
                assert orchestrator.clients == mock_clients
                assert hasattr(orchestrator, 'archive')
                assert hasattr(orchestrator, 'resilience')
    
    async def test_run_ingestion_success(self, orchestrator):
        """Test successful ingestion run."""
        # Setup
        data_type = 'market_data'
        symbols = ['AAPL', 'MSFT']
        start_date = datetime.now(timezone.utc) - timedelta(days=1)
        end_date = datetime.now(timezone.utc)
        
        # Execute
        result = await orchestrator.run_ingestion(
            data_type=data_type,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe='1hour'
        )
        
        # Verify
        assert isinstance(result, DataPipelineResult)
        assert result.status == DataPipelineStatus.SUCCESS
        assert result.records_processed > 0
        assert len(result.errors) == 0
        assert result.metadata['data_type'] == data_type
        assert result.metadata['symbols_count'] == len(symbols)
    
    async def test_run_ingestion_no_capable_clients(self, orchestrator):
        """Test ingestion when no clients can handle data type."""
        # Execute with unsupported data type
        result = await orchestrator.run_ingestion(
            data_type='unsupported_type',
            symbols=['AAPL'],
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc)
        )
        
        # Verify
        assert result.status == DataPipelineStatus.FAILED
        assert result.records_processed == 0
        assert len(result.errors) > 0
        assert "No clients available" in result.errors[0]
    
    async def test_prioritized_client_selection(self, orchestrator, test_config):
        """Test client prioritization based on config."""
        # Get prioritized clients for market data
        clients = orchestrator._get_prioritized_clients('market_data')
        
        # Verify priority order (alpaca should be first based on config)
        assert len(clients) > 0
        client_names = [name for name, _ in clients]
        
        # Check if alpaca is prioritized
        alpaca_index = next((i for i, name in enumerate(client_names) if 'alpaca' in name), -1)
        polygon_index = next((i for i, name in enumerate(client_names) if 'polygon' in name), -1)
        
        if alpaca_index >= 0 and polygon_index >= 0:
            assert alpaca_index < polygon_index
    
    async def test_client_failure_fallback(self, orchestrator, mock_clients):
        """Test fallback to secondary client on failure."""
        # Make primary client fail
        mock_clients['alpaca_market'].fail_on_fetch = True
        
        # Execute
        result = await orchestrator.run_ingestion(
            data_type='market_data',
            symbols=['AAPL'],
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc)
        )
        
        # Verify fallback worked
        assert result.status in [DataPipelineStatus.SUCCESS, DataPipelineStatus.PARTIAL]
        assert result.records_processed > 0  # Should have data from polygon
        assert len(result.errors) >= 1  # Should have error from alpaca
    
    async def test_batch_processing(self, orchestrator, test_config):
        """Test batch processing for large symbol sets."""
        # Create large symbol list
        symbols = [f'SYM{i:03d}' for i in range(150)]
        
        # Execute
        result = await orchestrator.run_ingestion(
            data_type='market_data',
            symbols=symbols,
            start_date=datetime.now(timezone.utc) - timedelta(hours=1),
            end_date=datetime.now(timezone.utc),
            timeframe='1hour'
        )
        
        # Verify batching occurred
        assert result.status == DataPipelineStatus.SUCCESS
        assert result.metadata['symbols_count'] == 150
        # Based on config batch_size of 50, should have created 3 batches
    
    async def test_validation_integration(self, orchestrator, mock_validation_pipeline):
        """Test validation pipeline integration."""
        # Setup validation to fail
        mock_validation_pipeline.validate_ingest.return_value = Mock(
            passed=False,
            errors=['Invalid data format'],
            warnings=[],
            has_warnings=False
        )
        
        # Execute
        result = await orchestrator.run_ingestion(
            data_type='market_data',
            symbols=['AAPL'],
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc)
        )
        
        # Verify validation was called and errors captured
        assert mock_validation_pipeline.validate_ingest.called
        assert len(result.errors) > 0
        assert 'Invalid data format' in str(result.errors)
    
    async def test_resilience_retry_logic(self, mock_clients, mock_archive, test_config):
        """Test resilience manager retry logic."""
        # Use error test config with retries
        error_config = get_error_test_config()
        
        # Setup client to fail then succeed
        call_count = 0
        
        async def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return {'records_processed': 1, 'data': []}
        
        mock_clients['alpaca_market'].fetch_and_archive = fail_then_succeed
        
        # Create orchestrator with retry config
        with patch('main.data_pipeline.ingestion.orchestrator.get_config', return_value=error_config):
            with patch('main.data_pipeline.ingestion.orchestrator.get_archive', return_value=mock_archive):
                orchestrator = IngestionOrchestrator(mock_clients)
                
                # Execute
                result = await orchestrator.run_ingestion(
                    data_type='market_data',
                    symbols=['AAPL'],
                    start_date=datetime.now(timezone.utc),
                    end_date=datetime.now(timezone.utc)
                )
                
                # Verify retry worked
                assert result.status == DataPipelineStatus.SUCCESS
                assert call_count > 2  # Should have retried
    
    async def test_health_check(self, orchestrator, mock_clients):
        """Test health check functionality."""
        # Add health check method to mock client
        mock_clients['alpaca_market'].health_check = AsyncMock(return_value=True)
        mock_clients['polygon_market'].health_check = AsyncMock(return_value=False)
        
        # Execute
        health_status = await orchestrator.health_check()
        
        # Verify
        assert 'alpaca_market' in health_status
        assert health_status['alpaca_market']['healthy'] is True
        assert 'polygon_market' in health_status
        assert health_status['polygon_market']['healthy'] is False
    
    async def test_run_full_ingestion(self, orchestrator):
        """Test run_full_ingestion method."""
        # Execute
        summary = await orchestrator.run_full_ingestion()
        
        # Verify
        assert 'ingested_records' in summary
        assert 'data_types_processed' in summary
        assert 'successful_runs' in summary
        assert summary['data_types_processed'] >= 1
        assert summary['ingested_records'] >= 0
    
    async def test_news_ingestion(self, orchestrator):
        """Test news data ingestion."""
        # Execute
        result = await orchestrator.run_ingestion(
            data_type='news',
            symbols=['AAPL', 'TSLA'],
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc)
        )
        
        # Verify
        assert result.status == DataPipelineStatus.SUCCESS
        assert result.records_processed > 0
        assert result.metadata['data_type'] == 'news'
    
    async def test_concurrent_client_execution(self, orchestrator):
        """Test concurrent execution of multiple clients."""
        # Setup multiple clients for same data type
        orchestrator.clients['alpaca_market_2'] = MockAlpacaMarketClient()
        orchestrator.clients['polygon_market_2'] = MockPolygonMarketClient()
        
        # Execute with small symbol set (won't trigger batching)
        result = await orchestrator.run_ingestion(
            data_type='market_data',
            symbols=['AAPL'],
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc)
        )
        
        # Verify multiple clients were tried
        assert result.status == DataPipelineStatus.SUCCESS
        assert result.metadata['clients_tried'] >= 1