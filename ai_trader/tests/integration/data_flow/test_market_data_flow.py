"""
Integration tests for market data flow through the system.

Tests the complete data pipeline from ingestion to storage:
1. Data fetching from source
2. Archive storage
3. Database loading
4. Data retrieval and verification
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from typing import Dict, Any

from main.data_pipeline.storage.archive import DataArchive, RawDataRecord
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.ingestion.loaders.market_data_split import MarketDataSplitBulkLoader
from main.data_pipeline.services.storage import (
    QualificationService,
    TableRoutingService,
    PartitionManager
)
from main.config import get_config_manager


@pytest.fixture
async def test_archive(tmp_path):
    """Create a test archive with temporary storage."""
    archive_config = {
        'storage_type': 'local',
        'local_path': str(tmp_path / 'test_archive')
    }
    return DataArchive(archive_config)


@pytest.fixture
async def test_db():
    """Create a test database connection."""
    import os
    config = {
        'database': {
            'host': os.getenv('TEST_DB_HOST', 'localhost'),
            'port': int(os.getenv('TEST_DB_PORT', '5432')),
            'name': os.getenv('TEST_DB_NAME', 'ai_trader_test'),
            'user': os.getenv('TEST_DB_USER', 'zachwade'),
            'password': os.getenv('TEST_DB_PASSWORD', 'ZachT$2002'),
            'pool_size': 5,
            'max_overflow': 10
        }
    }
    
    db_factory = DatabaseFactory()
    db_adapter = db_factory.create_async_database(config)
    
    yield db_adapter
    
    await db_adapter.close()


@pytest.fixture
async def bulk_loader(test_db, test_archive):
    """Create a market data bulk loader for testing."""
    # Mock services for testing
    class MockQualificationService:
        async def get_qualification(self, symbol):
            return type('Qualification', (), {
                'layer_qualified': 1,
                'should_store_interval': lambda x: True
            })
    
    class MockTableRoutingService:
        def get_table_for_interval(self, interval):
            return 'market_data_1h' if interval in ['1hour', '1day'] else 'market_data_5m'
        
        def get_interval_for_table(self, table, interval):
            return interval
    
    class MockPartitionManager:
        async def ensure_partitions_for_tables(self, table_ranges):
            return {}
    
    loader = MarketDataSplitBulkLoader(
        db_adapter=test_db,
        qualification_service=MockQualificationService(),
        routing_service=MockTableRoutingService(),
        partition_manager=MockPartitionManager(),
        archive=test_archive
    )
    
    return loader


class TestMarketDataFlow:
    """Test market data flow through the pipeline."""
    
    @pytest.mark.asyncio
    async def test_data_ingestion_to_archive(self, test_archive):
        """Test that market data can be stored in the archive."""
        # Create sample market data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=10, freq='1h'),
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [99.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000000] * 10,
            'vwap': [101.5] * 10,
            'trades': [500] * 10
        })
        
        # Create a raw data record
        record = RawDataRecord(
            source='polygon',
            data_type='market_data',
            symbol='TEST',
            timestamp=datetime.now(timezone.utc),
            data={'market_data': data.to_dict('records')},
            metadata={
                'interval': '1hour',
                'record_count': len(data),
                'start_time': data['timestamp'].min().isoformat(),
                'end_time': data['timestamp'].max().isoformat()
            }
        )
        
        # Save to archive
        await test_archive.save_raw_record_async(record)
        
        # Verify data was saved
        retrieved = await test_archive.query_raw_records(
            source='polygon',
            data_type='market_data',
            symbol='TEST',
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc) + timedelta(days=1)
        )
        
        assert len(retrieved) == 1
        assert retrieved[0].symbol == 'TEST'
        assert len(retrieved[0].data['market_data']) == 10
    
    @pytest.mark.asyncio
    async def test_archive_to_database_loading(self, bulk_loader, test_archive):
        """Test that archived data can be loaded to the database."""
        # First, store data in archive
        data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=5, freq='1h'),
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [105.0, 106.0, 107.0, 108.0, 109.0],
            'low': [99.0, 100.0, 101.0, 102.0, 103.0],
            'close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'vwap': [101.5, 102.5, 103.5, 104.5, 105.5],
            'trades': [500, 550, 600, 650, 700]
        })
        
        record = RawDataRecord(
            source='polygon',
            data_type='market_data',
            symbol='TEST2',
            timestamp=datetime.now(timezone.utc),
            data={'market_data': data.to_dict('records')},
            metadata={'interval': '1hour'}
        )
        
        await test_archive.save_raw_record_async(record)
        
        # Load from archive to database
        result = await bulk_loader.load(
            data=data,
            symbol='TEST2',
            interval='1hour',
            source='polygon'
        )
        
        assert result.success
        assert result.records_loaded > 0
        assert 'TEST2' in result.symbols_processed
    
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, bulk_loader, test_db):
        """Test complete flow from ingestion to retrieval."""
        # Create test data
        test_symbol = 'E2E_TEST'
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01 09:00', periods=3, freq='1h'),
            'open': [150.0, 151.0, 152.0],
            'high': [155.0, 156.0, 157.0],
            'low': [149.0, 150.0, 151.0],
            'close': [153.0, 154.0, 155.0],
            'volume': [2000000, 2100000, 2200000],
            'vwap': [152.0, 153.0, 154.0],
            'trades': [1000, 1100, 1200]
        })
        
        # Load data
        result = await bulk_loader.load(
            data=test_data,
            symbol=test_symbol,
            interval='1hour',
            source='test'
        )
        
        # Flush any buffered data
        await bulk_loader.flush()
        
        assert result.success
        
        # Query database to verify data was stored
        query = """
            SELECT COUNT(*) as count
            FROM market_data_1h
            WHERE symbol = $1
        """
        
        count_result = await test_db.fetch_one(query, test_symbol)
        assert count_result['count'] >= 3
    
    @pytest.mark.asyncio
    async def test_data_integrity(self, bulk_loader, test_db):
        """Test that data integrity is maintained through the pipeline."""
        # Create precise test data
        test_symbol = 'INTEGRITY_TEST'
        original_data = pd.DataFrame({
            'timestamp': [
                pd.Timestamp('2025-01-01 10:00:00', tz='UTC'),
                pd.Timestamp('2025-01-01 11:00:00', tz='UTC')
            ],
            'open': [200.50, 201.75],
            'high': [205.25, 206.50],
            'low': [199.75, 201.00],
            'close': [204.00, 205.25],
            'volume': [3000000, 3100000],
            'vwap': [202.125, 203.375],
            'trades': [1500, 1600]
        })
        
        # Load data
        await bulk_loader.load(
            data=original_data,
            symbol=test_symbol,
            interval='1hour',
            source='test'
        )
        
        # Flush buffer
        await bulk_loader.flush()
        
        # Retrieve and verify
        query = """
            SELECT timestamp, open, high, low, close, volume, vwap, trades
            FROM market_data_1h
            WHERE symbol = $1
            ORDER BY timestamp
        """
        
        rows = await test_db.fetch_all(query, test_symbol)
        
        assert len(rows) >= 2
        
        # Verify first row
        if rows:
            first_row = rows[0]
            assert float(first_row['open']) == 200.50
            assert float(first_row['high']) == 205.25
            assert float(first_row['low']) == 199.75
            assert float(first_row['close']) == 204.00
            assert int(first_row['volume']) == 3000000
            assert float(first_row['vwap']) == pytest.approx(202.125, rel=1e-3)
            assert int(first_row['trades']) == 1500
    
    @pytest.mark.asyncio
    async def test_duplicate_handling(self, bulk_loader, test_db):
        """Test that duplicate data is handled correctly."""
        test_symbol = 'DUP_TEST'
        test_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2025-01-01 12:00:00', tz='UTC')],
            'open': [100.0],
            'high': [105.0],
            'low': [99.0],
            'close': [102.0],
            'volume': [1000000],
            'vwap': [101.5],
            'trades': [500]
        })
        
        # Load same data twice
        await bulk_loader.load(test_data, test_symbol, '1hour', 'test')
        await bulk_loader.flush()
        
        await bulk_loader.load(test_data, test_symbol, '1hour', 'test')
        await bulk_loader.flush()
        
        # Should only have one record (upsert behavior)
        query = """
            SELECT COUNT(*) as count
            FROM market_data_1h
            WHERE symbol = $1
        """
        
        count_result = await test_db.fetch_one(query, test_symbol)
        assert count_result['count'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])