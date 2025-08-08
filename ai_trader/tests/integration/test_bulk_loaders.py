#!/usr/bin/env python3
"""
Comprehensive integration tests for all bulk loaders with archive functionality.

This test suite validates:
1. All bulk loaders properly archive data
2. Data integrity is maintained through the pipeline
3. Error recovery works correctly
4. Performance meets requirements
"""

import asyncio
import tempfile
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import time
from typing import List, Dict, Any
sys.path.insert(0, 'src')

from main.data_pipeline.storage.archive import DataArchive, RawDataRecord
from main.data_pipeline.ingestion.loaders.news import NewsBulkLoader
from main.data_pipeline.ingestion.loaders.corporate_actions import CorporateActionsBulkLoader
from main.data_pipeline.ingestion.loaders.fundamentals import FundamentalsBulkLoader
from main.data_pipeline.ingestion.loaders.market_data import MarketDataBulkLoader
from main.data_pipeline.ingestion.loaders.market_data_split import MarketDataSplitBulkLoader
from main.data_pipeline.services.ingestion import TextProcessingService, DeduplicationService
from main.interfaces.ingestion import BulkLoadConfig
from unittest.mock import AsyncMock, MagicMock, Mock


class TestMetrics:
    """Track test metrics for reporting."""
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.start_time = time.time()
        self.failures = []
    
    def record_pass(self, test_name: str):
        self.tests_run += 1
        self.tests_passed += 1
        print(f"✅ {test_name}")
    
    def record_fail(self, test_name: str, error: str):
        self.tests_run += 1
        self.tests_failed += 1
        self.failures.append((test_name, error))
        print(f"❌ {test_name}: {error}")
    
    def print_summary(self):
        duration = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Duration: {duration:.2f} seconds")
        
        if self.failures:
            print("\nFailed Tests:")
            for test, error in self.failures:
                print(f"  - {test}: {error}")
        
        print("=" * 60)
        return self.tests_failed == 0


async def test_market_data_archive(metrics: TestMetrics):
    """Test market data bulk loader with archive integration."""
    test_name = "Market Data Archive Integration"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            archive = DataArchive({'storage_type': 'local', 'local_path': temp_dir})
            db_adapter = create_mock_db_adapter()
            config = BulkLoadConfig(buffer_size=10, batch_timeout_seconds=1.0)
            
            loader = MarketDataBulkLoader(
                db_adapter=db_adapter,
                archive=archive,
                config=config
            )
            
            # Test data
            now = datetime.now(timezone.utc)
            test_data = [
                {
                    'symbol': 'AAPL',
                    'timestamp': now - timedelta(hours=i),
                    'open': 150.0 + i,
                    'high': 151.0 + i,
                    'low': 149.0 + i,
                    'close': 150.5 + i,
                    'volume': 1000000 + i * 10000,
                    'vwap': 150.3 + i * 0.1,
                    'interval': '1hour',
                    'source': 'polygon'
                }
                for i in range(5)
            ]
            
            # Load data
            result = await loader.load(
                data=test_data,
                symbols=['AAPL'],
                source='polygon'
            )
            
            # Force flush to trigger archive
            await loader.flush_all()
            
            # Verify archive
            archive_path = Path(temp_dir) / 'raw' / 'polygon' / 'market_data'
            files = list(archive_path.rglob('*.parquet')) if archive_path.exists() else []
            
            if len(files) > 0:
                # Read and verify archived data
                df = pd.read_parquet(files[0])
                if 'market_data' in df.columns:
                    # Data is nested
                    archived_records = df['market_data'].iloc[0]
                else:
                    archived_records = df.to_dict('records')
                
                if len(archived_records) >= len(test_data):
                    metrics.record_pass(f"{test_name} - Archive created with {len(archived_records)} records")
                else:
                    metrics.record_fail(test_name, f"Archive has {len(archived_records)} records, expected {len(test_data)}")
            else:
                metrics.record_fail(test_name, "No archive files created")
                
    except Exception as e:
        metrics.record_fail(test_name, str(e))


async def test_market_data_split_archive(metrics: TestMetrics):
    """Test market data split bulk loader with archive integration."""
    test_name = "Market Data Split Archive Integration"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            archive = DataArchive({'storage_type': 'local', 'local_path': temp_dir})
            db_adapter = create_mock_db_adapter()
            
            # Mock qualification service
            qualification_service = Mock()
            qualification_service.get_symbols_by_layer = AsyncMock(return_value=['AAPL'])
            qualification_service.clear_cache = AsyncMock()
            
            config = BulkLoadConfig(buffer_size=10, batch_timeout_seconds=1.0)
            
            loader = MarketDataSplitBulkLoader(
                db_adapter=db_adapter,
                qualification_service=qualification_service,
                archive=archive,
                config=config
            )
            
            # Test data with different intervals
            now = datetime.now(timezone.utc)
            test_data = []
            
            for interval in ['1min', '5min', '1hour', '1day']:
                for i in range(2):
                    test_data.append({
                        'symbol': 'AAPL',
                        'timestamp': now - timedelta(hours=i),
                        'open': 150.0,
                        'high': 151.0,
                        'low': 149.0,
                        'close': 150.5,
                        'volume': 1000000,
                        'interval': interval,
                        'source': 'polygon'
                    })
            
            # Load data
            result = await loader.load(
                data=test_data,
                symbols=['AAPL'],
                source='polygon'
            )
            
            # Force flush
            await loader.flush_all()
            
            # Verify archive has interval-specific data
            archive_path = Path(temp_dir) / 'raw' / 'polygon' / 'market_data'
            files = list(archive_path.rglob('*.parquet')) if archive_path.exists() else []
            
            # Check for interval-specific files
            intervals_found = set()
            for file in files:
                # Read metadata to check interval
                meta_file = file.with_suffix('.parquet.meta')
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                        if 'interval' in metadata:
                            intervals_found.add(metadata['interval'])
            
            if len(intervals_found) >= 2:  # Should have multiple intervals
                metrics.record_pass(f"{test_name} - Archived {len(intervals_found)} different intervals")
            else:
                metrics.record_fail(test_name, f"Only {len(intervals_found)} intervals archived")
                
    except Exception as e:
        metrics.record_fail(test_name, str(e))


async def test_news_loader_deduplication(metrics: TestMetrics):
    """Test news loader properly deduplicates and archives."""
    test_name = "News Loader Deduplication"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            archive = DataArchive({'storage_type': 'local', 'local_path': temp_dir})
            db_adapter = create_mock_db_adapter()
            
            # Mock services
            text_processor = MagicMock(spec=TextProcessingService)
            text_processor.process_article = lambda x: {
                'title': x.get('title', ''),
                'content': x.get('content', ''),
                'symbols': ['AAPL'],
                'keywords': ['tech'],
                'sentiment_positive': 0.7,
                'sentiment_negative': 0.1,
                'sentiment_overall': 'positive'
            }
            
            deduplicator = AsyncMock(spec=DeduplicationService)
            
            config = BulkLoadConfig(buffer_size=10)
            
            loader = NewsBulkLoader(
                db_adapter=db_adapter,
                text_processor=text_processor,
                deduplicator=deduplicator,
                archive=archive,
                config=config
            )
            
            # Test data with duplicates
            test_articles = [
                {
                    'id': 'article_1',
                    'title': 'Apple News 1',
                    'content': 'Content 1',
                    'published_utc': datetime.now(timezone.utc).isoformat()
                },
                {
                    'id': 'article_1',  # Duplicate ID
                    'title': 'Apple News 1',
                    'content': 'Content 1',
                    'published_utc': datetime.now(timezone.utc).isoformat()
                },
                {
                    'id': 'article_2',
                    'title': 'Apple News 2',
                    'content': 'Content 2',
                    'published_utc': datetime.now(timezone.utc).isoformat()
                }
            ]
            
            # Deduplicator should remove 1 duplicate
            deduplicator.deduplicate_batch = AsyncMock(
                return_value=([test_articles[0], test_articles[2]], 1)
            )
            
            # Load data
            result = await loader.load(
                data=test_articles,
                symbols=['AAPL'],
                source='polygon'
            )
            
            await loader.flush_all()
            
            # Verify deduplication
            if result.metadata and result.metadata.get('duplicates_removed') == 1:
                metrics.record_pass(f"{test_name} - Correctly removed 1 duplicate")
            else:
                metrics.record_fail(test_name, "Deduplication count incorrect")
                
    except Exception as e:
        metrics.record_fail(test_name, str(e))


async def test_error_recovery(metrics: TestMetrics):
    """Test that loaders handle errors gracefully."""
    test_name = "Error Recovery"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup with failing database
            archive = DataArchive({'storage_type': 'local', 'local_path': temp_dir})
            
            # Create a database adapter that fails on execute
            db_adapter = AsyncMock()
            db_adapter.acquire = MagicMock()
            conn_mock = AsyncMock()
            conn_mock.execute = AsyncMock(side_effect=Exception("Database error"))
            conn_mock.copy_records_to_table = AsyncMock(side_effect=Exception("COPY failed"))
            db_adapter.acquire.return_value.__aenter__ = AsyncMock(return_value=conn_mock)
            db_adapter.acquire.return_value.__aexit__ = AsyncMock()
            
            config = BulkLoadConfig(
                buffer_size=10,
                recovery_enabled=True,
                recovery_directory=str(Path(temp_dir) / 'recovery')
            )
            
            loader = MarketDataBulkLoader(
                db_adapter=db_adapter,
                archive=archive,
                config=config
            )
            
            # Test data
            test_data = [{
                'symbol': 'AAPL',
                'timestamp': datetime.now(timezone.utc),
                'open': 150.0,
                'high': 151.0,
                'low': 149.0,
                'close': 150.5,
                'volume': 1000000,
                'interval': '1hour',
                'source': 'polygon'
            }]
            
            # Load should not raise exception despite DB error
            result = await loader.load(
                data=test_data,
                symbols=['AAPL'],
                source='polygon'
            )
            
            await loader.flush_all()
            
            # Check if data was archived despite DB failure
            archive_path = Path(temp_dir) / 'raw'
            archive_files = list(archive_path.rglob('*.parquet')) if archive_path.exists() else []
            
            if len(archive_files) > 0:
                metrics.record_pass(f"{test_name} - Data archived despite DB failure")
            else:
                metrics.record_fail(test_name, "Archive failed when DB failed")
                
    except Exception as e:
        metrics.record_fail(test_name, str(e))


async def test_concurrent_loading(metrics: TestMetrics):
    """Test concurrent loading from multiple symbols."""
    test_name = "Concurrent Symbol Loading"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            archive = DataArchive({'storage_type': 'local', 'local_path': temp_dir})
            db_adapter = create_mock_db_adapter()
            config = BulkLoadConfig(buffer_size=100)
            
            loader = MarketDataBulkLoader(
                db_adapter=db_adapter,
                archive=archive,
                config=config
            )
            
            # Create data for multiple symbols
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            now = datetime.now(timezone.utc)
            
            async def load_symbol_data(symbol: str):
                data = [{
                    'symbol': symbol,
                    'timestamp': now - timedelta(hours=i),
                    'open': 100.0 + i,
                    'high': 101.0 + i,
                    'low': 99.0 + i,
                    'close': 100.5 + i,
                    'volume': 1000000,
                    'interval': '1hour',
                    'source': 'polygon'
                } for i in range(10)]
                
                return await loader.load(
                    data=data,
                    symbols=[symbol],
                    source='polygon'
                )
            
            # Load all symbols concurrently
            start_time = time.time()
            results = await asyncio.gather(
                *[load_symbol_data(symbol) for symbol in symbols]
            )
            duration = time.time() - start_time
            
            # Flush all data
            await loader.flush_all()
            
            # Check performance
            if duration < 2.0:  # Should complete within 2 seconds
                metrics.record_pass(f"{test_name} - Loaded {len(symbols)} symbols in {duration:.2f}s")
            else:
                metrics.record_fail(test_name, f"Too slow: {duration:.2f}s for {len(symbols)} symbols")
                
    except Exception as e:
        metrics.record_fail(test_name, str(e))


async def test_large_batch_handling(metrics: TestMetrics):
    """Test handling of large batches that exceed memory limits."""
    test_name = "Large Batch Handling"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup with small memory limit
            archive = DataArchive({'storage_type': 'local', 'local_path': temp_dir})
            db_adapter = create_mock_db_adapter()
            config = BulkLoadConfig(
                buffer_size=1000,
                max_memory_mb=1  # Very small limit to trigger flush
            )
            
            loader = MarketDataBulkLoader(
                db_adapter=db_adapter,
                archive=archive,
                config=config
            )
            
            # Create large dataset
            now = datetime.now(timezone.utc)
            large_data = [{
                'symbol': 'AAPL',
                'timestamp': now - timedelta(minutes=i),
                'open': 150.0,
                'high': 151.0,
                'low': 149.0,
                'close': 150.5,
                'volume': 1000000,
                'vwap': 150.3,
                'interval': '1min',
                'source': 'polygon'
            } for i in range(1000)]  # 1000 records
            
            # Track flushes
            flush_count = 0
            original_flush = loader._flush_buffer
            
            async def counting_flush():
                nonlocal flush_count
                flush_count += 1
                return await original_flush()
            
            loader._flush_buffer = counting_flush
            
            # Load large batch
            result = await loader.load(
                data=large_data,
                symbols=['AAPL'],
                source='polygon'
            )
            
            await loader.flush_all()
            
            # Should have triggered multiple flushes due to memory limit
            if flush_count > 1:
                metrics.record_pass(f"{test_name} - Triggered {flush_count} automatic flushes")
            else:
                metrics.record_fail(test_name, f"Only {flush_count} flushes for large batch")
                
    except Exception as e:
        metrics.record_fail(test_name, str(e))


def create_mock_db_adapter():
    """Create a mock database adapter for testing."""
    db_adapter = AsyncMock()
    db_adapter.acquire = MagicMock()
    
    # Mock connection
    conn_mock = AsyncMock()
    conn_mock.execute = AsyncMock(return_value="INSERT 0 10")
    conn_mock.copy_records_to_table = AsyncMock()
    
    db_adapter.acquire.return_value.__aenter__ = AsyncMock(return_value=conn_mock)
    db_adapter.acquire.return_value.__aexit__ = AsyncMock()
    
    return db_adapter


async def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Bulk Loader Integration Tests")
    print("=" * 60)
    
    metrics = TestMetrics()
    
    # Run all tests
    await test_market_data_archive(metrics)
    await test_market_data_split_archive(metrics)
    await test_news_loader_deduplication(metrics)
    await test_error_recovery(metrics)
    await test_concurrent_loading(metrics)
    await test_large_batch_handling(metrics)
    
    # Print summary
    success = metrics.print_summary()
    
    if success:
        print("\n✅ All integration tests passed!")
        return 0
    else:
        print(f"\n❌ {metrics.tests_failed} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)