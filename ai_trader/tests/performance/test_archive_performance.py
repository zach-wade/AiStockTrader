#!/usr/bin/env python3
"""
Archive Performance Tests

Comprehensive performance benchmarks for archive operations including:
- Single-record write speeds
- Batch write performance
- Read query speeds
- Concurrent operations
- Format comparison (parquet vs JSON)
"""

import pytest
import asyncio
import time
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any
import json
import sys
sys.path.insert(0, 'src')

from main.data_pipeline.storage.archive import DataArchive, RawDataRecord
from main.monitoring.metrics.archive_metrics_collector import ArchiveMetricsCollector


class TestArchivePerformance:
    """Performance tests for archive operations."""
    
    @pytest.fixture
    async def archive_setup(self):
        """Set up archive with temporary directory and metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create metrics collector
            metrics = ArchiveMetricsCollector(
                archive_path=temp_dir,
                metrics_window_size=10000
            )
            
            # Create archive with metrics
            archive = DataArchive(
                config={'local_path': temp_dir},
                metrics_collector=metrics
            )
            
            # Start metrics collection
            await metrics.start()
            
            yield archive, metrics
            
            # Stop metrics collection
            await metrics.stop()
    
    @pytest.mark.asyncio
    async def test_single_record_write_speed(self, archive_setup):
        """Test single record write performance."""
        archive, metrics = archive_setup
        
        # Create test records of varying sizes
        sizes = [100, 1000, 10000, 100000]  # Number of data points
        results = {}
        
        for size in sizes:
            # Create market data record
            data = {
                'market_data': [
                    {
                        'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                        'open': 100.0 + i * 0.1,
                        'high': 101.0 + i * 0.1,
                        'low': 99.0 + i * 0.1,
                        'close': 100.5 + i * 0.1,
                        'volume': 1000000 + i * 1000
                    }
                    for i in range(size)
                ]
            }
            
            record = RawDataRecord(
                source='test',
                data_type='market_data',
                symbol='TEST',
                timestamp=datetime.now(timezone.utc),
                data=data,
                metadata={'records': size}
            )
            
            # Measure write time
            start = time.time()
            success = await archive.save_raw_record(record)
            duration = time.time() - start
            
            assert success, f"Failed to save record with {size} data points"
            
            results[size] = {
                'duration_ms': duration * 1000,
                'throughput': size / duration if duration > 0 else 0
            }
            
            print(f"Size {size}: {duration*1000:.2f}ms, {results[size]['throughput']:.0f} records/sec")
        
        # Verify performance targets
        assert results[100]['duration_ms'] < 50, "Small record write too slow"
        assert results[100000]['throughput'] > 1000, "Large record throughput too low"
    
    @pytest.mark.asyncio
    async def test_batch_write_performance(self, archive_setup):
        """Test batch write performance."""
        archive, metrics = archive_setup
        
        batch_sizes = [10, 100, 1000]
        results = {}
        
        for batch_size in batch_sizes:
            records = []
            for i in range(batch_size):
                record = RawDataRecord(
                    source='test',
                    data_type='market_data',
                    symbol=f'SYMBOL{i}',
                    timestamp=datetime.now(timezone.utc),
                    data={'price': 100.0 + i, 'volume': 1000000},
                    metadata={'batch': batch_size, 'index': i}
                )
                records.append(record)
            
            # Measure batch write time
            start = time.time()
            tasks = [archive.save_raw_record(record) for record in records]
            results_batch = await asyncio.gather(*tasks)
            duration = time.time() - start
            
            success_rate = sum(results_batch) / len(results_batch)
            assert success_rate == 1.0, f"Some records failed in batch {batch_size}"
            
            results[batch_size] = {
                'duration_ms': duration * 1000,
                'throughput': batch_size / duration if duration > 0 else 0,
                'avg_per_record_ms': (duration * 1000) / batch_size
            }
            
            print(f"Batch {batch_size}: {duration*1000:.2f}ms total, "
                  f"{results[batch_size]['avg_per_record_ms']:.2f}ms/record, "
                  f"{results[batch_size]['throughput']:.0f} records/sec")
        
        # Performance targets
        assert results[1000]['throughput'] > 100, "Batch throughput too low"
        assert results[1000]['avg_per_record_ms'] < 10, "Per-record time too high in batch"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, archive_setup):
        """Test concurrent read/write operations."""
        archive, metrics = archive_setup
        
        # First, write some data
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        write_tasks = []
        
        for symbol in symbols:
            for i in range(20):  # 20 records per symbol
                record = RawDataRecord(
                    source='test',
                    data_type='market_data',
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                    data={'price': 100.0 + i, 'volume': 1000000}
                )
                write_tasks.append(archive.save_raw_record(record))
        
        # Execute writes concurrently
        start = time.time()
        write_results = await asyncio.gather(*write_tasks)
        write_duration = time.time() - start
        
        total_writes = len(write_tasks)
        successful_writes = sum(write_results)
        
        print(f"Concurrent writes: {total_writes} records in {write_duration:.2f}s "
              f"({total_writes/write_duration:.0f} records/sec)")
        
        # Now test concurrent reads
        read_tasks = []
        for symbol in symbols:
            read_tasks.append(
                archive.query_raw_records(
                    source='test',
                    data_type='market_data',
                    symbol=symbol
                )
            )
        
        start = time.time()
        read_results = await asyncio.gather(*read_tasks)
        read_duration = time.time() - start
        
        total_read = sum(len(result) for result in read_results)
        
        print(f"Concurrent reads: {total_read} records in {read_duration:.2f}s "
              f"({total_read/read_duration:.0f} records/sec)")
        
        # Check metrics for concurrent operations
        perf_summary = metrics.get_performance_summary()
        max_concurrent = perf_summary.get('concurrent', {}).get('max_observed', 0)
        
        print(f"Max concurrent operations: {max_concurrent}")
        
        # Targets
        assert successful_writes == total_writes, "Some concurrent writes failed"
        assert total_read == total_writes, "Read count doesn't match write count"
        assert max_concurrent >= min(10, total_writes), "Not enough concurrent operations"
    
    @pytest.mark.asyncio
    async def test_format_comparison(self, archive_setup):
        """Compare performance of parquet vs JSON formats."""
        archive, metrics = archive_setup
        
        # Create identical data for both formats
        data_sizes = [1000, 10000]
        formats = ['parquet', 'json']
        
        results = {}
        
        for size in data_sizes:
            for fmt in formats:
                # Create data
                if fmt == 'parquet':
                    data = {
                        'market_data': [
                            {
                                'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                                'price': 100.0 + i * 0.1,
                                'volume': 1000000 + i
                            }
                            for i in range(size)
                        ]
                    }
                    data_type = 'market_data'  # Will use parquet
                else:
                    data = {
                        'articles': [
                            {
                                'id': f'article_{i}',
                                'title': f'Title {i}',
                                'content': 'x' * 1000,  # 1KB per article
                                'timestamp': datetime.now(timezone.utc) - timedelta(hours=i)
                            }
                            for i in range(size)
                        ]
                    }
                    data_type = 'news'  # Will use JSON
                
                record = RawDataRecord(
                    source='test',
                    data_type=data_type,
                    symbol='TEST',
                    timestamp=datetime.now(timezone.utc),
                    data=data,
                    metadata={'format': fmt, 'size': size}
                )
                
                # Measure write
                start = time.time()
                success = await archive.save_raw_record(record)
                write_duration = time.time() - start
                
                assert success, f"Failed to save {fmt} record"
                
                # Measure read
                start = time.time()
                read_results = await archive.query_raw_records(
                    source='test',
                    data_type=data_type,
                    symbol='TEST'
                )
                read_duration = time.time() - start
                
                key = f"{fmt}_{size}"
                results[key] = {
                    'write_ms': write_duration * 1000,
                    'read_ms': read_duration * 1000,
                    'records_read': len(read_results)
                }
                
                print(f"{fmt.upper()} ({size} records): "
                      f"Write={results[key]['write_ms']:.2f}ms, "
                      f"Read={results[key]['read_ms']:.2f}ms")
        
        # Compare formats
        parquet_write = results['parquet_10000']['write_ms']
        json_write = results['json_10000']['write_ms']
        
        print(f"\nParquet is {json_write/parquet_write:.1f}x faster for writes")
        
        # Parquet should be faster for large datasets
        assert parquet_write < json_write * 2, "Parquet not efficient enough"
    
    @pytest.mark.asyncio
    async def test_query_performance(self, archive_setup):
        """Test query performance with different filters."""
        archive, metrics = archive_setup
        
        # Write test data across multiple days
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        days = 30
        records_per_day = 24  # Hourly data
        
        write_tasks = []
        for symbol in symbols:
            for day in range(days):
                for hour in range(records_per_day):
                    record = RawDataRecord(
                        source='test',
                        data_type='market_data',
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc) - timedelta(days=day, hours=hour),
                        data={'price': 100.0 + day + hour * 0.1, 'volume': 1000000}
                    )
                    write_tasks.append(archive.save_raw_record(record))
        
        # Write all data
        await asyncio.gather(*write_tasks)
        
        # Test different query patterns
        query_tests = [
            {
                'name': 'Single symbol, all time',
                'params': {'symbol': 'AAPL'},
                'expected_count': days * records_per_day
            },
            {
                'name': 'Single symbol, date range',
                'params': {
                    'symbol': 'AAPL',
                    'start_date': datetime.now(timezone.utc) - timedelta(days=7),
                    'end_date': datetime.now(timezone.utc)
                },
                'expected_count': 7 * records_per_day
            },
            {
                'name': 'All symbols, single day',
                'params': {
                    'start_date': datetime.now(timezone.utc) - timedelta(days=1),
                    'end_date': datetime.now(timezone.utc)
                },
                'expected_count': len(symbols) * records_per_day
            }
        ]
        
        for test in query_tests:
            start = time.time()
            results = await archive.query_raw_records(
                source='test',
                data_type='market_data',
                **test['params']
            )
            duration = time.time() - start
            
            print(f"{test['name']}: {len(results)} records in {duration*1000:.2f}ms "
                  f"({len(results)/duration if duration > 0 else 0:.0f} records/sec)")
            
            # Allow some tolerance for async timing
            assert len(results) >= test['expected_count'] * 0.9, \
                f"Query returned {len(results)}, expected ~{test['expected_count']}"
            
            # Performance target
            assert duration < 1.0, f"Query took too long: {duration:.2f}s"
    
    @pytest.mark.asyncio
    async def test_storage_efficiency(self, archive_setup):
        """Test storage efficiency and compression."""
        archive, metrics = archive_setup
        
        # Write data and measure storage
        data_types = {
            'market_data': {
                'records': 10000,
                'data_generator': lambda i: {
                    'timestamp': datetime.now(timezone.utc) - timedelta(minutes=i),
                    'open': 100.0 + i * 0.01,
                    'high': 101.0 + i * 0.01,
                    'low': 99.0 + i * 0.01,
                    'close': 100.5 + i * 0.01,
                    'volume': 1000000 + i * 100,
                    'vwap': 100.3 + i * 0.01
                }
            },
            'news': {
                'records': 1000,
                'data_generator': lambda i: {
                    'id': f'article_{i}',
                    'title': f'Breaking News {i}: ' + 'x' * 50,
                    'content': 'Lorem ipsum ' * 100,  # ~1KB
                    'author': f'Author {i % 10}',
                    'published': datetime.now(timezone.utc) - timedelta(hours=i)
                }
            }
        }
        
        storage_results = {}
        
        for data_type, config in data_types.items():
            # Generate data
            data = {
                data_type: [
                    config['data_generator'](i)
                    for i in range(config['records'])
                ]
            }
            
            # Calculate original size
            original_size = sys.getsizeof(json.dumps(data, default=str))
            
            # Save to archive
            record = RawDataRecord(
                source='test',
                data_type=data_type,
                symbol='TEST',
                timestamp=datetime.now(timezone.utc),
                data=data
            )
            
            await archive.save_raw_record(record)
            
            # Get storage metrics
            await asyncio.sleep(0.1)  # Let metrics update
            storage_metrics = await metrics.get_storage_metrics()
            
            storage_results[data_type] = {
                'original_size': original_size,
                'stored_size': storage_metrics.used_size_bytes,
                'compression_ratio': original_size / storage_metrics.used_size_bytes if storage_metrics.used_size_bytes > 0 else 1,
                'records': config['records']
            }
            
            print(f"{data_type}: Original={original_size/1024/1024:.2f}MB, "
                  f"Stored={storage_metrics.used_size_bytes/1024/1024:.2f}MB, "
                  f"Compression={storage_results[data_type]['compression_ratio']:.2f}x")
        
        # Verify compression efficiency
        assert storage_results['market_data']['compression_ratio'] > 2, \
            "Market data compression too low"
        
        # Check storage growth prediction
        storage_metrics = await metrics.get_storage_metrics()
        if storage_metrics.estimated_days_until_full:
            print(f"\nEstimated days until full: {storage_metrics.estimated_days_until_full:.1f}")


def run_performance_benchmarks():
    """Run all performance benchmarks and print summary."""
    print("=" * 60)
    print("Archive Performance Benchmarks")
    print("=" * 60)
    
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
    
    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_performance_benchmarks()