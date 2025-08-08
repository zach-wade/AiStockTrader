#!/usr/bin/env python3
"""
Archive Stress Tests

Push the archive system to its limits to identify breaking points
and ensure resilience under extreme conditions.

Run with: pytest tests/stress/test_archive_stress.py -v
Or: pytest -m stress
"""

import pytest
import asyncio
import time
import tempfile
import pandas as pd
import numpy as np
import psutil
import gc
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import random
import string
import sys
sys.path.insert(0, 'src')

from main.data_pipeline.storage.archive import DataArchive, RawDataRecord
# ArchiveMetricsCollector has UnifiedMetrics dependency - use archive without metrics for stress tests
# from main.monitoring.metrics.archive_metrics_collector import ArchiveMetricsCollector


@pytest.mark.stress
class TestArchiveStress:
    """Stress tests for archive system resilience."""
    
    @pytest.fixture
    async def stress_archive(self):
        """Set up archive for stress testing with monitoring."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create archive without metrics (UnifiedMetrics dependency not available)
            archive = DataArchive(
                config={'local_path': temp_dir}
            )
            
            # Track initial resource usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            initial_handles = process.num_fds() if hasattr(process, 'num_fds') else 0
            
            # Simple metrics tracking
            metrics = {
                'write_count': 0,
                'read_count': 0,
                'errors': [],
                'start_time': time.time()
            }
            
            yield {
                'archive': archive,
                'metrics': metrics,
                'temp_dir': temp_dir,
                'process': process,
                'initial_memory': initial_memory,
                'initial_handles': initial_handles
            }
            
            # Cleanup and resource check
            gc.collect()
            
            # Check for resource leaks
            final_memory = process.memory_info().rss / 1024 / 1024
            final_handles = process.num_fds() if hasattr(process, 'num_fds') else 0
            
            memory_leak = final_memory - initial_memory
            handle_leak = final_handles - initial_handles
            
            if memory_leak > 100:  # More than 100MB leak
                pytest.warn(f"Potential memory leak detected: {memory_leak:.2f}MB")
            if handle_leak > 10:
                pytest.warn(f"Potential file handle leak: {handle_leak} handles")
    
    def _generate_test_record(self, index: int, size_bytes: int = 1000) -> RawDataRecord:
        """Generate a test record with specified size."""
        # Create data of approximately the requested size
        data_points = size_bytes // 100  # Rough estimate
        
        data = {
            'index': index,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'values': [random.random() for _ in range(data_points)],
            'text': ''.join(random.choices(string.ascii_letters, k=size_bytes // 10))
        }
        
        return RawDataRecord(
            source='stress_test',
            data_type='test_data',
            symbol=f'TEST_{index % 100}',  # Distribute across 100 symbols
            timestamp=datetime.now(timezone.utc),
            data=data,
            metadata={'test_index': index, 'size_bytes': size_bytes}
        )
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 minute timeout
    async def test_high_volume_stress(self, stress_archive):
        """Test with extremely high volume of records."""
        archive = stress_archive['archive']
        metrics = stress_archive['metrics']
        process = stress_archive['process']
        
        print("\n=== HIGH VOLUME STRESS TEST ===")
        print(f"Writing 100,000 records rapidly...")
        
        num_records = 100000
        batch_size = 1000
        records_written = 0
        start_time = time.time()
        
        # Monitor resources
        max_memory = stress_archive['initial_memory']
        errors = []
        
        try:
            for batch_idx in range(num_records // batch_size):
                # Generate batch of records
                tasks = []
                for i in range(batch_size):
                    record_idx = batch_idx * batch_size + i
                    record = self._generate_test_record(record_idx, size_bytes=500)
                    tasks.append(archive.save_raw_record_async(record))
                
                # Write batch concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for errors
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        errors.append(f"Record {batch_idx * batch_size + idx}: {result}")
                    else:
                        records_written += 1
                
                # Monitor memory every 10 batches
                if batch_idx % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    max_memory = max(max_memory, current_memory)
                    
                    # Force garbage collection periodically
                    gc.collect()
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    rate = records_written / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {records_written}/{num_records} records "
                          f"({rate:.0f} records/sec, Memory: {current_memory:.1f}MB)")
        
        except Exception as e:
            pytest.fail(f"High volume test failed: {e}")
        
        # Final statistics
        total_time = time.time() - start_time
        write_rate = records_written / total_time
        memory_growth = max_memory - stress_archive['initial_memory']
        
        # Get metrics
        storage_metrics = await metrics.get_storage_metrics()
        operation_stats = metrics.get_operation_stats()
        
        print(f"\n=== RESULTS ===")
        print(f"Records written: {records_written}/{num_records}")
        print(f"Write rate: {write_rate:.0f} records/sec")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Memory growth: {memory_growth:.1f}MB (max: {max_memory:.1f}MB)")
        print(f"Errors: {len(errors)}")
        print(f"Storage used: {storage_metrics.total_size_mb:.2f}MB")
        print(f"Compression ratio: {storage_metrics.compression_ratio:.2f}")
        
        # Assertions
        assert records_written >= num_records * 0.99, f"Too many failures: {len(errors)}"
        assert write_rate > 100, f"Write rate too slow: {write_rate:.0f} records/sec"
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.1f}MB"
        
        if errors:
            print(f"First 10 errors: {errors[:10]}")
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_concurrent_operations_stress(self, stress_archive):
        """Test with many concurrent readers and writers."""
        archive = stress_archive['archive']
        metrics = stress_archive['metrics']
        
        print("\n=== CONCURRENT OPERATIONS STRESS TEST ===")
        print("Testing 100 concurrent writers and 100 concurrent readers...")
        
        num_writers = 100
        num_readers = 100
        records_per_writer = 100
        
        # First, write some data to read
        print("Seeding archive with initial data...")
        seed_tasks = []
        for i in range(1000):
            record = self._generate_test_record(i)
            seed_tasks.append(archive.save_raw_record_async(record))
        await asyncio.gather(*seed_tasks)
        
        # Track results
        write_results = {'success': 0, 'failure': 0}
        read_results = {'success': 0, 'failure': 0}
        
        async def writer_task(writer_id: int):
            """Individual writer task."""
            for i in range(records_per_writer):
                try:
                    record = self._generate_test_record(writer_id * 1000 + i)
                    success = await archive.save_raw_record_async(record)
                    if success:
                        write_results['success'] += 1
                    else:
                        write_results['failure'] += 1
                except Exception:
                    write_results['failure'] += 1
                
                # Small random delay
                await asyncio.sleep(random.uniform(0.001, 0.01))
        
        async def reader_task(reader_id: int):
            """Individual reader task."""
            for _ in range(50):  # Each reader performs 50 reads
                try:
                    symbol = f'TEST_{random.randint(0, 99)}'
                    records = await archive.query_raw_records(
                        source='stress_test',
                        data_type='test_data',
                        symbol=symbol,
                        start_date=datetime.now(timezone.utc) - timedelta(hours=1),
                        end_date=datetime.now(timezone.utc)
                    )
                    read_results['success'] += 1
                except Exception:
                    read_results['failure'] += 1
                
                # Small random delay
                await asyncio.sleep(random.uniform(0.001, 0.01))
        
        # Start all tasks concurrently
        start_time = time.time()
        
        all_tasks = []
        for i in range(num_writers):
            all_tasks.append(writer_task(i))
        for i in range(num_readers):
            all_tasks.append(reader_task(i))
        
        # Run all concurrently
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Get metrics
        operation_stats = metrics.get_operation_stats()
        
        print(f"\n=== RESULTS ===")
        print(f"Total operations time: {total_time:.2f} seconds")
        print(f"Write operations: {write_results['success']} success, {write_results['failure']} failure")
        print(f"Read operations: {read_results['success']} success, {read_results['failure']} failure")
        print(f"Write success rate: {write_results['success'] / (write_results['success'] + write_results['failure']) * 100:.1f}%")
        print(f"Read success rate: {read_results['success'] / (read_results['success'] + read_results['failure']) * 100:.1f}%")
        
        # Assertions
        total_writes = write_results['success'] + write_results['failure']
        assert write_results['success'] / total_writes > 0.95, "Too many write failures"
        
        total_reads = read_results['success'] + read_results['failure']
        assert read_results['success'] / total_reads > 0.95, "Too many read failures"
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_large_file_stress(self, stress_archive):
        """Test with extremely large individual files."""
        archive = stress_archive['archive']
        metrics = stress_archive['metrics']
        process = stress_archive['process']
        
        print("\n=== LARGE FILE STRESS TEST ===")
        print("Testing with 1GB+ individual files...")
        
        # Create a large DataFrame (approximately 1GB)
        rows = 5_000_000
        cols = 25
        
        print(f"Generating large DataFrame ({rows} rows x {cols} cols)...")
        large_df = pd.DataFrame(
            np.random.randn(rows, cols),
            columns=[f'col_{i}' for i in range(cols)]
        )
        
        # Add some string columns for variety
        large_df['symbol'] = np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], rows)
        large_df['timestamp'] = pd.date_range(
            start='2024-01-01', 
            periods=rows, 
            freq='1min'
        )
        
        # Estimate size
        size_mb = large_df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"DataFrame size in memory: {size_mb:.2f}MB")
        
        # Create record with large data
        large_record = RawDataRecord(
            source='stress_test',
            data_type='large_file',
            symbol='LARGE_TEST',
            timestamp=datetime.now(timezone.utc),
            data={'dataframe': large_df.to_dict('records')},
            metadata={'rows': rows, 'cols': cols, 'size_mb': size_mb}
        )
        
        # Monitor memory before write
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Write the large file
        print("Writing large file to archive...")
        start_time = time.time()
        
        try:
            success = await archive.save_raw_record_async(large_record)
            write_time = time.time() - start_time
            
            assert success, "Failed to write large file"
            
            # Monitor memory after write
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
            
            # Check the actual file size
            archive_path = Path(stress_archive['temp_dir'])
            parquet_files = list(archive_path.glob('**/*.parquet'))
            
            if parquet_files:
                file_size_mb = parquet_files[-1].stat().st_size / 1024 / 1024
                compression_ratio = size_mb / file_size_mb
            else:
                file_size_mb = 0
                compression_ratio = 0
            
            print(f"\n=== RESULTS ===")
            print(f"Write time: {write_time:.2f} seconds")
            print(f"Write speed: {size_mb / write_time:.2f}MB/sec")
            print(f"Memory used during write: {memory_used:.2f}MB")
            print(f"File size on disk: {file_size_mb:.2f}MB")
            print(f"Compression ratio: {compression_ratio:.2f}x")
            
            # Now try to read it back
            print("\nReading large file back...")
            read_start = time.time()
            
            records = await archive.query_raw_records(
                source='stress_test',
                data_type='large_file',
                symbol='LARGE_TEST'
            )
            
            read_time = time.time() - read_start
            
            assert len(records) == 1, f"Expected 1 record, got {len(records)}"
            assert 'dataframe' in records[0].data, "Data not properly stored"
            
            print(f"Read time: {read_time:.2f} seconds")
            print(f"Read speed: {size_mb / read_time:.2f}MB/sec")
            
            # Assertions
            assert write_time < 60, f"Write too slow: {write_time:.2f} seconds"
            assert read_time < 30, f"Read too slow: {read_time:.2f} seconds"
            assert compression_ratio > 1.5, f"Poor compression: {compression_ratio:.2f}x"
            
        except Exception as e:
            pytest.fail(f"Large file test failed: {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)  # 10 minute timeout
    async def test_sustained_load(self, stress_archive):
        """Test sustained load over extended period."""
        archive = stress_archive['archive']
        metrics = stress_archive['metrics']
        process = stress_archive['process']
        
        print("\n=== SUSTAINED LOAD TEST ===")
        print("Running continuous operations for 2 minutes...")
        
        duration_seconds = 120  # 2 minutes for testing (can be extended)
        write_interval = 0.01  # Write every 10ms
        read_interval = 0.05   # Read every 50ms
        
        # Track metrics over time
        memory_samples = []
        write_rates = []
        read_rates = []
        errors = []
        
        # Flags for async tasks
        stop_flag = False
        write_count = 0
        read_count = 0
        
        async def continuous_writer():
            """Continuously write records."""
            nonlocal write_count
            index = 0
            while not stop_flag:
                try:
                    record = self._generate_test_record(index)
                    success = await archive.save_raw_record_async(record)
                    if success:
                        write_count += 1
                    index += 1
                except Exception as e:
                    errors.append(f"Write error: {e}")
                await asyncio.sleep(write_interval)
        
        async def continuous_reader():
            """Continuously read records."""
            nonlocal read_count
            while not stop_flag:
                try:
                    symbol = f'TEST_{random.randint(0, 99)}'
                    records = await archive.query_raw_records(
                        source='stress_test',
                        data_type='test_data',
                        symbol=symbol,
                        start_date=datetime.now(timezone.utc) - timedelta(minutes=5)
                    )
                    read_count += 1
                except Exception as e:
                    errors.append(f"Read error: {e}")
                await asyncio.sleep(read_interval)
        
        async def monitor_resources():
            """Monitor resource usage over time."""
            sample_interval = 5  # Sample every 5 seconds
            while not stop_flag:
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)
                
                # Calculate rates
                if len(memory_samples) > 1:
                    time_elapsed = len(memory_samples) * sample_interval
                    write_rate = write_count / time_elapsed
                    read_rate = read_count / time_elapsed
                    write_rates.append(write_rate)
                    read_rates.append(read_rate)
                    
                    print(f"  [{len(memory_samples) * sample_interval}s] "
                          f"Memory: {memory_mb:.1f}MB, "
                          f"Writes: {write_rate:.1f}/s, "
                          f"Reads: {read_rate:.1f}/s")
                
                await asyncio.sleep(sample_interval)
        
        # Start all tasks
        start_time = time.time()
        
        tasks = [
            asyncio.create_task(continuous_writer()),
            asyncio.create_task(continuous_writer()),  # 2 writers
            asyncio.create_task(continuous_reader()),
            asyncio.create_task(continuous_reader()),   # 2 readers
            asyncio.create_task(monitor_resources())
        ]
        
        # Run for specified duration
        await asyncio.sleep(duration_seconds)
        stop_flag = True
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Analyze results
        if memory_samples:
            memory_growth = memory_samples[-1] - memory_samples[0]
            max_memory = max(memory_samples)
            avg_memory = np.mean(memory_samples)
            memory_std = np.std(memory_samples)
        else:
            memory_growth = max_memory = avg_memory = memory_std = 0
        
        # Get final metrics
        storage_metrics = await metrics.get_storage_metrics()
        operation_stats = metrics.get_operation_stats()
        
        print(f"\n=== RESULTS ===")
        print(f"Test duration: {total_time:.1f} seconds")
        print(f"Total writes: {write_count} ({write_count/total_time:.1f}/sec avg)")
        print(f"Total reads: {read_count} ({read_count/total_time:.1f}/sec avg)")
        print(f"Errors: {len(errors)}")
        print(f"Memory growth: {memory_growth:.1f}MB")
        print(f"Max memory: {max_memory:.1f}MB")
        print(f"Avg memory: {avg_memory:.1f}MB (std: {memory_std:.1f}MB)")
        print(f"Storage used: {storage_metrics.total_size_mb:.2f}MB")
        
        if write_rates:
            print(f"Write rate stability: min={min(write_rates):.1f}, max={max(write_rates):.1f}")
        if read_rates:
            print(f"Read rate stability: min={min(read_rates):.1f}, max={max(read_rates):.1f}")
        
        # Assertions
        assert write_count > 0, "No successful writes"
        assert read_count > 0, "No successful reads"
        assert len(errors) < write_count * 0.01, f"Too many errors: {len(errors)}"
        assert memory_growth < 100, f"Memory leak detected: {memory_growth:.1f}MB growth"
        
        # Check for performance degradation
        if len(write_rates) > 2:
            early_rate = np.mean(write_rates[:2])
            late_rate = np.mean(write_rates[-2:])
            degradation = (early_rate - late_rate) / early_rate * 100
            assert degradation < 20, f"Performance degraded by {degradation:.1f}%"
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_error_recovery_stress(self, stress_archive):
        """Test error handling and recovery under stress."""
        archive = stress_archive['archive']
        temp_dir = Path(stress_archive['temp_dir'])
        
        print("\n=== ERROR RECOVERY STRESS TEST ===")
        
        # Test 1: Corrupted file handling
        print("Testing corrupted file handling...")
        
        # Create a corrupted parquet file
        corrupted_path = temp_dir / 'raw' / 'stress_test' / 'test_data' / 'corrupted.parquet'
        corrupted_path.parent.mkdir(parents=True, exist_ok=True)
        corrupted_path.write_bytes(b'This is not a valid parquet file!')
        
        # Try to query (should handle gracefully)
        try:
            records = await archive.query_raw_records(
                source='stress_test',
                data_type='test_data'
            )
            print(f"  Handled corrupted file gracefully, returned {len(records)} valid records")
        except Exception as e:
            pytest.fail(f"Failed to handle corrupted file: {e}")
        
        # Test 2: Write with invalid data
        print("Testing invalid data handling...")
        
        invalid_records = [
            RawDataRecord(
                source='test',
                data_type='test',
                symbol='TEST',
                timestamp=datetime.now(timezone.utc),
                data={'invalid': float('inf')}  # Infinity not JSON serializable
            ),
            RawDataRecord(
                source='test',
                data_type='test',
                symbol='TEST',
                timestamp=datetime.now(timezone.utc),
                data={'invalid': float('nan')}  # NaN not JSON serializable
            ),
            RawDataRecord(
                source='test',
                data_type='test',
                symbol='TEST',
                timestamp=datetime.now(timezone.utc),
                data={'circular': None}  # Will add circular reference
            )
        ]
        
        # Add circular reference
        invalid_records[2].data['circular'] = invalid_records[2].data
        
        errors_handled = 0
        for record in invalid_records:
            try:
                result = await archive.save_raw_record_async(record)
                if not result:
                    errors_handled += 1
            except Exception:
                errors_handled += 1
        
        print(f"  Handled {errors_handled}/{len(invalid_records)} invalid records gracefully")
        
        # Test 3: Disk space simulation (create many files rapidly)
        print("Testing rapid file creation (simulating disk pressure)...")
        
        rapid_writes = 0
        rapid_errors = 0
        
        tasks = []
        for i in range(1000):
            record = self._generate_test_record(i, size_bytes=10000)  # 10KB each
            tasks.append(archive.save_raw_record_async(record))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                rapid_errors += 1
            elif result:
                rapid_writes += 1
            else:
                rapid_errors += 1
        
        print(f"  Rapid writes: {rapid_writes} success, {rapid_errors} errors")
        
        # Test 4: Concurrent access to same file
        print("Testing concurrent access to same file...")
        
        same_symbol = 'CONCURRENT_TEST'
        concurrent_tasks = []
        
        for i in range(50):
            record = RawDataRecord(
                source='stress_test',
                data_type='concurrent',
                symbol=same_symbol,  # All writing to same symbol
                timestamp=datetime.now(timezone.utc),
                data={'index': i},
                metadata={'thread': i}
            )
            concurrent_tasks.append(archive.save_raw_record_async(record))
        
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        concurrent_success = sum(1 for r in results if r is True)
        
        print(f"  Concurrent writes to same file: {concurrent_success}/50 succeeded")
        
        print("\n=== RESULTS ===")
        print("All error recovery tests completed")
        print("System demonstrated resilience to:")
        print("  - Corrupted files")
        print("  - Invalid data")
        print("  - Rapid file creation")
        print("  - Concurrent file access")
        
        # Assertions
        assert errors_handled > 0, "Should handle some invalid data"
        assert rapid_writes > 900, f"Too many failures in rapid writes: {rapid_errors}"
        assert concurrent_success > 45, f"Too many failures in concurrent access: {50 - concurrent_success}"


if __name__ == "__main__":
    # Run stress tests
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "-s", "--tb=short"],
        capture_output=False
    )
    sys.exit(result.returncode)