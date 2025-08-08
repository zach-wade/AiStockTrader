#!/usr/bin/env python3
"""
Archive Stress Tests (Simplified)

Push the archive system to its limits without external dependencies.
Focus on core stress testing functionality.

Run with: pytest tests/stress/test_archive_stress_simple.py -v
"""

import pytest
import pytest_asyncio
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
from typing import List, Dict, Any
import random
import string
import sys
sys.path.insert(0, 'src')

from main.data_pipeline.storage.archive import DataArchive, RawDataRecord


@pytest.mark.stress
class TestArchiveStressSimple:
    """Simplified stress tests for archive system."""
    
    @pytest_asyncio.fixture
    async def stress_archive(self):
        """Set up archive for stress testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = DataArchive(config={'local_path': temp_dir})
            
            # Track initial resource usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            yield {
                'archive': archive,
                'temp_dir': temp_dir,
                'process': process,
                'initial_memory': initial_memory
            }
            
            # Force cleanup
            gc.collect()
    
    def _generate_test_record(self, index: int) -> RawDataRecord:
        """Generate a test record."""
        return RawDataRecord(
            source='stress_test',
            data_type='test_data',
            symbol=f'TEST_{index % 100}',
            timestamp=datetime.now(timezone.utc),
            data={
                'index': index,
                'values': [random.random() for _ in range(100)],
                'text': ''.join(random.choices(string.ascii_letters, k=100))
            },
            metadata={'test_index': index}
        )
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_high_volume_writes(self, stress_archive):
        """Test high volume of writes."""
        archive = stress_archive['archive']
        process = stress_archive['process']
        
        print("\n=== HIGH VOLUME WRITE TEST ===")
        num_records = 10000  # Reduced for quick testing
        batch_size = 100
        
        start_time = time.time()
        success_count = 0
        
        for batch in range(num_records // batch_size):
            tasks = []
            for i in range(batch_size):
                record = self._generate_test_record(batch * batch_size + i)
                tasks.append(archive.save_raw_record_async(record))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count += sum(1 for r in results if r is True)
            
            if batch % 10 == 0:
                elapsed = time.time() - start_time
                rate = success_count / elapsed if elapsed > 0 else 0
                memory = process.memory_info().rss / 1024 / 1024
                print(f"  Progress: {success_count}/{batch * batch_size + batch_size} "
                      f"({rate:.0f} rec/s, Mem: {memory:.1f}MB)")
        
        total_time = time.time() - start_time
        write_rate = success_count / total_time
        
        print(f"\nResults:")
        print(f"  Records written: {success_count}/{num_records}")
        print(f"  Write rate: {write_rate:.0f} records/sec")
        print(f"  Total time: {total_time:.2f} seconds")
        
        assert success_count >= num_records * 0.95, f"Too many failures"
        assert write_rate > 50, f"Write rate too slow: {write_rate:.0f}/sec"
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_concurrent_operations(self, stress_archive):
        """Test concurrent reads and writes."""
        archive = stress_archive['archive']
        
        print("\n=== CONCURRENT OPERATIONS TEST ===")
        num_workers = 20
        operations_per_worker = 50
        
        # Seed with initial data
        for i in range(100):
            record = self._generate_test_record(i)
            await archive.save_raw_record_async(record)
        
        async def worker(worker_id: int):
            """Worker that does mixed operations."""
            successes = 0
            for i in range(operations_per_worker):
                if random.random() < 0.7:  # 70% writes
                    record = self._generate_test_record(worker_id * 1000 + i)
                    result = await archive.save_raw_record_async(record)
                    if result:
                        successes += 1
                else:  # 30% reads
                    try:
                        records = await archive.query_raw_records(
                            source='stress_test',
                            data_type='test_data',
                            symbol=f'TEST_{random.randint(0, 99)}'
                        )
                        successes += 1
                    except:
                        pass
            return successes
        
        start_time = time.time()
        tasks = [worker(i) for i in range(num_workers)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        total_successes = sum(results)
        total_operations = num_workers * operations_per_worker
        success_rate = total_successes / total_operations * 100
        
        print(f"\nResults:")
        print(f"  Total operations: {total_operations}")
        print(f"  Successful: {total_successes}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Time: {total_time:.2f} seconds")
        
        assert success_rate > 90, f"Success rate too low: {success_rate:.1f}%"
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_large_data_handling(self, stress_archive):
        """Test handling of large data objects."""
        archive = stress_archive['archive']
        process = stress_archive['process']
        
        print("\n=== LARGE DATA TEST ===")
        
        # Create a large DataFrame
        rows = 100000
        cols = 10
        df = pd.DataFrame(
            np.random.randn(rows, cols),
            columns=[f'col_{i}' for i in range(cols)]
        )
        
        size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"  DataFrame size: {size_mb:.2f}MB ({rows} rows x {cols} cols)")
        
        # Create record with large data
        large_record = RawDataRecord(
            source='stress_test',
            data_type='large_data',
            symbol='LARGE',
            timestamp=datetime.now(timezone.utc),
            data={'dataframe': df.to_dict('records')},
            metadata={'rows': rows, 'cols': cols}
        )
        
        # Write large data
        memory_before = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        success = await archive.save_raw_record_async(large_record)
        
        write_time = time.time() - start_time
        memory_after = process.memory_info().rss / 1024 / 1024
        
        print(f"\nWrite Results:")
        print(f"  Success: {success}")
        print(f"  Write time: {write_time:.2f} seconds")
        print(f"  Write speed: {size_mb / write_time:.2f}MB/sec")
        print(f"  Memory used: {memory_after - memory_before:.2f}MB")
        
        # Read it back
        start_time = time.time()
        records = await archive.query_raw_records(
            source='stress_test',
            data_type='large_data',
            symbol='LARGE'
        )
        read_time = time.time() - start_time
        
        print(f"\nRead Results:")
        print(f"  Records found: {len(records)}")
        print(f"  Read time: {read_time:.2f} seconds")
        print(f"  Read speed: {size_mb / read_time:.2f}MB/sec")
        
        assert success, "Failed to write large data"
        assert len(records) == 1, f"Expected 1 record, got {len(records)}"
        assert write_time < 30, f"Write too slow: {write_time:.2f}s"
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_sustained_load(self, stress_archive):
        """Test sustained operations over time."""
        archive = stress_archive['archive']
        process = stress_archive['process']
        
        print("\n=== SUSTAINED LOAD TEST ===")
        duration = 30  # 30 seconds
        
        stop_flag = False
        write_count = 0
        read_count = 0
        errors = 0
        
        async def writer():
            nonlocal write_count, errors
            index = 0
            while not stop_flag:
                try:
                    record = self._generate_test_record(index)
                    if await archive.save_raw_record_async(record):
                        write_count += 1
                    index += 1
                except:
                    errors += 1
                await asyncio.sleep(0.001)  # 1ms between writes
        
        async def reader():
            nonlocal read_count, errors
            while not stop_flag:
                try:
                    await archive.query_raw_records(
                        source='stress_test',
                        data_type='test_data',
                        symbol=f'TEST_{random.randint(0, 99)}'
                    )
                    read_count += 1
                except:
                    errors += 1
                await asyncio.sleep(0.01)  # 10ms between reads
        
        # Start workers
        tasks = [
            asyncio.create_task(writer()),
            asyncio.create_task(writer()),
            asyncio.create_task(reader()),
            asyncio.create_task(reader())
        ]
        
        initial_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Monitor for duration
        for i in range(duration // 5):
            await asyncio.sleep(5)
            memory = process.memory_info().rss / 1024 / 1024
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.0f}s] Writes: {write_count}, "
                  f"Reads: {read_count}, Errors: {errors}, "
                  f"Memory: {memory:.1f}MB")
        
        stop_flag = True
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        print(f"\nResults:")
        print(f"  Duration: {total_time:.1f} seconds")
        print(f"  Total writes: {write_count} ({write_count/total_time:.1f}/sec)")
        print(f"  Total reads: {read_count} ({read_count/total_time:.1f}/sec)")
        print(f"  Errors: {errors}")
        print(f"  Memory growth: {memory_growth:.1f}MB")
        
        assert write_count > 0, "No successful writes"
        assert read_count > 0, "No successful reads"
        assert errors < (write_count + read_count) * 0.01, "Too many errors"
        assert memory_growth < 100, f"Memory leak: {memory_growth:.1f}MB"


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "-s", "--tb=short"],
        capture_output=False
    )
    sys.exit(result.returncode)