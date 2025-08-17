"""
Streaming Data Processor for Large DataFrame Operations

This module provides memory-efficient streaming processing capabilities for large datasets,
specifically designed for high-frequency trading data processing where memory usage is critical.
"""

# Standard library imports
import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import gc
import logging
import threading
import time
from typing import Protocol, TypeVar, runtime_checkable

# Third-party imports
import pandas as pd
import psutil

# Local imports
from main.utils.monitoring import get_memory_monitor, memory_profiled

logger = logging.getLogger(__name__)

T = TypeVar("T")
DataFrame = TypeVar("DataFrame", bound=pd.DataFrame)


@dataclass
class StreamingConfig:
    """Configuration for streaming data processing"""

    chunk_size: int = 10000  # Number of rows per chunk
    max_memory_mb: float = 500.0  # Maximum memory usage before triggering optimization
    buffer_size: int = 3  # Number of chunks to buffer
    parallel_workers: int = 2  # Number of parallel processing workers
    enable_gc_per_chunk: bool = True  # Trigger GC after each chunk
    compression: str = "gzip"  # Compression for temporary storage

    # Performance monitoring
    log_progress_every: int = 5  # Log progress every N chunks
    memory_check_interval: int = 1  # Check memory every N chunks


@dataclass
class ProcessingStats:
    """Statistics for streaming processing operations"""

    chunks_processed: int = 0
    total_rows: int = 0
    processing_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    gc_collections: int = 0
    errors: list[str] = field(default_factory=list)

    def add_chunk_stats(self, chunk_size: int, chunk_time: float, memory_mb: float):
        """Add statistics for a processed chunk"""
        self.chunks_processed += 1
        self.total_rows += chunk_size
        self.processing_time += chunk_time
        self.memory_peak_mb = max(self.memory_peak_mb, memory_mb)


@runtime_checkable
class StreamProcessor(Protocol):
    """Protocol for stream processing functions"""

    async def __call__(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a data chunk"""
        ...


class DataFrameStreamer:
    """Efficient streaming processor for large DataFrames"""

    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        self.memory_monitor = get_memory_monitor()
        self.stats = ProcessingStats()
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        self._lock = threading.Lock()

    async def stream_chunks(
        self, data_source: pd.DataFrame | str | Callable[[], pd.DataFrame]
    ) -> AsyncIterator[pd.DataFrame]:
        """
        Stream data in chunks for memory-efficient processing

        Args:
            data_source: DataFrame, file path, or callable that returns DataFrame

        Yields:
            DataFrame chunks of configured size
        """
        # Get the source DataFrame
        if isinstance(data_source, pd.DataFrame):
            df = data_source
        elif isinstance(data_source, str):
            # File path - read in chunks directly
            async for chunk in self._stream_from_file(data_source):
                yield chunk
            return
        elif callable(data_source):
            df = data_source()
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")

        # Stream DataFrame in chunks
        total_rows = len(df)
        chunk_size = self.config.chunk_size

        logger.info(f"Streaming {total_rows} rows in chunks of {chunk_size}")

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)

            # Extract chunk with memory optimization
            with self.memory_monitor.memory_context(f"chunk_extraction_{start_idx}"):
                chunk = df.iloc[start_idx:end_idx].copy()

                # Optimize chunk memory usage
                chunk = self.memory_monitor.optimize_dataframe_memory(chunk)

                yield chunk

            # Memory management
            if self.config.enable_gc_per_chunk:
                gc.collect()

            # Progress logging
            chunk_num = (start_idx // chunk_size) + 1
            total_chunks = (total_rows + chunk_size - 1) // chunk_size

            if chunk_num % self.config.log_progress_every == 0:
                logger.info(f"Streamed chunk {chunk_num}/{total_chunks}")

    async def _stream_from_file(self, file_path: str) -> AsyncIterator[pd.DataFrame]:
        """Stream data from file in chunks"""
        try:
            if file_path.endswith(".csv"):
                # Use pandas read_csv with chunksize for memory efficiency
                chunk_reader = pd.read_csv(
                    file_path, chunksize=self.config.chunk_size, compression="infer"
                )

                for chunk_num, chunk in enumerate(chunk_reader, 1):
                    # Optimize memory usage
                    chunk = self.memory_monitor.optimize_dataframe_memory(chunk)
                    yield chunk

                    if chunk_num % self.config.log_progress_every == 0:
                        logger.info(f"Streamed file chunk {chunk_num}")

            elif file_path.endswith(".parquet"):
                # Use pyarrow for efficient parquet streaming
                # Third-party imports
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(file_path)

                for batch_num, batch in enumerate(
                    parquet_file.iter_batches(batch_size=self.config.chunk_size)
                ):
                    chunk = batch.to_pandas()
                    chunk = self.memory_monitor.optimize_dataframe_memory(chunk)
                    yield chunk

                    if (batch_num + 1) % self.config.log_progress_every == 0:
                        logger.info(f"Streamed parquet batch {batch_num + 1}")

            else:
                raise ValueError(f"Unsupported file format: {file_path}")

        except Exception as e:
            logger.error(f"Error streaming from file {file_path}: {e}")
            raise

    @memory_profiled(include_gc=True)
    async def process_stream(
        self,
        data_source: pd.DataFrame | str | Callable[[], pd.DataFrame],
        processor: StreamProcessor,
        output_path: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Process data stream with memory-efficient operations

        Args:
            data_source: Data source to stream
            processor: Processing function for each chunk
            output_path: Optional path to save results incrementally

        Returns:
            Combined results if output_path is None, otherwise None
        """
        start_time = time.time()
        process = psutil.Process()
        self.stats.memory_start_mb = process.memory_info().rss / 1024 / 1024

        logger.info(f"Starting streaming processing with config: {self.config}")

        results = []
        chunk_buffer = deque(maxlen=self.config.buffer_size)

        try:
            async for chunk in self.stream_chunks(data_source):
                chunk_start_time = time.time()

                # Process chunk
                with self.memory_monitor.memory_context(
                    f"process_chunk_{self.stats.chunks_processed}"
                ):
                    processed_chunk = await self._process_chunk_safely(chunk, processor)

                    if processed_chunk is not None and not processed_chunk.empty:
                        if output_path:
                            # Write incrementally to file
                            await self._append_to_file(
                                processed_chunk, output_path, self.stats.chunks_processed == 0
                            )
                        else:
                            # Buffer in memory (with size limits)
                            chunk_buffer.append(processed_chunk)

                            # Check memory usage and flush if needed
                            current_memory = process.memory_info().rss / 1024 / 1024
                            if current_memory > self.config.max_memory_mb:
                                logger.warning(
                                    f"Memory usage {current_memory:.1f}MB exceeds limit, flushing buffer"
                                )
                                if chunk_buffer:
                                    results.extend(list(chunk_buffer))
                                    chunk_buffer.clear()
                                    gc.collect()

                # Update statistics
                chunk_time = time.time() - chunk_start_time
                current_memory = process.memory_info().rss / 1024 / 1024
                self.stats.add_chunk_stats(len(chunk), chunk_time, current_memory)

                # Memory cleanup
                del chunk, processed_chunk
                if self.config.enable_gc_per_chunk:
                    self.stats.gc_collections += 1
                    gc.collect()

                # Progress reporting
                if self.stats.chunks_processed % self.config.log_progress_every == 0:
                    self._log_progress()

            # Final processing
            if not output_path:
                # Combine remaining buffered chunks
                if chunk_buffer:
                    results.extend(list(chunk_buffer))

                if results:
                    logger.info(f"Combining {len(results)} processed chunks")
                    with self.memory_monitor.memory_context("combine_results"):
                        combined_result = pd.concat(results, ignore_index=True)
                        return self.memory_monitor.optimize_dataframe_memory(combined_result)
                else:
                    return pd.DataFrame()

        except Exception as e:
            self.stats.errors.append(str(e))
            logger.error(f"Error in streaming processing: {e}", exc_info=True)
            raise

        finally:
            # Final statistics
            self.stats.processing_time = time.time() - start_time
            self.stats.memory_end_mb = process.memory_info().rss / 1024 / 1024
            self._log_final_stats()

        return None

    async def _process_chunk_safely(
        self, chunk: pd.DataFrame, processor: StreamProcessor
    ) -> pd.DataFrame | None:
        """Process chunk with error handling and timeout"""
        try:
            # Run processor with timeout protection
            timeout = 60.0  # 1 minute timeout per chunk

            if asyncio.iscoroutinefunction(processor):
                result = await asyncio.wait_for(processor(chunk), timeout=timeout)
            else:
                # Run synchronous processor in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, processor, chunk), timeout=timeout
                )

            return result

        except TimeoutError:
            error_msg = f"Chunk processing timeout after {timeout}s"
            self.stats.errors.append(error_msg)
            logger.error(error_msg)
            return None

        except Exception as e:
            error_msg = f"Chunk processing error: {e}"
            self.stats.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return None

    async def _append_to_file(self, chunk: pd.DataFrame, output_path: str, first_chunk: bool):
        """Append processed chunk to output file"""
        try:
            mode = "w" if first_chunk else "a"
            header = first_chunk

            if output_path.endswith(".csv"):
                chunk.to_csv(
                    output_path,
                    mode=mode,
                    header=header,
                    index=False,
                    compression=self.config.compression,
                )
            elif output_path.endswith(".parquet"):
                # For parquet, we need to handle append mode differently
                if first_chunk:
                    chunk.to_parquet(output_path, compression=self.config.compression)
                else:
                    # Read existing, append, and write back (not ideal for large files)
                    existing = pd.read_parquet(output_path)
                    combined = pd.concat([existing, chunk], ignore_index=True)
                    combined.to_parquet(output_path, compression=self.config.compression)
            else:
                raise ValueError(f"Unsupported output format: {output_path}")

        except Exception as e:
            logger.error(f"Error appending to file {output_path}: {e}")
            raise

    def _log_progress(self):
        """Log current processing progress"""
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024

        logger.info(
            f"Processed {self.stats.chunks_processed} chunks, "
            f"{self.stats.total_rows:,} rows, "
            f"Memory: {current_memory:.1f}MB, "
            f"Avg time/chunk: {self.stats.processing_time/max(self.stats.chunks_processed, 1):.3f}s"
        )

    def _log_final_stats(self):
        """Log final processing statistics"""
        logger.info("Streaming processing completed:")
        logger.info(f"  Chunks processed: {self.stats.chunks_processed}")
        logger.info(f"  Total rows: {self.stats.total_rows:,}")
        logger.info(f"  Total time: {self.stats.processing_time:.2f}s")
        logger.info(
            f"  Avg time per chunk: {self.stats.processing_time/max(self.stats.chunks_processed, 1):.3f}s"
        )
        logger.info(f"  Memory start: {self.stats.memory_start_mb:.1f}MB")
        logger.info(f"  Memory peak: {self.stats.memory_peak_mb:.1f}MB")
        logger.info(f"  Memory end: {self.stats.memory_end_mb:.1f}MB")
        logger.info(f"  GC collections: {self.stats.gc_collections}")
        logger.info(f"  Errors: {len(self.stats.errors)}")

    def get_stats(self) -> ProcessingStats:
        """Get processing statistics"""
        return self.stats

    async def close(self):
        """Cleanup resources"""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


class StreamingAggregator:
    """Streaming aggregation for large datasets"""

    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        self.memory_monitor = get_memory_monitor()

    async def aggregate_streaming(
        self,
        data_source: pd.DataFrame | str | Callable[[], pd.DataFrame],
        group_by: str | list[str],
        aggregations: dict[str, str | list[str]],
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """
        Perform aggregations on streaming data

        Args:
            data_source: Data source to aggregate
            group_by: Column(s) to group by
            aggregations: Dict mapping column names to aggregation functions
            output_path: Optional path to save results

        Returns:
            Aggregated DataFrame
        """
        logger.info(f"Starting streaming aggregation grouped by {group_by}")

        # Initialize streaming processor
        streamer = DataFrameStreamer(self.config)

        # Accumulator for partial aggregations
        partial_aggregations = {}

        async def aggregate_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
            """Aggregate a single chunk"""
            if chunk.empty:
                return pd.DataFrame()

            # Perform chunk aggregation
            chunk_agg = chunk.groupby(group_by).agg(aggregations)

            # Update running aggregations using vectorized operations
            chunk_records = chunk_agg.to_dict("index")

            for group_key, group_data in chunk_records.items():
                if group_key not in partial_aggregations:
                    partial_aggregations[group_key] = group_data
                else:
                    # Combine with existing aggregation
                    existing = partial_aggregations[group_key]
                    for col, value in group_data.items():
                        if col.endswith("_count") or col.endswith("_sum"):
                            existing[col] = existing.get(col, 0) + value
                        elif col.endswith("_mean"):
                            # Recalculate mean (simplified - assumes count is tracked)
                            existing[col] = (existing.get(col, 0) + value) / 2
                        elif col.endswith("_max"):
                            existing[col] = max(existing.get(col, value), value)
                        elif col.endswith("_min"):
                            existing[col] = min(existing.get(col, value), value)

            return pd.DataFrame()  # Return empty since we're accumulating

        # Process stream
        await streamer.process_stream(data_source, aggregate_chunk)

        # Convert accumulated results to DataFrame
        if partial_aggregations:
            result_df = pd.DataFrame.from_dict(partial_aggregations, orient="index")
            result_df.index.name = group_by if isinstance(group_by, str) else None

            if output_path:
                result_df.to_csv(output_path)
                logger.info(f"Streaming aggregation saved to {output_path}")

            return result_df
        else:
            return pd.DataFrame()


class StreamingJoiner:
    """Memory-efficient joins for large datasets"""

    def __init__(self, config: StreamingConfig = None):
        self.config = config or StreamingConfig()
        self.memory_monitor = get_memory_monitor()

    async def streaming_join(
        self,
        left_source: pd.DataFrame | str,
        right_source: pd.DataFrame | str,
        join_keys: str | list[str],
        how: str = "inner",
        output_path: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Perform memory-efficient join of large datasets

        Args:
            left_source: Left DataFrame or file path
            right_source: Right DataFrame or file path
            join_keys: Column(s) to join on
            how: Join type ('inner', 'left', 'right', 'outer')
            output_path: Optional path to save results

        Returns:
            Joined DataFrame if output_path is None
        """
        logger.info(f"Starting streaming join on {join_keys}")

        # Load right dataset into memory (assuming it's smaller)
        if isinstance(right_source, str):
            right_df = pd.read_csv(right_source)
        else:
            right_df = right_source

        right_df = self.memory_monitor.optimize_dataframe_memory(right_df)
        logger.info(f"Loaded right dataset: {len(right_df):,} rows")

        # Stream and join left dataset
        streamer = DataFrameStreamer(self.config)

        async def join_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
            """Join chunk with right dataset"""
            if chunk.empty:
                return pd.DataFrame()

            # Perform join
            joined = chunk.merge(right_df, on=join_keys, how=how)
            return self.memory_monitor.optimize_dataframe_memory(joined)

        # Process streaming join
        result = await streamer.process_stream(left_source, join_chunk, output_path)

        logger.info("Streaming join completed")
        return result


# Convenience functions
async def stream_process_dataframe(
    df: pd.DataFrame,
    processor: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: int = 10000,
    max_memory_mb: float = 500.0,
) -> pd.DataFrame:
    """
    Convenience function for streaming DataFrame processing

    Args:
        df: DataFrame to process
        processor: Processing function
        chunk_size: Size of each chunk
        max_memory_mb: Memory limit

    Returns:
        Processed DataFrame
    """
    config = StreamingConfig(chunk_size=chunk_size, max_memory_mb=max_memory_mb)

    streamer = DataFrameStreamer(config)

    try:
        result = await streamer.process_stream(df, processor)
        return result or pd.DataFrame()
    finally:
        await streamer.close()


@asynccontextmanager
async def streaming_context(config: StreamingConfig = None):
    """Context manager for streaming processing"""
    streamer = DataFrameStreamer(config)
    try:
        yield streamer
    finally:
        await streamer.close()


# Integration with existing feature orchestrator
def optimize_feature_calculation(original_func: Callable) -> Callable:
    """
    Decorator to optimize feature calculation using streaming processing
    """

    async def wrapper(self, symbol_data: pd.DataFrame, *args, **kwargs):
        # If data is large, use streaming processing
        if len(symbol_data) > 50000:  # 50k rows threshold
            logger.info(f"Using streaming processing for large dataset: {len(symbol_data):,} rows")

            def feature_processor(chunk: pd.DataFrame) -> pd.DataFrame:
                return original_func(self, chunk, *args, **kwargs)

            # Use streaming processing
            result = await stream_process_dataframe(
                symbol_data, feature_processor, chunk_size=10000, max_memory_mb=300.0
            )
            return result
        else:
            # Use original function for smaller datasets
            return original_func(self, symbol_data, *args, **kwargs)

    return wrapper
