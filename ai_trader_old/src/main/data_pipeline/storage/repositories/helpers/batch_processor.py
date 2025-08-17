"""
Batch Processor Helper

Handles efficient batch processing of large datasets with chunking and parallel execution.
"""

# Standard library imports
import asyncio
from collections.abc import AsyncGenerator, Callable
import time
from typing import Any, Generic, TypeVar

# Local imports
from main.utils.core import chunk_list, get_logger
from main.utils.monitoring import timer

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor(Generic[T, R]):
    """
    Processes large batches of data efficiently with chunking,
    parallel processing, and error handling.
    """

    def __init__(
        self,
        batch_size: int = 1000,
        max_parallel: int = 4,
        retry_failed: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        """
        Initialize the BatchProcessor.

        Args:
            batch_size: Size of each batch
            max_parallel: Maximum parallel workers
            retry_failed: Whether to retry failed items
            progress_callback: Optional callback for progress updates
        """
        self.batch_size = batch_size
        self.max_parallel = max_parallel
        self.retry_failed = retry_failed
        self.progress_callback = progress_callback

        # Statistics
        self.total_processed = 0
        self.total_succeeded = 0
        self.total_failed = 0
        self.processing_time = 0.0

        logger.debug(
            f"BatchProcessor initialized: batch_size={batch_size}, " f"max_parallel={max_parallel}"
        )

    async def process_batch(
        self,
        items: list[T],
        processor_func: Callable[[list[T]], R],
        validate_func: Callable[[T], bool] | None = None,
    ) -> dict[str, Any]:
        """
        Process a batch of items.

        Args:
            items: Items to process
            processor_func: Async function to process each batch
            validate_func: Optional validation function

        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()

        # Reset statistics
        self.total_processed = 0
        self.total_succeeded = 0
        self.total_failed = 0

        # Validate items if validator provided
        if validate_func:
            valid_items = [item for item in items if validate_func(item)]
            invalid_count = len(items) - len(valid_items)
            if invalid_count > 0:
                logger.warning(f"Filtered out {invalid_count} invalid items")
            items = valid_items

        # Split into chunks
        chunks = chunk_list(items, self.batch_size)
        total_chunks = len(chunks)

        logger.info(f"Processing {len(items)} items in {total_chunks} chunks")

        # Process chunks with concurrency limit
        results = []
        failed_items = []

        with timer("batch_processing"):
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_parallel)

            async def process_chunk_with_limit(chunk: list[T], chunk_idx: int) -> dict[str, Any]:
                async with semaphore:
                    return await self._process_single_chunk(
                        chunk, chunk_idx, total_chunks, processor_func
                    )

            # Process all chunks
            tasks = [process_chunk_with_limit(chunk, i) for i, chunk in enumerate(chunks)]

            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results and failures
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"Chunk {i} failed with exception: {result}")
                    failed_items.extend(chunks[i])
                    self.total_failed += len(chunks[i])
                elif isinstance(result, dict):
                    if result.get("success"):
                        results.append(result.get("data"))
                        self.total_succeeded += result.get("count", 0)
                    else:
                        failed_items.extend(chunks[i])
                        self.total_failed += len(chunks[i])

        # Retry failed items if enabled
        if self.retry_failed and failed_items:
            logger.info(f"Retrying {len(failed_items)} failed items")
            retry_results = await self._retry_failed_items(failed_items, processor_func)
            results.extend(retry_results.get("results", []))
            self.total_succeeded += retry_results.get("succeeded", 0)
            self.total_failed = retry_results.get("failed", 0)

        self.processing_time = time.time() - start_time
        self.total_processed = self.total_succeeded + self.total_failed

        return {
            "success": self.total_failed == 0,
            "results": results,
            "statistics": {
                "total_items": len(items),
                "processed": self.total_processed,
                "succeeded": self.total_succeeded,
                "failed": self.total_failed,
                "chunks_processed": total_chunks,
                "processing_time": self.processing_time,
                "items_per_second": (
                    self.total_processed / self.processing_time if self.processing_time > 0 else 0
                ),
            },
        }

    async def _process_single_chunk(
        self,
        chunk: list[T],
        chunk_idx: int,
        total_chunks: int,
        processor_func: Callable[[list[T]], R],
    ) -> dict[str, Any]:
        """Process a single chunk of items."""
        try:
            # Report progress
            if self.progress_callback:
                self.progress_callback(chunk_idx + 1, total_chunks)

            # Process the chunk
            result = await processor_func(chunk)

            return {"success": True, "data": result, "count": len(chunk), "chunk_idx": chunk_idx}

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx}: {e}")
            return {"success": False, "error": str(e), "count": 0, "chunk_idx": chunk_idx}

    async def _retry_failed_items(
        self, failed_items: list[T], processor_func: Callable[[list[T]], R]
    ) -> dict[str, Any]:
        """Retry processing of failed items with smart batching."""
        retry_results = []
        retry_succeeded = 0
        retry_failed = 0

        # First try with smaller batches (half the original batch size)
        smaller_batch_size = max(1, self.batch_size // 2)
        retry_chunks = chunk_list(failed_items, smaller_batch_size)

        still_failed_items = []

        # Try processing in smaller batches first
        for chunk in retry_chunks:
            try:
                result = await processor_func(chunk)
                retry_results.append(result)
                retry_succeeded += len(chunk)
            except Exception as e:
                logger.debug(f"Smaller batch retry failed, will try individually: {e}")
                still_failed_items.extend(chunk)

        # Only process individually if smaller batches also failed
        if still_failed_items:
            logger.debug(f"Processing {len(still_failed_items)} items individually as last resort")
            for item in still_failed_items:
                try:
                    result = await processor_func([item])
                    retry_results.append(result)
                    retry_succeeded += 1
                except Exception as e:
                    logger.debug(f"Individual retry failed for item: {e}")
                    retry_failed += 1

        return {"results": retry_results, "succeeded": retry_succeeded, "failed": retry_failed}

    def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_processed": self.total_processed,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "success_rate": (
                self.total_succeeded / self.total_processed if self.total_processed > 0 else 0
            ),
            "processing_time": self.processing_time,
            "throughput": (
                self.total_processed / self.processing_time if self.processing_time > 0 else 0
            ),
        }

    async def process_stream(
        self,
        stream_generator: AsyncGenerator[T, None],
        processor_func: Callable[[list[T]], R],
        buffer_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Process items from an async stream.

        Args:
            stream_generator: Async generator yielding items
            processor_func: Function to process batches
            buffer_size: Buffer size before processing (default: batch_size)

        Returns:
            Processing results
        """
        buffer_size = buffer_size or self.batch_size
        buffer = []
        all_results = []

        async for item in stream_generator:
            buffer.append(item)

            if len(buffer) >= buffer_size:
                # Process buffer
                result = await self.process_batch(buffer, processor_func)
                all_results.extend(result.get("results", []))
                buffer = []

        # Process remaining items
        if buffer:
            result = await self.process_batch(buffer, processor_func)
            all_results.extend(result.get("results", []))

        return {
            "success": self.total_failed == 0,
            "results": all_results,
            "statistics": self.get_statistics(),
        }
