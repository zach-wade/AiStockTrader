"""
Batch Processor for Prediction Engine

Handles batch processing of predictions for efficient inference on large datasets.
"""

# Standard library imports
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Handles batch processing for prediction engine.

    Features:
    - Configurable batch sizes
    - Parallel processing support
    - Progress tracking
    - Error handling and retry logic
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_workers: int = 4,
        timeout: float = 30.0,
        retry_count: int = 3,
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
            max_workers: Maximum parallel workers
            timeout: Timeout per batch in seconds
            retry_count: Number of retries on failure
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.retry_count = retry_count

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Progress tracking
        self.progress = {
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "start_time": None,
            "end_time": None,
        }

    async def process_batch(
        self, items: List[Any], processor_func: Callable, **kwargs
    ) -> List[Any]:
        """
        Process a batch of items asynchronously.

        Args:
            items: List of items to process
            processor_func: Function to process each batch
            **kwargs: Additional arguments for processor function

        Returns:
            List of processed results
        """
        self.progress["total_items"] = len(items)
        self.progress["processed_items"] = 0
        self.progress["failed_items"] = 0
        self.progress["start_time"] = time.time()

        # Create batches
        batches = self._create_batches(items)
        logger.info(f"Processing {len(items)} items in {len(batches)} batches")

        # Process batches in parallel
        results = []
        tasks = []

        for i, batch in enumerate(batches):
            task = self._process_single_batch(batch, processor_func, i, **kwargs)
            tasks.append(task)

        # Execute with concurrency control
        batch_results = await self._execute_with_concurrency(tasks)

        # Flatten results
        for batch_result in batch_results:
            if batch_result is not None:
                results.extend(batch_result)

        self.progress["end_time"] = time.time()

        logger.info(
            f"Batch processing completed: {self.progress['processed_items']}/{self.progress['total_items']} items processed"
        )

        return results

    def _create_batches(self, items: List[Any]) -> List[List[Any]]:
        """Create batches from items."""
        batches = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batches.append(batch)
        return batches

    async def _process_single_batch(
        self, batch: List[Any], processor_func: Callable, batch_idx: int, **kwargs
    ) -> Optional[List[Any]]:
        """Process a single batch with retry logic."""
        for attempt in range(self.retry_count):
            try:
                # Run processor function
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, processor_func, batch, **kwargs)

                # Update progress
                self.progress["processed_items"] += len(batch)

                return result

            except Exception as e:
                logger.error(
                    f"Batch {batch_idx} failed (attempt {attempt + 1}/{self.retry_count}): {e}"
                )

                if attempt == self.retry_count - 1:
                    # Final attempt failed
                    self.progress["failed_items"] += len(batch)
                    logger.error(f"Batch {batch_idx} failed after {self.retry_count} attempts")
                    return None

                # Wait before retry
                await asyncio.sleep(2**attempt)

    async def _execute_with_concurrency(self, tasks: List) -> List[Any]:
        """Execute tasks with concurrency limit."""
        semaphore = asyncio.Semaphore(self.max_workers)

        async def bounded_task(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_task(task) for task in tasks]
        return await asyncio.gather(*bounded_tasks, return_exceptions=True)

    def process_dataframe_batch(
        self, df: pd.DataFrame, processor_func: Callable, **kwargs
    ) -> pd.DataFrame:
        """
        Process a DataFrame in batches.

        Args:
            df: Input DataFrame
            processor_func: Function to process each batch
            **kwargs: Additional arguments

        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df

        # Split into batches
        num_batches = (len(df) + self.batch_size - 1) // self.batch_size
        batch_results = []

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            try:
                # Process batch
                result = processor_func(batch_df, **kwargs)
                batch_results.append(result)

            except Exception as e:
                logger.error(f"Failed to process batch {i}: {e}")
                # Return empty DataFrame for failed batch
                batch_results.append(pd.DataFrame())

        # Combine results
        if batch_results:
            return pd.concat(batch_results, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress."""
        progress = self.progress.copy()

        if progress["start_time"] and progress["end_time"]:
            progress["duration"] = progress["end_time"] - progress["start_time"]
        elif progress["start_time"]:
            progress["duration"] = time.time() - progress["start_time"]
        else:
            progress["duration"] = 0

        if progress["total_items"] > 0:
            progress["success_rate"] = (progress["processed_items"] / progress["total_items"]) * 100
        else:
            progress["success_rate"] = 0

        return progress

    def reset_progress(self):
        """Reset progress tracking."""
        self.progress = {
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "start_time": None,
            "end_time": None,
        }

    def shutdown(self):
        """Shutdown the batch processor."""
        self.executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


# Convenience functions
def create_batch_processor(config: Optional[Dict[str, Any]] = None) -> BatchProcessor:
    """
    Create a batch processor with configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        BatchProcessor instance
    """
    if config is None:
        config = {}

    return BatchProcessor(
        batch_size=config.get("batch_size", 100),
        max_workers=config.get("max_workers", 4),
        timeout=config.get("timeout", 30.0),
        retry_count=config.get("retry_count", 3),
    )


async def process_predictions_batch(
    predictions: List[Dict], model_func: Callable, batch_size: int = 100
) -> List[Dict]:
    """
    Process predictions in batches.

    Args:
        predictions: List of prediction requests
        model_func: Model prediction function
        batch_size: Size of each batch

    Returns:
        List of prediction results
    """
    processor = BatchProcessor(batch_size=batch_size)

    try:
        results = await processor.process_batch(predictions, model_func)
        return results
    finally:
        processor.shutdown()
