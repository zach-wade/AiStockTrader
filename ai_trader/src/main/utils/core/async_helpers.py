# File: ai_trader/utils/async_helpers.py

import asyncio
import functools
import time
from typing import List, Coroutine, Any, Callable, TypeVar, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Thread pool for I/O bound operations
_thread_pool = ThreadPoolExecutor(max_workers=10)
# Process pool for CPU bound operations  
_process_pool = ProcessPoolExecutor(max_workers=4)


async def process_in_batches(
    items: List[Any], 
    worker_coro: Callable[[Any], Coroutine], 
    batch_size: int
):
    """
    Processes a list of items in asynchronous batches.
    
    :param items: The list of items to process.
    :param worker_coro: The async function to call for each item.
    :param batch_size: The number of items to process concurrently in each batch.
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        tasks = [worker_coro(item) for item in batch]
        await asyncio.gather(*tasks)


async def run_in_executor(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a synchronous function in a thread pool executor.
    
    This is useful for I/O bound operations that don't have async versions.
    
    Args:
        func: The synchronous function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function
    """
    loop = asyncio.get_event_loop()
    # Use functools.partial to handle kwargs
    if kwargs:
        func_with_args = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(_thread_pool, func_with_args)
    else:
        return await loop.run_in_executor(_thread_pool, func, *args)


async def run_cpu_bound_task(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a CPU-intensive function in a process pool executor.
    
    This is useful for CPU bound operations that would block the event loop.
    
    Args:
        func: The function to run (must be picklable)
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        func_with_args = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(_process_pool, func_with_args)
    else:
        return await loop.run_in_executor(_process_pool, func, *args)


async def gather_with_exceptions(
    *coros: Coroutine[Any, Any, T],
    return_exceptions: bool = True
) -> List[Union[T, Exception]]:
    """
    Gather multiple coroutines and handle exceptions gracefully.
    
    Args:
        *coros: Coroutines to run concurrently
        return_exceptions: If True, exceptions are returned in results. If False, first exception is raised.
        
    Returns:
        List of results (and exceptions if return_exceptions=True)
    """
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)

async def timeout_coro(
    coro: Coroutine[Any, Any, T],
    timeout_seconds: float,
    timeout_msg: Optional[str] = None
) -> T:
    """
    Run a coroutine with a timeout.
    
    Args:
        coro: The coroutine to run
        timeout_seconds: Timeout in seconds
        timeout_msg: Optional custom timeout message
        
    Returns:
        The result of the coroutine
        
    Raises:
        asyncio.TimeoutError: If the coroutine times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        if timeout_msg:
            logger.error(timeout_msg)
        raise


class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.
    
    Example:
        limiter = RateLimiter(rate=5, per=1.0)  # 5 requests per second
        async with limiter:
            await make_api_call()
    """
    
    def __init__(self, rate: int, per: float):
        """
        Args:
            rate: Number of requests allowed
            per: Time period in seconds
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        async with self._lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


async def chunk_async_generator(async_gen, chunk_size: int):
    """
    Convert an async generator into chunks.
    
    Args:
        async_gen: The async generator to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Lists of items from the generator
    """
    chunk = []
    async for item in async_gen:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    
    if chunk:  # Yield remaining items
        yield chunk


class AsyncCircuitBreaker:
    """
    Circuit breaker pattern for async functions.
    
    Prevents repeated calls to a failing service.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., Coroutine[Any, Any, T]], *args, **kwargs) -> T:
        """
        Call the function through the circuit breaker.
        """
        async with self._lock:
            if self.state == 'open':
                if (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = 'half-open'
                else:
                    raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self.state == 'half-open':
                    self.state = 'closed'
                self.failure_count = 0
            return result
            
        except self.expected_exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            raise


def async_lru_cache(maxsize: int = 128):
    """
    LRU cache decorator for async functions.
    
    Args:
        maxsize: Maximum cache size
    """
    def decorator(func):
        # Use functools.lru_cache on the wrapped function
        cached_func = functools.lru_cache(maxsize=maxsize)(func)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a hashable key from args and kwargs
            key = (args, tuple(sorted(kwargs.items())))
            return await cached_func(*key)
        
        # Expose cache info
        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear
        
        return wrapper
    return decorator


# Cleanup function to properly close thread pools
def cleanup_executors():
    """Cleanup thread and process pools. Call this before program exit."""
    _thread_pool.shutdown(wait=True)
    _process_pool.shutdown(wait=True)


def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying async functions on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        
    Returns:
        Decorated function that retries on exception
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator