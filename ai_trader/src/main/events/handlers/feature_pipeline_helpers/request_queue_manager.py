"""
Feature computation request queue management.

This module manages the queue of feature computation requests,
handling prioritization, deduplication, and load balancing.
"""

import asyncio
import heapq
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from collections import defaultdict

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    timer
)
from main.utils.monitoring import record_metric

from main.events.handlers.feature_pipeline_helpers.feature_types import FeatureRequest
from main.events.handlers.feature_pipeline_helpers.queue_types import QueuedRequest, QueueStats
from main.events.handlers.feature_pipeline_helpers.deduplication_tracker import DeduplicationTracker

logger = get_logger(__name__)


class RequestQueueManager(ErrorHandlingMixin):
    """
    Manages the queue of feature computation requests.
    
    Features:
    - Priority-based queueing
    - Request deduplication
    - TTL-based expiration
    - Load balancing across symbols
    - Queue statistics and monitoring
    - Backpressure handling
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        request_ttl_seconds: int = 300,  # 5 minutes
        dedup_window_seconds: int = 60,  # 1 minute
        max_requests_per_symbol: int = 10
    ):
        """
        Initialize request queue manager.
        
        Args:
            max_queue_size: Maximum number of queued requests
            request_ttl_seconds: Time-to-live for requests
            dedup_window_seconds: Window for deduplication
            max_requests_per_symbol: Max concurrent requests per symbol
        """
        super().__init__()
        
        # Configuration
        self.max_queue_size = max_queue_size
        self.request_ttl_seconds = request_ttl_seconds
        self.max_requests_per_symbol = max_requests_per_symbol
        
        # Priority queue
        self._queue: List[QueuedRequest] = []
        self._request_map: Dict[str, QueuedRequest] = {}
        
        # Deduplication tracker
        self._dedup_tracker = DeduplicationTracker(
            dedup_window_seconds=dedup_window_seconds
        )
        
        # Symbol tracking for load balancing
        self._active_requests_by_symbol: Dict[str, int] = defaultdict(int)
        
        # Statistics
        self._stats = QueueStats()
        self._queue_times: List[float] = []  # For averaging
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.debug(
            f"RequestQueueManager initialized with max_size={max_queue_size}, "
            f"ttl={request_ttl_seconds}s"
        )
    
    @timer
    async def enqueue_request(
        self,
        request: FeatureRequest,
        request_id: Optional[str] = None
    ) -> bool:
        """
        Add a feature request to the queue.
        
        Args:
            request: Feature request to queue
            request_id: Optional unique request ID
            
        Returns:
            Success status
        """
        async with self._lock:
            try:
                # Clean up periodically
                self._dedup_tracker.cleanup_expired()
                
                # Generate request ID if not provided
                if not request_id:
                    request_id = self._dedup_tracker.generate_request_id(request)
                
                # Check for duplicates
                if self._dedup_tracker.is_duplicate(request, request_id, self._queue):
                    logger.debug(f"Skipping duplicate request: {request_id}")
                    return False
                
                # Check queue capacity
                if len(self._queue) >= self.max_queue_size:
                    logger.warning(
                        f"Queue full ({self.max_queue_size}), rejecting request"
                    )
                    record_metric('request_queue.rejected', 1)
                    return False
                
                # Check per-symbol limit
                if self._active_requests_by_symbol[request.symbol] >= self.max_requests_per_symbol:
                    logger.warning(
                        f"Too many requests for {request.symbol}, rejecting"
                    )
                    return False
                
                # Create queued request
                queued = QueuedRequest(
                    request=request,
                    request_id=request_id
                )
                
                # Add to queue and tracking
                heapq.heappush(self._queue, queued)
                self._request_map[request_id] = queued
                self._dedup_tracker.track_request(request_id)
                
                # Update stats
                self._stats.total_queued += 1
                self._stats.current_queue_size = len(self._queue)
                self._stats.requests_by_symbol[request.symbol] += 1
                
                for group in request.feature_groups:
                    self._stats.requests_by_group[group] += 1
                
                # Record metrics
                record_metric(
                    'request_queue.enqueued',
                    1,
                    tags={
                        'symbol': request.symbol,
                        'priority': request.priority
                    }
                )
                
                logger.debug(
                    f"Enqueued request {request_id} for {request.symbol} "
                    f"with priority {request.priority}"
                )
                
                return True
            except Exception as e:
                self.handle_error(e, "enqueueing request")
                return False
    
    async def dequeue_request(
        self,
        worker_id: Optional[int] = None
    ) -> Optional[QueuedRequest]:
        """
        Get the next request from the queue.
        
        Args:
            worker_id: Optional worker ID for logging
            
        Returns:
            Next queued request or None if empty
        """
        async with self._lock:
            try:
                # Find next valid request
                while self._queue:
                    queued = heapq.heappop(self._queue)
                    
                    # Check if expired
                    if self._is_expired(queued):
                        self._handle_expired_request(queued)
                        continue
                    
                    # Check symbol limit
                    if self._active_requests_by_symbol[queued.request.symbol] >= self.max_requests_per_symbol:
                        # Put back in queue
                        heapq.heappush(self._queue, queued)
                        
                        # Try to find request for different symbol
                        alternative = self._find_alternative_request()
                        if alternative:
                            queued = alternative
                        else:
                            return None
                    
                    # Update tracking
                    queued.attempt_count += 1
                    queued.last_attempt = datetime.now(timezone.utc)
                    self._active_requests_by_symbol[queued.request.symbol] += 1
                    
                    # Calculate queue time
                    queue_time = (queued.last_attempt - queued.queued_at).total_seconds()
                    self._queue_times.append(queue_time)
                    
                    # Update stats
                    self._stats.current_queue_size = len(self._queue)
                    if len(self._queue_times) > 1000:
                        self._queue_times = self._queue_times[-1000:]  # Keep last 1000
                    
                    # Record metrics
                    record_metric(
                        'request_queue.dequeued',
                        1,
                        tags={
                            'worker_id': worker_id,
                            'symbol': queued.request.symbol,
                            'queue_time': queue_time
                        }
                    )
                    
                    logger.debug(
                        f"Worker {worker_id} dequeued request {queued.request_id} "
                        f"(queued for {queue_time:.1f}s)"
                    )
                    
                    return queued
                
                return None
            except Exception as e:
                self.handle_error(e, "dequeueing request")
                return None
    
    async def complete_request(
        self,
        request_id: str,
        success: bool = True
    ) -> None:
        """
        Mark a request as completed.
        
        Args:
            request_id: Request ID
            success: Whether request completed successfully
        """
        async with self._lock:
            if request_id in self._request_map:
                queued = self._request_map[request_id]
                
                # Update symbol tracking
                self._active_requests_by_symbol[queued.request.symbol] -= 1
                if self._active_requests_by_symbol[queued.request.symbol] <= 0:
                    del self._active_requests_by_symbol[queued.request.symbol]
                
                # Remove from tracking
                del self._request_map[request_id]
                
                # Update stats
                if success:
                    self._stats.total_processed += 1
                else:
                    self._stats.total_failed += 1
                
                # Record metrics
                record_metric(
                    'request_queue.completed',
                    1,
                    tags={
                        'success': success,
                        'symbol': queued.request.symbol
                    }
                )
                
                logger.debug(
                    f"Completed request {request_id} "
                    f"(success={success})"
                )
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)
    
    def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        # Calculate average queue time
        if self._queue_times:
            self._stats.avg_queue_time_seconds = sum(self._queue_times) / len(self._queue_times)
        
        return self._stats
    
    def get_queued_symbols(self) -> List[str]:
        """Get list of symbols with queued requests."""
        symbols = set()
        for queued in self._queue:
            symbols.add(queued.request.symbol)
        return sorted(symbols)
    
    async def clear_symbol_requests(self, symbol: str) -> int:
        """
        Clear all requests for a specific symbol.
        
        Args:
            symbol: Symbol to clear
            
        Returns:
            Number of requests cleared
        """
        async with self._lock:
            cleared = 0
            
            # Remove from queue
            new_queue = []
            for queued in self._queue:
                if queued.request.symbol != symbol:
                    new_queue.append(queued)
                else:
                    del self._request_map[queued.request_id]
                    cleared += 1
            
            # Rebuild heap
            self._queue = new_queue
            heapq.heapify(self._queue)
            
            # Clear active tracking
            if symbol in self._active_requests_by_symbol:
                del self._active_requests_by_symbol[symbol]
            
            # Update stats
            self._stats.current_queue_size = len(self._queue)
            if symbol in self._stats.requests_by_symbol:
                self._stats.requests_by_symbol[symbol] = 0
            
            logger.info(f"Cleared {cleared} requests for {symbol}")
            
            return cleared
    
    def _is_expired(self, queued: QueuedRequest) -> bool:
        """Check if request has expired."""
        age = (datetime.now(timezone.utc) - queued.queued_at).seconds
        return age > self.request_ttl_seconds
    
    def _handle_expired_request(self, queued: QueuedRequest) -> None:
        """Handle an expired request."""
        # Remove from tracking
        if queued.request_id in self._request_map:
            del self._request_map[queued.request_id]
        
        # Update stats
        self._stats.total_expired += 1
        
        # Record metric
        record_metric(
            'request_queue.expired',
            1,
            tags={'symbol': queued.request.symbol}
        )
        
        logger.warning(
            f"Request {queued.request_id} expired after "
            f"{self.request_ttl_seconds}s"
        )
    
    def _find_alternative_request(self) -> Optional[QueuedRequest]:
        """Find request for different symbol when current is blocked."""
        # Look through queue for different symbol
        temp_queue = []
        alternative = None
        
        while self._queue and not alternative:
            candidate = heapq.heappop(self._queue)
            
            if self._active_requests_by_symbol[candidate.request.symbol] < self.max_requests_per_symbol:
                alternative = candidate
                break
            else:
                temp_queue.append(candidate)
        
        # Restore queue
        for item in temp_queue:
            heapq.heappush(self._queue, item)
        
        return alternative