"""
Request deduplication tracking for feature computation.

This module handles deduplication of feature requests to prevent
redundant computations and optimize resource usage.
"""

from datetime import datetime, timezone
from typing import Dict, List
import hashlib

from main.utils.core import get_logger
from main.events.handlers.feature_pipeline_helpers.feature_types import FeatureRequest
from main.events.handlers.feature_pipeline_helpers.queue_types import QueuedRequest

logger = get_logger(__name__)


class DeduplicationTracker:
    """
    Tracks recent requests to prevent duplicate processing.
    
    Features:
    - Time-window based deduplication
    - Request signature generation
    - Automatic cleanup of expired entries
    """
    
    def __init__(
        self,
        dedup_window_seconds: int = 60,
        cleanup_interval_seconds: int = 60
    ):
        """
        Initialize deduplication tracker.
        
        Args:
            dedup_window_seconds: Window for deduplication
            cleanup_interval_seconds: Interval for cleanup
        """
        self.dedup_window_seconds = dedup_window_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # Tracking structures
        self._recent_requests: Dict[str, datetime] = {}
        self._last_cleanup = datetime.now(timezone.utc)
        
        logger.debug(
            f"DeduplicationTracker initialized with window={dedup_window_seconds}s"
        )
    
    def is_duplicate(
        self,
        request: FeatureRequest,
        request_id: str,
        active_queue: List[QueuedRequest]
    ) -> bool:
        """
        Check if request is a duplicate.
        
        Args:
            request: Feature request to check
            request_id: Unique request ID
            active_queue: Current active queue for similarity check
            
        Returns:
            True if duplicate
        """
        # Check if exact ID exists
        if request_id in self._recent_requests:
            req_time = self._recent_requests[request_id]
            if (datetime.now(timezone.utc) - req_time).seconds < self.dedup_window_seconds:
                return True
        
        # Check for similar recent requests in active queue
        for queued in active_queue:
            if self._is_similar_request(request, queued.request):
                if (datetime.now(timezone.utc) - queued.queued_at).seconds < self.dedup_window_seconds:
                    return True
        
        return False
    
    def track_request(self, request_id: str) -> None:
        """
        Track a new request.
        
        Args:
            request_id: Unique request ID
        """
        self._recent_requests[request_id] = datetime.now(timezone.utc)
    
    def generate_request_id(self, request: FeatureRequest) -> str:
        """
        Generate unique request ID based on request content.
        
        Args:
            request: Feature request
            
        Returns:
            Unique request ID
        """
        # Create ID from request components
        components = [
            request.symbol,
            str(request.alert_type.value),
            str(sorted([g.value for g in request.feature_groups])),
            str(request.metadata.get('alert_timestamp', ''))
        ]
        
        hash_input = ':'.join(components)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired tracking data.
        
        Returns:
            Number of entries cleaned
        """
        now = datetime.now(timezone.utc)
        
        # Check if cleanup is needed
        if (now - self._last_cleanup).seconds < self.cleanup_interval_seconds:
            return 0
        
        # Find expired entries
        expired_ids = [
            req_id for req_id, timestamp in self._recent_requests.items()
            if (now - timestamp).seconds > self.dedup_window_seconds
        ]
        
        # Remove expired entries
        for req_id in expired_ids:
            del self._recent_requests[req_id]
        
        self._last_cleanup = now
        
        if expired_ids:
            logger.debug(f"Cleaned up {len(expired_ids)} expired request IDs")
        
        return len(expired_ids)
    
    def _is_similar_request(
        self,
        request1: FeatureRequest,
        request2: FeatureRequest
    ) -> bool:
        """
        Check if two requests are similar enough to be considered duplicates.
        
        Args:
            request1: First request
            request2: Second request
            
        Returns:
            True if similar
        """
        # Same symbol and same feature groups
        return (
            request1.symbol == request2.symbol and
            set(request1.feature_groups) == set(request2.feature_groups)
        )
    
    def get_tracked_count(self) -> int:
        """Get number of currently tracked requests."""
        return len(self._recent_requests)
    
    def clear(self) -> None:
        """Clear all tracking data."""
        self._recent_requests.clear()
        self._last_cleanup = datetime.now(timezone.utc)
        logger.debug("Deduplication tracker cleared")