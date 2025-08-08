"""
Queue data types for feature computation requests.

This module contains the core data types used in the request queue
management system including dataclasses and statistics types.
"""

from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from main.events.handlers.feature_pipeline_helpers.feature_types import FeatureRequest, FeatureGroup


@dataclass
class QueuedRequest:
    """Wrapper for queued feature requests with metadata."""
    request: FeatureRequest
    request_id: str
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempt_count: int = 0
    last_attempt: Optional[datetime] = None
    
    def __lt__(self, other):
        """Compare by priority (higher first) then by queue time (older first)."""
        if self.request.priority != other.request.priority:
            return self.request.priority > other.request.priority
        return self.queued_at < other.queued_at


@dataclass
class QueueStats:
    """Statistics for the request queue."""
    total_queued: int = 0
    total_processed: int = 0
    total_failed: int = 0
    total_expired: int = 0
    avg_queue_time_seconds: float = 0.0
    current_queue_size: int = 0
    requests_by_symbol: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    requests_by_group: Dict[FeatureGroup, int] = field(default_factory=lambda: defaultdict(int))