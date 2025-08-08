# File: events/scanner_feature_bridge.py

"""
Scanner-Feature Bridge implementation.

Bridges the gap between scanner alerts and the feature pipeline by converting
scanner alerts into feature computation requests with intelligent batching,
prioritization, and deduplication.
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    RateLimiter,
    ensure_utc
)
from main.events.types import (
    Event,
    EventType,
    ScannerAlertEvent,
    FeatureRequestEvent,
    ScanAlert
)
from main.interfaces.events import IEventBus
from main.events.handlers.scanner_bridge_helpers.alert_feature_mapper import AlertFeatureMapper
from main.events.handlers.scanner_bridge_helpers.feature_request_batcher import FeatureRequestBatcher
from main.events.handlers.scanner_bridge_helpers.priority_calculator import PriorityCalculator
from main.events.handlers.scanner_bridge_helpers.request_dispatcher import RequestDispatcher
from main.events.handlers.scanner_bridge_helpers.bridge_stats_tracker import BridgeStatsTracker

logger = get_logger(__name__)


class ScannerFeatureBridge(ErrorHandlingMixin):
    """
    Bridges scanner alerts to feature computation requests.
    
    This component:
    - Listens for scanner alert events
    - Maps alerts to required feature groups
    - Batches requests for efficiency
    - Prioritizes based on alert importance
    - Dispatches feature computation requests
    - Tracks performance metrics
    """
    
    def __init__(
        self,
        event_bus: IEventBus,
        batch_size: int = 50,
        batch_timeout_seconds: float = 5.0,
        max_symbols_per_batch: int = 100,
        rate_limit_per_second: int = 10,
        dedup_window_seconds: int = 60,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the scanner-feature bridge.
        
        Args:
            event_bus: Event bus for communication
            batch_size: Number of symbols to batch before sending
            batch_timeout_seconds: Max time to wait before sending partial batch
            max_symbols_per_batch: Maximum symbols in a single request
            rate_limit_per_second: Max feature requests per second
            dedup_window_seconds: Window for deduplicating symbols
            config: Optional configuration dictionary
        """
        super().__init__()
        
        self.event_bus = event_bus
        self.config = config or {}
        
        # Configuration
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        self.max_symbols_per_batch = max_symbols_per_batch
        self.dedup_window_seconds = dedup_window_seconds
        
        # Helper components
        self.alert_mapper = AlertFeatureMapper()
        self.request_batcher = FeatureRequestBatcher(
            batch_size=batch_size
        )
        self.priority_calculator = PriorityCalculator()
        self.request_dispatcher = RequestDispatcher(event_bus)
        self.stats_tracker = BridgeStatsTracker()
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            rate=rate_limit_per_second,
            per=1.0  # per second
        )
        
        # Deduplication tracking
        self._recent_symbols: Dict[str, datetime] = {}
        self._dedup_lock = asyncio.Lock()
        
        # Batch processing
        self._batch_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(
            f"ScannerFeatureBridge initialized with batch_size={batch_size}, "
            f"timeout={batch_timeout_seconds}s, rate_limit={rate_limit_per_second}/s"
        )
    
    async def start(self):
        """Start the bridge and subscribe to events."""
        if self._running:
            logger.warning("ScannerFeatureBridge already running")
            return
        
        self._running = True
        
        # Subscribe to scanner alerts
        self.event_bus.subscribe(
            EventType.SCANNER_ALERT,
            self._handle_scanner_alert,
            priority=100  # High priority
        )
        
        # Start batch processor
        self._batch_task = asyncio.create_task(self._process_batches())
        
        logger.info("ScannerFeatureBridge started")
    
    async def stop(self):
        """Stop the bridge and clean up resources."""
        if not self._running:
            return
        
        logger.info("Stopping ScannerFeatureBridge...")
        self._running = False
        
        # Unsubscribe from events
        self.event_bus.unsubscribe(
            EventType.SCANNER_ALERT,
            self._handle_scanner_alert
        )
        
        # Cancel batch processor
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        
        # Process any remaining batches
        await self._flush_pending_batches()
        
        logger.info("ScannerFeatureBridge stopped")
    
    async def _handle_scanner_alert(self, event: Event):
        """
        Handle incoming scanner alert events.
        
        Args:
            event: ScannerAlertEvent containing alerts
        """
        if not isinstance(event, ScannerAlertEvent):
            logger.error(f"Expected ScannerAlertEvent, got {type(event)}")
            return
        
        try:
            # Update stats
            self.stats_tracker.increment_alerts_received()
            
            # Process each alert
            for alert in event.alerts:
                await self._process_alert(alert)
        except Exception as e:
            self.handle_error(e, "processing scanner alert")
    
    async def _process_alert(self, alert: ScanAlert):
        """
        Process a single scanner alert.
        
        Args:
            alert: Scanner alert to process
        """
        # Check deduplication
        if await self._is_duplicate(alert.symbol):
            logger.debug(f"Skipping duplicate symbol: {alert.symbol}")
            return
        
        # Map alert to feature groups
        feature_groups = self.alert_mapper.map_alert_to_features(alert)
        
        if not feature_groups:
            logger.debug(f"No features mapped for alert type: {alert.alert_type}")
            return
        
        # Calculate priority
        priority = self.priority_calculator.calculate_priority(alert)
        
        # Add to batch
        self.request_batcher.add_request(
            symbol=alert.symbol,
            feature_groups=feature_groups,
            priority=priority,
            metadata={
                'alert_type': alert.alert_type.value,
                'alert_score': alert.score,
                'source_scanner': alert.source_scanner
            }
        )
        
        # Track symbol
        self.stats_tracker.add_symbol_processed(alert.symbol)
    
    async def _is_duplicate(self, symbol: str) -> bool:
        """
        Check if symbol was recently processed.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if duplicate within dedup window
        """
        async with self._dedup_lock:
            now = ensure_utc(datetime.now())
            cutoff = now - timedelta(seconds=self.dedup_window_seconds)
            
            # Clean old entries
            self._recent_symbols = {
                s: t for s, t in self._recent_symbols.items()
                if t > cutoff
            }
            
            # Check if duplicate
            if symbol in self._recent_symbols:
                return True
            
            # Add symbol
            self._recent_symbols[symbol] = now
            return False
    
    async def _process_batches(self):
        """
        Background task to process batches periodically.
        """
        logger.debug("Batch processor started")
        
        while self._running:
            try:
                # Wait for timeout or interruption
                await asyncio.sleep(self.batch_timeout_seconds)
                
                # Process any ready batches
                await self._send_ready_batches()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}", exc_info=True)
        
        logger.debug("Batch processor stopped")
    
    async def _send_ready_batches(self):
        """Send all ready batches respecting rate limits."""
        batches = self.request_batcher.get_ready_batches()
        
        for batch in batches:
            # Apply rate limiting
            async with self.rate_limiter:
                # Dispatch the batch
                await self.request_dispatcher.dispatch_batch(batch)
                
                # Update stats
                self.stats_tracker.increment_feature_requests_sent()
    
    async def _flush_pending_batches(self):
        """Force send all pending batches (used during shutdown)."""
        logger.info("Flushing pending batches...")
        
        # Get all pending batches
        all_batches = self.request_batcher.flush_all()
        
        # Send them all
        for batch in all_batches:
            try:
                await self.request_dispatcher.dispatch_batch(batch)
                self.stats_tracker.increment_feature_requests_sent()
            except Exception as e:
                logger.error(f"Failed to flush batch: {e}")
        
        logger.info(f"Flushed {len(all_batches)} batches")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current bridge statistics.
        
        Returns:
            Dictionary containing bridge metrics
        """
        # Get stats from components
        pending_batches = self.request_batcher.get_pending_batch_count()
        pending_symbols = self.request_batcher.get_pending_symbol_count()
        
        stats = self.stats_tracker.get_stats(
            pending_batches_count=pending_batches,
            pending_symbols_count=pending_symbols
        )
        
        # Record gauge metrics
        record_metric("scanner_bridge.pending_batches", pending_batches, metric_type="gauge")
        record_metric("scanner_bridge.pending_symbols", pending_symbols, metric_type="gauge")
        record_metric("scanner_bridge.dedup_cache_size", len(self._recent_symbols), metric_type="gauge")
        
        # Add additional metrics
        stats.update({
            'is_running': self._running,
            'dedup_cache_size': len(self._recent_symbols),
            'rate_limit_per_second': self.rate_limiter._calls_per_second,
            'batch_config': {
                'batch_size': self.batch_size,
                'timeout_seconds': self.batch_timeout_seconds,
                'max_symbols_per_batch': self.max_symbols_per_batch
            }
        })
        
        return stats
    
    async def process_manual_request(
        self,
        symbols: List[str],
        feature_groups: List[str],
        priority: int = 0
    ):
        """
        Manually request features for specific symbols.
        
        This bypasses the alert system and directly creates feature requests.
        
        Args:
            symbols: List of symbols to compute features for
            feature_groups: List of feature groups to compute
            priority: Request priority
        """
        logger.info(
            f"Processing manual feature request for {len(symbols)} symbols, "
            f"groups: {feature_groups}"
        )
        
        # Create a feature request event
        event = FeatureRequestEvent(
            symbols=symbols,
            feature_groups=feature_groups,
            priority=priority,
            requester="manual_bridge_request"
        )
        
        # Publish directly
        await self.event_bus.publish(event)
        
        # Update stats
        self.stats_tracker.increment_feature_requests_sent()
        for symbol in symbols:
            self.stats_tracker.add_symbol_processed(symbol)