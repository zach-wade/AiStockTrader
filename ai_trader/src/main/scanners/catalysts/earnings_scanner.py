# File: ai_trader/scanners/earnings_scanner.py

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import asyncio
from collections import defaultdict

from omegaconf import DictConfig
from main.interfaces.scanners import IScanner, IScannerRepository
from main.interfaces.events import IEventBus
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.utils.scanners import (
    ScannerDataAccess,
    ScannerCacheManager,
    ScannerMetricsCollector
)
from main.utils.core import timer
from ..catalyst_scanner_base import CatalystScannerBase
from main.events.types import AlertType, ScanAlert

logger = logging.getLogger(__name__)

class EarningsScanner(CatalystScannerBase):
    """
    Scans a universe of symbols for upcoming earnings announcements, which are
    often significant, short-term catalysts.
    
    Now uses the repository pattern with hot/cold storage awareness to
    efficiently access earnings data and historical patterns.
    """

    def __init__(
        self,
        config: DictConfig,
        repository: IScannerRepository,
        event_bus: Optional[IEventBus] = None,
        metrics_collector: Optional[ScannerMetricsCollector] = None,
        cache_manager: Optional[ScannerCacheManager] = None
    ):
        """
        Initializes the EarningsScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "EarningsScanner",
            config,
            repository,
            event_bus,
            metrics_collector,
            cache_manager
        )
        
        # Scanner-specific parameters
        self.params = self.config.get('scanners.earnings', {})
        self.days_ahead_threshold = self.params.get('days_ahead', 5)
        self.use_cache = self.params.get('use_cache', True)
        self.catalyst_weights = self.params.get('weights', {
            'today': 5.0,
            'upcoming': 3.0
        })
        
        # Track initialization
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize scanner resources."""
        if self._initialized:
            return
        
        logger.info(f"Initializing {self.name}")
        self._initialized = True
    
    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        logger.info(f"Cleaning up {self.name}")
        self._initialized = False

    async def scan(self, symbols: List[str], **kwargs) -> List[ScanAlert]:
        """
        Scan symbols for upcoming earnings announcements.
        
        Uses repository pattern for efficient earnings data access with hot storage
        for upcoming earnings and cold storage for historical patterns.
        
        Args:
            symbols: List of stock symbols to scan
            **kwargs: Additional scanner-specific parameters
            
        Returns:
            List of ScanAlert objects for earnings signals
        """
        if not self._initialized:
            await self.initialize()
            
        with timer() as t:
            logger.info(f"📈 Earnings Scanner: Checking {len(symbols)} symbols for upcoming earnings...")
            
            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(timezone.utc)
            
            try:
                # Check cache if enabled
                if self.cache and self.use_cache:
                    cache_key = f"earnings_scan:{','.join(sorted(symbols[:10]))}:{self.days_ahead_threshold}"
                    cached_alerts = await self.cache.get_cached_result(
                        self.name,
                        "batch",
                        cache_key
                    )
                    if cached_alerts is not None:
                        logger.info(f"Using cached results for earnings scan")
                        return cached_alerts
                
                # Build query filter for earnings data
                query_filter = QueryFilter(
                    symbols=symbols,
                    start_date=datetime.now(timezone.utc),
                    end_date=datetime.now(timezone.utc) + timedelta(days=self.days_ahead_threshold)
                )
                
                # Get earnings data from repository
                # This will use hot storage for upcoming earnings
                earnings_data = await self.repository.get_earnings_data(
                    symbols=symbols,
                    query_filter=query_filter
                )
                
                # Check if we got data
                if not earnings_data:
                    logger.warning("No earnings data returned from repository")
                    return []
                
                alerts = []
                for symbol, earnings_info in earnings_data.items():
                    if not earnings_info:
                        continue
                    
                    # Extract earnings events - handle dict structure
                    if isinstance(earnings_info, dict):
                        # Extract history list if present
                        events = earnings_info.get('history', [])
                        if not events and earnings_info.get('upcoming'):
                            events = [earnings_info['upcoming']]
                        if not events and earnings_info.get('recent'):
                            events = [earnings_info['recent']]
                    elif isinstance(earnings_info, list):
                        events = earnings_info
                    else:
                        # Single event
                        events = [earnings_info]
                    
                    if not events:
                        continue
                    
                    # Process earnings events for this symbol
                    symbol_alerts = await self._process_symbol_earnings(
                        symbol,
                        events
                    )
                    alerts.extend(symbol_alerts)
                
                # Deduplicate alerts
                alerts = self.deduplicate_alerts(alerts)
                
                # Cache results if enabled
                if self.cache and self.use_cache and alerts:
                    await self.cache.cache_result(
                        self.name,
                        "batch",
                        cache_key,
                        alerts,
                        ttl_seconds=3600  # 1 hour TTL for earnings
                    )
                
                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)
                
                logger.info(
                    f"✅ Earnings Scanner: Found {len(alerts)} symbols with upcoming earnings "
                    f"in {t.elapsed * 1000:.2f}ms"
                )
                
                # Record metrics
                if self.metrics:
                    self.metrics.record_scan_duration(
                        self.name,
                        t.elapsed * 1000,
                        len(symbols)
                    )
                
                return alerts
                
            except Exception as e:
                logger.error(f"❌ Error in Earnings Scanner: {e}", exc_info=True)
                
                # Record error metric
                if self.metrics:
                    self.metrics.record_scan_error(
                        self.name,
                        type(e).__name__,
                        str(e)
                    )
                
                return []
    
    async def run(self, universe: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Legacy method for backward compatibility.
        
        Args:
            universe: The list of symbols to scan.

        Returns:
            A dictionary mapping symbols to their catalyst signal data.
        """
        # Use the new scan method
        alerts = await self.scan(universe)
        
        # Convert to legacy format
        catalyst_signals = defaultdict(list)
        for alert in alerts:
            signal = {
                'score': alert.metadata.get('raw_score', alert.score * 5.0),  # Convert from 0-1 to 0-5 scale
                'reason': alert.metadata.get('reason', ''),
                'signal_type': 'earnings',
                'metadata': {
                    'earnings_date': alert.metadata.get('earnings_date'),
                    'days_until': alert.metadata.get('days_until'),
                    'source': alert.metadata.get('source')
                }
            }
            catalyst_signals[alert.symbol].append(signal)
        
        return dict(catalyst_signals)

    async def _process_symbol_earnings(
        self,
        symbol: str,
        earnings_events: List[Dict[str, Any]]
    ) -> List[ScanAlert]:
        """
        Process earnings events for a symbol and generate alerts.
        
        Args:
            symbol: Stock symbol
            earnings_events: List of earnings event data
            
        Returns:
            List of alerts for this symbol
        """
        alerts = []
        now_utc = datetime.now(timezone.utc)
        
        for event in earnings_events:
            # Extract earnings date
            earnings_date = event.get('earnings_date')
            if not earnings_date:
                continue
            
            # Ensure it's a datetime object
            if isinstance(earnings_date, str):
                try:
                    earnings_date = datetime.fromisoformat(earnings_date.replace('Z', '+00:00'))
                except Exception as e:
                    logger.debug(f"Failed to parse earnings date for {symbol}: {e}")
                    continue
            
            # Calculate days until earnings
            days_until = (earnings_date.date() - now_utc.date()).days
            
            # Skip if outside our threshold
            if days_until < 0 or days_until > self.days_ahead_threshold:
                continue
            
            # Calculate score based on proximity
            is_today = (days_until == 0)
            raw_score = self.catalyst_weights.get('today' if is_today else 'upcoming', 3.0)
            
            # Normalize score to 0-1 range
            normalized_score = min(raw_score / 5.0, 1.0)
            
            # Get additional metadata
            source = event.get('source', 'repository')
            eps_estimate = event.get('eps_estimate')
            revenue_estimate = event.get('revenue_estimate')
            
            # Create alert
            alert = self.create_alert(
                symbol=symbol,
                alert_type=AlertType.EARNINGS_ANNOUNCEMENT,
                score=normalized_score,
                metadata={
                    'earnings_date': earnings_date.isoformat(),
                    'days_until': days_until,
                    'source': source,
                    'reason': f"Earnings on {earnings_date.strftime('%Y-%m-%d')} ({days_until} days)",
                    'raw_score': raw_score,
                    'is_today': is_today,
                    'eps_estimate': eps_estimate,
                    'revenue_estimate': revenue_estimate
                }
            )
            alerts.append(alert)
            
            # Record metric
            if self.metrics:
                self.metrics.record_alert_generated(
                    self.name,
                    AlertType.EARNINGS_ANNOUNCEMENT,
                    symbol,
                    normalized_score
                )
        
        return alerts