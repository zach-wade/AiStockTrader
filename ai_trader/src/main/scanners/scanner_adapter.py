"""
Scanner system adapter with dependency injection support.

This module provides an adapter that integrates the scanner system with
the main trading engine, converting scan alerts to trading signals.

Key Features:
- Full dependency injection support via interfaces
- Event bus integration for real-time alert publishing
- Configurable parallel scanner engine
- Support for all new alert types
- Database-agnostic persistence

Usage:
    # Create with factory
    from main.scanners.scanner_adapter_factory import create_scanner_adapter
    
    adapter = create_scanner_adapter(
        config=config,
        database=db_interface,
        event_bus=event_bus,
        metrics_collector=metrics
    )
    
    # Register scanners
    adapter.register_scanner(scanner, priority=10)
    
    # Start scanning
    await adapter.start(universe=['AAPL', 'GOOGL', 'MSFT'])
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from main.scanners.base_scanner import BaseScanner
from main.scanners.layers.parallel_scanner_engine import (
    ParallelScannerEngine, ParallelEngineConfig, ScannerConfig
)
from main.events.types import ScanAlert, AlertType
from main.models.common import Signal, SignalType
from main.utils.core import create_event_tracker, create_task_safely
from main.utils.monitoring import MetricsCollector
from main.interfaces.database import IAsyncDatabase
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScannerAdapter, IScanner

logger = logging.getLogger(__name__)


@dataclass
class AlertToSignalConfig:
    """Configuration for converting alerts to signals."""
    min_confidence: float = 0.5
    alert_weights: Dict[AlertType, float] = field(default_factory=lambda: {
        # Volume alerts
        AlertType.VOLUME_SPIKE: 0.7,
        AlertType.UNUSUAL_ACTIVITY: 0.75,
        
        # Technical alerts
        AlertType.TECHNICAL_BREAKOUT: 0.8,
        AlertType.TECHNICAL_SIGNAL: 0.75,
        AlertType.MOMENTUM: 0.7,
        AlertType.MOMENTUM_SHIFT: 0.75,
        AlertType.TREND_CHANGE: 0.8,
        
        # News and sentiment
        AlertType.NEWS_CATALYST: 0.9,
        AlertType.SENTIMENT_SPIKE: 0.8,
        AlertType.SENTIMENT_SURGE: 0.85,
        AlertType.SOCIAL_BUZZ: 0.6,
        AlertType.SOCIAL_SENTIMENT: 0.65,
        AlertType.SOCIAL_VOLUME: 0.6,
        AlertType.SOCIAL_VIRAL: 0.7,
        
        # Corporate actions
        AlertType.EARNINGS_SURPRISE: 0.9,
        AlertType.EARNINGS_ANNOUNCEMENT: 0.85,
        AlertType.INSIDER_ACTIVITY: 0.8,
        AlertType.INSIDER_BUYING: 0.85,
        
        # Options flow
        AlertType.OPTIONS_FLOW: 0.75,
        AlertType.UNUSUAL_OPTIONS: 0.8,
        
        # Market structure
        AlertType.SECTOR_ROTATION: 0.65,
        AlertType.CORRELATION_BREAK: 0.7,
        AlertType.CORRELATION_ANOMALY: 0.7,
        AlertType.INTERMARKET_SIGNAL: 0.7,
        AlertType.DIVERGENCE: 0.75,
        AlertType.REGIME_CHANGE: 0.8,
        
        # Advanced analysis
        AlertType.COORDINATED_ACTIVITY: 0.85,
        
        # Price-based (if not covered above)
        AlertType.PRICE_BREAKOUT: 0.75,
        AlertType.PRICE_REVERSAL: 0.7,
        AlertType.SUPPORT_RESISTANCE: 0.65
    })
    signal_aggregation_window: float = 300.0  # 5 minutes
    min_alerts_for_signal: int = 2
    decay_rate: float = 0.95  # Confidence decay per minute
    
    
@dataclass
class ScannerAdapterConfig:
    """Configuration for scanner adapter."""
    engine_config: ParallelEngineConfig
    alert_to_signal_config: AlertToSignalConfig
    scan_interval: float = 60.0  # Seconds between scans
    universe_size_limit: int = 500
    enable_continuous_scanning: bool = True
    persist_alerts: bool = True
    
    
@dataclass
class AggregatedAlert:
    """Aggregated alerts for a symbol."""
    symbol: str
    alerts: List[ScanAlert]
    total_score: float
    first_seen: datetime
    last_updated: datetime
    
    @property
    def alert_count(self) -> int:
        """Get number of alerts."""
        return len(self.alerts)
        
    @property
    def alert_types(self) -> Set[AlertType]:
        """Get unique alert types."""
        return {alert.alert_type for alert in self.alerts}
        

class ScannerAdapter(IScannerAdapter):
    """
    Adapter that integrates scanners with the trading engine.
    
    Manages scanner execution, alert aggregation, and conversion
    of scan alerts into actionable trading signals.
    """
    
    def __init__(
        self,
        config: ScannerAdapterConfig,
        database: IAsyncDatabase,
        engine: Optional[ParallelScannerEngine] = None,
        event_bus: Optional[IEventBus] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize scanner adapter.
        
        Args:
            config: Adapter configuration
            database: Database interface for persistence
            engine: Optional scanner engine (created if not provided)
            event_bus: Optional event bus for integration
            metrics_collector: Optional metrics collector
        """
        self.config = config
        self.db = database
        self.event_bus = event_bus
        self.metrics = metrics_collector
        self.event_tracker = create_event_tracker("scanner_adapter")
        
        # Use provided engine or create new one
        if engine is not None:
            self.engine = engine
        else:
            self.engine = ParallelScannerEngine(
                config.engine_config,
                metrics_collector
            )
        
        # Alert aggregation
        self._alert_buffer: Dict[str, AggregatedAlert] = {}
        self._signal_callbacks: List[Callable[[Signal], None]] = []
        
        # Scanning state
        self._scanning_task: Optional[asyncio.Task] = None
        self._running = False
        self._current_universe: List[str] = []
        
    def register_scanner(self, scanner: BaseScanner, **kwargs) -> None:
        """
        Register a scanner with the system.
        
        Args:
            scanner: Scanner instance
            **kwargs: Additional scanner configuration
        """
        scanner_config = ScannerConfig(
            scanner=scanner,
            priority=kwargs.get('priority', 5),
            enabled=kwargs.get('enabled', True),
            timeout=kwargs.get('timeout', 30.0),
            retry_attempts=kwargs.get('retry_attempts', 3),
            batch_size=kwargs.get('batch_size', 50)
        )
        
        self.engine.register_scanner(scanner_config)
        logger.info(f"Registered scanner: {scanner.name}")
        
    def register_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """Register callback for generated signals."""
        self._signal_callbacks.append(callback)
        
    async def start(self, universe: List[str]) -> None:
        """
        Start scanner adapter.
        
        Args:
            universe: Initial universe of symbols to scan
        """
        logger.info(f"Starting scanner adapter with {len(universe)} symbols")
        
        self._running = True
        self._current_universe = universe[:self.config.universe_size_limit]
        
        if self.config.enable_continuous_scanning:
            self._scanning_task = create_task_safely(self._continuous_scanning())
            
    async def stop(self) -> None:
        """Stop scanner adapter."""
        logger.info("Stopping scanner adapter")
        
        self._running = False
        
        if self._scanning_task:
            self._scanning_task.cancel()
            await asyncio.gather(self._scanning_task, return_exceptions=True)
            
    async def scan_once(self, symbols: Optional[List[str]] = None) -> List[Signal]:
        """
        Perform a single scan and return signals.
        
        Args:
            symbols: Optional list of symbols to scan (uses universe if not provided)
            
        Returns:
            List of generated signals
        """
        scan_symbols = symbols or self._current_universe
        
        if not scan_symbols:
            logger.warning("No symbols to scan")
            return []
            
        # Run scanners
        alerts = await self.engine.scan_symbols(scan_symbols)
        
        # Process alerts
        await self._process_alerts(alerts)
        
        # Generate signals
        signals = self._generate_signals()
        
        # Publish alerts to event bus if available
        if self.event_bus and alerts:
            await self._publish_alerts_to_event_bus(alerts)
        
        # Persist alerts if configured
        if self.config.persist_alerts:
            await self._persist_alerts(alerts)
            
        return signals
        
    def update_universe(self, universe: List[str]) -> None:
        """Update the scanning universe."""
        self._current_universe = universe[:self.config.universe_size_limit]
        logger.info(f"Updated universe to {len(self._current_universe)} symbols")
        
    async def _continuous_scanning(self) -> None:
        """Continuous scanning loop."""
        while self._running:
            try:
                # Perform scan
                signals = await self.scan_once()
                
                # Trigger callbacks for signals
                for signal in signals:
                    await self._trigger_signal_callbacks(signal)
                    
                # Wait for next scan
                await asyncio.sleep(self.config.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous scanning: {e}")
                await asyncio.sleep(self.config.scan_interval)
                
    async def _process_alerts(self, alerts: List[ScanAlert]) -> None:
        """Process and aggregate alerts."""
        now = datetime.utcnow()
        
        for alert in alerts:
            symbol = alert.symbol
            
            # Update or create aggregated alert
            if symbol in self._alert_buffer:
                agg_alert = self._alert_buffer[symbol]
                agg_alert.alerts.append(alert)
                agg_alert.total_score += self._calculate_alert_score(alert)
                agg_alert.last_updated = now
            else:
                score = self._calculate_alert_score(alert)
                self._alert_buffer[symbol] = AggregatedAlert(
                    symbol=symbol,
                    alerts=[alert],
                    total_score=score,
                    first_seen=now,
                    last_updated=now
                )
                
        # Clean old alerts
        self._clean_alert_buffer(now)
        
    def _calculate_alert_score(self, alert: ScanAlert) -> float:
        """Calculate score for an alert."""
        base_weight = self.config.alert_to_signal_config.alert_weights.get(
            alert.alert_type,
            0.5
        )
        
        # Adjust by confidence
        score = base_weight * alert.confidence
        
        # Adjust by severity
        severity_multiplier = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.3,
            'critical': 1.5
        }
        
        score *= severity_multiplier.get(alert.severity, 1.0)
        
        return score
        
    def _generate_signals(self) -> List[Signal]:
        """Generate trading signals from aggregated alerts."""
        signals = []
        now = datetime.utcnow()
        config = self.config.alert_to_signal_config
        
        for symbol, agg_alert in self._alert_buffer.items():
            # Check minimum alert count
            if agg_alert.alert_count < config.min_alerts_for_signal:
                continue
                
            # Calculate time-decayed confidence
            minutes_elapsed = (now - agg_alert.first_seen).total_seconds() / 60
            decay_factor = config.decay_rate ** minutes_elapsed
            
            # Calculate final confidence
            avg_score = agg_alert.total_score / agg_alert.alert_count
            confidence = avg_score * decay_factor
            
            # Check minimum confidence
            if confidence < config.min_confidence:
                continue
                
            # Determine signal type based on alerts
            signal_type = self._determine_signal_type(agg_alert)
            
            # Create signal
            signal = Signal(
                signal_id=f"scan_{symbol}_{now.timestamp()}",
                timestamp=now,
                symbol=symbol,
                signal_type=signal_type,
                strength=confidence,
                source=f"scanner_adapter",
                metadata={
                    'alert_count': agg_alert.alert_count,
                    'alert_types': [at.value for at in agg_alert.alert_types],
                    'first_alert': agg_alert.first_seen.isoformat(),
                    'total_score': agg_alert.total_score
                }
            )
            
            signals.append(signal)
            
            # Track signal generation
            self._track_signal_generation(signal, agg_alert)
            
        return signals
        
    def _determine_signal_type(self, agg_alert: AggregatedAlert) -> SignalType:
        """Determine signal type from aggregated alerts."""
        # Count alert types
        type_counts = defaultdict(int)
        for alert in agg_alert.alerts:
            type_counts[alert.alert_type] += 1
            
        # Find dominant alert type
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # Map to signal type
        type_mapping = {
            # Volume-based entries
            AlertType.VOLUME_SPIKE: SignalType.ENTRY,
            AlertType.UNUSUAL_ACTIVITY: SignalType.ENTRY,
            
            # Technical entries
            AlertType.TECHNICAL_BREAKOUT: SignalType.ENTRY,
            AlertType.TECHNICAL_SIGNAL: SignalType.ENTRY,
            AlertType.MOMENTUM: SignalType.ENTRY,
            AlertType.MOMENTUM_SHIFT: SignalType.ENTRY,
            AlertType.TREND_CHANGE: SignalType.REBALANCE,
            
            # News and sentiment
            AlertType.NEWS_CATALYST: SignalType.ENTRY,
            AlertType.SENTIMENT_SPIKE: SignalType.ENTRY,
            AlertType.SENTIMENT_SURGE: SignalType.ENTRY,
            
            # Social signals
            AlertType.SOCIAL_BUZZ: SignalType.ENTRY,
            AlertType.SOCIAL_SENTIMENT: SignalType.ENTRY,
            AlertType.SOCIAL_VOLUME: SignalType.ENTRY,
            AlertType.SOCIAL_VIRAL: SignalType.ENTRY,
            
            # Corporate actions
            AlertType.EARNINGS_SURPRISE: SignalType.ENTRY,
            AlertType.EARNINGS_ANNOUNCEMENT: SignalType.ENTRY,
            AlertType.INSIDER_ACTIVITY: SignalType.ENTRY,
            AlertType.INSIDER_BUYING: SignalType.ENTRY,
            
            # Options flow
            AlertType.OPTIONS_FLOW: SignalType.ENTRY,
            AlertType.UNUSUAL_OPTIONS: SignalType.ENTRY,
            
            # Market structure - typically rebalance or exit
            AlertType.SECTOR_ROTATION: SignalType.REBALANCE,
            AlertType.CORRELATION_BREAK: SignalType.EXIT,
            AlertType.CORRELATION_ANOMALY: SignalType.REBALANCE,
            AlertType.INTERMARKET_SIGNAL: SignalType.REBALANCE,
            AlertType.DIVERGENCE: SignalType.REBALANCE,
            AlertType.REGIME_CHANGE: SignalType.REBALANCE,
            
            # Risk signals
            AlertType.COORDINATED_ACTIVITY: SignalType.EXIT,
            
            # Price-based
            AlertType.PRICE_BREAKOUT: SignalType.ENTRY,
            AlertType.PRICE_REVERSAL: SignalType.EXIT,
            AlertType.SUPPORT_RESISTANCE: SignalType.ENTRY,
            
            # Legacy mappings (backward compatibility)
            AlertType.EARNINGS_BEAT: SignalType.ENTRY,
            AlertType.SOCIAL_MOMENTUM: SignalType.ENTRY,
            AlertType.OPTIONS_ACTIVITY: SignalType.ENTRY
        }
        
        return type_mapping.get(dominant_type, SignalType.ENTRY)
        
    def _clean_alert_buffer(self, now: datetime) -> None:
        """Remove old alerts from buffer."""
        window_seconds = self.config.alert_to_signal_config.signal_aggregation_window
        cutoff = now - timedelta(seconds=window_seconds)
        
        # Find symbols to remove
        to_remove = [
            symbol for symbol, agg_alert in self._alert_buffer.items()
            if agg_alert.last_updated < cutoff
        ]
        
        # Remove old alerts
        for symbol in to_remove:
            del self._alert_buffer[symbol]
            
    async def _trigger_signal_callbacks(self, signal: Signal) -> None:
        """Trigger callbacks for a signal."""
        for callback in self._signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")
                
    async def _persist_alerts(self, alerts: List[ScanAlert]) -> None:
        """Persist alerts to database."""
        if not alerts:
            return
            
        try:
            # Convert alerts to dictionaries for database storage
            alert_records = []
            for alert in alerts:
                alert_dict = {
                    'symbol': alert.symbol,
                    'alert_type': alert.alert_type.value if hasattr(alert.alert_type, 'value') else str(alert.alert_type),
                    'confidence': alert.confidence,
                    'severity': alert.severity,
                    'source_scanner': getattr(alert, 'source_scanner', 'unknown'),
                    'metadata': alert.metadata or {},
                    'timestamp': alert.timestamp
                }
                alert_records.append(alert_dict)
            
            # Use database interface to save alerts
            # Note: The specific method depends on the database interface implementation
            if hasattr(self.db, 'execute_many'):
                await self.db.execute_many(
                    "INSERT INTO scan_alerts (symbol, alert_type, confidence, severity, source_scanner, metadata, timestamp) "
                    "VALUES (:symbol, :alert_type, :confidence, :severity, :source_scanner, :metadata, :timestamp)",
                    alert_records
                )
            else:
                # Fallback to individual inserts if batch not supported
                for record in alert_records:
                    await self.db.execute(
                        "INSERT INTO scan_alerts (symbol, alert_type, confidence, severity, source_scanner, metadata, timestamp) "
                        "VALUES (:symbol, :alert_type, :confidence, :severity, :source_scanner, :metadata, :timestamp)",
                        record
                    )
                    
        except Exception as e:
            logger.error(f"Error persisting alerts: {e}")
            # Optionally publish to event bus if available
            if self.event_bus:
                # Could publish a persistence error event here
                pass
    
    async def _publish_alerts_to_event_bus(self, alerts: List[ScanAlert]) -> None:
        """Publish scanner alerts to the event bus."""
        if not self.event_bus or not alerts:
            return
        
        try:
            # Import here to avoid circular dependency
            from main.events.types import ScannerAlertEvent, EventType
            
            # Group alerts by scanner for efficiency
            alerts_by_scanner = defaultdict(list)
            for alert in alerts:
                scanner_name = getattr(alert, 'source_scanner', 'unknown')
                alerts_by_scanner[scanner_name].append(alert)
            
            # Publish grouped alerts
            for scanner_name, scanner_alerts in alerts_by_scanner.items():
                event = ScannerAlertEvent(
                    alerts=scanner_alerts,
                    scanner_name=scanner_name,
                    event_type=EventType.SCANNER_ALERT,
                    timestamp=datetime.utcnow()
                )
                await self.event_bus.publish(event)
                
        except Exception as e:
            logger.error(f"Error publishing alerts to event bus: {e}")
            
    def _track_signal_generation(self, signal: Signal, agg_alert: AggregatedAlert) -> None:
        """Track signal generation metrics."""
        self.event_tracker.track("signal_generated", {
            "symbol": signal.symbol,
            "signal_type": signal.signal_type.value,
            "confidence": signal.strength,
            "alert_count": agg_alert.alert_count,
            "alert_types": list(agg_alert.alert_types)
        })
        
        if self.metrics:
            self.metrics.increment(
                "scanner_adapter.signals_generated",
                tags={
                    "signal_type": signal.signal_type.value,
                    "symbol": signal.symbol
                }
            )
            
    def get_alert_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the alert buffer."""
        return {
            "buffer_size": len(self._alert_buffer),
            "total_alerts": sum(a.alert_count for a in self._alert_buffer.values()),
            "symbols_with_alerts": list(self._alert_buffer.keys()),
            "avg_alerts_per_symbol": (
                sum(a.alert_count for a in self._alert_buffer.values()) / 
                len(self._alert_buffer) if self._alert_buffer else 0
            )
        }