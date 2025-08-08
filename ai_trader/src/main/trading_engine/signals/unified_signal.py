"""
Unified Signal Handler for Trading Engine

Aggregates, prioritizes, and manages trading signals from multiple sources
including strategies, scanners, and manual interventions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict

# Import signal type from strategies
from main.models.strategies.base_strategy import Signal
# from main.utils.core import get_utc_timestamp  # Not used

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Source of trading signals."""
    STRATEGY = "strategy"
    SCANNER = "scanner"
    MANUAL = "manual"
    REBALANCE = "rebalance"
    RISK_MANAGEMENT = "risk_management"


class SignalPriority(Enum):
    """Priority levels for signal processing."""
    CRITICAL = 1  # Risk management signals
    HIGH = 2      # Manual overrides
    NORMAL = 3    # Strategy signals
    LOW = 4       # Rebalancing signals


@dataclass
class UnifiedSignal:
    """Extended signal with metadata for unified handling."""
    signal: Signal
    source: SignalSource
    source_name: str  # e.g., "momentum_strategy", "volume_scanner"
    priority: SignalPriority
    timestamp: datetime
    expiry: Optional[datetime] = None
    correlation_group: Optional[str] = None  # For deduplication
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue sorting (lower priority value = higher priority)."""
        return self.priority.value < other.priority.value


class SignalAggregator:
    """Aggregates signals from multiple sources with deduplication."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signal_config = config.get('trading_engine', {}).get('signals', {})
        
        # Deduplication settings
        self.dedup_window = timedelta(
            seconds=self.signal_config.get('deduplication_window_seconds', 60)
        )
        self.correlation_threshold = self.signal_config.get('correlation_threshold', 0.8)
        
        # Signal history for deduplication
        self.recent_signals: List[UnifiedSignal] = []
        self.signal_groups: Dict[str, List[UnifiedSignal]] = defaultdict(list)
        
    def add_signals(self, signals: List[UnifiedSignal]) -> List[UnifiedSignal]:
        """Add signals with deduplication and grouping."""
        unique_signals = []
        current_time = datetime.now(timezone.utc)
        
        # Clean expired signals
        self._clean_expired_signals(current_time)
        
        for signal in signals:
            if self._is_duplicate(signal):
                logger.debug(f"Duplicate signal filtered: {signal.signal.symbol} {signal.signal.direction}")
                continue
                
            # Add to history and groups
            self.recent_signals.append(signal)
            if signal.correlation_group:
                self.signal_groups[signal.correlation_group].append(signal)
            
            unique_signals.append(signal)
            
        return unique_signals
    
    def _is_duplicate(self, signal: UnifiedSignal) -> bool:
        """Check if signal is a duplicate of recent signals."""
        for recent in self.recent_signals:
            # Same symbol and direction within time window
            if (recent.signal.symbol == signal.signal.symbol and
                recent.signal.direction == signal.signal.direction and
                abs((signal.timestamp - recent.timestamp).total_seconds()) < self.dedup_window.total_seconds()):
                
                # Check confidence correlation
                conf_diff = abs(recent.signal.confidence - signal.signal.confidence)
                if conf_diff < (1 - self.correlation_threshold):
                    return True
                    
        return False
    
    def _clean_expired_signals(self, current_time: datetime):
        """Remove old signals from history."""
        cutoff_time = current_time - self.dedup_window
        
        # Clean recent signals
        self.recent_signals = [
            s for s in self.recent_signals 
            if s.timestamp > cutoff_time
        ]
        
        # Clean signal groups
        for group in list(self.signal_groups.keys()):
            self.signal_groups[group] = [
                s for s in self.signal_groups[group]
                if s.timestamp > cutoff_time
            ]
            if not self.signal_groups[group]:
                del self.signal_groups[group]


class SignalRouter:
    """Routes signals to appropriate execution paths."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.routing_config = config.get('trading_engine', {}).get('routing', {})
        
        # Routing rules
        self.priority_routing = self.routing_config.get('priority_routing', {
            'CRITICAL': 'fast_execution',
            'HIGH': 'fast_execution',
            'NORMAL': 'standard',
            'LOW': 'batch'
        })
        
        # Size thresholds for execution algorithm selection
        self.size_thresholds = self.routing_config.get('size_thresholds', {
            'small': 0.01,   # < 1% of portfolio
            'medium': 0.05,  # 1-5% of portfolio
            'large': 0.10    # > 10% of portfolio
        })
        
    def route_signal(self, signal: UnifiedSignal, portfolio_value: float) -> Dict[str, Any]:
        """Determine execution routing for a signal."""
        routing = {
            'execution_path': self._get_execution_path(signal),
            'algorithm': self._select_algorithm(signal, portfolio_value),
            'urgency': self._calculate_urgency(signal),
            'constraints': self._get_constraints(signal)
        }
        
        return routing
    
    def _get_execution_path(self, signal: UnifiedSignal) -> str:
        """Select execution path based on priority."""
        return self.priority_routing.get(
            signal.priority.name,
            'standard'
        )
    
    def _select_algorithm(self, signal: UnifiedSignal, portfolio_value: float) -> str:
        """Select execution algorithm based on signal size."""
        if signal.signal.size is None:
            return 'market'
            
        position_value = signal.signal.size * portfolio_value
        
        # Large orders use VWAP/TWAP
        if signal.signal.size > self.size_thresholds['large']:
            return 'vwap' if signal.metadata.get('use_vwap', True) else 'twap'
        
        # Medium orders use TWAP
        elif signal.signal.size > self.size_thresholds['medium']:
            return 'twap'
        
        # Small orders use market
        else:
            return 'market'
    
    def _calculate_urgency(self, signal: UnifiedSignal) -> float:
        """Calculate urgency score (0-1) for execution timing."""
        if signal.priority == SignalPriority.CRITICAL:
            return 1.0
        
        if signal.expiry:
            time_remaining = (signal.expiry - datetime.now(timezone.utc)).total_seconds()
            total_window = (signal.expiry - signal.timestamp).total_seconds()
            
            if total_window > 0:
                urgency = 1 - (time_remaining / total_window)
                return max(0, min(1, urgency))
        
        # Default urgency based on priority
        priority_urgency = {
            SignalPriority.HIGH: 0.8,
            SignalPriority.NORMAL: 0.5,
            SignalPriority.LOW: 0.2
        }
        
        return priority_urgency.get(signal.priority, 0.5)
    
    def _get_constraints(self, signal: UnifiedSignal) -> Dict[str, Any]:
        """Extract execution constraints from signal metadata."""
        constraints = {
            'max_participation_rate': signal.metadata.get('max_participation', 0.1),
            'limit_price': signal.metadata.get('limit_price'),
            'stop_price': signal.metadata.get('stop_price'),
            'time_in_force': signal.metadata.get('time_in_force', 'DAY')
        }
        
        return constraints


class UnifiedSignalHandler:
    """Main handler for unified signal processing in the trading engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.aggregator = SignalAggregator(config)
        self.router = SignalRouter(config)
        
        # Signal queue (priority queue)
        self.signal_queue: List[UnifiedSignal] = []
        self.processing_lock = asyncio.Lock()
        
        # Callbacks
        self.execution_callback = None
        
        # Metrics
        self.signal_metrics = {
            'received': 0,
            'duplicates': 0,
            'processed': 0,
            'failed': 0
        }
        
        logger.info("UnifiedSignalHandler initialized")
    
    def set_execution_callback(self, callback):
        """Set callback for signal execution."""
        self.execution_callback = callback
    
    async def add_signals(self, 
                         signals: List[Signal],
                         source: SignalSource,
                         source_name: str,
                         priority: Optional[SignalPriority] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add signals to the processing queue."""
        async with self.processing_lock:
            # Convert to unified signals
            unified_signals = []
            current_time = datetime.now(timezone.utc)
            
            for signal in signals:
                unified_signal = UnifiedSignal(
                    signal=signal,
                    source=source,
                    source_name=source_name,
                    priority=priority or self._default_priority(source),
                    timestamp=current_time,
                    metadata=metadata or {}
                )
                unified_signals.append(unified_signal)
            
            self.signal_metrics['received'] += len(unified_signals)
            
            # Aggregate and deduplicate
            unique_signals = self.aggregator.add_signals(unified_signals)
            self.signal_metrics['duplicates'] += len(unified_signals) - len(unique_signals)
            
            # Add to priority queue
            for signal in unique_signals:
                heapq.heappush(self.signal_queue, signal)
            
            logger.info(f"Added {len(unique_signals)} unique signals from {source_name}")
            return len(unique_signals)
    
    async def process_signals(self, portfolio_value: float) -> List[Dict[str, Any]]:
        """Process queued signals and return execution instructions."""
        async with self.processing_lock:
            execution_instructions = []
            
            while self.signal_queue:
                signal = heapq.heappop(self.signal_queue)
                
                # Check expiry
                if signal.expiry and signal.expiry < datetime.now(timezone.utc):
                    logger.warning(f"Signal expired: {signal.signal.symbol} {signal.signal.direction}")
                    continue
                
                # Route signal
                routing = self.router.route_signal(signal, portfolio_value)
                
                # Create execution instruction
                instruction = {
                    'signal': signal.signal,
                    'source': signal.source.value,
                    'source_name': signal.source_name,
                    'routing': routing,
                    'timestamp': signal.timestamp,
                    'metadata': signal.metadata
                }
                
                execution_instructions.append(instruction)
                self.signal_metrics['processed'] += 1
                
                # Execute callback if set
                if self.execution_callback:
                    try:
                        await self.execution_callback(instruction)
                    except Exception as e:
                        logger.error(f"Execution callback failed: {e}")
                        self.signal_metrics['failed'] += 1
            
            return execution_instructions
    
    def _default_priority(self, source: SignalSource) -> SignalPriority:
        """Get default priority for signal source."""
        priority_map = {
            SignalSource.RISK_MANAGEMENT: SignalPriority.CRITICAL,
            SignalSource.MANUAL: SignalPriority.HIGH,
            SignalSource.STRATEGY: SignalPriority.NORMAL,
            SignalSource.SCANNER: SignalPriority.NORMAL,
            SignalSource.REBALANCE: SignalPriority.LOW
        }
        return priority_map.get(source, SignalPriority.NORMAL)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get signal processing metrics."""
        metrics = self.signal_metrics.copy()
        metrics['queue_size'] = len(self.signal_queue)
        metrics['active_groups'] = len(self.aggregator.signal_groups)
        return metrics
    
    def clear_queue(self):
        """Clear all pending signals."""
        self.signal_queue.clear()
        logger.info("Signal queue cleared")