"""
Broker Reconciler

Specialized component for broker position synchronization and discrepancy detection.
Uses existing broker interface for communication.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from main.models.common import Position
from main.trading_engine.brokers.broker_interface import BrokerInterface
from main.trading_engine.core.position_tracker import PositionTracker

logger = logging.getLogger(__name__)


class DiscrepancySeverity(Enum):
    """Severity levels for position discrepancies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionDiscrepancy:
    """Position discrepancy between local and broker states."""
    symbol: str
    local_position: Optional[Position]
    broker_position: Optional[Position]
    quantity_difference: Decimal
    value_difference: Decimal
    severity: DiscrepancySeverity
    auto_correctable: bool
    detected_at: datetime
    
    @property
    def discrepancy_type(self) -> str:
        """Get human-readable discrepancy type."""
        if not self.local_position and self.broker_position:
            return "missing_local"
        elif self.local_position and not self.broker_position:
            return "orphaned_local"
        else:
            return "quantity_mismatch"


@dataclass
class ReconciliationResult:
    """Result of position reconciliation."""
    discrepancies: List[PositionDiscrepancy]
    total_positions_checked: int
    discrepancies_found: int
    auto_corrected: int
    manual_review_required: int
    reconciliation_time: datetime
    execution_time_ms: int
    
    @property
    def success_rate(self) -> float:
        """Calculate reconciliation success rate."""
        if self.total_positions_checked == 0:
            return 100.0
        return ((self.total_positions_checked - self.discrepancies_found) / self.total_positions_checked) * 100


class BrokerReconciler:
    """
    Broker position reconciliation and synchronization.
    
    Responsibilities:
    - Compare local vs broker positions
    - Detect and classify discrepancies
    - Auto-correct minor discrepancies
    - Generate reconciliation reports
    """
    
    def __init__(self, 
                 broker_interface: BrokerInterface,
                 position_tracker: PositionTracker,
                 auto_correct_threshold: Decimal = Decimal('0.01'),
                 reconciliation_interval: int = 300):
        """
        Initialize broker reconciler.
        
        Args:
            broker_interface: Existing broker interface
            position_tracker: Position tracker for local state
            auto_correct_threshold: Threshold for auto-correction
            reconciliation_interval: Interval between reconciliations (seconds)
        """
        self.broker_interface = broker_interface
        self.position_tracker = position_tracker
        self.auto_correct_threshold = auto_correct_threshold
        self.reconciliation_interval = reconciliation_interval
        
        # Reconciliation state
        self.last_reconciliation: Optional[datetime] = None
        self.reconciliation_history: List[ReconciliationResult] = []
        
        # Background task
        self.reconciliation_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Tolerances for discrepancy detection
        self.quantity_tolerance = Decimal('0.001')  # 0.001 shares
        self.value_tolerance = Decimal('0.01')      # $0.01
        
        logger.info("✅ BrokerReconciler initialized")
    
    async def reconcile_positions(self, auto_correct: bool = False) -> ReconciliationResult:
        """
        Reconcile positions with broker.
        
        Args:
            auto_correct: Whether to auto-correct discrepancies
            
        Returns:
            ReconciliationResult with detailed results
        """
        start_time = datetime.now()
        
        try:
            # Get positions from both sources
            # Handle both sync and async position trackers
            if asyncio.iscoroutinefunction(self.position_tracker.get_all_positions):
                local_positions = await self.position_tracker.get_all_positions()
            else:
                local_positions = self.position_tracker.get_all_positions()
            
            broker_positions = await self._get_broker_positions()
            
            # Find all unique symbols
            all_symbols = set(local_positions.keys()) | set(broker_positions.keys())
            
            # Analyze discrepancies
            discrepancies = []
            auto_corrected = 0
            
            for symbol in all_symbols:
                local_pos = local_positions.get(symbol)
                broker_pos = broker_positions.get(symbol)
                
                discrepancy = self._analyze_discrepancy(local_pos, broker_pos)
                
                if discrepancy:
                    discrepancies.append(discrepancy)
                    
                    # Auto-correct if enabled and safe
                    if auto_correct and discrepancy.auto_correctable:
                        if await self._auto_correct_discrepancy(discrepancy):
                            auto_corrected += 1
            
            # Calculate metrics
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            result = ReconciliationResult(
                discrepancies=discrepancies,
                total_positions_checked=len(all_symbols),
                discrepancies_found=len(discrepancies),
                auto_corrected=auto_corrected,
                manual_review_required=len(discrepancies) - auto_corrected,
                reconciliation_time=start_time,
                execution_time_ms=execution_time
            )
            
            # Update state
            self.last_reconciliation = start_time
            self.reconciliation_history.append(result)
            
            # Keep only last 100 reconciliations
            if len(self.reconciliation_history) > 100:
                self.reconciliation_history = self.reconciliation_history[-50:]
            
            # Log results
            if discrepancies:
                logger.warning(f"Reconciliation found {len(discrepancies)} discrepancies, {auto_corrected} auto-corrected")
                for disc in discrepancies:
                    logger.warning(f"  {disc.symbol}: {disc.discrepancy_type} ({disc.severity.value})")
            else:
                logger.info("Position reconciliation completed successfully - no discrepancies found")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during position reconciliation: {e}")
            return ReconciliationResult(
                discrepancies=[],
                total_positions_checked=0,
                discrepancies_found=0,
                auto_corrected=0,
                manual_review_required=0,
                reconciliation_time=start_time,
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    async def _get_broker_positions(self) -> Dict[str, Position]:
        """Get positions from broker."""
        try:
            broker_positions = await self.broker_interface.get_positions()
            return {pos.symbol: pos for pos in broker_positions} if broker_positions else {}
        except Exception as e:
            logger.error(f"Error getting broker positions: {e}")
            return {}
    
    def _analyze_discrepancy(self, 
                           local_pos: Optional[Position],
                           broker_pos: Optional[Position]) -> Optional[PositionDiscrepancy]:
        """Analyze discrepancy between local and broker positions."""
        if not local_pos and not broker_pos:
            return None
        
        detected_at = datetime.now()
        
        if not local_pos and broker_pos:
            # Missing local position
            return PositionDiscrepancy(
                symbol=broker_pos.symbol,
                local_position=None,
                broker_position=broker_pos,
                quantity_difference=Decimal(str(broker_pos.quantity)),
                value_difference=Decimal(str(broker_pos.market_value)),
                severity=DiscrepancySeverity.MEDIUM,
                auto_correctable=True,
                detected_at=detected_at
            )
        
        if local_pos and not broker_pos:
            # Orphaned local position
            return PositionDiscrepancy(
                symbol=local_pos.symbol,
                local_position=local_pos,
                broker_position=None,
                quantity_difference=Decimal(str(-local_pos.quantity)),
                value_difference=Decimal(str(-local_pos.market_value)),
                severity=DiscrepancySeverity.HIGH,
                auto_correctable=True,
                detected_at=detected_at
            )
        
        # Both positions exist - check for differences
        if local_pos and broker_pos:
            qty_diff = abs(Decimal(str(local_pos.quantity)) - Decimal(str(broker_pos.quantity)))
            value_diff = abs(Decimal(str(local_pos.market_value)) - Decimal(str(broker_pos.market_value)))
            
            if qty_diff > self.quantity_tolerance or value_diff > self.value_tolerance:
                # Determine severity and auto-correction eligibility
                severity, auto_correctable = self._assess_discrepancy_severity(qty_diff, value_diff)
                
                return PositionDiscrepancy(
                    symbol=local_pos.symbol,
                    local_position=local_pos,
                    broker_position=broker_pos,
                    quantity_difference=Decimal(str(broker_pos.quantity)) - Decimal(str(local_pos.quantity)),
                    value_difference=Decimal(str(broker_pos.market_value)) - Decimal(str(local_pos.market_value)),
                    severity=severity,
                    auto_correctable=auto_correctable,
                    detected_at=detected_at
                )
        
        return None
    
    def _assess_discrepancy_severity(self, 
                                   qty_diff: Decimal, 
                                   value_diff: Decimal) -> Tuple[DiscrepancySeverity, bool]:
        """Assess severity and auto-correction eligibility."""
        if qty_diff > Decimal('10') or value_diff > Decimal('1000'):
            return DiscrepancySeverity.CRITICAL, False
        elif qty_diff > Decimal('1') or value_diff > Decimal('100'):
            return DiscrepancySeverity.HIGH, False
        elif qty_diff > Decimal('0.1') or value_diff > Decimal('10'):
            return DiscrepancySeverity.MEDIUM, True
        else:
            return DiscrepancySeverity.LOW, True
    
    async def _auto_correct_discrepancy(self, discrepancy: PositionDiscrepancy) -> bool:
        """Auto-correct a discrepancy if safe to do so."""
        try:
            if discrepancy.discrepancy_type == "missing_local":
                # Add missing local position
                await self.position_tracker.update_position(discrepancy.broker_position)
                logger.info(f"Auto-corrected missing local position for {discrepancy.symbol}")
                return True
            
            elif discrepancy.discrepancy_type == "orphaned_local":
                # Remove orphaned local position
                await self.position_tracker.remove_position(discrepancy.symbol)
                logger.info(f"Auto-corrected orphaned local position for {discrepancy.symbol}")
                return True
            
            elif discrepancy.discrepancy_type == "quantity_mismatch":
                # Update local position to match broker
                if discrepancy.severity in [DiscrepancySeverity.LOW, DiscrepancySeverity.MEDIUM]:
                    await self.position_tracker.update_position(discrepancy.broker_position)
                    logger.info(f"Auto-corrected quantity mismatch for {discrepancy.symbol}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error auto-correcting discrepancy for {discrepancy.symbol}: {e}")
            return False
    
    async def start_continuous_reconciliation(self):
        """Start continuous background reconciliation."""
        if self.is_running:
            logger.warning("Continuous reconciliation already running")
            return
        
        self.is_running = True
        self.reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        logger.info("✅ Continuous reconciliation started")
    
    async def stop_continuous_reconciliation(self):
        """Stop continuous background reconciliation."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.reconciliation_task:
            self.reconciliation_task.cancel()
            try:
                await self.reconciliation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Continuous reconciliation stopped")
    
    async def _reconciliation_loop(self):
        """Background reconciliation loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.reconciliation_interval)
                
                if self.is_running:
                    await self.reconcile_positions(auto_correct=True)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def get_reconciliation_status(self) -> Dict[str, any]:
        """Get current reconciliation status."""
        return {
            'is_running': self.is_running,
            'last_reconciliation': self.last_reconciliation.isoformat() if self.last_reconciliation else None,
            'reconciliation_interval': self.reconciliation_interval,
            'total_reconciliations': len(self.reconciliation_history),
            'recent_success_rate': self._calculate_recent_success_rate(),
            'auto_correct_threshold': float(self.auto_correct_threshold)
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate for recent reconciliations."""
        if not self.reconciliation_history:
            return 0.0
        
        # Get last 10 reconciliations
        recent = self.reconciliation_history[-10:]
        total_rate = sum(result.success_rate for result in recent)
        return total_rate / len(recent)
    
    def get_discrepancy_summary(self) -> Dict[str, any]:
        """Get summary of recent discrepancies."""
        if not self.reconciliation_history:
            return {'total_discrepancies': 0, 'by_severity': {}, 'by_type': {}}
        
        recent_discrepancies = []
        for result in self.reconciliation_history[-5:]:  # Last 5 reconciliations
            recent_discrepancies.extend(result.discrepancies)
        
        # Count by severity
        by_severity = {}
        for severity in DiscrepancySeverity:
            by_severity[severity.value] = sum(1 for d in recent_discrepancies if d.severity == severity)
        
        # Count by type
        by_type = {}
        for discrepancy in recent_discrepancies:
            disc_type = discrepancy.discrepancy_type
            by_type[disc_type] = by_type.get(disc_type, 0) + 1
        
        return {
            'total_discrepancies': len(recent_discrepancies),
            'by_severity': by_severity,
            'by_type': by_type
        }
    
    async def cleanup(self):
        """Clean up reconciler resources."""
        await self.stop_continuous_reconciliation()
        logger.info("✅ BrokerReconciler cleaned up")