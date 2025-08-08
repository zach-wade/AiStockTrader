"""
Fill Processor

Handles order fill processing and position updates.
Integrates with existing order management and position tracking systems.
"""

import logging
from decimal import Decimal
from typing import Dict, Optional, Tuple
from datetime import datetime

from main.models.common import Position, Order, OrderSide
from main.trading_engine.core.position_tracker import PositionTracker
from main.trading_engine.core.position_events import (
    PositionEvent, PositionEventType, FillProcessedEvent, 
    PositionOpenedEvent, PositionClosedEvent, PositionIncreasedEvent,
    PositionDecreasedEvent, PositionReversedEvent
)

logger = logging.getLogger(__name__)


class FillResult:
    """Result of fill processing."""
    
    def __init__(self, 
                 old_position: Optional[Position],
                 new_position: Optional[Position],
                 realized_pnl: Decimal,
                 event: PositionEvent):
        self.old_position = old_position
        self.new_position = new_position
        self.realized_pnl = realized_pnl
        self.event = event
        
    @property
    def position_change_type(self) -> PositionEventType:
        """Determine the type of position change."""
        if not self.old_position and self.new_position:
            return PositionEventType.POSITION_OPENED
        elif self.old_position and not self.new_position:
            return PositionEventType.POSITION_CLOSED
        elif self.old_position and self.new_position:
            old_qty = Decimal(str(self.old_position.quantity))
            new_qty = Decimal(str(self.new_position.quantity))
            
            if abs(new_qty) > abs(old_qty):
                return PositionEventType.POSITION_INCREASED
            elif abs(new_qty) < abs(old_qty):
                return PositionEventType.POSITION_DECREASED
            elif (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
                return PositionEventType.POSITION_REVERSED
        
        return PositionEventType.FILL_PROCESSED


class FillProcessor:
    """
    Handles order fill processing and position updates.
    
    Responsibilities:
    - Process order fills into position updates
    - Calculate realized P&L
    - Generate position events
    - Update position state through tracker
    """
    
    def __init__(self, position_tracker: PositionTracker):
        """
        Initialize fill processor.
        
        Args:
            position_tracker: Position tracker for state management
        """
        self.position_tracker = position_tracker
        
        # P&L tracking
        self.realized_pnl_today = Decimal('0')
        self.realized_pnl_total = Decimal('0')
        
        logger.info("✅ FillProcessor initialized")
    
    async def process_fill(self, 
                          order: Order,
                          fill_price: Decimal,
                          fill_quantity: Decimal,
                          commission: Decimal = Decimal('0'),
                          fees: Decimal = Decimal('0')) -> FillResult:
        """
        Process an order fill and update positions.
        
        Args:
            order: The order that was filled
            fill_price: Price at which the fill occurred
            fill_quantity: Quantity that was filled
            commission: Commission paid
            fees: Fees paid
            
        Returns:
            FillResult with position changes and realized P&L
        """
        try:
            symbol = order.symbol
            
            # Get current position
            current_position = self.position_tracker.get_position(symbol)
            
            # Calculate new position
            new_position = await self._calculate_new_position(
                current_position, order, fill_price, fill_quantity, commission, fees
            )
            
            # Calculate realized P&L
            realized_pnl = self._calculate_realized_pnl(
                current_position, new_position, order, fill_price, fill_quantity
            )
            
            # Update position tracker
            if new_position and new_position.quantity != 0:
                await self.position_tracker.update_position(new_position)
            elif current_position:
                # Position closed
                await self.position_tracker.remove_position(symbol)
                new_position = None
            
            # Create appropriate event
            event = self._create_position_event(
                current_position, new_position, order, fill_price, fill_quantity, realized_pnl
            )
            
            # Update P&L tracking
            self.realized_pnl_today += realized_pnl
            self.realized_pnl_total += realized_pnl
            
            logger.info(f"Processed fill for {symbol}: {fill_quantity} @ {fill_price}, P&L: {realized_pnl}")
            
            return FillResult(
                old_position=current_position,
                new_position=new_position,
                realized_pnl=realized_pnl,
                event=event
            )
            
        except Exception as e:
            logger.error(f"Error processing fill for {order.symbol}: {e}")
            # Return error result
            return FillResult(
                old_position=current_position,
                new_position=current_position,
                realized_pnl=Decimal('0'),
                event=FillProcessedEvent(
                    symbol=order.symbol,
                    old_position=current_position,
                    new_position=current_position,
                    fill_price=fill_price,
                    fill_quantity=fill_quantity,
                    metadata={'error': str(e)}
                )
            )
    
    async def _calculate_new_position(self,
                                     current_position: Optional[Position],
                                     order: Order,
                                     fill_price: Decimal,
                                     fill_quantity: Decimal,
                                     commission: Decimal,
                                     fees: Decimal) -> Optional[Position]:
        """Calculate new position after fill."""
        try:
            # Convert order side to signed quantity
            signed_fill_quantity = fill_quantity if order.side == OrderSide.BUY else -fill_quantity
            
            if not current_position:
                # Opening new position
                return Position(
                    symbol=order.symbol,
                    quantity=float(signed_fill_quantity),
                    avg_entry_price=float(fill_price),
                    current_price=float(fill_price),
                    market_value=float(signed_fill_quantity * fill_price),
                    cost_basis=float(abs(signed_fill_quantity) * fill_price + commission + fees),
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    realized_pnl=0.0,
                    side='long' if signed_fill_quantity > 0 else 'short',
                    timestamp=datetime.now()
                )
            else:
                # Modifying existing position
                old_quantity = Decimal(str(current_position.quantity))
                new_quantity = old_quantity + signed_fill_quantity
                
                if new_quantity == 0:
                    # Position closed
                    return None
                
                # Calculate new average entry price
                old_cost_basis = Decimal(str(current_position.cost_basis))
                new_cost_basis = old_cost_basis + (abs(signed_fill_quantity) * fill_price + commission + fees)
                new_avg_entry_price = new_cost_basis / abs(new_quantity)
                
                # Calculate unrealized P&L
                current_price = Decimal(str(current_position.current_price))
                unrealized_pnl = (current_price - new_avg_entry_price) * new_quantity
                unrealized_pnl_pct = ((current_price / new_avg_entry_price) - 1) * 100 if new_avg_entry_price > 0 else 0
                
                return Position(
                    symbol=order.symbol,
                    quantity=float(new_quantity),
                    avg_entry_price=float(new_avg_entry_price),
                    current_price=float(current_price),
                    market_value=float(new_quantity * current_price),
                    cost_basis=float(new_cost_basis),
                    unrealized_pnl=float(unrealized_pnl),
                    unrealized_pnl_pct=float(unrealized_pnl_pct),
                    realized_pnl=current_position.realized_pnl,
                    side='long' if new_quantity > 0 else 'short',
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error calculating new position: {e}")
            return current_position
    
    def _calculate_realized_pnl(self,
                               current_position: Optional[Position],
                               new_position: Optional[Position],
                               order: Order,
                               fill_price: Decimal,
                               fill_quantity: Decimal) -> Decimal:
        """Calculate realized P&L from position change."""
        try:
            if not current_position:
                # Opening new position - no realized P&L
                return Decimal('0')
            
            old_quantity = Decimal(str(current_position.quantity))
            new_quantity = Decimal(str(new_position.quantity)) if new_position else Decimal('0')
            
            # Check if position is being reduced
            if abs(new_quantity) >= abs(old_quantity):
                # Position increased or same size - no realized P&L
                return Decimal('0')
            
            # Calculate realized P&L for the portion closed
            closed_quantity = abs(old_quantity) - abs(new_quantity)
            avg_entry_price = Decimal(str(current_position.avg_entry_price))
            price_diff = fill_price - avg_entry_price
            
            # Sign based on position side
            if old_quantity > 0:  # Long position
                realized_pnl = closed_quantity * price_diff
            else:  # Short position
                realized_pnl = closed_quantity * -price_diff
            
            return realized_pnl
            
        except Exception as e:
            logger.error(f"Error calculating realized P&L: {e}")
            return Decimal('0')
    
    def _create_position_event(self,
                              old_position: Optional[Position],
                              new_position: Optional[Position],
                              order: Order,
                              fill_price: Decimal,
                              fill_quantity: Decimal,
                              realized_pnl: Decimal) -> PositionEvent:
        """Create appropriate position event based on position changes."""
        try:
            # Determine event type
            if not old_position and new_position:
                event_type = PositionEventType.POSITION_OPENED
                event_class = PositionOpenedEvent
            elif old_position and not new_position:
                event_type = PositionEventType.POSITION_CLOSED
                event_class = PositionClosedEvent
            elif old_position and new_position:
                old_qty = Decimal(str(old_position.quantity))
                new_qty = Decimal(str(new_position.quantity))
                
                if abs(new_qty) > abs(old_qty):
                    event_type = PositionEventType.POSITION_INCREASED
                    event_class = PositionIncreasedEvent
                elif abs(new_qty) < abs(old_qty):
                    event_type = PositionEventType.POSITION_DECREASED
                    event_class = PositionDecreasedEvent
                elif (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
                    event_type = PositionEventType.POSITION_REVERSED
                    event_class = PositionReversedEvent
                else:
                    event_type = PositionEventType.FILL_PROCESSED
                    event_class = FillProcessedEvent
            else:
                event_type = PositionEventType.FILL_PROCESSED
                event_class = FillProcessedEvent
            
            # Create event with common parameters
            event_data = {
                'symbol': order.symbol,
                'old_position': old_position,
                'new_position': new_position,
                'trigger_order_id': order.order_id,
                'realized_pnl': realized_pnl,
                'metadata': {
                    'order_side': order.side.value,
                    'order_type': order.order_type.value,
                    'fill_price': float(fill_price),
                    'fill_quantity': float(fill_quantity)
                }
            }
            
            # Add event-specific data for FillProcessedEvent
            if event_class == FillProcessedEvent:
                event_data.update({
                    'fill_price': fill_price,
                    'fill_quantity': fill_quantity,
                    'commission': Decimal('0'),  # Would be passed from order
                    'fees': Decimal('0')
                })
            
            return event_class(**event_data)
            
        except Exception as e:
            logger.error(f"Error creating position event: {e}")
            # Return generic fill processed event
            return FillProcessedEvent(
                symbol=order.symbol,
                old_position=old_position,
                new_position=new_position,
                fill_price=fill_price,
                fill_quantity=fill_quantity,
                realized_pnl=realized_pnl,
                metadata={'error': str(e)}
            )
    
    def get_realized_pnl_today(self) -> Decimal:
        """Get realized P&L for today."""
        return self.realized_pnl_today
    
    def get_realized_pnl_total(self) -> Decimal:
        """Get total realized P&L."""
        return self.realized_pnl_total
    
    def reset_daily_pnl(self):
        """Reset daily P&L tracking (call at start of trading day)."""
        self.realized_pnl_today = Decimal('0')
        logger.info("Daily P&L tracking reset")
    
    def get_processing_metrics(self) -> Dict[str, float]:
        """Get fill processing metrics."""
        return {
            'realized_pnl_today': float(self.realized_pnl_today),
            'realized_pnl_total': float(self.realized_pnl_total),
            'processing_status': 'active'
        }