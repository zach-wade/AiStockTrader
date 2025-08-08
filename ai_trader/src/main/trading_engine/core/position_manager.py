"""
Position Manager

Manages individual position state and lifecycle, working in coordination with
the existing PortfolioManager. Focuses on position-level operations while
PortfolioManager handles portfolio-level aggregation.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# Import existing models and systems
from main.models.common import Position, Order, OrderSide, OrderStatus
from main.trading_engine.core.position_events import (
    PositionEvent, PositionEventType,
    create_position_opened_event, create_position_closed_event, 
    create_fill_processed_event
)
from main.utils.cache import CacheType, get_global_cache

logger = logging.getLogger(__name__)


class PositionTracker:
    """
    Tracks position state and changes for individual positions.
    Lightweight component focused on state tracking, not portfolio aggregation.
    """
    
    def __init__(self, cache_ttl: int = 300):
        """
        Initialize position tracker.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.positions: Dict[str, Position] = {}
        self.position_history: Dict[str, List[Position]] = {}
        self.cache = get_global_cache()
        self.cache_ttl = cache_ttl
        
        # Thread safety
        self._lock = asyncio.Lock()
        self._position_locks: Dict[str, asyncio.Lock] = {}
        
        logger.info(" Position tracker initialized")
    
    async def _get_position_lock(self, symbol: str) -> asyncio.Lock:
        """Get or create lock for specific position."""
        if symbol not in self._position_locks:
            self._position_locks[symbol] = asyncio.Lock()
        return self._position_locks[symbol]
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        # Try cache first
        try:
            cached_pos = await self.cache.get(CacheType.POSITIONS, f"position:{symbol}")
            if cached_pos:
                return cached_pos
        except Exception as e:
            logger.debug(f"Cache retrieval failed for {symbol}: {e}")
        
        # Return from memory
        return self.positions.get(symbol)
    
    async def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return dict(self.positions)
    
    async def update_position(self, position: Position):
        """Update position state."""
        symbol = position.symbol
        position_lock = await self._get_position_lock(symbol)
        
        async with position_lock:
            # Store previous position for history
            old_position = self.positions.get(symbol)
            if old_position:
                if symbol not in self.position_history:
                    self.position_history[symbol] = []
                self.position_history[symbol].append(old_position)
                
                # Keep only last 100 historical positions
                if len(self.position_history[symbol]) > 100:
                    self.position_history[symbol] = self.position_history[symbol][-50:]
            
            # Update current position
            self.positions[symbol] = position
            
            # Update cache
            try:
                await self.cache.set(CacheType.POSITIONS, f"position:{symbol}", position, self.cache_ttl)
            except Exception as e:
                logger.debug(f"Cache update failed for {symbol}: {e}")
            
            logger.debug(f"Updated position for {symbol}: {position.quantity} @ {position.avg_entry_price}")
    
    async def remove_position(self, symbol: str):
        """Remove position (when fully closed)."""
        position_lock = await self._get_position_lock(symbol)
        
        async with position_lock:
            if symbol in self.positions:
                # Store final position in history
                final_position = self.positions[symbol]
                if symbol not in self.position_history:
                    self.position_history[symbol] = []
                self.position_history[symbol].append(final_position)
                
                # Remove from current positions
                del self.positions[symbol]
                
                # Remove from cache
                try:
                    await self.cache.delete(CacheType.POSITIONS, f"position:{symbol}")
                except Exception as e:
                    logger.debug(f"Cache deletion failed for {symbol}: {e}")
                
                logger.info(f"Removed position for {symbol}")
    
    def get_position_count(self) -> int:
        """Get number of active positions."""
        return len(self.positions)
    
    def get_symbols(self) -> Set[str]:
        """Get set of symbols with active positions."""
        return set(self.positions.keys())
    
    async def get_position_history(self, symbol: str, limit: int = 10) -> List[Position]:
        """Get historical positions for symbol."""
        history = self.position_history.get(symbol, [])
        return history[-limit:] if history else []


class PositionManager:
    """
    High-level position management coordinating with existing systems.
    
    Coordinates between:
    - PositionTracker (state tracking)
    - Existing PortfolioManager (portfolio-level operations)
    - Position events system
    """
    
    def __init__(self, portfolio_manager=None):
        """
        Initialize position manager.
        
        Args:
            portfolio_manager: Existing PortfolioManager instance
        """
        self.portfolio_manager = portfolio_manager
        self.position_tracker = PositionTracker()
        
        # Event handling
        self.event_handlers: List[callable] = []
        
        # Performance tracking
        self.position_metrics: Dict[str, Dict[str, Any]] = {}
        
        logger.info(" Position manager initialized")
    
    def add_event_handler(self, handler: callable):
        """Add event handler for position events."""
        self.event_handlers.append(handler)
        logger.debug(f"Added position event handler: {handler.__name__}")
    
    async def _emit_event(self, event: PositionEvent):
        """Emit position event to all handlers."""
        for handler in self.event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler {handler.__name__}: {e}")
    
    async def open_position(self, 
                          symbol: str, 
                          quantity: float, 
                          entry_price: float, 
                          side: OrderSide,
                          order_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Position:
        """
        Open a new position or add to existing position.
        
        Coordinates with PortfolioManager if available.
        """
        try:
            current_position = await self.position_tracker.get_position(symbol)
            
            if current_position:
                # Add to existing position
                new_position = await self._add_to_position(
                    current_position, quantity, entry_price, side
                )
                
                # Update portfolio manager if available
                if self.portfolio_manager:
                    await self.portfolio_manager.open_position(
                        symbol, quantity, entry_price, side
                    )
                
            else:
                # Create new position
                new_position = Position(
                    symbol=symbol,
                    quantity=quantity if side == OrderSide.BUY else -quantity,
                    avg_entry_price=entry_price,
                    current_price=entry_price,
                    market_value=abs(quantity) * entry_price,
                    cost_basis=abs(quantity) * entry_price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    realized_pnl=0.0,
                    side='long' if side == OrderSide.BUY else 'short',
                    timestamp=datetime.now(timezone.utc)
                )
                
                # Update portfolio manager if available
                if self.portfolio_manager:
                    await self.portfolio_manager.open_position(
                        symbol, quantity, entry_price, side
                    )
            
            # Update position tracker
            await self.position_tracker.update_position(new_position)
            
            # Emit event
            event = create_position_opened_event(
                symbol=symbol,
                new_position=new_position,
                trigger_order_id=order_id,
                metadata=metadata
            )
            await self._emit_event(event)
            
            logger.info(f"Opened/updated position: {symbol} {side.value} {quantity} @ {entry_price}")
            return new_position
            
        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {e}")
            raise
    
    async def close_position(self,
                           symbol: str,
                           quantity: Optional[float] = None,
                           exit_price: Optional[float] = None,
                           order_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Optional[Decimal]:
        """
        Close position (partial or full).
        
        Returns realized P&L from closed portion.
        """
        try:
            current_position = await self.position_tracker.get_position(symbol)
            if not current_position:
                logger.warning(f"No position found to close for {symbol}")
                return None
            
            # Default to current market price if not provided
            if exit_price is None:
                exit_price = current_position.current_price
            
            # Default to full close if quantity not specified
            close_quantity = quantity or abs(current_position.quantity)
            
            # Calculate realized P&L
            realized_pnl = self._calculate_realized_pnl(
                current_position, close_quantity, exit_price
            )
            
            # Update portfolio manager if available
            portfolio_pnl = None
            if self.portfolio_manager:
                portfolio_pnl = await self.portfolio_manager.close_position(
                    symbol, close_quantity, exit_price
                )
            
            # Determine if position is fully closed
            remaining_quantity = abs(current_position.quantity) - close_quantity
            
            if remaining_quantity <= 0.001:  # Essentially zero
                # Full close
                await self.position_tracker.remove_position(symbol)
                
                event = create_position_closed_event(
                    symbol=symbol,
                    old_position=current_position,
                    realized_pnl=realized_pnl,
                    trigger_order_id=order_id,
                    fill_price=exit_price,
                    metadata=metadata
                )
                await self._emit_event(event)
                
                logger.info(f"Fully closed position: {symbol} @ {exit_price}, P&L: {realized_pnl}")
                
            else:
                # Partial close - update position
                new_quantity = remaining_quantity if current_position.quantity > 0 else -remaining_quantity
                
                updated_position = Position(
                    symbol=symbol,
                    quantity=new_quantity,
                    avg_entry_price=current_position.avg_entry_price,
                    current_price=exit_price,
                    market_value=abs(new_quantity) * exit_price,
                    cost_basis=abs(new_quantity) * current_position.avg_entry_price,
                    unrealized_pnl=(exit_price - current_position.avg_entry_price) * new_quantity,
                    unrealized_pnl_pct=(exit_price / current_position.avg_entry_price - 1) * 100,
                    realized_pnl=current_position.realized_pnl + float(realized_pnl),
                    side=current_position.side,
                    timestamp=datetime.now(timezone.utc)
                )
                
                await self.position_tracker.update_position(updated_position)
                
                logger.info(f"Partially closed position: {symbol} {close_quantity} @ {exit_price}, remaining: {new_quantity}")
            
            return realized_pnl
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            raise
    
    async def update_market_prices(self, prices: Dict[str, float]):
        """Update current market prices for positions."""
        try:
            positions = await self.position_tracker.get_all_positions()
            
            for symbol, price in prices.items():
                if symbol in positions:
                    position = positions[symbol]
                    old_unrealized_pnl = position.unrealized_pnl
                    
                    # Create updated position
                    updated_position = Position(
                        symbol=symbol,
                        quantity=position.quantity,
                        avg_entry_price=position.avg_entry_price,
                        current_price=price,
                        market_value=abs(position.quantity) * price,
                        cost_basis=position.cost_basis,
                        unrealized_pnl=(price - position.avg_entry_price) * position.quantity,
                        unrealized_pnl_pct=(price / position.avg_entry_price - 1) * 100,
                        realized_pnl=position.realized_pnl,
                        side=position.side,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    await self.position_tracker.update_position(updated_position)
            
            # Update portfolio manager if available
            if self.portfolio_manager:
                await self.portfolio_manager.update_prices(prices)
                
        except Exception as e:
            logger.error(f"Error updating market prices: {e}")
    
    async def _add_to_position(self, 
                             current_position: Position, 
                             add_quantity: float, 
                             add_price: float,
                             side: OrderSide) -> Position:
        """Add to existing position with proper averaging."""
        signed_add_quantity = add_quantity if side == OrderSide.BUY else -add_quantity
        
        old_quantity = current_position.quantity
        new_quantity = old_quantity + signed_add_quantity
        
        # Calculate new average entry price
        old_cost = abs(old_quantity) * current_position.avg_entry_price
        add_cost = abs(signed_add_quantity) * add_price
        new_avg_price = (old_cost + add_cost) / abs(new_quantity)
        
        return Position(
            symbol=current_position.symbol,
            quantity=new_quantity,
            avg_entry_price=new_avg_price,
            current_price=add_price,
            market_value=abs(new_quantity) * add_price,
            cost_basis=abs(new_quantity) * new_avg_price,
            unrealized_pnl=(add_price - new_avg_price) * new_quantity,
            unrealized_pnl_pct=(add_price / new_avg_price - 1) * 100,
            realized_pnl=current_position.realized_pnl,
            side='long' if new_quantity > 0 else 'short',
            timestamp=datetime.now(timezone.utc)
        )
    
    def _calculate_realized_pnl(self, 
                               position: Position, 
                               close_quantity: float, 
                               exit_price: float) -> Decimal:
        """Calculate realized P&L for closing portion of position."""
        price_diff = exit_price - position.avg_entry_price
        
        # P&L calculation depends on position side
        if position.quantity > 0:  # Long position
            realized_pnl = close_quantity * price_diff
        else:  # Short position
            realized_pnl = close_quantity * -price_diff
        
        return Decimal(str(realized_pnl))
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        positions = await self.position_tracker.get_all_positions()
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions.values())
        total_market_value = sum(abs(pos.market_value) for pos in positions.values())
        
        return {
            'position_count': len(positions),
            'total_market_value': total_market_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'symbols': list(positions.keys()),
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'side': pos.side,
                    'avg_entry_price': pos.avg_entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'market_value': pos.market_value
                }
                for symbol, pos in positions.items()
            }
        }
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        return await self.position_tracker.get_position(symbol)
    
    async def get_all_positions(self) -> List[Position]:
        """Get list of all positions."""
        positions = await self.position_tracker.get_all_positions()
        return list(positions.values())
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info(" Position manager cleaned up")


def create_position_manager(portfolio_manager) -> PositionManager:
    """
    Factory function to create a position manager instance.
    
    Args:
        portfolio_manager: Portfolio manager instance
        
    Returns:
        PositionManager instance
    """
    return PositionManager(portfolio_manager)
