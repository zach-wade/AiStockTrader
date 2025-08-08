"""
Time-Weighted Average Price (TWAP) Execution Algorithm

Implements TWAP algorithm for optimal execution of large orders
by distributing trades evenly across time intervals.
"""

import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from dataclasses import dataclass

from main.trading_engine.brokers.broker_interface import BrokerInterface
from main.models.common import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class TWAPSchedule:
    """Schedule for TWAP execution."""
    time_slices: List[datetime]
    quantities: List[int]
    slice_duration: timedelta
    

class TWAPAlgorithm:
    """TWAP execution algorithm for large orders."""
    
    def __init__(self, broker: BrokerInterface, config: Optional[Dict[str, Any]] = None):
        self.broker = broker
        self.config = config or {}
        self.active_executions = {}
        
        # Algorithm parameters
        self.min_order_size = self.config.get('min_order_size', 100)
        self.max_slices = self.config.get('max_slices', 100)  # Max number of time slices
        self.min_slice_duration = self.config.get('min_slice_duration', 60)  # Seconds
        self.randomize_size = self.config.get('randomize_size', True)  # Add randomness to hide pattern
        self.randomize_timing = self.config.get('randomize_timing', True)  # Add timing randomness
        self.size_variance = self.config.get('size_variance', 0.2)  # 20% variance in slice sizes
        self.timing_variance = self.config.get('timing_variance', 0.1)  # 10% variance in timing
        
    async def execute(self,
                     symbol: str,
                     total_quantity: int,
                     side: str,
                     duration_minutes: int = 60,
                     start_time: Optional[datetime] = None,
                     limit_price: Optional[float] = None,
                     participation_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute order using TWAP algorithm.
        
        Args:
            symbol: Stock symbol
            total_quantity: Total shares to execute
            side: 'BUY' or 'SELL'
            duration_minutes: Total execution duration
            start_time: When to start execution (default: now)
            limit_price: Maximum price for buys, minimum for sells
            participation_rate: Max percentage of average volume (if provided)
            
        Returns:
            Execution summary with fills and performance metrics
        """
        execution_id = f"TWAP_{symbol}_{datetime.now().timestamp()}"
        start_time = start_time or datetime.now(timezone.utc)
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Create execution schedule
        schedule = self._create_twap_schedule(
            total_quantity, 
            start_time, 
            end_time,
            duration_minutes
        )
        
        # Initialize execution tracking
        self.active_executions[execution_id] = {
            'symbol': symbol,
            'total_quantity': total_quantity,
            'executed_quantity': 0,
            'side': side,
            'start_time': start_time,
            'end_time': end_time,
            'limit_price': limit_price,
            'participation_rate': participation_rate,
            'fills': [],
            'vwap_price': 0.0,
            'twap_price': 0.0,
            'status': 'ACTIVE',
            'schedule': schedule
        }
        
        logger.info(f"Starting TWAP execution: {execution_id}")
        logger.info(f"Symbol: {symbol}, Quantity: {total_quantity}, Duration: {duration_minutes}min")
        logger.info(f"Schedule: {len(schedule.time_slices)} slices, ~{total_quantity/len(schedule.time_slices):.0f} shares/slice")
        
        try:
            # Execute according to TWAP schedule
            await self._execute_twap_schedule(
                execution_id, 
                symbol, 
                side,
                schedule,
                limit_price,
                participation_rate
            )
            
            # Calculate final metrics
            self._calculate_execution_metrics(execution_id)
            
            # Mark as complete
            self.active_executions[execution_id]['status'] = 'COMPLETED'
            
        except Exception as e:
            logger.error(f"TWAP execution failed: {e}")
            self.active_executions[execution_id]['status'] = 'FAILED'
            self.active_executions[execution_id]['error'] = str(e)
            
        return self._get_execution_summary(execution_id)
    
    def _create_twap_schedule(self, 
                             total_quantity: int, 
                             start_time: datetime, 
                             end_time: datetime,
                             duration_minutes: int) -> TWAPSchedule:
        """
        Create TWAP execution schedule with equal time slices.
        """
        # Calculate number of slices
        duration_seconds = duration_minutes * 60
        slice_duration_seconds = max(self.min_slice_duration, duration_seconds / self.max_slices)
        num_slices = int(duration_seconds / slice_duration_seconds)
        
        # Ensure at least one slice
        num_slices = max(1, num_slices)
        
        # Calculate base quantity per slice
        base_quantity = total_quantity // num_slices
        remainder = total_quantity % num_slices
        
        # Create time slices
        slice_duration = timedelta(seconds=slice_duration_seconds)
        time_slices = []
        quantities = []
        
        for i in range(num_slices):
            # Calculate slice time (with optional randomization)
            slice_time = start_time + i * slice_duration
            
            if self.randomize_timing and i > 0:  # Don't randomize first slice
                # Add random variance to timing (except for first slice)
                max_variance_seconds = slice_duration_seconds * self.timing_variance
                random_offset = np.secure_uniform(-max_variance_seconds, max_variance_seconds)
                slice_time += timedelta(seconds=random_offset)
            
            time_slices.append(slice_time)
            
            # Calculate slice quantity
            slice_quantity = base_quantity
            
            # Distribute remainder across first slices
            if i < remainder:
                slice_quantity += 1
            
            # Add randomness to quantity if enabled
            if self.randomize_size and num_slices > 1:
                # Add variance while ensuring we don't go below min_order_size
                variance = np.secure_uniform(-self.size_variance, self.size_variance)
                adjusted_quantity = int(slice_quantity * (1 + variance))
                adjusted_quantity = max(self.min_order_size, adjusted_quantity)
                quantities.append(adjusted_quantity)
            else:
                quantities.append(slice_quantity)
        
        # Adjust last slice to ensure total matches exactly
        if self.randomize_size:
            total_scheduled = sum(quantities[:-1])
            quantities[-1] = total_quantity - total_scheduled
            
            # Ensure last slice isn't negative or too small
            if quantities[-1] < self.min_order_size and len(quantities) > 1:
                # Redistribute from previous slices
                redistribution = self.min_order_size - quantities[-1]
                quantities[-1] = self.min_order_size
                
                # Take from earlier slices
                for j in range(len(quantities) - 2, -1, -1):
                    if quantities[j] > self.min_order_size + redistribution:
                        quantities[j] -= redistribution
                        break
                    elif quantities[j] > self.min_order_size:
                        take = min(redistribution, quantities[j] - self.min_order_size)
                        quantities[j] -= take
                        redistribution -= take
                        if redistribution <= 0:
                            break
        
        return TWAPSchedule(
            time_slices=time_slices,
            quantities=quantities,
            slice_duration=slice_duration
        )
    
    async def _execute_twap_schedule(self,
                                    execution_id: str,
                                    symbol: str,
                                    side: str,
                                    schedule: TWAPSchedule,
                                    limit_price: Optional[float],
                                    participation_rate: Optional[float]):
        """Execute orders according to TWAP schedule."""
        execution = self.active_executions[execution_id]
        
        for i, (slice_time, slice_quantity) in enumerate(zip(schedule.time_slices, schedule.quantities)):
            # Check if execution has been cancelled
            if execution['status'] != 'ACTIVE':
                logger.info(f"TWAP execution {execution_id} cancelled or failed")
                break
            
            # Wait until slice time
            current_time = datetime.now(timezone.utc)
            if current_time < slice_time:
                wait_seconds = (slice_time - current_time).total_seconds()
                if wait_seconds > 0:
                    logger.debug(f"Waiting {wait_seconds:.1f}s until slice {i+1}/{len(schedule.time_slices)}")
                    await asyncio.sleep(wait_seconds)
            
            # Skip if we're too far behind schedule (e.g., system was down)
            if i < len(schedule.time_slices) - 1:  # Not the last slice
                next_slice_time = schedule.time_slices[i + 1]
                if datetime.now(timezone.utc) > next_slice_time:
                    logger.warning(f"Skipping slice {i+1} - too far behind schedule")
                    continue
            
            # Check participation rate if specified
            if participation_rate:
                can_trade = await self._check_participation_rate(
                    symbol, slice_quantity, participation_rate
                )
                if not can_trade:
                    logger.warning(f"Skipping slice {i+1} due to participation rate limit")
                    continue
            
            # Check market conditions before placing order
            if limit_price:
                market_price = await self._get_market_price(symbol)
                if not self._check_price_condition(market_price, limit_price, side):
                    logger.warning(f"Skipping slice {i+1} due to price condition: {market_price} vs {limit_price}")
                    continue
            
            # Submit child order
            await self._submit_child_order(
                execution_id,
                symbol,
                slice_quantity,
                side,
                limit_price,
                slice_number=i+1,
                total_slices=len(schedule.time_slices)
            )
            
            execution['executed_quantity'] += slice_quantity
            
            # Log progress
            progress = execution['executed_quantity'] / execution['total_quantity'] * 100
            logger.info(f"TWAP progress: {progress:.1f}% ({execution['executed_quantity']}/{execution['total_quantity']})")
    
    async def _check_participation_rate(self, symbol: str, quantity: int, max_rate: float) -> bool:
        """
        Check if trading quantity would exceed participation rate limit.
        
        Args:
            symbol: Stock symbol
            quantity: Intended order quantity
            max_rate: Maximum participation rate (0.0 to 1.0)
            
        Returns:
            True if trade is within participation limits
        """
        try:
            # Get recent volume data
            market_data = await self.broker.get_market_data(symbol)
            if market_data and hasattr(market_data, 'volume'):
                # Estimate volume rate (assuming uniform distribution)
                # In production, would use more sophisticated volume profiles
                trading_hours = 6.5  # Regular trading hours
                minutes_elapsed = (datetime.now() - datetime.now().replace(
                    hour=9, minute=30, second=0, microsecond=0
                )).total_seconds() / 60
                
                if minutes_elapsed > 0 and minutes_elapsed < trading_hours * 60:
                    # Estimate current volume rate
                    volume_rate = market_data.volume / minutes_elapsed  # shares per minute
                    
                    # Check if our order would exceed participation rate
                    our_rate = quantity  # Our shares per minute (for this slice)
                    participation = our_rate / (volume_rate + our_rate)
                    
                    return participation <= max_rate
            
            # If no volume data, allow trade
            return True
            
        except Exception as e:
            logger.error(f"Error checking participation rate: {e}")
            return True  # Allow trade on error
    
    async def _submit_child_order(self,
                                 execution_id: str,
                                 symbol: str,
                                 quantity: int,
                                 side: str,
                                 limit_price: Optional[float],
                                 slice_number: int,
                                 total_slices: int):
        """Submit individual child order for a time slice."""
        try:
            # Determine order type
            if limit_price:
                order_type = OrderType.LIMIT
                price = limit_price
            else:
                order_type = OrderType.MARKET
                price = None
            
            order = Order(
                symbol=symbol,
                quantity=quantity,
                side=OrderSide(side),
                order_type=order_type,
                limit_price=price,
                metadata={
                    'execution_id': execution_id,
                    'algorithm': 'TWAP',
                    'slice_number': slice_number,
                    'total_slices': total_slices,
                    'parent_order': execution_id
                }
            )
            
            order_id = await self.broker.submit_order(order)
            
            if order_id:
                logger.info(f"TWAP child order submitted: {order_id} - Slice {slice_number}/{total_slices} - {quantity} shares")
                
                # Track fill (in production, would get actual fill price)
                fill_price = await self._get_market_price(symbol)
                
                fill_data = {
                    'order_id': order_id,
                    'quantity': quantity,
                    'price': fill_price,
                    'timestamp': datetime.now(timezone.utc),
                    'slice_number': slice_number
                }
                
                self.active_executions[execution_id]['fills'].append(fill_data)
                
                # Update running TWAP calculation
                self._update_twap_price(execution_id)
            
        except Exception as e:
            logger.error(f"Failed to submit TWAP child order: {e}")
    
    async def _get_market_price(self, symbol: str) -> float:
        """Get current market price for the symbol."""
        try:
            # Get current quote from broker
            quote = await self.broker.get_quote(symbol)
            if quote:
                # Use mid-price between bid and ask
                bid_price = quote.get('bid_price', 0)
                ask_price = quote.get('ask_price', 0)
                
                if bid_price > 0 and ask_price > 0:
                    return (bid_price + ask_price) / 2
                elif bid_price > 0:
                    return bid_price
                elif ask_price > 0:
                    return ask_price
            
            # Fallback to last trade price
            market_data = await self.broker.get_market_data(symbol)
            if market_data and hasattr(market_data, 'last'):
                return market_data.last
                
        except Exception as e:
            logger.error(f"Failed to get market price for {symbol}: {e}")
        
        # Return a default if all else fails
        logger.warning(f"Using default price for {symbol}")
        return 100.0
    
    def _check_price_condition(self, market_price: float, limit_price: float, side: str) -> bool:
        """Check if market price meets limit price condition."""
        if side == 'BUY':
            return market_price <= limit_price
        else:  # SELL
            return market_price >= limit_price
    
    def _update_twap_price(self, execution_id: str):
        """Update running TWAP calculation after each fill."""
        execution = self.active_executions[execution_id]
        fills = execution['fills']
        
        if not fills:
            return
        
        # Calculate time-weighted average price
        # For true TWAP, we weight by time duration, but since our slices
        # are equal duration, this simplifies to arithmetic mean
        total_value = sum(fill['quantity'] * fill['price'] for fill in fills)
        total_quantity = sum(fill['quantity'] for fill in fills)
        
        if total_quantity > 0:
            execution['twap_price'] = total_value / total_quantity
    
    def _calculate_execution_metrics(self, execution_id: str):
        """Calculate final execution metrics."""
        execution = self.active_executions[execution_id]
        fills = execution['fills']
        
        if not fills:
            return
        
        # Calculate VWAP (Volume-Weighted Average Price)
        total_value = sum(fill['quantity'] * fill['price'] for fill in fills)
        total_quantity = sum(fill['quantity'] for fill in fills)
        
        if total_quantity > 0:
            execution['vwap_price'] = total_value / total_quantity
        
        # TWAP is already calculated incrementally
        
        # Calculate implementation shortfall
        if len(fills) >= 2:
            arrival_price = fills[0]['price']
            avg_price = execution['vwap_price']
            
            if execution['side'] == 'BUY':
                # For buys, positive shortfall means we paid more
                execution['implementation_shortfall'] = (avg_price - arrival_price) / arrival_price
            else:
                # For sells, positive shortfall means we received less
                execution['implementation_shortfall'] = (arrival_price - avg_price) / arrival_price
        else:
            execution['implementation_shortfall'] = 0.0
        
        # Calculate price improvement vs arrival price
        if fills:
            arrival_price = fills[0]['price']
            if execution['side'] == 'BUY':
                execution['price_improvement'] = (arrival_price - execution['vwap_price']) / arrival_price
            else:
                execution['price_improvement'] = (execution['vwap_price'] - arrival_price) / arrival_price
    
    def _get_execution_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get execution summary."""
        execution = self.active_executions[execution_id]
        
        return {
            'execution_id': execution_id,
            'symbol': execution['symbol'],
            'side': execution['side'],
            'total_quantity': execution['total_quantity'],
            'executed_quantity': execution['executed_quantity'],
            'execution_rate': execution['executed_quantity'] / execution['total_quantity'],
            'num_fills': len(execution['fills']),
            'num_slices': len(execution['schedule'].time_slices),
            'vwap_price': execution['vwap_price'],
            'twap_price': execution['twap_price'],
            'implementation_shortfall': execution.get('implementation_shortfall', 0.0),
            'price_improvement': execution.get('price_improvement', 0.0),
            'duration': (datetime.now(timezone.utc) - execution['start_time']).total_seconds() / 60,
            'status': execution['status']
        }
    
    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs."""
        return [
            exec_id for exec_id, execution in self.active_executions.items()
            if execution['status'] == 'ACTIVE'
        ]
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active TWAP execution."""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            if execution['status'] == 'ACTIVE':
                execution['status'] = 'CANCELLED'
                logger.info(f"TWAP execution cancelled: {execution_id}")
                return True
        return False
    
    async def modify_execution(self, 
                             execution_id: str,
                             new_limit_price: Optional[float] = None,
                             new_participation_rate: Optional[float] = None) -> bool:
        """
        Modify parameters of an active TWAP execution.
        
        Args:
            execution_id: ID of the execution to modify
            new_limit_price: New limit price constraint
            new_participation_rate: New participation rate limit
            
        Returns:
            True if modification was successful
        """
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            if execution['status'] == 'ACTIVE':
                if new_limit_price is not None:
                    execution['limit_price'] = new_limit_price
                    logger.info(f"Updated limit price for {execution_id} to {new_limit_price}")
                
                if new_participation_rate is not None:
                    execution['participation_rate'] = new_participation_rate
                    logger.info(f"Updated participation rate for {execution_id} to {new_participation_rate}")
                
                return True
        return False