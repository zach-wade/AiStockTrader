"""
Volume-Weighted Average Price (VWAP) Execution Algorithm

Implements VWAP algorithm for optimal execution of large orders
by distributing trades according to historical volume patterns.
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
class VolumeProfile:
    """Historical volume profile for VWAP calculation."""
    time_buckets: List[datetime]
    volume_distribution: List[float]  # Percentage of daily volume
    cumulative_volume: List[float]    # Cumulative percentage
    

class VWAPAlgorithm:
    """VWAP execution algorithm for large orders."""
    
    def __init__(self, broker: BrokerInterface, config: Optional[Dict[str, Any]] = None):
        self.broker = broker
        self.config = config or {}
        self.active_executions = {}
        
        # Algorithm parameters
        self.default_buckets = self.config.get('default_buckets', 30)  # 30 time buckets
        self.min_order_size = self.config.get('min_order_size', 100)
        self.max_participation_rate = self.config.get('max_participation_rate', 0.1)  # 10% of volume
        self.use_historical_volume = self.config.get('use_historical_volume', True)
        
    async def execute(self,
                     symbol: str,
                     total_quantity: int,
                     side: str,
                     duration_minutes: int = 60,
                     start_time: Optional[datetime] = None,
                     volume_profile: Optional[VolumeProfile] = None,
                     limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute order using VWAP algorithm.
        
        Args:
            symbol: Stock symbol
            total_quantity: Total shares to execute
            side: 'BUY' or 'SELL'
            duration_minutes: Total execution duration
            start_time: When to start execution (default: now)
            volume_profile: Historical volume profile (optional)
            limit_price: Maximum price for buys, minimum for sells
            
        Returns:
            Execution summary with fills and performance metrics
        """
        execution_id = f"VWAP_{symbol}_{datetime.now().timestamp()}"
        start_time = start_time or datetime.now(timezone.utc)
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Get or create volume profile
        if volume_profile is None:
            volume_profile = await self._get_volume_profile(symbol, duration_minutes)
        
        # Initialize execution tracking
        self.active_executions[execution_id] = {
            'symbol': symbol,
            'total_quantity': total_quantity,
            'executed_quantity': 0,
            'side': side,
            'start_time': start_time,
            'end_time': end_time,
            'limit_price': limit_price,
            'fills': [],
            'vwap_price': 0.0,
            'status': 'ACTIVE',
            'slippage': 0.0
        }
        
        logger.info(f"Starting VWAP execution: {execution_id}")
        logger.info(f"Symbol: {symbol}, Quantity: {total_quantity}, Duration: {duration_minutes}min")
        
        try:
            # Execute according to volume profile
            await self._execute_vwap_schedule(
                execution_id, 
                symbol, 
                total_quantity,
                side,
                volume_profile,
                start_time,
                end_time,
                limit_price
            )
            
            # Calculate final metrics
            self._calculate_execution_metrics(execution_id)
            
            # Mark as complete
            self.active_executions[execution_id]['status'] = 'COMPLETED'
            
        except Exception as e:
            logger.error(f"VWAP execution failed: {e}")
            self.active_executions[execution_id]['status'] = 'FAILED'
            self.active_executions[execution_id]['error'] = str(e)
            
        return self._get_execution_summary(execution_id)
    
    async def _get_volume_profile(self, symbol: str, duration_minutes: int) -> VolumeProfile:
        """
        Get historical volume profile for the symbol.
        If historical data unavailable, use uniform distribution.
        """
        if self.use_historical_volume:
            # In production, fetch historical intraday volume data
            # For now, create a typical intraday volume profile
            return self._create_typical_volume_profile(duration_minutes)
        else:
            # Uniform distribution
            return self._create_uniform_volume_profile(duration_minutes)
    
    def _create_typical_volume_profile(self, duration_minutes: int) -> VolumeProfile:
        """
        Create a typical U-shaped intraday volume profile.
        Higher volume at open and close, lower in the middle.
        """
        num_buckets = min(duration_minutes, self.default_buckets)
        bucket_duration = duration_minutes / num_buckets
        
        # Generate U-shaped volume distribution
        x = np.linspace(0, 1, num_buckets)
        # U-shape: higher at ends, lower in middle
        volume_shape = 1.5 - np.cos(2 * np.pi * x)
        volume_shape = volume_shape / volume_shape.sum()
        
        # Create time buckets
        current_time = datetime.now(timezone.utc)
        time_buckets = [
            current_time + timedelta(minutes=i * bucket_duration)
            for i in range(num_buckets)
        ]
        
        # Calculate cumulative volume
        cumulative_volume = np.cumsum(volume_shape).tolist()
        
        return VolumeProfile(
            time_buckets=time_buckets,
            volume_distribution=volume_shape.tolist(),
            cumulative_volume=cumulative_volume
        )
    
    def _create_uniform_volume_profile(self, duration_minutes: int) -> VolumeProfile:
        """Create uniform volume distribution."""
        num_buckets = min(duration_minutes, self.default_buckets)
        bucket_duration = duration_minutes / num_buckets
        
        # Uniform distribution
        volume_per_bucket = 1.0 / num_buckets
        volume_distribution = [volume_per_bucket] * num_buckets
        
        # Create time buckets
        current_time = datetime.now(timezone.utc)
        time_buckets = [
            current_time + timedelta(minutes=i * bucket_duration)
            for i in range(num_buckets)
        ]
        
        # Calculate cumulative volume
        cumulative_volume = [
            (i + 1) * volume_per_bucket for i in range(num_buckets)
        ]
        
        return VolumeProfile(
            time_buckets=time_buckets,
            volume_distribution=volume_distribution,
            cumulative_volume=cumulative_volume
        )
    
    async def _execute_vwap_schedule(self,
                                    execution_id: str,
                                    symbol: str,
                                    total_quantity: int,
                                    side: str,
                                    volume_profile: VolumeProfile,
                                    start_time: datetime,
                                    end_time: datetime,
                                    limit_price: Optional[float]):
        """Execute orders according to VWAP schedule."""
        execution = self.active_executions[execution_id]
        remaining_quantity = total_quantity
        
        for i, (bucket_time, volume_pct) in enumerate(
            zip(volume_profile.time_buckets, volume_profile.volume_distribution)
        ):
            # Wait until bucket time
            current_time = datetime.now(timezone.utc)
            if current_time < bucket_time:
                wait_seconds = (bucket_time - current_time).total_seconds()
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
            
            # Calculate quantity for this bucket
            bucket_quantity = int(total_quantity * volume_pct)
            
            # Ensure minimum order size
            if bucket_quantity < self.min_order_size and remaining_quantity >= self.min_order_size:
                bucket_quantity = self.min_order_size
            
            # Don't exceed remaining quantity
            bucket_quantity = min(bucket_quantity, remaining_quantity)
            
            if bucket_quantity > 0:
                # Check market conditions before placing order
                if limit_price:
                    market_price = await self._get_market_price(symbol)
                    if not self._check_price_condition(market_price, limit_price, side):
                        logger.warning(f"Skipping bucket due to price condition: {market_price} vs {limit_price}")
                        continue
                
                # Submit child order
                await self._submit_child_order(
                    execution_id,
                    symbol,
                    bucket_quantity,
                    side,
                    limit_price
                )
                
                remaining_quantity -= bucket_quantity
                execution['executed_quantity'] += bucket_quantity
            
            # Check if execution is complete
            if remaining_quantity <= 0:
                logger.info(f"VWAP execution complete: {execution_id}")
                break
        
        # Handle any remaining quantity
        if remaining_quantity > 0:
            logger.warning(f"VWAP execution incomplete. Remaining: {remaining_quantity}")
            await self._submit_child_order(
                execution_id,
                symbol,
                remaining_quantity,
                side,
                limit_price
            )
    
    async def _submit_child_order(self,
                                 execution_id: str,
                                 symbol: str,
                                 quantity: int,
                                 side: str,
                                 limit_price: Optional[float]):
        """Submit individual child order."""
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
                    'algorithm': 'VWAP',
                    'parent_order': execution_id
                }
            )
            
            order_id = await self.broker.submit_order(order)
            
            if order_id:
                logger.info(f"VWAP child order submitted: {order_id} - {quantity} shares")
                
                # Track fill (in production, would get actual fill price)
                fill_price = await self._get_market_price(symbol)
                
                self.active_executions[execution_id]['fills'].append({
                    'order_id': order_id,
                    'quantity': quantity,
                    'price': fill_price,
                    'timestamp': datetime.now(timezone.utc)
                })
            
        except Exception as e:
            logger.error(f"Failed to submit VWAP child order: {e}")
    
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
    
    def _calculate_execution_metrics(self, execution_id: str):
        """Calculate VWAP and slippage metrics."""
        execution = self.active_executions[execution_id]
        fills = execution['fills']
        
        if not fills:
            return
        
        # Calculate executed VWAP
        total_value = sum(fill['quantity'] * fill['price'] for fill in fills)
        total_quantity = sum(fill['quantity'] for fill in fills)
        
        if total_quantity > 0:
            execution['vwap_price'] = total_value / total_quantity
        
        # Calculate slippage (would need market VWAP for comparison)
        # For now, calculate price drift from first to last fill
        if len(fills) > 1:
            first_price = fills[0]['price']
            last_price = fills[-1]['price']
            
            if execution['side'] == 'BUY':
                # For buys, positive drift is bad (paying more)
                execution['slippage'] = (last_price - first_price) / first_price
            else:
                # For sells, negative drift is bad (receiving less)
                execution['slippage'] = (first_price - last_price) / first_price
    
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
            'vwap_price': execution['vwap_price'],
            'slippage': execution['slippage'],
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
        """Cancel an active VWAP execution."""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            if execution['status'] == 'ACTIVE':
                execution['status'] = 'CANCELLED'
                logger.info(f"VWAP execution cancelled: {execution_id}")
                return True
        return False