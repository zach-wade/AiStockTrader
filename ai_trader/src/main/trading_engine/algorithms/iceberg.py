"""Iceberg order execution algorithm."""
import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
from typing import Dict, Optional, Any
from datetime import datetime
from main.utils.core import secure_uniform  # DEPRECATED - use secure_random
from main.utils.core import secure_uniform, secure_randint, secure_choice, secure_sample, secure_shuffle
import logging

from main.trading_engine.brokers.broker_interface import BrokerInterface
from main.models.common import Order

logger = logging.getLogger(__name__)

class IcebergAlgorithm:
    """Iceberg order algorithm for hiding large order size."""
    
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
        self.active_executions = {}
    
    async def execute(self,
                     symbol: str,
                     total_quantity: int,
                     side: str,
                     visible_quantity: int,
                     price_limit: Optional[float] = None,
                     randomize: bool = True) -> Dict[str, Any]:
        """
        Execute iceberg order.
        
        Args:
            symbol: Stock symbol
            total_quantity: Total shares to execute
            side: 'BUY' or 'SELL'
            visible_quantity: Visible order size
            price_limit: Optional price limit
            randomize: Randomize child order sizes
        """
        execution_id = f"ICEBERG_{symbol}_{datetime.now().timestamp()}"
        
        # Store execution details
        self.active_executions[execution_id] = {
            'symbol': symbol,
            'total_quantity': total_quantity,
            'executed_quantity': 0,
            'side': side,
            'visible_quantity': visible_quantity,
            'price_limit': price_limit,
            'start_time': datetime.now(),
            'fills': [],
            'status': 'ACTIVE'
        }
        
        logger.info(f"Starting Iceberg execution: {execution_id}")
        
        # Execute iceberg slices
        while self.active_executions[execution_id]['executed_quantity'] < total_quantity:
            remaining = total_quantity - self.active_executions[execution_id]['executed_quantity']
            
            # Calculate slice size
            if randomize:
                # Randomize between 50% and 150% of visible quantity
                slice_size = int(visible_quantity * (0.5 + secure_uniform(0, 1)))
            else:
                slice_size = visible_quantity
            
            slice_size = min(slice_size, remaining)
            
            # Submit slice order
            await self._submit_slice_order(execution_id, symbol, slice_size, side, price_limit)
            
            # Update execution status
            self.active_executions[execution_id]['executed_quantity'] += slice_size
            
            # Wait before next slice (randomized delay)
            if remaining > slice_size:
                delay = secure_uniform(5, 30)  # 5-30 seconds
                await asyncio.sleep(delay)
        
        # Mark execution as complete
        self.active_executions[execution_id]['status'] = 'COMPLETED'
        
        return self._get_execution_summary(execution_id)
    
    async def _submit_slice_order(self, execution_id: str, symbol: str, 
                                 quantity: int, side: str, price_limit: Optional[float]):
        """Submit individual iceberg slice."""
        try:
            if price_limit:
                order = Order(
                    symbol=symbol,
                    quantity=quantity,
                    side=side,
                    order_type='LIMIT',
                    limit_price=price_limit,
                    metadata={'execution_id': execution_id, 'algorithm': 'ICEBERG'}
                )
            else:
                order = Order(
                    symbol=symbol,
                    quantity=quantity,
                    side=side,
                    order_type='MARKET',
                    metadata={'execution_id': execution_id, 'algorithm': 'ICEBERG'}
                )
            
            order_id = await self.broker.submit_order(order)
            
            if order_id:
                logger.info(f"Iceberg slice submitted: {order_id} - {quantity} shares")
                
                self.active_executions[execution_id]['fills'].append({
                    'order_id': order_id,
                    'quantity': quantity,
                    'timestamp': datetime.now()
                })
            
        except Exception as e:
            logger.error(f"Failed to submit iceberg slice: {e}")
    
    def _get_execution_summary(self, execution_id: str) -> Dict[str, Any]:
        """Get execution summary."""
        execution = self.active_executions[execution_id]
        
        return {
            'execution_id': execution_id,
            'symbol': execution['symbol'],
            'side': execution['side'],
            'total_quantity': execution['total_quantity'],
            'executed_quantity': execution['executed_quantity'],
            'num_slices': len(execution['fills']),
            'avg_slice_size': execution['executed_quantity'] / len(execution['fills']) if execution['fills'] else 0,
            'duration': (datetime.now() - execution['start_time']).total_seconds() / 60,
            'status': execution['status']
        }