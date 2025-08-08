

# File: utils/base_algorithm.py

"""
BaseAlgorithm class consolidating common trading algorithm patterns.

This consolidates patterns from TWAP, VWAP, and Iceberg algorithms into a unified
base class that provides standard execution lifecycle, state management, and monitoring.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from uuid import uuid4

# SECURITY FIX: Import secure random for G2.4 vulnerability fix
from main.utils.core import secure_uniform

import pandas as pd

from main.utils.core import async_retry as async_retry_decorator
from main.utils.core import secure_dumps as to_json, secure_loads as from_json

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class ExecutionStatus(Enum):
    """Execution status states."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class SlicingStrategy(Enum):
    """Order slicing strategies."""
    EQUAL_SIZE = "equal_size"
    EQUAL_TIME = "equal_time"
    VOLUME_WEIGHTED = "volume_weighted"
    RANDOMIZED = "randomized"
    ADAPTIVE = "adaptive"


@dataclass
class ExecutionParameters:
    """Standard execution parameters for all algorithms."""
    symbol: str
    total_quantity: int
    side: OrderSide
    time_horizon_minutes: int = 60
    max_slices: int = 10
    min_slice_size: int = 1
    randomize_timing: bool = True
    randomize_sizing: bool = False
    max_participation_rate: float = 0.1  # 10% of volume
    price_limit: Optional[float] = None
    emergency_stop_loss: Optional[float] = None
    algorithm_specific_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildOrder:
    """Represents a child order within an execution."""
    order_id: str
    execution_id: str
    symbol: str
    quantity: int
    side: OrderSide
    price: Optional[float] = None
    order_type: str = "MARKET"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = "PENDING"
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class ExecutionState:
    """Tracks the state of an algorithm execution."""
    execution_id: str
    algorithm_name: str
    parameters: ExecutionParameters
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    executed_quantity: int = 0
    remaining_quantity: int = 0
    child_orders: List[ChildOrder] = field(default_factory=list)
    fills: List[Dict[str, Any]] = field(default_factory=list)
    current_slice: int = 0
    total_slices: int = 0
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize remaining quantity."""
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.parameters.total_quantity


@dataclass
class ExecutionSummary:
    """Summary of execution results."""
    execution_id: str
    algorithm_name: str
    symbol: str
    total_quantity: int
    executed_quantity: int
    side: OrderSide
    status: ExecutionStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: float
    num_child_orders: int
    num_fills: int
    avg_fill_price: Optional[float]
    total_value: float
    participation_rate: float
    slippage_bps: Optional[float] = None
    benchmark_performance: Optional[float] = None
    error_messages: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BASE ALGORITHM CLASS
# =============================================================================

class BaseAlgorithm(ABC):
    """
    Base class for all trading execution algorithms.
    
    Consolidates common patterns from:
    - TWAP: Time-weighted average price execution
    - VWAP: Volume-weighted average price execution  
    - Iceberg: Hidden quantity execution with small visible slices
    """
    
    def __init__(self, 
                 broker_interface,
                 algorithm_name: str,
                 enable_metrics: bool = True,
                 enable_adaptive_sizing: bool = False):
        """
        Initialize base algorithm.
        
        Args:
            broker_interface: Broker interface for order execution
            algorithm_name: Name of the specific algorithm
            enable_metrics: Enable performance metrics collection
            enable_adaptive_sizing: Enable adaptive slice sizing
        """
        self.broker = broker_interface
        self.algorithm_name = algorithm_name
        self.enable_metrics = enable_metrics
        self.enable_adaptive_sizing = enable_adaptive_sizing
        
        # State tracking
        self.active_executions: Dict[str, ExecutionState] = {}
        self.completed_executions: Dict[str, ExecutionState] = {}
        
        # Performance metrics
        self.execution_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_quantity_executed': 0,
            'average_execution_time': 0.0,
            'average_slippage_bps': 0.0
        }
        
        # Market data cache
        from main.utils.cache import MemoryBackend
        from main.utils.cache import CacheType
        self.cache = get_global_cache()
        
        logger.info(f"Initialized {algorithm_name} algorithm")
    
    # =============================================================================
    # PUBLIC INTERFACE
    # =============================================================================
    
    async def execute(self, 
                     symbol: str,
                     total_quantity: int,
                     side: Union[str, OrderSide],
                     **kwargs) -> ExecutionSummary:
        """
        Main execution entry point.
        
        Args:
            symbol: Trading symbol
            total_quantity: Total quantity to execute
            side: Order side (BUY/SELL)
            **kwargs: Algorithm-specific parameters
            
        Returns:
            ExecutionSummary with results
        """
        # Convert side to enum if needed
        if isinstance(side, str):
            side = OrderSide(side.upper())
        
        # Create execution parameters
        params = ExecutionParameters(
            symbol=symbol,
            total_quantity=total_quantity,
            side=side,
            **kwargs
        )
        
        # Validate parameters
        validation_error = await self._validate_parameters(params)
        if validation_error:
            raise ValueError(f"Parameter validation failed: {validation_error}")
        
        # Initialize execution
        execution_id = self._generate_execution_id(symbol)
        execution_state = ExecutionState(
            execution_id=execution_id,
            algorithm_name=self.algorithm_name,
            parameters=params
        )
        
        # Store execution state
        self.active_executions[execution_id] = execution_state
        
        try:
            # Set start time and status
            execution_state.start_time = datetime.utcnow()
            execution_state.status = ExecutionStatus.ACTIVE
            
            # Call lifecycle hook: execution start
            await self.on_execution_start(execution_state)
            
            # Execute algorithm-specific logic
            await self._execute_algorithm(execution_state)
            
            # Mark as completed
            execution_state.status = ExecutionStatus.COMPLETED
            execution_state.end_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Algorithm execution failed: {e}")
            execution_state.status = ExecutionStatus.FAILED
            execution_state.error_messages.append(str(e))
            execution_state.end_time = datetime.utcnow()
        
        finally:
            # Call lifecycle hook: execution complete
            await self.on_execution_complete(execution_state)
            
            # Move to completed executions
            self.completed_executions[execution_id] = execution_state
            del self.active_executions[execution_id]
            
            # Update metrics
            if self.enable_metrics:
                await self._update_execution_metrics(execution_state)
        
        return await self._create_execution_summary(execution_state)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active execution.
        
        Args:
            execution_id: ID of execution to cancel
            
        Returns:
            True if successfully cancelled
        """
        if execution_id not in self.active_executions:
            logger.warning(f"Execution {execution_id} not found or not active")
            return False
        
        execution_state = self.active_executions[execution_id]
        
        try:
            # Cancel pending child orders
            for child_order in execution_state.child_orders:
                if child_order.status == "PENDING":
                    await self._cancel_child_order(child_order)
            
            # Update status
            execution_state.status = ExecutionStatus.CANCELLED
            execution_state.end_time = datetime.utcnow()
            
            # Move to completed
            self.completed_executions[execution_id] = execution_state
            del self.active_executions[execution_id]
            
            logger.info(f"Successfully cancelled execution {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel execution {execution_id}: {e}")
            return False
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause an active execution."""
        if execution_id not in self.active_executions:
            return False
        
        execution_state = self.active_executions[execution_id]
        execution_state.status = ExecutionStatus.PAUSED
        
        logger.info(f"Paused execution {execution_id}")
        return True
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution."""
        if execution_id not in self.active_executions:
            return False
        
        execution_state = self.active_executions[execution_id]
        if execution_state.status == ExecutionStatus.PAUSED:
            execution_state.status = ExecutionStatus.ACTIVE
            
            # Continue execution from current state
            await self._execute_algorithm(execution_state)
            
            logger.info(f"Resumed execution {execution_id}")
            return True
        
        return False
    
    async def get_execution_status(self, execution_id: str) -> Optional[ExecutionSummary]:
        """Get current status of an execution."""
        execution_state = (self.active_executions.get(execution_id) or 
                          self.completed_executions.get(execution_id))
        
        if execution_state:
            return await self._create_execution_summary(execution_state)
        return None
    
    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs."""
        return list(self.active_executions.keys())
    
    def get_algorithm_metrics(self) -> Dict[str, Any]:
        """Get algorithm performance metrics."""
        return {
            'algorithm_name': self.algorithm_name,
            'metrics': self.execution_metrics.copy(),
            'active_executions': len(self.active_executions),
            'completed_executions': len(self.completed_executions)
        }
    
    # =============================================================================
    # ABSTRACT METHODS - TO BE IMPLEMENTED BY SUBCLASSES
    # =============================================================================
    
    @abstractmethod
    async def _execute_algorithm(self, execution_state: ExecutionState) -> None:
        """
        Algorithm-specific execution logic.
        
        Args:
            execution_state: Current execution state
        """
        pass
    
    @abstractmethod
    def _calculate_slice_size(self, 
                            execution_state: ExecutionState, 
                            current_slice: int) -> int:
        """
        Calculate the size of the current slice.
        
        Args:
            execution_state: Current execution state
            current_slice: Current slice number (0-indexed)
            
        Returns:
            Size of the slice in shares
        """
        pass
    
    @abstractmethod
    def _get_next_delay(self, 
                       execution_state: ExecutionState, 
                       current_slice: int) -> float:
        """
        Calculate delay before next slice.
        
        Args:
            execution_state: Current execution state
            current_slice: Current slice number
            
        Returns:
            Delay in seconds
        """
        pass
    
    # =============================================================================
    # COMMON INFRASTRUCTURE METHODS
    # =============================================================================
    
    def _generate_execution_id(self, symbol: str) -> str:
        """Generate unique execution ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid4())[:8]
        return f"{self.algorithm_name}_{symbol}_{timestamp}_{unique_id}"
    
    async def _validate_parameters(self, params: ExecutionParameters) -> Optional[str]:
        """
        Validate execution parameters.
        
        Returns:
            Error message if validation fails, None if valid
        """
        if params.total_quantity <= 0:
            return "Total quantity must be positive"
        
        if params.time_horizon_minutes <= 0:
            return "Time horizon must be positive"
        
        if params.max_slices <= 0:
            return "Max slices must be positive"
        
        if params.min_slice_size <= 0:
            return "Min slice size must be positive"
        
        if params.max_participation_rate <= 0 or params.max_participation_rate > 1:
            return "Participation rate must be between 0 and 1"
        
        # Algorithm-specific validation
        return await self._validate_algorithm_parameters(params)
    
    async def _validate_algorithm_parameters(self, params: ExecutionParameters) -> Optional[str]:
        """Algorithm-specific parameter validation. Override in subclasses."""
        return None
    
    @async_retry_decorator(max_attempts=3, delay=1.0)
    async def _submit_child_order(self, 
                                execution_state: ExecutionState,
                                slice_size: int,
                                price: Optional[float] = None) -> ChildOrder:
        """
        Submit a child order as part of the execution.
        
        Args:
            execution_state: Current execution state
            slice_size: Size of this slice
            price: Optional limit price
            
        Returns:
            ChildOrder object with submission results
        """
        params = execution_state.parameters
        
        # Create child order
        child_order = ChildOrder(
            order_id=f"{execution_state.execution_id}_{execution_state.current_slice}",
            execution_id=execution_state.execution_id,
            symbol=params.symbol,
            quantity=slice_size,
            side=params.side,
            price=price,
            order_type="LIMIT" if price else "MARKET"
        )
        
        try:
            # Submit order through broker
            order_result = await self.broker.submit_order(
                symbol=params.symbol,
                quantity=slice_size,
                side=params.side.value,
                order_type=child_order.order_type,
                price=price,
                metadata={'execution_id': execution_state.execution_id}
            )
            
            # Update child order with result
            child_order.order_id = order_result.get('order_id', child_order.order_id)
            child_order.status = "SUBMITTED"
            
            # Add to execution state
            execution_state.child_orders.append(child_order)
            
            # Call lifecycle hook: slice submitted
            await self.on_slice_submitted(execution_state, child_order)
            
            logger.info(f"Submitted child order {child_order.order_id} for {slice_size} shares")
            
            return child_order
            
        except Exception as e:
            child_order.status = "FAILED"
            child_order.error_message = str(e)
            execution_state.error_messages.append(f"Child order failed: {e}")
            
            logger.error(f"Failed to submit child order: {e}")
            raise
    
    async def _cancel_child_order(self, child_order: ChildOrder) -> bool:
        """Cancel a child order."""
        try:
            await self.broker.cancel_order(child_order.order_id)
            child_order.status = "CANCELLED"
            return True
        except Exception as e:
            logger.error(f"Failed to cancel child order {child_order.order_id}: {e}")
            return False
    
    async def _update_execution_progress(self, 
                                       execution_state: ExecutionState,
                                       fill_quantity: int,
                                       fill_price: float) -> None:
        """Update execution progress with a new fill."""
        execution_state.executed_quantity += fill_quantity
        execution_state.remaining_quantity -= fill_quantity
        
        # Add fill record
        fill_record = {
            'timestamp': datetime.utcnow(),
            'quantity': fill_quantity,
            'price': fill_price,
            'value': fill_quantity * fill_price
        }
        execution_state.fills.append(fill_record)
        
        # Update metadata
        execution_state.metadata['avg_fill_price'] = (
            sum(f['price'] * f['quantity'] for f in execution_state.fills) /
            execution_state.executed_quantity
        )
        
        logger.debug(f"Execution {execution_state.execution_id}: "
                    f"{execution_state.executed_quantity}/{execution_state.parameters.total_quantity} filled")
    
    def _calculate_remaining_quantity(self, execution_state: ExecutionState) -> int:
        """Calculate remaining quantity to execute."""
        return execution_state.parameters.total_quantity - execution_state.executed_quantity
    
    def _add_randomization(self, 
                          base_value: Union[int, float], 
                          randomization_factor: float = 0.1) -> Union[int, float]:
        """Add randomization to a value."""
        if randomization_factor <= 0:
            return base_value
        
        variation = base_value * randomization_factor
        # SECURITY FIX: G2.4 - Replace insecure secure_uniform() with cryptographically secure alternative
        random_adjustment = secure_uniform(-variation, variation)
        
        if isinstance(base_value, int):
            return max(1, int(base_value + random_adjustment))
        else:
            return max(0.1, base_value + random_adjustment)
    
    async def _create_execution_summary(self, execution_state: ExecutionState) -> ExecutionSummary:
        """Create execution summary from state."""
        duration = 0.0
        if execution_state.start_time and execution_state.end_time:
            duration = (execution_state.end_time - execution_state.start_time).total_seconds()
        
        # Calculate average fill price
        avg_fill_price = None
        if execution_state.fills:
            total_value = sum(f['price'] * f['quantity'] for f in execution_state.fills)
            avg_fill_price = total_value / execution_state.executed_quantity
        
        # Calculate participation rate based on execution data
        participation_rate = await self._calculate_participation_rate(execution_state)
        
        return ExecutionSummary(
            execution_id=execution_state.execution_id,
            algorithm_name=execution_state.algorithm_name,
            symbol=execution_state.parameters.symbol,
            total_quantity=execution_state.parameters.total_quantity,
            executed_quantity=execution_state.executed_quantity,
            side=execution_state.parameters.side,
            status=execution_state.status,
            start_time=execution_state.start_time,
            end_time=execution_state.end_time,
            duration_seconds=duration,
            num_child_orders=len(execution_state.child_orders),
            num_fills=len(execution_state.fills),
            avg_fill_price=avg_fill_price,
            total_value=sum(f['value'] for f in execution_state.fills),
            participation_rate=participation_rate,
            error_messages=execution_state.error_messages.copy()
        )
    
    async def _update_execution_metrics(self, execution_state: ExecutionState) -> None:
        """Update algorithm performance metrics."""
        self.execution_metrics['total_executions'] += 1
        
        if execution_state.status == ExecutionStatus.COMPLETED:
            self.execution_metrics['successful_executions'] += 1
        else:
            self.execution_metrics['failed_executions'] += 1
        
        self.execution_metrics['total_quantity_executed'] += execution_state.executed_quantity
        
        # Update average execution time
        if execution_state.start_time and execution_state.end_time:
            duration = (execution_state.end_time - execution_state.start_time).total_seconds()
            total_execs = self.execution_metrics['total_executions']
            current_avg = self.execution_metrics['average_execution_time']
            new_avg = ((current_avg * (total_execs - 1)) + duration) / total_execs
            self.execution_metrics['average_execution_time'] = new_avg
    
    async def _calculate_participation_rate(self, execution_state: ExecutionState) -> float:
        """Calculate the participation rate for this execution."""
        try:
            # Get volume profile for the symbol
            volume_profile = await self._get_volume_profile(
                execution_state.parameters.symbol, 
                days=5  # Use 5-day average
            )
            
            if volume_profile is None or volume_profile.empty:
                logger.warning(f"No volume data available for {execution_state.parameters.symbol}")
                return 0.0
            
            # Calculate average daily volume
            avg_daily_volume = volume_profile['volume'].mean()
            
            if avg_daily_volume <= 0:
                return 0.0
            
            # Calculate execution duration in trading hours
            duration_hours = 0
            if execution_state.start_time and execution_state.end_time:
                duration_seconds = (execution_state.end_time - execution_state.start_time).total_seconds()
                duration_hours = duration_seconds / 3600
            
            # Estimate volume during execution period
            # Assuming 6.5 trading hours per day
            trading_hours_per_day = 6.5
            estimated_period_volume = avg_daily_volume * (duration_hours / trading_hours_per_day)
            
            if estimated_period_volume <= 0:
                return 0.0
            
            # Calculate participation rate
            participation_rate = execution_state.executed_quantity / estimated_period_volume
            
            # Cap at reasonable bounds
            return min(participation_rate, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating participation rate: {e}")
            return 0.0
    
    # =============================================================================
    # ALGORITHM LIFECYCLE HOOKS
    # =============================================================================
    
    async def on_execution_start(self, execution_state: ExecutionState) -> None:
        """
        Hook called when execution starts.
        Override in subclasses for algorithm-specific initialization.
        """
        logger.debug(f"Execution started: {execution_state.execution_id}")
        
        # Default behavior: log execution parameters
        params = execution_state.parameters
        logger.info(
            f"Starting {self.algorithm_name} execution for {params.symbol}: "
            f"{params.total_quantity} shares over {params.time_horizon_minutes} minutes"
        )
    
    async def on_slice_submitted(self, execution_state: ExecutionState, 
                               child_order: ChildOrder) -> None:
        """
        Hook called when a slice (child order) is submitted.
        Override in subclasses for algorithm-specific behavior.
        """
        logger.debug(f"Slice submitted: {child_order.order_id} for {child_order.quantity} shares")
        
        # Default behavior: update progress metrics
        progress_pct = (execution_state.current_slice / execution_state.parameters.max_slices) * 100
        logger.info(f"Execution {execution_state.execution_id} progress: {progress_pct:.1f}%")
    
    async def on_execution_complete(self, execution_state: ExecutionState) -> None:
        """
        Hook called when execution completes (successfully or with error).
        Override in subclasses for algorithm-specific cleanup.
        """
        logger.debug(f"Execution completed: {execution_state.execution_id}")
        
        # Default behavior: log execution summary
        fill_rate = (execution_state.executed_quantity / execution_state.parameters.total_quantity) * 100
        logger.info(
            f"Execution {execution_state.execution_id} completed: "
            f"{execution_state.executed_quantity}/{execution_state.parameters.total_quantity} "
            f"({fill_rate:.1f}%) filled with status {execution_state.status.value}"
        )
    
    # =============================================================================
    # MARKET DATA INTEGRATION
    # =============================================================================
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        try:
            market_data = await self.broker.get_current_quote(symbol)
            return market_data.get('last_price') or market_data.get('mid_price')
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    async def _get_volume_profile(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical volume profile for symbol."""
        cache_key = f"volume_profile:{symbol}_{days}d"
        
        # Check cache first
        cached_profile = await self.cache.get(CacheType.MARKET_DATA, cache_key)
        if cached_profile is not None:
            return cached_profile
        
        try:
            # Fetch volume data from broker
            volume_data = await self.broker.get_historical_volume(symbol, days)
            
            if volume_data:
                # Cache the result for 1 hour
                await self.cache.set(CacheType.MARKET_DATA, cache_key, volume_data, 3600)
                return volume_data
                
        except Exception as e:
            logger.error(f"Failed to get volume profile for {symbol}: {e}")
        
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_execution_parameters(
    symbol: str,
    total_quantity: int,
    side: str,
    time_horizon_minutes: int = 60,
    max_slices: int = 10,
    **kwargs
) -> ExecutionParameters:
    """Create execution parameters with defaults."""
    return ExecutionParameters(
        symbol=symbol,
        total_quantity=total_quantity,
        side=OrderSide(side.upper()),
        time_horizon_minutes=time_horizon_minutes,
        max_slices=max_slices,
        **kwargs
    )


def calculate_equal_slices(total_quantity: int, num_slices: int) -> List[int]:
    """Calculate equal slice sizes with remainder handling."""
    base_size = total_quantity // num_slices
    remainder = total_quantity % num_slices
    
    slices = [base_size] * num_slices
    
    # Distribute remainder across first slices
    for i in range(remainder):
        slices[i] += 1
    
    return slices


def calculate_time_intervals(total_minutes: int, num_intervals: int) -> List[float]:
    """Calculate time intervals in seconds."""
    interval_minutes = total_minutes / num_intervals
    return [interval_minutes * 60] * num_intervals