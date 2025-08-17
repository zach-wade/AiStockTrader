# Circuit Breaker Components Documentation

## Overview

This document provides detailed documentation for all components in the modular circuit breaker system, including APIs, configuration options, and usage examples.

## Core Infrastructure Components

### 1. Types Module (`types.py`)

#### Purpose

Defines all core data structures, enums, and type definitions used throughout the circuit breaker system.

#### Key Classes

**BreakerType (Enum)**

```python
class BreakerType(Enum):
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    LOSS_RATE = "loss_rate"
    POSITION_LIMIT = "position_limit"
    KILL_SWITCH = "kill_switch"
    # ... additional types
```

**BreakerStatus (Enum)**

```python
class BreakerStatus(Enum):
    ACTIVE = "active"
    WARNING = "warning"
    TRIPPED = "tripped"
    COOLDOWN = "cooldown"
    EMERGENCY_HALT = "emergency_halt"
```

**BreakerEvent (Dataclass)**

```python
@dataclass
class BreakerEvent:
    timestamp: datetime
    breaker_type: BreakerType
    status: BreakerStatus
    message: str
    metrics: Dict[str, float]
    auto_reset_time: Optional[datetime] = None
```

**MarketConditions (Dataclass)**

```python
@dataclass
class MarketConditions:
    timestamp: datetime
    volatility: float
    correlation_matrix: Optional[np.ndarray] = None
    volume_ratio: float = 1.0
    # ... additional market data fields
```

### 2. Configuration Module (`config.py`)

#### Purpose

Centralized configuration management with validation and risk parameter handling.

#### Key Class: BreakerConfig

**Initialization**

```python
config = BreakerConfig({
    'volatility_threshold': 0.05,
    'max_drawdown': 0.08,
    'loss_rate_threshold': 0.03,
    'max_positions': 20
})
```

**Core Properties**

```python
# Risk thresholds
config.volatility_threshold    # 5% volatility threshold
config.max_drawdown           # 8% maximum drawdown
config.loss_rate_threshold    # 3% loss rate threshold
config.max_positions          # 20 maximum positions

# Time windows
config.loss_rate_window       # timedelta(minutes=5)
config.volatility_window      # timedelta(minutes=30)
config.cooldown_period        # timedelta(minutes=15)
```

**Configuration Validation**

```python
warnings = config.validate_limits()
if warnings:
    for param, warning in warnings.items():
        logger.warning(f"Config warning - {param}: {warning}")
```

### 3. Events Module (`events.py`)

#### Purpose

Manages events, callbacks, and state transitions for the circuit breaker system.

#### Key Classes

**BreakerEventManager**

```python
# Initialize event manager
event_manager = BreakerEventManager(max_events=1000)

# Register callbacks
event_manager.register_callback(my_callback_function)

# Record events
await event_manager.record_event(breaker_event)

# Get recent events
recent_events = event_manager.get_recent_events(limit=10)
```

**BreakerStateManager**

```python
# Initialize state manager
state_manager = BreakerStateManager(event_manager)

# Update breaker state
await state_manager.update_breaker_state(
    BreakerType.VOLATILITY,
    is_tripped=True,
    market_conditions,
    cooldown_time
)

# Check trading allowed
is_allowed = state_manager.is_trading_allowed()
```

### 4. Registry Module (`registry.py`)

#### Purpose

Manages registration and lifecycle of individual circuit breaker components.

#### Key Classes

**BaseBreaker (Abstract Base Class)**

```python
class BaseBreaker(ABC):
    @abstractmethod
    async def check(self, portfolio_value: float,
                   positions: Dict[str, Any],
                   market_conditions: MarketConditions) -> bool:
        pass

    @abstractmethod
    def get_metrics(self) -> BreakerMetrics:
        pass
```

**BreakerRegistry**

```python
# Initialize registry
registry = BreakerRegistry(config, event_manager, state_manager)

# Register breaker classes
registry.register_breaker_class(BreakerType.VOLATILITY, VolatilityBreaker)

# Initialize all breakers
await registry.initialize_breakers()

# Check all breakers
results = await registry.check_all_breakers(
    portfolio_value, positions, market_conditions
)
```

### 5. Facade Module (`facade.py`)

#### Purpose

Provides backward compatibility with the original monolithic circuit breaker interface.

#### Key Class: CircuitBreakerFacade

**Initialization**

```python
circuit_breaker = CircuitBreakerFacade(config_dict)
```

**Main Methods**

```python
# Check all breakers
results = await circuit_breaker.check_all_breakers(
    portfolio_value, positions, market_conditions
)

# Manual controls
await circuit_breaker.manual_trip("Emergency stop")
await circuit_breaker.manual_reset()

# Status checking
is_allowed = circuit_breaker.is_trading_allowed()
status = circuit_breaker.get_breaker_status()
```

## Specialized Breaker Components

### 1. VolatilityBreaker (`volatility_breaker.py`)

#### Purpose

Monitors market volatility and triggers protection when volatility exceeds safe levels or accelerates rapidly.

#### Key Features

- Spot volatility monitoring
- Volatility acceleration detection
- Breakout analysis with statistical thresholds
- Historical volatility trend analysis

#### Configuration

```python
config = {
    'volatility_threshold': 0.05,  # 5% volatility threshold
    'volatility_window_minutes': 30,  # 30-minute window
    'min_history_length': 10  # Minimum data points
}
```

#### Usage Example

```python
volatility_breaker = VolatilityBreaker(BreakerType.VOLATILITY, config)

# Check volatility conditions
is_tripped = await volatility_breaker.check(
    portfolio_value, positions, market_conditions
)

# Get volatility statistics
stats = volatility_breaker.get_volatility_statistics()
print(f"Current volatility: {stats['current_volatility']:.2%}")
```

#### Metrics Available

- Current volatility level
- Mean and standard deviation
- Volatility trend analysis
- Time above threshold
- Acceleration indicators

### 2. DrawdownBreaker (`drawdown_breaker.py`)

#### Purpose

Monitors portfolio drawdown and triggers protection when drawdown exceeds safe levels or accelerates rapidly.

#### Key Features

- Real-time drawdown calculation
- Drawdown acceleration monitoring
- Underwater period tracking
- Recovery pattern analysis

#### Configuration

```python
config = {
    'max_drawdown': 0.08,  # 8% maximum drawdown
    'warning_threshold': 0.064,  # 80% of max drawdown
    'acceleration_threshold': 0.004  # 0.4% acceleration
}
```

#### Usage Example

```python
drawdown_breaker = DrawdownBreaker(BreakerType.DRAWDOWN, config)

# Check drawdown conditions
is_tripped = await drawdown_breaker.check(
    portfolio_value, positions, market_conditions
)

# Get drawdown statistics
stats = drawdown_breaker.get_drawdown_statistics()
print(f"Current drawdown: {stats['current_drawdown']:.2%}")
print(f"Recovery factor: {stats['recovery_factor']:.2f}")
```

#### Metrics Available

- Current and maximum drawdown
- Recovery factor and underwater duration
- Drawdown trend analysis
- Recovery pattern statistics
- Portfolio peak tracking

### 3. LossRateBreaker (`loss_rate_breaker.py`)

#### Purpose

Monitors the velocity of losses and triggers protection when losses occur too rapidly within a specified time window.

#### Key Features

- Loss velocity monitoring
- Consecutive loss detection
- Loss acceleration analysis
- Pattern recognition for loss trends

#### Configuration

```python
config = {
    'loss_rate_threshold': 0.03,  # 3% loss rate threshold
    'loss_rate_window_minutes': 5,  # 5-minute window
    'consecutive_loss_limit': 5,  # Max consecutive losses
    'severe_loss_threshold': 0.06  # 6% severe loss threshold
}
```

#### Usage Example

```python
loss_rate_breaker = LossRateBreaker(BreakerType.LOSS_RATE, config)

# Check loss rate conditions
is_tripped = await loss_rate_breaker.check(
    portfolio_value, positions, market_conditions
)

# Get loss statistics
stats = loss_rate_breaker.get_loss_statistics()
print(f"Current loss rate: {stats['current_loss_rate']:.2%}")
print(f"Consecutive losses: {stats['consecutive_losses']}")
```

#### Metrics Available

- Current loss rate and maximum seen
- Consecutive loss count
- Loss frequency and magnitude
- Loss pattern analysis
- Time since last profit

### 4. PositionLimitBreaker (`position_limit_breaker.py`)

#### Purpose

Monitors position count and concentration limits to prevent overexposure and ensure proper risk diversification.

#### Key Features

- Position count monitoring
- Concentration risk analysis
- Sector diversification tracking
- Exposure limit enforcement

#### Configuration

```python
config = {
    'max_positions': 20,  # Maximum position count
    'max_position_size': 0.10,  # 10% maximum position size
    'max_sector_concentration': 0.30,  # 30% sector limit
    'max_long_exposure': 1.0,  # 100% long exposure
    'max_short_exposure': 0.5  # 50% short exposure
}
```

#### Usage Example

```python
position_limit_breaker = PositionLimitBreaker(BreakerType.POSITION_LIMIT, config)

# Check position limits
is_tripped = await position_limit_breaker.check(
    portfolio_value, positions, market_conditions
)

# Get position statistics
stats = position_limit_breaker.get_position_statistics()
print(f"Current positions: {stats['current_position_count']}")
print(f"Max concentration: {stats['max_concentration']:.2%}")
```

#### Metrics Available

- Position count and concentration
- Sector exposure analysis
- Long/short exposure ratios
- Diversification metrics (HHI)
- Risk contribution analysis

## Integration Examples

### Basic Usage

```python
from ai_trader.risk_management.real_time.circuit_breaker import CircuitBreaker

# Initialize circuit breaker (backward compatible)
config = {
    'volatility_threshold': 0.05,
    'max_drawdown': 0.08,
    'loss_rate_threshold': 0.03,
    'max_positions': 20
}

circuit_breaker = CircuitBreaker(config)

# Check all breakers
results = await circuit_breaker.check_all_breakers(
    portfolio_value=100000,
    positions=current_positions,
    market_conditions=market_data
)

# Handle results
if not circuit_breaker.is_trading_allowed():
    logger.warning("Trading halted by circuit breaker")
    # Implement halt logic
```

### Advanced Usage with Individual Breakers

```python
from ai_trader.risk_management.real_time.circuit_breaker.breakers import (
    VolatilityBreaker, DrawdownBreaker
)

# Initialize individual breakers
volatility_breaker = VolatilityBreaker(BreakerType.VOLATILITY, config)
drawdown_breaker = DrawdownBreaker(BreakerType.DRAWDOWN, config)

# Check specific breakers
vol_tripped = await volatility_breaker.check(
    portfolio_value, positions, market_conditions
)

dd_tripped = await drawdown_breaker.check(
    portfolio_value, positions, market_conditions
)

# Get detailed metrics
vol_stats = volatility_breaker.get_volatility_statistics()
dd_stats = drawdown_breaker.get_drawdown_statistics()
```

### Event Handling

```python
from ai_trader.risk_management.real_time.circuit_breaker.events import BreakerEventManager

# Set up event handling
event_manager = BreakerEventManager()

def handle_breaker_event(event):
    logger.warning(f"Breaker event: {event.breaker_type.value} - {event.message}")
    # Send alerts, update dashboard, etc.

event_manager.register_callback(handle_breaker_event)
```

## Error Handling

### Common Exceptions

```python
try:
    results = await circuit_breaker.check_all_breakers(
        portfolio_value, positions, market_conditions
    )
except ValueError as e:
    logger.error(f"Invalid input data: {e}")
except Exception as e:
    logger.error(f"Circuit breaker error: {e}")
    # Implement fallback logic
```

### Graceful Degradation

The system is designed to fail gracefully:

- Invalid configuration values use defaults
- Missing data is handled with appropriate warnings
- Individual breaker failures don't affect others
- Comprehensive logging for debugging

## Performance Considerations

### Optimization Tips

1. **Selective Execution**: Enable only needed breakers
2. **Efficient Data Structures**: Use appropriate data types
3. **Caching**: Cache expensive calculations
4. **Batch Processing**: Process multiple checks together

### Memory Management

- Historical data is limited by deque maxlen
- Old events are automatically cleaned up
- Memory usage is monitored and optimized

---
*Component Documentation*
*Version: 2.0*
*Last Updated: 2025-07-15*
*Author: AI Trading System Development Team*
