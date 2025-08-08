# Execution Engine Integration Guide

## Overview

The AI Trader execution engine provides a comprehensive framework for managing trading operations across multiple brokers, with integrated risk management, position tracking, and performance monitoring.

## Architecture

### Core Components

1. **ExecutionEngine** (`execution_engine.py`)
   - Top-level orchestrator for all trading operations
   - Manages multiple TradingSystems (one per broker)
   - Coordinates cross-system operations
   - Provides emergency controls

2. **TradingSystem** (`trading_system.py`)
   - Per-broker trading coordinator
   - Integrates portfolio, position, and order management
   - Handles broker-specific operations

3. **ExecutionManager** (`execution_manager.py`)
   - Orchestration layer component
   - Provides high-level execution interface
   - Integrates with monitoring and risk management

### Supporting Components

- **PortfolioManager**: Tracks portfolio state and P&L
- **PositionManager**: Manages position lifecycle and events
- **OrderManager**: Handles order routing and tracking
- **FastExecutionPath**: Low-latency execution for time-sensitive orders
- **Risk Management**: Circuit breakers, drawdown control, position limits

## Integration Points

### 1. Trading Algorithms

Algorithms can integrate with the execution engine through the ExecutionManager:

```python
from main.orchestration.managers.execution_manager import ExecutionManager

# In your algorithm
async def execute_signal(self, signal):
    order_id = await self.execution_manager.process_signal(signal)
    return order_id
```

### 2. Broker Integration

The execution engine supports multiple brokers simultaneously:

- Alpaca
- Interactive Brokers
- Paper Trading (built-in simulator)

Each broker is managed by a dedicated TradingSystem instance.

### 3. Risk Management

Risk controls are applied at multiple levels:

```python
# Order-level risk checks
- Position size limits
- Order size limits
- Daily loss limits

# System-level controls
- Circuit breakers
- Drawdown control
- Emergency stop/liquidation
```

### 4. Event System

The execution engine publishes events for:

- Order status updates
- Position changes
- Fill notifications
- System status changes

Subscribe to events:

```python
engine.add_event_handler(your_event_handler)
```

## Usage Examples

### Basic Usage

```python
# Create and initialize execution engine
engine = await create_execution_engine(
    config=config,
    trading_mode=TradingMode.PAPER,
    execution_mode=ExecutionMode.SEMI_AUTO
)

# Start trading
await engine.start_trading()

# Submit an order
order = Order(
    symbol='AAPL',
    side=OrderSide.BUY,
    quantity=100,
    order_type=OrderType.MARKET
)
order_id = await engine.submit_cross_system_order(order)

# Check status
status = await engine.get_comprehensive_status()
```

### Advanced Features

#### Multi-Broker Execution

```python
# Submit order with broker preference
order_id = await engine.submit_cross_system_order(
    order,
    preferred_broker='alpaca'
)
```

#### Emergency Controls

```python
# Emergency stop - cancels all orders, disables trading
await engine.emergency_stop()

# Emergency liquidation - closes all positions
await engine.emergency_liquidate_all()
```

#### Position Synchronization

The engine automatically synchronizes positions across all brokers:

```python
# Positions are synchronized every 60 seconds
# Access aggregated position data
total_positions = engine.session_metrics['active_positions']
```

## Configuration

### Basic Configuration

```yaml
brokers:
  alpaca:
    enabled: true
    api_key: ${ALPACA_API_KEY}
    api_secret: ${ALPACA_API_SECRET}
    base_url: https://paper-api.alpaca.markets

execution:
  fast_path_enabled: true
  max_order_size: 10000
  max_position_size: 50000

risk_management:
  max_drawdown: 0.10
  max_daily_loss: 0.05
  circuit_breaker:
    enabled: true
    max_loss_threshold: 0.02
```

### Execution Modes

- **MANUAL**: All orders require manual approval
- **SEMI_AUTO**: Automated execution with human oversight
- **FULL_AUTO**: Fully automated execution
- **RESEARCH**: Analysis only, no real orders
- **EMERGENCY**: Emergency shutdown/liquidation mode

## Performance Monitoring

The execution engine tracks:

- Order execution times
- Fill rates
- Slippage
- Position P&L
- System health metrics

Access metrics:

```python
metrics = engine.session_metrics
print(f"Total orders: {metrics['total_orders_submitted']}")
print(f"Fill rate: {metrics['total_orders_filled'] / metrics['total_orders_submitted']}")
```

## Error Handling

The execution engine implements comprehensive error handling:

1. **Broker Connection Failures**: Automatic retry with exponential backoff
2. **Order Rejections**: Logged and reported through events
3. **Risk Limit Violations**: Orders blocked, alerts triggered
4. **System Errors**: Graceful degradation, emergency stop if critical

## Testing

### Unit Tests

```bash
pytest tests/unit/test_execution_engine.py
```

### Integration Tests

```bash
pytest tests/integration/test_execution_engine_integration.py -m integration
```

### Paper Trading

Always test strategies in paper trading mode first:

```python
engine = await create_execution_engine(
    trading_mode=TradingMode.PAPER
)
```

## Best Practices

1. **Always Initialize Before Trading**
   ```python
   success = await engine.initialize()
   if not success:
       raise RuntimeError("Initialization failed")
   ```

2. **Monitor System Health**
   ```python
   # Regular health checks
   status = await engine.get_comprehensive_status()
   if status['engine_status'] != 'active':
       logger.warning(f"Engine not active: {status}")
   ```

3. **Handle Events Asynchronously**
   ```python
   async def handle_position_event(event):
       # Process event without blocking
       await process_event(event)
   ```

4. **Implement Proper Shutdown**
   ```python
   # Always shutdown gracefully
   await engine.shutdown()
   ```

5. **Use Risk Controls**
   - Set appropriate position limits
   - Configure circuit breakers
   - Monitor drawdowns
   - Test emergency procedures

## Troubleshooting

### Common Issues

1. **Broker Connection Failed**
   - Check API credentials
   - Verify network connectivity
   - Check broker API status

2. **Orders Not Executing**
   - Verify trading is enabled
   - Check risk limits
   - Review order validation logs

3. **Position Mismatch**
   - Force position sync: `await engine._synchronize_positions()`
   - Check broker reconciliation logs

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('main.trading_engine').setLevel(logging.DEBUG)
```

## Next Steps

1. Review example implementations in `examples/execution_engine_example.py`
2. Configure brokers in your environment
3. Run integration tests to verify setup
4. Start with paper trading before live trading
5. Monitor system performance and adjust parameters