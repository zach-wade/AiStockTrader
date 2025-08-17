# Broker Implementations

This module provides broker adapters for order execution in the AI Trading System MVP.

## Components

### 1. **IBroker Interface** (`src/application/interfaces/broker.py`)

- Defines the contract all broker implementations must follow
- Key methods:
  - `submit_order()` - Submit orders for execution
  - `cancel_order()` - Cancel pending orders
  - `get_positions()` - Get current positions
  - `get_account_info()` - Get account balances and metrics
  - `is_market_open()` - Check market status

### 2. **AlpacaBroker** (`alpaca_broker.py`)

- Production-ready integration with Alpaca Markets API
- Supports both paper and live trading
- Features:
  - Real-time order execution
  - Position tracking
  - Account management
  - Rate limiting compliance
  - Comprehensive error handling

### 3. **PaperBroker** (`paper_broker.py`)

- Simulated broker for testing without real money
- Features:
  - Realistic order fills with configurable slippage
  - Commission calculation
  - Position and P&L tracking
  - Market hours simulation
  - Support for market, limit, and stop orders

### 4. **BrokerFactory** (`broker_factory.py`)

- Factory pattern for creating broker instances
- Supports multiple broker types:
  - `"alpaca"` - Alpaca Markets integration
  - `"paper"` - Paper trading simulation
  - `"backtest"` - Optimized for backtesting

## Setup

### Environment Variables

```bash
# Broker type selection
export BROKER_TYPE=paper  # Options: alpaca, paper, backtest

# Alpaca configuration
export ALPACA_API_KEY=your_api_key
export ALPACA_SECRET_KEY=your_secret_key
export ALPACA_PAPER=true  # Use paper trading

# Paper trading configuration
export PAPER_INITIAL_CAPITAL=100000
export PAPER_SLIPPAGE_PCT=0.001
export PAPER_FILL_DELAY=1
export PAPER_COMMISSION_PER_SHARE=0.01
export PAPER_MIN_COMMISSION=1.0
```

## Usage Examples

### Basic Paper Trading

```python
from decimal import Decimal
from src.infrastructure.brokers import BrokerFactory
from src.domain.entities.order import Order, OrderSide

# Create paper broker
broker = BrokerFactory.create_broker("paper")

# Set market price (in production, comes from data feed)
broker.set_market_price("AAPL", Decimal("150.00"))

# Submit a market order
order = Order.create_market_order(
    symbol="AAPL",
    quantity=Decimal("10"),
    side=OrderSide.BUY,
    reason="Entry signal"
)
order = broker.submit_order(order)

# Check positions
positions = broker.get_positions()
for position in positions:
    print(f"{position.symbol}: {position.quantity} @ ${position.average_entry_price}")

# Get account info
account = broker.get_account_info()
print(f"Cash: ${account.cash}, Equity: ${account.equity}")
```

### Alpaca Integration

```python
# Create Alpaca broker
broker = BrokerFactory.create_broker(
    broker_type="alpaca",
    paper=True,  # Use paper trading
    api_key="your_key",  # Or use environment variable
    secret_key="your_secret"
)

# Check market status
if broker.is_market_open():
    # Submit a limit order
    order = Order.create_limit_order(
        symbol="AAPL",
        quantity=Decimal("5"),
        side=OrderSide.BUY,
        limit_price=Decimal("149.50")
    )
    order = broker.submit_order(order)

    # Monitor order status
    status = broker.get_order_status(order.id)
    print(f"Order status: {status}")
```

### Using the Factory

```python
# Create from configuration
config = {
    "type": "paper",
    "initial_capital": "50000",
    "slippage_pct": "0.002",
    "auto_connect": True
}
broker = BrokerFactory.create_from_config(config)

# Or use environment variables
broker = BrokerFactory.create_broker()  # Uses BROKER_TYPE env var
```

## Error Handling

The broker implementations provide specific exceptions for different error scenarios:

- `BrokerConnectionError` - Connection issues with broker
- `InsufficientFundsError` - Not enough buying power
- `InvalidOrderError` - Order validation failed
- `OrderNotFoundError` - Order doesn't exist
- `MarketClosedError` - Market is closed for trading
- `RateLimitError` - API rate limit exceeded

Example:

```python
from src.application.interfaces.broker import (
    InsufficientFundsError,
    MarketClosedError
)

try:
    order = broker.submit_order(order)
except InsufficientFundsError as e:
    print(f"Not enough funds: {e}")
except MarketClosedError as e:
    print(f"Market is closed: {e}")
```

## Testing

The paper broker is ideal for testing strategies without risk:

```python
# Create test broker
broker = BrokerFactory.create_broker(
    broker_type="paper",
    initial_capital=Decimal("10000"),
    slippage_pct=Decimal("0"),  # No slippage for unit tests
    fill_delay_seconds=0  # Immediate fills
)

# Run your strategy
# ... strategy logic ...

# Check results
account = broker.get_account_info()
assert account.realized_pnl > 0  # Profitable strategy
```

## Integration with Order Entity

The brokers work seamlessly with the domain Order entity:

```python
# Order lifecycle
order = Order.create_market_order(...)  # Create
order = broker.submit_order(order)      # Submit to broker
order = broker.update_order(order)      # Update with latest status

# Order states are automatically managed
if order.is_active():
    broker.cancel_order(order.id)

if order.status == OrderStatus.FILLED:
    print(f"Filled at ${order.average_fill_price}")
```

## Best Practices

1. **Always use the factory** for creating brokers to ensure proper configuration
2. **Handle broker exceptions** appropriately in your trading logic
3. **Check market hours** before submitting market orders
4. **Monitor rate limits** when using real APIs
5. **Use paper trading** for development and testing
6. **Track order IDs** for status updates and cancellations

## Production Considerations

- **Credentials Security**: Never hardcode API credentials; use environment variables or secure vaults
- **Rate Limiting**: Implement proper backoff strategies for API rate limits
- **Error Recovery**: Implement retry logic with exponential backoff
- **Logging**: Enable comprehensive logging for audit trails
- **Monitoring**: Set up alerts for connection issues and failed orders
- **Testing**: Thoroughly test with paper trading before going live
