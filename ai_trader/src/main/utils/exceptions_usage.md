# Exception Usage Guide

This guide explains how to use the standardized exception classes in AI Trader.

## Exception Hierarchy

```
AITraderError (base)
├── ConfigurationError
│   ├── EnvironmentError
│   └── ValidationError
├── DataPipelineError
│   ├── DataFetchError
│   ├── DataProcessingError
│   └── FeatureCalculationError
├── MarketDataError
│   └── APIError
│       ├── RateLimitError
│       └── AuthenticationError
├── TradingError
│   ├── OrderError
│   ├── PositionError
│   └── RiskLimitError
├── ScannerError
│   ├── ScanTimeoutError
│   └── FilterError
├── StrategyError
│   ├── SignalGenerationError
│   └── BacktestError
├── DatabaseError
│   ├── ConnectionError
│   ├── QueryError
│   └── IntegrityError
├── RepositoryError
│   ├── RecordNotFoundError
│   └── DuplicateRecordError
├── CacheError
│   ├── CacheConnectionError
│   └── CacheKeyError
├── MonitoringError
│   ├── MetricsError
│   └── AlertingError
└── ResilienceError
    ├── CircuitOpenError
    └── RetryExhaustedError
```

## Usage Examples

### Basic Exception Usage

```python
from main.utils.exceptions import DataFetchError, APIError

# Simple exception
raise DataFetchError("Failed to fetch data from provider")

# Exception with context
raise DataFetchError(
    "Failed to fetch historical data",
    error_code="HIST_DATA_FETCH_FAIL",
    context={
        'symbol': 'AAPL',
        'timeframe': '1Day',
        'start_date': '2024-01-01'
    }
)
```

### API Error Handling

```python
from main.utils.exceptions import APIError, RateLimitError, AuthenticationError

# API error with status code
raise APIError(
    "API request failed",
    status_code=500,
    api_name="alpaca",
    context={'endpoint': '/v2/bars'}
)

# Rate limit error
raise RateLimitError(
    "Rate limit exceeded",
    status_code=429,
    api_name="polygon",
    context={'retry_after': 60}
)

# Authentication error
raise AuthenticationError(
    "Invalid API credentials",
    status_code=401,
    api_name="alpaca"
)
```

### Trading Errors

```python
from main.utils.exceptions import OrderError, RiskLimitError

# Order error
raise OrderError(
    "Order rejected by broker",
    order_type="LIMIT",
    symbol="AAPL",
    quantity=100,
    context={'reason': 'insufficient_funds'}
)

# Risk limit error
raise RiskLimitError(
    "Position size exceeds risk limit",
    limit_type="position_size",
    current_value=0.25,
    limit_value=0.20,
    context={'symbol': 'TSLA'}
)
```

### Repository Errors

```python
from main.utils.exceptions import RecordNotFoundError, DuplicateRecordError

# Record not found
raise RecordNotFoundError(
    "Company not found",
    entity_type="Company",
    entity_id="AAPL"
)

# Duplicate record
raise DuplicateRecordError(
    "Company already exists",
    context={'symbol': 'AAPL'}
)
```

### Circuit Breaker Errors

```python
from main.utils.exceptions import CircuitOpenError, RetryExhaustedError

# Circuit open
raise CircuitOpenError(
    "Circuit breaker is open",
    service_name="alpaca_api",
    retry_after=30.0
)

# Retry exhausted
raise RetryExhaustedError(
    "All retry attempts failed",
    attempts=3,
    last_error=original_exception
)
```

## Exception Handler Decorator

```python
from main.utils.exceptions import handle_exceptions, DataPipelineError

# Basic usage
@handle_exceptions(default_return=None)
def fetch_data(symbol: str):
    # Function that might raise exceptions
    pass

# Custom error class and behavior
@handle_exceptions(
    default_return=[],
    log_errors=True,
    reraise=False,
    error_class=DataPipelineError
)
async def process_data(data: list):
    # Only catches DataPipelineError and subclasses
    pass

# Reraise after logging
@handle_exceptions(reraise=True)
def critical_operation():
    # Logs error and then reraises it
    pass
```

## Error Formatting

```python
from main.utils.exceptions import format_error_message, DataFetchError

error = DataFetchError(
    "Connection timeout",
    error_code="CONN_TIMEOUT",
    context={'host': 'api.example.com', 'timeout': 30}
)

# Basic formatting
message = format_error_message(error)
# Output: "CONN_TIMEOUT: Connection timeout | Context: host=api.example.com, timeout=30"

# Without context
message = format_error_message(error, include_context=False)
# Output: "CONN_TIMEOUT: Connection timeout"

# With traceback
message = format_error_message(error, include_traceback=True)
# Includes full traceback information
```

## Best Practices

1. **Use specific exceptions**: Choose the most specific exception class for your use case.

2. **Include context**: Always provide relevant context information.

3. **Use error codes**: Define consistent error codes for programmatic handling.

4. **Handle at appropriate level**: Catch exceptions at the level where you can handle them meaningfully.

5. **Log and monitor**: Use the exception handler decorator for consistent logging.

6. **Document exceptions**: Document which exceptions your functions might raise.

```python
async def fetch_market_data(symbol: str) -> pd.DataFrame:
    """
    Fetch market data for a symbol.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        DataFrame with market data
        
    Raises:
        DataFetchError: If data fetching fails
        RateLimitError: If rate limit is exceeded
        AuthenticationError: If authentication fails
    """
    pass
```

## Migration Guide

To migrate existing code to use standardized exceptions:

1. Replace generic exceptions:
   ```python
   # Old
   raise Exception("Failed to fetch data")
   
   # New
   raise DataFetchError("Failed to fetch data")
   ```

2. Add context to exceptions:
   ```python
   # Old
   raise ValueError(f"Invalid symbol: {symbol}")
   
   # New
   raise InputValidationError(
       "Invalid symbol format",
       field_name="symbol",
       invalid_value=symbol
   )
   ```

3. Use specific error types:
   ```python
   # Old
   if not record:
       return None
   
   # New
   if not record:
       raise RecordNotFoundError(
           "Record not found",
           entity_type="Trade",
           entity_id=trade_id
       )
   ```