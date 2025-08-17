# AI Trader Examples

This directory contains comprehensive examples for various AI Trader components.

## Directory Structure

- `configs/` - Configuration examples for different environments
- `indicators/` - Custom indicator implementations
- `scanners/` - Scanner configuration and usage examples
- `strategies/` - Trading strategy examples
- `backtests/` - Backtesting setup examples
- `monitoring/` - Monitoring and alerting examples

## Quick Start

### 1. Basic Configuration

See `configs/basic_config.py` for a minimal configuration example.

### 2. Custom Indicator

See `indicators/custom_rsi.py` for implementing a custom RSI indicator.

### 3. Scanner Setup

See `scanners/volume_scanner.py` for creating a custom volume scanner.

### 4. Trading Strategy

See `strategies/mean_reversion.py` for a complete mean reversion strategy.

### 5. Backtesting

See `backtests/strategy_backtest.py` for running backtests on your strategies.

## Running Examples

All examples can be run from the project root:

```bash
# Activate virtual environment
source ../venv/bin/activate

# Run an example
python examples/strategies/mean_reversion.py
```

## Environment Setup

Most examples require environment variables. Create a `.env` file:

```bash
AI_TRADER_ENV=dev
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

## Common Patterns

### Async Operations

Most AI Trader components use async/await:

```python
import asyncio
from main.data_pipeline import DataPipeline

async def main():
    pipeline = DataPipeline()
    await pipeline.initialize()
    # Your code here

asyncio.run(main())
```

### Error Handling

Always handle potential errors:

```python
try:
    result = await scanner.scan()
except ScannerError as e:
    logger.error(f"Scanner failed: {e}")
```

### Logging

Use structured logging:

```python
import structlog
logger = structlog.get_logger(__name__)

logger.info("Processing", symbol=symbol, count=count)
```

## Need Help?

- Check the main documentation in `/docs/`
- Review test files for more examples
- Submit issues on GitHub
