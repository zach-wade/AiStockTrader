# Integration Tests

This directory contains integration tests for the AI Trader system. Integration tests verify that different components work correctly together.

## Structure

- `data_pipeline/` - Tests for data ingestion, processing, and storage
- `scanners/` - Tests for scanner layers and universe management
- `trading/` - Tests for trading engine and order execution
- `monitoring/` - Tests for metrics, alerts, and system monitoring
- `fixtures/` - Shared test data and mock services

## Running Integration Tests

### Run all integration tests
```bash
pytest tests/integration -v
```

### Run specific test category
```bash
pytest tests/integration/data_pipeline -v
pytest tests/integration/scanners -v
```

### Run with coverage
```bash
pytest tests/integration --cov=src --cov-report=html
```

### Run in parallel
```bash
pytest tests/integration -n auto
```

## Test Categories

### Data Pipeline Tests
- End-to-end data flow from ingestion to storage
- Feature calculation and validation
- Cache behavior and performance
- Error handling and recovery

### Scanner Tests
- Multi-layer scanner coordination
- Universe management and updates
- Real-time vs batch scanning
- Performance under load

### Trading Tests
- Order lifecycle management
- Risk management integration
- Portfolio tracking
- Paper trading mode

### Monitoring Tests
- Metrics collection and aggregation
- Alert triggering and delivery
- System health checks
- Performance monitoring

## Writing Integration Tests

### Test Structure
```python
import pytest
from tests.integration.fixtures import test_config, mock_market_data

@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_pipeline_end_to_end(test_config, mock_market_data):
    """Test complete data flow from ingestion to storage."""
    # Setup
    pipeline = DataPipeline(test_config)
    
    # Execute
    result = await pipeline.process(mock_market_data)
    
    # Verify
    assert result.success
    assert len(result.processed_records) > 0
```

### Best Practices

1. **Use markers**: Mark integration tests with `@pytest.mark.integration`
2. **Mock external services**: Use fixtures to mock APIs and databases
3. **Test realistic scenarios**: Include error cases and edge conditions
4. **Clean up resources**: Ensure tests clean up after themselves
5. **Document test purpose**: Clear docstrings explaining what's being tested

## Environment Setup

Integration tests require certain environment variables:

```bash
# Test database
TEST_DB_HOST=localhost
TEST_DB_PORT=5432
TEST_DB_NAME=ai_trader_test

# Test Redis
TEST_REDIS_HOST=localhost
TEST_REDIS_PORT=6379

# Mock API keys
TEST_ALPACA_API_KEY=test_key
TEST_ALPACA_SECRET_KEY=test_secret
```

## CI/CD Integration

Integration tests run in CI with:
- PostgreSQL and Redis containers
- Mock external API responses
- Parallel execution for speed
- Coverage reporting

## Debugging Tips

### Enable detailed logging
```bash
pytest tests/integration -v -s --log-cli-level=DEBUG
```

### Run single test
```bash
pytest tests/integration/path/to/test.py::test_function_name
```

### Use debugger
```python
import pdb; pdb.set_trace()  # Add breakpoint
```