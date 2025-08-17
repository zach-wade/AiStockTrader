# AI Trader Test Suite

Comprehensive test suite for the AI Trader system with unit tests, integration tests, and performance benchmarks.

## Test Structure

```
tests/
├── unit/                   # Unit tests for individual components
├── integration/            # Integration tests for system workflows
│   ├── events/            # Event system integration tests
│   ├── risk/              # Risk management integration tests
│   ├── monitoring/        # Monitoring & alerting tests
│   ├── orchestration/     # System orchestration tests
│   ├── backtesting/       # Backtesting engine tests
│   └── models/            # ML model pipeline tests
├── performance/           # Performance and stress tests
├── fixtures/              # Shared test fixtures and mock data
├── run_all_tests.py       # Main test runner
├── coverage_report.py     # Coverage analysis tool
└── pytest.ini            # Pytest configuration
```

## Running Tests

### Run All Tests with Coverage

```bash
# From project root
python tests/run_all_tests.py

# Or using pytest directly
pytest --cov=src/main --cov-report=html
```

### Run Specific Test Types

```bash
# Unit tests only
python tests/run_all_tests.py -t unit

# Integration tests only
python tests/run_all_tests.py -t integration

# Tests with specific marker
python tests/run_all_tests.py -m slow
python tests/run_all_tests.py -m requires_db
```

### Run Specific Test Files

```bash
# Run single test file
python tests/run_all_tests.py -s tests/unit/test_order_manager.py

# Run with pytest
pytest tests/integration/events/test_event_bus_integration.py -v
```

### Re-run Failed Tests

```bash
# Re-run only failed tests from last run
python tests/run_all_tests.py -f

# Or with pytest
pytest --lf
```

## Coverage Analysis

### Generate Coverage Report

```bash
# Run tests and generate coverage
python tests/coverage_report.py --run

# Generate report from existing coverage data
python tests/coverage_report.py

# Generate GitHub Actions summary
python tests/coverage_report.py --github-summary
```

### View Coverage Report

After running tests with coverage, open `htmlcov/index.html` in a browser.

## Test Markers

Tests are marked with various markers for selective execution:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.requires_db` - Tests requiring database
- `@pytest.mark.requires_redis` - Tests requiring Redis
- `@pytest.mark.requires_api` - Tests requiring external APIs
- `@pytest.mark.asyncio` - Async tests

## Writing Tests

### Unit Test Example

```python
import pytest
from main.trading_engine.core.order_manager import OrderManager

class TestOrderManager:
    @pytest.mark.unit
    def test_create_order(self):
        manager = OrderManager()
        order = manager.create_order(
            symbol="AAPL",
            quantity=100,
            side="buy"
        )
        assert order.symbol == "AAPL"
        assert order.quantity == 100
```

### Integration Test Example

```python
import pytest
import asyncio
from main.events.event_bus import EventBus

class TestEventBusIntegration:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_event_flow(self, event_bus):
        # Test complete event flow
        events_received = []

        async def handler(event):
            events_received.append(event)

        event_bus.subscribe("order_placed", handler)
        await event_bus.publish({"type": "order_placed", "symbol": "AAPL"})

        await asyncio.sleep(0.1)
        assert len(events_received) == 1
```

### Using Fixtures

```python
import pytest

@pytest.fixture
def mock_broker():
    """Create mock broker for testing."""
    broker = MagicMock()
    broker.submit_order = AsyncMock(return_value="ORDER123")
    return broker

def test_with_broker(mock_broker):
    # Use the fixture
    result = mock_broker.submit_order("AAPL", 100)
    assert result == "ORDER123"
```

## Test Data

Test data and fixtures are located in `tests/fixtures/`:

- `market_data.py` - Mock market data
- `portfolio_data.py` - Sample portfolios
- `order_data.py` - Test orders and fills

## Performance Testing

Run performance benchmarks:

```bash
# Run performance tests
pytest tests/performance/ -v

# Run with profiling
pytest tests/performance/ --profile
```

## Continuous Integration

Tests are automatically run on:

- Every push to main branch
- Every pull request
- Nightly for full test suite including slow tests

## Coverage Requirements

- Minimum overall coverage: 80%
- Critical components (risk, execution): 90%
- New code must include tests

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   ```

2. **Database Tests Failing**

   ```bash
   # Ensure test database is running
   docker-compose up -d test-db
   ```

3. **Async Test Warnings**

   ```bash
   # Install pytest-asyncio
   pip install pytest-asyncio
   ```

## Test Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Names**: Test names should describe what they test
3. **Arrange-Act-Assert**: Follow AAA pattern
4. **Mock External Dependencies**: Don't hit real APIs in tests
5. **Test Edge Cases**: Include boundary conditions
6. **Performance**: Keep tests fast (< 1 second each)

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Update test documentation
