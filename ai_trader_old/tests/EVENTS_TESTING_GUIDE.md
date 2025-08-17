# Events Module Testing Guide

Complete guide for running tests on the AI Trader events module.

## 🚀 Quick Start

### 1. Setup Test Environment

```bash
# Setup test dependencies and environment
python tests/setup_test_environment.py

# Or check environment without installing
python tests/setup_test_environment.py --check-only
```

### 2. Run All Events Tests

```bash
# Using events-specific runner (recommended)
python tests/run_events_tests.py

# Using general test runner
python tests/run_all_tests.py -t all -v

# Using pytest directly
pytest tests/unit/events/ tests/integration/events/ -v
```

## 📋 Test Structure

### Test Organization

```
tests/
├── unit/events/                 # Unit tests (28 files)
│   ├── test_event_bus.py       # Core event bus
│   ├── test_event_types.py     # Event type definitions
│   ├── test_*_helpers.py       # Helper modules
│   └── ...
├── integration/events/          # Integration tests (8 files)
│   ├── test_event_coordination_integration.py
│   ├── test_error_handling_integration.py
│   ├── test_performance_integration.py
│   └── ...
└── fixtures/events/             # Test fixtures and mocks
    ├── mock_events.py
    ├── mock_database.py
    └── mock_configs.py
```

### Test Coverage

- **37 total test files** for events module
- **Complete coverage** of all 27 source files
- **Unit tests:** Individual component testing
- **Integration tests:** End-to-end workflows
- **Performance tests:** Load and scalability testing

## 🎯 Running Specific Tests

### By Test Type

```bash
# Unit tests only
python tests/run_events_tests.py -t unit -v

# Integration tests only
python tests/run_events_tests.py -t integration -v

# Performance tests
python tests/run_events_tests.py --performance
```

### By Component

```bash
# Event bus tests
python tests/run_events_tests.py -c bus -v

# Scanner-feature bridge tests
python tests/run_events_tests.py -c bridge -v

# Feature pipeline tests
python tests/run_events_tests.py -c pipeline -v

# Helper modules
python tests/run_events_tests.py -c helpers -v
```

### By Test File

```bash
# Specific test file
pytest tests/unit/events/test_event_bus.py -v

# Multiple files
pytest tests/unit/events/test_event_bus.py tests/unit/events/test_event_types.py -v

# Pattern matching
pytest tests/unit/events/test_*_helpers.py -v
```

### Using Test Markers

```bash
# Events module tests only
pytest -m events -v

# Unit tests in events module
pytest -m "unit and events" -v

# Integration tests in events module
pytest -m "integration and events" -v

# Async tests
pytest -m asyncio tests/integration/events/ -v

# Performance tests
pytest -m performance -v

# Exclude slow tests
pytest -m "not slow" tests/unit/events/ -v
```

## 📊 Coverage Analysis

### Generate Coverage Reports

```bash
# Coverage with HTML report
pytest tests/unit/events/ tests/integration/events/ \
  --cov=src/main/events \
  --cov-report=html \
  --cov-report=term-missing

# Events-specific coverage analysis
python tests/run_events_tests.py --coverage-report

# View HTML report
open htmlcov/index.html
```

### Coverage Targets

- **Overall coverage:** 90%+
- **Unit test coverage:** 95%+
- **Critical components:** 98%+
- **Integration coverage:** 85%+

## ⚡ Performance Testing

### Run Performance Tests

```bash
# Performance test suite
python tests/run_events_tests.py --performance

# Specific performance tests
pytest tests/integration/events/test_performance_integration.py -v

# With benchmarking (if pytest-benchmark installed)
pytest tests/integration/events/test_performance_integration.py --benchmark-only
```

### Performance Benchmarks

- **Throughput:** 500+ events/second
- **Latency:** <10ms average
- **Memory:** <150MB growth under load
- **Scalability:** 1000+ concurrent events

## 🔧 Advanced Testing Options

### Parallel Testing

```bash
# Run tests in parallel (faster execution)
python tests/run_events_tests.py -p

# With pytest-xdist
pytest tests/unit/events/ -n auto
```

### Debug Mode

```bash
# Debug mode (no capture, full tracebacks)
python tests/run_events_tests.py --debug

# Debug specific test
pytest tests/unit/events/test_event_bus.py::TestEventBus::test_publish_event -s --pdb
```

### Smoke Testing

```bash
# Quick smoke tests for basic functionality
python tests/run_events_tests.py --smoke

# Basic functionality check
pytest tests/unit/events/test_event_types.py tests/unit/events/test_event_bus.py -x
```

### Continuous Testing

```bash
# Watch for file changes and re-run tests
pytest-watch tests/unit/events/

# Re-run only failed tests
pytest --lf tests/unit/events/
```

## 🐛 Debugging and Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Or use the setup script
python tests/setup_test_environment.py
```

#### 2. Async Test Issues

```bash
# Install async test support
pip install pytest-asyncio

# Check pytest.ini configuration
cat tests/pytest.ini | grep asyncio
```

#### 3. Missing Dependencies

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Or use requirements
pip install -r requirements.txt
```

#### 4. Database Test Failures

```bash
# Tests use mock database by default
# Check mock database fixtures in tests/fixtures/events/mock_database.py
```

### Debugging Specific Tests

```bash
# Run single test with debugging
pytest tests/unit/events/test_event_bus.py::TestEventBus::test_publish_event -s -vv

# Add breakpoints in code
import pdb; pdb.set_trace()

# Use pytest debugger
pytest --pdb tests/unit/events/test_event_bus.py
```

### Performance Debugging

```bash
# Profile test execution
pytest tests/integration/events/ --profile

# Memory profiling (if memory-profiler installed)
pytest tests/integration/events/test_performance_integration.py --profile-svg

# Detailed timing
pytest tests/unit/events/ --durations=10
```

## 📈 Test Reports

### Generate Test Reports

```bash
# Comprehensive test report
python tests/run_events_tests.py --test-report

# HTML test report (if pytest-html installed)
pytest tests/unit/events/ --html=reports/events_unit_tests.html

# JUnit XML (for CI/CD)
pytest tests/unit/events/ --junit-xml=reports/events_junit.xml
```

### Coverage Reports

```bash
# Terminal coverage report
pytest tests/unit/events/ --cov=src/main/events --cov-report=term

# HTML coverage report
pytest tests/unit/events/ --cov=src/main/events --cov-report=html:htmlcov/events

# XML coverage report (for CI/CD)
pytest tests/unit/events/ --cov=src/main/events --cov-report=xml:coverage_events.xml
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Events Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: python tests/setup_test_environment.py
      - name: Run events tests
        run: python tests/run_events_tests.py --no-coverage
      - name: Generate coverage
        run: pytest tests/unit/events/ tests/integration/events/ --cov=src/main/events --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Docker Testing

```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN python tests/setup_test_environment.py
CMD ["python", "tests/run_events_tests.py"]
```

## 📚 Test Development Guidelines

### Writing New Tests

#### Unit Test Template

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.unit
@pytest.mark.events
class TestMyComponent:
    """Test MyComponent class."""

    @pytest.fixture
    def mock_dependency(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_async_method(self, mock_dependency):
        # Arrange
        component = MyComponent(mock_dependency)

        # Act
        result = await component.async_method()

        # Assert
        assert result is not None
        mock_dependency.method.assert_called_once()
```

#### Integration Test Template

```python
import pytest
import asyncio

@pytest.mark.integration
@pytest.mark.events
class TestComponentIntegration:
    """Test integrated component behavior."""

    @pytest.fixture
    async def integrated_system(self):
        # Setup integrated system
        system = await create_system()
        yield system
        await system.cleanup()

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, integrated_system):
        # Test complete workflow
        result = await integrated_system.process_workflow()
        assert result.success
```

### Test Best Practices

1. **Use descriptive test names**

   ```python
   def test_event_bus_publishes_events_to_all_subscribers()
   def test_scanner_bridge_handles_high_volume_alerts()
   ```

2. **Follow AAA pattern** (Arrange, Act, Assert)

   ```python
   def test_method():
       # Arrange
       setup_data()

       # Act
       result = method_under_test()

       # Assert
       assert result == expected
   ```

3. **Use appropriate fixtures**

   ```python
   @pytest.fixture
   async def event_bus():
       bus = EventBus()
       await bus.start()
       yield bus
       await bus.stop()
   ```

4. **Mock external dependencies**

   ```python
   @patch('external.service.api_call')
   def test_with_mocked_api(self, mock_api):
       mock_api.return_value = expected_response
       # Test logic here
   ```

5. **Test error conditions**

   ```python
   def test_handles_invalid_input():
       with pytest.raises(ValueError):
           component.process_invalid_input()
   ```

## 🔍 Test Markers Reference

Available pytest markers for events tests:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.events` - Events module tests
- `@pytest.mark.asyncio` - Async tests (auto-applied)
- `@pytest.mark.slow` - Slow tests (>5 seconds)
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.requires_db` - Tests needing database
- `@pytest.mark.requires_api` - Tests needing external APIs

### Using Markers

```bash
# Run specific marker
pytest -m "unit and events"

# Exclude markers
pytest -m "not slow"

# Multiple markers
pytest -m "integration and events and not performance"
```

## 🎯 Testing Checklist

Before committing code, ensure:

- [ ] All existing tests pass
- [ ] New features have unit tests
- [ ] Integration tests cover workflows
- [ ] Coverage meets minimum thresholds
- [ ] Performance tests pass benchmarks
- [ ] Error handling is tested
- [ ] Edge cases are covered
- [ ] Documentation is updated

### Quick Validation

```bash
# Run full validation suite
python tests/run_events_tests.py -v

# Check coverage
python tests/run_events_tests.py --coverage-report

# Run smoke tests
python tests/run_events_tests.py --smoke
```

## 📞 Support and Resources

### Getting Help

- **Test failures:** Check `tests/logs/pytest.log`
- **Coverage issues:** Review `htmlcov/index.html`
- **Performance problems:** Run `--performance` tests
- **Environment issues:** Run `setup_test_environment.py --check-only`

### Useful Commands Summary

```bash
# Essential commands
python tests/setup_test_environment.py          # Setup
python tests/run_events_tests.py               # Run all
python tests/run_events_tests.py -c bus -v     # Component
python tests/run_events_tests.py --smoke       # Quick check
python tests/run_events_tests.py --performance # Performance
pytest -m "unit and events" -v                 # Unit tests
pytest --cov=src/main/events --cov-report=html # Coverage
```

This comprehensive testing setup ensures the events module maintains high quality, performance, and reliability! 🎉
