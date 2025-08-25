# End-to-End Trading Workflow Integration Tests

## Overview

This test suite (`test_e2e_trading_workflows.py`) provides comprehensive end-to-end testing of the complete trading system, validating real-world trading scenarios with actual PostgreSQL database connections.

## Test Coverage

### 1. Complete Trading Lifecycle Test (`TestCompleteTradingLifecycle`)

- **Purpose**: Validates the entire trading workflow from portfolio creation to P&L calculation
- **Test Flow**:
  1. Create portfolio with initial capital
  2. Place and execute buy order
  3. Process order fill and create position
  4. Simulate market price movement
  5. Place and execute sell order to close position
  6. Calculate realized P&L
  7. Verify final portfolio state and statistics

### 2. Risk Management Workflow (`TestRiskManagementWorkflow`)

- **Purpose**: Tests risk limit enforcement and position sizing
- **Test Coverage**:
  - Position size limits validation
  - Portfolio risk percentage limits
  - Maximum positions limit enforcement
  - Position sizing based on Kelly Criterion
  - Risk-adjusted position entry

### 3. Market Simulation Workflow (`TestMarketSimulationWorkflow`)

- **Purpose**: Simulates realistic market conditions with limit order execution
- **Test Scenarios**:
  - Multiple limit orders at different price levels
  - Market price movements triggering orders
  - Buy limit orders executing when price drops
  - Sell limit orders executing when price rises
  - Proper fill price validation

### 4. Portfolio Management Workflow (`TestPortfolioManagementWorkflow`)

- **Purpose**: Tests multi-position portfolio management
- **Features Tested**:
  - Creating diversified portfolio across sectors
  - Portfolio metrics calculation (value, P&L, returns)
  - Position weighting and concentration analysis
  - Profit-taking logic (closing winning positions)
  - Portfolio statistics tracking

## Prerequisites

### Database Setup

1. PostgreSQL database must be running
2. Database named `ai_trader` must exist
3. User must have proper permissions

### Environment Variables

```bash
# Required for running integration tests
export RUN_INTEGRATION_TESTS=true

# Database configuration (optional, defaults shown)
export TEST_DATABASE_HOST=localhost
export TEST_DATABASE_PORT=5432
export TEST_DATABASE_NAME=ai_trader
export TEST_DATABASE_USER=zachwade
export TEST_DATABASE_PASSWORD=""
```

## Running the Tests

### Run All E2E Tests

```bash
export RUN_INTEGRATION_TESTS=true
pytest tests/integration/test_e2e_trading_workflows.py -v
```

### Run Specific Test Class

```bash
export RUN_INTEGRATION_TESTS=true
pytest tests/integration/test_e2e_trading_workflows.py::TestCompleteTradingLifecycle -v
```

### Run with Coverage

```bash
export RUN_INTEGRATION_TESTS=true
pytest tests/integration/test_e2e_trading_workflows.py --cov=src --cov-report=html -v
```

### Run with Output

```bash
export RUN_INTEGRATION_TESTS=true
pytest tests/integration/test_e2e_trading_workflows.py -v -s
```

## Test Data Isolation

- All test symbols are prefixed with `E2E_` (e.g., `E2E_AAPL`)
- All test portfolios are prefixed with `E2E_`
- Database is cleaned before and after each test
- Tests use isolated transactions

## Key Components Tested

### Domain Layer

- `Portfolio` entity management
- `Position` lifecycle (open, update, close)
- `Order` processing (submit, fill, cancel)
- `RiskCalculator` service
- `PositionManager` service

### Infrastructure Layer

- `PostgreSQLAdapter` database operations
- `PostgreSQLUnitOfWork` transaction management
- Repository implementations (Order, Position, Portfolio)
- `PaperBroker` for simulated trading

### Business Logic

- Order execution workflow
- Position P&L calculation
- Risk limit validation
- Portfolio value computation
- Trade statistics tracking

## Performance Considerations

- Tests use connection pooling (min: 2, max: 10 connections)
- Transactions are kept as short as possible
- Bulk operations are batched where applicable
- Indexes are assumed on key columns (symbol, portfolio_id)

## Troubleshooting

### Tests Skip with "Integration tests require database"

- Ensure `RUN_INTEGRATION_TESTS=true` is set
- Verify PostgreSQL is running
- Check database connection parameters

### Database Connection Errors

```bash
# Test database connection
psql -h localhost -U zachwade -d ai_trader -c "SELECT 1;"
```

### Clean Test Data Manually

```sql
DELETE FROM orders WHERE symbol LIKE 'E2E_%';
DELETE FROM positions WHERE symbol LIKE 'E2E_%';
DELETE FROM portfolios WHERE name LIKE 'E2E_%';
```

## Expected Test Output

When all tests pass, you should see:

```
tests/integration/test_e2e_trading_workflows.py::TestCompleteTradingLifecycle::test_full_trading_cycle PASSED
tests/integration/test_e2e_trading_workflows.py::TestRiskManagementWorkflow::test_risk_limits_enforcement PASSED
tests/integration/test_e2e_trading_workflows.py::TestMarketSimulationWorkflow::test_limit_order_execution PASSED
tests/integration/test_e2e_trading_workflows.py::TestPortfolioManagementWorkflow::test_multi_position_portfolio PASSED
```

## Continuous Integration

For CI/CD pipelines, ensure:

1. PostgreSQL service is available
2. Database is initialized with schema
3. Environment variables are set
4. Test database is separate from production

Example GitHub Actions step:

```yaml
- name: Run E2E Trading Tests
  env:
    RUN_INTEGRATION_TESTS: true
    TEST_DATABASE_HOST: localhost
    TEST_DATABASE_PORT: 5432
    TEST_DATABASE_NAME: ai_trader_test
    TEST_DATABASE_USER: postgres
    TEST_DATABASE_PASSWORD: postgres
  run: |
    pytest tests/integration/test_e2e_trading_workflows.py -v
```
