# Coverage Testing Instructions for AI Trader

## Prerequisites

1. **Activate your virtual environment**:

   ```bash
   cd /Users/zachwade/StockMonitoring/ai_trader
   source venv/bin/activate
   ```

2. **Install coverage dependencies**:

   ```bash
   pip install pytest pytest-asyncio pytest-cov coverage[toml]
   ```

## Running Tests with Coverage

### Option 1: Using the Setup Script (Recommended)

```bash
# Run all tests with coverage
./tests/setup_and_run.sh

# Run only unit tests
./tests/setup_and_run.sh unit

# Run only integration tests
./tests/setup_and_run.sh integration

# Generate report from existing coverage data
./tests/setup_and_run.sh report
```

### Option 2: Using Quick Test Script

```bash
# Interactive test runner
python tests/quick_test.py
```

### Option 3: Direct pytest Commands

```bash
# Set PYTHONPATH first
export PYTHONPATH=/Users/zachwade/StockMonitoring/ai_trader/src:$PYTHONPATH

# Run all tests with coverage
pytest --cov=src/main --cov-report=xml --cov-report=html --cov-report=term

# Run specific test file
pytest tests/integration/test_infrastructure.py -v

# Run with specific marker
pytest -m integration --cov=src/main
```

### Option 4: Using the Python Test Runner

```bash
# Run all tests with coverage
python tests/run_all_tests.py

# Run with coverage report
python tests/coverage_report.py --run
```

## Viewing Coverage Results

1. **Terminal Report**: Shown immediately after tests run
2. **HTML Report**: Open `htmlcov/index.html` in your browser
3. **Summary Report**: Run `python tests/coverage_report.py`

## Troubleshooting

### "No module named 'main'"

Set PYTHONPATH:

```bash
export PYTHONPATH=/Users/zachwade/StockMonitoring/ai_trader/src:$PYTHONPATH
```

### "Coverage file not found"

Run tests first:

```bash
python tests/coverage_report.py --run
```

### "pytest-cov not installed"

Install it:

```bash
pip install pytest-cov coverage[toml]
```

## Expected Coverage Results

After running tests, you should see:

- Overall coverage percentage
- Package-by-package breakdown
- Uncovered lines in each file
- HTML report in `htmlcov/` directory

## Next Steps

1. Run the infrastructure test first to verify setup:

   ```bash
   pytest tests/integration/test_infrastructure.py -v
   ```

2. Run full test suite with coverage:

   ```bash
   ./tests/setup_and_run.sh
   ```

3. View the HTML coverage report:

   ```bash
   open htmlcov/index.html  # On macOS
   ```

4. Focus on improving coverage for critical components:
   - Risk management
   - Event system
   - Trading engine
   - Monitoring
