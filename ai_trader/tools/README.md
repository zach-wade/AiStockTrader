# AI Trader Development Tools

This directory contains utility scripts and tools for AI Trader development.

## Available Tools

### setup.sh
Initial development environment setup script.

```bash
./tools/setup.sh
```

Features:
- Checks Python version compatibility
- Creates virtual environment
- Installs all dependencies
- Sets up pre-commit hooks
- Creates necessary directories
- Generates .env template

### validate_config.py
Configuration validation tool for different environments.

```bash
# Validate development configuration
./tools/validate_config.py --env dev

# Validate production configuration with strict mode
./tools/validate_config.py --env prod --strict

# Validate a specific config file
./tools/validate_config.py --file src/main/config/environments/staging.yaml
```

Options:
- `--env`: Environment to validate (dev, staging, prod, paper)
- `--config`: Configuration name to load
- `--strict`: Treat warnings as errors
- `--file`: Validate specific file instead of full config

### lint.sh
Code quality checker that runs multiple linting tools.

```bash
./tools/lint.sh
```

Runs:
- Black (code formatting)
- Flake8 (style guide enforcement)
- isort (import sorting)
- mypy (type checking) - if installed
- pylint (code analysis) - if installed
- bandit (security checks) - if installed

### run_tests.sh
Test runner with coverage and filtering options.

```bash
# Run all tests
./tools/run_tests.sh

# Run with coverage report
./tools/run_tests.sh --coverage

# Run only unit tests
./tools/run_tests.sh --unit

# Run specific test file
./tools/run_tests.sh tests/unit/test_config.py

# Verbose output
./tools/run_tests.sh -v
```

Options:
- `-c, --coverage`: Generate coverage report
- `-v, --verbose`: Verbose output
- `--unit`: Run only unit tests
- `--integration`: Run only integration tests
- `--slow`: Include slow tests

## Quick Start

1. **First time setup**:
   ```bash
   ./tools/setup.sh
   ```

2. **Validate configuration**:
   ```bash
   ./tools/validate_config.py --env dev
   ```

3. **Check code quality**:
   ```bash
   ./tools/lint.sh
   ```

4. **Run tests**:
   ```bash
   ./tools/run_tests.sh --coverage
   ```

## Adding New Tools

When adding new tools:

1. Create script in this directory
2. Make it executable: `chmod +x tools/your_script.sh`
3. Add documentation to this README
4. Include proper error handling and help text
5. Use consistent styling with existing scripts

## Tool Requirements

Most tools require the development dependencies to be installed:

```bash
pip install -r requirements-dev.txt
```

Some tools have specific requirements:
- `lint.sh`: Requires black, flake8, isort
- `run_tests.sh`: Requires pytest, pytest-cov
- `validate_config.py`: No additional requirements

## Best Practices

- Always run from the project root directory
- Use virtual environment for consistency
- Run `lint.sh` before committing code
- Run tests before pushing changes
- Validate configuration after changes