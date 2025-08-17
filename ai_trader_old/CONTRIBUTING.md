# Contributing to AI Trader

Thank you for your interest in contributing to the AI Trader project! This document provides guidelines and instructions for contributing to this enterprise-grade algorithmic trading platform.

## 📋 Table of Contents

- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Contribution Workflow](#contribution-workflow)
- [Architecture Guidelines](#architecture-guidelines)
- [Security Considerations](#security-considerations)
- [Documentation Standards](#documentation-standards)

## 🚀 Development Setup

### Prerequisites

- **Python 3.8+** (Python 3.11 recommended)
- **Git** for version control
- **PostgreSQL** for database (development)
- **Redis** (optional, for caching)

### Initial Setup

1. **Clone and Setup Environment**

   ```bash
   git clone <repository-url>
   cd ai_trader

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -e ".[dev]"
   ```

2. **Configure Environment**

   ```bash
   # Copy environment template
   cp .env.example .env

   # Edit .env with your API keys and database settings
   # Note: Never commit .env file - it's in .gitignore
   ```

3. **Install Pre-commit Hooks**

   ```bash
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

4. **Initialize Database** (if needed)

   ```bash
   python scripts/init_database.py
   ```

5. **Validate Setup**

   ```bash
   python ai_trader.py validate
   pytest tests/unit/test_setup.py
   ```

### Development Dependencies

The project uses modern Python development tools:

- **Code Formatting**: Black, isort
- **Linting**: flake8, pylint
- **Type Checking**: mypy
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Security**: bandit, safety
- **Pre-commit Hooks**: Automated code quality checks

## 📝 Code Standards

### Code Formatting

We use **Black** for code formatting with these settings:

- Line length: 88 characters
- Target Python versions: 3.8+

```bash
# Format code
black src tests

# Check formatting
black --check src tests
```

### Import Organization

We use **isort** with Black compatibility:

```bash
# Sort imports
isort src tests

# Check import sorting
isort --check-only src tests
```

### Type Hints

- All new code **must include type hints**
- Use `from __future__ import annotations` for forward references
- Type check with mypy: `mypy src/ai_trader`

### Code Style Guidelines

1. **Function/Method Design**
   - Keep functions focused and single-purpose
   - Maximum 20 lines per function (exceptions for complex algorithms)
   - Use descriptive names: `calculate_sharpe_ratio()` not `calc_sr()`

2. **Class Design**
   - Follow single responsibility principle
   - Use composition over inheritance where possible
   - Implement `__repr__` for debugging

3. **Error Handling**
   - Use specific exception types
   - Log errors with context
   - Never use bare `except:` clauses

4. **Documentation**
   - All public functions/classes need docstrings
   - Use Google-style docstrings
   - Include examples for complex functions

## 🧪 Testing Guidelines

### Test Organization

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component integration tests
├── performance/    # Performance and stress tests
└── fixtures/       # Test data and mocks
```

### Test Categories

**Unit Tests**: Fast, isolated, no external dependencies

```bash
pytest tests/unit/ -v
```

**Integration Tests**: Test component interactions

```bash
pytest tests/integration/ -v
```

**Performance Tests**: Validate performance requirements

```bash
pytest tests/performance/ -v
```

### Test Standards

1. **Naming Convention**
   - Test files: `test_<module_name>.py`
   - Test functions: `test_<functionality>`
   - Test classes: `Test<ClassName>`

2. **Test Structure** (Arrange-Act-Assert)

   ```python
   def test_calculate_sharpe_ratio():
       # Arrange
       returns = [0.01, 0.02, -0.005, 0.015]
       risk_free_rate = 0.02

       # Act
       result = calculate_sharpe_ratio(returns, risk_free_rate)

       # Assert
       assert result == pytest.approx(0.365, rel=1e-3)
   ```

3. **Test Coverage**
   - Minimum 80% coverage for new code
   - 100% coverage for critical trading logic
   - Use pytest-cov: `pytest --cov=ai_trader`

4. **Test Data**
   - Use fixtures for reusable test data
   - No real API calls in tests (use mocks)
   - No real trading in tests (paper mode only)

### Running Tests

```bash
# All tests
pytest

# Specific test categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# With coverage
pytest --cov=ai_trader --cov-report=html

# Parallel execution (faster)
pytest -n auto
```

## 🔄 Contribution Workflow

### Branch Strategy

1. **Main Branches**
   - `main`: Production-ready code
   - `develop`: Integration branch for features
   - `staging`: Pre-production testing

2. **Feature Branches**
   - `feature/feature-name`: New features
   - `bugfix/bug-description`: Bug fixes
   - `hotfix/critical-fix`: Critical production fixes

### Making Contributions

1. **Create Feature Branch**

   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Development Cycle**

   ```bash
   # Make your changes
   # Run tests frequently
   pytest tests/unit/

   # Pre-commit checks
   pre-commit run --all-files

   # Commit with descriptive message
   git commit -m "feat: add portfolio optimization algorithm"
   ```

3. **Quality Checks Before PR**

   ```bash
   # Run full test suite
   pytest

   # Type checking
   mypy src/ai_trader

   # Security scan
   bandit -r src/ai_trader

   # Code quality
   flake8 src tests
   ```

4. **Pull Request**
   - Clear, descriptive title
   - Detailed description of changes
   - Link to relevant issues
   - Include test results
   - Update documentation if needed

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

**Examples**:

- `feat(trading): add iceberg order execution algorithm`
- `fix(risk): correct position size calculation for futures`
- `docs(api): update trading engine documentation`

## 🏗️ Architecture Guidelines

### Project Structure

```
ai_trader/
├── src/main/           # Main source code
│   ├── app/                 # CLI applications
│   ├── config/              # Configuration management
│   ├── data_pipeline/       # Data ingestion and processing
│   ├── feature_pipeline/    # Feature engineering
│   ├── models/              # ML models and strategies
│   ├── trading_engine/      # Order execution
│   ├── risk_management/     # Risk controls
│   ├── monitoring/          # System monitoring
│   └── utils/              # Shared utilities
├── tests/                   # Test suites
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
└── deployment/              # Deployment configs
```

### Design Principles

1. **Separation of Concerns**
   - Each module has a single, well-defined responsibility
   - Minimal coupling between components
   - Clear interfaces and abstractions

2. **Async-First Design**
   - Use async/await for I/O operations
   - Support concurrent processing
   - Non-blocking data pipeline operations

3. **Configuration-Driven**
   - All behavior configurable via YAML files
   - Environment-specific configurations
   - Runtime parameter adjustment

4. **Error Resilience**
   - Graceful degradation on failures
   - Circuit breaker patterns
   - Comprehensive logging and monitoring

### Adding New Components

1. **Data Sources**
   - Implement `BaseDataSource` interface
   - Add to `data_pipeline/ingestion/clients/`
   - Include rate limiting and error handling
   - Add comprehensive tests

2. **Feature Calculators**
   - Extend `BaseCalculator` class
   - Add to appropriate feature category
   - Include validation and error handling
   - Document algorithm and parameters

3. **Trading Strategies**
   - Implement `BaseStrategy` interface
   - Include backtesting capabilities
   - Add risk management integration
   - Comprehensive performance testing

4. **Risk Controls**
   - Follow existing risk management patterns
   - Include pre-trade and real-time checks
   - Add alerting and circuit breaker logic
   - Extensive testing with edge cases

## 🔒 Security Considerations

### Credential Management

1. **Environment Variables**
   - All secrets via environment variables
   - Never hardcode credentials
   - Use `.env` file for development (not committed)

2. **API Key Security**
   - Rotate keys regularly
   - Use separate keys for different environments
   - Monitor for unauthorized usage

3. **Trading Safety**
   - Always test with paper trading first
   - Implement position limits and stop losses
   - Monitor for unusual trading patterns
   - Include emergency shutdown procedures

### Code Security

1. **Input Validation**
   - Validate all external inputs
   - Sanitize data from external sources
   - Use type hints and runtime validation

2. **Error Handling**
   - Don't expose internal details in errors
   - Log security events
   - Implement rate limiting

3. **Dependencies**
   - Regular security updates
   - Scan with `safety check`
   - Review new dependencies carefully

## 📚 Documentation Standards

### Code Documentation

1. **Docstrings** (Google Style)

   ```python
   def calculate_portfolio_var(
       positions: Dict[str, float],
       correlation_matrix: np.ndarray,
       confidence_level: float = 0.95
   ) -> float:
       """Calculate portfolio Value at Risk (VaR).

       Args:
           positions: Dict mapping symbols to position sizes
           correlation_matrix: Asset correlation matrix
           confidence_level: VaR confidence level (0.0-1.0)

       Returns:
           Portfolio VaR as positive value

       Raises:
           ValueError: If confidence_level not in valid range

       Example:
           >>> positions = {"AAPL": 1000, "GOOGL": 500}
           >>> corr_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
           >>> calculate_portfolio_var(positions, corr_matrix)
           15420.67
       """
   ```

2. **Module Documentation**
   - Include module purpose and usage
   - Document key classes and functions
   - Provide usage examples

3. **Configuration Documentation**
   - Document all configuration options
   - Provide examples and defaults
   - Explain parameter interactions

### README Updates

When adding significant features:

1. Update feature list in main README
2. Add configuration examples
3. Update usage instructions
4. Include performance considerations

## 🤝 Code Review Process

### For Contributors

1. **Self-Review Checklist**
   - [ ] All tests pass
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No security issues
   - [ ] Performance considered

2. **PR Description**
   - Clear problem statement
   - Solution approach
   - Testing performed
   - Breaking changes noted

### For Reviewers

1. **Review Focus**
   - Code correctness and logic
   - Security implications
   - Performance impact
   - Architectural consistency
   - Test coverage

2. **Feedback Guidelines**
   - Be constructive and specific
   - Suggest alternatives
   - Explain reasoning for requested changes
   - Acknowledge good practices

## 🐛 Issue Reporting

### Bug Reports

Include:

- Detailed description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant log outputs
- Configuration details (sanitized)

### Feature Requests

Include:

- Use case description
- Proposed solution approach
- Alternative solutions considered
- Impact on existing functionality
- Implementation complexity estimate

## 📞 Getting Help

1. **Documentation**: Check `/docs/` directory first
2. **Issues**: Search existing issues before creating new ones
3. **Discussions**: Use GitHub Discussions for questions
4. **Code Examples**: Check `/examples/` directory

## 🏆 Recognition

Contributors will be recognized in:

- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes for major features

---

## Final Notes

- **Quality over Speed**: Take time to do things right
- **Security First**: Always consider security implications
- **Test Everything**: Comprehensive testing prevents issues
- **Document Decisions**: Help future contributors understand your choices

Thank you for contributing to making AI Trader a world-class algorithmic trading platform!

---

*For questions about contributing, please open a GitHub issue or discussion.*
