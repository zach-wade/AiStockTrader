# CLAUDE-TECHNICAL.md - Technical Implementation Details

This document provides detailed technical specifications for the AI Trading System.

**Last Updated**: 2025-08-09 (Phase 5 Week 5 Day 2 - Documentation Reorganized)

---

## 🔧 Language & Runtime Requirements

### Python Version
- **Minimum**: Python 3.8
- **Recommended**: Python 3.11+ (performance improvements)
- **Tested On**: Python 3.8, 3.9, 3.10, 3.11

### Key Dependencies
```
# Core Libraries
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical operations
asyncio                # Async operations (built-in)
asyncpg>=0.27.0        # PostgreSQL async driver

# ML/AI Libraries
scikit-learn>=1.3.0    # Machine learning
xgboost>=1.7.0         # Gradient boosting
lightgbm>=4.0.0        # Gradient boosting alternative

# Trading & Market Data
alpaca-py>=0.13.0      # Alpaca trading API
polygon-api-client     # Polygon.io data

# Infrastructure
redis>=4.5.0           # Caching
prometheus-client      # Metrics
omegaconf>=2.3.0       # Configuration management
pydantic>=2.0.0        # Data validation
```

---

## 🧪 Testing Philosophy & Best Practices

### Core Principle: Real Components Over Mocks

**Why This Matters**: Mocks hide real bugs. The system had a critical bug (PositionEventType.ALL) that was hidden by mock usage for months.

### Testing Hierarchy (Best to Worst)

1. **Real Components** (BEST)
   - Use actual production components
   - Catches real integration issues
   - Validates actual behavior

2. **Test Implementations** (ACCEPTABLE)
   - Minimal but real implementations (e.g., TestPositionManager)
   - Implements actual interfaces
   - No external dependencies
   - **WARNING**: Must be replaced before production!

3. **Mocks** (LAST RESORT)
   - Only for external services you can't control
   - Must accurately reflect real behavior
   - Update when interfaces change

### Current Test Implementations

| Component | Test Implementation | Production Risk | Status |
|-----------|-------------------|-----------------|---------|
| PositionManager | TestPositionManager | HIGH - Will fail in production | ISSUE-059 |

### Testing Rules

1. **Never mock to make tests pass** - Fix the actual bug
2. **Document all test implementations** - Mark with clear warnings
3. **Track test implementations** - Create issues for production replacement
4. **Integration test before production** - No test implementations in live trading

### Example: TestPositionManager Pattern

```python
class TestPositionManager:
    """
    WARNING: TEST IMPLEMENTATION ONLY!
    DO NOT USE IN PRODUCTION
    Issue: ISSUE-059
    Replace with: main.trading_engine.core.position_manager.PositionManager
    """
```

---

## 📊 Project Statistics (Updated 2025-08-09)

**Source**: Automated code analysis via scripts/code_analyzer.py

### Code Review Status (Week 3 Complete)
| Category | Files | Status | Notes |
|----------|-------|--------|-------|
| Total Files | 787 | - | 233,439 lines |
| Files Reviewed | 238 | 30.2% | Actually examined for correctness |
| Files Not Reviewed | 549 | 69.8% | Never examined, functionality unknown |
| Test Files | 158 | - | 23% test-to-code ratio |

### Code Distribution
| Category | Files | Lines | Review Status |
|----------|-------|-------|--------------|
| data_pipeline | 170 | 40,305 | 75.3% REVIEWED (128/170 files) |
| utils | 145 | 36,628 | NOT REVIEWED |
| models | 101 | 24,406 | NOT REVIEWED |
| feature_pipeline | 90 | 44,393 | 11.1% REVIEWED (10/90 files) |
| risk_management | 51 | 16,554 | 100% REVIEWED |
| Examples | 11 | 2,940 | Example implementations |

### Code Quality Metrics (Phase 4 Analysis)
| Metric | Count | Status |
|--------|-------|--------|
| Large Files (>500 lines) | 146 | 🟡 18.5% of codebase |
| Circular Imports | 0 | 🟢 Excellent! |
| Duplicate Code Blocks | 10 | 🟡 Mainly in scanners |
| Empty Modules | 0 | 🟢 None found |
| Files >1000 lines | 4 | 🔴 Dashboard files need splitting |

### Top Refactoring Targets
1. **system_dashboard_v2.py** - 1153 lines
2. **dataloader.py** - 1069 lines  
3. **models/common.py** - 1044 lines
4. **trading_dashboard_v2.py** - 1038 lines

### Module Size Analysis
#### Top 5 Largest Modules
1. **feature_pipeline**: 44,393 lines (19.2% of codebase) 🔴
2. **data_pipeline**: 40,305 lines (17.4% of codebase) 🔴
3. **utils**: 36,628 lines (15.8% of codebase) 🔴
4. **models**: 24,406 lines (10.5% of codebase)
5. **risk_management**: 16,554 lines (7.1% of codebase)

#### Empty Modules (Need Investigation)
- **core/**: 0 files ❓
- **services/**: 0 files ❓
- **migrations/**: 0 files ❓

### Known Technical Debt
- **50+ documented issues** - See [ISSUE_REGISTRY.md](ISSUE_REGISTRY.md)
- **10+ files over 500 lines** - Refactoring candidates
- **Circular import patterns** - Detected in 5+ files
- **Module size imbalance** - Top 3 modules contain 52% of code

---

## 📁 Detailed Directory Structure

### Project Root Structure
```
/Users/zachwade/StockMonitoring/ai_trader/
├── ai_trader.py              # Main CLI entry point
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Multi-container orchestration
├── requirements.txt           # Python dependencies
├── setup.py                   # Package configuration
├── CLAUDE.md                  # Main AI assistant reference
├── CLAUDE-TECHNICAL.md        # This file
├── CLAUDE-OPERATIONS.md       # Operational procedures
├── CLAUDE-SETUP.md           # Setup instructions
│
├── src/main/                  # Main source code
│   ├── __init__.py
│   ├── app/                   # Application entry points
│   ├── config/                # Configuration system
│   ├── data_pipeline/         # Data ingestion & processing
│   ├── feature_pipeline/      # Feature engineering
│   ├── models/                # ML models & strategies
│   ├── trading_engine/        # Order execution
│   └── [other modules...]
│
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test fixtures
│
├── scripts/                   # Utility scripts
│   ├── init_database.py      # Database initialization
│   ├── migrations/            # Database migrations
│   └── monitoring/            # Monitoring scripts
│
├── docs/                      # Documentation
│   ├── database/              # Database documentation
│   │   └── schema/           # SQL schema definitions
│   ├── api/                   # API documentation
│   └── architecture/          # Architecture diagrams
│
├── deployment/                # Deployment configurations
│   ├── kubernetes/            # K8s manifests
│   ├── terraform/             # Infrastructure as code
│   └── scripts/               # Deployment scripts
│
├── logs/                      # Application logs
│   ├── ai_trader.log         # Main application log
│   ├── trading/              # Trading-specific logs
│   ├── backfill/             # Data backfill logs
│   └── errors/               # Error logs
│
├── data_lake/                 # Raw data archive
│   └── raw/
│       ├── market_data/      # OHLCV data
│       ├── news/             # News articles
│       ├── fundamentals/     # Company financials
│       └── corporate_actions/ # Dividends, splits, etc.
│
└── venv/                      # Python virtual environment
```

### Key Code Locations

#### Executable Entry Points
- **Main CLI**: `ai_trader.py`
- **Commands**: `src/main/app/commands/`
  - `data_commands.py` - Data operations
  - `trading_commands.py` - Trading operations
  - `scanner_commands.py` - Market scanning
  - `universe_commands.py` - Universe management
  - `utility_commands.py` - System utilities

#### Core Business Logic
- **Data Pipeline**: `src/main/data_pipeline/`
  - Ingestion: `ingestion/clients/`
  - Processing: `processing/`
  - Storage: `storage/`
  
- **Feature Engineering**: `src/main/feature_pipeline/`
  - Calculators: `calculators/`
  - Orchestration: `feature_orchestrator.py`
  
- **Trading Logic**: `src/main/trading_engine/`
  - Execution: `core/execution_engine.py`
  - Brokers: `brokers/`
  - Algorithms: `algorithms/`

- **Orchestration**: `src/main/orchestration/`
  - ML Orchestrator: `ml_orchestrator.py`
  - Job Scheduler: `job_scheduler.py` (relocated from /scripts/scheduler/)
  - Scripts: `/scripts/scheduler/master_scheduler.py` (CLI wrapper only)

#### Database Schema Locations
- **Main Schemas**: `docs/database/schema/`
  - `companies_table.sql`
  - `market_data_tables.sql`
  - `scanner_qualifications_table.sql`
  
- **Migrations**: `scripts/migrations/`
  - `add_layer_column.sql`
  - `create_partitions.sql`
  
- **Initialization**: `deployment/sql/init.sql`

---

## 🐳 Docker Architecture

### Container Services

```yaml
# docker-compose.yml structure
services:
  postgres:       # PostgreSQL database
    ports: 5432
    container: aitrader-db
    
  redis:          # Redis cache
    ports: 6379
    container: aitrader-cache
    
  grafana:        # Monitoring dashboard
    ports: 3000
    container: aitrader-grafana
    
  prometheus:     # Metrics collection
    ports: 9090
    container: aitrader-prometheus
    
  app:           # Main application
    ports: 8000
    container: aitrader-app
```

### Volume Mounts
```
postgres_data    → /var/lib/postgresql/data
redis_data       → /data
grafana_data     → /var/lib/grafana
prometheus_data  → /prometheus
app_logs         → /app/logs
data_lake        → /app/data_lake
```

### Network Configuration
- **Network Name**: `aitrader-network`
- **Driver**: bridge
- **Internal DNS**: Service names resolve to container IPs

---

## 🌐 Service Architecture

### Service Ports & Endpoints

| Service | Container Name | Internal Port | External Port | Protocol |
|---------|---------------|---------------|---------------|----------|
| PostgreSQL | aitrader-db | 5432 | 5432 | TCP |
| Redis | aitrader-cache | 6379 | 6379 | TCP |
| Grafana | aitrader-grafana | 3000 | 3000 | HTTP |
| Prometheus | aitrader-prometheus | 9090 | 9090 | HTTP |
| Main App | aitrader-app | 8000 | 8000 | HTTP |
| WebSocket | aitrader-app | 8001 | 8001 | WS |

### Health Check Endpoints
- **App Health**: `http://localhost:8000/health`
- **Metrics**: `http://localhost:8000/metrics`
- **Database**: `pg_isready -h localhost -p 5432`
- **Redis**: `redis-cli ping`

---

## 💻 Development Tools & Preferences

### Version Control
```bash
# Git is available and required
git --version  # Should be 2.0+

# Preferred commit message format
<type>(<scope>): <subject>

# Types: feat, fix, docs, style, refactor, test, chore
# Example: feat(scanner): add momentum scanner for layer 2

# Multi-line format:
feat(trading): implement VWAP execution algorithm

- Add VWAP calculator
- Integrate with order manager
- Add unit tests
```

### Command Line Tools
```bash
# Available tools (in order of preference)
grep      # Preferred for simple searches
rg        # Ripgrep for complex patterns (if installed)
find      # File searching
awk       # Text processing
sed       # Stream editing
jq        # JSON processing

# Python tools
python    # Python interpreter
pip       # Package manager
pytest    # Test runner
black     # Code formatter
ruff      # Linter
```

### Database Tools
```bash
# PostgreSQL client
psql -h localhost -U ai_trader -d ai_trader

# Redis client
redis-cli -h localhost

# Database migrations
python scripts/run_migrations.py
```

---

## 🎨 Coding Style & Conventions

### Python Style Guide
```python
# PEP 8 with modifications
MAX_LINE_LENGTH = 120  # Extended from 79
INDENT = 4 spaces      # No tabs

# Naming conventions
module_name            # snake_case
ClassName             # PascalCase
function_name         # snake_case
CONSTANT_NAME         # UPPER_SNAKE_CASE
_private_method       # Leading underscore
__internal_method     # Double underscore (name mangling)

# Type hints required for all public functions
def calculate_returns(prices: pd.DataFrame, 
                      period: int = 20) -> pd.Series:
    """Calculate period returns.
    
    Args:
        prices: DataFrame with OHLCV data
        period: Lookback period in days
        
    Returns:
        Series of calculated returns
    """
    pass

# Async function naming
async def fetch_data():  # Prefix with async
    await some_operation()

# Import organization (isort style)
# 1. Standard library
import os
import sys
from datetime import datetime

# 2. Third-party libraries
import pandas as pd
import numpy as np

# 3. Local application imports
from main.config import get_config
from main.utils import logger
```

### SQL Style Guide
```sql
-- Keywords in UPPERCASE
SELECT symbol, close_price, volume
FROM market_data
WHERE symbol = $1
  AND timestamp >= $2
ORDER BY timestamp DESC
LIMIT 100;

-- Table names: snake_case
-- Column names: snake_case
-- Indexes: idx_table_column
-- Constraints: table_column_constraint_type
```

### Configuration Style
```yaml
# YAML files use snake_case for keys
database:
  host: localhost
  port: 5432
  connection_pool:
    min_size: 10
    max_size: 50
    
# Environment variables use UPPER_SNAKE_CASE
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
```

---

## 🔍 Testing Conventions

### Test Structure
```python
# Test file naming
test_<module_name>.py  # Unit tests
integration_test_<feature>.py  # Integration tests

# Test class and method naming
class TestMarketDataRepository:
    def test_fetch_returns_dataframe(self):
        """Test that fetch returns a DataFrame."""
        pass
        
    def test_fetch_handles_empty_result(self):
        """Test fetch with no data returns empty DataFrame."""
        pass

# Fixture naming
@pytest.fixture
def mock_database():
    """Provide mock database for testing."""
    return Mock(spec=IAsyncDatabase)
```

### Test Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=main --cov-report=html

# Run specific test file
pytest tests/unit/test_market_data.py

# Run with verbose output
pytest -v

# Run only marked tests
pytest -m "not slow"
```

---

## 📊 Performance Characteristics

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 100GB+ for data lake
- **Network**: Low latency connection for trading

### Performance Targets
| Operation | Target | Actual |
|-----------|--------|--------|
| Feature Calculation | >1M features/sec | 9M features/sec |
| Database Writes | >10K records/sec | 15K records/sec |
| Order Latency | <100ms | 50-80ms |
| Scanner Cycle | <5 seconds | 3-4 seconds |
| Backfill Rate | >10K records/sec | 12K records/sec |

### Resource Usage
- **Database Connections**: Pool of 10-50
- **Redis Memory**: 2GB max
- **Python Memory**: 4-8GB typical
- **CPU Usage**: 40-60% during trading hours

---

## 🏛️ Validation Architecture (Phase 5 Week 4 Findings)

### Multi-Stage Validation Pipeline
The system implements a sophisticated three-stage validation pipeline:

1. **INGEST Stage**: Raw data validation at entry point
   - OHLCV data integrity checks
   - Required field validation
   - Source-specific field mapping

2. **POST_ETL Stage**: Post-transformation validation
   - DataFrame structure validation
   - Data continuity checks
   - Trading hours validation

3. **FEATURE_READY Stage**: Pre-feature calculation validation
   - Feature completeness checks
   - Distribution validation
   - Correlation analysis
   - Feature drift detection

### Key Architectural Patterns Found

#### Clean Interface Implementation
- All validators implement `IValidator` base interface
- Specialized interfaces: `IFeatureValidator`, `IMarketDataValidator`
- Stage-specific validators with proper routing
- Dependency injection throughout

#### Statistical Drift Detection
```python
# Feature drift detection with statistical analysis
drift_score = abs(current_mean - reference_mean) / reference_std
if drift_score > threshold:
    # Drift detected
```

#### No SQL Injection Risk
- All validators work exclusively with DataFrames
- No direct SQL query construction in validation layer
- Data access through repository pattern only

### Validation Metrics
- **NaN ratio checking**: Configurable thresholds per data type
- **Infinite value detection**: Prevents invalid calculations
- **Constant feature detection**: Identifies non-informative features
- **Quality scoring**: Comprehensive quality metrics with penalties

---

## 🔐 Security Considerations

### Known Security Vulnerabilities (Found in Phase 5 Review)

#### Critical Issues Requiring Immediate Attention (8 Total)
1. **SQL Injection - Partition Manager** (ISSUE-144)
   - File: data_pipeline/services/storage/partition_manager.py
   - Pattern: Direct f-string interpolation in CREATE/DROP TABLE statements
   - Impact: CRITICAL - Arbitrary SQL execution possible
   - Fix: Use identifier validation before query construction

2. **Code Execution via eval()** (ISSUE-103)
   - File: data_pipeline/validation/rules/rule_executor.py
   - Impact: Complete system compromise via malicious rule expressions
   - Fix: Replace eval() with safe expression parser (asteval/sympy)

3. **YAML Deserialization Attack** (ISSUE-104)
   - File: data_pipeline/validation/rules/rule_parser.py
   - Impact: Code execution via malicious YAML files
   - Fix: Use yaml.safe_load() with validation

4. **Path Traversal Vulnerability** (ISSUE-095)
   - File: data_pipeline/validation/config/validation_profile_manager.py
   - Impact: Arbitrary file system access
   - Fix: Path validation and sandboxing

5. **Additional SQL Injection Risks** (ISSUE-076, 061, 063, 066, 069, 078)
   - Files affected: Multiple ingestion and repository files
   - Pattern: Direct f-string interpolation of table names
   - Fix: Use validate_table_name() from sql_validator.py

#### Secure Components (Found in Week 2-3)
✅ **Bulk Loading System** - Uses PostgreSQL COPY commands with proper parameterization
✅ **Metrics & Dashboard System** - Clean configuration management, no security issues
✅ **Validation Metrics Exporters** - Professional implementation with proper validation
✅ **SQL Validator Module** - EXCELLENT validation with comprehensive whitelisting
✅ **Circuit Breaker Pattern** - Well-implemented resilience patterns throughout

### Sensitive File Locations
```
.env                    # Environment variables (git-ignored)
config/secrets.yaml     # API keys (git-ignored)
~/.alpaca/             # Alpaca credentials
~/.polygon/            # Polygon credentials
```

### Security Patterns
```python
# Never hardcode secrets
API_KEY = os.getenv('ALPACA_API_KEY')  # ✅ Correct
API_KEY = 'PKY1234567890'              # ❌ Never do this

# Use parameterized queries
query = "SELECT * FROM users WHERE id = $1"  # ✅ Safe
query = f"SELECT * FROM users WHERE id = {user_id}"  # ❌ SQL injection risk

# Validate table names before use
from main.utils.security.sql_security import validate_table_name
table = validate_table_name(table_name)  # ✅ Safe
query = f"SELECT * FROM {table}"

# Validate all inputs
from main.utils.security.sql_security import validate_table_name
table = validate_table_name(user_input)  # ✅ Validated
```

---

## 🏆 Excellent Architectural Patterns (Found in Week 3)

### Outstanding Implementations
1. **SQL Validator Module** (sql_validator.py)
   - Comprehensive whitelisting of safe table names
   - Dangerous keyword detection
   - Pattern validation for identifiers
   - Zero SQL injection vulnerabilities

2. **Circuit Breaker Implementation**
   - Exponential backoff with jitter
   - Success threshold recovery
   - Per-service isolation
   - Monitoring integration

3. **Storage Router** (storage_router.py) 
   - Intelligent hot/cold data routing
   - Fallback tier support
   - Performance estimation
   - Streaming for large datasets

4. **Layer-Aware Processing**
   - Different strategies per layer (0-3)
   - Automatic resource allocation
   - Efficient data retention
   - Progressive enhancement

5. **Service Locator Pattern** (service_container.py)
   - Clean dependency injection
   - Lazy initialization
   - Type-safe service retrieval
   - Easy testing/mocking

### Design Patterns Properly Used
- **Factory Pattern**: All repositories use factory creation
- **Repository Pattern**: Clean data access abstraction
- **Command Pattern**: ETL processors with configurable pipelines
- **Strategy Pattern**: Multiple validators with common interface
- **Observer Pattern**: Event-driven ingestion system

---

## 📝 File Formats & Conventions

### Data Storage Formats
- **Market Data**: Parquet files (compressed, columnar)
- **Configuration**: YAML with environment variable interpolation
- **Logs**: JSON lines format for structured logging
- **Cache**: Redis with JSON serialization
- **Archive**: Parquet with metadata sidecar files

### Naming Conventions
```
# Parquet files
market_data_YYYYMMDD_HHMMSS.parquet

# Log files
ai_trader_YYYYMMDD.log

# Backup files
backup_YYYYMMDD_HHMMSS.sql

# Cache keys
cache:market_data:AAPL:1h:20240101
```

---

*Last Updated: 2025-08-09 (Phase 5 Week 4 Complete - Historical & Validation Subsystems)*  
*System Status: TESTS PASSING (10/10 components) - 11 CRITICAL SECURITY FIXES REQUIRED*
*Code Review Progress: 261/787 files (33.2%) - 526 files never examined*
*Version: 2.4*