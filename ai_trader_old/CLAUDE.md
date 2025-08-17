# CLAUDE.md - AI Assistant Guidelines & Project Reference

This document provides comprehensive context and guidelines for AI assistants working on the AI Trading System codebase.

## 📚 Documentation Structure

This is the main reference document. For detailed information, see:

- **[CLAUDE-TECHNICAL.md](CLAUDE-TECHNICAL.md)** - Technical implementation details, language specs, Docker architecture
- **[CLAUDE-OPERATIONS.md](CLAUDE-OPERATIONS.md)** - Operational procedures, troubleshooting, monitoring
- **[CLAUDE-SETUP.md](CLAUDE-SETUP.md)** - Initial setup, configuration, development environment

### 📝 Documentation Update Protocol (`/update-claude-doc` command)

When updating CLAUDE documentation, the AI assistant MUST:

1. **Update ALL CLAUDE files**:
   - CLAUDE.md (main reference)
   - CLAUDE-TECHNICAL.md (if technical changes)
   - CLAUDE-OPERATIONS.md (if operational changes)
   - CLAUDE-SETUP.md (if setup/repository changes)

2. **Check for placeholders** in all files:
   - Repository URLs (replace `[username]` with actual username)
   - File paths (update to actual paths)
   - Version numbers and dates
   - Any TODO or PLACEHOLDER markers

3. **Synchronize information** across files:
   - Repository information must match
   - Version numbers should be consistent
   - Cross-references should be valid

4. **Update these sections** when relevant:
   - Current project status
   - Known issues and priorities
   - Recent changes and commits
   - Configuration examples with actual values
   - Setup instructions with real repository URLs

## 🚀 Repository Information

- **GitHub Repository**: [https://github.com/zach-wade/AiStockTrader](https://github.com/zach-wade/AiStockTrader)
- **Local Path**: `/Users/zachwade/StockMonitoring`
- **Main Branch**: `main`
- **Last Updated**: 2025-08-09 (Phase 5 Week 5 Day 2 - Documentation Reorganized)

## 📋 Current Project Status (August 2025)

### ⚠️ IMPORTANT: Tests Pass But Code Not Fully Reviewed (2025-08-09)

- **Test Status**: 10/10 components pass initialization tests
- **Code Review Status**: 295 of 787 files reviewed (37.5%)
- **Reality Check**: We know it doesn't crash, NOT that it works correctly
- **Production Ready**: NO - 12 CRITICAL security vulnerabilities including eval() code execution
- **Current Phase**: Phase 5 Week 5 Day 2 - feature_pipeline module in progress (27.8% complete)

### What We Actually Know vs Don't Know

**VERIFIED** ✅:

- System initializes without crashes
- Database tables exist and connect
- Modules can be imported
- 501 model files exist on disk
- 26 scanner files exist
- No circular imports detected

**NOT VERIFIED** ⚠️:

- Whether data pipeline processes data correctly
- Whether features are calculated accurately
- Whether models make sensible predictions
- Whether trading logic is sound
- Whether risk management actually works
- Performance under real load
- Security vulnerabilities
- Memory leaks or inefficiencies

### Phase 5: Deep Code Review (Week 5 Day 2 IN PROGRESS)

**Completed Reviews (295 files total)**:

- ✅ Week 1 Day 1: Storage/Repositories (25 files) - Found SQL injection risks, placeholder implementations
- ✅ Week 1 Day 2: Ingestion (17 files) - Found critical SQL injection, undefined functions
- ✅ Week 1 Day 3: Orchestration & Historical (15 files) - Found more SQL injection, deadlock risks
- ✅ Week 1 Day 4: Validation System (20 files) - Found CRITICAL code execution vulnerabilities
- ✅ Week 1 Day 5: Processing/Services (25 files) - Found runtime errors, security issues
- ✅ Week 2 Batch 1: Core Infrastructure (20 files) - Found CRITICAL SQL injection in partition manager
- ✅ Week 2 Batch 2: Bulk Loaders & Validation Metrics (21 files) - EXCELLENT security, no critical issues
- ✅ Week 3 Batch 3: Core Module Files (5 files) - EXCELLENT architecture, no issues
- ✅ Week 3 Batch 4: Base Classes & Storage Router (5 files) - One minor MD5 issue
- ✅ Week 3 Batch 5: Storage & Database Components (5 files) - 2 CRITICAL SQL injection issues
- ✅ Week 3 Batch 6: Processing Validators (5 files) - 1 undefined variable issue
- ✅ Week 3 Batch 7: Processing Features (5 files) - Excellent patterns found
- ✅ Week 3 Batch 8: Processing Services (5 files) - Good text processing implementation
- ✅ Week 4 Batch 1: Validation Core Components (4 files) - Config injection risks, excellent architecture
- ✅ Week 4 Batch 2: Validation Validators (5 files) - No SQL injection, minor issues only
- ✅ Week 4 Batch 3: Historical Module Part 1 (5 files) - CRITICAL SQL injection in data_existence_checker
- ✅ Week 4 Batch 4: Historical Module Part 2 (4 files) - Cache management issues only
- ✅ Week 4 Batch 5: Validation Quality & Coverage (5 files) - MD5 usage, global state issues
- ✅ Week 4 Batch 6: Validation Rules Engine (6 files) - ⚠️ CRITICAL eval() vulnerability CONFIRMED
- ✅ Week 4 Batch 7-8: Validation Config (3 files) - Path traversal risks
- ✅ Week 5 Day 1: feature_pipeline Core (20 files) - No critical issues
- ✅ Week 5 Day 2 Batch 5: Technical Facade & Cross-Sectional (5 files) - No security issues

**Modules Complete**:

- ✅ data_pipeline: 170/170 files (100%) - 35+ issues, 7 critical
- 🔄 feature_pipeline: 25/90 files (27.8%) - 24 issues, 0 critical

### Week 5 Plan: feature_pipeline Module

**Target**: 90 files, ~44K lines of critical feature calculation code
**Timeline**: 5 days, 18 batches of 5 files each
**Priority**: HIGH - Features directly impact trading accuracy

**Schedule**:

- Day 1: Core infrastructure (20 files)
- Day 2: Technical & Statistical calculators (20 files)
- Day 3: Correlation & News analysis (20 files)
- Day 4: Options & Integration (20 files)
- Day 5: Remaining components (10 files)

**Focus**: eval() usage, performance bottlenecks, placeholder implementations, calculation accuracy

**Still Unreviewed (492 files, 62.5%)**:

1. **feature_pipeline/** - 65 files remaining (25/90 complete)
2. **utils/** - 145 files, 36K lines
3. **models/** - 101 files, 24K lines
4. **trading_engine/** - 33 files, 13K lines
5. **monitoring/** - 36 files, 10K lines
6. **All other modules** - 112 files

**Critical Issues Found So Far** (216 total, 12 CRITICAL):

- ISSUE-162: SQL injection in data_existence_checker.py (CRITICAL - Week 4 Batch 3) - IMMEDIATE FIX REQUIRED
- ISSUE-144: SQL injection in partition manager (CRITICAL - Week 2 Batch 1) - IMMEDIATE FIX REQUIRED
- ISSUE-171: eval() code execution in rule_executor.py (CRITICAL - Week 4 Batch 6) - CONFIRMED & IMMEDIATE FIX REQUIRED
- ISSUE-104: YAML deserialization FALSE POSITIVE - yaml.safe_load() used correctly
- ISSUE-095: Path traversal vulnerability (CRITICAL - Day 4)
- ISSUE-096: JSON deserialization attack (CRITICAL - Day 4)
- ISSUE-078: SQL injection in retention_manager.py (CRITICAL - Day 3)
- ISSUE-076: SQL injection in market_data_split.py (CRITICAL - Day 2)
- ISSUE-071: Technical analyzer returns RANDOM data (CRITICAL - Day 1)
- ISSUE-153: SQL injection in update() method (CRITICAL - Week 3 Batch 5) - IMMEDIATE FIX REQUIRED
- ISSUE-154: SQL injection in delete() method (CRITICAL - Week 3 Batch 5) - IMMEDIATE FIX REQUIRED
- ISSUE-059: TestPositionManager in production path
- Week 4 issues: Config injection, undefined variables, cache issues (9 new issues)
- Multiple validation system vulnerabilities (Day 4: 32 new issues)
- Multiple SQL injection risks in 10+ files
- 146 files >500 lines need refactoring

### Documentation Reorganization (2025-08-09)

The issue tracking has been reorganized for better navigation:

- **ISSUE_REGISTRY.md** - Now serves as index/summary only
- **ISSUES_data_pipeline.md** - All data_pipeline issues (35+)
- **ISSUES_feature_pipeline.md** - All feature_pipeline issues (24)
- Module-specific files will be created as reviews progress

### Critical Issues Status (2025-08-09)

- ✅ **FIXED**: Risk management - All abstract methods implemented
- ✅ **FIXED**: Feature calculation - FeatureStoreV2 base_path parameter added
- ✅ **FIXED**: Import errors - All calculator imports working
- ✅ **IMPROVED**: Test environment - DATABASE_URL made optional
- 🚨 **NEW**: TestPositionManager in use (ISSUE-059) - MUST replace before production
- **KNOWN**: Health metrics module not implemented (ISSUE-005) - documented limitation

## 📊 Project Audit & Review (August 2025)

### Comprehensive Review Initiative

A systematic audit of the entire codebase is underway to ensure code quality, identify issues, and plan improvements.

### Key Audit Documents

- **[ISSUE_REGISTRY.md](ISSUE_REGISTRY.md)** - Index/summary of all issues (216 total)
- **[ISSUES_data_pipeline.md](ISSUES_data_pipeline.md)** - data_pipeline module issues (35+)
- **[ISSUES_feature_pipeline.md](ISSUES_feature_pipeline.md)** - feature_pipeline module issues (24)
- **[PROJECT_AUDIT.md](PROJECT_AUDIT.md)** - Complete audit methodology and findings
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed code metrics and structure analysis
- **[COMPONENT_MAPPING.md](COMPONENT_MAPPING.md)** - Expected vs actual component mapping
- **[review_progress.json](review_progress.json)** - Real-time tracking of review progress

### Current Audit Status (Updated: 2025-08-09)

- **Files in Project**: 787 Python files, 233,439 lines
- **Files Actually Reviewed**: 295 files (37.5% coverage)
- **Files Never Reviewed**: 492 files (62.5%)
- **Issues Documented**: 216 (12 critical, ~35 high, ~45 medium, ~124 low)
- **Large Files**: 146 files >500 lines need refactoring
- **Progress**: Phase 5 Week 5 Day 2 - feature_pipeline 27.8% complete
- **System Status**: 🔴 **NOT PRODUCTION READY** - eval() vulnerability confirmed

### Latest Test Results (2025-08-09)

- **Components Tested**: 10 major subsystems
- **Currently Passing**: 10/10 components (100% pass rate)
  - ✅ Configuration, Database, Data Ingestion, Trading Engine
  - ✅ Features, Models, Risk Management, Scanners
  - ✅ Monitoring, Job Scheduler
- **Outstanding Warnings**: 4 minor non-blocking items
  - ⚠️ DATABASE_URL optional (by design)
  - ⚠️ Polygon API needs manual verification
  - ⚠️ Feature calculation needs live test
  - ⚠️ Health metrics not implemented (ISSUE-005)
- **System Status**: 100% FUNCTIONAL - All systems operational

### Priority Issues (P0 - Blocking System Function)

1. ✅ **FIXED**: Features database table - migration successful
2. ✅ **FIXED**: Data ingestion - AsyncCircuitBreaker parameter mapping resolved
3. ✅ **FIXED**: ResilienceStrategies - Complete parameter and config extraction fix
4. ✅ **FIXED**: Risk circuit breakers - Now working with proper config
5. ✅ **FIXED**: FeatureStoreV2 - base_path parameter implemented
6. ✅ **FIXED**: Risk management - All abstract methods implemented
7. ✅ **FIXED**: Import system - All calculator imports working
8. ✅ **FIXED**: Test environment - DATABASE_URL made optional
9. 🚨 **NEW P0**: TestPositionManager usage - Production blocker (ISSUE-059)

**WARNING: System uses test implementations - replace before production!**

---

## 🚨 PRODUCTION READINESS WARNING

### Current System Status: NOT PRODUCTION READY

The system currently uses test implementations that MUST be replaced before live trading:

1. **TestPositionManager (ISSUE-059)**
   - Location: `test_helpers/test_position_manager.py`
   - Usage: LiveRiskMonitor position tracking
   - Risk: Will fail in production with real trades
   - Fix: Replace with full PositionManager implementation

### Pre-Production Checklist

- [ ] Replace TestPositionManager with real implementation
- [ ] Verify Polygon API with live market data
- [ ] Implement health metrics module
- [ ] Run full integration tests
- [ ] Test with paper trading for 1+ week
- [ ] Validate all risk management systems
- [ ] Review and remove all test implementations

**DO NOT DEPLOY TO PRODUCTION UNTIL ALL ITEMS ARE COMPLETE**

---

## 📊 Project Overview

### System Purpose

The AI Trading System is an **enterprise-grade algorithmic trading platform** that combines:

- Advanced machine learning models for market prediction
- Real-time data processing from multiple sources
- Automated trade execution with risk management
- Comprehensive monitoring and alerting

### Core Capabilities

- **Data Pipeline**: Ingests 15+ data sources (market, news, fundamentals, corporate actions)
- **Feature Engineering**: 16 specialized calculators generating 227+ features
- **ML Models**: Ensemble learning with multiple strategies (momentum, mean reversion, etc.)
- **Trading Engine**: Automated execution via Alpaca with algorithms (TWAP, VWAP, Iceberg)
- **Risk Management**: Real-time position monitoring, VaR, stress testing, circuit breakers
- **Performance**: 9+ million features/second, 250K+ rows in <3 seconds

### Key External Dependencies

- **Polygon.io**: Primary market data provider (historical & real-time)
- **Alpaca**: Broker for trade execution and portfolio management
- **PostgreSQL**: Time-series database with partitioned tables
- **Data Lake**: Local/S3 storage for raw data archival

---

## 🏗️ System Architecture

### High-Level Data Flow

```
Market Data Sources → Ingestion → Processing → Feature Engineering → ML Models → Trading Signals → Risk Checks → Order Execution
        ↓                  ↓           ↓              ↓                   ↓              ↓              ↓              ↓
   Data Archive      Validation   Database      Feature Store       Model Registry   Monitoring    Risk Limits   Portfolio
```

### Layer-Based Architecture

The system uses a 4-layer hierarchy for symbol management:

| Layer | Name | Symbols | Retention | Data Types | Purpose |
|-------|------|---------|-----------|------------|---------|
| **0** | Universe | ~10,000 | 30 days | 1hour, 1day | Basic monitoring of all tradable symbols |
| **1** | Liquid | ~2,000 | 60 days | 15min, 1hour, 1day | High liquidity symbols with enhanced data |
| **2** | Catalyst | ~500 | 90 days | 5min, 15min, 1hour, 1day | Active catalyst detection and monitoring |
| **3** | Active | ~50 | 180 days | tick, 1min, 5min, all | Real-time trading with full data |

### Module Organization

```
src/main/
├── app/                  # CLI commands and application entry points
├── backtesting/          # Backtesting engine and analysis
├── config/               # Configuration management (YAML + Pydantic)
├── data_pipeline/        # Data ingestion, processing, storage
├── events/               # Event-driven architecture components
├── feature_pipeline/     # Feature calculation and orchestration
├── interfaces/           # Abstract interfaces and protocols
├── models/               # ML models and trading strategies
├── monitoring/           # Metrics, alerts, dashboards
├── risk_management/      # Pre/post trade risk controls
├── scanners/             # Market scanning and symbol selection
├── trading_engine/       # Order execution and portfolio management
├── universe/             # Universe management and layer transitions
└── utils/                # Shared utilities and helpers
```

---

## 🔧 Key Systems & Components

### Data Pipeline

**Purpose**: Reliable data ingestion and storage

**Components**:

- **Ingestion Clients**: Polygon (market, news, fundamentals), Alpaca (assets)
- **Bulk Loaders**: Efficient batch loading to PostgreSQL
- **Archive System**: Raw data storage in Parquet format
- **Validation**: Multi-stage data quality checks

**Key Files**:

- `data_pipeline/ingestion/clients/` - API clients
- `data_pipeline/storage/bulk_loaders/` - Database loaders
- `data_pipeline/storage/archive/` - Archive system

### Feature Pipeline

**Purpose**: Calculate technical indicators and ML features

**Calculator Categories**:

1. **Technical** (momentum, trend, volatility, volume)
2. **Statistical** (entropy, fractals, time series)
3. **Risk** (VaR, drawdown, stress tests)
4. **Correlation** (beta, lead-lag, PCA)
5. **News** (sentiment, volume, credibility)
6. **Options** (Greeks, IV, flow analysis)

**Key Files**:

- `feature_pipeline/calculators/` - Feature calculators
- `feature_pipeline/feature_store.py` - Feature storage
- `feature_pipeline/feature_orchestrator.py` - Orchestration

### Trading Engine

**Purpose**: Signal generation and order execution

**Components**:

- **Strategies**: ML-based, technical, sentiment, pairs trading
- **Execution Algorithms**: TWAP, VWAP, Iceberg
- **Brokers**: Alpaca (live), Paper (simulation), Backtest
- **Portfolio Manager**: Position tracking and P&L

**Key Files**:

- `trading_engine/core/execution_engine.py` - Main engine
- `trading_engine/brokers/` - Broker implementations
- `models/strategies/` - Trading strategies

### Risk Management

**Purpose**: Protect capital and ensure compliance

**Pre-Trade Checks**:

- Position limits
- Exposure limits
- Liquidity checks

**Real-Time Monitoring**:

- Drawdown control
- Stop-loss management
- Anomaly detection
- Circuit breakers

**Key Files**:

- `risk_management/pre_trade/` - Pre-trade validations
- `risk_management/real_time/` - Live monitoring
- `risk_management/metrics/` - Risk metrics

---

## 💻 Common Tasks & Commands

### CLI Command Reference

```bash
# Check system status
python ai_trader.py status

# Backfill historical data
python ai_trader.py backfill --symbols AAPL,MSFT --days 30 --stage market_data
python ai_trader.py backfill --layer layer1 --days 90  # Backfill all Layer 1 symbols

# Calculate features
python ai_trader.py features --symbols AAPL --lookback 30

# Train ML models
python ai_trader.py train --model-type xgboost --lookback-days 90

# Run backtesting
python ai_trader.py backtest --strategy ml_momentum --start-date 2024-01-01

# Start trading
python ai_trader.py trade --mode paper  # Paper trading
python ai_trader.py trade --mode live   # Live trading (use with caution!)

# Universe management
python ai_trader.py universe scan       # Run layer scanners
python ai_trader.py universe promote    # Promote symbols between layers

# Emergency shutdown
python ai_trader.py shutdown --force
```

### Common Workflows

#### 1. Setting Up for Trading

```bash
# 1. Configure environment
export ALPACA_API_KEY="your_key"
export POLYGON_API_KEY="your_key"

# 2. Initialize database
python scripts/init_database.py

# 3. Backfill initial data
python ai_trader.py backfill --layer layer1 --days 30

# 4. Calculate features
python ai_trader.py features --layer layer1 --lookback 30

# 5. Train models
python ai_trader.py train --lookback-days 30

# 6. Start paper trading
python ai_trader.py trade --mode paper
```

#### 2. Adding a New Symbol to Active Trading

```bash
# 1. Check symbol eligibility
python ai_trader.py universe check --symbol NVDA

# 2. Backfill symbol data
python ai_trader.py backfill --symbols NVDA --days 90 --stage all

# 3. Calculate features
python ai_trader.py features --symbols NVDA --lookback 30

# 4. Add to layer
python ai_trader.py universe add --symbol NVDA --layer layer2
```

---

## 📁 File Organization Guide

### Import Patterns

```python
# ✅ CORRECT - Import from interfaces
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories import IRepository

# ✅ CORRECT - Import from public APIs
from main.data_pipeline.storage.repositories import get_repository_factory
from main.utils.monitoring import MetricType

# ❌ WRONG - Direct concrete imports
from main.data_pipeline.storage.repositories.company_repository import CompanyRepository

# ❌ WRONG - Internal module imports
from main.utils.monitoring.metrics_utils.buffer import MetricsBuffer
```

### Module Structure Pattern

```
module_name/
├── __init__.py           # Public API exports
├── interfaces.py         # Module interfaces (if any)
├── core/                 # Core implementation
│   ├── __init__.py
│   └── implementation.py
├── services/             # Service layer
├── models/               # Data models
├── utils/                # Module-specific utilities
└── tests/                # Module tests
```

### Repository Pattern

All repositories must:

1. Extend from `BaseRepository` or implement `IRepository`
2. Be created via `RepositoryFactory`
3. Use `IAsyncDatabase` for database operations
4. Never be instantiated directly

---

## 🔧 Configuration System

### Configuration Files

```
config/yaml/
├── layer_definitions.yaml      # Symbol layer definitions
├── app_context_config.yaml     # Application settings
├── event_config.yaml           # Event system config
├── environments/               # Environment overrides
│   ├── development.yaml
│   ├── paper.yaml
│   └── production.yaml
├── services/                   # Service configurations
│   ├── backfill.yaml
│   ├── ml_trading.yaml
│   └── scanners.yaml
└── defaults/                   # Default values
    ├── data.yaml
    ├── risk.yaml
    └── trading.yaml
```

### Usage

```python
from main.config import get_config_manager

# Load configuration
manager = get_config_manager()
config = manager.load_config("layer_definitions")

# Access values
layer1_retention = config.layers.layer_1.retention.hot_storage_days

# With validation
manager = get_config_manager(use_validation=True)
config = manager.load_config("unified_config")  # Returns AITraderConfig
```

---

## 🗄️ Database Schema

### Key Tables

| Table | Purpose | Partitioning |
|-------|---------|--------------|
| `companies` | Symbol metadata and layer assignments | None |
| `market_data_1h` | Hourly/daily OHLCV data | Weekly by timestamp |
| `market_data_5m` | 5-minute bars | Weekly by timestamp |
| `market_data_1m` | 1-minute bars | Daily by timestamp |
| `market_data_tick` | Tick data | Hourly by timestamp |
| `news_data` | News articles and sentiment | Monthly by published_at |
| `financials_data` | Company financials | Quarterly by period_end |
| `features` | Calculated features | Daily by timestamp |
| `scanner_alerts` | Scanner detections | Daily by created_at |

### Partition Management

- Automatic partition creation for time-series tables
- Retention policies by layer (30-180 days)
- Archive to data lake for long-term storage

---

## 🐛 Troubleshooting Guide

### Common Issues

#### 1. Import Errors

**Error**: `ModuleNotFoundError: No module named 'main'`
**Solution**: Ensure you're running from the project root with `python ai_trader.py`

#### 2. Database Connection Issues

**Error**: `asyncpg.exceptions.InvalidPasswordError`
**Solution**: Check DATABASE_URL environment variable and PostgreSQL credentials

#### 3. API Rate Limits

**Error**: `HTTP 429: Too Many Requests`
**Solution**: Reduce concurrent symbols or adjust rate limits in config

#### 4. Memory Issues

**Error**: `MemoryError` during feature calculation
**Solution**: Process symbols in smaller batches, increase system memory

#### 5. Missing Partitions

**Error**: `no partition of relation 'market_data_1h' found`
**Solution**: Run partition creation script or wait for automatic creation

### Debug Commands

```bash
# Check system logs
tail -f logs/ai_trader.log

# Database diagnostics
python scripts/check_database.py

# API connectivity test
python scripts/test_apis.py

# Memory profiling
python ai_trader.py status --memory-profile

# Configuration validation
python scripts/validate_config.py
```

---

## 📋 Quick Reference

### Environment Variables

```bash
# Required
ALPACA_API_KEY          # Alpaca API key
ALPACA_API_SECRET       # Alpaca secret
POLYGON_API_KEY         # Polygon.io API key
DATABASE_URL            # PostgreSQL connection string

# Optional
ALPACA_BASE_URL         # Paper/live trading URL
DATA_LAKE_PATH          # Archive storage path
LOG_LEVEL               # DEBUG, INFO, WARNING, ERROR
ENVIRONMENT             # development, paper, production
```

### Key Metrics & Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Drawdown | >10% | Reduce position sizes |
| Sharpe Ratio | <0.5 | Review strategy |
| API Errors | >5/min | Circuit breaker activation |
| Memory Usage | >80% | Trigger garbage collection |
| Query Time | >1s | Optimize query/add index |

### Performance Targets

- Feature calculation: >1M features/second
- Backfill: >10K records/second
- Order latency: <100ms
- Scanner cycle: <5 seconds

---

## 🔴 MANDATORY `/update-claude-doc` Checklist

When the user runs `/update-claude-doc`, you MUST:

- [ ] Check ALL CLAUDE*.md files exist (CLAUDE.md, CLAUDE-TECHNICAL.md, CLAUDE-OPERATIONS.md, CLAUDE-SETUP.md)
- [ ] Update repository URLs in all files (especially CLAUDE-SETUP.md)
- [ ] Update version numbers and dates consistently across all files
- [ ] Replace ALL placeholders like `[username]`, `[path]`, `TODO`, etc.
- [ ] Update current project status section with recent changes
- [ ] Verify all cross-references between files are valid
- [ ] Commit all updated files together with descriptive message
- [ ] Report which files were updated and what changed

## 🔴 MANDATORY Code Review Checklist (2025-08-08)

### 🔴 CRITICAL: Follow This Checklist for EVERY Code Review

**IMPORTANT**: This checklist must be followed for ALL code reviews, regardless of:

- How the review request is phrased
- Current conversation context
- Apparent scope of changes
- Time constraints

#### Review Triggers

- **After ANY code changes**: Run Levels 1-2 minimum
- **After refactoring**: Run Levels 1-5 minimum
- **Before commits**: Run Levels 1-6 (complete)
- **On user request for review**: ALWAYS run Levels 1-6 complete

#### Level 1: Syntax & Compilation ✓ ALWAYS CHECK

- [ ] No undefined variables (check for NameError risks)
- [ ] No missing imports or incorrect import paths
- [ ] No Python syntax errors
- [ ] Type hints present for function parameters and returns
- [ ] No missing `await` for async function calls
- [ ] No mismatched brackets, quotes, or indentation

#### Level 2: Runtime Safety ✓ ALWAYS CHECK

- [ ] No bare `except:` clauses (must specify exception type)
- [ ] Null/None checks before accessing attributes/methods
- [ ] No functions returning None on error (raise exceptions instead)
- [ ] Proper error messages with context (not generic messages)
- [ ] Resource cleanup (close files, database connections, etc.)
- [ ] No infinite loops or recursion without exit conditions

#### Level 3: Design Patterns ✓ ALWAYS CHECK

- [ ] Using interfaces (IRepository, IDatabase) not concrete classes
- [ ] Factory pattern for all repository/service creation
- [ ] No direct instantiation of repositories/services
- [ ] Dependency injection preferred over singleton patterns
- [ ] Config dataclasses have default values for all fields
- [ ] Following Interface → Implementation → Public API pattern

#### Level 4: Code Quality ✓ ALWAYS CHECK

- [ ] No duplicate code blocks (>10 lines)
- [ ] No star imports (`from module import *`)
- [ ] No SQL injection vulnerabilities (use parameterized queries)
- [ ] No hardcoded secrets, API keys, or credentials
- [ ] Functions < 50 lines (or has justification comment)
- [ ] Files < 500 lines (or properly modularized)
- [ ] No commented-out code blocks
- [ ] Clear variable and function names

#### Level 5: Architecture ✓ ALWAYS CHECK

- [ ] Proper module organization (/interfaces/, /impl/, /public API/)
- [ ] No circular dependencies between modules
- [ ] Clear separation of concerns (SRP)
- [ ] Following three-layer architecture
- [ ] Consistent naming conventions across codebase
- [ ] No business logic in data access layer
- [ ] No data access in presentation layer

#### Level 6: Testing & Documentation ✓ ALWAYS CHECK

- [ ] Critical paths have unit tests
- [ ] Mocking uses interfaces, not concrete implementations
- [ ] Docstrings for all public classes and methods
- [ ] Complex algorithms have explanatory comments
- [ ] README updated if public API changed
- [ ] No TODO comments without issue numbers
- [ ] Integration tests for cross-module interactions

### Review Output Template

```markdown
=== CODE REVIEW RESULTS ===

**Review Scope**: [List files/directories reviewed]
**Review Date**: [YYYY-MM-DD HH:MM]
**Checklist Version**: 1.0

## 🔴 Critical Issues (Must Fix Immediately)
- [Issue Description]: [file_path:line_number]
  - Impact: [Runtime error/Data loss/Security risk]
  - Fix: [Specific solution]

## 🟡 Major Issues (Should Fix Before Deploy)
- [Issue Description]: [file_path:line_number]
  - Impact: [Performance/Maintainability/Best practices]
  - Fix: [Specific solution]

## 🔵 Minor Issues (Consider Fixing)
- [Issue Description]: [file_path:line_number]
  - Impact: [Code clarity/Future maintenance]
  - Fix: [Specific solution]

## Checklist Summary
✅ Level 1: Syntax & Compilation - [Passed/X issues]
✅ Level 2: Runtime Safety - [Passed/X issues]
⚠️ Level 3: Design Patterns - [X issues found]
✅ Level 4: Code Quality - [Passed/X issues]
❌ Level 5: Architecture - [X issues found]
⚠️ Level 6: Testing & Documentation - [X issues found]

## Metrics
- Files Reviewed: [X]
- Total Issues: [Y]
- Distribution: Critical: [A], Major: [B], Minor: [C]
- Estimated Fix Time: [Z hours]
- Code Coverage: [X%] (if available)

## Recommendations
1. [Priority 1 action]
2. [Priority 2 action]
3. [Priority 3 action]
```

### Common Anti-Patterns to Flag

1. **Bare Except Clauses**

   ```python
   # ❌ WRONG
   try:
       result = operation()
   except:
       return None

   # ✅ CORRECT
   try:
       result = operation()
   except SpecificException as e:
       logger.error(f"Operation failed: {e}")
       raise
   ```

2. **Direct Instantiation**

   ```python
   # ❌ WRONG
   repo = CompanyRepository(db)

   # ✅ CORRECT
   factory = get_repository_factory()
   repo = factory.create_company_repository(db)
   ```

3. **Missing Interface Usage**

   ```python
   # ❌ WRONG
   def process(db: AsyncDatabaseAdapter):

   # ✅ CORRECT
   def process(db: IAsyncDatabase):
   ```

4. **SQL Injection Risk**

   ```python
   # ❌ WRONG
   query = f"SELECT * FROM {table_name}"

   # ✅ CORRECT
   from main.utils.security.sql_security import validate_table_name
   table = validate_table_name(table_name)
   query = f"SELECT * FROM {table}"
   ```

### Review Priority Matrix

| Issue Type | Severity | Fix Priority | Example |
|------------|----------|--------------|---------|
| NameError/Import Error | Critical | Immediate | Undefined variable |
| SQL Injection | Critical | Immediate | Unvalidated table names |
| Bare Except | Major | Before Deploy | Hiding exceptions |
| Missing Factory | Major | Before Deploy | Direct instantiation |
| No Type Hints | Minor | Next Sprint | Function parameters |
| Long Functions | Minor | Refactoring | >50 lines |

---

## MANDATORY Architecture Guidelines (2025-08-08)

### 🔴 CRITICAL: Follow These Principles for ALL Code

#### 1. **Three-Layer Architecture (REQUIRED)**

```
src/main/
├── interfaces/           # Layer 1: Pure abstractions (NO implementation)
│   ├── database.py      # IAsyncDatabase, ISyncDatabase interfaces
│   ├── repository.py    # IRepository base interface
│   └── events.py        # IEventBus, IEventPublisher interfaces
├── */module_name/       # Layer 2: Implementations (business logic)
│   ├── concrete.py      # Implements interfaces
│   └── internal.py      # Internal helpers (not exported)
└── */__init__.py        # Layer 3: Public APIs (what others import)
```

#### 2. **Interface-First Development (ALWAYS)**

**✅ CORRECT Pattern:**

```python
# Step 1: Define interface
from abc import ABC, abstractmethod

class IRepository(ABC):
    @abstractmethod
    async def get(self, id: str) -> Optional[Dict]:
        pass

# Step 2: Implement interface
class ConcreteRepository(IRepository):
    async def get(self, id: str) -> Optional[Dict]:
        return await self.db.fetch_one(...)

# Step 3: Use factory for instantiation
class RepositoryFactory:
    def create_repository(self, db: IAsyncDatabase) -> IRepository:
        return ConcreteRepository(db)  # Let exceptions bubble up!
```

**❌ NEVER Do This:**

```python
# WRONG: Direct instantiation
repository = ConcreteRepository(db)  # NO!

# WRONG: Returning None on error
try:
    return ConcreteRepository(db)
except:
    return None  # NEVER return None - raise exceptions!

# WRONG: Importing concrete classes
from main.data_pipeline.storage.repositories.company_repository import CompanyRepository  # NO!
```

#### 3. **Factory Pattern (REQUIRED for all repositories/services)**

```python
# ALWAYS use factories
from main.data_pipeline.storage.repositories import get_repository_factory

factory = get_repository_factory()
company_repo = factory.create_company_repository(db_adapter)  # Returns ICompanyRepository

# NEVER direct instantiation
company_repo = CompanyRepository(db_adapter)  # WRONG!
```

#### 4. **Configuration with Dataclasses (REQUIRED)**

```python
from dataclasses import dataclass

@dataclass
class RepositoryConfig:
    """All fields MUST have defaults."""
    max_workers: int = 4
    cache_ttl: int = 300
    batch_size: int = 1000

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'RepositoryConfig':
        """Safe construction from dict."""
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered)
```

#### 5. **Public API Pattern (REQUIRED for all modules)**

```python
# /src/main/utils/monitoring/__init__.py
"""
Public API for monitoring module.
External code should ONLY import from here.
"""

# Re-export public types
from .metrics import MetricType, MetricRecord
from .metrics_utils import MetricsBuffer

__all__ = ['MetricType', 'MetricRecord', 'MetricsBuffer']  # Explicit exports

# Internal modules should NOT be imported directly
```

#### 6. **Error Handling**

**✅ CORRECT:**

```python
def create_repository(db: IAsyncDatabase) -> IRepository:
    """Let exceptions bubble up with context."""
    if not db:
        raise ValueError("Database adapter is required")
    return ConcreteRepository(db)  # Exceptions bubble up
```

**❌ WRONG:**

```python
def create_repository(db):
    try:
        return ConcreteRepository(db)
    except:
        return None  # NEVER swallow exceptions with None!
```

#### 7. **Testing Pattern**

```python
# ALWAYS mock interfaces
mock_db = Mock(spec=IAsyncDatabase)
mock_repo = Mock(spec=IRepository)

# NEVER mock concrete classes
mock_db = Mock(spec=PostgreSQLAdapter)  # WRONG!
```

### 📋 Refactoring Checklist

When refactoring existing code or adding new features:

- [ ] Create interface first (in `/interfaces/`)
- [ ] Implement interface (in module directory)
- [ ] Create factory method (no None returns!)
- [ ] Add to public API (`__init__.py`)
- [ ] Use dataclass for configuration (with defaults)
- [ ] Update all imports to use public API
- [ ] Add type hints using interface types
- [ ] Write tests mocking interfaces
- [ ] Document in module's README

### 🚫 Common Mistakes to AVOID

1. **Returning None instead of raising exceptions**
2. **Direct instantiation of concrete classes**
3. **Importing from internal modules instead of public API**
4. **Missing default values in config classes**
5. **Not using type hints with interface types**
6. **Try/except blocks that hide errors**
7. **Circular imports** (use interfaces to break cycles)
8. **Not using factories for object creation**

---

## 📝 Example: Adding a New Repository

```python
# 1. Create interface (/src/main/interfaces/repositories/order.py)
from abc import ABC, abstractmethod

class IOrderRepository(ABC):
    @abstractmethod
    async def get_order(self, id: str) -> Optional[Order]:
        pass

# 2. Implement (/src/main/trading/repositories/order_repository.py)
from main.interfaces.repositories.order import IOrderRepository

class OrderRepository(IOrderRepository):
    def __init__(self, db: IAsyncDatabase):
        self.db = db

    async def get_order(self, id: str) -> Optional[Order]:
        return await self.db.fetch_one("SELECT * FROM orders WHERE id = $1", id)

# 3. Add to factory (/src/main/trading/repositories/repository_factory.py)
def create_order_repository(self, db: IAsyncDatabase) -> IOrderRepository:
    return OrderRepository(db)

# 4. Export from public API (/src/main/trading/repositories/__init__.py)
from .repository_factory import get_repository_factory
__all__ = ['get_repository_factory']

# 5. Use in code
from main.trading.repositories import get_repository_factory

factory = get_repository_factory()
order_repo = factory.create_order_repository(db)  # Type: IOrderRepository
```

---

## 📚 Additional Resources

### Internal Documentation

#### Core Documentation

- `README.md` - Project overview and quick start
- `PRODUCTION_USAGE_GUIDE.md` - Production deployment guide
- `config/docs/CONFIG_ARCHITECTURE.md` - Configuration system details
- `monitoring/README.md` - Monitoring setup guide

#### Audit & Review Documents (August 2025)

- `PROJECT_AUDIT.md` - Comprehensive project audit and methodology
- `ISSUE_REGISTRY.md` - Complete issue tracking with priorities (53+ issues)
- `PROJECT_STRUCTURE.md` - Detailed code structure and metrics
- `CRITICAL_PATH_TEST.md` - Phase 2 test results showing system failures
- `review_progress.json` - Live review progress tracking
- `test_trading_flow.py` - End-to-end system test script
- `test_results.json` - Latest test execution results
- `code_inventory_v2.sh` - Automated code analysis script

### Key Integration Points

- Alpaca API: <https://alpaca.markets/docs/>
- Polygon.io API: <https://polygon.io/docs/>
- PostgreSQL: <https://www.postgresql.org/docs/>

### Performance Optimization

- Use batch operations for database writes
- Implement caching for frequently accessed data
- Process symbols in parallel with semaphore control
- Archive old data to reduce database size

---

## 📖 See Also

### Specialized Documentation

- **[CLAUDE-TECHNICAL.md](CLAUDE-TECHNICAL.md)** - Language versions, Docker setup, coding conventions, tool preferences
- **[CLAUDE-OPERATIONS.md](CLAUDE-OPERATIONS.md)** - Service management, log analysis, troubleshooting procedures
- **[CLAUDE-SETUP.md](CLAUDE-SETUP.md)** - Repository setup, environment configuration, first-run guide

### Quick Navigation

- **Technical Questions** → See CLAUDE-TECHNICAL.md
- **Operational Issues** → See CLAUDE-OPERATIONS.md
- **Setup & Configuration** → See CLAUDE-SETUP.md
- **Code Standards** → This document (scroll up)

---

*Last Updated: 2025-08-09 (Phase 5 Week 5 Day 2 - Documentation Reorganized)*
*Version: 5.4*
*GitHub: <https://github.com/zach-wade/AiStockTrader>*
*System Status: NOT PRODUCTION READY - 37.5% of code reviewed, 12 CRITICAL vulnerabilities*
*Production Ready: NO - eval() code execution vulnerability CONFIRMED*
