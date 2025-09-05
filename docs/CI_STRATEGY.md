# CI/CD Strategy Documentation

## Executive Summary

This document outlines the comprehensive CI/CD strategy for the AI Trading System, designed to catch breaking changes early while maintaining fast feedback loops for developers.

## Strategy Overview

### Three-Tier Testing Approach

1. **Tier 1: Commit Validation** (ci-commit.yml)
   - Runtime: 2-3 minutes
   - Triggers: Every push to any branch
   - Purpose: Immediate feedback on basic functionality

2. **Tier 2: Pull Request Validation** (ci-pull-request.yml)
   - Runtime: 5-8 minutes
   - Triggers: PR creation/update to main or develop
   - Purpose: Comprehensive testing before merge

3. **Tier 3: Paper Trading Validation** (ci-paper-trading.yml)
   - Runtime: 10-15 minutes
   - Triggers: Scheduled (market hours) or manual
   - Purpose: Real-world trading system validation

## Workflow Details

### ci-commit.yml - Fast Feedback Loop

**Purpose**: Catch obvious breaks immediately after commit

**Tests**:

- Code formatting (Black)
- Critical imports validation
- Core domain tests (value objects, entities)
- Basic smoke test

**Key Features**:

- Runs on all branches
- Fails fast on critical errors
- Provides summary in GitHub UI

### ci-pull-request.yml - Comprehensive Validation

**Purpose**: Ensure code is ready for production

**Tests**:

- All commit validation tests
- Type checking (MyPy) - non-blocking
- Security scanning (Bandit)
- Integration tests
- Paper trading simulation
- Coverage reporting

**Key Features**:

- Detailed test results
- Artifact uploads for debugging
- Performance metrics
- PR-specific summary

### ci-paper-trading.yml - Production Readiness

**Purpose**: Validate trading system with real market data

**Tests**:

- Alpaca API connection
- Real-time price fetching
- Order submission workflow
- Portfolio management
- Performance metrics

**Key Features**:

- Scheduled during market hours
- Manual trigger with parameters
- Creates issues on failure
- Comprehensive metrics reporting

## Test Organization

### Test Categories

| Category | Location | Stability | In CI |
|----------|----------|-----------|-------|
| Unit Tests - Value Objects | tests/unit/domain/value_objects/ | 98.6% | ✅ |
| Unit Tests - Entities | tests/unit/domain/entities/ | 94.7% | ✅ |
| Unit Tests - Core Services | tests/unit/domain/services/ | 100% | ✅ (partial) |
| Integration Tests | tests/integration/ | 90% | ✅ |
| Smoke Tests | tests/smoke/ | 100% | ✅ |
| Paper Broker Tests | tests/unit/infrastructure/brokers/ | 32.5% | ⚠️ (partial) |

### Known Test Exclusions

These tests are excluded from CI due to known issues:

- `test_less_than_or_equal` - Quantity comparison edge case
- `test_quantity_can_compare_with_numbers` - Type comparison issue
- `test_value_object_copy_behavior` - Copy semantics issue
- `test_round_to_tick_zero_tick_size` - Zero tick size edge case

## Branch Protection Strategy

### Main Branch

- **Required Checks**:
  - Fast Validation (Python 3.11 & 3.12)
  - PR Validation (Python 3.11 & 3.12)
- **Settings**:
  - Require up-to-date branches
  - Dismiss stale reviews
  - Require 1 approval

### Feature Branches

- **Automatic Checks**:
  - ci-commit on every push
  - ci-pull-request when PR created
- **Manual Checks**:
  - ci-paper-trading for trading features

## Performance Targets

| Workflow | Target Runtime | Actual Runtime | Status |
|----------|---------------|----------------|--------|
| ci-commit | < 3 min | 2-3 min | ✅ |
| ci-pull-request | < 10 min | 5-8 min | ✅ |
| ci-paper-trading | < 15 min | 10-15 min | ✅ |

## Monitoring & Alerts

### Success Metrics

- **CI Pass Rate**: Target > 95%
- **Mean Time to Feedback**: < 5 minutes
- **False Positive Rate**: < 5%

### Alert Triggers

1. **Scheduled Test Failures**: Creates GitHub issue
2. **Multiple Consecutive Failures**: Email notification (future)
3. **Performance Degradation**: Logged in metrics

## Migration Path

### Phase 1: Current State (December 2024)

- New workflows created and tested
- Running in parallel with legacy workflows
- Branch protection not yet updated

### Phase 2: Transition (Next Steps)

1. Push new workflows to main branch
2. Let them run once to register with GitHub
3. Update branch protection rules
4. Monitor for 1 week

### Phase 3: Cleanup

1. Remove legacy workflows (ci-quick, ci-progressive)
2. Archive old test reports
3. Update all documentation

## Best Practices

### For Developers

1. **Before Pushing**:
   - Run `black .` for formatting
   - Run core tests locally
   - Check for import errors

2. **Creating PRs**:
   - Wait for ci-commit to pass
   - Review ci-pull-request results
   - Address any MyPy warnings

3. **Trading Features**:
   - Manually trigger ci-paper-trading
   - Verify with real market data
   - Check performance metrics

### For Maintainers

1. **Weekly**:
   - Review CI metrics
   - Check for flaky tests
   - Update exclusion list if needed

2. **Monthly**:
   - Analyze test coverage trends
   - Review CI runtime performance
   - Update documentation

## Troubleshooting Guide

### Common Issues

**Issue**: Status checks not appearing

- **Solution**: Ensure workflows have run on main branch

**Issue**: Tests passing locally but failing in CI

- **Solution**: Check Python version differences, environment variables

**Issue**: Timeout in ci-paper-trading

- **Solution**: Check Alpaca API status, reduce iteration count

**Issue**: Branch protection blocking valid PRs

- **Solution**: Verify job names match exactly in settings

## Cost Optimization

### GitHub Actions Usage

- **Free Tier**: 2,000 minutes/month
- **Current Usage**: ~500 minutes/month
- **Optimization**: Use matrix strategy efficiently

### Caching Strategy

- pip packages cached per Python version
- Cache invalidated on requirements change
- Reduces install time by 60%

## Security Considerations

### Secrets Management

- API keys stored in GitHub Secrets
- Never logged or exposed in artifacts
- Rotated quarterly

### Code Scanning

- Bandit runs on every PR
- Security issues flagged but non-blocking
- Critical issues require immediate fix

## Future Enhancements

### Short Term (Q1 2025)

- Add performance benchmarking
- Implement test result trending
- Add Slack notifications

### Medium Term (Q2 2025)

- Parallel test execution
- Dynamic test selection
- Integration with monitoring tools

### Long Term (2025+)

- ML-based test prioritization
- Automated test generation
- Production deployment pipeline

## Appendix

### Useful Commands

```bash
# Run smoke tests locally
PYTHONPATH=. pytest tests/smoke/ -v

# Run integration tests
PYTHONPATH=. pytest tests/integration/ -v

# Check what CI would run
PYTHONPATH=. pytest tests/unit/domain/value_objects/ tests/unit/domain/entities/ -v

# Trigger paper trading workflow manually
gh workflow run ci-paper-trading.yml -f iterations=5 -f symbols="AAPL,MSFT"
```

### Related Documentation

- [TEST_STATUS.md](../TEST_STATUS.md) - Current test health
- [BRANCH_PROTECTION_SETUP.md](../BRANCH_PROTECTION_SETUP.md) - Branch protection configuration
- [README.md](../README.md) - Project overview

### Contact

For CI/CD questions or issues, create a GitHub issue with the `ci-cd` label.
