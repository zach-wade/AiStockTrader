# CI/CD Pipeline Setup & Branch Protection Guide

## Current CI/CD Status

- **MyPy Errors**: ✅ 0 errors (FIXED)
- **Smoke Tests**: ✅ 100% passing (10/10)
- **Core Domain Tests**: ✅ ~95% passing
- **Overall Test Status**: 🟡 ~50% passing (working on categorization)

## Active CI Workflows

### 1. Commit Validation (ci-commit.yml)

**Trigger**: Every push to main/develop/feature branches
**Duration**: 2-3 minutes
**Checks**:

- Black formatting
- MyPy type checking (0 errors required)
- Critical imports verification
- Core domain tests (value objects & entities)
- Smoke tests
- Paper trading validation

### 2. Pull Request Validation (ci-pull-request.yml)

**Trigger**: PRs to main/develop
**Duration**: 5-8 minutes
**Checks**:

- All commit validation checks
- Extended integration tests
- Security scanning
- Test coverage reporting

### 3. Paper Trading Tests (ci-paper-trading.yml)

**Trigger**: Scheduled (hourly) or manual
**Duration**: 10-15 minutes
**Checks**:

- Live paper trading validation
- Full integration suite
- Performance benchmarks

## Branch Protection Setup

### GitHub Settings Configuration

1. **Navigate to Repository Settings**
   - Go to: <https://github.com/zach-wade/AiStockTrader/settings>
   - Click on "Branches" in the left sidebar

2. **Add/Edit Branch Protection Rule for `main`**
   - Click "Add rule" or edit existing rule
   - Branch name pattern: `main`

3. **Required Status Checks**
   ✅ Enable "Require status checks to pass before merging"

   Add these required checks:
   - `Fast Validation (3.11)`
   - `Fast Validation (3.12)`

   ✅ Enable "Require branches to be up to date before merging"

4. **Additional Protection Settings**
   - ✅ Require pull request reviews before merging (1 review)
   - ✅ Dismiss stale pull request approvals when new commits are pushed
   - ✅ Require review from CODEOWNERS
   - ✅ Restrict who can dismiss pull request reviews
   - ✅ Include administrators in restrictions
   - ✅ Require linear history
   - ✅ Require deployments to succeed before merging (if applicable)
   - ✅ Do not allow bypassing the above settings

5. **Save Changes**
   - Click "Create" or "Save changes"

## Local Testing Commands

### Run Tests Matching CI

```bash
# Fast validation (what CI runs on every commit)
PYTHONPATH=/Users/zachwade/StockMonitoring python -m pytest \
  tests/unit/domain/value_objects/ \
  tests/unit/domain/entities/ \
  tests/smoke/ \
  -v --tb=short

# Check MyPy (must pass with 0 errors)
python -m mypy src --ignore-missing-imports --show-error-codes

# Check Black formatting
black --check --diff .
```

### Test Categories

- **Stable Tests**: Currently passing, run in CI
- **Unstable Tests**: Known failures, excluded from CI
- **Smoke Tests**: Critical path validation

## GitHub Actions Secrets Required

For paper trading tests to work, add these secrets:

1. Go to Settings → Secrets and variables → Actions
2. Add:
   - `ALPACA_API_KEY`: Your Alpaca paper trading API key
   - `ALPACA_SECRET_KEY`: Your Alpaca secret key
   - `POLYGON_API_KEY`: (Optional) Polygon API key

## Monitoring CI Status

### Check Workflow Runs

```bash
# List recent workflow runs
gh run list --limit 10

# Watch a specific run
gh run watch <run-id>

# View run details
gh run view <run-id>
```

### Badge for README

Add to your README.md:

```markdown
[![CI Status](https://github.com/zach-wade/AiStockTrader/workflows/Commit%20Validation/badge.svg)](https://github.com/zach-wade/AiStockTrader/actions)
```

## Troubleshooting

### If CI Fails

1. Check the specific job that failed in GitHub Actions
2. Run the same test locally using commands above
3. Fix the issue
4. Push fix and verify CI passes

### Common Issues

- **MyPy errors**: Run `python -m mypy src --show-error-codes` locally
- **Black formatting**: Run `black .` to auto-format
- **Test failures**: Check if test is marked as `@pytest.mark.unstable`

## Next Steps

1. ✅ Removed duplicate CI workflows
2. ✅ Fixed all MyPy errors (0 remaining)
3. ✅ Added MyPy to CI checks
4. ⏳ Continue fixing unstable tests
5. ⏳ Increase test coverage to 80%
6. ⏳ Add more integration tests

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| MyPy Errors | 0 | 0 | ✅ |
| Smoke Tests | 100% | 100% | ✅ |
| Core Tests | 95% | 100% | 🟡 |
| Overall Tests | 50% | 100% | 🔴 |
| Test Coverage | 12% | 80% | 🔴 |
| CI Duration | 3 min | < 5 min | ✅ |
