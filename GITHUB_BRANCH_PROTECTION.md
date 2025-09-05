# GitHub Branch Protection Rules

## Overview

This document outlines the recommended branch protection rules for the AI Trading System repository to ensure code quality and prevent breaking changes from reaching production.

## Branch Protection for `main`

Navigate to: Settings → Branches → Add rule

### Rule Pattern

- **Branch name pattern**: `main`

### Protection Settings

#### ✅ Required Status Checks

Enable "Require status checks to pass before merging" and select:

- `quality-gates / quality-gates (3.11)` - Core CI pipeline
- `quality-gates / quality-gates (3.12)` - Python 3.12 compatibility
- `validate-paper-trading / validate-paper-trading (3.11)` - Paper trading validation
- `code-quality-summary` - Overall quality check

Settings:

- [x] Require branches to be up to date before merging
- [x] Strict - All selected checks must pass

#### ✅ Pull Request Reviews

Enable "Require a pull request before merging" with:

- [x] Required approving reviews: **1**
- [x] Dismiss stale pull request approvals when new commits are pushed
- [x] Require review from CODEOWNERS (if configured)
- [x] Require approval of the most recent reviewable push

#### ✅ Additional Protections

- [x] Require conversation resolution before merging
- [x] Require signed commits (recommended for financial systems)
- [x] Include administrators (enforce for everyone)
- [x] Restrict who can push to matching branches
  - Add specific users/teams who can merge to main

#### ✅ Force Push Protection

- [x] Do not allow force pushes
- [x] Do not allow deletions

## Branch Protection for `develop`

### Rule Pattern

- **Branch name pattern**: `develop`

### Protection Settings (Less Strict)

- [x] Require status checks (same as main but non-blocking failures allowed)
- [x] Require pull request (but no review required)
- [x] Allow force pushes from administrators only
- [x] Do not allow deletions

## Automated Enforcement

### Pre-commit Hooks (Local)

Already configured in `.pre-commit-config.yaml`:

- Black formatting
- Ruff linting
- MyPy type checking
- Security scanning (Bandit)
- Trailing whitespace removal
- File size limits

To install locally:

```bash
pre-commit install
```

### CI/CD Pipeline

The following checks run automatically on every push/PR:

1. **Code Quality** (`ci.yml`):
   - Black formatting check
   - Ruff linting
   - MyPy type checking (continue-on-error for now)
   - Bandit security scan
   - Vulnerability scanning
   - Unit tests (must maintain 99%+ pass rate)
   - Coverage reporting

2. **Paper Trading Validation** (`paper-trading-validation.yml`):
   - Core domain entity tests
   - Value object tests
   - PaperBroker tests
   - Smoke test execution
   - Integration tests on PRs

3. **Schedule**:
   - Daily validation runs at 2 AM UTC
   - Catches any environment-related regressions

## Setting Up Protection Rules

1. Go to your GitHub repository
2. Click **Settings** → **Branches**
3. Click **Add rule**
4. Configure as described above
5. Click **Create** or **Save changes**

## Monitoring & Alerts

### Recommended GitHub Actions

1. **Status Badge**: Add to README.md

```markdown
![CI Status](https://github.com/zach-wade/AiStockTrader/workflows/Quality%20Gates%20CI/badge.svg)
![Paper Trading](https://github.com/zach-wade/AiStockTrader/workflows/Paper%20Trading%20Validation/badge.svg)
```

2. **Slack/Email Notifications**: Configure in Settings → Notifications for:
   - CI failures on main branch
   - Paper trading validation failures
   - Security vulnerabilities detected

## Emergency Procedures

### If CI is Broken on Main

1. Revert the breaking commit immediately
2. Fix in a feature branch
3. Ensure all checks pass before re-merging

### Override Protection (Emergency Only)

Administrators can bypass protection rules, but this should be logged:

1. Document reason in commit message
2. Create incident report
3. Review in next team meeting

## Quality Gates Summary

| Check | Current Status | Target | Enforcement |
|-------|---------------|--------|-------------|
| Unit Tests | 99.9% passing | 100% | Required |
| MyPy Errors | 2 errors | 0 | Warning only |
| Coverage | 28.3% | 80% | Info only |
| Security | Passing | No high/critical | Required |
| Paper Trading | Working | Must pass smoke test | Required |

## Maintenance

Review and update these rules:

- Weekly: Check CI failure rates
- Monthly: Review coverage trends
- Quarterly: Audit security scanning results
- Annually: Review and update protection rules

## Contact

For questions or to request changes to branch protection:

- Repository Owner: @zach-wade
- CI/CD Issues: Create issue with `ci/cd` label
