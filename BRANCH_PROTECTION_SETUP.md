# Branch Protection Rules Configuration

## GitHub Repository Settings

Navigate to: <https://github.com/zach-wade/AiStockTrader/settings/branches>

## Current CI/CD Workflows

### New Workflow Structure (December 2024)

- **ci-commit.yml** - Fast validation on every commit (2-3 min)
- **ci-pull-request.yml** - Comprehensive PR validation (5-8 min)
- **ci-paper-trading.yml** - Scheduled paper trading tests (10-15 min)

### Legacy Workflows (Still Active)

- **ci-quick.yml** - Quick validation (being replaced by ci-commit)
- **ci-progressive.yml** - Progressive validation (being replaced by ci-pull-request)

## Main Branch Protection Rules

### Required Status Checks

**IMPORTANT**: The current branch protection is configured with job names that need updating.

Current (Incorrect):

- `quick-validation (3.11)`
- `quick-validation (3.12)`
- `progressive-validation (3.11)`
- `progressive-validation (3.12)`

Should Be Updated To:

1. **Fast Validation (3.11)** - From ci-commit.yml
2. **Fast Validation (3.12)** - From ci-commit.yml
3. **PR Validation (3.11)** - From ci-pull-request.yml
4. **PR Validation (3.12)** - From ci-pull-request.yml

### Additional Settings

- ✅ **Require branches to be up to date before merging**
- ✅ **Dismiss stale reviews when new commits are pushed**
- ✅ **Require review from 1 reviewer** (currently set)
- ✅ **Require conversation resolution before merging**

## Pull Request Workflow

For all PRs to main:

1. **ci-commit.yml** runs immediately (fast feedback)
2. **ci-pull-request.yml** runs comprehensive tests
3. Both must pass before merge is allowed

## Feature Branch Strategy

Feature branches automatically trigger:

- **ci-commit.yml** on every push (fast validation)
- **ci-pull-request.yml** when PR is opened

Trading-related changes also trigger:

- **ci-paper-trading.yml** (can be manually triggered)

## CI Workflow Summary

| Workflow | Trigger | Runtime | Purpose |
|----------|---------|---------|---------|
| ci-commit | Every push | 2-3 min | Fast validation |
| ci-pull-request | PR to main/develop | 5-8 min | Comprehensive tests |
| ci-paper-trading | Schedule/Manual | 10-15 min | Trading validation |

## Manual Configuration Steps

To update branch protection with new workflow names:

1. Go to Settings → Branches
2. Edit rule for `main` branch
3. Remove old status checks:
   - Remove `quick-validation` checks
   - Remove `progressive-validation` checks
4. Add new status checks under "Require status checks to pass":
   - Search and add `Fast Validation` (from ci-commit workflow)
   - Search and add `PR Validation` (from ci-pull-request workflow)
5. Ensure these settings are checked:
   - ✅ Require branches to be up to date before merging
   - ✅ Require status checks to pass before merging
   - ✅ Dismiss stale pull request approvals when new commits are pushed
6. Save changes

## Verification Steps

After updating branch protection:

1. Create a test branch with a small change
2. Push to trigger ci-commit workflow
3. Create PR to trigger ci-pull-request workflow
4. Verify both checks appear and must pass
5. Confirm merge is blocked until checks pass
6. Verify merge is allowed once all checks pass

## Troubleshooting

If status checks don't appear:

1. Ensure workflows have run at least once on the default branch
2. Check workflow file syntax is correct
3. Verify job names match exactly in branch protection settings
4. Wait a few minutes for GitHub to index new workflows

## Scheduled Tests

**ci-paper-trading.yml** runs automatically:

- 9 AM EST weekdays (market open)
- 3 PM EST weekdays (before close)
- Can be manually triggered with custom parameters
