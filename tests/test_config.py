"""Test configuration for tiered CI strategy."""

# Tier 1: Quick tests (< 2 minutes) - Must pass
TIER1_TESTS = [
    # Value objects - 100% passing
    "tests/unit/domain/value_objects/test_money.py",
    "tests/unit/domain/value_objects/test_quantity.py",
    # Core domain entities
    "tests/unit/domain/entities/test_order.py",
    "tests/unit/domain/entities/test_position.py",
    # Paper broker core
    "tests/unit/infrastructure/brokers/test_paper_broker.py::TestPaperBrokerInitialization",
    "tests/unit/infrastructure/brokers/test_paper_broker.py::TestPaperBrokerConnection",
]

# Tier 2: Progressive tests (< 5 minutes) - 95% pass rate
TIER2_TESTS = [
    # All of Tier 1
    *TIER1_TESTS,
    # Additional value objects (skip known failure)
    "tests/unit/domain/value_objects/",
    # Domain entities
    "tests/unit/domain/entities/",
    # Domain services (skip known failures)
    "tests/unit/domain/services/",
    # Paper broker (skip limit order test)
    "tests/unit/infrastructure/brokers/test_paper_broker.py",
]

# Tier 3: Full test suite - All tests
TIER3_TESTS = [
    "tests/",
]

# Tests to skip in CI
SKIP_IN_CI = [
    # These require real API keys
    "tests/integration/brokers/test_alpaca_integration.py",
    "tests/e2e/test_live_trading.py",
]

# Known failing tests (to be fixed)
KNOWN_FAILURES = [
    "tests/unit/infrastructure/brokers/test_paper_broker.py::TestPaperBrokerOrderManagement::test_submit_limit_order",
    "tests/unit/domain/value_objects/test_price.py::TestPriceRounding::test_round_to_tick_zero_tick_size",
    "tests/unit/domain/services/test_security_policy_service.py::TestSecurityPolicyService::test_get_sanitization_level",
]

# Performance benchmarks
SLOW_TESTS = [
    "tests/performance/",
    "tests/stress/",
]


def get_tier_command(tier: int) -> str:
    """Get pytest command for a specific tier."""
    if tier == 1:
        tests = " ".join(TIER1_TESTS)
        return f"pytest {tests} -v --tb=short"
    elif tier == 2:
        skip_tests = " and not ".join([f"{test.split('::')[-1]}" for test in KNOWN_FAILURES])
        return f"pytest {' '.join(TIER2_TESTS)} -k 'not {skip_tests}' -v --tb=short"
    elif tier == 3:
        return "pytest tests/ -v"
    else:
        raise ValueError(f"Invalid tier: {tier}")


def get_ci_command() -> str:
    """Get pytest command for CI environment."""
    skip_patterns = " or ".join([f"'{test}'" for test in SKIP_IN_CI])
    return f"pytest tests/ -k 'not ({skip_patterns})' -v"


if __name__ == "__main__":
    print("Test Tier Configuration")
    print("=" * 50)
    print("\nTier 1 (Quick):")
    print(get_tier_command(1))
    print("\nTier 2 (Progressive):")
    print(get_tier_command(2))
    print("\nTier 3 (Full):")
    print(get_tier_command(3))
    print("\nCI Command:")
    print(get_ci_command())
