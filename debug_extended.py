#!/usr/bin/env python3
"""Extended debug script to test more calls."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from infrastructure.rate_limiting.decorators import get_rate_limit_manager, _build_context_from_function, _build_custom_identifier, _create_custom_limiter
from infrastructure.rate_limiting.config import RateLimitConfig, RateLimitRule, TimeWindow, RateLimitAlgorithm
from infrastructure.rate_limiting.decorators import initialize_rate_limiting

# Initialize rate limiting
config = RateLimitConfig.from_env()
initialize_rate_limiting(config)

def test_function(user_id=None):
    return "success"

manager = get_rate_limit_manager()

# Simulate the exact same flow as the decorator
args = ()
kwargs = {"user_id": "user1"}
context = _build_context_from_function(test_function, args, kwargs, "user", None)

# Create the rule with limit 5
custom_rule = RateLimitRule(
    limit=5,
    window=TimeWindow("1min"),
    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
    burst_allowance=None,
    identifier=f"custom:{test_function.__name__}",
)

# Create the limiter
limiter_id = _create_custom_limiter(manager, "token_bucket", custom_rule)

# Build identifier
identifier = _build_custom_identifier(context, "user", test_function.__name__)

# Test up to 10 calls to see the pattern
print("Testing rate limits (limit should be 5):")
for i in range(10):
    result = manager._check_limiter(limiter_id, identifier, 1)
    print(f"Check {i+1}: allowed={result.allowed}, current_count={result.current_count}")
    if not result.allowed:
        print(f"Rate limit exceeded at call {i+1}")
        break