#!/usr/bin/env python3
"""Debug script to understand rate limiting behavior."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from infrastructure.rate_limiting.decorators import rate_limit
from infrastructure.rate_limiting.config import RateLimitConfig
from infrastructure.rate_limiting.decorators import initialize_rate_limiting
from infrastructure.rate_limiting.exceptions import RateLimitExceeded

# Initialize rate limiting
config = RateLimitConfig.from_env()
initialize_rate_limiting(config)

@rate_limit(limit=5, window="1min")
def test_function(user_id=None):
    print(f"Called with user_id={user_id}")
    return "success"

print("Testing rate limiting behavior...")
try:
    # Call the function 6 times like in the test
    for i in range(6):
        print(f"Call {i+1}:")
        result = test_function(user_id="user1")
        print(f"  Result: {result}")
except RateLimitExceeded as e:
    print(f"  Rate limit exceeded: {e}")