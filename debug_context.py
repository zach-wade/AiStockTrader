#!/usr/bin/env python3
"""Debug script to understand context building."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from infrastructure.rate_limiting.decorators import _build_context_from_function, _build_custom_identifier
import inspect

def test_function(user_id=None):
    return "success"

# Test context building
args = ("user1",)  # positional argument
kwargs = {}
context = _build_context_from_function(test_function, args, kwargs, "user", None)
print(f"Context from positional args: user_id={context.user_id}")

# Test with keyword argument
args = ()
kwargs = {"user_id": "user1"}
context = _build_context_from_function(test_function, args, kwargs, "user", None)
print(f"Context from kwargs: user_id={context.user_id}")

# Test identifier building
identifier = _build_custom_identifier(context, "user", "test_function")
print(f"Identifier: {identifier}")

# Test function signature inspection
sig = inspect.signature(test_function)
bound_args = sig.bind(*(), **{"user_id": "user1"})
bound_args.apply_defaults()
print(f"Bound args: {bound_args.arguments}")