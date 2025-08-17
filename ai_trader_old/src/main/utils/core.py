"""
Core utilities for the AI Trader system.

This module provides a unified interface to all core utilities including:
- Async helpers for concurrent processing
- Exception types for proper error handling
- Time helpers for market-aware datetime operations
- File helpers for safe file operations
- Secure random number generation
- Secure serialization
- Error handling and circuit breaker patterns
- Logging configuration

This is the main interface module that imports all core utilities from the
core/ subdirectory for easy access throughout the system.
"""

# Import and explicitly re-export core utilities

# Version info
__version__ = "2.0.0"
__author__ = "AI Trader Team"
