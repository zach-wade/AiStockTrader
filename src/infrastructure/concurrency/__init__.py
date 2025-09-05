"""
Infrastructure layer concurrency utilities.

This module contains adapters and utilities for handling concurrency concerns
at the infrastructure layer, keeping the domain and application layers pure.
"""

from .thread_safe_portfolio_adapter import ThreadSafePortfolioAdapter

__all__ = ["ThreadSafePortfolioAdapter"]
