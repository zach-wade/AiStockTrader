#!/usr/bin/env python3
"""
AI Trader Setup Script

Enterprise-Grade Algorithmic Trading Platform
Backward compatibility setup.py for older build systems.

For modern installations, use: pip install -e .
This will use pyproject.toml configuration.
"""

# Third-party imports
from setuptools import setup

# All configuration is now in pyproject.toml
# This file exists for backward compatibility with older tools

if __name__ == "__main__":
    setup()
