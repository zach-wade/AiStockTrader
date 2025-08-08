# File: ai_trader/__init__.py

# This file marks the 'ai_trader' directory as the root of your Python package.
# In a strict, modular design, it should be kept absolutely minimal
# to prevent circular import issues and to make dependencies explicit.

# Do NOT import sub-modules directly here if they create deep dependency graphs
# or if they will be imported by modules that are also part of your common models/interfaces.

# You can keep it entirely empty, or define __version__ etc.
# Remove these lines if they exist:
# from .models import create_strategy, AVAILABLE_STRATEGIES
# from .data_pipeline import ... (any top-level imports you had here)
# from .trading_engine import ...
# __all__ = [...] # If you had an __all__, make it empty or remove it.

# Minimal example:
# __version__ = "0.1.0"
# import logging
# logger = logging.getLogger(__name__)

# Leave it empty for now to break the cycle