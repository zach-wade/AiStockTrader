# File: main/trading_engine/brokers/__init__.py

# This file marks the 'brokers' directory as a Python package.
# In a strict, modular design, it should be kept minimal.
# Do NOT import specific broker implementations (like AlpacaBroker) directly here.
# Instead, import them explicitly where they are used (e.g., in ExecutionEngine
# or a broker factory).

# Remove lines like:
# from .alpaca_broker import AlpacaBroker
# from .ib_broker import IBBroker
# etc.

# Export broker registry for backward compatibility
from .broker_factory import broker_registry

__all__ = ["broker_registry"]
