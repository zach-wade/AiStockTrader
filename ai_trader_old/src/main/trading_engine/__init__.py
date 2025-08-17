"""
AI Trader Trading Engine

Core trading execution system providing:
- Order execution and management
- Multiple broker integrations
- Trading algorithms (TWAP, VWAP, Iceberg)
- Signal generation and processing
- Portfolio management
"""

# Core trading system
from .algorithms.iceberg import IcebergAlgorithm

# Execution algorithms
from .algorithms.twap import TWAPAlgorithm
from .algorithms.vwap import VWAPAlgorithm
from .brokers.alpaca_broker import AlpacaBroker

# Broker interfaces
from .brokers.broker_interface import BrokerInterface
from .brokers.paper_broker import PaperBroker
from .core.order_manager import OrderManager
from .core.portfolio_manager import PortfolioManager
from .core.trading_system import TradingSystem

# Trading signals
from .signals.unified_signal import SignalAggregator

__all__ = [
    # Core components
    "TradingSystem",
    "OrderManager",
    "PortfolioManager",
    # Signals
    "SignalAggregator",
    # Algorithms
    "TWAPAlgorithm",
    "VWAPAlgorithm",
    "IcebergAlgorithm",
    # Brokers
    "BrokerInterface",
    "AlpacaBroker",
    "PaperBroker",
]
