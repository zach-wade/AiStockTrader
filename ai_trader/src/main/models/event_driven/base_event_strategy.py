"""
Defines the Abstract Base Class for all real-time, event-driven strategies.

These strategies react to discrete events like news articles, trades, or
order book updates, rather than operating on historical bars.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseEventStrategy(ABC):
    """
    The contract for event-driven, low-latency strategies.
    """
    def __init__(self, config: Dict, strategy_specific_config: Dict):
        """
        Initializes the Event-Driven strategy.

        Args:
            config: The global application configuration.
            strategy_specific_config: The configuration block for this specific strategy.
        """
        self.config = config
        self.params = strategy_specific_config
        self.name = "base_event_strategy" # Should be overridden by child classes

    async def on_news_event(self, event: Dict) -> List[Dict]:
        """
        Handles a new, enriched news event from the engine.
        Returns a list of execution-ready orders.
        """
        return [] # Default implementation does nothing

    async def on_orderbook_update(self, symbol: str, orderbook: Dict) -> List[Dict]:
        """
        Handles a new order book tick from the engine.
        Returns a list of execution-ready orders.
        """
        return [] # Default implementation does nothing

    async def on_trade_update(self, symbol: str, trade: Dict) -> List[Dict]:
        """
        Handles a new market trade from the engine.
        Returns a list of execution-ready orders.
        """
        return [] # Default implementation does nothing