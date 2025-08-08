"""
Defines the Abstract Base Class for all High-Frequency Trading (HFT) strategies.

These strategies are event-driven, reacting to real-time market data ticks
(order book updates, trades) rather than historical bars.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseHFTStrategy(ABC):
    """
    The contract for event-driven, high-frequency strategies.
    """
    def __init__(self, config: Dict, strategy_specific_config: Dict):
        """
        Initializes the HFT strategy.

        Args:
            config: The global application configuration.
            strategy_specific_config: The configuration block for this specific strategy.
        """
        self.config = config
        self.params = strategy_specific_config
        self.name = "base_hft" # Should be overridden by child classes

    @abstractmethod
    async def on_orderbook_update(self, symbol: str, orderbook_data: Dict) -> List[Dict]:
        """
        Called by the HFT engine on every order book update for a subscribed symbol.
        This is where the core alpha logic for order book strategies will live.

        Args:
            symbol: The symbol that was updated.
            orderbook_data: The raw order book update data from the source.

        Returns:
            A list of execution-ready order dictionaries.
        """
        pass

    @abstractmethod
    async def on_trade_update(self, symbol: str, trade_data: Dict) -> List[Dict]:
        """
        Called by the HFT engine on every new trade execution in the market.

        Args:
            symbol: The symbol that was updated.
            trade_data: The raw trade data from the source.

        Returns:
            A list of execution-ready order dictionaries.
        """
        pass