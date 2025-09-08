"""
Alpaca Data Provider

Provides market data through the Alpaca API.
This is a lighter alternative to Polygon for basic market data needs.
"""

import logging
import os
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
from typing import Any

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading import TradingClient

from src.application.interfaces.market_data import (
    Bar,
    IMarketDataProvider,
    MarketDataError,
    PriceUpdate,
    Quote,
)
from src.domain.value_objects import Price

logger = logging.getLogger(__name__)


class AlpacaDataProvider(IMarketDataProvider):
    """
    Alpaca market data provider implementation.

    Provides access to market data through the Alpaca API.
    Simpler than Polygon but included with trading account.
    """

    def __init__(
        self, api_key: str | None = None, secret_key: str | None = None, paper: bool = True
    ):
        """
        Initialize Alpaca data provider.

        Args:
            api_key: Alpaca API key. If None, reads from ALPACA_API_KEY env var.
            secret_key: Alpaca secret key. If None, reads from ALPACA_SECRET_KEY env var.
            paper: Whether to use paper trading endpoint
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials are required")

        # Create both data and trading clients
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key, secret_key=self.secret_key
        )

        self.trading_client = TradingClient(
            api_key=self.api_key, secret_key=self.secret_key, paper=paper
        )

        self.connected = False

    async def connect(self) -> None:
        """Establish connection to Alpaca API."""
        try:
            # Test connection by getting account info
            account = self.trading_client.get_account()
            self.connected = True
            logger.info(f"Connected to Alpaca API (Account: {account.account_number})")
        except Exception as e:
            raise MarketDataError(f"Failed to connect to Alpaca: {e}")

    async def disconnect(self) -> None:
        """Close connection to Alpaca API."""
        self.connected = False
        logger.info("Disconnected from Alpaca API")

    async def get_current_price(self, symbol: str) -> Price:
        """
        Get the latest price for a symbol.

        Args:
            symbol: The symbol to get price for

        Returns:
            Latest price

        Raises:
            MarketDataError: If price fetch fails
        """
        if not self.connected:
            await self.connect()

        try:
            # Get latest trade
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trade = self.data_client.get_stock_latest_trade(request)

            if symbol in trade:
                trade_data = trade[symbol]
                return Price(Decimal(str(trade_data.price)))
            else:
                raise MarketDataError(f"No price data for {symbol}")

        except Exception as e:
            raise MarketDataError(f"Failed to get price: {e}")

    async def get_current_quote(self, symbol: str) -> Quote:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: The symbol to get quote for

        Returns:
            Latest quote with bid/ask
        """
        if not self.connected:
            await self.connect()

        try:
            # Get latest quote
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)

            if symbol in quotes:
                quote = quotes[symbol]
                from src.domain.value_objects import Symbol

                return Quote(
                    symbol=Symbol(symbol),
                    bid_price=Price(Decimal(str(quote.bid_price))),
                    ask_price=Price(Decimal(str(quote.ask_price))),
                    bid_size=quote.bid_size,
                    ask_size=quote.ask_size,
                    timestamp=quote.timestamp,
                )
            else:
                raise MarketDataError(f"No quote data for {symbol}")

        except Exception as e:
            raise MarketDataError(f"Failed to get quote: {e}")

    async def get_historical_bars(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> list[Bar]:
        """
        Get historical price bars for a symbol.

        Args:
            symbol: The symbol to get bars for
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)

        Returns:
            List of price bars
        """
        if not self.connected:
            await self.connect()

        # Convert timeframe to Alpaca format
        # TimeFrame enum values from alpaca-py
        from alpaca.data.timeframe import TimeFrameUnit

        timeframe_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }

        alpaca_timeframe = timeframe_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Day))

        try:
            # Create bars request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_timeframe,
                start=start,
                end=end,
                adjustment="raw",
                limit=10000,
            )

            # Get bars from Alpaca
            bars_response = self.data_client.get_stock_bars(request)

            bars = []
            # The response has a 'data' attribute that contains the bars
            if hasattr(bars_response, "data") and bars_response.data:
                if symbol in bars_response.data:
                    for bar_data in bars_response.data[symbol]:
                        bar = Bar(
                            symbol=symbol,
                            open=Price(Decimal(str(bar_data.open))),
                            high=Price(Decimal(str(bar_data.high))),
                            low=Price(Decimal(str(bar_data.low))),
                            close=Price(Decimal(str(bar_data.close))),
                            volume=bar_data.volume,
                            timestamp=bar_data.timestamp,
                        )
                        bars.append(bar)
            elif hasattr(bars_response, symbol):
                # Alternative structure
                for bar_data in getattr(bars_response, symbol):
                    bar = Bar(
                        symbol=symbol,
                        open=Price(Decimal(str(bar_data.open))),
                        high=Price(Decimal(str(bar_data.high))),
                        low=Price(Decimal(str(bar_data.low))),
                        close=Price(Decimal(str(bar_data.close))),
                        volume=bar_data.volume,
                        timestamp=bar_data.timestamp,
                    )
                    bars.append(bar)

            # Log if no data
            if not bars:
                logger.warning(f"No bars returned from Alpaca for {symbol}")
            else:
                logger.info(f"Got {len(bars)} bars from Alpaca for {symbol}")

            return bars

        except Exception as e:
            raise MarketDataError(f"Failed to get historical bars: {e}")

    async def stream_prices(self, symbols: list[str]) -> AsyncIterator[PriceUpdate]:
        """Stream real-time price updates."""
        raise NotImplementedError("Real-time streaming not yet implemented")
        # This would require WebSocket implementation
        yield  # type: ignore

    async def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        """Stream real-time quote updates."""
        raise NotImplementedError("Real-time streaming not yet implemented")
        # This would require WebSocket implementation
        yield  # type: ignore

    async def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        if not self.connected:
            await self.connect()

        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get detailed information about a symbol."""
        if not self.connected:
            await self.connect()

        try:
            from alpaca.trading.requests import GetAssetsRequest

            request = GetAssetsRequest(symbol_or_symbols=symbol)
            assets = self.trading_client.get_all_assets(request)

            if assets:
                asset = assets[0]
                return {
                    "symbol": asset.symbol,
                    "name": asset.name if hasattr(asset, "name") else None,
                    "exchange": asset.exchange,
                    "tradable": asset.tradable,
                    "marginable": asset.marginable,
                    "shortable": asset.shortable,
                    "easy_to_borrow": asset.easy_to_borrow,
                    "fractionable": asset.fractionable,
                }
            return {}
        except Exception as e:
            raise MarketDataError(f"Failed to get symbol info: {e}")

    # Additional Alpaca-specific methods

    async def get_market_status(self) -> dict[str, Any]:
        """
        Get current market status.

        Returns:
            Dictionary with market status information
        """
        if not self.connected:
            await self.connect()

        try:
            clock = self.trading_client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
                "timestamp": clock.timestamp,
            }
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return {}

    async def get_tradable_assets(self) -> list[str]:
        """
        Get list of tradable assets.

        Returns:
            List of tradable symbol strings
        """
        if not self.connected:
            await self.connect()

        try:
            from alpaca.trading.enums import AssetClass, AssetStatus
            from alpaca.trading.requests import GetAssetsRequest

            request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
            assets = self.trading_client.get_all_assets(request)
            return [asset.symbol for asset in assets if asset.tradable]
        except Exception as e:
            logger.error(f"Failed to get tradable assets: {e}")
            return []
