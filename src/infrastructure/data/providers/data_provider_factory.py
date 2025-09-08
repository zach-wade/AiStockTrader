"""
Data Provider Factory

Factory for creating data provider instances based on configuration.
Supports multiple data sources with fallback options.
"""

import logging
import os

from src.application.interfaces.market_data import IMarketDataProvider

from .alpaca_provider import AlpacaDataProvider
from .polygon_provider import PolygonDataProvider

logger = logging.getLogger(__name__)


class DataProviderFactory:
    """
    Factory for creating data provider instances.

    Manages the creation and configuration of different data providers
    based on available credentials and user preferences.
    """

    @staticmethod
    def create_provider(provider_type: str | None = None, **kwargs) -> IMarketDataProvider:
        """
        Create a data provider instance.

        Args:
            provider_type: Type of provider ('polygon', 'alpaca', or None for auto)
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured data provider instance

        Raises:
            ValueError: If no valid provider can be created
        """
        if provider_type is None:
            # Auto-detect based on available credentials
            provider_type = DataProviderFactory._detect_provider()

        provider_type = provider_type.lower()

        if provider_type == "polygon":
            return DataProviderFactory._create_polygon_provider(**kwargs)
        elif provider_type == "alpaca":
            return DataProviderFactory._create_alpaca_provider(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def _detect_provider() -> str:
        """
        Auto-detect the best available provider based on credentials.

        Returns:
            Provider type string
        """
        # Prefer Polygon if available (more comprehensive data)
        if os.getenv("POLYGON_API_KEY"):
            logger.info("Auto-detected Polygon provider")
            return "polygon"

        # Fall back to Alpaca if available
        if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
            logger.info("Auto-detected Alpaca provider")
            return "alpaca"

        raise ValueError(
            "No data provider credentials found. "
            "Please set POLYGON_API_KEY or ALPACA_API_KEY/ALPACA_SECRET_KEY"
        )

    @staticmethod
    def _create_polygon_provider(**kwargs) -> PolygonDataProvider:
        """
        Create a Polygon data provider.

        Args:
            **kwargs: Provider-specific arguments

        Returns:
            Configured Polygon provider
        """
        api_key = kwargs.get("api_key") or os.getenv("POLYGON_API_KEY")

        if not api_key:
            raise ValueError("Polygon API key is required")

        logger.info("Creating Polygon data provider")
        return PolygonDataProvider(api_key=api_key)

    @staticmethod
    def _create_alpaca_provider(**kwargs) -> AlpacaDataProvider:
        """
        Create an Alpaca data provider.

        Args:
            **kwargs: Provider-specific arguments

        Returns:
            Configured Alpaca provider
        """
        api_key = kwargs.get("api_key") or os.getenv("ALPACA_API_KEY")
        secret_key = kwargs.get("secret_key") or os.getenv("ALPACA_SECRET_KEY")
        paper = kwargs.get("paper", True)

        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials are required")

        logger.info(f"Creating Alpaca data provider (paper={paper})")
        return AlpacaDataProvider(api_key=api_key, secret_key=secret_key, paper=paper)

    @staticmethod
    def create_multi_provider(
        primary: str = "polygon", fallback: str = "alpaca"
    ) -> "MultiSourceProvider":
        """
        Create a multi-source provider with fallback.

        Args:
            primary: Primary provider type
            fallback: Fallback provider type

        Returns:
            Multi-source provider with fallback capability
        """
        primary_provider = DataProviderFactory.create_provider(primary)
        fallback_provider = DataProviderFactory.create_provider(fallback)

        return MultiSourceProvider(primary=primary_provider, fallback=fallback_provider)


class MultiSourceProvider(IMarketDataProvider):
    """
    Data provider that uses multiple sources with fallback.

    Attempts to use the primary provider first, falling back to
    secondary provider if the primary fails.
    """

    def __init__(self, primary: IMarketDataProvider, fallback: IMarketDataProvider):
        """
        Initialize multi-source provider.

        Args:
            primary: Primary data provider
            fallback: Fallback data provider
        """
        self.primary = primary
        self.fallback = fallback
        self.current = primary

    async def connect(self) -> None:
        """Connect to data providers."""
        try:
            await self.primary.connect()
            self.current = self.primary
            logger.info("Connected to primary data provider")
        except Exception as e:
            logger.warning(f"Primary provider connection failed: {e}")
            await self.fallback.connect()
            self.current = self.fallback
            logger.info("Connected to fallback data provider")

    async def disconnect(self) -> None:
        """Disconnect from data providers."""
        await self.primary.disconnect()
        await self.fallback.disconnect()

    async def get_latest_price(self, symbol):
        """Get latest price with fallback."""
        try:
            return await self.current.get_latest_price(symbol)
        except Exception as e:
            if self.current == self.primary:
                logger.warning(f"Primary provider failed: {e}, trying fallback")
                self.current = self.fallback
                return await self.fallback.get_latest_price(symbol)
            raise

    async def get_latest_quote(self, symbol):
        """Get latest quote with fallback."""
        try:
            return await self.current.get_latest_quote(symbol)
        except Exception as e:
            if self.current == self.primary:
                logger.warning(f"Primary provider failed: {e}, trying fallback")
                self.current = self.fallback
                return await self.fallback.get_latest_quote(symbol)
            raise

    async def get_historical_bars(self, symbol, start, end, timeframe="1Day"):
        """Get historical bars with fallback."""
        try:
            return await self.current.get_historical_bars(symbol, start, end, timeframe)
        except Exception as e:
            if self.current == self.primary:
                logger.warning(f"Primary provider failed: {e}, trying fallback")
                self.current = self.fallback
                return await self.fallback.get_historical_bars(symbol, start, end, timeframe)
            raise

    async def subscribe_to_prices(self, symbols, handler):
        """Subscribe to prices (not implemented)."""
        raise NotImplementedError("Real-time subscription not yet implemented")

    async def unsubscribe_from_prices(self, symbols):
        """Unsubscribe from prices (not implemented)."""
        raise NotImplementedError("Real-time subscription not yet implemented")
