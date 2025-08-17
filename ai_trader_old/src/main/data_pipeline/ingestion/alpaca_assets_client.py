"""
Alpaca Assets Client

Provides access to Alpaca's asset listing functionality for discovering
tradable securities. This is primarily used by the Layer 0 scanner to
build the initial universe of symbols.
"""

# Standard library imports
from typing import Any

# Third-party imports
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

# Local imports
from main.utils.core import get_logger

logger = get_logger(__name__)


class AlpacaAssetsClient:
    """
    Client for fetching asset information from Alpaca.

    This client wraps Alpaca's TradingClient to provide asset discovery
    functionality for the scanner system.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the Alpaca assets client.

        Args:
            config: Configuration dictionary with Alpaca credentials
        """
        self.config = config
        self.logger = logger

        # Extract Alpaca credentials from config
        api_key = config.get("alpaca", {}).get("api_key") or config.get("alpaca.api_key")
        api_secret = config.get("alpaca", {}).get("api_secret") or config.get("alpaca.api_secret")
        base_url = config.get("alpaca", {}).get("base_url") or config.get(
            "alpaca.base_url", "https://paper-api.alpaca.markets"
        )

        if not api_key or not api_secret:
            raise ValueError("Alpaca API credentials not found in config")

        # Initialize the trading client
        self.trading_client = TradingClient(
            api_key=api_key, secret_key=api_secret, url_override=base_url
        )

        self.logger.info(f"AlpacaAssetsClient initialized with base URL: {base_url}")

    async def get_all_tradable_assets(self) -> list[dict[str, Any]]:
        """
        Fetch all tradable US equity assets from Alpaca.

        Returns:
            List of asset dictionaries with fields:
            - symbol: Stock ticker symbol
            - name: Company name
            - exchange: Exchange where the asset is traded
            - asset_class: Type of asset (us_equity)
            - tradable: Whether the asset is tradable
            - status: Asset status (active, inactive)
            - easy_to_borrow: Whether the asset is easy to borrow for shorting
            - fractionable: Whether fractional shares are allowed
        """
        try:
            # Create request for US equities only
            request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)

            # Fetch assets from Alpaca
            self.logger.info("Fetching tradable assets from Alpaca...")
            assets = self.trading_client.get_all_assets(request)

            # Convert to list of dictionaries
            asset_list = []
            for asset in assets:
                asset_dict = {
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "exchange": (
                        asset.exchange.value
                        if hasattr(asset.exchange, "value")
                        else str(asset.exchange)
                    ),
                    "asset_class": "us_equity",  # Standardized
                    "tradable": asset.tradable,
                    "status": "active" if asset.status == AssetStatus.ACTIVE else "inactive",
                    "easy_to_borrow": getattr(asset, "easy_to_borrow", False),
                    "fractionable": getattr(asset, "fractionable", False),
                    "shortable": getattr(asset, "shortable", False),
                    "marginable": getattr(asset, "marginable", False),
                    "maintenance_margin_requirement": getattr(
                        asset, "maintenance_margin_requirement", None
                    ),
                }
                asset_list.append(asset_dict)

            self.logger.info(f"Successfully fetched {len(asset_list)} assets from Alpaca")
            return asset_list

        except Exception as e:
            self.logger.error(f"Failed to fetch assets from Alpaca: {e}")
            raise

    async def get_asset(self, symbol: str) -> dict[str, Any] | None:
        """
        Get information for a specific asset.

        Args:
            symbol: The stock ticker symbol

        Returns:
            Asset dictionary if found, None otherwise
        """
        try:
            asset = self.trading_client.get_asset(symbol)

            if asset:
                return {
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "exchange": (
                        asset.exchange.value
                        if hasattr(asset.exchange, "value")
                        else str(asset.exchange)
                    ),
                    "asset_class": "us_equity",
                    "tradable": asset.tradable,
                    "status": "active" if asset.status == AssetStatus.ACTIVE else "inactive",
                    "easy_to_borrow": getattr(asset, "easy_to_borrow", False),
                    "fractionable": getattr(asset, "fractionable", False),
                    "shortable": getattr(asset, "shortable", False),
                    "marginable": getattr(asset, "marginable", False),
                }

            return None

        except Exception as e:
            self.logger.warning(f"Failed to get asset info for {symbol}: {e}")
            raise

    async def is_tradable(self, symbol: str) -> bool:
        """
        Check if a symbol is tradable.

        Args:
            symbol: The stock ticker symbol

        Returns:
            True if the symbol is tradable, False otherwise
        """
        asset_info = await self.get_asset(symbol)
        return asset_info.get("tradable", False) if asset_info else False
