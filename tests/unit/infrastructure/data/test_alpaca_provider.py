"""
Unit tests for AlpacaDataProvider
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.application.interfaces.market_data import MarketDataError
from src.domain.value_objects import Price
from src.infrastructure.data.providers.alpaca_provider import AlpacaDataProvider


class TestAlpacaDataProvider:
    """Test suite for AlpacaDataProvider."""

    @pytest.fixture
    def provider(self):
        """Create an AlpacaDataProvider instance for testing."""
        with patch.dict(
            "os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}
        ):
            return AlpacaDataProvider()

    def test_initialization_without_credentials(self):
        """Test that initialization without credentials raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Alpaca API credentials are required"):
                AlpacaDataProvider()

    @pytest.mark.asyncio
    async def test_connect_success(self, provider):
        """Test successful connection."""
        mock_account = MagicMock()
        mock_account.account_number = "TEST123"

        with patch.object(provider.trading_client, "get_account", return_value=mock_account):
            await provider.connect()
            assert provider.connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, provider):
        """Test connection failure."""
        with patch.object(
            provider.trading_client, "get_account", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(MarketDataError, match="Failed to connect"):
                await provider.connect()

    @pytest.mark.asyncio
    async def test_get_current_price_success(self, provider):
        """Test successful price retrieval."""
        mock_trade_data = MagicMock()
        mock_trade_data.price = 150.50

        mock_response = {"AAPL": mock_trade_data}

        with patch.object(
            provider.data_client, "get_stock_latest_trade", return_value=mock_response
        ):
            provider.connected = True
            price = await provider.get_current_price("AAPL")

            assert isinstance(price, Price)
            assert price.value == Decimal("150.50")

    @pytest.mark.asyncio
    async def test_get_current_price_no_data(self, provider):
        """Test price retrieval with no data."""
        mock_response = {}

        with patch.object(
            provider.data_client, "get_stock_latest_trade", return_value=mock_response
        ):
            provider.connected = True

            with pytest.raises(MarketDataError, match="No price data"):
                await provider.get_current_price("INVALID")

    @pytest.mark.asyncio
    async def test_get_current_quote_success(self, provider):
        """Test successful quote retrieval."""
        mock_quote_data = MagicMock()
        mock_quote_data.bid_price = 150.00
        mock_quote_data.ask_price = 150.50
        mock_quote_data.bid_size = 100
        mock_quote_data.ask_size = 200
        mock_quote_data.timestamp = datetime.now()

        mock_response = {"AAPL": mock_quote_data}

        with patch.object(
            provider.data_client, "get_stock_latest_quote", return_value=mock_response
        ):
            provider.connected = True
            quote = await provider.get_current_quote("AAPL")

            assert quote.bid_price.value == Decimal("150.00")
            assert quote.ask_price.value == Decimal("150.50")
            assert quote.bid_size == 100
            assert quote.ask_size == 200

    @pytest.mark.asyncio
    async def test_get_historical_bars_success(self, provider):
        """Test successful historical bars retrieval."""
        mock_bar = MagicMock()
        mock_bar.open = 150.0
        mock_bar.high = 155.0
        mock_bar.low = 149.0
        mock_bar.close = 154.0
        mock_bar.volume = 1000000
        mock_bar.timestamp = datetime(2021, 1, 1)

        mock_response = MagicMock()
        mock_response.data = {"AAPL": [mock_bar]}

        with patch.object(provider.data_client, "get_stock_bars", return_value=mock_response):
            provider.connected = True

            start = datetime(2021, 1, 1)
            end = datetime(2021, 1, 2)
            bars = await provider.get_historical_bars("AAPL", start, end)

            assert len(bars) == 1
            assert bars[0].open.value == Decimal("150.0")
            assert bars[0].high.value == Decimal("155.0")
            assert bars[0].low.value == Decimal("149.0")
            assert bars[0].close.value == Decimal("154.0")
            assert bars[0].volume == 1000000

    @pytest.mark.asyncio
    async def test_is_market_open(self, provider):
        """Test market open check."""
        mock_clock = MagicMock()
        mock_clock.is_open = True

        with patch.object(provider.trading_client, "get_clock", return_value=mock_clock):
            provider.connected = True
            is_open = await provider.is_market_open()
            assert is_open is True

    @pytest.mark.asyncio
    async def test_get_symbol_info(self, provider):
        """Test symbol info retrieval."""
        mock_asset = MagicMock()
        mock_asset.symbol = "AAPL"
        mock_asset.name = "Apple Inc."
        mock_asset.exchange = "NASDAQ"
        mock_asset.tradable = True
        mock_asset.marginable = True
        mock_asset.shortable = True
        mock_asset.easy_to_borrow = True
        mock_asset.fractionable = True

        with patch.object(provider.trading_client, "get_all_assets", return_value=[mock_asset]):
            provider.connected = True
            info = await provider.get_symbol_info("AAPL")

            assert info["symbol"] == "AAPL"
            assert info["name"] == "Apple Inc."
            assert info["tradable"] is True

    @pytest.mark.asyncio
    async def test_stream_methods_not_implemented(self, provider):
        """Test that streaming methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            async for _ in provider.stream_prices(["AAPL"]):
                pass

        with pytest.raises(NotImplementedError):
            async for _ in provider.stream_quotes(["AAPL"]):
                pass
