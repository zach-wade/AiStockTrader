"""
Unit tests for PolygonDataProvider
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.application.interfaces.market_data import MarketDataError, RateLimitError
from src.domain.value_objects import Price
from src.infrastructure.data.providers.polygon_provider import PolygonDataProvider


class TestPolygonDataProvider:
    """Test suite for PolygonDataProvider."""

    @pytest.fixture
    def provider(self):
        """Create a PolygonDataProvider instance for testing."""
        with patch.dict("os.environ", {"POLYGON_API_KEY": "test_key"}):
            return PolygonDataProvider()

    @pytest.mark.asyncio
    async def test_connect(self, provider):
        """Test provider connection."""
        await provider.connect()
        assert provider._client is not None
        await provider.disconnect()

    @pytest.mark.asyncio
    async def test_get_current_price_success(self, provider):
        """Test successful price retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": {"p": 150.50}}
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_client") as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)

            price = await provider.get_current_price("AAPL")

            assert isinstance(price, Price)
            assert price.value == Decimal("150.50")
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_price_no_data(self, provider):
        """Test price retrieval with no data."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": None}
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_client") as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)

            with pytest.raises(MarketDataError, match="No price data"):
                await provider.get_current_price("INVALID")

    @pytest.mark.asyncio
    async def test_get_current_price_rate_limit(self, provider):
        """Test rate limit error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        error = httpx.HTTPStatusError(
            "Too Many Requests", request=MagicMock(), response=mock_response
        )

        with patch.object(provider, "_client") as mock_client:
            mock_client.get = AsyncMock(side_effect=error)

            with pytest.raises(RateLimitError):
                await provider.get_current_price("AAPL")

    @pytest.mark.asyncio
    async def test_get_historical_bars_success(self, provider):
        """Test successful historical bars retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "o": 150.0,
                    "h": 155.0,
                    "l": 149.0,
                    "c": 154.0,
                    "v": 1000000,
                    "t": 1609459200000,  # 2021-01-01 00:00:00 UTC
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(provider, "_client") as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)

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
    async def test_rate_limiter(self, provider):
        """Test rate limiter functionality."""
        # Should allow requests within limit
        for _ in range(5):
            await provider._rate_limiter.acquire()

        # The 6th request should be delayed (but we won't wait for it)
        # Just verify the rate limiter is working
        assert len(provider._rate_limiter.requests) == 5

    def test_initialization_without_api_key(self):
        """Test that initialization without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Polygon API key is required"):
                PolygonDataProvider()

    @pytest.mark.asyncio
    async def test_stream_methods_not_implemented(self, provider):
        """Test that streaming methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            async for _ in provider.stream_prices(["AAPL"]):
                pass

        with pytest.raises(NotImplementedError):
            async for _ in provider.stream_quotes(["AAPL"]):
                pass
