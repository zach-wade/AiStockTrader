"""
Unit tests for DataProviderFactory
"""

import os
from unittest.mock import patch

import pytest

from src.infrastructure.data.providers import DataProviderFactory
from src.infrastructure.data.providers.alpaca_provider import AlpacaDataProvider
from src.infrastructure.data.providers.polygon_provider import PolygonDataProvider


class TestDataProviderFactory:
    """Test suite for DataProviderFactory."""

    def test_create_provider_auto_detect_polygon(self):
        """Test auto-detection prefers Polygon when available."""
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            provider = DataProviderFactory.create_provider()
            assert isinstance(provider, PolygonDataProvider)

    def test_create_provider_auto_detect_alpaca(self):
        """Test auto-detection falls back to Alpaca."""
        with patch.dict(
            os.environ,
            {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"},
            clear=True,
        ):
            provider = DataProviderFactory.create_provider()
            assert isinstance(provider, AlpacaDataProvider)

    def test_create_provider_explicit_polygon(self):
        """Test explicit Polygon provider creation."""
        with patch.dict(os.environ, {"POLYGON_API_KEY": "test_key"}):
            provider = DataProviderFactory.create_provider("polygon")
            assert isinstance(provider, PolygonDataProvider)

    def test_create_provider_explicit_alpaca(self):
        """Test explicit Alpaca provider creation."""
        with patch.dict(
            os.environ, {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"}
        ):
            provider = DataProviderFactory.create_provider("alpaca")
            assert isinstance(provider, AlpacaDataProvider)

    def test_create_provider_no_credentials_raises_error(self):
        """Test that missing credentials raises an error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No data provider credentials found"):
                DataProviderFactory.create_provider()

    def test_create_provider_invalid_type_raises_error(self):
        """Test that invalid provider type raises an error."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            DataProviderFactory.create_provider("invalid")
