"""
Comprehensive unit tests for BrokerFactory implementation.

Tests cover:
- Factory initialization
- Broker creation for different types
- Configuration handling
- Environment variable usage
- Auto-connection functionality
- Error handling
"""

import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.domain.services.broker_configuration_service import BrokerType
from src.domain.services.trading_calendar import Exchange
from src.infrastructure.brokers.broker_factory import BrokerFactory


class TestBrokerFactoryInitialization:
    """Test BrokerFactory initialization."""

    def test_initialization_default(self):
        """Test factory initialization with defaults."""
        factory = BrokerFactory()

        assert factory.config_service is not None

    def test_initialization_with_config_service(self):
        """Test factory initialization with custom config service."""
        mock_service = MagicMock()
        factory = BrokerFactory(config_service=mock_service)

        assert factory.config_service == mock_service


class TestBrokerFactoryCreateBroker:
    """Test create_broker method."""

    @pytest.fixture
    def factory(self):
        """Create factory with mock config service."""
        mock_service = MagicMock()
        return BrokerFactory(config_service=mock_service)

    @patch("src.infrastructure.brokers.broker_factory.AlpacaBroker")
    def test_create_alpaca_broker(self, mock_alpaca_class, factory):
        """Test creating Alpaca broker."""
        factory.config_service.determine_broker_type.return_value = BrokerType.ALPACA
        factory.config_service.determine_paper_mode.return_value = True

        mock_broker = MagicMock()
        mock_alpaca_class.return_value = mock_broker

        with patch.dict(
            os.environ,
            {
                "ALPACA_API_KEY": "test_key",
                "ALPACA_SECRET_KEY": "test_secret",
                "ALPACA_PAPER": "true",
            },
        ):
            broker = factory.create_broker("alpaca")

        assert broker == mock_broker
        mock_alpaca_class.assert_called_once_with(
            api_key="test_key", secret_key="test_secret", paper=True
        )
        mock_broker.connect.assert_called_once()

    @patch("src.infrastructure.brokers.broker_factory.AlpacaBroker")
    def test_create_alpaca_broker_with_kwargs(self, mock_alpaca_class, factory):
        """Test creating Alpaca broker with kwargs."""
        factory.config_service.determine_broker_type.return_value = BrokerType.ALPACA
        factory.config_service.determine_paper_mode.return_value = False

        mock_broker = MagicMock()
        mock_alpaca_class.return_value = mock_broker

        broker = factory.create_broker(
            "alpaca",
            api_key="custom_key",
            secret_key="custom_secret",
            paper=False,
            auto_connect=False,
        )

        assert broker == mock_broker
        mock_alpaca_class.assert_called_once_with(
            api_key="custom_key", secret_key="custom_secret", paper=False
        )
        mock_broker.connect.assert_not_called()

    @patch("src.infrastructure.brokers.broker_factory.PaperBroker")
    def test_create_paper_broker(self, mock_paper_class, factory):
        """Test creating paper broker."""
        factory.config_service.determine_broker_type.return_value = BrokerType.PAPER
        factory.config_service.normalize_initial_capital.return_value = Decimal("100000")

        mock_broker = MagicMock()
        mock_paper_class.return_value = mock_broker

        broker = factory.create_broker("paper")

        assert broker == mock_broker
        mock_paper_class.assert_called_once_with(
            initial_capital=Decimal("100000"), exchange=Exchange.NYSE
        )
        mock_broker.connect.assert_called_once()

    @patch("src.infrastructure.brokers.broker_factory.PaperBroker")
    def test_create_paper_broker_with_kwargs(self, mock_paper_class, factory):
        """Test creating paper broker with kwargs."""
        factory.config_service.determine_broker_type.return_value = BrokerType.PAPER
        factory.config_service.normalize_initial_capital.return_value = Decimal("250000")

        mock_broker = MagicMock()
        mock_paper_class.return_value = mock_broker

        broker = factory.create_broker(
            "paper", initial_capital=Decimal("250000"), exchange=Exchange.NASDAQ, auto_connect=False
        )

        assert broker == mock_broker
        mock_paper_class.assert_called_once_with(
            initial_capital=Decimal("250000"), exchange=Exchange.NASDAQ
        )
        mock_broker.connect.assert_not_called()

    @patch("src.infrastructure.brokers.broker_factory.PaperBroker")
    def test_create_backtest_broker(self, mock_paper_class, factory):
        """Test creating backtest broker."""
        factory.config_service.determine_broker_type.return_value = BrokerType.BACKTEST
        factory.config_service.normalize_initial_capital.return_value = Decimal("100000")

        mock_broker = MagicMock()
        mock_paper_class.return_value = mock_broker

        broker = factory.create_broker("backtest")

        assert broker == mock_broker
        # Backtest uses paper broker internally
        mock_paper_class.assert_called_once_with(
            initial_capital=Decimal("100000"), exchange=Exchange.NYSE
        )
        mock_broker.connect.assert_called_once()

    @patch.dict(os.environ, {"BROKER_TYPE": "paper"})
    @patch("src.infrastructure.brokers.broker_factory.PaperBroker")
    def test_create_broker_from_environment(self, mock_paper_class, factory):
        """Test creating broker from environment variable."""
        factory.config_service.determine_broker_type.return_value = BrokerType.PAPER
        factory.config_service.normalize_initial_capital.return_value = Decimal("100000")

        mock_broker = MagicMock()
        mock_paper_class.return_value = mock_broker

        # No broker_type argument, should use env var
        broker = factory.create_broker()

        assert broker == mock_broker
        factory.config_service.determine_broker_type.assert_called_with("paper", "paper")

    def test_create_broker_invalid_type(self, factory):
        """Test creating broker with invalid type."""
        # Mock service to return unexpected enum value
        factory.config_service.determine_broker_type.return_value = MagicMock()

        with pytest.raises(ValueError, match="Unexpected broker type"):
            factory.create_broker("invalid")


class TestBrokerFactoryEnvironmentIntegration:
    """Test environment variable integration."""

    @pytest.fixture
    def factory(self):
        """Create factory with mock config service."""
        mock_service = MagicMock()
        return BrokerFactory(config_service=mock_service)

    @patch.dict(
        os.environ,
        {"ALPACA_API_KEY": "env_key", "ALPACA_SECRET_KEY": "env_secret", "ALPACA_PAPER": "false"},
    )
    @patch("src.infrastructure.brokers.broker_factory.AlpacaBroker")
    def test_alpaca_broker_env_vars(self, mock_alpaca_class, factory):
        """Test Alpaca broker uses environment variables."""
        factory.config_service.determine_broker_type.return_value = BrokerType.ALPACA
        factory.config_service.determine_paper_mode.return_value = False

        mock_broker = MagicMock()
        mock_alpaca_class.return_value = mock_broker

        broker = factory.create_broker("alpaca")

        mock_alpaca_class.assert_called_once_with(
            api_key="env_key", secret_key="env_secret", paper=False
        )

    @patch.dict(os.environ, {"PAPER_INITIAL_CAPITAL": "500000"})
    @patch("src.infrastructure.brokers.broker_factory.PaperBroker")
    def test_paper_broker_env_vars(self, mock_paper_class, factory):
        """Test paper broker uses environment variables."""
        factory.config_service.determine_broker_type.return_value = BrokerType.PAPER
        factory.config_service.normalize_initial_capital.return_value = Decimal("500000")

        mock_broker = MagicMock()
        mock_paper_class.return_value = mock_broker

        broker = factory.create_broker("paper")

        factory.config_service.normalize_initial_capital.assert_called_with("500000")
        mock_paper_class.assert_called_once_with(
            initial_capital=Decimal("500000"), exchange=Exchange.NYSE
        )


class TestBrokerFactoryCreateFromConfig:
    """Test create_from_config method."""

    @pytest.fixture
    def factory(self):
        """Create factory with mock config service."""
        mock_service = MagicMock()
        return BrokerFactory(config_service=mock_service)

    @patch("src.infrastructure.brokers.broker_factory.AlpacaBroker")
    def test_create_from_config_alpaca(self, mock_alpaca_class, factory):
        """Test creating Alpaca broker from config dict."""
        factory.config_service.process_broker_config.return_value = {
            "type": "alpaca",
            "api_key": "config_key",
            "secret_key": "config_secret",
            "paper": True,
            "auto_connect": True,
        }
        factory.config_service.determine_broker_type.return_value = BrokerType.ALPACA
        factory.config_service.determine_paper_mode.return_value = True

        mock_broker = MagicMock()
        mock_alpaca_class.return_value = mock_broker

        config = {
            "type": "alpaca",
            "paper": True,
            "auto_connect": True,
            "api_key": "config_key",
            "secret_key": "config_secret",
        }

        broker = factory.create_from_config(config)

        assert broker == mock_broker
        factory.config_service.process_broker_config.assert_called_once_with(config)
        mock_alpaca_class.assert_called_once()

    @patch("src.infrastructure.brokers.broker_factory.PaperBroker")
    def test_create_from_config_paper(self, mock_paper_class, factory):
        """Test creating paper broker from config dict."""
        factory.config_service.process_broker_config.return_value = {
            "type": "paper",
            "initial_capital": Decimal("750000"),
            "exchange": Exchange.NASDAQ,
            "auto_connect": False,
        }
        factory.config_service.determine_broker_type.return_value = BrokerType.PAPER
        factory.config_service.normalize_initial_capital.return_value = Decimal("750000")

        mock_broker = MagicMock()
        mock_paper_class.return_value = mock_broker

        config = {
            "type": "paper",
            "initial_capital": "750000",
            "exchange": Exchange.NASDAQ,
            "auto_connect": False,
        }

        broker = factory.create_from_config(config)

        assert broker == mock_broker
        mock_paper_class.assert_called_once_with(
            initial_capital=Decimal("750000"), exchange=Exchange.NASDAQ
        )
        mock_broker.connect.assert_not_called()

    def test_create_from_config_no_type(self, factory):
        """Test creating broker from config without type."""
        factory.config_service.process_broker_config.return_value = {"auto_connect": True}
        factory.config_service.determine_broker_type.return_value = BrokerType.PAPER
        factory.config_service.normalize_initial_capital.return_value = Decimal("100000")

        with patch("src.infrastructure.brokers.broker_factory.PaperBroker") as mock_paper:
            mock_broker = MagicMock()
            mock_paper.return_value = mock_broker

            config = {"auto_connect": True}
            broker = factory.create_from_config(config)

            # Should default to paper broker
            assert broker == mock_broker


class TestBrokerFactoryGetDefaultConfig:
    """Test get_default_config method."""

    @pytest.fixture
    def factory(self):
        """Create factory with mock config service."""
        mock_service = MagicMock()
        return BrokerFactory(config_service=mock_service)

    def test_get_default_config_alpaca(self, factory):
        """Test getting default config for Alpaca."""
        factory.config_service.determine_broker_type.return_value = BrokerType.ALPACA
        factory.config_service.get_default_config.return_value = {
            "type": "alpaca",
            "paper": True,
            "auto_connect": True,
        }

        config = factory.get_default_config("alpaca")

        assert config["type"] == "alpaca"
        assert config["paper"] is True
        assert config["auto_connect"] is True
        factory.config_service.determine_broker_type.assert_called_once_with("alpaca")
        factory.config_service.get_default_config.assert_called_once_with(BrokerType.ALPACA)

    def test_get_default_config_paper(self, factory):
        """Test getting default config for paper broker."""
        factory.config_service.determine_broker_type.return_value = BrokerType.PAPER
        factory.config_service.get_default_config.return_value = {
            "type": "paper",
            "initial_capital": "100000",
            "exchange": "NYSE",
            "auto_connect": True,
        }

        config = factory.get_default_config("paper")

        assert config["type"] == "paper"
        assert config["initial_capital"] == "100000"
        assert config["exchange"] == "NYSE"

    def test_get_default_config_backtest(self, factory):
        """Test getting default config for backtest."""
        factory.config_service.determine_broker_type.return_value = BrokerType.BACKTEST
        factory.config_service.get_default_config.return_value = {
            "type": "backtest",
            "initial_capital": "100000",
            "exchange": "NYSE",
        }

        config = factory.get_default_config("backtest")

        assert config["type"] == "backtest"
        assert config["initial_capital"] == "100000"


class TestBrokerFactoryEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def factory(self):
        """Create factory with mock config service."""
        mock_service = MagicMock()
        return BrokerFactory(config_service=mock_service)

    @patch("src.infrastructure.brokers.broker_factory.PaperBroker")
    def test_create_broker_with_none_type(self, mock_paper_class, factory):
        """Test creating broker with None type defaults to paper."""
        factory.config_service.determine_broker_type.return_value = BrokerType.PAPER
        factory.config_service.normalize_initial_capital.return_value = Decimal("100000")

        mock_broker = MagicMock()
        mock_paper_class.return_value = mock_broker

        with patch.dict(os.environ, {}, clear=True):
            broker = factory.create_broker(None)

        assert broker == mock_broker
        factory.config_service.determine_broker_type.assert_called_with(None, "paper")

    def test_create_broker_config_service_exception(self, factory):
        """Test handling config service exceptions."""
        factory.config_service.determine_broker_type.side_effect = ValueError("Invalid type")

        with pytest.raises(ValueError, match="Invalid type"):
            factory.create_broker("bad_type")

    @patch("src.infrastructure.brokers.broker_factory.AlpacaBroker")
    def test_create_alpaca_connection_failure(self, mock_alpaca_class, factory):
        """Test handling connection failure during auto-connect."""
        factory.config_service.determine_broker_type.return_value = BrokerType.ALPACA
        factory.config_service.determine_paper_mode.return_value = True

        mock_broker = MagicMock()
        mock_broker.connect.side_effect = Exception("Connection failed")
        mock_alpaca_class.return_value = mock_broker

        with patch.dict(os.environ, {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "secret"}):
            with pytest.raises(Exception, match="Connection failed"):
                factory.create_broker("alpaca", auto_connect=True)

    def test_create_from_empty_config(self, factory):
        """Test creating broker from empty config."""
        factory.config_service.process_broker_config.return_value = {}
        factory.config_service.determine_broker_type.return_value = BrokerType.PAPER
        factory.config_service.normalize_initial_capital.return_value = Decimal("100000")

        with patch("src.infrastructure.brokers.broker_factory.PaperBroker") as mock_paper:
            mock_broker = MagicMock()
            mock_paper.return_value = mock_broker

            broker = factory.create_from_config({})

            assert broker == mock_broker
            # Should use all defaults
            mock_paper.assert_called_once_with(
                initial_capital=Decimal("100000"), exchange=Exchange.NYSE
            )
