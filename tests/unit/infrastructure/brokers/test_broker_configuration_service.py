"""
Comprehensive unit tests for BrokerConfigurationService.

This test module provides comprehensive coverage for the BrokerConfigurationService,
testing all business logic related to broker configuration and setup.
"""

from decimal import Decimal

import pytest

from src.infrastructure.brokers.broker_configuration_service import (
    BrokerConfigurationService,
    BrokerType,
)


class TestBrokerType:
    """Test BrokerType enum."""

    def test_broker_type_values(self):
        """Test that all broker types have expected values."""
        assert BrokerType.ALPACA.value == "alpaca"
        assert BrokerType.PAPER.value == "paper"
        assert BrokerType.BACKTEST.value == "backtest"

    def test_broker_type_from_string(self):
        """Test creating BrokerType from string value."""
        assert BrokerType("alpaca") == BrokerType.ALPACA
        assert BrokerType("paper") == BrokerType.PAPER
        assert BrokerType("backtest") == BrokerType.BACKTEST

    def test_broker_type_invalid_value(self):
        """Test that invalid broker type raises ValueError."""
        with pytest.raises(ValueError):
            BrokerType("invalid")


class TestDetermineBrokerType:
    """Test determine_broker_type method."""

    def test_determine_broker_type_valid(self):
        """Test determining valid broker types."""
        assert BrokerConfigurationService.determine_broker_type("alpaca") == BrokerType.ALPACA
        assert BrokerConfigurationService.determine_broker_type("PAPER") == BrokerType.PAPER
        assert BrokerConfigurationService.determine_broker_type(" Backtest ") == BrokerType.BACKTEST

    def test_determine_broker_type_none_uses_fallback(self):
        """Test that None uses fallback type."""
        assert BrokerConfigurationService.determine_broker_type(None) == BrokerType.PAPER
        assert (
            BrokerConfigurationService.determine_broker_type(None, "backtest")
            == BrokerType.BACKTEST
        )

    def test_determine_broker_type_case_insensitive(self):
        """Test that broker type is case insensitive."""
        assert BrokerConfigurationService.determine_broker_type("ALPACA") == BrokerType.ALPACA
        assert BrokerConfigurationService.determine_broker_type("Paper") == BrokerType.PAPER
        assert BrokerConfigurationService.determine_broker_type("bAcKtEsT") == BrokerType.BACKTEST

    def test_determine_broker_type_strips_whitespace(self):
        """Test that whitespace is stripped from broker type."""
        assert BrokerConfigurationService.determine_broker_type("  alpaca  ") == BrokerType.ALPACA
        assert BrokerConfigurationService.determine_broker_type("\tpaper\n") == BrokerType.PAPER

    def test_determine_broker_type_invalid(self):
        """Test that invalid broker type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BrokerConfigurationService.determine_broker_type("invalid")
        assert "Unsupported broker type: invalid" in str(exc_info)
        assert "alpaca, paper, backtest" in str(exc_info)

    def test_determine_broker_type_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError):
            BrokerConfigurationService.determine_broker_type("")


class TestNormalizeInitialCapital:
    """Test normalize_initial_capital method."""

    def test_normalize_initial_capital_valid_values(self):
        """Test normalizing valid capital values."""
        assert BrokerConfigurationService.normalize_initial_capital(10000) == Decimal("10000")
        assert BrokerConfigurationService.normalize_initial_capital("50000") == Decimal("50000")
        assert BrokerConfigurationService.normalize_initial_capital(Decimal("25000")) == Decimal(
            "25000"
        )
        assert BrokerConfigurationService.normalize_initial_capital(100.50) == Decimal("100.5")

    def test_normalize_initial_capital_none_uses_default(self):
        """Test that None returns default capital."""
        default = BrokerConfigurationService.DEFAULT_INITIAL_CAPITAL
        assert BrokerConfigurationService.normalize_initial_capital(None) == default

    def test_normalize_initial_capital_zero_raises_error(self):
        """Test that zero capital raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BrokerConfigurationService.normalize_initial_capital(0)
        assert "Initial capital must be positive" in str(exc_info)

    def test_normalize_initial_capital_negative_raises_error(self):
        """Test that negative capital raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BrokerConfigurationService.normalize_initial_capital(-1000)
        assert "Initial capital must be positive" in str(exc_info)

    def test_normalize_initial_capital_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            BrokerConfigurationService.normalize_initial_capital("invalid")
        assert "Invalid initial capital" in str(exc_info)

        with pytest.raises(ValueError):
            BrokerConfigurationService.normalize_initial_capital([1000])

        with pytest.raises(ValueError):
            BrokerConfigurationService.normalize_initial_capital({"amount": 1000})

    def test_normalize_initial_capital_precision(self):
        """Test that decimal precision is maintained."""
        assert BrokerConfigurationService.normalize_initial_capital("100.123456789") == Decimal(
            "100.123456789"
        )
        assert BrokerConfigurationService.normalize_initial_capital(100.99) == Decimal("100.99")


class TestDeterminePaperMode:
    """Test determine_paper_mode method."""

    def test_determine_paper_mode_explicit_values(self):
        """Test explicit paper mode values."""
        assert BrokerConfigurationService.determine_paper_mode(True) is True
        assert BrokerConfigurationService.determine_paper_mode(False) is False

    def test_determine_paper_mode_none_uses_default(self):
        """Test that None uses default value."""
        assert BrokerConfigurationService.determine_paper_mode(None) is True
        assert BrokerConfigurationService.determine_paper_mode(None, True) is True
        assert BrokerConfigurationService.determine_paper_mode(None, False) is False

    def test_determine_paper_mode_converts_to_bool(self):
        """Test that values are converted to boolean."""
        assert BrokerConfigurationService.determine_paper_mode(1) is True
        assert BrokerConfigurationService.determine_paper_mode(0) is False
        assert BrokerConfigurationService.determine_paper_mode("yes") is True
        assert BrokerConfigurationService.determine_paper_mode("") is False
        assert BrokerConfigurationService.determine_paper_mode([]) is False
        assert BrokerConfigurationService.determine_paper_mode([1]) is True


class TestGetDefaultConfig:
    """Test get_default_config method."""

    def test_get_default_config_alpaca(self):
        """Test default config for Alpaca broker."""
        config = BrokerConfigurationService.get_default_config(BrokerType.ALPACA)

        assert config["type"] == "alpaca"
        assert config["paper"] is True
        assert config["auto_connect"] is True
        assert config["api_key"] is None
        assert config["secret_key"] is None

    def test_get_default_config_paper(self):
        """Test default config for Paper broker."""
        config = BrokerConfigurationService.get_default_config(BrokerType.PAPER)

        assert config["type"] == "paper"
        assert config["auto_connect"] is True
        assert config["initial_capital"] == str(BrokerConfigurationService.DEFAULT_INITIAL_CAPITAL)
        assert config["exchange"] == "NYSE"
        assert "api_key" not in config
        assert "secret_key" not in config

    def test_get_default_config_backtest(self):
        """Test default config for Backtest broker."""
        config = BrokerConfigurationService.get_default_config(BrokerType.BACKTEST)

        assert config["type"] == "backtest"
        assert config["auto_connect"] is True
        assert config["initial_capital"] == str(BrokerConfigurationService.DEFAULT_INITIAL_CAPITAL)
        assert config["exchange"] == "NYSE"
        assert "api_key" not in config
        assert "secret_key" not in config

    def test_get_default_config_uses_class_defaults(self):
        """Test that default config uses class constants."""
        # Temporarily modify defaults to verify they're being used
        original_capital = BrokerConfigurationService.DEFAULT_INITIAL_CAPITAL
        original_paper = BrokerConfigurationService.DEFAULT_PAPER_TRADING

        try:
            BrokerConfigurationService.DEFAULT_INITIAL_CAPITAL = Decimal("50000")
            BrokerConfigurationService.DEFAULT_PAPER_TRADING = False

            alpaca_config = BrokerConfigurationService.get_default_config(BrokerType.ALPACA)
            assert alpaca_config["paper"] is False

            paper_config = BrokerConfigurationService.get_default_config(BrokerType.PAPER)
            assert paper_config["initial_capital"] == "50000"

        finally:
            # Restore original values
            BrokerConfigurationService.DEFAULT_INITIAL_CAPITAL = original_capital
            BrokerConfigurationService.DEFAULT_PAPER_TRADING = original_paper


class TestValidateAlpacaConfig:
    """Test validate_alpaca_config method."""

    def test_validate_alpaca_config_valid(self):
        """Test validation with valid Alpaca credentials."""
        assert (
            BrokerConfigurationService.validate_alpaca_config("valid_api_key", "valid_secret_key")
            is True
        )

    def test_validate_alpaca_config_missing_api_key(self):
        """Test validation with missing API key."""
        with pytest.raises(ValueError) as exc_info:
            BrokerConfigurationService.validate_alpaca_config(None, "secret")
        assert "Alpaca API credentials are required" in str(exc_info)

        with pytest.raises(ValueError):
            BrokerConfigurationService.validate_alpaca_config("", "secret")

    def test_validate_alpaca_config_missing_secret_key(self):
        """Test validation with missing secret key."""
        with pytest.raises(ValueError) as exc_info:
            BrokerConfigurationService.validate_alpaca_config("api_key", None)
        assert "Alpaca API credentials are required" in str(exc_info)

        with pytest.raises(ValueError):
            BrokerConfigurationService.validate_alpaca_config("api_key", "")

    def test_validate_alpaca_config_both_missing(self):
        """Test validation with both credentials missing."""
        with pytest.raises(ValueError) as exc_info:
            BrokerConfigurationService.validate_alpaca_config(None, None)
        assert "Alpaca API credentials are required" in str(exc_info)

        with pytest.raises(ValueError):
            BrokerConfigurationService.validate_alpaca_config("", "")


class TestProcessBrokerConfig:
    """Test process_broker_config method."""

    def test_process_broker_config_alpaca(self):
        """Test processing Alpaca broker config."""
        config = {
            "type": "alpaca",
            "paper": True,
            "api_key": "test_key",
            "secret_key": "test_secret",
        }

        processed = BrokerConfigurationService.process_broker_config(config)

        assert processed["type"] == "alpaca"
        assert processed["paper"] is True
        assert processed["api_key"] == "test_key"
        assert processed["secret_key"] == "test_secret"

    def test_process_broker_config_paper(self):
        """Test processing Paper broker config."""
        config = {"type": "paper", "initial_capital": "25000"}

        processed = BrokerConfigurationService.process_broker_config(config)

        assert processed["type"] == "paper"
        assert processed["initial_capital"] == Decimal("25000")

    def test_process_broker_config_backtest(self):
        """Test processing Backtest broker config."""
        config = {"type": "backtest", "initial_capital": 50000}

        processed = BrokerConfigurationService.process_broker_config(config)

        assert processed["type"] == "backtest"
        assert processed["initial_capital"] == Decimal("50000")

    def test_process_broker_config_normalizes_type(self):
        """Test that broker type is normalized."""
        config = {"type": "ALPACA"}
        processed = BrokerConfigurationService.process_broker_config(config)
        assert processed["type"] == "alpaca"

    def test_process_broker_config_preserves_extra_fields(self):
        """Test that extra fields are preserved."""
        config = {
            "type": "paper",
            "initial_capital": "10000",
            "extra_field": "value",
            "another_field": 123,
        }

        processed = BrokerConfigurationService.process_broker_config(config)

        assert processed["extra_field"] == "value"
        assert processed["another_field"] == 123

    def test_process_broker_config_applies_defaults(self):
        """Test that defaults are applied when fields are missing."""
        config = {"type": "alpaca"}
        processed = BrokerConfigurationService.process_broker_config(config)
        assert processed["paper"] is True  # Default value

    def test_process_broker_config_invalid_type(self):
        """Test that invalid broker type raises error."""
        config = {"type": "invalid"}

        with pytest.raises(ValueError) as exc_info:
            BrokerConfigurationService.process_broker_config(config)
        assert "Unsupported broker type" in str(exc_info)

    def test_process_broker_config_invalid_capital(self):
        """Test that invalid capital raises error."""
        config = {"type": "paper", "initial_capital": "invalid"}

        with pytest.raises(ValueError) as exc_info:
            BrokerConfigurationService.process_broker_config(config)
        assert "Invalid initial capital" in str(exc_info)

    def test_process_broker_config_does_not_mutate_original(self):
        """Test that original config is not modified."""
        original = {"type": "paper", "initial_capital": "10000"}
        original_copy = original.copy()

        processed = BrokerConfigurationService.process_broker_config(original)

        # Original should be unchanged
        assert original == original_copy
        # Processed should have Decimal for capital
        assert isinstance(processed["initial_capital"], Decimal)


class TestClassConstants:
    """Test class constants."""

    def test_default_initial_capital(self):
        """Test default initial capital value."""
        assert Decimal("100000") == BrokerConfigurationService.DEFAULT_INITIAL_CAPITAL

    def test_default_paper_trading(self):
        """Test default paper trading value."""
        assert BrokerConfigurationService.DEFAULT_PAPER_TRADING is True

    def test_default_auto_connect(self):
        """Test default auto connect value."""
        assert BrokerConfigurationService.DEFAULT_AUTO_CONNECT is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_initial_capital(self):
        """Test handling of very large capital amounts."""
        large_capital = "999999999999.99"
        result = BrokerConfigurationService.normalize_initial_capital(large_capital)
        assert result == Decimal(large_capital)

    def test_small_initial_capital(self):
        """Test handling of very small capital amounts."""
        small_capital = "0.01"
        result = BrokerConfigurationService.normalize_initial_capital(small_capital)
        assert result == Decimal(small_capital)

    def test_unicode_in_broker_type(self):
        """Test that unicode in broker type is handled."""
        with pytest.raises(ValueError):
            BrokerConfigurationService.determine_broker_type("papeř")

    def test_scientific_notation_capital(self):
        """Test handling of scientific notation in capital."""
        result = BrokerConfigurationService.normalize_initial_capital("1e6")
        assert result == Decimal("1000000")

    def test_process_config_with_none_values(self):
        """Test processing config with None values."""
        config = {"type": "paper", "initial_capital": None, "exchange": None}  # Should use default

        processed = BrokerConfigurationService.process_broker_config(config)
        assert processed["initial_capital"] == BrokerConfigurationService.DEFAULT_INITIAL_CAPITAL
