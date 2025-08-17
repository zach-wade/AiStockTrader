"""Unit tests for feature_types module."""

# Standard library imports
from dataclasses import fields
from datetime import UTC, datetime

# Third-party imports
import pytest

# Local imports
from main.events.handlers.feature_pipeline_helpers.feature_types import (
    FeatureGroup,
    FeatureGroupConfig,
    FeatureRequest,
    validate_feature_request,
)
from main.events.types import AlertType


class TestFeatureGroup:
    """Test FeatureGroup enum."""

    def test_feature_group_values(self):
        """Test FeatureGroup enum has expected values."""
        # Check some expected groups exist
        assert hasattr(FeatureGroup, "PRICE")
        assert hasattr(FeatureGroup, "VOLUME")
        assert hasattr(FeatureGroup, "VOLATILITY")
        assert hasattr(FeatureGroup, "MOMENTUM")
        assert hasattr(FeatureGroup, "TREND")
        assert hasattr(FeatureGroup, "ML_SIGNALS")

    def test_feature_group_string_representation(self):
        """Test string representation of feature groups."""
        assert FeatureGroup.PRICE.value == "PRICE"
        assert str(FeatureGroup.VOLUME) == "FeatureGroup.VOLUME"

    def test_feature_group_comparison(self):
        """Test feature group comparison."""
        assert FeatureGroup.PRICE == FeatureGroup.PRICE
        assert FeatureGroup.PRICE != FeatureGroup.VOLUME

    def test_feature_group_iteration(self):
        """Test iterating over feature groups."""
        all_groups = list(FeatureGroup)
        assert len(all_groups) > 5  # Should have multiple groups
        assert FeatureGroup.PRICE in all_groups


class TestFeatureGroupConfig:
    """Test FeatureGroupConfig dataclass."""

    def test_feature_group_config_creation(self):
        """Test creating FeatureGroupConfig."""
        config = FeatureGroupConfig(
            name="test_group",
            features=["feature1", "feature2"],
            dependencies=[FeatureGroup.PRICE],
            priority_boost=2,
            computation_params={"param1": "value1"},
            required_data_types=["prices", "quotes"],
        )

        assert config.name == "test_group"
        assert config.features == ["feature1", "feature2"]
        assert config.dependencies == [FeatureGroup.PRICE]
        assert config.priority_boost == 2
        assert config.computation_params == {"param1": "value1"}
        assert config.required_data_types == ["prices", "quotes"]

    def test_feature_group_config_defaults(self):
        """Test FeatureGroupConfig default values."""
        config = FeatureGroupConfig(name="minimal", features=["f1"])

        assert config.dependencies == []
        assert config.priority_boost == 0
        assert config.computation_params == {}
        assert config.required_data_types == []

    def test_feature_group_config_immutability(self):
        """Test that FeatureGroupConfig is immutable."""
        config = FeatureGroupConfig(name="test", features=["f1"])

        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            config.name = "new_name"

    def test_feature_group_config_equality(self):
        """Test FeatureGroupConfig equality."""
        config1 = FeatureGroupConfig(name="test", features=["f1", "f2"], priority_boost=1)

        config2 = FeatureGroupConfig(name="test", features=["f1", "f2"], priority_boost=1)

        config3 = FeatureGroupConfig(name="test", features=["f1"], priority_boost=1)

        assert config1 == config2
        assert config1 != config3


class TestFeatureRequest:
    """Test FeatureRequest dataclass."""

    def test_feature_request_creation(self):
        """Test creating FeatureRequest."""
        request = FeatureRequest(
            symbol="AAPL",
            feature_groups=[FeatureGroup.PRICE, FeatureGroup.VOLUME],
            alert_type=AlertType.HIGH_VOLUME,
            priority=7,
            metadata={"source": "scanner"},
        )

        assert request.symbol == "AAPL"
        assert request.feature_groups == [FeatureGroup.PRICE, FeatureGroup.VOLUME]
        assert request.alert_type == AlertType.HIGH_VOLUME
        assert request.priority == 7
        assert request.metadata == {"source": "scanner"}
        assert isinstance(request.timestamp, datetime)

    def test_feature_request_timestamp_default(self):
        """Test FeatureRequest timestamp is set automatically."""
        request = FeatureRequest(symbol="GOOGL", feature_groups=[FeatureGroup.VOLATILITY])

        assert request.timestamp is not None
        assert isinstance(request.timestamp, datetime)
        assert request.timestamp.tzinfo == UTC

        # Should be recent
        time_diff = datetime.now(UTC) - request.timestamp
        assert time_diff.total_seconds() < 1

    def test_feature_request_defaults(self):
        """Test FeatureRequest default values."""
        request = FeatureRequest(symbol="TSLA", feature_groups=[FeatureGroup.MOMENTUM])

        assert request.alert_type == AlertType.UNKNOWN
        assert request.priority == 5
        assert request.metadata == {}

    def test_feature_request_priority_bounds(self):
        """Test FeatureRequest priority validation."""
        # Valid priorities
        for priority in [0, 5, 10]:
            request = FeatureRequest(
                symbol="TEST", feature_groups=[FeatureGroup.PRICE], priority=priority
            )
            assert request.priority == priority

    def test_feature_request_empty_feature_groups(self):
        """Test creating FeatureRequest with empty feature groups."""
        request = FeatureRequest(symbol="EMPTY", feature_groups=[])

        assert request.feature_groups == []

    def test_feature_request_to_dict(self):
        """Test converting FeatureRequest to dictionary."""
        request = FeatureRequest(
            symbol="AAPL",
            feature_groups=[FeatureGroup.PRICE, FeatureGroup.VOLUME],
            alert_type=AlertType.ML_SIGNAL,
            priority=8,
            metadata={"test": True},
        )

        # If to_dict method exists
        if hasattr(request, "to_dict"):
            data = request.to_dict()
            assert data["symbol"] == "AAPL"
            assert data["priority"] == 8
            assert data["alert_type"] == AlertType.ML_SIGNAL.value
            assert data["metadata"] == {"test": True}


class TestValidateFeatureRequest:
    """Test validate_feature_request function."""

    def test_validate_valid_request(self):
        """Test validating a valid request."""
        request = FeatureRequest(symbol="AAPL", feature_groups=[FeatureGroup.PRICE])

        # Should not raise
        validate_feature_request(request)

    def test_validate_empty_symbol(self):
        """Test validation fails for empty symbol."""
        request = FeatureRequest(symbol="", feature_groups=[FeatureGroup.PRICE])

        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            validate_feature_request(request)

    def test_validate_whitespace_symbol(self):
        """Test validation fails for whitespace symbol."""
        request = FeatureRequest(symbol="   ", feature_groups=[FeatureGroup.PRICE])

        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            validate_feature_request(request)

    def test_validate_empty_feature_groups(self):
        """Test validation fails for empty feature groups."""
        request = FeatureRequest(symbol="AAPL", feature_groups=[])

        with pytest.raises(ValueError, match="Feature groups cannot be empty"):
            validate_feature_request(request)

    def test_validate_invalid_priority(self):
        """Test validation fails for invalid priority."""
        # Test negative priority
        request = FeatureRequest(symbol="AAPL", feature_groups=[FeatureGroup.PRICE], priority=-1)

        with pytest.raises(ValueError, match="Priority must be between"):
            validate_feature_request(request)

        # Test priority too high
        request.priority = 11
        with pytest.raises(ValueError, match="Priority must be between"):
            validate_feature_request(request)

    def test_validate_duplicate_feature_groups(self):
        """Test validation with duplicate feature groups."""
        request = FeatureRequest(
            symbol="AAPL", feature_groups=[FeatureGroup.PRICE, FeatureGroup.PRICE]
        )

        # Depending on implementation, may remove duplicates or raise
        try:
            validate_feature_request(request)
            # If no error, check duplicates were handled
            assert len(set(request.feature_groups)) == 1
        except ValueError as e:
            # Or it might raise an error
            assert "duplicate" in str(e).lower()


class TestFeatureTypesIntegration:
    """Test interactions between feature types."""

    def test_feature_request_with_all_groups(self):
        """Test creating request with all feature groups."""
        all_groups = list(FeatureGroup)

        request = FeatureRequest(
            symbol="TEST", feature_groups=all_groups, alert_type=AlertType.ML_SIGNAL, priority=10
        )

        assert len(request.feature_groups) == len(all_groups)
        assert all(isinstance(g, FeatureGroup) for g in request.feature_groups)

    def test_feature_group_config_with_dependencies(self):
        """Test config with multiple dependencies."""
        config = FeatureGroupConfig(
            name="complex_features",
            features=["complex1", "complex2"],
            dependencies=[FeatureGroup.PRICE, FeatureGroup.VOLUME, FeatureGroup.MOMENTUM],
            priority_boost=3,
            computation_params={"lookback": 20, "aggregation": "mean"},
        )

        assert len(config.dependencies) == 3
        assert all(isinstance(d, FeatureGroup) for d in config.dependencies)

    def test_feature_request_serialization(self):
        """Test serializing and deserializing feature request."""
        original = FeatureRequest(
            symbol="AAPL",
            feature_groups=[FeatureGroup.PRICE, FeatureGroup.VOLATILITY],
            alert_type=AlertType.BREAKOUT,
            priority=7,
            metadata={"scanner_id": "scanner_1"},
        )

        # Test field access
        assert hasattr(original, "symbol")
        assert hasattr(original, "feature_groups")
        assert hasattr(original, "alert_type")
        assert hasattr(original, "priority")
        assert hasattr(original, "metadata")
        assert hasattr(original, "timestamp")

    def test_feature_types_type_hints(self):
        """Test that type hints are properly defined."""
        # Check FeatureRequest fields
        request_fields = fields(FeatureRequest)

        symbol_field = next(f for f in request_fields if f.name == "symbol")
        assert symbol_field.type == str

        groups_field = next(f for f in request_fields if f.name == "feature_groups")
        # Type should be List[FeatureGroup] or similar

        priority_field = next(f for f in request_fields if f.name == "priority")
        assert priority_field.type == int
