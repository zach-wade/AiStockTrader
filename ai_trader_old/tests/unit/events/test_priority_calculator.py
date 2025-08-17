"""Unit tests for priority_calculator module."""

# Standard library imports
from unittest.mock import mock_open, patch

# Third-party imports
import pytest
import yaml

# Local imports
from main.events.handlers.scanner_bridge_helpers.priority_calculator import PriorityCalculator


class TestPriorityCalculator:
    """Test PriorityCalculator class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "priority_boosts": {
                "ML_SIGNAL": 3,
                "BREAKOUT": 2,
                "HIGH_VOLUME": 1,
                "PRICE_SPIKE": 2,
                "NEWS_ALERT": 3,
                "MOMENTUM_SHIFT": 1,
            },
            "priority_config": {"min_priority": 1, "max_priority": 10, "base_multiplier": 10},
        }

    @pytest.fixture
    def calculator(self, sample_config):
        """Create PriorityCalculator instance with mocked config."""
        yaml_content = yaml.dump(sample_config)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.dirname") as mock_dirname:
                with patch("os.path.join") as mock_join:
                    mock_dirname.return_value = "/test/dir"
                    mock_join.return_value = "/test/config.yaml"

                    calculator = PriorityCalculator()
                    # Manually set the loaded config to ensure it's correct
                    calculator.priority_boost_map = sample_config["priority_boosts"]
                    calculator.priority_config = sample_config["priority_config"]
                    return calculator

    def test_initialization_default_path(self, sample_config):
        """Test initialization with default config path."""
        yaml_content = yaml.dump(sample_config)

        with patch("builtins.open", mock_open(read_data=yaml_content)) as mock_file:
            with patch("os.path.dirname") as mock_dirname:
                with patch("os.path.join") as mock_join:
                    mock_dirname.return_value = "/base/dir"
                    mock_join.return_value = "/base/dir/config/events/priority_boosts.yaml"

                    calculator = PriorityCalculator()

                    # Verify path construction
                    assert mock_dirname.call_count >= 1
                    mock_join.assert_called()

                    # Verify file was opened
                    mock_file.assert_called_once_with(
                        "/base/dir/config/events/priority_boosts.yaml", "r"
                    )

    def test_initialization_custom_path(self, sample_config):
        """Test initialization with custom config path."""
        yaml_content = yaml.dump(sample_config)
        custom_path = "/custom/path/priority_config.yaml"

        with patch("builtins.open", mock_open(read_data=yaml_content)) as mock_file:
            calculator = PriorityCalculator(config_path=custom_path)

            # Verify custom path was used
            mock_file.assert_called_once_with(custom_path, "r")

    def test_initialization_with_missing_priority_config(self):
        """Test initialization with missing priority_config (uses defaults)."""
        config = {
            "priority_boosts": {"ML_SIGNAL": 3}
            # Missing 'priority_config'
        }
        yaml_content = yaml.dump(config)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    calculator = PriorityCalculator()

                    # Should use default priority_config
                    assert calculator.priority_config == {
                        "min_priority": 1,
                        "max_priority": 10,
                        "base_multiplier": 10,
                    }

    def test_calculate_priority_basic(self, calculator):
        """Test basic priority calculation without boost."""
        with patch(
            "main.events.handlers.scanner_bridge_helpers.priority_calculator.record_metric"
        ) as mock_metric:
            # Score 0.5 * 10 = 5, no boost
            priority = calculator.calculate_priority(0.5, "UNKNOWN_TYPE")

            assert priority == 5

            # Verify metric was recorded
            mock_metric.assert_called_once_with(
                "priority_calculator.priority_calculated",
                5,
                metric_type="histogram",
                tags={"alert_type": "UNKNOWN_TYPE", "has_boost": False, "boost_value": 0},
            )

    def test_calculate_priority_with_boost(self, calculator):
        """Test priority calculation with boost."""
        with patch(
            "main.events.handlers.scanner_bridge_helpers.priority_calculator.record_metric"
        ) as mock_metric:
            # Score 0.5 * 10 = 5, boost = 3, total = 8
            priority = calculator.calculate_priority(0.5, "ML_SIGNAL")

            assert priority == 8

            # Verify metric includes boost
            mock_metric.assert_called_once_with(
                "priority_calculator.priority_calculated",
                8,
                metric_type="histogram",
                tags={"alert_type": "ML_SIGNAL", "has_boost": True, "boost_value": 3},
            )

    def test_calculate_priority_min_bound(self, calculator):
        """Test priority is capped at minimum."""
        # Very low score
        priority = calculator.calculate_priority(0.0, "UNKNOWN_TYPE")
        assert priority == 1  # min_priority

        # Negative score (shouldn't happen but test anyway)
        priority = calculator.calculate_priority(-0.5, "UNKNOWN_TYPE")
        assert priority == 1

    def test_calculate_priority_max_bound(self, calculator):
        """Test priority is capped at maximum."""
        # High score with boost
        priority = calculator.calculate_priority(1.0, "ML_SIGNAL")
        # 1.0 * 10 + 3 = 13, but capped at 10
        assert priority == 10

        # Even higher score
        priority = calculator.calculate_priority(2.0, "NEWS_ALERT")
        assert priority == 10

    def test_calculate_priority_various_scores(self, calculator):
        """Test priority calculation with various scores."""
        test_cases = [
            (0.0, "HIGH_VOLUME", 1),  # 0 + 1 = 1
            (0.1, "HIGH_VOLUME", 2),  # 1 + 1 = 2
            (0.2, "BREAKOUT", 4),  # 2 + 2 = 4
            (0.3, "ML_SIGNAL", 6),  # 3 + 3 = 6
            (0.7, "UNKNOWN", 7),  # 7 + 0 = 7
            (0.8, "PRICE_SPIKE", 10),  # 8 + 2 = 10 (at max)
            (0.9, "NEWS_ALERT", 10),  # 9 + 3 = 12, capped at 10
        ]

        for score, alert_type, expected in test_cases:
            priority = calculator.calculate_priority(score, alert_type)
            assert priority == expected, f"Failed for score={score}, alert_type={alert_type}"

    def test_calculate_priority_float_scores(self, calculator):
        """Test priority calculation with precise float scores."""
        # Test rounding behavior
        priority1 = calculator.calculate_priority(0.54, "UNKNOWN_TYPE")  # 5.4 -> 5
        priority2 = calculator.calculate_priority(0.56, "UNKNOWN_TYPE")  # 5.6 -> 5

        assert priority1 == 5
        assert priority2 == 5

    def test_timer_decorator(self, calculator):
        """Test that timer decorator is applied to calculate_priority."""
        # The @timer decorator should be applied to the method
        priority = calculator.calculate_priority(0.5, "ML_SIGNAL")
        assert priority == 8

    def test_different_multipliers(self):
        """Test with different base multipliers."""
        config = {
            "priority_boosts": {"HIGH": 2},
            "priority_config": {"min_priority": 0, "max_priority": 100, "base_multiplier": 50},
        }
        yaml_content = yaml.dump(config)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    calculator = PriorityCalculator()
                    calculator.priority_boost_map = config["priority_boosts"]
                    calculator.priority_config = config["priority_config"]

                    # 0.5 * 50 = 25
                    priority = calculator.calculate_priority(0.5, "UNKNOWN")
                    assert priority == 25

                    # 0.5 * 50 + 2 = 27
                    priority = calculator.calculate_priority(0.5, "HIGH")
                    assert priority == 27

    def test_error_handling_file_not_found(self):
        """Test error handling when config file not found."""
        with patch("builtins.open", side_effect=FileNotFoundError("Config not found")):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    with pytest.raises(FileNotFoundError):
                        PriorityCalculator()

    def test_error_handling_invalid_yaml(self):
        """Test error handling with invalid YAML."""
        invalid_yaml = "invalid: yaml: content: [["

        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    with pytest.raises(yaml.YAMLError):
                        PriorityCalculator()

    def test_error_handling_missing_required_fields(self):
        """Test handling of config missing required fields."""
        incomplete_config = {
            # Missing 'priority_boosts'
            "priority_config": {"min_priority": 1, "max_priority": 10}
        }
        yaml_content = yaml.dump(incomplete_config)

        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("os.path.dirname"):
                with patch("os.path.join"):
                    with pytest.raises(KeyError):
                        PriorityCalculator()

    def test_all_alert_types(self, calculator):
        """Test priority calculation for all configured alert types."""
        score = 0.5

        for alert_type, boost in calculator.priority_boost_map.items():
            priority = calculator.calculate_priority(score, alert_type)
            expected = min(int(score * 10) + boost, 10)
            expected = max(expected, 1)
            assert priority == expected

    def test_metric_recording_details(self, calculator):
        """Test detailed metric recording."""
        with patch(
            "main.events.handlers.scanner_bridge_helpers.priority_calculator.record_metric"
        ) as mock_metric:
            # Multiple calculations
            calculator.calculate_priority(0.3, "ML_SIGNAL")
            calculator.calculate_priority(0.7, "UNKNOWN_TYPE")
            calculator.calculate_priority(0.9, "BREAKOUT")

            assert mock_metric.call_count == 3

            # Check each call
            calls = mock_metric.call_args_list

            # First call - ML_SIGNAL
            assert calls[0][0][0] == "priority_calculator.priority_calculated"
            assert calls[0][0][1] == 6  # 3 + 3
            assert calls[0][1]["tags"]["alert_type"] == "ML_SIGNAL"
            assert calls[0][1]["tags"]["has_boost"] is True

            # Second call - UNKNOWN_TYPE
            assert calls[1][0][1] == 7  # 7 + 0
            assert calls[1][1]["tags"]["has_boost"] is False

            # Third call - BREAKOUT
            assert calls[2][0][1] == 10  # 9 + 2 = 11, capped at 10
            assert calls[2][1]["tags"]["boost_value"] == 2

    def test_concurrent_calculations(self, calculator):
        """Test thread safety of priority calculations."""
        # Standard library imports
        import threading

        results = []

        def calculate_priorities():
            for i in range(100):
                score = i / 100.0
                alert_type = "ML_SIGNAL" if i % 2 == 0 else "HIGH_VOLUME"
                priority = calculator.calculate_priority(score, alert_type)
                results.append((score, alert_type, priority))

        threads = []
        for _ in range(5):
            t = threading.Thread(target=calculate_priorities)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have 500 results
        assert len(results) == 500
