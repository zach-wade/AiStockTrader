# File: src/main/events/scanner_bridge_helpers/alert_feature_mapper.py

# Standard library imports
import os
from typing import Any

# Third-party imports
import yaml

# Local imports
from main.utils.core import ErrorHandlingMixin, get_logger
from main.utils.monitoring import record_metric

logger = get_logger(__name__)


class AlertFeatureMapper(ErrorHandlingMixin):
    """
    Maps scanner alert types to the specific feature sets required for computation.
    Also provides a list of all available features.

    Now loads mappings from configuration file instead of hardcoding.
    """

    def __init__(self, config_path: str | None = None):
        """Initializes the AlertFeatureMapper from configuration file."""
        super().__init__()

        # Load configuration
        if config_path is None:
            # Go up to ai_trader directory
            current_dir = os.path.dirname(__file__)  # scanner_bridge_helpers
            events_handlers_dir = os.path.dirname(current_dir)  # handlers
            events_dir = os.path.dirname(events_handlers_dir)  # events
            main_dir = os.path.dirname(events_dir)  # main
            src_dir = os.path.dirname(main_dir)  # src
            base_dir = os.path.dirname(src_dir)  # ai_trader
            config_path = os.path.join(
                base_dir, "src", "config", "events", "alert_feature_mappings.yaml"
            )

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            self._alert_feature_map = config["alert_feature_mappings"]
            self._all_features_list = config["all_features_list"]
            logger.info(f"AlertFeatureMapper initialized from {config_path}")
            logger.debug(f"Loaded {len(self._alert_feature_map)} alert type mappings")
        except Exception as e:
            self.handle_error(e, "loading alert feature mappings")
            # Provide defaults if config loading fails
            self._alert_feature_map = {
                "default": ["price_features", "volume_features", "volatility_features"]
            }
            self._all_features_list = ["price_features", "volume_features", "volatility_features"]
            logger.warning("Using default alert feature mappings")

    def reload_config(self, config_path: str | None = None) -> None:
        """
        Reload configuration from file.
        Useful for updating mappings without restarting.
        """
        if config_path is None:
            # Go up to ai_trader directory
            current_dir = os.path.dirname(__file__)  # scanner_bridge_helpers
            events_handlers_dir = os.path.dirname(current_dir)  # handlers
            events_dir = os.path.dirname(events_handlers_dir)  # events
            main_dir = os.path.dirname(events_dir)  # main
            src_dir = os.path.dirname(main_dir)  # src
            base_dir = os.path.dirname(src_dir)  # ai_trader
            config_path = os.path.join(
                base_dir, "src", "config", "events", "alert_feature_mappings.yaml"
            )

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            self._alert_feature_map = config["alert_feature_mappings"]
            self._all_features_list = config["all_features_list"]
            logger.info(f"Reloaded alert feature mappings from {config_path}")
            logger.debug(f"Loaded {len(self._alert_feature_map)} alert type mappings")
        except Exception as e:
            self.handle_error(e, "reloading alert feature mappings")
            logger.warning("Failed to reload config, keeping existing mappings")

    def get_features_for_alert_type(self, alert_type: str) -> list[str]:
        """
        Retrieves the list of feature sets associated with a given alert type.
        Returns default features if the alert type is not explicitly mapped.

        Args:
            alert_type: The type of scanner alert (string).

        Returns:
            A list of strings representing the feature sets to compute.
        """
        features = self._alert_feature_map.get(alert_type, self._alert_feature_map["default"])

        # Record alert type lookup
        record_metric(
            "alert_feature_mapper.lookup",
            1,
            tags={
                "alert_type": alert_type,
                "is_default": alert_type not in self._alert_feature_map,
            },
        )

        # Expand 'all_features' keyword if present
        if "all_features" in features:
            logger.debug(
                f"Alert type '{alert_type}' requested 'all_features'. Expanding to full list."
            )
            record_metric(
                "alert_feature_mapper.all_features_requested", 1, tags={"alert_type": alert_type}
            )
            return self._all_features_list

        return features

    def get_all_features(self) -> list[str]:
        """Returns the comprehensive list of all available feature sets."""
        return list(self._all_features_list)  # Return a copy

    def map_alert_to_features(self, alert: Any) -> list[str]:
        """
        Maps a ScanAlert to the required feature groups.

        Args:
            alert: ScanAlert object with alert_type attribute

        Returns:
            List of feature groups to compute
        """
        # Extract alert type - handle both string and enum
        if hasattr(alert, "alert_type"):
            alert_type = alert.alert_type
            # If it's an enum, get the value
            if hasattr(alert_type, "value"):
                alert_type = alert_type.value
        else:
            logger.warning(f"Alert missing alert_type attribute: {alert}")
            return self._alert_feature_map.get("default", [])

        return self.get_features_for_alert_type(alert_type)

    def get_supported_alert_types(self) -> list[str]:
        """
        Returns a list of all explicitly mapped alert types.

        Returns:
            List of alert type strings that have explicit mappings
        """
        # Exclude 'default' from the list
        return [
            alert_type for alert_type in self._alert_feature_map.keys() if alert_type != "default"
        ]

    def is_alert_type_supported(self, alert_type: str) -> bool:
        """
        Check if an alert type has an explicit mapping.

        Args:
            alert_type: The alert type to check

        Returns:
            True if the alert type has an explicit mapping, False otherwise
        """
        return alert_type in self._alert_feature_map and alert_type != "default"
