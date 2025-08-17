# config/field_mappings.py
"""
Field mapping configuration for different data sources.
This centralizes all field mappings and makes them easily configurable.
"""

# Standard library imports
import json
import os
from pathlib import Path


class FieldMappingConfig:
    """Manages field mappings for different data sources."""

    def __init__(self, config_file_path: str | None = None):
        """
        Initialize field mapping configuration.

        Args:
            config_file_path: Optional path to JSON config file.
                            If None, uses default mappings.
        """
        self.config_file_path = config_file_path
        self._mappings = self._load_mappings()

    def _get_default_mappings(self) -> dict[str, dict[str, str]]:
        """Get default field mappings for known data sources."""
        return {
            "yahoo": {
                "Date": "timestamp",
                "adj_close": "adjusted_close",
                "close": "close",
                "volume": "volume",
                "low": "low",
                "open": "open",
                "high": "high",
            },
            "alpha_vantage": {
                "timestamp": "timestamp",
                "datetime": "timestamp",
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            },
            "iex": {
                "date": "timestamp",
                "uOpen": "open",
                "uHigh": "high",
                "uLow": "low",
                "uClose": "close",
                "uVolume": "volume",
            },
            "polygon": {
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            },
            "quandl": {
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            "finnhub": {
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            },
            "tiingo": {
                "date": "timestamp",
                "adjOpen": "open",
                "adjHigh": "high",
                "adjLow": "low",
                "adjClose": "close",
                "adjVolume": "volume",
            },
            "benzinga": {
                # TODO: Fill in after testing Benzinga API
                # Common possibilities:
                # 'date': 'timestamp',
                # 'datetime': 'timestamp',
                # 'time': 'timestamp',
                # These might already be correct:
                # 'open': 'open',
                # 'high': 'high',
                # 'low': 'low',
                # 'close': 'close',
                # 'volume': 'volume'
            },
            "twelve_data": {
                "datetime": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
            "fmp": {  # Financial Modeling Prep
                "date": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
        }

    def _load_mappings(self) -> dict[str, dict[str, str]]:
        """Load field mappings from config file or use defaults."""
        if self.config_file_path and os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load field mappings from {self.config_file_path}: {e}")
                print("Falling back to default mappings.")

        return self._get_default_mappings()

    def get_mapping(self, source: str) -> dict[str, str]:
        """Get field mapping for a specific source."""
        return self._mappings.get(source, {})

    def add_mapping(self, source: str, mapping: dict[str, str]) -> None:
        """Add or update mapping for a source."""
        self._mappings[source] = mapping

    def update_mapping(self, source: str, field_updates: dict[str, str]) -> None:
        """Update specific fields in an existing mapping."""
        if source not in self._mappings:
            self._mappings[source] = {}
        self._mappings[source].update(field_updates)

    def save_mappings(self, file_path: str | None = None) -> None:
        """Save current mappings to JSON file."""
        save_path = file_path or self.config_file_path
        if not save_path:
            raise ValueError("No file path provided for saving mappings")

        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self._mappings, f, indent=2)

    def get_supported_sources(self) -> list:
        """Get list of supported data sources."""
        return list(self._mappings.keys())

    def validate_source(self, source: str) -> bool:
        """Check if source has field mappings configured."""
        return source in self._mappings and bool(self._mappings[source])


def get_field_mapping_config(config_file_path: str | None = None) -> FieldMappingConfig:
    """
    Create a field mapping configuration instance.

    Simple factory function that creates a new FieldMappingConfig instance.
    This replaces the previous singleton pattern for better testability.

    Args:
        config_file_path: Optional path to JSON config file

    Returns:
        New FieldMappingConfig instance
    """
    return FieldMappingConfig(config_file_path)
