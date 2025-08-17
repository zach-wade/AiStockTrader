"""
Layer Utilities

Centralized utilities for layer-based data pipeline operations.
Provides layer configuration, rate limits, and transition helpers.
"""

# Standard library imports
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Any

# Standard library imports
from pathlib import Path

# Third-party imports
import yaml

# Local imports
from main.data_pipeline.core.enums import DataLayer
from main.utils.core import get_logger

logger = get_logger(__name__)


def get_api_rate_limit(layer: DataLayer) -> float:
    """
    Get API rate limit for a specific layer (requests per second).

    Layer-based rate limits correspond to Polygon API subscription tiers:
    - Layer 0 (BASIC): Conservative rate for universe scanning
    - Layer 1 (LIQUID): Moderate rate for liquid symbols
    - Layer 2 (CATALYST): Higher rate for active catalyst symbols
    - Layer 3 (ACTIVE): Maximum rate for actively traded symbols

    Args:
        layer: The data layer

    Returns:
        Rate limit in requests per second
    """
    # Rate limits in requests per minute, converted to per second
    rate_limits = {
        DataLayer.BASIC: 5 / 60,  # 5 req/min = ~0.083 req/sec
        DataLayer.LIQUID: 100 / 60,  # 100 req/min = ~1.67 req/sec
        DataLayer.CATALYST: 1000 / 60,  # 1000 req/min = ~16.67 req/sec
        DataLayer.ACTIVE: 10000 / 60,  # 10000 req/min = ~166.67 req/sec
    }

    rate_limit = rate_limits.get(layer, 5 / 60)
    logger.debug(f"Layer {layer.name} rate limit: {rate_limit:.2f} req/sec")
    return rate_limit


def get_layer_config(layer: DataLayer, config_manager: Any | None = None) -> dict[str, Any]:
    """
    Get comprehensive configuration for a specific layer.

    Loads configuration from layer_definitions.yaml and combines
    with programmatic defaults from the DataLayer enum.

    Args:
        layer: The data layer
        config_manager: Optional config manager to use for loading additional configuration

    Returns:
        Dictionary containing layer configuration
    """
    # Start with enum properties as base configuration
    base_config = {
        "layer": layer.value,
        "name": layer.name,
        "description": layer.description,
        "retention_days": layer.retention_days,
        "hot_storage_days": layer.hot_storage_days,
        "supported_intervals": layer.supported_intervals,
        "max_symbols": layer.max_symbols,
        "api_rate_limit": get_api_rate_limit(layer),
    }

    # Only try to load additional config if config_manager is provided
    if config_manager is not None:
        try:
            # Try multiple approaches to find the config file
            config_path = None

            # Approach 1: Try through config manager if it has a method
            try:
                if hasattr(config_manager, "get_config_path"):
                    config_path = Path(config_manager.get_config_path("layer_definitions.yaml"))
            except (AttributeError, TypeError, FileNotFoundError):
                pass

            # Approach 2: Use standard config location
            if config_path is None or not config_path.exists():
                base_path = Path(__file__).parent.parent.parent
                config_path = base_path / "config" / "yaml" / "layer_definitions.yaml"

            # Load and merge configuration if file exists
            if config_path and config_path.exists():
                with open(config_path) as f:
                    layer_defs = yaml.safe_load(f)

                # Get specific layer config
                layer_key = f"layer_{layer.value}"
                file_config = layer_defs.get("layers", {}).get(layer_key, {})

                # Merge file config with base config (file config takes precedence)
                for key, value in file_config.items():
                    if value is not None:  # Only override if value is explicitly set
                        base_config[key] = value

                logger.debug(f"Loaded layer {layer.name} config from {config_path}")

        except OSError as e:
            logger.warning(f"Could not read layer config file: {e}")
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in layer config file: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error loading layer config: {e}")
    else:
        # When no config_manager provided, just use base config from enum
        logger.debug(f"Using base layer config for {layer.name} (no config_manager provided)")

    return base_config


def get_layer_batch_size(layer: DataLayer) -> int:
    """
    Get recommended batch size for API operations based on layer.

    Higher layers can use larger batches due to higher rate limits.

    Args:
        layer: The data layer

    Returns:
        Recommended batch size
    """
    batch_sizes = {
        DataLayer.BASIC: 10,  # Small batches for basic tier
        DataLayer.LIQUID: 50,  # Medium batches
        DataLayer.CATALYST: 100,  # Large batches
        DataLayer.ACTIVE: 200,  # Maximum batches
    }

    return batch_sizes.get(layer, 10)


def get_layer_max_concurrent(layer: DataLayer) -> int:
    """
    Get maximum concurrent requests based on layer.

    Args:
        layer: The data layer

    Returns:
        Maximum concurrent requests
    """
    max_concurrent = {
        DataLayer.BASIC: 1,  # Sequential for basic
        DataLayer.LIQUID: 5,  # Some concurrency
        DataLayer.CATALYST: 10,  # Higher concurrency
        DataLayer.ACTIVE: 20,  # Maximum concurrency
    }

    return max_concurrent.get(layer, 1)


def get_layer_cache_ttl(layer: DataLayer) -> int:
    """
    Get cache TTL in seconds based on layer.

    Higher layers need fresher data, so shorter cache TTL.

    Args:
        layer: The data layer

    Returns:
        Cache TTL in seconds
    """
    cache_ttls = {
        DataLayer.BASIC: 3600,  # 1 hour cache for basic
        DataLayer.LIQUID: 900,  # 15 minutes for liquid
        DataLayer.CATALYST: 300,  # 5 minutes for catalyst
        DataLayer.ACTIVE: 60,  # 1 minute for active trading
    }

    return cache_ttls.get(layer, 3600)


def validate_layer_assignment(symbol: str, layer: DataLayer, current_symbols: int) -> bool:
    """
    Validate if a symbol can be assigned to a layer.

    Checks if the layer has capacity for additional symbols.

    Args:
        symbol: The symbol to assign
        layer: The target layer
        current_symbols: Current number of symbols in the layer

    Returns:
        True if assignment is valid, False otherwise
    """
    if current_symbols >= layer.max_symbols:
        logger.warning(
            f"Layer {layer.name} at capacity ({current_symbols}/{layer.max_symbols}), "
            f"cannot assign {symbol}"
        )
        return False

    logger.debug(f"Symbol {symbol} can be assigned to layer {layer.name}")
    return True


def validate_layer_promotion(from_layer: DataLayer, to_layer: DataLayer) -> bool:
    """
    Validate if promotion from one layer to another is valid.

    Promotions can only go up in layer hierarchy (lower number to higher number).

    Args:
        from_layer: Current layer
        to_layer: Target layer for promotion

    Returns:
        True if promotion is valid, False otherwise
    """
    if to_layer.value <= from_layer.value:
        logger.error(
            f"Invalid promotion: {from_layer.name} (layer {from_layer.value}) -> "
            f"{to_layer.name} (layer {to_layer.value}). Can only promote to higher layers."
        )
        return False

    logger.info(f"Valid promotion: {from_layer.name} -> {to_layer.name}")
    return True


def get_layer_by_value(layer_value: int) -> DataLayer | None:
    """
    Get DataLayer enum by its numeric value.

    Args:
        layer_value: The numeric layer value (0-3)

    Returns:
        DataLayer enum or None if invalid
    """
    try:
        return DataLayer(layer_value)
    except ValueError:
        logger.error(f"Invalid layer value: {layer_value}")
        return None
