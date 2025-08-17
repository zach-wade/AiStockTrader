# File: src/main/utils/services_factory.py

"""
Centralized factory functions for creating and wiring core application services.

This module helps manage complex dependency graphs by encapsulating
the instantiation and wiring logic for various service components,
promoting cleaner code at the call site.
"""

# Standard library imports
import logging
from typing import Union

# Third-party imports
# Import necessary types from config_manager.py
from omegaconf import DictConfig  # For Hydra/OmegaConf config objects

# Local imports
# Corrected imports for DataFetcher and its dependencies
from main.data_pipeline.historical.data_fetcher import DataFetcher
from main.data_pipeline.processing.transformer import DataTransformer
from main.data_pipeline.storage.archive import DataArchive  # Import DataArchive type for clarity

# Corrected import for get_archive (now from archive_initializer)
from main.data_pipeline.storage.archive_helpers.archive_initializer import get_archive
from main.data_pipeline.storage.repositories.market_data import MarketDataRepository
from main.interfaces.database import IAsyncDatabase
from main.utils.resilience import ErrorRecoveryManager

# Assuming AITraderConfig is the Pydantic validated type returned by load_config when validation is on
try:
    # Local imports
    from main.config.validation_models import AITraderConfig

    # Use Union[DictConfig, AITraderConfig] for config type if validation can return either
    # Or just AITraderConfig if load_config always returns validated type when used by components
    ConfigType = Union[DictConfig, AITraderConfig]
except ImportError:
    # Fallback if AITraderConfig is not present or not desired for strict typing
    ConfigType = DictConfig

logger = logging.getLogger(__name__)


# Corrected make_data_fetcher signature to use ConfigType
def make_data_fetcher(config: ConfigType, db_adapter: IAsyncDatabase) -> DataFetcher:
    """
    Factory function to create and wire up a DataFetcher instance with all its dependencies.

    This centralizes the complex instantiation logic, making the DataFetcher's
    usage simpler at the call site.

    Args:
        config: The main application configuration object (OmegaConf DictConfig or AITraderConfig).
        db_adapter: An initialized asynchronous database adapter instance.

    Returns:
        An initialized DataFetcher instance.
    """
    logger.info("Creating DataFetcher instance and wiring its dependencies...")

    # 1. Instantiate ErrorRecoveryManager
    # ErrorRecoveryManager provides retry logic and error handling
    resilience_strategy = ErrorRecoveryManager()
    logger.debug("Instantiated ErrorRecoveryManager.")

    # 2. Instantiate DataTransformer
    # DataTransformer's __init__ should also accept the OmegaConf DictConfig or AITraderConfig.
    # From your code, it's `config: Any`, so passing ConfigType is fine.
    data_transformer = DataTransformer(config=config)
    logger.debug("Instantiated DataTransformer.")

    # 3. Get DataArchive singleton
    # The get_archive() function is now explicitly imported from archive_initializer.py.
    # Its signature `get_archive(config: Optional[Any] = None)` means passing `config` is correct.
    data_archive: DataArchive = get_archive(config=config)  # Pass config to ensure correct setup
    logger.debug("Instantiated DataArchive (via get_archive).")

    # 4. Instantiate MarketDataRepository
    # MarketDataRepository's __init__ typically takes `db_adapter` and `config`.
    # Your DataFetcher `__init__` also passes `config` to MarketDataRepository.
    # Let's ensure MarketDataRepository takes `config` as well if it needs it.
    # Your current MarketDataRepository only takes `db_adapter` and `config` (optional).
    # This is fine, we pass the main `config` object.
    market_data_repo = MarketDataRepository(db_adapter=db_adapter, config=config)
    logger.debug("Instantiated MarketDataRepository.")

    # 5. Create DataFetcher with all wired dependencies
    # DataFetcher's __init__ expects `config: Dict`. We need to align this.
    # If DataFetcher expects `Dict`, and `config` is `DictConfig`, we might need conversion `OmegaConf.to_container(config, resolve=True)`.
    # Let's assume DataFetcher's __init__ can accept `ConfigType` (DictConfig/AITraderConfig).
    # If not, its signature needs to be updated, or conversion added here.
    data_fetcher = DataFetcher(
        config=config,  # Pass the OmegaConf DictConfig or AITraderConfig object
        resilience=resilience_strategy,
        standardizer=data_transformer,
        archive=data_archive,
        market_data_repo=market_data_repo,
    )
    logger.info("DataFetcher instance successfully created and wired.")

    return data_fetcher
