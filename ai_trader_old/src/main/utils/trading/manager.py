"""
Trading Universe Manager

Core universe management system for creating, filtering, and managing trading universes.
"""

# Standard library imports
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
import logging

# Third-party imports
import numpy as np
import pandas as pd

from .filters import create_high_volume_filters, create_large_cap_filters
from .types import Filter, UniverseConfig, UniverseSnapshot, UniverseType

logger = logging.getLogger(__name__)


class UniverseManager:
    """
    Comprehensive universe management system.

    Handles creation, filtering, and management of trading universes
    with support for dynamic rebalancing and custom criteria.
    """

    def __init__(self, data_provider: Callable | None = None):
        """
        Initialize universe manager.

        Args:
            data_provider: Function to provide market data
        """
        self.data_provider = data_provider
        self.universes: dict[str, UniverseConfig] = {}
        self.universe_history: dict[str, list[UniverseSnapshot]] = defaultdict(list)
        self.symbol_cache: dict[str, pd.DataFrame] = {}
        self.custom_filters: dict[str, Callable] = {}

        # Initialize predefined universes
        self._init_predefined_universes()

        logger.info("Universe manager initialized")

    def _init_predefined_universes(self):
        """Initialize predefined universe configurations."""
        # S&P 500 universe
        self.universes["sp500"] = UniverseConfig(
            name="S&P 500",
            universe_type=UniverseType.INDEX_BASED,
            max_symbols=500,
            rebalance_frequency="monthly",
        )

        # Large cap universe
        self.universes["large_cap"] = UniverseConfig(
            name="Large Cap",
            universe_type=UniverseType.MARKET_CAP_BASED,
            filters=create_large_cap_filters(),
            max_symbols=1000,
            ranking_criteria="market_cap",
            ranking_ascending=False,
        )

        # High volume universe
        self.universes["high_volume"] = UniverseConfig(
            name="High Volume",
            universe_type=UniverseType.SCREENED,
            filters=create_high_volume_filters(),
            max_symbols=500,
            ranking_criteria="volume",
            ranking_ascending=False,
        )

    def add_universe(self, config: UniverseConfig):
        """Add a universe configuration."""
        self.universes[config.name] = config
        logger.info(f"Added universe: {config.name}")

    def get_universe(self, name: str) -> UniverseConfig | None:
        """Get universe configuration by name."""
        return self.universes.get(name)

    def list_universes(self) -> list[str]:
        """List all available universes."""
        return list(self.universes.keys())

    def add_custom_filter(self, name: str, filter_func: Callable[[pd.DataFrame], pd.Series]):
        """Add a custom filter function."""
        self.custom_filters[name] = filter_func
        logger.info(f"Added custom filter: {name}")

    async def construct_universe(
        self, config: UniverseConfig, reference_date: datetime | None = None
    ) -> set[str]:
        """
        Construct universe based on configuration.

        Args:
            config: Universe configuration
            reference_date: Date for universe construction

        Returns:
            Set of symbols in the universe
        """
        if reference_date is None:
            reference_date = datetime.now()

        logger.info(f"Constructing universe: {config.name}")

        # Get candidate symbols
        candidates = await self._get_candidate_symbols(config, reference_date)

        if candidates.empty:
            logger.warning(f"No candidate symbols found for universe: {config.name}")
            return set()

        # Apply filters
        filtered_symbols = self._apply_filters(candidates, config.filters)

        # Apply ranking and size constraints
        final_symbols = self._apply_ranking_and_constraints(filtered_symbols, config)

        # Create snapshot
        snapshot = UniverseSnapshot(
            timestamp=reference_date,
            symbols=final_symbols,
            metadata={
                "config": config.to_dict(),
                "candidates_count": len(candidates),
                "filtered_count": len(filtered_symbols),
                "final_count": len(final_symbols),
            },
        )

        self.universe_history[config.name].append(snapshot)

        logger.info(f"Universe constructed: {config.name} with {len(final_symbols)} symbols")
        return final_symbols

    async def _get_candidate_symbols(
        self, config: UniverseConfig, reference_date: datetime
    ) -> pd.DataFrame:
        """Get candidate symbols for universe construction."""
        if config.universe_type == UniverseType.STATIC:
            # Static universe - symbols are predefined
            return pd.DataFrame({"symbol": config.filters[0].value if config.filters else []})

        elif config.universe_type == UniverseType.INDEX_BASED:
            # Index-based universe - get symbols from index
            return await self._get_index_symbols(config.name, reference_date)

        else:
            # Dynamic universe - get all available symbols
            return await self._get_all_symbols(reference_date)

    async def _get_index_symbols(self, index_name: str, reference_date: datetime) -> pd.DataFrame:
        """Get symbols from a specific index."""
        # This would typically call an external data provider
        if self.data_provider:
            try:
                return await self.data_provider(
                    f"index_symbols_{index_name.lower()}", reference_date
                )
            except Exception as e:
                logger.error(f"Error fetching index symbols: {e}")

        # Fallback to cached data or empty DataFrame
        return pd.DataFrame(columns=["symbol", "market_cap", "volume", "price"])

    async def _get_all_symbols(self, reference_date: datetime) -> pd.DataFrame:
        """Get all available symbols with market data."""
        # This would typically call an external data provider
        if self.data_provider:
            try:
                return await self.data_provider("all_symbols", reference_date)
            except Exception as e:
                logger.error(f"Error fetching all symbols: {e}")

        # Fallback to cached data or sample data
        return self._get_sample_data()

    def _get_sample_data(self) -> pd.DataFrame:
        """Get sample data for testing."""
        sample_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
        return pd.DataFrame(
            {
                "symbol": sample_symbols,
                "market_cap": np.secure_uniform(
                    100_000_000_000, 3_000_000_000_000, len(sample_symbols)
                ),
                "volume": np.secure_uniform(1_000_000, 100_000_000, len(sample_symbols)),
                "price": np.secure_uniform(50, 500, len(sample_symbols)),
                "sector": np.secure_choice(
                    ["Technology", "Healthcare", "Finance"], len(sample_symbols)
                ),
                "exchange": ["NASDAQ"] * len(sample_symbols),
            }
        )

    def _apply_filters(self, data: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
        """Apply filters to candidate symbols."""
        if not filters:
            return data

        mask = pd.Series(True, index=data.index)

        for filter_obj in filters:
            if not filter_obj.enabled:
                continue

            criteria_name = filter_obj.criteria.value

            if criteria_name == "custom":
                # Apply custom filter
                if isinstance(filter_obj.value, str) and filter_obj.value in self.custom_filters:
                    custom_mask = self.custom_filters[filter_obj.value](data)
                    mask = mask & custom_mask
            elif criteria_name in data.columns:
                filter_mask = filter_obj.apply(data[criteria_name])
                mask = mask & filter_mask
            else:
                logger.warning(f"Column '{criteria_name}' not found in data")

        return data[mask]

    def _apply_ranking_and_constraints(
        self, data: pd.DataFrame, config: UniverseConfig
    ) -> set[str]:
        """Apply ranking and size constraints."""
        if data.empty:
            return set()

        # Apply ranking if specified
        if config.ranking_criteria and config.ranking_criteria in data.columns:
            data = data.sort_values(config.ranking_criteria, ascending=config.ranking_ascending)

        # Apply size constraints
        if config.max_symbols:
            data = data.head(config.max_symbols)

        if config.min_symbols and len(data) < config.min_symbols:
            logger.warning(
                f"Universe has fewer symbols ({len(data)}) than minimum ({config.min_symbols})"
            )

        return set(data["symbol"].tolist())

    def get_universe_symbols(self, name: str, reference_date: datetime | None = None) -> set[str]:
        """Get current symbols for a universe."""
        if name not in self.universe_history:
            return set()

        history = self.universe_history[name]
        if not history:
            return set()

        if reference_date is None:
            # Return most recent
            return history[-1].symbols

        # Find closest snapshot
        closest_snapshot = min(
            history, key=lambda x: abs((x.timestamp - reference_date).total_seconds())
        )

        return closest_snapshot.symbols

    def get_universe_history(self, name: str) -> list[UniverseSnapshot]:
        """Get universe history."""
        return self.universe_history.get(name, [])

    def create_filtered_universe(
        self, base_universe: str, additional_filters: list[Filter], new_name: str
    ) -> UniverseConfig:
        """Create a new universe by adding filters to an existing one."""
        if base_universe not in self.universes:
            raise ValueError(f"Base universe '{base_universe}' not found")

        base_config = self.universes[base_universe]

        new_config = UniverseConfig(
            name=new_name,
            universe_type=UniverseType.SCREENED,
            filters=base_config.filters + additional_filters,
            max_symbols=base_config.max_symbols,
            min_symbols=base_config.min_symbols,
            rebalance_frequency=base_config.rebalance_frequency,
            ranking_criteria=base_config.ranking_criteria,
            ranking_ascending=base_config.ranking_ascending,
        )

        self.add_universe(new_config)
        return new_config

    def get_symbol_universe_membership(self, symbol: str) -> dict[str, bool]:
        """Get universe membership for a symbol."""
        membership = {}

        for universe_name in self.universes:
            symbols = self.get_universe_symbols(universe_name)
            membership[universe_name] = symbol in symbols

        return membership

    def get_sector_distribution(self, name: str) -> dict[str, int]:
        """Get sector distribution for a universe."""
        symbols = self.get_universe_symbols(name)

        if not symbols:
            return {}

        # This would typically fetch sector data from data provider
        # For now, return a mock distribution
        sectors = ["Technology", "Healthcare", "Finance", "Consumer", "Energy", "Industrials"]
        distribution = {}

        for sector in sectors:
            count = len([s for s in symbols if hash(s) % len(sectors) == sectors.index(sector)])
            if count > 0:
                distribution[sector] = count

        return distribution
