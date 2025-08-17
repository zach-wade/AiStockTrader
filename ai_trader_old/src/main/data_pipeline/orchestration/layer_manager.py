"""
Layer Manager

Manages layer-based data processing with symbol assignment, promotion/demotion,
and layer-specific configuration. Implements the layer system (0-3) that
replaces the previous tier-based architecture.
"""

# Standard library imports
from datetime import UTC, datetime
from typing import Any

# Local imports
from main.config import get_config_manager
from main.data_pipeline.core.enums import DataLayer
from main.data_pipeline.core.exceptions import LayerConfigurationError, convert_exception
from main.interfaces.data_pipeline.orchestration import ILayerManager
from main.interfaces.database import IAsyncDatabase
from main.utils.core import ensure_utc, get_logger, timer
from main.utils.math_utils import safe_divide


class LayerManager(ILayerManager):
    """
    Manages layer-based data processing and symbol assignments.

    Implements the layer system:
    - Layer 0: Basic Tradable (~10,000 symbols) - 30 days retention
    - Layer 1: Liquid Symbols (~2,000 symbols) - 365 days retention
    - Layer 2: Catalyst-Driven (~500 symbols) - 730 days retention
    - Layer 3: Active Trading (~50 symbols) - 1825 days retention
    """

    def __init__(self, db_adapter: IAsyncDatabase, config: dict[str, Any] | None = None):
        self.db_adapter = db_adapter
        self.config = config or {}
        self.logger = get_logger(__name__)

        # Initialize config loader
        self.config_manager = get_config_manager()

        # Cache for symbol assignments
        self._symbol_layer_cache = {}
        self._cache_expiry = {}
        self._cache_ttl_seconds = self.config.get("cache_ttl_seconds", 300)  # 5 minutes

        self.logger.info("LayerManager initialized with unified configuration system")

    async def get_layer_config(self, layer: DataLayer) -> dict[str, Any]:
        """Get configuration for a specific layer."""
        try:
            with timer(f"get_layer_config({layer.name})"):
                # Get configuration from unified config loader
                base_config = self.config_manager.load_config("layer_configs").get(
                    f"layer_{layer}", {}
                )

                # Enhance with database-specific configuration if available
                query = """
                    SELECT config_value
                    FROM layer_configurations
                    WHERE layer = $1 AND is_active = true
                """

                db_config_row = await self.db_adapter.fetch_one(query, layer.value)
                if db_config_row:
                    # Merge database config with base config
                    # Standard library imports
                    import json

                    db_config = json.loads(db_config_row["config_value"])
                    base_config.update(db_config)

                self.logger.debug(f"Retrieved configuration for layer {layer.value}")
                return base_config

        except Exception as e:
            error = LayerConfigurationError(
                f"Failed to get configuration for layer {layer.value}",
                layer=layer.value,
                configuration_key="layer_config",
                original_error=e,
            )
            self.logger.error(f"Layer configuration error: {error}")
            raise error

    async def get_symbols_for_layer(self, layer: DataLayer) -> list[str]:
        """Get symbols assigned to a specific layer."""
        try:
            cache_key = f"symbols_layer_{layer.value}"
            if self._is_cache_valid(cache_key):
                return self._symbol_layer_cache[cache_key]

            # Query symbols for the layer from companies table
            query = """
                SELECT symbol
                FROM companies
                WHERE layer = $1 AND is_active = true
                ORDER BY symbol
            """

            rows = await self.db_adapter.fetch_all(query, layer.value)
            symbols = [row["symbol"] for row in rows]

            # Cache the result
            self._symbol_layer_cache[cache_key] = symbols
            self._cache_expiry[cache_key] = ensure_utc(datetime.now(UTC))

            self.logger.debug(f"Retrieved {len(symbols)} symbols for layer {layer.value}")
            return symbols

        except Exception as e:
            error = LayerConfigurationError(
                f"Failed to get symbols for layer {layer.value}",
                layer=layer.value,
                original_error=e,
            )
            self.logger.error(f"Symbol retrieval error: {error}")
            raise error

    async def promote_symbol(self, symbol: str, from_layer: DataLayer, to_layer: DataLayer) -> bool:
        """Promote a symbol from one layer to another."""
        try:
            if to_layer.value <= from_layer.value:
                raise LayerConfigurationError(
                    f"Cannot promote symbol {symbol} from layer {from_layer.value} to {to_layer.value} (not an upgrade)",
                    layer=to_layer.value,
                    context={"symbol": symbol, "from_layer": from_layer.value},
                )

            # Check layer capacity
            if not await self.validate_layer_capacity(to_layer):
                self.logger.warning(f"Layer {to_layer.value} at capacity, cannot promote {symbol}")
                return False

            # Update symbol's layer in database
            update_query = """
                UPDATE companies
                SET layer = $1,
                    layer_updated_at = $2,
                    layer_history = array_append(
                        COALESCE(layer_history, ARRAY[]::text[]),
                        $3
                    )
                WHERE symbol = $4 AND layer = $5
            """

            history_entry = (
                f"{from_layer.value}→{to_layer.value}@{ensure_utc(datetime.now(UTC)).isoformat()}"
            )

            result = await self.db_adapter.execute_query(
                update_query,
                to_layer.value,
                ensure_utc(datetime.now(UTC)),
                history_entry,
                symbol,
                from_layer.value,
            )

            # Record transition in layer_transitions table for event sourcing
            await self._record_layer_transition(
                symbol=symbol,
                from_layer=from_layer.value,
                to_layer=to_layer.value,
                reason=f"Promoted from {from_layer.name} to {to_layer.name}",
                transitioned_by="LayerManager.promote_symbol",
                metadata={"promotion": True},
            )

            if result.rowcount == 0:
                self.logger.warning(f"Symbol {symbol} not found in layer {from_layer.value}")
                return False

            # Clear relevant caches
            self._clear_layer_caches()

            self.logger.info(
                f"Promoted symbol {symbol} from layer {from_layer.value} to {to_layer.value}"
            )
            return True

        except Exception as e:
            error = LayerConfigurationError(
                f"Failed to promote symbol {symbol}",
                layer=to_layer.value,
                original_error=e,
                context={
                    "symbol": symbol,
                    "from_layer": from_layer.value,
                    "to_layer": to_layer.value,
                },
            )
            self.logger.error(f"Symbol promotion error: {error}")
            raise error

    async def demote_symbol(self, symbol: str, from_layer: DataLayer, to_layer: DataLayer) -> bool:
        """Demote a symbol from one layer to another."""
        try:
            if to_layer.value >= from_layer.value:
                raise LayerConfigurationError(
                    f"Cannot demote symbol {symbol} from layer {from_layer.value} to {to_layer.value} (not a downgrade)",
                    layer=to_layer.value,
                    context={"symbol": symbol, "from_layer": from_layer.value},
                )

            # Update symbol's layer in database
            update_query = """
                UPDATE companies
                SET layer = $1,
                    layer_updated_at = $2,
                    layer_history = array_append(
                        COALESCE(layer_history, ARRAY[]::text[]),
                        $3
                    )
                WHERE symbol = $4 AND layer = $5
            """

            history_entry = (
                f"{from_layer.value}→{to_layer.value}@{ensure_utc(datetime.now(UTC)).isoformat()}"
            )

            result = await self.db_adapter.execute_query(
                update_query,
                to_layer.value,
                ensure_utc(datetime.now(UTC)),
                history_entry,
                symbol,
                from_layer.value,
            )

            if result.rowcount == 0:
                self.logger.warning(f"Symbol {symbol} not found in layer {from_layer.value}")
                return False

            # Record transition in layer_transitions table for event sourcing
            await self._record_layer_transition(
                symbol=symbol,
                from_layer=from_layer.value,
                to_layer=to_layer.value,
                reason=f"Demoted from {from_layer.name} to {to_layer.name}",
                transitioned_by="LayerManager.demote_symbol",
                metadata={"demotion": True},
            )

            # Clear relevant caches
            self._clear_layer_caches()

            self.logger.info(
                f"Demoted symbol {symbol} from layer {from_layer.value} to {to_layer.value}"
            )
            return True

        except Exception as e:
            error = LayerConfigurationError(
                f"Failed to demote symbol {symbol}",
                layer=to_layer.value,
                original_error=e,
                context={
                    "symbol": symbol,
                    "from_layer": from_layer.value,
                    "to_layer": to_layer.value,
                },
            )
            self.logger.error(f"Symbol demotion error: {error}")
            raise error

    async def get_layer_limits(self, layer: DataLayer) -> dict[str, Any]:
        """Get processing limits for a layer."""
        try:
            config = await self.get_layer_config(layer)

            return {
                "max_symbols": layer.max_symbols,
                "retention_days": layer.retention_days,
                "hot_storage_days": layer.hot_storage_days,
                "supported_intervals": layer.supported_intervals,
                "max_concurrent_operations": config.get("max_concurrent_operations", 10),
                "rate_limit_per_minute": config.get("rate_limit_per_minute", 100),
                "priority_weight": config.get("priority_weight", layer.value + 1),
            }

        except Exception as e:
            error = convert_exception(e, f"Failed to get limits for layer {layer.value}")
            self.logger.error(f"Layer limits error: {error}")
            raise error

    async def validate_layer_capacity(self, layer: DataLayer) -> bool:
        """Check if layer has capacity for new symbols."""
        try:
            current_symbols = await self.get_symbols_for_layer(layer)
            max_symbols = layer.max_symbols

            has_capacity = len(current_symbols) < max_symbols

            self.logger.debug(
                f"Layer {layer.value} capacity: {len(current_symbols)}/{max_symbols} "
                f"({'available' if has_capacity else 'full'})"
            )

            return has_capacity

        except Exception as e:
            error = convert_exception(e, f"Failed to validate capacity for layer {layer.value}")
            self.logger.error(f"Capacity validation error: {error}")
            raise error

    async def get_layer_statistics(self) -> dict[str, Any]:
        """Get statistics for all layers."""
        try:
            with timer("get_layer_statistics"):
                stats = {}

                for layer in DataLayer:
                    symbols = await self.get_symbols_for_layer(layer)
                    config = await self.get_layer_config(layer)

                    stats[layer.name.lower()] = {
                        "layer_number": layer.value,
                        "current_symbols": len(symbols),
                        "max_symbols": layer.max_symbols,
                        "capacity_used_percent": safe_divide(
                            len(symbols) * 100, layer.max_symbols, default=0.0
                        ),
                        "retention_days": layer.retention_days,
                        "hot_storage_days": layer.hot_storage_days,
                        "supported_intervals": layer.supported_intervals,
                        "description": layer.description,
                    }

                return {
                    "layer_statistics": stats,
                    "total_symbols": sum(
                        len(await self.get_symbols_for_layer(layer)) for layer in DataLayer
                    ),
                    "generated_at": ensure_utc(datetime.now(UTC)).isoformat(),
                }

        except Exception as e:
            error = convert_exception(e, "Failed to get layer statistics")
            self.logger.error(f"Statistics error: {error}")
            raise error

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_expiry:
            return False

        expiry_time = self._cache_expiry[cache_key]
        now = ensure_utc(datetime.now(UTC))

        return (now - expiry_time).total_seconds() < self._cache_ttl_seconds

    def _clear_layer_caches(self) -> None:
        """Clear all layer-related caches."""
        self._symbol_layer_cache.clear()
        self._cache_expiry.clear()
        # Also clear config loader cache
        # Config manager handles its own caching
        self.logger.debug("Cleared layer caches")

    async def refresh_caches(self) -> None:
        """Manually refresh all caches."""
        self._clear_layer_caches()

        # Pre-warm caches
        for layer in DataLayer:
            await self.get_layer_config(layer)
            await self.get_symbols_for_layer(layer)

        self.logger.info("Refreshed layer manager caches")

    async def _record_layer_transition(
        self,
        symbol: str,
        from_layer: int | None,
        to_layer: int,
        reason: str,
        transitioned_by: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a layer transition in the layer_transitions table for event sourcing.

        Args:
            symbol: The symbol that transitioned
            from_layer: Previous layer (None for initial qualification)
            to_layer: New layer
            reason: Human-readable reason for the transition
            transitioned_by: Component that triggered the transition
            metadata: Additional data to store with the transition
        """
        try:
            insert_query = """
                INSERT INTO layer_transitions (
                    symbol,
                    from_layer,
                    to_layer,
                    reason,
                    metadata,
                    transitioned_at,
                    transitioned_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """

            # Standard library imports
            import json

            metadata_json = json.dumps(metadata) if metadata else None

            await self.db_adapter.execute_query(
                insert_query,
                symbol,
                from_layer,
                to_layer,
                reason,
                metadata_json,
                ensure_utc(datetime.now(UTC)),
                transitioned_by,
            )

            self.logger.debug(
                f"Recorded layer transition: {symbol} from {from_layer} to {to_layer} ({reason})"
            )

        except Exception as e:
            # Log error but don't fail the main operation
            self.logger.error(
                f"Failed to record layer transition for {symbol}: {e}. "
                "Continuing with main operation."
            )
