"""
Event Coordinator

Coordinates event-driven operations including automatic backfill triggers
when symbols are qualified, promoted, or when data gaps are detected.
Implements the event-driven flow: Scanner → Events → Automatic Backfill.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from typing import Any

# Local imports
from main.config import get_config_manager
from main.data_pipeline.core.enums import DataLayer, DataType
from main.data_pipeline.core.exceptions import EventProcessingError, convert_exception
from main.interfaces.data_pipeline.orchestration import (
    IEventCoordinator,
    ILayerManager,
    IRetentionManager,
)
from main.interfaces.events import EventType, IEventBus
from main.interfaces.events.event_types import (
    DataGapDetectedEvent,
    SymbolPromotedEvent,
    SymbolQualifiedEvent,
)
from main.utils.core import (
    AsyncCircuitBreaker,
    RateLimiter,
    async_retry,
    ensure_utc,
    gather_with_exceptions,
    get_logger,
)
from main.utils.layer_utils import get_layer_config


class EventCoordinator(IEventCoordinator):
    """
    Coordinates event-driven data pipeline operations.

    Event Flow:
    1. Scanner qualifies symbol → SymbolQualifiedEvent
    2. EventCoordinator handles event → Automatic backfill scheduled
    3. Symbol promoted → SymbolPromotedEvent → Enhanced data collection
    4. Data gap detected → DataGapDetectedEvent → Gap fill backfill
    """

    def __init__(
        self,
        event_bus: IEventBus,
        layer_manager: ILayerManager,
        retention_manager: IRetentionManager,
        config: dict[str, Any] | None = None,
    ):
        self.event_bus = event_bus
        self.layer_manager = layer_manager
        self.retention_manager = retention_manager
        self.config = config or {}
        self.logger = get_logger(__name__)

        # Initialize config manager
        self.config_manager = get_config_manager()

        # Event processing statistics
        self._event_stats = {
            "symbol_qualified_events": 0,
            "symbol_promoted_events": 0,
            "data_gap_events": 0,
            "backfills_scheduled": 0,
            "failed_events": 0,
        }

        # Configuration from unified system
        try:
            # Try to load event config
            event_config = self.config_manager.get_config().get("event_config", {})
        except (AttributeError, KeyError, TypeError) as e:
            # Fallback to defaults if config not available
            self.logger.debug(f"Event config not available: {e}")
            event_config = {}
        backfill_config = event_config.get("backfill", {})

        self.auto_backfill_enabled = backfill_config.get("auto_enabled", True)
        self.backfill_delay_minutes = backfill_config.get("delay_minutes", 5)
        self.max_concurrent_backfills = backfill_config.get("max_concurrent", 3)

        # Get rate limits for event processing
        try:
            rate_limit_config = (
                self.config_manager.get_config().get("rate_limits", {}).get("events", {})
            )
        except (AttributeError, KeyError, TypeError) as e:
            self.logger.debug(f"Rate limit config not available: {e}")
            rate_limit_config = {}

        # Rate limiter for backfill scheduling
        self.backfill_rate_limiter = RateLimiter(
            rate=rate_limit_config.get("backfill_rate_per_minute", 10), per=60  # per minute
        )

        # Circuit breaker for event publishing
        circuit_config = rate_limit_config.get("circuit_breaker", {})
        self.event_circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=circuit_config.get("failure_threshold", 5),
            recovery_timeout=circuit_config.get("recovery_timeout", 60),
            expected_exception=Exception,
        )

        self.logger.info("EventCoordinator initialized with unified configuration")

    async def initialize(self) -> None:
        """Initialize event subscriptions."""
        if not self.event_bus:
            self.logger.warning("No event bus configured, cannot subscribe to events")
            return

        try:
            # Subscribe to symbol qualification events (subscribe is synchronous)
            self.event_bus.subscribe(
                EventType.SYMBOL_QUALIFIED, self._handle_symbol_qualified_event
            )

            # Subscribe to symbol promotion events (subscribe is synchronous)
            self.event_bus.subscribe(EventType.SYMBOL_PROMOTED, self._handle_symbol_promoted_event)

            # Subscribe to data gap detection events (subscribe is synchronous)
            self.event_bus.subscribe(EventType.DATA_GAP_DETECTED, self._handle_data_gap_event)

            self.logger.info("EventCoordinator subscribed to qualification and backfill events")

        except Exception as e:
            self.logger.error(f"Failed to initialize event subscriptions: {e}")
            raise

    async def _handle_symbol_qualified_event(self, event: SymbolQualifiedEvent) -> None:
        """Handle symbol qualified event from event bus."""
        try:
            layer = DataLayer(event.layer)
            await self.handle_symbol_qualified(event.symbol, layer)
        except Exception as e:
            self.logger.error(f"Error handling symbol qualified event: {e}")

    async def _handle_symbol_promoted_event(self, event: SymbolPromotedEvent) -> None:
        """Handle symbol promoted event from event bus."""
        try:
            to_layer = DataLayer(event.to_layer)
            await self.handle_symbol_promoted(event.symbol, to_layer)
        except Exception as e:
            self.logger.error(f"Error handling symbol promoted event: {e}")

    async def _handle_data_gap_event(self, event: DataGapDetectedEvent) -> None:
        """Handle data gap detected event from event bus."""
        try:
            await self.handle_data_gap_detected(
                event.symbol, event.data_type, event.gap_start, event.gap_end
            )
        except Exception as e:
            self.logger.error(f"Error handling data gap event: {e}")

    async def handle_symbol_qualified(self, symbol: str, layer: DataLayer) -> None:
        """Handle symbol qualification event - trigger automatic backfill."""
        try:
            self.logger.info(f"Handling symbol qualification: {symbol} → Layer {layer.value}")

            if not self.auto_backfill_enabled:
                self.logger.debug("Auto-backfill disabled, skipping automatic backfill")
                return

            # Get layer configuration to determine backfill scope
            layer_config = await self.layer_manager.get_layer_config(layer)

            # Determine data types to backfill based on layer
            data_types = self._get_data_types_for_layer(layer)

            # Schedule automatic backfill with layer-appropriate lookback period
            backfill_days = self._get_backfill_days_for_layer(layer)
            backfill_id = await self.schedule_automatic_backfill(symbol, layer)

            # Publish backfill scheduled event concurrently with stats update
            tasks = [
                self._publish_backfill_scheduled_event(symbol, layer, backfill_id, data_types),
                self._update_stats_async("backfills_scheduled", 1),
            ]
            await gather_with_exceptions(*tasks)

            # Update statistics
            self._event_stats["symbol_qualified_events"] += 1
            self._event_stats["backfills_scheduled"] += 1

            self.logger.info(
                f"Scheduled automatic backfill for {symbol} (layer {layer.value}): "
                f"backfill_id={backfill_id}, data_types={[dt.value for dt in data_types]}"
            )

        except Exception as e:
            self._event_stats["failed_events"] += 1
            error = EventProcessingError(
                f"Failed to handle symbol qualification for {symbol}",
                event_type="symbol_qualified",
                event_handler="EventCoordinator",
                original_error=e,
                context={"symbol": symbol, "layer": layer.value},
            )
            self.logger.error(f"Symbol qualification handling error: {error}")
            raise error

    async def handle_symbol_promoted(
        self, symbol: str, from_layer: DataLayer, to_layer: DataLayer
    ) -> None:
        """Handle symbol promotion event - enhance data collection."""
        try:
            self.logger.info(
                f"Handling symbol promotion: {symbol} from Layer {from_layer.value} → {to_layer.value}"
            )

            # Get enhanced data types for the new layer
            new_data_types = self._get_data_types_for_layer(to_layer)
            old_data_types = self._get_data_types_for_layer(from_layer)

            # Find additional data types needed for the higher layer
            additional_data_types = [dt for dt in new_data_types if dt not in old_data_types]

            if additional_data_types:
                # Schedule backfill for additional data types
                backfill_id = await self._schedule_enhancement_backfill(
                    symbol, to_layer, additional_data_types
                )

                self.logger.info(
                    f"Scheduled enhancement backfill for {symbol}: "
                    f"additional_data_types={[dt.value for dt in additional_data_types]}"
                )

            # Apply new retention policy
            await self.retention_manager.apply_retention_policy(symbol, to_layer)

            # Publish promotion completed event
            await self._publish_promotion_completed_event(symbol, from_layer, to_layer)

            # Update statistics
            self._event_stats["symbol_promoted_events"] += 1
            if additional_data_types:
                self._event_stats["backfills_scheduled"] += 1

        except Exception as e:
            self._event_stats["failed_events"] += 1
            error = EventProcessingError(
                f"Failed to handle symbol promotion for {symbol}",
                event_type="symbol_promoted",
                event_handler="EventCoordinator",
                original_error=e,
                context={
                    "symbol": symbol,
                    "from_layer": from_layer.value,
                    "to_layer": to_layer.value,
                },
            )
            self.logger.error(f"Symbol promotion handling error: {error}")
            raise error

    async def handle_data_gap_detected(
        self, symbol: str, data_type: DataType, gap_info: dict[str, Any]
    ) -> None:
        """Handle data gap detection event - trigger gap fill backfill."""
        try:
            self.logger.info(
                f"Handling data gap detection: {symbol}, {data_type.value}, "
                f"gap_size={gap_info.get('gap_size_hours', 'unknown')}"
            )

            # Get symbol's current layer
            symbol_layer = await self._get_symbol_layer(symbol)
            if symbol_layer is None:
                self.logger.warning(f"Symbol {symbol} not found in any layer, skipping gap fill")
                return

            # Determine if gap is significant enough to trigger backfill
            if not self._is_gap_significant(gap_info, symbol_layer):
                self.logger.debug(f"Gap for {symbol} not significant enough for backfill")
                return

            # Schedule gap fill backfill
            backfill_id = await self._schedule_gap_fill_backfill(
                symbol, symbol_layer, data_type, gap_info
            )

            # Publish gap fill scheduled event
            await self._publish_gap_fill_scheduled_event(symbol, data_type, backfill_id, gap_info)

            # Update statistics
            self._event_stats["data_gap_events"] += 1
            self._event_stats["backfills_scheduled"] += 1

            self.logger.info(
                f"Scheduled gap fill backfill for {symbol}: "
                f"backfill_id={backfill_id}, data_type={data_type.value}"
            )

        except Exception as e:
            self._event_stats["failed_events"] += 1
            error = EventProcessingError(
                f"Failed to handle data gap detection for {symbol}",
                event_type="data_gap_detected",
                event_handler="EventCoordinator",
                original_error=e,
                context={"symbol": symbol, "data_type": data_type.value, "gap_info": gap_info},
            )
            self.logger.error(f"Data gap handling error: {error}")
            raise error

    async def schedule_automatic_backfill(self, symbol: str, layer: DataLayer) -> str:
        """Schedule automatic backfill based on layer policies."""
        try:
            # Apply rate limiting
            async with self.backfill_rate_limiter:
                # Generate unique backfill ID
                now = ensure_utc(datetime.now(UTC))
                backfill_id = f"auto_backfill_{symbol}_{layer.value}_{int(now.timestamp())}"

                # Get data types and date range for the layer
                data_types = self._get_data_types_for_layer(layer)
                backfill_days = self._get_backfill_days_for_layer(layer)

                end_date = now
                start_date = end_date - timedelta(days=backfill_days)

                # Create a BackfillRequested event
                # We use a generic Event with custom event_type string
                # Local imports
                from main.interfaces.events import Event

                backfill_data = {
                    "backfill_id": backfill_id,
                    "symbol": symbol,
                    "layer": layer.value,
                    "data_types": [dt.value for dt in data_types],
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "priority": "normal",
                }

                # Create event - use string event type for custom events
                request_event = Event(
                    event_type="BackfillRequested",  # String type that handler listens for
                    metadata=backfill_data,  # Use metadata field instead of data
                )

                # Publish the event
                await self.event_bus.publish(request_event)

                self.logger.debug(f"Scheduled automatic backfill: {backfill_id}")
                return backfill_id

        except Exception as e:
            error = convert_exception(e, f"Failed to schedule automatic backfill for {symbol}")
            self.logger.error(f"Backfill scheduling error: {error}")
            raise error

    async def get_event_statistics(self) -> dict[str, Any]:
        """Get event processing statistics."""
        return {
            "event_statistics": self._event_stats.copy(),
            "configuration": {
                "auto_backfill_enabled": self.auto_backfill_enabled,
                "backfill_delay_minutes": self.backfill_delay_minutes,
                "max_concurrent_backfills": self.max_concurrent_backfills,
            },
            "generated_at": ensure_utc(datetime.now(UTC)).isoformat(),
        }

    def _get_data_types_for_layer(self, layer: DataLayer) -> list[DataType]:
        """Get appropriate data types for a layer."""
        # Get from unified layer configuration
        layer_config = get_layer_config(layer)
        processing_config = layer_config.get("processing", {})

        data_type_names = processing_config.get("data_types", ["market_data"])

        # Convert string names to DataType enums
        data_types = []
        for type_name in data_type_names:
            try:
                # Convert snake_case to uppercase enum name
                enum_name = type_name.upper()
                data_types.append(DataType[enum_name])
            except KeyError:
                self.logger.warning(f"Unknown data type in config: {type_name}")

        return data_types or [DataType.MARKET_DATA]

    def _get_backfill_days_for_layer(self, layer: DataLayer) -> int:
        """Get backfill lookback days for a layer."""
        # Get from unified layer configuration
        layer_config = get_layer_config(layer)
        backfill_config = layer_config.get("backfill", {})

        return backfill_config.get("default_days", layer.retention_days)

    async def _get_symbol_layer(self, symbol: str) -> DataLayer | None:
        """Get the current layer for a symbol."""
        try:
            # Try each layer to find the symbol
            for layer in DataLayer:
                symbols = await self.layer_manager.get_symbols_for_layer(layer)
                if symbol in symbols:
                    return layer
            return None

        except Exception:
            return None

    def _is_gap_significant(self, gap_info: dict[str, Any], layer: DataLayer) -> bool:
        """Determine if a gap is significant enough to trigger backfill."""
        gap_hours = gap_info.get("gap_size_hours", 0)

        # Get gap threshold from unified layer configuration
        layer_config = get_layer_config(layer)
        processing_config = layer_config.get("processing", {})

        threshold = processing_config.get("gap_threshold_hours", 24)
        return gap_hours >= threshold

    async def _schedule_enhancement_backfill(
        self, symbol: str, layer: DataLayer, data_types: list[DataType]
    ) -> str:
        """Schedule backfill for enhanced data collection after promotion."""
        backfill_id = (
            f"enhancement_{symbol}_{layer.value}_{int(ensure_utc(datetime.now(UTC)).timestamp())}"
        )

        # Use full retention period for enhancement backfill
        end_date = ensure_utc(datetime.now(UTC))

        # Get retention days from layer config
        layer_config = get_layer_config(layer)
        retention_days = layer_config.get("retention", {}).get(
            "total_retention_days", layer.retention_days
        )

        start_date = end_date - timedelta(days=retention_days)

        backfill_event = {
            "event_type": "BackfillRequested",
            "backfill_id": backfill_id,
            "symbol": symbol,
            "layer": layer.value,
            "data_types": [dt.value for dt in data_types],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "priority": "high",
            "trigger": "symbol_promotion",
            "scheduled_at": ensure_utc(datetime.now(UTC)).isoformat(),
        }

        @async_retry(max_attempts=3, delay=1.0)
        async def publish_with_retry():
            await self.event_circuit_breaker.call(
                self.event_bus.publish, "BackfillRequested", backfill_event
            )

        await publish_with_retry()
        return backfill_id

    async def _schedule_gap_fill_backfill(
        self, symbol: str, layer: DataLayer, data_type: DataType, gap_info: dict[str, Any]
    ) -> str:
        """Schedule backfill to fill detected data gaps."""
        backfill_id = (
            f"gap_fill_{symbol}_{data_type.value}_{int(ensure_utc(datetime.now(UTC)).timestamp())}"
        )

        # Use gap-specific date range
        start_date = datetime.fromisoformat(gap_info["gap_start"])
        end_date = datetime.fromisoformat(gap_info["gap_end"])

        backfill_event = {
            "event_type": "BackfillRequested",
            "backfill_id": backfill_id,
            "symbol": symbol,
            "layer": layer.value,
            "data_types": [data_type.value],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "priority": "normal",
            "trigger": "data_gap_detected",
            "gap_info": gap_info,
            "scheduled_at": ensure_utc(datetime.now(UTC)).isoformat(),
        }

        @async_retry(max_attempts=3, delay=1.0)
        async def publish_with_retry():
            await self.event_circuit_breaker.call(
                self.event_bus.publish, "BackfillRequested", backfill_event
            )

        await publish_with_retry()
        return backfill_id

    async def _publish_backfill_scheduled_event(
        self, symbol: str, layer: DataLayer, backfill_id: str, data_types: list[DataType]
    ) -> None:
        """Publish backfill scheduled event."""
        event_data = {
            "symbol": symbol,
            "layer": layer.value,
            "backfill_id": backfill_id,
            "data_types": [dt.value for dt in data_types],
            "scheduled_at": ensure_utc(datetime.now(UTC)).isoformat(),
        }

        await self.event_bus.publish("BackfillScheduled", event_data)

    async def _publish_promotion_completed_event(
        self, symbol: str, from_layer: DataLayer, to_layer: DataLayer
    ) -> None:
        """Publish symbol promotion completed event."""
        event_data = {
            "symbol": symbol,
            "from_layer": from_layer.value,
            "to_layer": to_layer.value,
            "completed_at": ensure_utc(datetime.now(UTC)).isoformat(),
        }

        await self.event_bus.publish("SymbolPromotionCompleted", event_data)

    async def _publish_gap_fill_scheduled_event(
        self, symbol: str, data_type: DataType, backfill_id: str, gap_info: dict[str, Any]
    ) -> None:
        """Publish gap fill scheduled event."""
        event_data = {
            "symbol": symbol,
            "data_type": data_type.value,
            "backfill_id": backfill_id,
            "gap_info": gap_info,
            "scheduled_at": ensure_utc(datetime.now(UTC)).isoformat(),
        }

        await self.event_bus.publish("GapFillScheduled", event_data)

    async def _update_stats_async(self, stat_name: str, increment: int = 1) -> None:
        """Update statistics asynchronously."""
        self._event_stats[stat_name] = self._event_stats.get(stat_name, 0) + increment
