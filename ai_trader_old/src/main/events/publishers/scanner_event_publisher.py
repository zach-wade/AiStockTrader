"""
Scanner Event Publisher

Publishes events when scanner qualifies symbols for different layers.
Integrates with the event bus to trigger automatic backfills.
"""

# Standard library imports
from datetime import UTC, datetime
from typing import Any

# Local imports
from main.data_pipeline.core.enums import DataLayer
from main.interfaces.events import IEventBus
from main.interfaces.events.event_types import SymbolPromotedEvent, SymbolQualifiedEvent
from main.utils.core import ErrorHandlingMixin, get_logger
from main.utils.layer_metrics import LayerMetricsCollector


class ScannerEventPublisher(ErrorHandlingMixin):
    """
    Publishes events when scanner qualifies or promotes symbols.

    This bridges the scanner layer qualification system with the
    event-driven backfill system in the data pipeline.
    """

    def __init__(self, event_bus: IEventBus | None = None):
        """
        Initialize the scanner event publisher.

        Args:
            event_bus: Optional event bus for publishing events
        """
        super().__init__()  # Initialize ErrorHandlingMixin
        self.event_bus = event_bus
        self.logger = get_logger(__name__)

        # Track published events to avoid duplicates
        self._published_qualifications = set()
        self._published_promotions = set()

        self.logger.info("ScannerEventPublisher initialized")

    async def publish_symbol_qualified(
        self,
        symbol: str,
        layer: DataLayer,
        qualification_reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Publish a symbol qualification event.

        Args:
            symbol: The symbol that was qualified
            layer: The layer the symbol was qualified for
            qualification_reason: Reason for qualification
            metrics: Optional metrics that led to qualification
        """
        if not self.event_bus:
            self.logger.debug("No event bus configured, skipping event publication")
            return

        # Create unique key to track this qualification
        qualification_key = f"{symbol}:{layer.value}"

        # Skip if already published recently (within session)
        if qualification_key in self._published_qualifications:
            self.logger.debug(f"Already published qualification for {symbol} to layer {layer.name}")
            return

        try:
            # Create the event
            event = SymbolQualifiedEvent(
                symbol=symbol,
                layer=layer.value,
                qualification_reason=qualification_reason,
                metrics=metrics or {},
                source="scanner",
                timestamp=datetime.now(UTC),
            )

            # Publish to event bus
            await self.event_bus.publish(event)

            # Track as published
            self._published_qualifications.add(qualification_key)

            self.logger.info(
                f"Published symbol qualification: {symbol} → Layer {layer.name} "
                f"(reason: {qualification_reason})"
            )

        except Exception as e:
            self.logger.error(f"Failed to publish symbol qualification event: {e}")

    async def publish_symbol_promoted(
        self,
        symbol: str,
        from_layer: DataLayer,
        to_layer: DataLayer,
        promotion_reason: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Publish a symbol promotion event.

        Args:
            symbol: The symbol that was promoted
            from_layer: The layer the symbol was promoted from
            to_layer: The layer the symbol was promoted to
            promotion_reason: Reason for promotion
            metrics: Optional metrics that led to promotion
        """
        if not self.event_bus:
            self.logger.debug("No event bus configured, skipping event publication")
            return

        # Create unique key to track this promotion
        promotion_key = f"{symbol}:{from_layer.value}→{to_layer.value}"

        # Skip if already published recently
        if promotion_key in self._published_promotions:
            self.logger.debug(
                f"Already published promotion for {symbol} from "
                f"layer {from_layer.name} to {to_layer.name}"
            )
            return

        try:
            # Create the event
            event = SymbolPromotedEvent(
                symbol=symbol,
                from_layer=from_layer.value,
                to_layer=to_layer.value,
                promotion_reason=promotion_reason,
                metrics=metrics or {},
                source="scanner",
                timestamp=datetime.now(UTC),
            )

            # Publish to event bus
            await self.event_bus.publish(event)

            # Track as published
            self._published_promotions.add(promotion_key)

            # Record metrics for the promotion
            LayerMetricsCollector.record_layer_promotion(
                symbol=symbol,
                from_layer=from_layer,
                to_layer=to_layer,
                reason=promotion_reason,
                metrics=metrics,
            )

            self.logger.info(
                f"Published symbol promotion: {symbol} from Layer {from_layer.name} "
                f"to Layer {to_layer.name} (reason: {promotion_reason})"
            )

        except Exception as e:
            self.logger.error(f"Failed to publish symbol promotion event: {e}")

    async def publish_batch_qualifications(self, qualifications: list[dict[str, Any]]) -> int:
        """
        Publish multiple symbol qualifications.

        Args:
            qualifications: List of qualification dicts with keys:
                - symbol: str
                - layer: DataLayer
                - reason: str
                - metrics: Optional[Dict]

        Returns:
            Number of events successfully published
        """
        if not self.event_bus:
            self.logger.debug("No event bus configured, skipping batch publication")
            return 0

        published_count = 0

        for qualification in qualifications:
            try:
                await self.publish_symbol_qualified(
                    symbol=qualification["symbol"],
                    layer=qualification["layer"],
                    qualification_reason=qualification.get("reason", "Scanner qualification"),
                    metrics=qualification.get("metrics"),
                )
                published_count += 1
            except Exception as e:
                self.logger.error(
                    f"Failed to publish qualification for {qualification.get('symbol', 'unknown')}: {e}"
                )

        self.logger.info(f"Published {published_count}/{len(qualifications)} qualification events")
        return published_count

    def clear_published_cache(self) -> None:
        """Clear the cache of published events (for new sessions)."""
        self._published_qualifications.clear()
        self._published_promotions.clear()
        self.logger.debug("Cleared published event cache")
