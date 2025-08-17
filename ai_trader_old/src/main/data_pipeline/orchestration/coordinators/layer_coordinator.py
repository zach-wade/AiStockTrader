"""
Layer Coordinator

Coordinates layer-based processing using the QualificationService.
"""

# Standard library imports
from dataclasses import dataclass

# Local imports
from main.data_pipeline.core.enums import DataLayer
from main.data_pipeline.services.storage import QualificationConfig, QualificationService
from main.interfaces.database import IAsyncDatabase
from main.utils.core import get_logger


@dataclass
class LayerSymbols:
    """Symbols for each data layer."""

    basic: list[str]
    liquid: list[str]
    catalyst: list[str]
    active: list[str]

    def get_layer(self, layer: DataLayer) -> list[str]:
        """Get symbols for a specific layer."""
        if layer == DataLayer.BASIC:
            return self.basic
        elif layer == DataLayer.LIQUID:
            return self.liquid
        elif layer == DataLayer.CATALYST:
            return self.catalyst
        elif layer == DataLayer.ACTIVE:
            return self.active
        return []


class LayerCoordinator:
    """
    Coordinates layer-based symbol management.

    Uses QualificationService to determine which symbols belong
    to which processing layers.
    """

    def __init__(
        self, db_adapter: IAsyncDatabase, qualification_service: QualificationService | None = None
    ):
        """
        Initialize the layer coordinator.

        Args:
            db_adapter: Database adapter
            qualification_service: Optional qualification service
        """
        self.db_adapter = db_adapter
        self.qualification_service = qualification_service or QualificationService(
            db_adapter=db_adapter, config=QualificationConfig()
        )
        self.logger = get_logger(__name__)

    async def get_layer_symbols(
        self, max_per_layer: dict[DataLayer, int] | None = None
    ) -> LayerSymbols:
        """
        Get symbols for each data layer.

        Args:
            max_per_layer: Optional max symbols per layer

        Returns:
            LayerSymbols containing symbols for each layer
        """
        # Default limits if not specified (using DataLayer max_symbols)
        if max_per_layer is None:
            max_per_layer = {
                DataLayer.BASIC: DataLayer.BASIC.max_symbols,
                DataLayer.LIQUID: DataLayer.LIQUID.max_symbols,
                DataLayer.CATALYST: DataLayer.CATALYST.max_symbols,
                DataLayer.ACTIVE: DataLayer.ACTIVE.max_symbols,
            }

        # Get qualified symbols from database
        basic = await self._get_symbols_for_layer(
            DataLayer.BASIC, max_per_layer.get(DataLayer.BASIC, 10000)
        )
        liquid = await self._get_symbols_for_layer(
            DataLayer.LIQUID, max_per_layer.get(DataLayer.LIQUID, 2000)
        )
        catalyst = await self._get_symbols_for_layer(
            DataLayer.CATALYST, max_per_layer.get(DataLayer.CATALYST, 500)
        )
        active = await self._get_symbols_for_layer(
            DataLayer.ACTIVE, max_per_layer.get(DataLayer.ACTIVE, 50)
        )

        self.logger.info(
            f"Retrieved symbols - Basic: {len(basic)}, Liquid: {len(liquid)}, "
            f"Catalyst: {len(catalyst)}, Active: {len(active)}"
        )

        return LayerSymbols(basic=basic, liquid=liquid, catalyst=catalyst, active=active)

    async def _get_symbols_for_layer(self, layer: DataLayer, limit: int) -> list[str]:
        """
        Get symbols qualified for a specific layer.

        Args:
            layer: DataLayer enum value
            limit: Maximum symbols to return

        Returns:
            List of qualified symbols
        """
        # Map DataLayer to qualified column names
        # Note: This assumes the DB schema uses layer0_qualified, layer1_qualified, etc.
        # Use the new layer column with integer values (0-3)
        layer_value = layer.value  # DataLayer enum value (0, 1, 2, or 3)

        query = """
            SELECT symbol
            FROM companies
            WHERE layer = %s
            AND is_active = TRUE
            ORDER BY symbol
            LIMIT %s
        """

        rows = await self.db_adapter.fetch_all(query, layer_value, limit)
        return [row["symbol"] for row in rows]

    async def check_symbol_qualification(self, symbol: str) -> int | None:
        """
        Check which layer a symbol is qualified for.

        Args:
            symbol: Symbol to check

        Returns:
            Highest layer number the symbol is qualified for, or None
        """
        qualification = await self.qualification_service.get_symbol_qualification(symbol)

        if qualification:
            return qualification.layer_qualified

        return None
