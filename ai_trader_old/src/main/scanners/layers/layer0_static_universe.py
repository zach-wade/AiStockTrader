# File: ai_trader/scanners/layer0_static_universe.py (Final Code)

# Standard library imports
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re
from typing import Any

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.data_pipeline.core.enums import DataLayer
from main.data_pipeline.ingestion.alpaca_assets_client import AlpacaAssetsClient
from main.data_pipeline.storage.archive import DataArchive
from main.data_pipeline.storage.archive_initializer import get_archive
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.data_pipeline.storage.repositories import get_repository_factory
from main.events.publishers.scanner_event_publisher import ScannerEventPublisher
from main.interfaces.events import IEventBus
from main.interfaces.repositories import ICompanyRepository

logger = logging.getLogger(__name__)


class Layer0StaticUniverseScanner:
    """
    Builds the static universe of all potentially tradable US equities and ETFs.
    This is the foundational layer of the screening funnel. Its only job is to
    fetch all assets from a primary source and apply basic, non-financial filters.
    """

    def __init__(self, config: DictConfig, event_bus: IEventBus = None):
        """Initializes the Layer 0 scanner.

        Args:
            config: Configuration object
            event_bus: Optional event bus for publishing layer qualification events
        """
        self.config = config
        # The scanner should be initialized with the clients it needs
        self.alpaca_client = AlpacaAssetsClient(config)
        self.archive: DataArchive = get_archive()

        # Initialize database adapter and company repository using factory
        db_factory = DatabaseFactory()
        self.db_adapter = db_factory.create_async_database(config)
        repo_factory = get_repository_factory()
        self.company_repository: ICompanyRepository = repo_factory.create_company_repository(
            self.db_adapter
        )

        # Initialize event publisher for layer qualification events
        self.event_publisher = ScannerEventPublisher(event_bus) if event_bus else None

        # Load filter criteria from config for flexibility
        self.filter_config = self.config.get("universe.layer0_filters", {})
        self.excluded_exchanges = self.filter_config.get(
            "excluded_exchanges", ["OTC", "OTCBB", "PINK", "GREY"]
        )
        self.excluded_patterns = self.filter_config.get(
            "excluded_patterns", ["W", "RT", "CV", "Q"]
        )  # Warrants, Rights, Convertibles, Bankrupt
        self.included_asset_classes = self.filter_config.get(
            "included_asset_classes", ["us_equity"]
        )

        self.output_dir = Path(self.config.get("paths.universe_dir", "data/universe")) / "layer0"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> list[dict[str, Any]]:
        """
        The main public method to generate the Layer 0 static universe.

        Returns:
            A list of asset dictionaries that passed the initial filters.
        """
        logger.info("🏛️ Starting Layer 0 Static Universe scan...")
        start_time = datetime.now(UTC)

        try:
            # 1. Fetch all tradable assets from the primary source (Alpaca)
            all_assets = await self.alpaca_client.get_all_tradable_assets()
            logger.info(f"Fetched {len(all_assets)} total assets from Alpaca.")

            # 2. Apply basic, non-financial filters
            # This logic is now self-contained and testable.
            filtered_assets = self._apply_static_filters(all_assets)

            # 3. Save the raw asset data to the Data Lake for archival
            await self._archive_raw_assets(filtered_assets)

            # 4. Populate companies table in database
            companies_result = await self._populate_companies_table(filtered_assets)
            logger.info(
                f"Database population result: {companies_result.processed_records} companies processed, success: {companies_result.success}"
            )

            # 5. Extract the final list of symbols
            final_symbols = [asset["symbol"] for asset in filtered_assets]

            # 5a. Update all symbols to Layer 0 (BASIC)
            await self._update_layer_qualifications(final_symbols)

            # 6. Save the final list of symbols to the layer's output file (for backward compatibility)
            await self._save_layer0_output(final_symbols, len(all_assets))

            duration = (datetime.now(UTC) - start_time).total_seconds()
            logger.info(
                f"✅ Layer 0 complete in {duration:.2f}s. Found {len(final_symbols)} tradable symbols."
            )

            return final_symbols

        except Exception as e:
            logger.error(f"❌ Layer 0 scan failed: {e}", exc_info=True)
            return []

    def _apply_static_filters(self, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Applies a strict set of non-financial filters to create the static universe.
        This includes exchange, asset class, and ticker format checks.
        """
        filtered = []
        logger.info(f"Applying static filters to {len(assets)} raw assets...")

        for asset in assets:
            symbol = asset.get("symbol", "")
            exchange = str(asset.get("exchange", "")).upper()
            asset_class = asset.get("asset_class", "")
            status = asset.get("status", "")

            # Filter 1: Must be active and tradable
            if status != "active" or not asset.get("tradable", False):
                continue

            # Filter 2: Must be in an included asset class
            if asset_class not in self.included_asset_classes:
                continue

            # Filter 3: Exclude OTC and other non-major exchanges
            if exchange in self.excluded_exchanges:
                continue

            # Filter 4: Exclude tickers with patterns indicating non-common stock
            if any(pattern in symbol for pattern in self.excluded_patterns):
                continue

            # Filter 5: Simple regex for standard ticker format (1-5 uppercase letters)
            if not re.match(r"^[A-Z]{1,5}$", symbol):
                continue

            filtered.append(asset)

        logger.info(f"Filtering complete. {len(filtered)} symbols passed Layer 0 checks.")
        return filtered

    async def _archive_raw_assets(self, assets: list[dict[str, Any]]):
        """Saves the raw asset data from the API to the Data Lake."""
        if not assets:
            return

        # Save as a universe snapshot
        timestamp = datetime.now(UTC)
        snapshot_data = {
            "timestamp": timestamp.isoformat(),
            "source": "alpaca",
            "asset_count": len(assets),
            "assets": assets,
        }

        # Use a date-based key for universe snapshots
        key = f"universe/snapshots/{timestamp.strftime('%Y/%m/%d')}/alpaca_assets_{timestamp.strftime('%H%M%S')}.json"

        # Import the ArchiveDataType enum
        # Local imports
        from main.data_pipeline.storage.data_archiver_types import ArchiveDataType

        self.archive.save(
            key=key,
            data=snapshot_data,
            data_type=ArchiveDataType.OTHER,
            format="json",
            metadata={
                "source": "alpaca",
                "data_type": "universe_snapshot",
                "asset_count": str(len(assets)),
                "timestamp": timestamp.isoformat(),
            },
        )
        logger.info(f"Archived {len(assets)} assets as universe snapshot")

    async def _save_layer0_output(self, symbols: list[str], raw_count: int):
        """Saves the final Layer 0 symbol list to a JSON file for the next layer."""
        output_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "layer": "0",
            "description": "Static universe of all potentially tradable US equities.",
            "raw_asset_count": raw_count,
            "final_symbol_count": len(symbols),
            "symbols": symbols,
        }

        output_file = self.output_dir / "layer0_universe.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved Layer 0 output for {len(symbols)} symbols to: {output_file}")

    async def _populate_companies_table(self, assets_data: list[dict[str, Any]]):
        """
        Populate companies table from Layer 0 assets data using the existing repository.

        Args:
            assets_data: List of asset dictionaries from Layer 0 scan

        Returns:
            OperationResult from the repository
        """
        if not assets_data:
            return None

        # Transform assets data to company records format
        company_records = []
        for asset in assets_data:
            company_record = {
                "symbol": asset.get("symbol", ""),
                "name": asset.get("name", ""),
                "exchange": asset.get("exchange", ""),
                "quote_type": asset.get("asset_class", "stock"),
                "is_active": True,
                # Initialize layer to 0 (BASIC) - will be updated by subsequent scanners
                "layer": 0,
                # Initialize scores to 0
                "liquidity_score": 0.0,
                "catalyst_score": 0.0,
                "premarket_score": 0.0,
            }

            # Only add valid records with symbol
            if company_record["symbol"]:
                company_records.append(company_record)

        if not company_records:
            logger.warning("No valid company records to insert")
            return None

        # Use the repository's bulk_upsert method
        logger.info(f"Upserting {len(company_records)} companies to database")
        return await self.company_repository.bulk_upsert(
            records=company_records,
            constraint_name=None,  # Will use default from repository
            update_fields=[
                "name",
                "exchange",
                "quote_type",
                "is_active",
                "layer",
            ],  # Update these fields on conflict
        )

    async def _update_layer_qualifications(self, symbols: list[str]):
        """Update layer qualifications for all symbols in Layer 0.

        Args:
            symbols: List of symbols to update to Layer 0
        """
        try:
            logger.info(f"Updating {len(symbols)} symbols to Layer 0 (BASIC)")

            # Update all symbols to Layer 0 using the new layer column
            for symbol in symbols:
                result = await self.company_repository.update_layer(
                    symbol=symbol,
                    layer=DataLayer.BASIC,
                    metadata={
                        "source": "layer0_scanner",
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

                if result.success:
                    # Publish qualification event if event publisher is available
                    if self.event_publisher:
                        await self.event_publisher.publish_symbol_qualified(
                            symbol=symbol,
                            layer=DataLayer.BASIC,
                            qualification_reason="Symbol included in static universe",
                            metrics={"source": "alpaca", "asset_class": "us_equity"},
                        )
                else:
                    logger.warning(f"Failed to update layer for {symbol}: {result.errors}")

            logger.info(f"✅ Updated all {len(symbols)} symbols to Layer 0")

        except Exception as e:
            logger.error(f"Error updating layer qualifications: {e}", exc_info=True)
