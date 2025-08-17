# File: ai_trader/scanners/layer1_liquidity_filter.py (Final Code)

# Standard library imports
from datetime import UTC, datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.config.database_field_mappings import map_company_fields
from main.data_pipeline.core.enums import DataLayer
from main.events.publishers.scanner_event_publisher import ScannerEventPublisher
from main.interfaces.database import IAsyncDatabase
from main.interfaces.events import IEventBus

logger = logging.getLogger(__name__)


class Layer1LiquidityFilter:
    """
    Filters the static universe down to the most liquid symbols.
    Runs nightly to identify ~1,500 symbols with sufficient liquidity for trading.
    Its sole responsibility is to apply liquidity-based filters.
    """

    def __init__(self, config: DictConfig, db_adapter: IAsyncDatabase, event_bus: IEventBus = None):
        """
        Initializes the Layer 1 Liquidity Filter.

        Args:
            config: The main application configuration object.
            db_adapter: An instance of IAsyncDatabase for database operations.
            event_bus: Optional event bus for publishing layer qualification events.
        """
        self.config = config
        self.db_adapter = db_adapter

        # Initialize event publisher for layer qualification events
        self.event_publisher = ScannerEventPublisher(event_bus) if event_bus else None

        # Load all filter parameters from the configuration
        self.params = self.config.get("universe.layer1_filters", {})
        self.min_avg_dollar_volume = self.params.get("min_avg_dollar_volume", 5_000_000)
        self.min_price = self.params.get("min_price", 1.0)
        self.max_price = self.params.get("max_price", 2000.0)
        self.lookback_days = self.params.get("lookback_days", 20)
        self.target_universe_size = self.params.get("target_universe_size", 2000)

        self.output_dir = Path(self.config.get("paths.universe_dir", "data/universe")) / "layer1"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _validate_prerequisites(self) -> bool:
        """Validate that all prerequisites are met before running scanner."""
        try:
            # Check if market_data view exists
            query = "SELECT 1 FROM market_data LIMIT 1"
            await self.db_adapter.fetch_one(query)
            return True

        except Exception as e:
            if 'relation "market_data" does not exist' in str(e):
                logger.error("❌ Market data view does not exist!")
                logger.error("   Please run: psql -f create_market_data_views.sql")
                return False
            else:
                # Other errors might be OK (e.g., empty table)
                return True

    async def run(self, input_symbols: list[str]) -> list[str]:
        """
        The main public method to execute the liquidity filtering process.

        Args:
            input_symbols: The list of symbols from the Layer 0 scan.

        Returns:
            A list of symbols that passed the liquidity filters.
        """
        logger.info(f"🏛️  Starting Layer 1 Liquidity Filter for {len(input_symbols)} symbols...")
        if not input_symbols:
            logger.warning("Input symbol list is empty. Aborting Layer 1 filter.")
            return []

        # Validate prerequisites
        if not await self._validate_prerequisites():
            raise ValueError("Prerequisites not met for Layer 1 scanner")

        start_time = datetime.now(UTC)

        try:
            # 1. Get liquidity metrics directly from database
            liquidity_data = await self._get_liquidity_metrics(input_symbols)

            # Handle case where no liquidity data is found
            if not liquidity_data:
                logger.error("No liquidity data found in database. This usually means:")
                logger.error(
                    "  1. The market_data view doesn't exist (run create_market_data_views.sql)"
                )
                logger.error("  2. No market data has been loaded yet (run backfill)")
                logger.error("  3. The date range has no data")
                raise ValueError(
                    "Cannot run Layer 1 scanner without market data. Please ensure market data is loaded."
                )

            logger.info(
                f"Successfully calculated liquidity metrics for {len(liquidity_data)} symbols."
            )

            # 2. Apply the filtering logic
            filtered_symbols_data = self._apply_liquidity_filters(liquidity_data)
            logger.info(f"Filtering complete. {len(filtered_symbols_data)} symbols passed.")

            # 3. Sort by the calculated liquidity score and limit the final count
            sorted_symbols_data = sorted(
                filtered_symbols_data, key=lambda x: x.get("liquidity_score", 0), reverse=True
            )
            final_symbols_data = sorted_symbols_data[: self.target_universe_size]

            final_symbols = [s["symbol"] for s in final_symbols_data]

            # 4. Save the output file for the next layer
            await self._save_layer1_output(final_symbols)

            # 5. Update company qualifications in database
            await self._update_company_qualifications(
                qualified_symbols=final_symbols,
                qualified_data=final_symbols_data,
                all_input_symbols=input_symbols,
            )

            duration = (datetime.now(UTC) - start_time).total_seconds()
            logger.info(
                f"✅ Layer 1 complete in {duration:.2f}s. Final universe size: {len(final_symbols)}"
            )

            return final_symbols

        except Exception as e:
            logger.error(f"❌ Layer 1 filter failed: {e}", exc_info=True)
            return []

    async def _get_liquidity_metrics(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Get liquidity metrics directly from database."""
        if not symbols:
            return []

        # Calculate date range
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=self.lookback_days)

        # Query to get liquidity metrics
        query = """
            SELECT
                symbol,
                AVG(close * volume) as avg_dollar_volume,
                AVG(close) as avg_price,
                AVG(volume) as avg_volume,
                COUNT(*) as data_points,
                MAX(close * volume) as max_dollar_volume,
                MIN(close) as min_price,
                MAX(close) as max_price,
                -- Calculate liquidity score as normalized average dollar volume
                AVG(close * volume) / 1000000 as liquidity_score
            FROM market_data
            WHERE symbol = ANY($1::text[])
                AND "timestamp" >= $2
                AND "timestamp" <= $3
                AND close > 0
                AND volume > 0
            GROUP BY symbol
            HAVING COUNT(*) >= $4
        """

        # Require at least 50% of expected data points
        # For minute data, we expect many more data points
        min_data_points = max(5, int(self.lookback_days * 0.5))  # At least 5 data points

        logger.info(f"Querying liquidity data for {len(symbols)} symbols")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Minimum data points required: {min_data_points}")

        try:
            # Execute query directly using connection pool
            async with self.db_adapter._pool.acquire() as conn:
                results = await conn.fetch(
                    query, symbols, start_date, end_date, min_data_points  # $1  # $2  # $3  # $4
                )

            liquidity_data = []
            for row in results:
                liquidity_data.append(
                    {
                        "symbol": row["symbol"],
                        "avg_dollar_volume": (
                            float(row["avg_dollar_volume"]) if row["avg_dollar_volume"] else 0
                        ),
                        "avg_price": float(row["avg_price"]) if row["avg_price"] else 0,
                        "avg_volume": float(row["avg_volume"]) if row["avg_volume"] else 0,
                        "data_points": row["data_points"],
                        "max_dollar_volume": (
                            float(row["max_dollar_volume"]) if row["max_dollar_volume"] else 0
                        ),
                        "min_price": float(row["min_price"]) if row["min_price"] else 0,
                        "max_price": float(row["max_price"]) if row["max_price"] else 0,
                        "liquidity_score": (
                            float(row["liquidity_score"]) if row["liquidity_score"] else 0
                        ),
                    }
                )

            logger.info(f"Query returned {len(liquidity_data)} rows")
            return liquidity_data

        except Exception as e:
            logger.error(f"Error getting liquidity metrics: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            if 'relation "market_data" does not exist' in str(e):
                logger.error(
                    "The market_data view is missing. Please run: create_market_data_views.sql"
                )
            elif "column" in str(e).lower():
                logger.error("Database schema mismatch. The market_data view may be outdated.")
            return []

    def _apply_liquidity_filters(
        self, liquidity_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Applies a series of professional liquidity filters."""
        filtered = []
        for data in liquidity_data:
            # Map database fields to expected field names
            mapped_data = map_company_fields(data, to_db=False)

            # Skip if essential data is missing
            if not all(k in mapped_data for k in ["avg_dollar_volume", "avg_price"]):
                continue

            # Primary Filter: Average Daily Dollar Volume
            if mapped_data["avg_dollar_volume"] < self.min_avg_dollar_volume:
                continue

            # Secondary Filter: Price Range
            if not (self.min_price <= mapped_data["avg_price"] <= self.max_price):
                continue

            # Store the original data (not mapped) to preserve all fields
            filtered.append(data)

        return filtered

    async def _save_layer1_output(self, symbols: list[str]):
        """Saves the final Layer 1 symbol list to a standardized JSON file."""
        output_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "layer": "1",
            "description": "Liquid symbols after volume and price filtering",
            "symbol_count": len(symbols),
            "symbols": symbols,
        }

        output_file = self.output_dir / "layer1_universe.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved Layer 1 output for {len(symbols)} symbols to: {output_file}")

    async def _update_company_qualifications(
        self,
        qualified_symbols: list[str],
        qualified_data: list[dict[str, Any]],
        all_input_symbols: list[str],
    ):
        """
        Update company layer status in database to Layer 1 (LIQUID).

        Args:
            qualified_symbols: Symbols that passed Layer 1
            qualified_data: Full data for qualified symbols (includes liquidity_score)
            all_input_symbols: All symbols that were evaluated
        """
        try:
            # Import repository factory and interface
            # Local imports
            from main.data_pipeline.storage.repositories import get_repository_factory
            from main.interfaces.repositories import ICompanyRepository

            # Create company repository instance using factory
            repo_factory = get_repository_factory()
            company_repo: ICompanyRepository = repo_factory.create_company_repository(
                self.db_adapter
            )

            # Extract liquidity scores from qualified data
            liquidity_scores = {}
            for data in qualified_data:
                symbol = data["symbol"]
                # Use the liquidity_score that was already calculated
                liquidity_scores[symbol] = data.get("liquidity_score", 0.0)

            # Update qualified symbols to Layer 1
            qualified_count = 0
            for symbol in qualified_symbols:
                result = await company_repo.update_layer(
                    symbol=symbol,
                    layer=DataLayer.LIQUID,
                    metadata={
                        "liquidity_score": liquidity_scores.get(symbol, 0.0),
                        "source": "layer1_scanner",
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

                if result.success:
                    qualified_count += 1

                    # Publish qualification event if event publisher is available
                    if self.event_publisher:
                        await self.event_publisher.publish_symbol_qualified(
                            symbol=symbol,
                            layer=DataLayer.LIQUID,
                            qualification_reason="Symbol meets liquidity requirements",
                            metrics={
                                "liquidity_score": liquidity_scores.get(symbol, 0.0),
                                "avg_dollar_volume": self.min_avg_dollar_volume,
                            },
                        )
                else:
                    logger.warning(f"Failed to update layer for {symbol}: {result.errors}")

            # Optionally downgrade non-qualified symbols back to Layer 0
            non_qualified = set(all_input_symbols) - set(qualified_symbols)
            for symbol in non_qualified:
                # Only downgrade if currently at Layer 1 or higher
                # This prevents downgrading symbols that are already at higher layers
                current = await company_repo.get_by_symbol(symbol)
                if current and current.get("layer", 0) == 1:
                    await company_repo.update_layer(
                        symbol=symbol,
                        layer=DataLayer.BASIC,
                        metadata={"reason": "Did not meet Layer 1 liquidity requirements"},
                    )

            logger.info(
                f"✅ Updated Layer 1 qualifications: "
                f"{qualified_count} qualified, "
                f"{len(non_qualified)} non-qualified"
            )

        except Exception as e:
            logger.error(f"Error updating company qualifications: {e}", exc_info=True)
            # Don't fail the entire scan if qualification update fails
            # The scan results are still valid
