"""
Company Repository

Repository for company data management with layer qualification system.
"""

# Standard library imports
from datetime import UTC, datetime
import time
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import OperationResult, RepositoryConfig
from main.interfaces.repositories.company import ICompanyRepository
from main.utils.core import get_logger

from .base_repository import BaseRepository
from .helpers import (
    BatchProcessor,
    CrudExecutor,
    QueryBuilder,
    RepositoryMetricsCollector,
    validate_table_column,
)

logger = get_logger(__name__)


class CompanyRepository(BaseRepository, ICompanyRepository):
    """
    Repository for company information with layer qualification management.

    Handles company data, layer qualifications (1, 2, 3), and provides
    search and filtering capabilities.
    """

    def __init__(
        self,
        db_adapter: IAsyncDatabase,
        config: RepositoryConfig | None = None,
        event_publisher: Any | None = None,
    ):
        """
        Initialize the CompanyRepository.

        Args:
            db_adapter: Database adapter
            config: Optional repository configuration
            event_publisher: Optional scanner event publisher for layer change events
        """
        # Initialize with companies table
        super().__init__(db_adapter, type("Company", (), {"__tablename__": "companies"}), config)

        # Additional components
        self.query_builder = QueryBuilder("companies")
        self.crud_executor = CrudExecutor(
            db_adapter,
            "companies",
            transaction_strategy=config.transaction_strategy if config else None,
        )
        self.batch_processor = BatchProcessor(
            batch_size=config.batch_size if config else 500,
            max_parallel=config.max_parallel_workers if config else 2,
        )
        self.metrics = RepositoryMetricsCollector(
            "CompanyRepository", enable_metrics=config.enable_metrics if config else True
        )

        # Event publisher for layer change events
        self.event_publisher = event_publisher

        logger.info("CompanyRepository initialized with layer qualification support")

    # ================================================================
    # INITIALIZATION & VALIDATION
    # ================================================================

    def get_required_fields(self) -> list[str]:
        """Get required fields for company data."""
        return ["symbol", "name", "is_active"]

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        """Validate company record."""
        errors = []

        # Check required fields
        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate symbol format
        if record.get("symbol"):
            symbol = record["symbol"]
            if not isinstance(symbol, str) or len(symbol) > 10:
                errors.append("Symbol must be a string of max 10 characters")
            if not symbol.replace("-", "").replace(".", "").isalnum():
                errors.append("Symbol contains invalid characters")

        # Validate layer field (new system)
        if "layer" in record and record["layer"] is not None:
            if not isinstance(record["layer"], int) or record["layer"] < 0 or record["layer"] > 3:
                errors.append("layer must be an integer between 0 and 3")

        return errors

    # ================================================================
    # CORE CRUD OPERATIONS
    # ================================================================

    async def get_company(self, symbol: str) -> dict[str, Any] | None:
        """Get company information by symbol."""
        start_time = time.time()

        try:
            # Check cache
            cache_key = self._get_cache_key(f"company_{symbol}")
            cached_data = await self._get_from_cache(cache_key)

            if cached_data is not None:
                await self.metrics.record_cache_access(hit=True)
                return cached_data

            await self.metrics.record_cache_access(hit=False)

            # Query database
            query = "SELECT * FROM companies WHERE symbol = $1"
            result = await self.db_adapter.fetch_one(query, self._normalize_symbol(symbol))

            if result:
                company_data = dict(result)
                # Cache the result
                await self._set_in_cache(cache_key, company_data)

                # Record metrics
                duration = time.time() - start_time
                await self.metrics.record_operation(
                    "get_company", duration, success=True, records=1
                )

                return company_data

            # Record metrics for not found
            duration = time.time() - start_time
            await self.metrics.record_operation("get_company", duration, success=True, records=0)

            return None

        except Exception as e:
            logger.error(f"Error getting company {symbol}: {e}")
            duration = time.time() - start_time
            await self.metrics.record_operation("get_company", duration, success=False)
            # Re-raise the exception - let it bubble up
            raise

    async def get_companies(
        self, symbols: list[str] | None = None, layer: int | None = None, is_active: bool = True
    ) -> pd.DataFrame:
        """Get multiple companies with optional filtering."""
        start_time = time.time()

        try:
            # Build query
            conditions = []
            params = []
            param_count = 1

            if is_active:
                conditions.append(f"is_active = ${param_count}")
                params.append(True)
                param_count += 1

            if symbols:
                placeholders = [f"${i}" for i in range(param_count, param_count + len(symbols))]
                conditions.append(f"symbol IN ({','.join(placeholders)})")
                params.extend([self._normalize_symbol(s) for s in symbols])
                param_count += len(symbols)

            if layer is not None and 0 <= layer <= 3:
                # Use new layer column
                conditions.append(f"layer = ${param_count}")
                params.append(layer)
                param_count += 1

            where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"SELECT * FROM companies{where_clause} ORDER BY symbol"

            # Execute query
            results = await self.db_adapter.fetch_all(query, *params)

            # Convert to DataFrame
            df = pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

            # Record metrics
            duration = time.time() - start_time
            await self.metrics.record_operation(
                "get_companies", duration, success=True, records=len(df)
            )

            return df

        except Exception as e:
            logger.error(f"Error getting companies: {e}")
            duration = time.time() - start_time
            await self.metrics.record_operation("get_companies", duration, success=False)
            # Return empty DataFrame for data query methods
            return pd.DataFrame()

    # ================================================================
    # LAYER MANAGEMENT
    # ================================================================

    # Method removed - use update_layer() instead

    async def update_layer(
        self, symbol: str, layer: int, reason: str | None = None
    ) -> OperationResult:
        """
        Update a company's layer assignment (new layer system).

        Args:
            symbol: Company symbol
            layer: New layer (0-3)
            reason: Optional reason for the layer change

        Returns:
            OperationResult with success status
        """
        start_time = time.time()

        try:
            if layer not in [0, 1, 2, 3]:
                return OperationResult(
                    success=False, error=f"Invalid layer: {layer}. Must be 0, 1, 2, or 3"
                )

            # First get current layer for transition tracking
            current_query = "SELECT layer FROM companies WHERE symbol = $1"
            current_result = await self.db_adapter.fetch_one(
                current_query, self._normalize_symbol(symbol)
            )
            current_layer = current_result["layer"] if current_result else None

            # Update using new layer column
            query = """
                UPDATE companies
                SET layer = $1,
                    layer_updated_at = $2,
                    layer_reason = $3,
                    updated_at = $4
                WHERE symbol = $5
                RETURNING symbol
            """

            params = [
                layer,
                datetime.now(UTC),
                reason or f"Updated to layer {layer}",
                datetime.now(UTC),
                self._normalize_symbol(symbol),
            ]

            # Execute update
            result = await self.db_adapter.fetch_one(query, *params)

            if result:
                # Record layer transition for event sourcing
                if current_layer != layer:
                    await self._record_layer_transition(
                        symbol=symbol,
                        from_layer=current_layer,
                        to_layer=layer,
                        reason=reason or f"Updated to layer {layer}",
                        transitioned_by="CompanyRepository.update_layer",
                    )

                # Invalidate relevant caches
                await self._invalidate_cache(f"company_{symbol}*")
                await self._invalidate_cache("layer_*")

                # Record metrics
                duration = time.time() - start_time
                await self.metrics.record_operation(
                    "update_layer", duration, success=True, records=1
                )

                logger.info(f"Updated {symbol} to layer {layer}: {reason}")

                return OperationResult(success=True, records_affected=1, duration_seconds=duration)
            else:
                return OperationResult(
                    success=False,
                    error=f"Symbol {symbol} not found",
                    duration_seconds=time.time() - start_time,
                )

        except Exception as e:
            logger.error(f"Error updating layer for {symbol}: {e}")
            return OperationResult(
                success=False, error=str(e), duration_seconds=time.time() - start_time
            )

    async def get_symbols_by_layer(self, layer: int, is_active: bool = True) -> list[str]:
        """
        Get all symbols at a specific layer.

        Args:
            layer: Layer number (0-3)
            is_active: Only return active symbols

        Returns:
            List of symbols at the specified layer
        """
        try:
            if layer not in [0, 1, 2, 3]:
                raise ValueError(f"Invalid layer: {layer}")

            # Check cache
            cache_key = f"layer_{layer}_symbols_active_{is_active}"
            cached_data = await self._get_from_cache(cache_key)

            if cached_data is not None:
                await self.metrics.record_cache_access(hit=True)
                return cached_data

            await self.metrics.record_cache_access(hit=False)

            # Query using new layer column
            query = """
                SELECT symbol FROM companies
                WHERE layer = $2
                AND ($1 = false OR is_active = true)
                ORDER BY symbol
            """

            params = [is_active, layer]

            results = await self.db_adapter.fetch_all(query, *params)

            symbols = [row["symbol"] for row in results] if results else []

            # Cache the results
            await self._set_in_cache(cache_key, symbols, ttl=300)  # 5 minute cache

            return symbols

        except Exception as e:
            logger.error(f"Error getting symbols for layer {layer}: {e}")
            return []

    async def get_symbols_above_layer(self, min_layer: int, is_active: bool = True) -> list[str]:
        """
        Get all symbols at or above a minimum layer.

        Args:
            min_layer: Minimum layer (0-3)
            is_active: Only return active symbols

        Returns:
            List of symbols at or above the minimum layer
        """
        try:
            if min_layer not in [0, 1, 2, 3]:
                raise ValueError(f"Invalid layer: {min_layer}")

            # Build query for symbols >= min_layer
            query = """
                SELECT symbol FROM companies
                WHERE layer >= $1
                AND ($2 = false OR is_active = true)
                ORDER BY layer DESC, symbol
            """

            results = await self.db_adapter.fetch_all(query, min_layer, is_active)

            return [row["symbol"] for row in results] if results else []

        except Exception as e:
            logger.error(f"Error getting symbols above layer {min_layer}: {e}")
            return []

    async def get_layer_qualified_symbols(self, layer: int) -> list[str]:
        """
        Get symbols qualified for a specific layer (backward compatibility).
        This method maintains compatibility with old layer[1-3]_qualified system.
        """
        try:
            if layer not in [1, 2, 3]:
                raise ValueError(f"Invalid layer: {layer}")

            # Use the new method but maintain backward compatibility
            return await self.get_symbols_by_layer(layer, is_active=True)

            # Query database
            query = f"""
                SELECT symbol FROM companies
                WHERE layer{layer}_qualified = TRUE
                AND is_active = TRUE
                ORDER BY symbol
            """

            results = await self.db_adapter.fetch_all(query)
            symbols = [row["symbol"] for row in results]

            # Cache the result
            await self._set_in_cache(cache_key, symbols)

            return symbols

        except Exception as e:
            logger.error(f"Error getting layer {layer} symbols: {e}")
            return []

    # Method removed - use batch update with update_layer() instead

    # ================================================================
    # SECTOR & INDUSTRY QUERIES
    # ================================================================

    async def get_sector_companies(self, sector: str, layer: int | None = None) -> pd.DataFrame:
        """Get companies in a specific sector."""
        try:
            conditions = ["sector = $1", "is_active = TRUE"]
            params = [sector]

            if layer in [1, 2, 3]:
                conditions.append(f"layer{layer}_qualified = ${len(params) + 1}")
                params.append(True)

            query = f"""
                SELECT * FROM companies
                WHERE {' AND '.join(conditions)}
                ORDER BY market_cap DESC NULLS LAST
            """

            results = await self.db_adapter.fetch_all(query, *params)
            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting sector companies: {e}")
            return pd.DataFrame()

    async def get_industry_companies(self, industry: str, layer: int | None = None) -> pd.DataFrame:
        """Get companies in a specific industry."""
        try:
            conditions = ["industry = $1", "is_active = TRUE"]
            params = [industry]

            if layer in [1, 2, 3]:
                conditions.append(f"layer{layer}_qualified = ${len(params) + 1}")
                params.append(True)

            query = f"""
                SELECT * FROM companies
                WHERE {' AND '.join(conditions)}
                ORDER BY market_cap DESC NULLS LAST
            """

            results = await self.db_adapter.fetch_all(query, *params)
            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting industry companies: {e}")
            return pd.DataFrame()

    # ================================================================
    # METADATA & AUXILIARY DATA
    # ================================================================

    async def update_company_metadata(
        self, symbol: str, metadata: dict[str, Any]
    ) -> OperationResult:
        """Update company metadata."""
        try:
            # Build update query dynamically
            set_clauses = []
            params = []
            param_count = 1

            for key, value in metadata.items():
                # Validate column name for SQL injection prevention
                validate_table_column("companies", key)
                set_clauses.append(f"{key} = ${param_count}")
                params.append(value)
                param_count += 1

            # Add updated_at
            set_clauses.append(f"updated_at = ${param_count}")
            params.append(datetime.now(UTC))
            param_count += 1

            # Add symbol condition
            params.append(self._normalize_symbol(symbol))

            query = f"""
                UPDATE companies
                SET {', '.join(set_clauses)}
                WHERE symbol = ${param_count}
            """

            result = await self.crud_executor.execute_update(query, params)

            # Invalidate cache
            await self._invalidate_cache(f"company_{symbol}*")

            return result

        except Exception as e:
            logger.error(f"Error updating company metadata: {e}")
            return OperationResult(success=False, error=str(e))

    # ================================================================
    # ANALYTICS & STATISTICS
    # ================================================================

    async def get_company_statistics(self) -> dict[str, Any]:
        """Get statistics about companies in the database."""
        try:
            query = """
                SELECT
                    COUNT(*) as total_companies,
                    COUNT(CASE WHEN is_active THEN 1 END) as active_companies,
                    COUNT(CASE WHEN layer = 0 THEN 1 END) as layer0_count,
                    COUNT(CASE WHEN layer = 1 THEN 1 END) as layer1_count,
                    COUNT(CASE WHEN layer = 2 THEN 1 END) as layer2_count,
                    COUNT(CASE WHEN layer = 3 THEN 1 END) as layer3_count,
                    COUNT(DISTINCT sector) as unique_sectors,
                    COUNT(DISTINCT industry) as unique_industries
                FROM companies
            """

            result = await self.db_adapter.fetch_one(query)

            if result:
                return dict(result)

            return {}

        except Exception as e:
            logger.error(f"Error getting company statistics: {e}")
            return {}

    # ================================================================
    # SEARCH & DISCOVERY
    # ================================================================

    async def search_companies(
        self, query: str, fields: list[str] | None = None, limit: int = 100
    ) -> pd.DataFrame:
        """Search companies by text query."""
        try:
            # Default search fields
            if not fields:
                fields = ["symbol", "name", "description"]

            # Build search conditions
            conditions = []
            for i, field in enumerate(fields, 1):
                conditions.append(f"{field} ILIKE ${i}")

            search_query = f"""
                SELECT * FROM companies
                WHERE ({' OR '.join(conditions)})
                AND is_active = TRUE
                ORDER BY
                    CASE WHEN symbol ILIKE $1 THEN 0 ELSE 1 END,
                    market_cap DESC NULLS LAST
                LIMIT ${len(fields) + 1}
            """

            # Add wildcards to search term
            search_term = f"%{query}%"
            params = [search_term] * len(fields) + [limit]

            results = await self.db_adapter.fetch_all(search_query, *params)

            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error searching companies: {e}")
            return pd.DataFrame()

    async def _record_layer_transition(
        self,
        symbol: str,
        from_layer: int | None,
        to_layer: int,
        reason: str,
        transitioned_by: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a layer transition in the layer_transitions table.

        This provides event sourcing for layer changes, allowing us to track
        the full history of symbol layer movements.
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
                ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
            """

            # Standard library imports
            import json

            metadata_json = json.dumps(metadata) if metadata else None

            await self.db_adapter.execute(
                insert_query,
                self._normalize_symbol(symbol),
                from_layer,
                to_layer,
                reason,
                metadata_json,
                datetime.now(UTC),
                transitioned_by,
            )

            logger.debug(f"Recorded layer transition: {symbol} from {from_layer} to {to_layer}")

        except Exception as e:
            # Log error but don't fail the main operation
            logger.warning(
                f"Failed to record layer transition for {symbol}: {e}. "
                "Layer transitions table may not exist yet."
            )

    # ================================================================
    # INTERFACE COMPLIANCE METHODS
    # ================================================================

    async def update_layer_qualification(
        self, symbol: str, layer: int, qualified: bool, reason: str | None = None
    ) -> OperationResult:
        """
        Update layer qualification status (maps to update_layer).

        This method exists to satisfy the ICompanyRepository interface.
        In our implementation, qualification is binary - you're either
        at a layer or not, so we map this to update_layer.

        Args:
            symbol: Stock symbol
            layer: Layer number (0-3)
            qualified: If True, sets to this layer; if False, sets to layer-1
            reason: Optional reason for change

        Returns:
            Operation result
        """
        if qualified:
            # Qualify for this layer
            return await self.update_layer(symbol, layer, reason)
        else:
            # Disqualify - move to previous layer
            target_layer = max(0, layer - 1)
            return await self.update_layer(
                symbol, target_layer, reason or f"Disqualified from layer {layer}"
            )

    async def batch_update_layer_qualifications(
        self, updates: list[dict[str, Any]]
    ) -> OperationResult:
        """
        Batch update layer qualifications.

        Args:
            updates: List of update dictionaries with symbol, layer, qualified

        Returns:
            Operation result with update count
        """
        success_count = 0
        failed_count = 0
        errors = []

        for update in updates:
            try:
                symbol = update.get("symbol")
                layer = update.get("layer")
                qualified = update.get("qualified", True)
                reason = update.get("reason")

                if not symbol or layer is None:
                    failed_count += 1
                    errors.append(f"Invalid update: {update}")
                    continue

                result = await self.update_layer_qualification(
                    symbol=symbol, layer=layer, qualified=qualified, reason=reason
                )

                if result.success:
                    success_count += 1
                else:
                    failed_count += 1
                    errors.append(f"Failed to update {symbol}: {result.error}")

            except Exception as e:
                failed_count += 1
                errors.append(f"Error updating {update}: {e}")

        return OperationResult(
            success=failed_count == 0,
            affected_count=success_count,
            error="; ".join(errors) if errors else None,
            metadata={
                "success_count": success_count,
                "failed_count": failed_count,
                "total_updates": len(updates),
            },
        )
