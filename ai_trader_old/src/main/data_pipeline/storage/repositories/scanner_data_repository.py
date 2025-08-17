"""
Scanner Data Repository

Repository for scanner qualification data management.
"""

# Standard library imports
from datetime import UTC, datetime
import json
import time
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import OperationResult, RepositoryConfig
from main.interfaces.repositories.scanner import IScannerDataRepository
from main.utils.core import ensure_utc, get_logger

from .base_repository import BaseRepository
from .helpers import BatchProcessor, CrudExecutor, RepositoryMetricsCollector
from .helpers.pattern_detector import PatternDetector
from .helpers.technical_analyzer import TechnicalAnalyzer

logger = get_logger(__name__)


class ScannerDataRepository(BaseRepository, IScannerDataRepository):
    """
    Repository for scanner qualification data.

    Manages scanner qualifications and delegates complex analysis
    to specialized helper components.
    """

    def __init__(self, db_adapter: IAsyncDatabase, config: RepositoryConfig | None = None):
        """Initialize the ScannerDataRepository."""
        super().__init__(
            db_adapter,
            type("ScannerQualification", (), {"__tablename__": "scanner_qualifications"}),
            config,
        )

        # Core components
        self.crud_executor = CrudExecutor(
            db_adapter,
            "scanner_qualifications",
            transaction_strategy=config.transaction_strategy if config else None,
        )
        self.batch_processor = BatchProcessor(batch_size=config.batch_size if config else 500)
        self.metrics = RepositoryMetricsCollector("ScannerDataRepository")

        # Specialized helpers
        self.technical_analyzer = TechnicalAnalyzer(db_adapter)
        self.pattern_detector = PatternDetector(db_adapter)

        logger.info("ScannerDataRepository initialized")

    def get_required_fields(self) -> list[str]:
        """Get required fields for scanner data."""
        return ["symbol", "qualification_date", "layer", "qualified"]

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        """Validate scanner data record."""
        errors = []

        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")

        if "layer" in record and record["layer"] not in [1, 2, 3]:
            errors.append(f"Invalid layer: {record['layer']}")

        return errors

    async def get_scanner_data(
        self, symbols: list[str], date: datetime, indicators: list[str] | None = None
    ) -> pd.DataFrame:
        """Get scanner data for multiple symbols."""
        start_time = time.time()

        try:
            # Build query
            placeholders = [f"${i+1}" for i in range(len(symbols))]

            query = f"""
                SELECT sq.*, c.name, c.sector, c.industry, c.market_cap
                FROM scanner_qualifications sq
                JOIN companies c ON sq.symbol = c.symbol
                WHERE sq.symbol IN ({','.join(placeholders)})
                AND sq.qualification_date = ${len(symbols) + 1}
                ORDER BY sq.composite_score DESC
            """

            params = [self._normalize_symbol(s) for s in symbols]
            params.append(ensure_utc(date))

            results = await self.db_adapter.fetch_all(query, *params)

            if not results:
                return pd.DataFrame()

            df = pd.DataFrame([dict(r) for r in results])

            # Add indicators if requested (delegate to helper)
            if indicators:
                df = await self.technical_analyzer.add_indicators(df, indicators)

            # Record metrics
            duration = time.time() - start_time
            await self.metrics.record_operation("get_scanner_data", duration, True, len(df))

            return df

        except Exception as e:
            logger.error(f"Error getting scanner data: {e}")
            await self.metrics.record_operation("get_scanner_data", time.time() - start_time, False)
            raise

    async def store_scanner_data(self, data: pd.DataFrame, date: datetime) -> OperationResult:
        """Store scanner data."""
        start_time = time.time()

        try:
            records = []
            qualification_date = ensure_utc(date)

            for _, row in data.iterrows():
                record = {
                    "symbol": self._normalize_symbol(row["symbol"]),
                    "qualification_date": qualification_date,
                    "layer": int(row.get("layer", 1)),
                    "qualified": bool(row.get("qualified", False)),
                    "technical_score": float(row.get("technical_score", 0)),
                    "fundamental_score": float(row.get("fundamental_score", 0)),
                    "sentiment_score": float(row.get("sentiment_score", 0)),
                    "composite_score": float(row.get("composite_score", 0)),
                    "signals": json.dumps(row.get("signals", [])),
                    "metadata": json.dumps(row.get("metadata", {})),
                    "created_at": datetime.now(UTC),
                }
                records.append(record)

            # Process in batches
            result = await self.batch_processor.process_batch(records, self._store_batch)

            # Invalidate cache
            await self._invalidate_cache("scanner_*")

            return OperationResult(
                success=result["success"],
                records_affected=result["statistics"]["succeeded"],
                records_created=result["statistics"]["succeeded"],
                duration_seconds=time.time() - start_time,
                metadata=result["statistics"],
            )

        except Exception as e:
            logger.error(f"Error storing scanner data: {e}")
            return OperationResult(
                success=False, error=str(e), duration_seconds=time.time() - start_time
            )

    async def scan_by_criteria(
        self, criteria: dict[str, Any], date: datetime | None = None, limit: int = 100
    ) -> pd.DataFrame:
        """Scan for symbols matching criteria."""
        try:
            # Get latest date if not specified
            if not date:
                date_query = "SELECT MAX(qualification_date) FROM scanner_qualifications"
                result = await self.db_adapter.fetch_one(date_query)
                date = result.get("max") if result else datetime.now(UTC)
                if date is None:
                    date = datetime.now(UTC)

            # Build dynamic query
            conditions = ["qualification_date = $1"]
            params = [ensure_utc(date)]
            param_count = 2

            # Add criteria conditions
            for field in ["min_technical_score", "min_composite_score", "layer", "qualified"]:
                if field in criteria:
                    col_name = field.replace("min_", "")
                    op = ">=" if "min_" in field else "="
                    conditions.append(f"{col_name} {op} ${param_count}")
                    params.append(criteria[field])
                    param_count += 1

            query = f"""
                SELECT sq.*, c.name, c.sector, c.industry
                FROM scanner_qualifications sq
                JOIN companies c ON sq.symbol = c.symbol
                WHERE {' AND '.join(conditions)}
                ORDER BY composite_score DESC
                LIMIT ${param_count}
            """
            params.append(limit)

            results = await self.db_adapter.fetch_all(query, *params)
            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error scanning by criteria: {e}")
            return pd.DataFrame()

    async def get_technical_indicators(
        self, symbol: str, start_date: datetime, end_date: datetime, indicators: list[str]
    ) -> pd.DataFrame:
        """Get technical indicators for a symbol."""
        # Delegate to technical analyzer
        return await self.technical_analyzer.get_indicators(
            symbol, start_date, end_date, indicators
        )

    async def get_relative_strength(
        self, symbols: list[str], benchmark: str, period_days: int = 30
    ) -> dict[str, float]:
        """Calculate relative strength vs benchmark."""
        # Delegate to technical analyzer
        return await self.technical_analyzer.calculate_relative_strength(
            symbols, benchmark, period_days
        )

    async def get_momentum_scores(
        self, symbols: list[str], lookback_periods: list[int] = [20, 50, 200]
    ) -> pd.DataFrame:
        """Calculate momentum scores."""
        # Delegate to technical analyzer
        return await self.technical_analyzer.calculate_momentum_scores(symbols, lookback_periods)

    async def get_volatility_metrics(self, symbol: str, period_days: int = 30) -> dict[str, float]:
        """Calculate volatility metrics."""
        # Delegate to technical analyzer
        return await self.technical_analyzer.calculate_volatility_metrics(symbol, period_days)

    async def get_support_resistance(
        self, symbol: str, lookback_days: int = 100
    ) -> dict[str, list[float]]:
        """Identify support and resistance levels."""
        # Delegate to technical analyzer
        return await self.technical_analyzer.find_support_resistance(symbol, lookback_days)

    async def get_pattern_signals(
        self, symbol: str, patterns: list[str], lookback_days: int = 30
    ) -> list[dict[str, Any]]:
        """Detect technical patterns."""
        # Delegate to pattern detector
        return await self.pattern_detector.detect_patterns(symbol, patterns, lookback_days)

    async def get_scanner_rankings(
        self, metric: str, date: datetime | None = None, top_n: int = 50, ascending: bool = False
    ) -> pd.DataFrame:
        """Get rankings by a specific metric."""
        try:
            if not date:
                date_query = "SELECT MAX(qualification_date) FROM scanner_qualifications"
                result = await self.db_adapter.fetch_one(date_query)
                date = result.get("max") if result else datetime.now(UTC)
                if date is None:
                    date = datetime.now(UTC)

            order_dir = "ASC" if ascending else "DESC"

            query = f"""
                SELECT
                    sq.*,
                    c.name, c.sector,
                    RANK() OVER (ORDER BY sq.{metric} {order_dir}) as rank
                FROM scanner_qualifications sq
                JOIN companies c ON sq.symbol = c.symbol
                WHERE sq.qualification_date = $1
                ORDER BY sq.{metric} {order_dir}
                LIMIT $2
            """

            results = await self.db_adapter.fetch_all(query, ensure_utc(date), top_n)
            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting scanner rankings: {e}")
            return pd.DataFrame()

    async def get_hot_cold_storage_data(
        self, symbol: str, lookback_days: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get data from hot and cold storage."""
        try:
            cutoff_date = self._get_hot_storage_cutoff()
            end_date = datetime.now(UTC)
            start_date = end_date - pd.Timedelta(days=lookback_days)

            # Hot storage (recent data)
            hot_query = """
                SELECT * FROM scanner_qualifications
                WHERE symbol = $1 AND qualification_date >= $2
                ORDER BY qualification_date DESC
            """

            hot_results = await self.db_adapter.fetch_all(
                hot_query, self._normalize_symbol(symbol), max(start_date, cutoff_date)
            )

            hot_df = pd.DataFrame([dict(r) for r in hot_results]) if hot_results else pd.DataFrame()

            # Cold storage (historical data)
            cold_df = pd.DataFrame()
            if start_date < cutoff_date:
                cold_query = """
                    SELECT * FROM scanner_qualifications_history
                    WHERE symbol = $1
                    AND qualification_date >= $2
                    AND qualification_date < $3
                    ORDER BY qualification_date DESC
                """

                cold_results = await self.db_adapter.fetch_all(
                    cold_query, self._normalize_symbol(symbol), start_date, cutoff_date
                )

                cold_df = (
                    pd.DataFrame([dict(r) for r in cold_results])
                    if cold_results
                    else pd.DataFrame()
                )

            return hot_df, cold_df

        except Exception as e:
            logger.error(f"Error getting hot/cold storage data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    # Private helper methods
    async def _store_batch(self, records: list[dict[str, Any]]) -> Any:
        """Store a batch of records."""
        for record in records:
            # Upsert to current table
            await self._upsert_qualification(record)
            # Append to history
            await self._append_to_history(record)
        return len(records)

    async def _upsert_qualification(self, record: dict[str, Any]) -> None:
        """Upsert scanner qualification record."""
        query = """
            INSERT INTO scanner_qualifications
            (symbol, qualification_date, layer, qualified, technical_score,
             fundamental_score, sentiment_score, composite_score, signals, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (symbol, qualification_date) DO UPDATE
            SET layer = EXCLUDED.layer,
                qualified = EXCLUDED.qualified,
                technical_score = EXCLUDED.technical_score,
                fundamental_score = EXCLUDED.fundamental_score,
                sentiment_score = EXCLUDED.sentiment_score,
                composite_score = EXCLUDED.composite_score,
                signals = EXCLUDED.signals,
                metadata = EXCLUDED.metadata
        """

        await self.db_adapter.execute_query(query, *list(record.values()))

    async def _append_to_history(self, record: dict[str, Any]) -> None:
        """Append record to history table."""
        query = """
            INSERT INTO scanner_qualifications_history
            (symbol, qualification_date, layer, qualified, technical_score,
             fundamental_score, sentiment_score, composite_score, signals, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """

        await self.db_adapter.execute_query(query, *list(record.values()))
