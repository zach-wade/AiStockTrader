"""
Financials Repository

Repository for financial statement data management.
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
from main.interfaces.repositories.financials import IFinancialsRepository
from main.utils.core import ensure_utc, get_logger

from .base_repository import BaseRepository
from .helpers import BatchProcessor, CrudExecutor, RepositoryMetricsCollector

logger = get_logger(__name__)


class FinancialsRepository(BaseRepository, IFinancialsRepository):
    """
    Repository for financial statements and fundamental data.

    Handles income statements, balance sheets, cash flows, and ratios.
    """

    def __init__(self, db_adapter: IAsyncDatabase, config: RepositoryConfig | None = None):
        """Initialize the FinancialsRepository."""
        super().__init__(
            db_adapter, type("Financials", (), {"__tablename__": "financials_data"}), config
        )

        self.crud_executor = CrudExecutor(
            db_adapter,
            "financials_data",
            transaction_strategy=config.transaction_strategy if config else None,
        )
        self.batch_processor = BatchProcessor(batch_size=config.batch_size if config else 100)
        self.metrics = RepositoryMetricsCollector("FinancialsRepository")

        logger.info("FinancialsRepository initialized")

    def get_required_fields(self) -> list[str]:
        """Get required fields for financial data."""
        return ["symbol", "period_end", "period_type", "filing_date"]

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        """Validate financial record."""
        errors = []

        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate period type
        if "period_type" in record:
            if record["period_type"] not in ["annual", "quarterly", "ttm"]:
                errors.append(f"Invalid period_type: {record['period_type']}")

        # Validate numeric fields
        numeric_fields = ["revenue", "net_income", "total_assets", "total_liabilities"]
        for field in numeric_fields:
            if field in record and record[field] is not None:
                if not isinstance(record[field], (int, float)):
                    errors.append(f"{field} must be numeric")

        return errors

    # IFinancialsRepository interface implementation
    async def get_financial_statements(
        self, symbol: str, statement_type: str, period: str = "quarterly", limit: int = 8
    ) -> pd.DataFrame:
        """Get financial statements for a symbol (interface method)."""
        # Map to existing get_financials method
        return await self.get_financials(symbol, period, limit)

    async def store_financial_statement(
        self, symbol: str, statement_type: str, period_date: datetime, data: dict[str, Any]
    ) -> OperationResult:
        """Store a financial statement (interface method)."""
        # Prepare data with required fields
        statement_data = {
            "symbol": symbol,
            "period_end": period_date,
            "period_type": "quarterly" if statement_type != "annual" else "annual",
            "filing_date": period_date,  # Use period_date as filing_date if not provided
            **data,
        }

        # Use existing store_financials method
        return await self.store_financials([statement_data])

    async def get_latest_financials(self, symbol: str) -> dict[str, Any] | None:
        """Get latest financial data for a symbol (interface method)."""
        try:
            query = """
                SELECT * FROM financials_data
                WHERE symbol = $1
                ORDER BY period_end DESC
                LIMIT 1
            """

            result = await self.db_adapter.fetch_one(query, self._normalize_symbol(symbol))

            if result:
                return dict(result)

            return None

        except Exception as e:
            logger.error(f"Error getting latest financials for {symbol}: {e}")
            raise

    async def get_financials(
        self, symbol: str, period_type: str = "quarterly", limit: int = 8
    ) -> pd.DataFrame:
        """Get financial statements for a symbol."""
        start_time = time.time()

        try:
            query = """
                SELECT * FROM financials_data
                WHERE symbol = $1 AND period_type = $2
                ORDER BY period_end DESC
                LIMIT $3
            """

            results = await self.db_adapter.fetch_all(
                query, self._normalize_symbol(symbol), period_type, limit
            )

            df = pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

            # Record metrics
            duration = time.time() - start_time
            await self.metrics.record_operation("get_financials", duration, True, len(df))

            return df

        except Exception as e:
            logger.error(f"Error getting financials: {e}")
            await self.metrics.record_operation("get_financials", time.time() - start_time, False)
            return pd.DataFrame()

    async def store_financials(self, financials: list[dict[str, Any]]) -> OperationResult:
        """Store financial statements."""
        start_time = time.time()

        try:
            records = []
            for statement in financials:
                record = self._prepare_financial_record(statement)
                records.append(record)

            # Process in batches
            async def store_batch(batch: list[dict[str, Any]]) -> Any:
                for record in batch:
                    await self._upsert_financial(record)
                return len(batch)

            result = await self.batch_processor.process_batch(records, store_batch)

            return OperationResult(
                success=result["success"],
                records_affected=result["statistics"]["succeeded"],
                records_created=result["statistics"]["succeeded"],
                duration_seconds=time.time() - start_time,
                metadata=result["statistics"],
            )

        except Exception as e:
            logger.error(f"Error storing financials: {e}")
            return OperationResult(
                success=False, error=str(e), duration_seconds=time.time() - start_time
            )

    async def get_latest_financials(
        self, symbol: str, period_type: str = "quarterly"
    ) -> dict[str, Any] | None:
        """Get the latest financial statement."""
        try:
            query = """
                SELECT * FROM financials_data
                WHERE symbol = $1 AND period_type = $2
                ORDER BY period_end DESC
                LIMIT 1
            """

            result = await self.db_adapter.fetch_one(
                query, self._normalize_symbol(symbol), period_type
            )

            return dict(result) if result else None

        except Exception as e:
            logger.error(f"Error getting latest financials: {e}")
            raise

    async def get_financial_ratios(self, symbol: str, period_end: datetime) -> dict[str, float]:
        """Calculate financial ratios for a period."""
        try:
            query = """
                SELECT * FROM financials_data
                WHERE symbol = $1 AND period_end = $2
                LIMIT 1
            """

            result = await self.db_adapter.fetch_one(
                query, self._normalize_symbol(symbol), ensure_utc(period_end)
            )

            if not result:
                return {}

            data = dict(result)
            return self._calculate_ratios(data)

        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
            return {}

    async def get_growth_metrics(self, symbol: str, periods: int = 4) -> dict[str, float]:
        """Calculate growth metrics over periods."""
        try:
            query = """
                SELECT period_end, revenue, net_income, eps_basic
                FROM financials_data
                WHERE symbol = $1 AND period_type = 'quarterly'
                ORDER BY period_end DESC
                LIMIT $2
            """

            results = await self.db_adapter.fetch_all(
                query, self._normalize_symbol(symbol), periods + 1
            )

            if len(results) < 2:
                return {}

            # Calculate year-over-year growth
            latest = dict(results[0])
            year_ago = dict(results[min(4, len(results) - 1)])

            growth = {}

            # Revenue growth
            if latest["revenue"] and year_ago["revenue"]:
                growth["revenue_growth"] = (
                    (latest["revenue"] - year_ago["revenue"]) / year_ago["revenue"]
                ) * 100

            # Net income growth
            if latest["net_income"] and year_ago["net_income"]:
                growth["net_income_growth"] = (
                    (latest["net_income"] - year_ago["net_income"]) / year_ago["net_income"]
                ) * 100

            # EPS growth
            if latest["eps_basic"] and year_ago["eps_basic"]:
                growth["eps_growth"] = (
                    (latest["eps_basic"] - year_ago["eps_basic"]) / year_ago["eps_basic"]
                ) * 100

            return growth

        except Exception as e:
            logger.error(f"Error calculating growth metrics: {e}")
            return {}

    async def compare_financials(
        self, symbols: list[str], period_type: str = "quarterly"
    ) -> pd.DataFrame:
        """Compare latest financials across symbols."""
        try:
            data = []

            for symbol in symbols:
                latest = await self.get_latest_financials(symbol, period_type)
                if latest:
                    # Extract key metrics
                    data.append(
                        {
                            "symbol": symbol,
                            "period_end": latest["period_end"],
                            "revenue": latest.get("revenue"),
                            "net_income": latest.get("net_income"),
                            "profit_margin": (
                                latest["net_income"] / latest["revenue"] * 100
                                if latest.get("revenue") and latest.get("net_income")
                                else None
                            ),
                            "roe": latest.get("return_on_equity"),
                            "debt_to_equity": latest.get("debt_to_equity"),
                            "current_ratio": latest.get("current_ratio"),
                        }
                    )

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error comparing financials: {e}")
            return pd.DataFrame()

    async def get_earnings_history(self, symbol: str, quarters: int = 8) -> pd.DataFrame:
        """Get earnings history with estimates vs actuals."""
        try:
            query = """
                SELECT
                    period_end,
                    eps_basic as actual_eps,
                    eps_estimate,
                    revenue as actual_revenue,
                    revenue_estimate,
                    CASE
                        WHEN eps_estimate IS NOT NULL
                        THEN eps_basic > eps_estimate
                        ELSE NULL
                    END as eps_beat,
                    CASE
                        WHEN revenue_estimate IS NOT NULL
                        THEN revenue > revenue_estimate
                        ELSE NULL
                    END as revenue_beat
                FROM financials_data
                WHERE symbol = $1 AND period_type = 'quarterly'
                ORDER BY period_end DESC
                LIMIT $2
            """

            results = await self.db_adapter.fetch_all(
                query, self._normalize_symbol(symbol), quarters
            )

            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting earnings history: {e}")
            return pd.DataFrame()

    async def cleanup_old_financials(self, years_to_keep: int = 10) -> OperationResult:
        """Clean up old financial data."""
        try:
            cutoff_date = datetime.now(UTC).replace(year=datetime.now().year - years_to_keep)

            query = "DELETE FROM financials_data WHERE period_end < $1"
            result = await self.crud_executor.execute_delete(query, [cutoff_date])

            logger.info(f"Cleaned up {result.records_deleted} old financial records")

            return result

        except Exception as e:
            logger.error(f"Error cleaning up old financials: {e}")
            return OperationResult(success=False, error=str(e))

    # Private helper methods
    def _prepare_financial_record(self, statement: dict[str, Any]) -> dict[str, Any]:
        """Prepare financial record for storage."""
        return {
            "symbol": self._normalize_symbol(statement["symbol"]),
            "period_end": ensure_utc(statement["period_end"]),
            "period_type": statement["period_type"],
            "filing_date": ensure_utc(statement["filing_date"]),
            "fiscal_year": statement.get("fiscal_year"),
            "fiscal_quarter": statement.get("fiscal_quarter"),
            # Income statement
            "revenue": statement.get("revenue"),
            "cost_of_revenue": statement.get("cost_of_revenue"),
            "gross_profit": statement.get("gross_profit"),
            "operating_expenses": statement.get("operating_expenses"),
            "operating_income": statement.get("operating_income"),
            "net_income": statement.get("net_income"),
            "eps_basic": statement.get("eps_basic"),
            "eps_diluted": statement.get("eps_diluted"),
            "shares_outstanding": statement.get("shares_outstanding"),
            # Balance sheet
            "total_assets": statement.get("total_assets"),
            "current_assets": statement.get("current_assets"),
            "total_liabilities": statement.get("total_liabilities"),
            "current_liabilities": statement.get("current_liabilities"),
            "total_equity": statement.get("total_equity"),
            "retained_earnings": statement.get("retained_earnings"),
            # Cash flow
            "operating_cash_flow": statement.get("operating_cash_flow"),
            "investing_cash_flow": statement.get("investing_cash_flow"),
            "financing_cash_flow": statement.get("financing_cash_flow"),
            "free_cash_flow": statement.get("free_cash_flow"),
            # Ratios (pre-calculated if available)
            "profit_margin": statement.get("profit_margin"),
            "return_on_assets": statement.get("return_on_assets"),
            "return_on_equity": statement.get("return_on_equity"),
            "debt_to_equity": statement.get("debt_to_equity"),
            "current_ratio": statement.get("current_ratio"),
            "quick_ratio": statement.get("quick_ratio"),
            # Estimates
            "eps_estimate": statement.get("eps_estimate"),
            "revenue_estimate": statement.get("revenue_estimate"),
            "created_at": datetime.now(UTC),
        }

    def _calculate_ratios(self, data: dict[str, Any]) -> dict[str, float]:
        """Calculate financial ratios from statement data."""
        ratios = {}

        # Profitability ratios
        if data.get("revenue") and data.get("net_income"):
            ratios["profit_margin"] = data["net_income"] / data["revenue"]

        if data.get("total_assets") and data.get("net_income"):
            ratios["return_on_assets"] = data["net_income"] / data["total_assets"]

        if data.get("total_equity") and data.get("net_income"):
            ratios["return_on_equity"] = data["net_income"] / data["total_equity"]

        # Liquidity ratios
        if data.get("current_assets") and data.get("current_liabilities"):
            ratios["current_ratio"] = data["current_assets"] / data["current_liabilities"]

            # Quick ratio (assuming inventory is small portion)
            ratios["quick_ratio"] = ratios["current_ratio"] * 0.8

        # Leverage ratios
        if data.get("total_liabilities") and data.get("total_equity"):
            ratios["debt_to_equity"] = data["total_liabilities"] / data["total_equity"]

        if data.get("total_assets") and data.get("total_liabilities"):
            ratios["debt_to_assets"] = data["total_liabilities"] / data["total_assets"]

        return ratios

    async def _upsert_financial(self, record: dict[str, Any]) -> None:
        """Upsert a financial statement."""
        # Build dynamic query based on available fields
        fields = [k for k, v in record.items() if v is not None]
        placeholders = [f"${i+1}" for i in range(len(fields))]

        update_fields = [
            f"{f} = EXCLUDED.{f}"
            for f in fields
            if f not in ["symbol", "period_end", "period_type"]
        ]

        query = f"""
            INSERT INTO financials_data ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (symbol, period_end, period_type) DO UPDATE
            SET {', '.join(update_fields)}
        """

        values = [record[f] for f in fields]
        await self.db_adapter.execute_query(query, *values)
