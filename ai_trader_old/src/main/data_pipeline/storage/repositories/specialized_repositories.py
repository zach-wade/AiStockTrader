"""
Specialized Repositories

Additional repositories for specific data types.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
import json
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.interfaces.database import IAsyncDatabase
from main.interfaces.repositories.base import OperationResult, RepositoryConfig
from main.utils.core import ensure_utc, get_logger

from .base_repository import BaseRepository
from .helpers import BatchProcessor, CrudExecutor, RepositoryMetricsCollector

logger = get_logger(__name__)


class SentimentRepository(BaseRepository):
    """Repository for sentiment analysis data."""

    def __init__(self, db_adapter: IAsyncDatabase, config: RepositoryConfig | None = None):
        super().__init__(
            db_adapter, type("Sentiment", (), {"__tablename__": "sentiment_data"}), config
        )
        self.crud_executor = CrudExecutor(db_adapter, "sentiment_data")
        self.batch_processor = BatchProcessor()
        self.metrics = RepositoryMetricsCollector("SentimentRepository")

    def get_required_fields(self) -> list[str]:
        return ["symbol", "timestamp", "sentiment_score", "source"]

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        errors = []
        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")

        if "sentiment_score" in record:
            score = record["sentiment_score"]
            if not isinstance(score, (int, float)) or score < -1 or score > 1:
                errors.append("sentiment_score must be between -1 and 1")

        return errors

    async def get_sentiment_history(
        self, symbol: str, days: int = 30, source: str | None = None
    ) -> pd.DataFrame:
        """Get sentiment history for a symbol."""
        try:
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=days)

            conditions = ["symbol = $1", "timestamp >= $2", "timestamp <= $3"]
            params = [self._normalize_symbol(symbol), start_date, end_date]

            if source:
                conditions.append(f"source = ${len(params) + 1}")
                params.append(source)

            query = f"""
                SELECT * FROM sentiment_data
                WHERE {' AND '.join(conditions)}
                ORDER BY timestamp DESC
            """

            results = await self.db_adapter.fetch_all(query, *params)
            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting sentiment history: {e}")
            return pd.DataFrame()

    async def store_sentiment(self, sentiment_data: list[dict[str, Any]]) -> OperationResult:
        """Store sentiment data."""
        try:
            records = []
            for item in sentiment_data:
                records.append(
                    {
                        "symbol": self._normalize_symbol(item["symbol"]),
                        "timestamp": ensure_utc(item["timestamp"]),
                        "sentiment_score": item["sentiment_score"],
                        "source": item["source"],
                        "confidence": item.get("confidence"),
                        "volume": item.get("volume"),
                        "metadata": json.dumps(item.get("metadata", {})),
                        "created_at": datetime.now(UTC),
                    }
                )

            async def store_batch(batch):
                for record in batch:
                    query = """
                        INSERT INTO sentiment_data
                        (symbol, timestamp, sentiment_score, source, confidence, volume, metadata, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (symbol, timestamp, source) DO UPDATE
                        SET sentiment_score = EXCLUDED.sentiment_score,
                            confidence = EXCLUDED.confidence,
                            volume = EXCLUDED.volume,
                            metadata = EXCLUDED.metadata
                    """
                    await self.db_adapter.execute_query(query, *list(record.values()))
                return len(batch)

            result = await self.batch_processor.process_batch(records, store_batch)

            return OperationResult(
                success=result["success"], records_affected=result["statistics"]["succeeded"]
            )

        except Exception as e:
            logger.error(f"Error storing sentiment: {e}")
            return OperationResult(success=False, error=str(e))


class RatingsRepository(BaseRepository):
    """Repository for analyst ratings data."""

    def __init__(self, db_adapter: IAsyncDatabase, config: RepositoryConfig | None = None):
        super().__init__(
            db_adapter, type("Rating", (), {"__tablename__": "analyst_ratings"}), config
        )
        self.crud_executor = CrudExecutor(db_adapter, "analyst_ratings")
        self.metrics = RepositoryMetricsCollector("RatingsRepository")

    def get_required_fields(self) -> list[str]:
        return ["symbol", "date", "analyst", "rating", "price_target"]

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        errors = []
        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")

        if "rating" in record:
            valid_ratings = ["buy", "strong_buy", "hold", "sell", "strong_sell"]
            if record["rating"].lower() not in valid_ratings:
                errors.append(f"Invalid rating: {record['rating']}")

        return errors

    async def get_latest_ratings(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """Get latest analyst ratings."""
        try:
            query = """
                SELECT * FROM analyst_ratings
                WHERE symbol = $1
                ORDER BY date DESC
                LIMIT $2
            """

            results = await self.db_adapter.fetch_all(query, self._normalize_symbol(symbol), limit)

            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting ratings: {e}")
            return pd.DataFrame()

    async def get_consensus_rating(self, symbol: str, days: int = 90) -> dict[str, Any]:
        """Get consensus rating for a symbol."""
        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=days)

            query = """
                SELECT
                    COUNT(*) as total_ratings,
                    AVG(price_target) as avg_price_target,
                    COUNT(CASE WHEN rating IN ('buy', 'strong_buy') THEN 1 END) as buy_count,
                    COUNT(CASE WHEN rating = 'hold' THEN 1 END) as hold_count,
                    COUNT(CASE WHEN rating IN ('sell', 'strong_sell') THEN 1 END) as sell_count
                FROM analyst_ratings
                WHERE symbol = $1 AND date >= $2
            """

            result = await self.db_adapter.fetch_one(
                query, self._normalize_symbol(symbol), cutoff_date
            )

            if result and result["total_ratings"] > 0:
                return {
                    "symbol": symbol,
                    "total_ratings": result["total_ratings"],
                    "avg_price_target": (
                        float(result["avg_price_target"]) if result["avg_price_target"] else None
                    ),
                    "buy_percentage": result["buy_count"] / result["total_ratings"] * 100,
                    "hold_percentage": result["hold_count"] / result["total_ratings"] * 100,
                    "sell_percentage": result["sell_count"] / result["total_ratings"] * 100,
                    "consensus": self._calculate_consensus(
                        result["buy_count"], result["hold_count"], result["sell_count"]
                    ),
                }

            return {}

        except Exception as e:
            logger.error(f"Error getting consensus rating: {e}")
            return {}

    def _calculate_consensus(self, buy_count: int, hold_count: int, sell_count: int) -> str:
        """Calculate consensus rating."""
        total = buy_count + hold_count + sell_count
        if total == 0:
            return "none"

        buy_pct = buy_count / total
        if buy_pct > 0.6:
            return "strong_buy"
        elif buy_pct > 0.4:
            return "buy"
        elif sell_count / total > 0.4:
            return "sell"
        else:
            return "hold"


class DividendsRepository(BaseRepository):
    """Repository for dividend data."""

    def __init__(self, db_adapter: IAsyncDatabase, config: RepositoryConfig | None = None):
        super().__init__(db_adapter, type("Dividend", (), {"__tablename__": "dividends"}), config)
        self.crud_executor = CrudExecutor(db_adapter, "dividends")
        self.metrics = RepositoryMetricsCollector("DividendsRepository")

    def get_required_fields(self) -> list[str]:
        return ["symbol", "ex_date", "amount"]

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        errors = []
        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")

        if "amount" in record and record["amount"] is not None:
            if not isinstance(record["amount"], (int, float)) or record["amount"] < 0:
                errors.append("Dividend amount must be positive")

        return errors

    async def get_dividend_history(self, symbol: str, years: int = 5) -> pd.DataFrame:
        """Get dividend history."""
        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=years * 365)

            query = """
                SELECT * FROM dividends
                WHERE symbol = $1 AND ex_date >= $2
                ORDER BY ex_date DESC
            """

            results = await self.db_adapter.fetch_all(
                query, self._normalize_symbol(symbol), cutoff_date
            )

            return pd.DataFrame([dict(r) for r in results]) if results else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting dividend history: {e}")
            return pd.DataFrame()

    async def calculate_dividend_yield(self, symbol: str, current_price: float) -> float | None:
        """Calculate trailing 12-month dividend yield."""
        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=365)

            query = """
                SELECT SUM(amount) as annual_dividend
                FROM dividends
                WHERE symbol = $1 AND ex_date >= $2
            """

            result = await self.db_adapter.fetch_one(
                query, self._normalize_symbol(symbol), cutoff_date
            )

            if result and result["annual_dividend"] and current_price > 0:
                return (result["annual_dividend"] / current_price) * 100

            return None

        except Exception as e:
            logger.error(f"Error calculating dividend yield: {e}")
            raise


class SocialSentimentRepository(BaseRepository):
    """Repository for social media sentiment data."""

    def __init__(self, db_adapter: IAsyncDatabase, config: RepositoryConfig | None = None):
        super().__init__(
            db_adapter, type("SocialSentiment", (), {"__tablename__": "social_sentiment"}), config
        )
        self.crud_executor = CrudExecutor(db_adapter, "social_sentiment")
        self.metrics = RepositoryMetricsCollector("SocialSentimentRepository")

    def get_required_fields(self) -> list[str]:
        return ["symbol", "timestamp", "platform", "mentions"]

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        errors = []
        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")

        if "platform" in record:
            valid_platforms = ["twitter", "reddit", "stocktwits", "facebook", "instagram"]
            if record["platform"].lower() not in valid_platforms:
                errors.append(f"Invalid platform: {record['platform']}")

        return errors

    async def get_social_metrics(
        self, symbol: str, platform: str | None = None, days: int = 7
    ) -> dict[str, Any]:
        """Get social media metrics."""
        try:
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=days)

            conditions = ["symbol = $1", "timestamp >= $2", "timestamp <= $3"]
            params = [self._normalize_symbol(symbol), start_date, end_date]

            if platform:
                conditions.append(f"platform = ${len(params) + 1}")
                params.append(platform)

            query = f"""
                SELECT
                    platform,
                    SUM(mentions) as total_mentions,
                    AVG(sentiment_score) as avg_sentiment,
                    MAX(mentions) as peak_mentions
                FROM social_sentiment
                WHERE {' AND '.join(conditions)}
                GROUP BY platform
            """

            results = await self.db_adapter.fetch_all(query, *params)

            metrics = {"symbol": symbol, "period_days": days, "platforms": {}}

            for row in results:
                metrics["platforms"][row["platform"]] = {
                    "total_mentions": row["total_mentions"],
                    "avg_sentiment": float(row["avg_sentiment"]) if row["avg_sentiment"] else 0,
                    "peak_mentions": row["peak_mentions"],
                }

            return metrics

        except Exception as e:
            logger.error(f"Error getting social metrics: {e}")
            return {}


class GuidanceRepository(BaseRepository):
    """Repository for earnings guidance data."""

    def __init__(self, db_adapter: IAsyncDatabase, config: RepositoryConfig | None = None):
        super().__init__(
            db_adapter, type("Guidance", (), {"__tablename__": "earnings_guidance"}), config
        )
        self.crud_executor = CrudExecutor(db_adapter, "earnings_guidance")
        self.metrics = RepositoryMetricsCollector("GuidanceRepository")

    def get_required_fields(self) -> list[str]:
        return ["symbol", "issued_date", "period_end", "metric", "guidance_value"]

    def validate_record(self, record: dict[str, Any]) -> list[str]:
        errors = []
        for field in self.get_required_fields():
            if field not in record or record[field] is None:
                errors.append(f"Missing required field: {field}")

        if "metric" in record:
            valid_metrics = ["revenue", "eps", "ebitda", "free_cash_flow"]
            if record["metric"].lower() not in valid_metrics:
                errors.append(f"Invalid metric: {record['metric']}")

        return errors

    async def get_latest_guidance(self, symbol: str, metric: str = "eps") -> dict[str, Any] | None:
        """Get latest guidance for a metric."""
        try:
            query = """
                SELECT * FROM earnings_guidance
                WHERE symbol = $1 AND metric = $2
                ORDER BY issued_date DESC
                LIMIT 1
            """

            result = await self.db_adapter.fetch_one(query, self._normalize_symbol(symbol), metric)

            return dict(result) if result else None

        except Exception as e:
            logger.error(f"Error getting guidance: {e}")
            raise

    async def compare_guidance_to_actual(self, symbol: str, period_end: datetime) -> dict[str, Any]:
        """Compare guidance to actual results."""
        try:
            # Get guidance
            guidance_query = """
                SELECT metric, guidance_value, guidance_low, guidance_high
                FROM earnings_guidance
                WHERE symbol = $1 AND period_end = $2
            """

            guidance_results = await self.db_adapter.fetch_all(
                guidance_query, self._normalize_symbol(symbol), ensure_utc(period_end)
            )

            if not guidance_results:
                return {}

            # Get actuals from financials
            actual_query = """
                SELECT eps_basic, revenue
                FROM financials_data
                WHERE symbol = $1 AND period_end = $2
                LIMIT 1
            """

            actual_result = await self.db_adapter.fetch_one(
                actual_query, self._normalize_symbol(symbol), ensure_utc(period_end)
            )

            if not actual_result:
                return {}

            comparison = {"symbol": symbol, "period_end": period_end, "metrics": {}}

            for guidance in guidance_results:
                metric = guidance["metric"]
                guidance_val = guidance["guidance_value"]

                actual_val = None
                if metric == "eps" and actual_result["eps_basic"]:
                    actual_val = actual_result["eps_basic"]
                elif metric == "revenue" and actual_result["revenue"]:
                    actual_val = actual_result["revenue"]

                if actual_val is not None:
                    comparison["metrics"][metric] = {
                        "guidance": guidance_val,
                        "actual": actual_val,
                        "beat": actual_val > guidance_val,
                        "beat_percentage": (
                            ((actual_val - guidance_val) / guidance_val * 100)
                            if guidance_val
                            else 0
                        ),
                    }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing guidance: {e}")
            return {}
