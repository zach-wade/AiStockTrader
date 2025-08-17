# File: src/main/features/precompute_engine.py
"""
Feature Pre-computation Engine for Hunter-Killer Strategy

Provides pre-computed features with Redis caching for instant access during trading.
"""

# Standard library imports
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import logging
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
from sqlalchemy import text

# Local imports
from main.config.config_manager import get_config
from main.data_pipeline.storage.database_factory import DatabaseFactory
from main.feature_pipeline.feature_store_compat import FeatureStore
from main.interfaces.database import IAsyncDatabase
from main.utils.cache import CacheType, features_key, get_global_cache, scanner_key

logger = logging.getLogger(__name__)


@dataclass
class FeatureComputeJob:
    """Feature computation job definition."""

    symbol: str
    feature_types: list[str]
    priority: str = "normal"  # "high", "normal", "low"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: str = "pending"  # "pending", "running", "completed", "failed"
    error_message: str | None = None


@dataclass
class PrecomputeMetrics:
    """Metrics for feature pre-computation performance."""

    jobs_completed: int = 0
    jobs_failed: int = 0
    total_compute_time_ms: float = 0
    cache_hit_rate: float = 0
    avg_feature_compute_time_ms: float = 0
    symbols_processed: set[str] = field(default_factory=set)


class FeaturePrecomputeEngine:
    """
    High-performance feature pre-computation engine with Redis caching.

    Features:
    - Parallel feature calculation across multiple workers
    - Smart priority queue for time-sensitive computations
    - Redis caching with TTL management
    - Batch processing for efficiency
    - Real-time cache warming for active symbols
    """

    def __init__(self, config=None):
        """Initialize feature pre-computation engine."""
        self.config = config or get_config()
        db_factory = DatabaseFactory()
        self.db_adapter: IAsyncDatabase = db_factory.create_async_database(self.config)
        self.cache = get_global_cache()
        self.feature_store = FeatureStore(self.config)

        # Engine configuration
        features_config = self.config._raw_config.get("features", {})
        self.parallel_workers = features_config.get("parallel_calculators", 8)
        self.batch_size = features_config.get("batch_size", 50)
        self.cache_ttl = features_config.get("cache_ttl_seconds", 60)
        self.precompute_interval = features_config.get("precompute_interval_seconds", 300)

        # Feature types to precompute
        self.precompute_features = [
            "technical_indicators",
            "momentum_features",
            "volatility_features",
            "volume_features",
            "price_action_features",
        ]

        # Job management
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.active_jobs: dict[str, FeatureComputeJob] = {}
        self.completed_jobs: list[FeatureComputeJob] = []

        # Worker management
        self.workers: list[asyncio.Task] = []
        self.is_running = False

        # Performance tracking
        self.metrics = PrecomputeMetrics()

        # Thread pool for CPU-intensive calculations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.parallel_workers)

        logger.info(f"Feature precompute engine initialized with {self.parallel_workers} workers")

    async def start(self):
        """Start the precompute engine."""
        if self.is_running:
            logger.warning("Precompute engine already running")
            return

        self.is_running = True
        logger.info("Starting feature precompute engine")

        # Start worker tasks
        self.workers = [
            asyncio.create_task(self._worker(f"worker-{i}")) for i in range(self.parallel_workers)
        ]

        # Start background tasks
        background_tasks = [
            asyncio.create_task(self._precompute_scheduler()),
            asyncio.create_task(self._cache_maintenance()),
            asyncio.create_task(self._metrics_collector()),
        ]

        self.workers.extend(background_tasks)

        logger.info(f"Started {len(self.workers)} worker tasks")

    async def stop(self):
        """Stop the precompute engine."""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Stopping feature precompute engine")

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        # Clean up thread pool
        self.thread_pool.shutdown(wait=True)

        logger.info("Feature precompute engine stopped")

    async def precompute_features(
        self, symbols: list[str], feature_types: list[str] | None = None, priority: str = "normal"
    ) -> list[str]:
        """
        Queue features for pre-computation.

        Args:
            symbols: List of symbols to precompute
            feature_types: Types of features to compute (default: all)
            priority: Job priority ("high", "normal", "low")

        Returns:
            List of job IDs created
        """
        if not feature_types:
            feature_types = self.precompute_features

        job_ids = []

        for symbol in symbols:
            # Check if already computing
            if symbol in self.active_jobs:
                logger.debug(f"Features already computing for {symbol}")
                continue

            # Create job
            job = FeatureComputeJob(
                symbol=symbol, feature_types=feature_types.copy(), priority=priority
            )

            job_id = f"{symbol}_{int(datetime.now().timestamp())}"
            self.active_jobs[job_id] = job

            # Queue job
            await self.job_queue.put((priority, job_id, job))
            job_ids.append(job_id)

        logger.info(f"Queued {len(job_ids)} feature computation jobs")
        return job_ids

    async def get_cached_features(self, symbol: str, feature_type: str) -> pd.DataFrame | None:
        """
        Get pre-computed features from cache.

        Args:
            symbol: Trading symbol
            feature_type: Type of features to retrieve

        Returns:
            Cached features DataFrame or None
        """
        cache_key = features_key(symbol, feature_type)

        try:
            cached_data = await self.cache.get(CacheType.FEATURES, cache_key)
            if cached_data:
                # Deserialize DataFrame
                return pd.read_json(cached_data, orient="records")

            return None

        except Exception as e:
            logger.error(f"Error retrieving cached features for {symbol}: {e}")
            return None

    async def warm_cache_for_symbols(self, symbols: list[str]) -> dict[str, bool]:
        """
        Warm cache for specific symbols with high priority.

        Args:
            symbols: Symbols to warm cache for

        Returns:
            Dict mapping symbol to success status
        """
        logger.info(f"Warming cache for {len(symbols)} symbols")

        job_ids = await self.precompute_features(symbols, priority="high")

        # Wait for completion with timeout
        results = {}
        timeout = 30.0  # 30 second timeout

        for symbol in symbols:
            start_time = datetime.now()
            success = False

            while (datetime.now() - start_time).total_seconds() < timeout:
                # Check if all feature types are cached
                all_cached = True
                for feature_type in self.precompute_features:
                    cached = await self.get_cached_features(symbol, feature_type)
                    if cached is None or cached.empty:
                        all_cached = False
                        break

                if all_cached:
                    success = True
                    break

                await asyncio.sleep(0.1)  # Check every 100ms

            results[symbol] = success

        logger.info(f"Cache warming completed: {sum(results.values())}/{len(symbols)} successful")
        return results

    async def _worker(self, worker_name: str):
        """Worker task for processing feature computation jobs."""
        logger.debug(f"Worker {worker_name} started")

        while self.is_running:
            try:
                # Get job from queue (with timeout)
                try:
                    priority, job_id, job = await asyncio.wait_for(
                        self.job_queue.get(), timeout=1.0
                    )
                except TimeoutError:
                    continue

                # Process job
                await self._process_job(worker_name, job_id, job)

                # Mark task done
                self.job_queue.task_done()

            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {e}")
                await asyncio.sleep(1)

        logger.debug(f"Worker {worker_name} stopped")

    async def _process_job(self, worker_name: str, job_id: str, job: FeatureComputeJob):
        """Process a single feature computation job."""
        start_time = datetime.now(UTC)
        job.started_at = start_time
        job.status = "running"

        logger.debug(f"Worker {worker_name} processing {job.symbol}")

        try:
            # Get market data for symbol
            market_data = await self._get_market_data(job.symbol)

            if market_data.empty:
                raise ValueError(f"No market data available for {job.symbol}")

            # Compute features for each type
            for feature_type in job.feature_types:
                try:
                    # Check cache first
                    cached = await self.get_cached_features(job.symbol, feature_type)
                    if cached is not None and not cached.empty:
                        logger.debug(f"Features already cached for {job.symbol}:{feature_type}")
                        continue

                    # Compute features
                    features_df = await self._compute_features(
                        market_data, job.symbol, feature_type
                    )

                    if not features_df.empty:
                        # Cache the features
                        await self._cache_features(job.symbol, feature_type, features_df)

                        # Also store in feature store for persistence
                        await self._store_features_async(job.symbol, feature_type, features_df)

                except Exception as e:
                    logger.error(f"Error computing {feature_type} for {job.symbol}: {e}")
                    continue

            # Mark job completed
            job.completed_at = datetime.now(UTC)
            job.status = "completed"

            # Update metrics
            self.metrics.jobs_completed += 1
            self.metrics.symbols_processed.add(job.symbol)
            compute_time = (job.completed_at - start_time).total_seconds() * 1000
            self.metrics.total_compute_time_ms += compute_time

            logger.debug(f"Completed features for {job.symbol} in {compute_time:.1f}ms")

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            self.metrics.jobs_failed += 1
            logger.error(f"Job failed for {job.symbol}: {e}")

        finally:
            # Move job to completed list
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            self.completed_jobs.append(job)

            # Limit completed jobs history
            if len(self.completed_jobs) > 1000:
                self.completed_jobs = self.completed_jobs[-500:]

    async def _get_market_data(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """Get market data for feature computation."""
        cutoff_date = datetime.now(UTC) - timedelta(days=lookback_days)

        query = text(
            """
            SELECT timestamp, open, high, low, close, volume, vwap
            FROM market_data
            WHERE symbol = :symbol
            AND timestamp >= :cutoff_date
            ORDER BY timestamp ASC
        """
        )

        def execute_query(session):
            result = session.execute(query, {"symbol": symbol, "cutoff_date": cutoff_date})

            data = []
            for row in result:
                data.append(
                    {
                        "timestamp": row.timestamp,
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                        "volume": float(row.volume),
                        "vwap": float(row.vwap) if row.vwap else row.close,
                    }
                )

            return pd.DataFrame(data)

        return await self.db_adapter.run_sync(execute_query)

    async def _compute_features(
        self, market_data: pd.DataFrame, symbol: str, feature_type: str
    ) -> pd.DataFrame:
        """Compute specific feature type."""
        # Run computation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        if feature_type == "technical_indicators":
            return await loop.run_in_executor(
                self.thread_pool, self._compute_technical_indicators, market_data
            )
        elif feature_type == "momentum_features":
            return await loop.run_in_executor(
                self.thread_pool, self._compute_momentum_features, market_data
            )
        elif feature_type == "volatility_features":
            return await loop.run_in_executor(
                self.thread_pool, self._compute_volatility_features, market_data
            )
        elif feature_type == "volume_features":
            return await loop.run_in_executor(
                self.thread_pool, self._compute_volume_features, market_data
            )
        elif feature_type == "price_action_features":
            return await loop.run_in_executor(
                self.thread_pool, self._compute_price_action_features, market_data
            )
        else:
            logger.warning(f"Unknown feature type: {feature_type}")
            return pd.DataFrame()

    def _compute_technical_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        if len(market_data) < 20:
            return pd.DataFrame()

        df = market_data.copy()

        # Simple Moving Averages
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_20"] = df["close"].rolling(20).mean()

        # Exponential Moving Averages
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Return only feature columns
        feature_cols = [
            col
            for col in df.columns
            if col not in ["timestamp", "open", "high", "low", "close", "volume", "vwap"]
        ]
        result = df[["timestamp"] + feature_cols].copy()

        return result.dropna()

    def _compute_momentum_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum-based features."""
        if len(market_data) < 10:
            return pd.DataFrame()

        df = market_data.copy()

        # Price momentum
        df["momentum_1d"] = df["close"].pct_change(1)
        df["momentum_3d"] = df["close"].pct_change(3)
        df["momentum_5d"] = df["close"].pct_change(5)
        df["momentum_10d"] = df["close"].pct_change(10)

        # Rate of change
        df["roc_5"] = ((df["close"] - df["close"].shift(5)) / df["close"].shift(5)) * 100
        df["roc_10"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100

        # Momentum oscillator
        df["momentum_osc"] = ((df["close"] - df["close"].shift(10)) / df["close"].shift(10)) * 100

        # Price acceleration
        df["price_accel"] = df["momentum_1d"].diff()

        feature_cols = [
            col for col in df.columns if col.startswith(("momentum_", "roc_", "price_accel"))
        ]
        result = df[["timestamp"] + feature_cols].copy()

        return result.dropna()

    def _compute_volatility_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility-based features."""
        if len(market_data) < 20:
            return pd.DataFrame()

        df = market_data.copy()

        # True Range and ATR
        df["prev_close"] = df["close"].shift(1)
        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = abs(df["high"] - df["prev_close"])
        df["tr3"] = abs(df["low"] - df["prev_close"])
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr"] = df["true_range"].rolling(14).mean()

        # Volatility measures
        df["volatility_5d"] = df["close"].rolling(5).std()
        df["volatility_10d"] = df["close"].rolling(10).std()
        df["volatility_20d"] = df["close"].rolling(20).std()

        # Relative volatility
        df["rel_vol_5d"] = df["volatility_5d"] / df["close"]
        df["rel_vol_10d"] = df["volatility_10d"] / df["close"]

        # Volatility ratio
        df["vol_ratio_5_20"] = df["volatility_5d"] / df["volatility_20d"]
        df["vol_ratio_10_20"] = df["volatility_10d"] / df["volatility_20d"]

        feature_cols = [
            col
            for col in df.columns
            if col.startswith(("true_range", "atr", "volatility_", "rel_vol_", "vol_ratio_"))
        ]
        result = df[["timestamp"] + feature_cols].copy()

        return result.dropna()

    def _compute_volume_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features."""
        if len(market_data) < 20:
            return pd.DataFrame()

        df = market_data.copy()

        # Volume moving averages
        df["vol_sma_5"] = df["volume"].rolling(5).mean()
        df["vol_sma_10"] = df["volume"].rolling(10).mean()
        df["vol_sma_20"] = df["volume"].rolling(20).mean()

        # Relative volume
        df["rvol_5d"] = df["volume"] / df["vol_sma_5"]
        df["rvol_10d"] = df["volume"] / df["vol_sma_10"]
        df["rvol_20d"] = df["volume"] / df["vol_sma_20"]

        # Volume price trend
        df["vpt"] = (
            (df["close"] - df["close"].shift(1)) / df["close"].shift(1) * df["volume"]
        ).cumsum()

        # On Balance Volume
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

        # Volume rate of change
        df["vol_roc_5"] = df["volume"].pct_change(5)
        df["vol_roc_10"] = df["volume"].pct_change(10)

        feature_cols = [
            col for col in df.columns if col.startswith(("vol_", "rvol_", "vpt", "obv"))
        ]
        result = df[["timestamp"] + feature_cols].copy()

        return result.dropna()

    def _compute_price_action_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute price action features."""
        if len(market_data) < 10:
            return pd.DataFrame()

        df = market_data.copy()

        # Candlestick patterns
        df["body_size"] = abs(df["close"] - df["open"]) / df["open"]
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]

        # Price gaps
        df["gap_up"] = (df["low"] > df["high"].shift(1)).astype(int)
        df["gap_down"] = (df["high"] < df["low"].shift(1)).astype(int)
        df["gap_size"] = df["open"] - df["close"].shift(1)

        # Price ranges
        df["daily_range"] = (df["high"] - df["low"]) / df["open"]
        df["range_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

        # Price vs VWAP
        df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]

        # Intraday strength
        df["intraday_strength"] = (df["close"] - df["open"]) / (df["high"] - df["low"])

        feature_cols = [
            col
            for col in df.columns
            if col.startswith(
                ("body_", "upper_", "lower_", "gap_", "daily_", "range_", "price_vs_", "intraday_")
            )
        ]
        result = df[["timestamp"] + feature_cols].copy()

        return result.dropna()

    async def _cache_features(self, symbol: str, feature_type: str, features_df: pd.DataFrame):
        """Cache computed features in Redis."""
        cache_key = features_key(symbol, feature_type)

        try:
            # Serialize DataFrame to JSON
            features_json = features_df.to_json(orient="records", date_format="iso")

            # Cache with TTL
            await self.cache.set(CacheType.FEATURES, cache_key, features_json, self.cache_ttl)

            logger.debug(f"Cached {len(features_df)} features for {symbol}:{feature_type}")

        except Exception as e:
            logger.error(f"Error caching features for {symbol}: {e}")

    async def _store_features_async(
        self, symbol: str, feature_type: str, features_df: pd.DataFrame
    ):
        """Store features in persistent storage."""
        try:
            # Use feature store for persistence
            success = await asyncio.to_thread(
                self.feature_store.save_features, symbol, features_df, feature_type
            )

            if success:
                logger.debug(f"Stored {len(features_df)} features for {symbol}:{feature_type}")
            else:
                logger.warning(f"Failed to store features for {symbol}:{feature_type}")

        except Exception as e:
            logger.error(f"Error storing features for {symbol}: {e}")

    async def _precompute_scheduler(self):
        """Background task to schedule regular pre-computation."""
        while self.is_running:
            try:
                # Get currently active symbols (Layer 2+ qualified)
                active_symbols = await self._get_active_symbols()

                if active_symbols:
                    logger.info(f"Scheduling precompute for {len(active_symbols)} active symbols")
                    await self.precompute_features(active_symbols, priority="normal")

                # Wait for next interval
                await asyncio.sleep(self.precompute_interval)

            except Exception as e:
                logger.error(f"Error in precompute scheduler: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _get_active_symbols(self) -> list[str]:
        """Get list of currently active symbols."""
        # Get from Redis cache first
        scanner_cache_key = scanner_key(2, "qualified")
        active_symbols = await self.cache.get(CacheType.CUSTOM, scanner_cache_key) or []

        if not active_symbols:
            # Fallback to database query
            query = text(
                """
                SELECT symbol
                FROM companies
                WHERE layer >= 2
                ORDER BY premarket_score DESC
                LIMIT 100
            """
            )

            def execute_query(session):
                result = session.execute(query)
                return [row.symbol for row in result]

            active_symbols = await self.db_adapter.run_sync(execute_query)

        return active_symbols

    async def _cache_maintenance(self):
        """Background task for cache maintenance."""
        while self.is_running:
            try:
                # Clean up expired cache entries
                # This is handled by Redis TTL, but we could add custom logic here

                await asyncio.sleep(600)  # Every 10 minutes

            except Exception as e:
                logger.error(f"Error in cache maintenance: {e}")

    async def _metrics_collector(self):
        """Background task to collect and update metrics."""
        while self.is_running:
            try:
                # Calculate cache hit rate
                total_requests = self.metrics.jobs_completed + self.metrics.jobs_failed
                if total_requests > 0:
                    self.metrics.avg_feature_compute_time_ms = (
                        self.metrics.total_compute_time_ms / total_requests
                    )

                # Store metrics in Redis
                metrics_data = {
                    "jobs_completed": self.metrics.jobs_completed,
                    "jobs_failed": self.metrics.jobs_failed,
                    "avg_compute_time_ms": self.metrics.avg_feature_compute_time_ms,
                    "symbols_processed": len(self.metrics.symbols_processed),
                    "active_jobs": len(self.active_jobs),
                    "queue_size": self.job_queue.qsize(),
                }

                await self.cache.set(CacheType.CUSTOM, "precompute:metrics", metrics_data, ttl=300)

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get current engine status."""
        return {
            "is_running": self.is_running,
            "active_jobs": len(self.active_jobs),
            "queue_size": self.job_queue.qsize(),
            "workers": len(self.workers),
            "metrics": {
                "jobs_completed": self.metrics.jobs_completed,
                "jobs_failed": self.metrics.jobs_failed,
                "avg_compute_time_ms": self.metrics.avg_feature_compute_time_ms,
                "symbols_processed": len(self.metrics.symbols_processed),
            },
            "configuration": {
                "parallel_workers": self.parallel_workers,
                "batch_size": self.batch_size,
                "cache_ttl": self.cache_ttl,
                "precompute_interval": self.precompute_interval,
            },
        }
