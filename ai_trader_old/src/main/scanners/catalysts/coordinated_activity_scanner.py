# File: ai_trader/scanners/coordinated_activity_scanner.py
"""
Coordinated Activity Scanner

Analyzes author interaction networks to detect coordinated behavior,
such as pump and dump schemes.

Now uses the repository pattern with hot/cold storage awareness to
efficiently access social sentiment data for network analysis.
"""

# Standard library imports
from collections import defaultdict
from datetime import UTC, datetime, timedelta
import logging
from typing import Any

# Third-party imports
import networkx as nx
import pandas as pd

# Sklearn is an optional dependency for this advanced scanner
try:
    # Third-party imports
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Third-party imports
from omegaconf import DictConfig

# Local imports
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.events.types import AlertType, ScanAlert
from main.interfaces.events import IEventBus
from main.interfaces.scanners import IScannerRepository
from main.scanners.catalyst_scanner_base import CatalystScannerBase
from main.utils.core import timer
from main.utils.scanners import ScannerCacheManager, ScannerMetricsCollector

logger = logging.getLogger(__name__)


class CoordinatedActivityScanner(CatalystScannerBase):
    """
    Analyzes author interaction networks to detect coordinated behavior,
    such as pump and dump schemes.

    Now uses the repository pattern with hot/cold storage awareness to
    efficiently access social sentiment data for network analysis.
    """

    def __init__(
        self,
        config: DictConfig,
        repository: IScannerRepository,
        event_bus: IEventBus | None = None,
        metrics_collector: ScannerMetricsCollector | None = None,
        cache_manager: ScannerCacheManager | None = None,
    ):
        """
        Initializes the CoordinatedActivityScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "CoordinatedActivityScanner",
            config,
            repository,
            event_bus,
            metrics_collector,
            cache_manager,
        )

        self.params = self.config.get("scanners.coordinated_activity", {})
        self.cluster_min_size = self.params.get("cluster_min_size", 3)
        self.dbscan_eps = self.params.get("dbscan_eps", 0.5)
        self.lookback_days = self.params.get("lookback_days", 7)
        self.use_cache = self.params.get("use_cache", True)

        # Track initialization
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize scanner resources."""
        if self._initialized:
            return

        logger.info(f"Initializing {self.name}")

        if not SKLEARN_AVAILABLE:
            logger.warning(
                "Scikit-learn not installed. CoordinatedActivityScanner requires sklearn for clustering."
            )

        self._initialized = True

    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        logger.info(f"Cleaning up {self.name}")
        self._initialized = False

    async def scan(self, symbols: list[str], **kwargs) -> list[ScanAlert]:
        """
        Scan for coordinated activity signals.

        Uses repository pattern for efficient social data access with hot storage
        for recent posts needed for network analysis.

        Args:
            symbols: List of symbols to scan
            **kwargs: Additional parameters

        Returns:
            List of ScanAlert objects
        """
        if not self._initialized:
            await self.initialize()

        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not installed. CoordinatedActivityScanner cannot run.")
            return []

        with timer() as t:
            logger.info(f"👥 Coordinated Activity Scanner: Analyzing {len(symbols)} symbols...")

            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(UTC)

            try:
                # Check cache if enabled
                if self.cache and self.use_cache:
                    cache_key = f"coordinated_activity:{self.lookback_days}"
                    cached_alerts = await self.cache.get_cached_result(
                        self.name, "batch", cache_key
                    )
                    if cached_alerts is not None:
                        logger.info("Using cached results for coordinated activity scan")
                        # Filter cached alerts for requested symbols
                        return [alert for alert in cached_alerts if alert.symbol in symbols]

                # Build query filter for social data
                query_filter = QueryFilter(
                    symbols=None,  # Get all symbols to build network
                    start_date=datetime.now(UTC) - timedelta(days=self.lookback_days),
                    end_date=datetime.now(UTC),
                )

                # Get social sentiment data from repository
                # This will primarily use hot storage for recent data
                social_data = await self.repository.get_social_sentiment(
                    symbols=[],  # Empty list means get all available symbols
                    query_filter=query_filter,
                )

                if not social_data:
                    logger.info("No recent social media posts found to build author network.")
                    return []

                # Convert to DataFrame format for analysis
                posts_df = self._convert_social_data_to_df(social_data)

                if posts_df.empty:
                    return []

                # 2. Build the author interaction graph
                author_graph, author_features = self._build_author_network(posts_df)

                # 3. Find suspicious clusters using DBSCAN
                suspicious_clusters = self._find_suspicious_clusters(author_graph, author_features)

                if not suspicious_clusters:
                    logger.info("No suspicious author clusters detected.")
                    return []

                # 4. Identify which symbols are being promoted by these clusters
                promoted_symbols = self._get_promoted_symbols(suspicious_clusters, posts_df)

                # 5. Generate alerts for promoted symbols in our universe
                alerts = []
                for symbol, data in promoted_symbols.items():
                    if symbol in symbols:
                        # Normalize score to 0-1 range (assuming max score of ~5)
                        normalized_score = min(data["score"] / 5.0, 1.0)

                        alert = self.create_alert(
                            symbol=symbol,
                            alert_type=AlertType.COORDINATED_ACTIVITY,
                            score=normalized_score,
                            metadata={
                                "cluster_size": data["cluster_size"],
                                "cluster_authors": data["authors"][:10],  # Limit to 10 authors
                                "reason": f"Coordinated promotion by author cluster of size {data['cluster_size']}",
                                "catalyst_type": "coordinated_activity",
                                "raw_score": data["score"],
                            },
                        )
                        alerts.append(alert)

                        # Record metric
                        if self.metrics:
                            self.metrics.record_alert_generated(
                                self.name, AlertType.COORDINATED_ACTIVITY, symbol, normalized_score
                            )

                # Cache results if enabled
                if self.cache and self.use_cache and alerts:
                    await self.cache.cache_result(
                        self.name,
                        "batch",
                        cache_key,
                        alerts,
                        ttl_seconds=3600,  # 1 hour TTL for network analysis
                    )

                # Deduplicate alerts
                alerts = self.deduplicate_alerts(alerts)

                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)

                logger.info(
                    f"✅ Coordinated Activity Scanner: Found {len(alerts)} symbols with suspicious activity "
                    f"in {t.elapsed_ms:.2f}ms"
                )

                # Record metrics
                if self.metrics:
                    self.metrics.record_scan_duration(self.name, t.elapsed_ms, len(symbols))

                return alerts

            except Exception as e:
                logger.error(f"❌ Error in Coordinated Activity Scanner: {e}", exc_info=True)

                # Record error metric
                if self.metrics:
                    self.metrics.record_scan_error(self.name, type(e).__name__, str(e))

                return []

    async def run(self, universe: list[str]) -> dict[str, dict[str, Any]]:
        """
        Legacy method for backward compatibility.

        Args:
            universe: The list of symbols to scan.

        Returns:
            A dictionary mapping symbols to a catalyst signal if detected.
        """
        # Use the new scan method
        alerts = await self.scan(universe)

        # Convert to legacy format
        catalyst_signals = {}
        for alert in alerts:
            catalyst_signals[alert.symbol] = {
                "score": alert.metadata.get(
                    "raw_score", alert.score * 5.0
                ),  # Convert from 0-1 to 0-5 scale
                "reason": alert.metadata.get("reason", ""),
                "signal_type": "coordinated_activity",
                "metadata": {"cluster_authors": alert.metadata.get("cluster_authors", [])},
            }

        return catalyst_signals

    def _convert_social_data_to_df(
        self, social_data: dict[str, list[dict[str, Any]]]
    ) -> pd.DataFrame:
        """Convert social data from repository format to DataFrame."""
        rows = []

        for symbol, posts in social_data.items():
            for post in posts:
                rows.append(
                    {
                        "symbol": symbol,
                        "author": post.get("author", "unknown"),
                        "content": post.get("content", ""),
                        "timestamp": post.get("timestamp"),
                        "platform": post.get("platform", "unknown"),
                        "sentiment_score": post.get("sentiment_score", 0.5),
                    }
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def _build_author_network(self, posts_df: pd.DataFrame) -> tuple[nx.Graph, pd.DataFrame]:
        """Builds a graph of authors who post about the same symbols."""
        G = nx.Graph()

        # Vectorized author features calculation
        author_features = (
            posts_df.groupby("author")
            .agg({"symbol": lambda x: set(x), "author": "count"})
            .rename(columns={"author": "post_count"})
        )
        author_features.columns = ["symbols", "post_count"]

        # Add all authors as nodes
        G.add_nodes_from(author_features.index)

        # Create edges between authors who post about the same tickers
        symbol_to_authors = posts_df.groupby("symbol")["author"].unique().to_dict()
        for symbol, authors in symbol_to_authors.items():
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    G.add_edge(authors[i], authors[j])

        features_df = author_features
        features_df["unique_symbols"] = features_df["symbols"].apply(len)
        return G, features_df.drop(columns=["symbols"])

    def _find_suspicious_clusters(self, G: nx.Graph, features_df: pd.DataFrame) -> list[list[str]]:
        """Uses DBSCAN clustering to find dense clusters of authors."""
        if features_df.empty:
            return []

        # Scale features for clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)

        # DBSCAN clustering
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.cluster_min_size).fit(features_scaled)

        clusters = defaultdict(list)
        for i, label in enumerate(db.labels_):
            if label != -1:  # -1 is noise
                clusters[label].append(features_df.index[i])

        return [members for members in clusters.values() if len(members) >= self.cluster_min_size]

    def _get_promoted_symbols(
        self, clusters: list[list[str]], posts_df: pd.DataFrame
    ) -> dict[str, dict]:
        """Identifies which symbols are being heavily promoted by each cluster."""
        promoted = defaultdict(lambda: {"score": 0, "cluster_size": 0, "authors": []})

        author_to_cluster = {
            author: i for i, member_list in enumerate(clusters) for author in member_list
        }

        # Count mentions per symbol within each cluster
        cluster_mentions = posts_df[posts_df["author"].isin(author_to_cluster.keys())].copy()
        cluster_mentions["cluster_id"] = cluster_mentions["author"].map(author_to_cluster)

        symbol_counts = (
            cluster_mentions.groupby(["symbol", "cluster_id"])
            .size()
            .reset_index(name="mention_count")
        )

        # Vectorized scoring calculation
        symbol_counts["cluster_size"] = symbol_counts["cluster_id"].apply(
            lambda x: len(clusters[x])
        )
        symbol_counts["score"] = (symbol_counts["cluster_size"] * 0.5) + (
            symbol_counts["mention_count"] * 0.2
        )

        # Find best score for each symbol
        best_scores = symbol_counts.groupby("symbol")["score"].idxmax()
        best_rows = symbol_counts.loc[best_scores]

        # Update promoted dictionary with vectorized results
        for _, row in best_rows.iterrows():
            symbol = row["symbol"]
            cluster_id = row["cluster_id"]

            if row["score"] > promoted[symbol]["score"]:
                promoted[symbol]["score"] = row["score"]
                promoted[symbol]["cluster_size"] = row["cluster_size"]
                promoted[symbol]["authors"] = clusters[cluster_id]

        return dict(promoted)
