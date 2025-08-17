# analysis/correlation_matrix.py
"""
Correlation Matrix System for cross-asset signal generation.
Analyzes correlations between different assets to identify trading opportunities.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class CorrelationSignal:
    """Represents a trading signal generated from correlation analysis"""

    timestamp: datetime
    signal_type: str  # 'divergence', 'convergence', 'breakdown', 'regime_shift'
    primary_asset: str
    related_assets: list[str]
    strength: float  # 0-1 signal strength
    direction: str  # 'long', 'short', 'neutral'
    metadata: dict


@dataclass
class CorrelationPair:
    """Represents a correlation relationship between two assets"""

    asset1: str
    asset2: str
    correlation: float
    rolling_mean: float
    rolling_std: float
    z_score: float
    is_significant: bool
    lookback_period: int


class CorrelationMatrix:
    """
    Main correlation matrix system for signal generation.
    Analyzes correlations across multiple asset classes and timeframes.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

        # Correlation parameters
        self.lookback_periods = self.config.get("lookback_periods", [20, 60, 120, 252])
        self.min_correlation = self.config.get("min_correlation", 0.3)
        self.significance_level = self.config.get("significance_level", 0.05)
        self.z_score_threshold = self.config.get("z_score_threshold", 2.0)

        # Asset classification
        self.asset_classes = {
            "equities": ["SPY", "QQQ", "IWM", "DIA"],
            "bonds": ["TLT", "IEF", "SHY", "TIP", "HYG"],
            "commodities": ["GLD", "SLV", "USO", "DBA", "UNG"],
            "currencies": ["DXY", "UUP", "FXE", "FXY"],
            "volatility": ["VXX", "UVXY", "SVXY"],
            "international": ["EFA", "EEM", "FXI", "EWJ"],
            "crypto": ["BTC", "ETH", "BNB"],
            "sectors": ["XLF", "XLT", "XLE", "XLI", "XLK", "XLV", "XLY", "XLP", "XLU", "XLB"],
        }

        # Known important relationships
        self.key_relationships = [
            ("SPY", "TLT"),  # Risk on/off
            ("GLD", "DXY"),  # Dollar vs Gold
            ("VXX", "SPY"),  # Fear gauge
            ("BTC", "QQQ"),  # Crypto-tech correlation
            ("USO", "XLE"),  # Oil and energy sector
            ("TLT", "TIP"),  # Inflation expectations
            ("EEM", "DXY"),  # EM vs Dollar
        ]

        # State tracking
        self._correlation_history = {}
        self._regime_history = []
        self._signals = []

    def calculate_correlation_matrix(
        self, data: dict[str, pd.DataFrame], lookback_period: int = 60
    ) -> pd.DataFrame:
        """Calculate correlation matrix for all assets"""

        # Prepare returns data
        returns = self._prepare_returns(data)

        if returns.empty or len(returns.columns) < 2:
            logger.warning("Insufficient data for correlation matrix")
            return pd.DataFrame()

        # Calculate correlation matrix
        correlation_matrix = returns.tail(lookback_period).corr()

        return correlation_matrix

    def analyze_correlations(self, data: dict[str, pd.DataFrame]) -> list[CorrelationSignal]:
        """Main analysis function that generates trading signals"""

        self._signals = []

        # Prepare data
        returns = self._prepare_returns(data)
        if returns.empty:
            return []

        # Run different analyses
        self._analyze_correlation_breakdowns(returns)
        self._analyze_divergences(returns)
        self._analyze_regime_shifts(returns)
        self._analyze_sector_rotations(returns)
        self._analyze_cross_asset_momentum(returns)
        self._detect_correlation_clusters(returns)

        return self._signals

    def _prepare_returns(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Convert price data to returns"""
        returns = pd.DataFrame()

        for symbol, df in data.items():
            if isinstance(df, pd.DataFrame) and "close" in df.columns:
                returns[symbol] = df["close"].pct_change()

        # Remove assets with insufficient data
        min_periods = max(self.lookback_periods)
        returns = returns.dropna(axis=1, thresh=min_periods)

        return returns

    def _analyze_correlation_breakdowns(self, returns: pd.DataFrame):
        """Detect when historically correlated assets diverge"""

        for asset1, asset2 in self.key_relationships:
            if asset1 not in returns.columns or asset2 not in returns.columns:
                continue

            for period in self.lookback_periods:
                # Calculate rolling correlation
                rolling_corr = returns[asset1].rolling(period).corr(returns[asset2])

                # Calculate historical statistics
                hist_mean = rolling_corr.rolling(period * 3).mean()
                hist_std = rolling_corr.rolling(period * 3).std()

                # Calculate z-score
                z_score = (rolling_corr - hist_mean) / (hist_std + 1e-6)

                # Detect breakdowns
                latest_z = z_score.iloc[-1] if len(z_score) > 0 else 0
                latest_corr = rolling_corr.iloc[-1] if len(rolling_corr) > 0 else 0

                if abs(latest_z) > self.z_score_threshold:
                    signal = CorrelationSignal(
                        timestamp=returns.index[-1],
                        signal_type="breakdown",
                        primary_asset=asset1,
                        related_assets=[asset2],
                        strength=min(abs(latest_z) / 3, 1.0),
                        direction="neutral",
                        metadata={
                            "correlation": latest_corr,
                            "z_score": latest_z,
                            "historical_mean": hist_mean.iloc[-1] if len(hist_mean) > 0 else 0,
                            "lookback_period": period,
                        },
                    )
                    self._signals.append(signal)

    def _analyze_divergences(self, returns: pd.DataFrame):
        """Detect divergences within asset classes"""

        for asset_class, symbols in self.asset_classes.items():
            available_symbols = [s for s in symbols if s in returns.columns]

            if len(available_symbols) < 2:
                continue

            # Calculate average correlation within asset class
            class_returns = returns[available_symbols]

            for period in [20, 60]:
                # Rolling correlation matrix
                correlations = []

                for i in range(len(available_symbols)):
                    for j in range(i + 1, len(available_symbols)):
                        corr = (
                            class_returns.iloc[:, i].rolling(period).corr(class_returns.iloc[:, j])
                        )
                        correlations.append(corr)

                if correlations:
                    avg_correlation = pd.concat(correlations, axis=1).mean(axis=1)

                    # Detect low correlation periods
                    if len(avg_correlation) > 0 and avg_correlation.iloc[-1] < 0.3:
                        signal = CorrelationSignal(
                            timestamp=returns.index[-1],
                            signal_type="divergence",
                            primary_asset=asset_class,
                            related_assets=available_symbols,
                            strength=1 - avg_correlation.iloc[-1],
                            direction="neutral",
                            metadata={
                                "average_correlation": avg_correlation.iloc[-1],
                                "period": period,
                            },
                        )
                        self._signals.append(signal)

    def _analyze_regime_shifts(self, returns: pd.DataFrame):
        """Detect market regime changes based on correlation patterns"""

        # Focus on major assets
        major_assets = ["SPY", "TLT", "GLD", "DXY", "VXX"]
        available = [a for a in major_assets if a in returns.columns]

        if len(available) < 3:
            return

        # Calculate correlation regime over time
        regime_lookback = 60
        regime_data = []

        for i in range(regime_lookback, len(returns)):
            window = returns[available].iloc[i - regime_lookback : i]
            corr_matrix = window.corr()

            # Extract features from correlation matrix
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            avg_corr = upper_triangle.stack().mean()
            std_corr = upper_triangle.stack().std()

            regime_data.append(
                {
                    "timestamp": returns.index[i],
                    "avg_correlation": avg_corr,
                    "correlation_dispersion": std_corr,
                }
            )

        if len(regime_data) < 20:
            return

        regime_df = pd.DataFrame(regime_data).set_index("timestamp")

        # Detect regime changes using rolling statistics
        regime_mean = regime_df["avg_correlation"].rolling(20).mean()
        regime_std = regime_df["avg_correlation"].rolling(20).std()
        regime_z = (regime_df["avg_correlation"] - regime_mean) / (regime_std + 1e-6)

        if len(regime_z) > 0 and abs(regime_z.iloc[-1]) > 2:
            signal = CorrelationSignal(
                timestamp=returns.index[-1],
                signal_type="regime_shift",
                primary_asset="MARKET",
                related_assets=available,
                strength=min(abs(regime_z.iloc[-1]) / 3, 1.0),
                direction="risk_off" if regime_z.iloc[-1] > 2 else "risk_on",
                metadata={
                    "correlation_level": regime_df["avg_correlation"].iloc[-1],
                    "z_score": regime_z.iloc[-1],
                },
            )
            self._signals.append(signal)

    def _analyze_sector_rotations(self, returns: pd.DataFrame):
        """Detect sector rotation opportunities"""

        sectors = self.asset_classes.get("sectors", [])
        available_sectors = [s for s in sectors if s in returns.columns]

        if len(available_sectors) < 4:
            return

        # Calculate sector momentum
        momentum_period = 20
        sector_momentum = returns[available_sectors].rolling(momentum_period).mean()

        if len(sector_momentum) < momentum_period:
            return

        # Rank sectors by momentum
        latest_momentum = sector_momentum.iloc[-1].sort_values(ascending=False)

        # Check for rotation signals
        if len(latest_momentum) >= 2:
            top_sector = latest_momentum.index[0]
            bottom_sector = latest_momentum.index[-1]

            # Calculate correlation between top and bottom
            corr = returns[top_sector].rolling(60).corr(returns[bottom_sector]).iloc[-1]

            if corr < -0.3:  # Negative correlation suggests rotation
                signal = CorrelationSignal(
                    timestamp=returns.index[-1],
                    signal_type="rotation",
                    primary_asset=top_sector,
                    related_assets=[bottom_sector],
                    strength=abs(corr),
                    direction="long",
                    metadata={
                        "top_momentum": latest_momentum.iloc[0],
                        "bottom_momentum": latest_momentum.iloc[-1],
                        "correlation": corr,
                    },
                )
                self._signals.append(signal)

    def _analyze_cross_asset_momentum(self, returns: pd.DataFrame):
        """Detect momentum spillovers across asset classes"""

        # Define lead-lag relationships
        lead_lag_pairs = [
            ("VXX", "SPY", -1),  # VXX leads SPY inversely
            ("DXY", "GLD", -1),  # Dollar leads gold inversely
            ("TLT", "TIPS", 1),  # Bonds lead inflation expectations
            ("BTC", "QQQ", 1),  # Crypto leads tech
        ]

        for leader, follower, expected_sign in lead_lag_pairs:
            if leader not in returns.columns or follower not in returns.columns:
                continue

            # Calculate lagged correlations
            lag_correlations = []
            for lag in range(1, 11):
                corr = returns[follower].corr(returns[leader].shift(lag))
                lag_correlations.append((lag, corr))

            # Find optimal lag
            optimal_lag = max(lag_correlations, key=lambda x: abs(x[1]))

            if abs(optimal_lag[1]) > 0.3 and np.sign(optimal_lag[1]) == expected_sign:
                # Check recent momentum in leader
                leader_momentum = returns[leader].rolling(5).mean().iloc[-1]

                if abs(leader_momentum) > returns[leader].std():
                    signal = CorrelationSignal(
                        timestamp=returns.index[-1],
                        signal_type="momentum_spillover",
                        primary_asset=follower,
                        related_assets=[leader],
                        strength=abs(optimal_lag[1]),
                        direction="long" if leader_momentum * expected_sign > 0 else "short",
                        metadata={
                            "leader_momentum": leader_momentum,
                            "optimal_lag": optimal_lag[0],
                            "lag_correlation": optimal_lag[1],
                        },
                    )
                    self._signals.append(signal)

    def _detect_correlation_clusters(self, returns: pd.DataFrame):
        """Identify clusters of assets moving together"""

        if len(returns.columns) < 10:
            return

        # Calculate recent correlation matrix
        recent_corr = returns.tail(60).corr()

        # Remove NaN values
        recent_corr = recent_corr.fillna(0)

        # Convert correlation to distance
        distance_matrix = 1 - abs(recent_corr)

        # Perform clustering
        n_clusters = min(5, len(returns.columns) // 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        try:
            clusters = kmeans.fit_predict(distance_matrix)

            # Analyze each cluster
            for cluster_id in range(n_clusters):
                cluster_assets = [
                    returns.columns[i] for i, c in enumerate(clusters) if c == cluster_id
                ]

                if len(cluster_assets) >= 3:
                    # Calculate cluster cohesion
                    cluster_returns = returns[cluster_assets]
                    avg_correlation = (
                        cluster_returns.corr()
                        .values[np.triu_indices_from(cluster_returns.corr().values, k=1)]
                        .mean()
                    )

                    if avg_correlation > 0.6:
                        signal = CorrelationSignal(
                            timestamp=returns.index[-1],
                            signal_type="cluster",
                            primary_asset=f"CLUSTER_{cluster_id}",
                            related_assets=cluster_assets,
                            strength=avg_correlation,
                            direction="neutral",
                            metadata={
                                "cluster_size": len(cluster_assets),
                                "average_correlation": avg_correlation,
                                "assets": cluster_assets,
                            },
                        )
                        self._signals.append(signal)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")

    def get_correlation_pairs(
        self, returns: pd.DataFrame, min_correlation: float = 0.5, lookback_period: int = 60
    ) -> list[CorrelationPair]:
        """Get all significant correlation pairs"""

        pairs = []
        correlation_matrix = returns.tail(lookback_period).corr()

        # Get all unique pairs
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                asset1 = correlation_matrix.columns[i]
                asset2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]

                if abs(correlation) >= min_correlation:
                    # Calculate rolling statistics
                    rolling_corr = returns[asset1].rolling(lookback_period).corr(returns[asset2])
                    rolling_mean = rolling_corr.rolling(lookback_period * 2).mean().iloc[-1]
                    rolling_std = rolling_corr.rolling(lookback_period * 2).std().iloc[-1]

                    z_score = (
                        (correlation - rolling_mean) / (rolling_std + 1e-6)
                        if rolling_std > 0
                        else 0
                    )

                    # Test significance
                    n = lookback_period
                    t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

                    pair = CorrelationPair(
                        asset1=asset1,
                        asset2=asset2,
                        correlation=correlation,
                        rolling_mean=rolling_mean,
                        rolling_std=rolling_std,
                        z_score=z_score,
                        is_significant=p_value < self.significance_level,
                        lookback_period=lookback_period,
                    )
                    pairs.append(pair)

        return sorted(pairs, key=lambda x: abs(x.correlation), reverse=True)

    def export_analysis(self, output_dir: str = "data/analysis"):
        """Export correlation analysis results"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export signals
        signals_data = []
        for signal in self._signals:
            signals_data.append(
                {
                    "timestamp": signal.timestamp.isoformat(),
                    "signal_type": signal.signal_type,
                    "primary_asset": signal.primary_asset,
                    "related_assets": signal.related_assets,
                    "strength": signal.strength,
                    "direction": signal.direction,
                    "metadata": signal.metadata,
                }
            )

        with open(output_path / "correlation_signals.json", "w") as f:
            json.dump(signals_data, f, indent=2)

        logger.info(f"Exported {len(signals_data)} correlation signals to {output_path}")
