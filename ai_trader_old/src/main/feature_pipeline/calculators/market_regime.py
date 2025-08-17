"""
Market Regime Calculator

Advanced market regime detection and analysis using machine learning techniques,
hidden Markov models, and sophisticated regime transition analytics.
"""

# Standard library imports
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd

try:
    # Third-party imports
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except ImportError:
    hmm = None
    HMM_AVAILABLE = False
# Third-party imports
from sklearn.preprocessing import StandardScaler
import talib

# Local imports
from main.utils.core import get_logger

from .base_calculator import BaseFeatureCalculator

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime types."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITIONING = "transitioning"


class MarketRegimeCalculator(BaseFeatureCalculator):
    """Advanced market regime detection and analytics."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize calculator with configuration."""
        super().__init__(config)

        # Default parameters
        self.n_regimes = self.config.get("n_regimes", 4)
        self.lookback_window = self.config.get("lookback_window", 252)
        self.transition_window = self.config.get("transition_window", 20)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.vol_regimes = self.config.get("vol_regimes", {"low": 0.1, "medium": 0.2, "high": 0.3})

    def get_feature_names(self) -> list[str]:
        """
        Get list of feature names this calculator produces.

        Returns:
            List of feature names
        """
        return [
            # HMM regime features
            "hmm_regime",
            "hmm_confidence",
            "hmm_regime_0_prob",
            "hmm_regime_1_prob",
            "hmm_regime_2_prob",
            "hmm_regime_3_prob",
            "regime_0_avg_return",
            "regime_0_avg_volatility",
            "regime_1_avg_return",
            "regime_1_avg_volatility",
            "regime_2_avg_return",
            "regime_2_avg_volatility",
            "regime_3_avg_return",
            "regime_3_avg_volatility",
            # Volatility regime features
            "volatility_20d_percentile",
            "volatility_60d_percentile",
            "volatility_120d_percentile",
            "vol_regime",
            "vol_regime_change",
            "garch_vol_momentum",
            "vol_acceleration",
            "high_vol_persistence",
            "vol_term_structure",
            "vol_inverted",
            # Trend regime features
            "market_above_sma20",
            "market_above_sma50",
            "market_above_sma200",
            "market_adx",
            "strong_trend",
            "trend_regime",
            "trend_consistency",
            # Correlation regime features
            "market_correlation",
            "correlation_dispersion",
            "correlation_regime",
            "correlation_breakdown",
            # Liquidity regime features
            "volume_zscore",
            "high_volume_regime",
            "low_volume_regime",
            "spread_zscore",
            "liquidity_stress",
            "volume_concentration",
            "liquidity_regime",
            # Regime stability features
            "regime_stability_score",
            "multi_regime_agreement",
            # Conditional analytics features
            "forward_return_1d",
            "combined_regime",
            "combined_regime_expected_return",
            "combined_regime_risk",
            "regime_downside_risk",
            "regime_var_95",
        ]

    def get_required_columns(self) -> list[str]:
        """
        Get the list of required columns for market regime analytics calculations.

        Returns:
            List of column names required for regime detection and analysis
        """
        return ["symbol", "timestamp", "close", "volume"]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market regime analytics features.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with regime analytics features
        """
        if data.empty:
            return pd.DataFrame()

        # Prepare data
        data = self._prepare_data(data)

        # Initialize results
        features = data[["symbol", "timestamp"]].copy()

        # Calculate returns if not present
        if "returns" not in data.columns:
            data["returns"] = data.groupby("symbol")["close"].pct_change()

        # Calculate regime features
        features = self._calculate_hmm_regimes(features, data)
        features = self._calculate_volatility_regimes(features, data)
        features = self._calculate_trend_regimes(features, data)
        features = self._calculate_correlation_regimes(features, data)
        features = self._calculate_liquidity_regimes(features, data)
        features = self._calculate_regime_transitions(features, data)
        features = self._calculate_regime_stability(features, data)
        features = self._calculate_conditional_analytics(features, data)

        return features

    def _calculate_hmm_regimes(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Hidden Markov Model based regimes."""
        # Prepare market-wide features for HMM
        market_data = (
            data.groupby("timestamp")
            .agg({"returns": "mean", "volume": "sum", "close": "mean"})
            .sort_index()
        )

        # Calculate additional market features
        market_data["volatility"] = market_data["returns"].rolling(20).std()
        market_data["volume_change"] = market_data["volume"].pct_change()
        market_data["trend"] = market_data["close"].pct_change(20)

        # Prepare HMM features
        hmm_features = market_data[["returns", "volatility", "volume_change", "trend"]].dropna()

        if len(hmm_features) < self.lookback_window:
            return features

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(hmm_features)

        try:
            # Check if HMM is available
            if not HMM_AVAILABLE:
                logger.warning("hmmlearn not available, skipping HMM regime detection")
                return pd.DataFrame()

            # Fit Gaussian HMM
            model = hmm.GaussianHMM(
                n_components=self.n_regimes, covariance_type="full", n_iter=100, random_state=42
            )
            model.fit(scaled_features)

            # Predict regimes
            regime_states = model.predict(scaled_features)
            regime_probs = model.predict_proba(scaled_features)

            # Create regime DataFrame
            regime_df = pd.DataFrame(
                {
                    "timestamp": hmm_features.index,
                    "hmm_regime": regime_states,
                    "hmm_confidence": np.max(regime_probs, axis=1),
                }
            )

            # Add regime probabilities
            for i in range(self.n_regimes):
                regime_df[f"hmm_regime_{i}_prob"] = regime_probs[:, i]

            # Merge with features
            features = features.merge(regime_df, on="timestamp", how="left")

            # Calculate regime characteristics
            for i in range(self.n_regimes):
                regime_mask = regime_states == i
                if np.any(regime_mask):
                    # Average returns in regime
                    regime_returns = hmm_features.loc[regime_mask, "returns"].mean()
                    regime_vol = hmm_features.loc[regime_mask, "volatility"].mean()

                    features[f"regime_{i}_avg_return"] = regime_returns
                    features[f"regime_{i}_avg_volatility"] = regime_vol

        except Exception as e:
            warnings.warn(f"HMM regime detection failed: {e!s}")
            features["hmm_regime"] = 0
            features["hmm_confidence"] = 0.5

        return features

    def _calculate_volatility_regimes(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate volatility-based regimes."""
        # Calculate rolling volatility
        for window in [20, 60, 120]:
            vol_col = f"volatility_{window}d"
            data[vol_col] = (
                data.groupby("symbol")["returns"]
                .rolling(window, min_periods=window // 2)
                .std()
                .reset_index(drop=True)
            )

            # Volatility percentiles
            features[f"{vol_col}_percentile"] = data.groupby("timestamp")[vol_col].rank(pct=True)

        # Market-wide volatility regime
        market_vol = data.groupby("timestamp")["returns"].std()

        # Define volatility regimes based on historical percentiles
        vol_percentiles = market_vol.rolling(252, min_periods=60).rank(pct=True)

        features["vol_regime"] = pd.cut(
            vol_percentiles.reindex(features["timestamp"]),
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["low_vol", "normal_vol", "elevated_vol", "high_vol"],
        )

        # Volatility regime transitions
        features["vol_regime_change"] = (
            features["vol_regime"] != features.groupby("symbol")["vol_regime"].shift(1)
        ).astype(int)

        # GARCH-based volatility regime
        features = self._calculate_garch_regimes(features, data)

        # Volatility term structure
        features["vol_term_structure"] = (data["volatility_20d"] / data["volatility_60d"]).fillna(1)

        features["vol_inverted"] = features["vol_term_structure"] > 1.2

        return features

    def _calculate_garch_regimes(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate GARCH-based volatility regimes."""
        # Simplified GARCH proxy using exponentially weighted volatility
        market_returns = data.groupby("timestamp")["returns"].mean()

        # Calculate EWMA volatility with different decay factors
        ewma_fast = market_returns.ewm(span=10).std()
        ewma_slow = market_returns.ewm(span=30).std()

        # Volatility momentum
        vol_momentum = ewma_fast / ewma_slow

        features["garch_vol_momentum"] = features["timestamp"].map(vol_momentum)

        # Volatility acceleration
        vol_accel = vol_momentum.diff(5)
        features["vol_acceleration"] = features["timestamp"].map(vol_accel)

        # High volatility persistence
        high_vol_threshold = ewma_slow.quantile(0.75)
        high_vol_periods = ewma_slow > high_vol_threshold

        # Count consecutive high vol periods
        high_vol_streaks = high_vol_periods.groupby((~high_vol_periods).cumsum()).cumsum()

        features["high_vol_persistence"] = features["timestamp"].map(high_vol_streaks)

        return features

    def _calculate_trend_regimes(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based market regimes."""
        # Market trend indicators
        market_close = data.groupby("timestamp")["close"].mean()

        # Multiple timeframe trends
        for period in [20, 50, 200]:
            sma = market_close.rolling(period, min_periods=period // 2).mean()
            features[f"market_above_sma{period}"] = (
                features["timestamp"].map(lambda x: market_close.get(x, 0))
                > features["timestamp"].map(lambda x: sma.get(x, 0))
            ).astype(int)

        # Trend strength using ADX
        if all(col in data.columns for col in ["high", "low", "close"]):
            market_high = data.groupby("timestamp")["high"].max()
            market_low = data.groupby("timestamp")["low"].min()

            adx = talib.ADX(
                market_high.values, market_low.values, market_close.values, timeperiod=14
            )
            adx_series = pd.Series(adx, index=market_close.index)

            features["market_adx"] = features["timestamp"].map(adx_series)
            features["strong_trend"] = features["market_adx"] > 25

        # Trend regime classification
        trend_score = (
            features["market_above_sma20"].astype(int) * 0.3
            + features["market_above_sma50"].astype(int) * 0.3
            + features["market_above_sma200"].astype(int) * 0.4
        )

        features["trend_regime"] = pd.cut(
            trend_score, bins=[-0.1, 0.3, 0.7, 1.1], labels=["bear", "neutral", "bull"]
        )

        # Trend consistency
        trend_changes = (
            market_close.pct_change(20).rolling(60).apply(lambda x: np.sum(x > 0) / len(x))
        )
        features["trend_consistency"] = features["timestamp"].map(trend_changes)

        return features

    def _calculate_correlation_regimes(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate correlation-based market regimes."""
        returns_pivot = data.pivot(index="timestamp", columns="symbol", values="returns")

        if returns_pivot.shape[1] < 10:
            return features

        # Rolling average correlation
        rolling_corr_mean = []
        rolling_corr_std = []

        window = 60
        for i in range(window, len(returns_pivot)):
            window_data = returns_pivot.iloc[i - window : i]
            corr_matrix = window_data.corr()

            # Get upper triangle
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            corr_values = upper_triangle.stack()

            rolling_corr_mean.append(corr_values.mean())
            rolling_corr_std.append(corr_values.std())

        # Create correlation regime series
        corr_mean_series = pd.Series(rolling_corr_mean, index=returns_pivot.index[window:])
        corr_std_series = pd.Series(rolling_corr_std, index=returns_pivot.index[window:])

        features["market_correlation"] = features["timestamp"].map(corr_mean_series)
        features["correlation_dispersion"] = features["timestamp"].map(corr_std_series)

        # Correlation regime
        corr_percentiles = corr_mean_series.rolling(252, min_periods=60).rank(pct=True)

        features["correlation_regime"] = pd.cut(
            features["timestamp"].map(corr_percentiles),
            bins=[0, 0.33, 0.67, 1.0],
            labels=["low_corr", "normal_corr", "high_corr"],
        )

        # Correlation breakdown indicator
        features["correlation_breakdown"] = (
            (features["market_correlation"] < corr_mean_series.rolling(60).quantile(0.1))
            & (features["correlation_dispersion"] > corr_std_series.rolling(60).quantile(0.9))
        ).astype(int)

        return features

    def _calculate_liquidity_regimes(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate liquidity-based market regimes."""
        # Market-wide liquidity metrics
        market_volume = data.groupby("timestamp")["volume"].sum()
        market_dollar_volume = (data["close"] * data["volume"]).groupby(data["timestamp"]).sum()

        # Volume regime
        volume_ma = market_volume.rolling(20).mean()
        volume_zscore = (market_volume - volume_ma) / market_volume.rolling(20).std()

        features["volume_zscore"] = features["timestamp"].map(volume_zscore)
        features["high_volume_regime"] = features["volume_zscore"] > 1
        features["low_volume_regime"] = features["volume_zscore"] < -1

        # Liquidity stress indicators
        if "bid" in data.columns and "ask" in data.columns:
            # Average bid-ask spread
            spreads = ((data["ask"] - data["bid"]) / data["mid"]).groupby(data["timestamp"]).mean()
            spread_zscore = (spreads - spreads.rolling(20).mean()) / spreads.rolling(20).std()

            features["spread_zscore"] = features["timestamp"].map(spread_zscore)
            features["liquidity_stress"] = features["spread_zscore"] > 2

        # Volume concentration
        volume_hhi = data.groupby("timestamp").apply(
            lambda x: np.sum((x["volume"] / x["volume"].sum()) ** 2)
        )
        features["volume_concentration"] = features["timestamp"].map(volume_hhi)

        # Liquidity regime classification
        liquidity_score = (
            StandardScaler()
            .fit_transform(features[["volume_zscore", "volume_concentration"]].fillna(0))
            .mean(axis=1)
        )

        features["liquidity_regime"] = pd.cut(
            liquidity_score,
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=["low_liquidity", "normal_liquidity", "high_liquidity"],
        )

        return features

    def _calculate_regime_transitions(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate regime transition probabilities and analytics."""
        # Transition matrices for different regime types
        for regime_type in ["hmm_regime", "vol_regime", "trend_regime"]:
            if regime_type not in features.columns:
                continue

            # Calculate transition probabilities
            regime_series = features.groupby("symbol")[regime_type].first()

            # One-step transition counts
            transitions = {}
            for symbol in features["symbol"].unique():
                symbol_regimes = features[features["symbol"] == symbol][regime_type]

                for i in range(len(symbol_regimes) - 1):
                    from_regime = symbol_regimes.iloc[i]
                    to_regime = symbol_regimes.iloc[i + 1]

                    key = (from_regime, to_regime)
                    transitions[key] = transitions.get(key, 0) + 1

            # Calculate transition probabilities
            if transitions:
                from_counts = {}
                for (from_r, to_r), count in transitions.items():
                    from_counts[from_r] = from_counts.get(from_r, 0) + count

                for (from_r, to_r), count in transitions.items():
                    prob_col = f"{regime_type}_trans_{from_r}_to_{to_r}_prob"
                    prob = count / from_counts[from_r] if from_counts[from_r] > 0 else 0

                    features[prob_col] = features[regime_type].map(
                        lambda x: prob if x == from_r else 0
                    )

        # Time in regime
        for regime_type in ["hmm_regime", "vol_regime", "trend_regime"]:
            if regime_type not in features.columns:
                continue

            # Calculate consecutive days in current regime
            regime_changes = features[regime_type] != features.groupby("symbol")[regime_type].shift(
                1
            )
            regime_groups = regime_changes.groupby(features["symbol"]).cumsum()

            features[f"{regime_type}_duration"] = (
                features.groupby(["symbol", regime_groups]).cumcount() + 1
            )

            # Average duration by regime
            avg_duration = features.groupby(regime_type)[f"{regime_type}_duration"].mean()
            features[f"{regime_type}_avg_duration"] = features[regime_type].map(avg_duration)

        return features

    def _calculate_regime_stability(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate regime stability metrics."""
        # Regime change frequency
        for regime_type in ["hmm_regime", "vol_regime", "trend_regime"]:
            if regime_type not in features.columns:
                continue

            # Rolling regime changes
            regime_changes = (
                features[regime_type] != features.groupby("symbol")[regime_type].shift(1)
            ).astype(int)

            features[f"{regime_type}_change_freq_20d"] = (
                regime_changes.groupby(features["symbol"])
                .rolling(20, min_periods=10)
                .sum()
                .reset_index(drop=True)
            )

            features[f"{regime_type}_change_freq_60d"] = (
                regime_changes.groupby(features["symbol"])
                .rolling(60, min_periods=30)
                .sum()
                .reset_index(drop=True)
            )

        # Regime confidence/stability score
        if "hmm_confidence" in features.columns:
            features["regime_stability_score"] = (
                features["hmm_confidence"] * (1 - features["hmm_regime_change_freq_20d"] / 20)
            ).clip(0, 1)

        # Multi-regime agreement
        regime_cols = [col for col in features.columns if col.endswith("_regime")]
        if len(regime_cols) > 1:
            # Encode regimes to numeric for comparison
            encoded_regimes = []
            for col in regime_cols:
                if features[col].dtype == "category" or features[col].dtype == "object":
                    encoded = pd.Categorical(features[col]).codes
                else:
                    encoded = features[col]
                encoded_regimes.append(encoded)

            # Calculate pairwise agreement
            regime_agreement = 0
            for i in range(len(encoded_regimes)):
                for j in range(i + 1, len(encoded_regimes)):
                    agreement = (encoded_regimes[i] == encoded_regimes[j]).astype(int)
                    regime_agreement += agreement

            features["multi_regime_agreement"] = regime_agreement / (
                len(regime_cols) * (len(regime_cols) - 1) / 2
            )

        return features

    def _calculate_conditional_analytics(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate conditional statistics based on regimes."""
        # Conditional returns by regime
        for regime_type in ["hmm_regime", "vol_regime", "trend_regime"]:
            if regime_type not in features.columns:
                continue

            # Forward returns by regime
            features["forward_return_1d"] = (
                features.groupby("symbol")["close"].shift(-1) / features["close"] - 1
            )

            # Average forward returns by regime
            regime_forward_returns = features.groupby(regime_type)["forward_return_1d"].mean()
            features[f"{regime_type}_expected_return"] = features[regime_type].map(
                regime_forward_returns
            )

            # Conditional volatility by regime
            regime_volatility = data.groupby([data["timestamp"], features[regime_type]])[
                "returns"
            ].std()
            features[f"{regime_type}_expected_volatility"] = features.apply(
                lambda row: regime_volatility.get((row["timestamp"], row[regime_type]), np.nan),
                axis=1,
            )

            # Win rate by regime
            positive_returns = (features["forward_return_1d"] > 0).astype(int)
            regime_win_rate = positive_returns.groupby(features[regime_type]).mean()
            features[f"{regime_type}_win_rate"] = features[regime_type].map(regime_win_rate)

        # Cross-regime analytics
        if "vol_regime" in features.columns and "trend_regime" in features.columns:
            # Combined regime state
            features["combined_regime"] = (
                features["vol_regime"].astype(str) + "_" + features["trend_regime"].astype(str)
            )

            # Statistics for combined regimes
            combined_stats = features.groupby("combined_regime").agg(
                {"forward_return_1d": ["mean", "std", "count"]}
            )

            features["combined_regime_expected_return"] = features["combined_regime"].map(
                combined_stats[("forward_return_1d", "mean")]
            )

            features["combined_regime_risk"] = features["combined_regime"].map(
                combined_stats[("forward_return_1d", "std")]
            )

        # Regime-specific risk metrics
        if "hmm_regime" in features.columns:
            # Calculate downside risk by regime
            negative_returns = features[features["returns"] < 0].copy()

            if len(negative_returns) > 0:
                regime_downside_vol = negative_returns.groupby("hmm_regime")["returns"].std()
                features["regime_downside_risk"] = features["hmm_regime"].map(regime_downside_vol)

                # Tail risk by regime
                regime_var_95 = features.groupby("hmm_regime")["returns"].quantile(0.05)
                features["regime_var_95"] = features["hmm_regime"].map(regime_var_95)

        return features

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate input data."""
        # Ensure required columns
        required_cols = ["symbol", "timestamp", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort by timestamp and symbol
        data = data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        # Add synthetic columns if missing
        if "high" not in data.columns:
            data["high"] = data["close"]
        if "low" not in data.columns:
            data["low"] = data["close"]
        if "open" not in data.columns:
            data["open"] = data["close"]

        # Calculate mid price if bid/ask available
        if "bid" in data.columns and "ask" in data.columns:
            data["mid"] = (data["bid"] + data["ask"]) / 2
        else:
            data["mid"] = data["close"]

        return data

    async def get_current_market_regime(self) -> str:
        """
        Get the current market regime classification.

        Returns:
            Market regime string (e.g., 'bull', 'bear', 'neutral', 'volatile')
        """
        try:
            # Check cache first
            if hasattr(self, "_regime_cache") and hasattr(self, "_cache_timestamp"):
                # Standard library imports
                from datetime import timedelta

                cache_duration = timedelta(hours=1)  # Cache for 1 hour
                if (
                    self._cache_timestamp
                    and datetime.now(UTC) - self._cache_timestamp < cache_duration
                ):
                    return self._regime_cache.get("current_regime", "unknown")

            # Analyze current market conditions
            regime = await self._analyze_current_market_regime()

            # Update cache
            if not hasattr(self, "_regime_cache"):
                self._regime_cache = {}
            self._regime_cache["current_regime"] = regime
            self._cache_timestamp = datetime.now(UTC)

            logger.info(f"📊 Current market regime: {regime}")
            return regime

        except Exception as e:
            logger.error(f"Error determining market regime: {e}", exc_info=True)
            return "unknown"

    async def _analyze_current_market_regime(self) -> str:
        """Analyze current market conditions to determine regime."""
        try:
            # Import here to avoid circular dependencies
            # Local imports
            from main.data_pipeline.storage.database_factory import DatabaseFactory
            from main.data_pipeline.storage.repositories.market_data import MarketDataRepository

            # Get database access if not already available
            if not hasattr(self, "_temp_db_adapter"):
                # Local imports
                from main.config.config_manager import get_config

                config = get_config()
                db_factory = DatabaseFactory()
                self._temp_db_adapter = db_factory.create_async_database(config)
                self._temp_market_repo = MarketDataRepository(self._temp_db_adapter)

            # Benchmark symbols for regime analysis
            benchmark_symbols = ["SPY", "QQQ", "IWM"]
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=60)  # 60-day lookback

            regime_votes = []

            for symbol in benchmark_symbols:
                try:
                    # Get recent market data for benchmark
                    market_data = await self._temp_market_repo.get_by_date_range(
                        symbol=symbol, start_date=start_date, end_date=end_date
                    )

                    if not market_data or len(market_data) < 30:
                        logger.warning(f"Insufficient data for {symbol} regime analysis")
                        continue

                    # Analyze regime for this benchmark
                    regime = self._analyze_single_benchmark_regime(market_data, symbol)
                    regime_votes.append(regime)
                    logger.debug(f"{symbol} regime: {regime}")

                except Exception as e:
                    logger.warning(f"Error analyzing regime for {symbol}: {e}")
                    continue

            # Determine consensus regime
            if not regime_votes:
                logger.warning("No valid regime votes, defaulting to 'unknown'")
                return "unknown"

            final_regime = self._determine_consensus_regime(regime_votes)
            logger.info(f"Market regime consensus: {final_regime} (votes: {regime_votes})")

            return final_regime

        except Exception as e:
            logger.error(f"Error in market regime analysis: {e}")
            return "unknown"

    def _analyze_single_benchmark_regime(self, market_data: list[dict], symbol: str) -> str:
        """Analyze regime for a single benchmark symbol."""
        try:
            # Third-party imports
            import pandas as pd

            # Convert to DataFrame
            df = pd.DataFrame(market_data)
            df = df.sort_values("timestamp")

            # Convert decimal columns to float for pandas operations
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Calculate technical indicators
            df["returns"] = df["close"].pct_change()
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()
            df["volatility"] = df["returns"].rolling(window=20).std() * np.sqrt(252)

            # Current values
            current_price = df["close"].iloc[-1]
            current_vol = df["volatility"].iloc[-1]
            sma_20 = df["sma_20"].iloc[-1]
            sma_50 = df["sma_50"].iloc[-1]

            # Recent performance
            price_change_1m = (
                (current_price - df["close"].iloc[-20]) / df["close"].iloc[-20]
                if len(df) >= 20
                else 0
            )
            price_change_3m = (current_price - df["close"].iloc[0]) / df["close"].iloc[0]

            # Moving average relationships
            ma_bullish = current_price > sma_20 > sma_50
            ma_bearish = current_price < sma_20 < sma_50

            # Volatility analysis
            avg_volatility = df["volatility"].mean()
            is_high_vol = current_vol > avg_volatility * 1.5

            # Regime determination logic
            if is_high_vol and abs(price_change_1m) > 0.1:  # High vol + big moves
                return "volatile"
            elif ma_bullish and price_change_3m > 0.05 and price_change_1m > 0:  # Bullish trend
                return "bull"
            elif ma_bearish and price_change_3m < -0.05 and price_change_1m < 0:  # Bearish trend
                return "bear"
            elif abs(price_change_3m) < 0.05 and abs(price_change_1m) < 0.03:  # Sideways
                return "sideways"
            elif price_change_3m > 0.02:
                return "bull"
            elif price_change_3m < -0.02:
                return "bear"
            else:
                return "sideways"

        except Exception as e:
            logger.error(f"Error analyzing regime for {symbol}: {e}")
            return "unknown"

    def _determine_consensus_regime(self, regime_votes: list[str]) -> str:
        """Determine consensus regime from individual benchmark regimes."""
        if not regime_votes:
            return "unknown"

        # Count votes
        vote_counts = {}
        for regime in regime_votes:
            vote_counts[regime] = vote_counts.get(regime, 0) + 1

        # Remove 'unknown' votes for consensus
        valid_votes = {k: v for k, v in vote_counts.items() if k != "unknown"}

        if not valid_votes:
            return "unknown"

        # Find most common regime
        max_votes = max(valid_votes.values())
        consensus_regimes = [regime for regime, votes in valid_votes.items() if votes == max_votes]

        # Handle ties with preference order
        if len(consensus_regimes) == 1:
            return consensus_regimes[0]
        elif "volatile" in consensus_regimes:
            return "volatile"  # Prefer volatile in mixed high-volatility conditions
        elif "sideways" in consensus_regimes:
            return "sideways"  # Prefer sideways as neutral default
        else:
            return consensus_regimes[0]  # Return first option


# Legacy alias for backward compatibility
MarketRegimeDetector = MarketRegimeCalculator
