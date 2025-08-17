"""
Cross-Sectional Calculator

Calculates cross-sectional features across multiple assets including relative
performance metrics, rankings, and statistical arbitrage opportunities.
"""

# Standard library imports
from typing import Any
import warnings

# Third-party imports
import pandas as pd

# Local imports
from main.utils.core import RateLimiter, get_logger

from .base_calculator import BaseFeatureCalculator
from .helpers import create_feature_dataframe, safe_divide

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class CrossSectionalCalculator(BaseFeatureCalculator):
    """
    Calculates cross-sectional features across multiple assets.

    Features include:
    - Relative performance rankings
    - Cross-sectional momentum and reversal
    - Statistical arbitrage opportunities
    - Sector relative strength
    - Cross-sectional volatility
    - Mean reversion indicators
    - Cross-asset correlations
    - Relative value metrics
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize cross-sectional calculator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Cross-sectional parameters
        self.ranking_windows = config.get("ranking_windows", [5, 20, 60])
        self.momentum_windows = config.get("momentum_windows", [5, 10, 20])
        self.reversion_windows = config.get("reversion_windows", [5, 10])

        # Universe parameters
        self.min_assets = config.get("min_assets", 5)
        self.ranking_method = config.get("ranking_method", "percentile")

        # Statistical arbitrage parameters
        self.zscore_threshold = config.get("zscore_threshold", 2.0)
        self.cointegration_window = config.get("cointegration_window", 252)

        # Rate limiter for intensive calculations
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60)

        logger.info("Initialized CrossSectionalCalculator")

    def get_feature_names(self) -> list[str]:
        """Get list of cross-sectional feature names."""
        features = []

        # Ranking features
        for window in self.ranking_windows:
            features.extend(
                [
                    f"rank_return_{window}d",
                    f"rank_percentile_{window}d",
                    f"rank_zscore_{window}d",
                    f"rank_decile_{window}d",
                ]
            )

        # Momentum features
        for window in self.momentum_windows:
            features.extend(
                [
                    f"momentum_rank_{window}d",
                    f"momentum_strength_{window}d",
                    f"momentum_persistence_{window}d",
                ]
            )

        # Mean reversion features
        for window in self.reversion_windows:
            features.extend(
                [
                    f"reversion_score_{window}d",
                    f"oversold_ratio_{window}d",
                    f"overbought_ratio_{window}d",
                ]
            )

        # Relative performance
        features.extend(
            [
                "relative_strength_index",
                "sector_relative_return",
                "market_relative_return",
                "beta_adjusted_return",
                "alpha_estimate",
            ]
        )

        # Cross-sectional volatility
        features.extend(
            ["relative_volatility_rank", "volatility_zscore", "vol_adjusted_return", "sharpe_rank"]
        )

        # Statistical arbitrage
        features.extend(
            [
                "pairs_zscore",
                "cointegration_score",
                "spread_mean_reversion",
                "arbitrage_opportunity",
            ]
        )

        # Cross-sectional dispersion
        features.extend(
            [
                "cross_sectional_dispersion",
                "return_dispersion_rank",
                "correlation_breakdown",
                "idiosyncratic_risk",
            ]
        )

        # Composite scores
        features.extend(
            [
                "cross_sectional_score",
                "relative_value_score",
                "statistical_edge",
                "cross_momentum_signal",
            ]
        )

        return features

    def calculate(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate cross-sectional features from multi-asset data.

        Args:
            data: Dictionary mapping asset names to DataFrames with OHLCV data

        Returns:
            DataFrame with cross-sectional features for each asset
        """
        try:
            if not data or len(data) < self.min_assets:
                logger.warning(f"Insufficient assets for cross-sectional analysis: {len(data)}")
                return pd.DataFrame()

            # Prepare cross-sectional dataset
            cs_data = self._prepare_cross_sectional_data(data)
            if cs_data.empty:
                return pd.DataFrame()

            # Initialize results dictionary
            all_features = {}

            for asset in data:
                if asset not in cs_data.columns:
                    continue

                # Initialize features for this asset
                asset_data = data[asset]
                features = create_feature_dataframe(asset_data.index)

                # Calculate ranking features
                ranking_features = self._calculate_ranking_features(
                    cs_data, asset, asset_data.index
                )
                features = pd.concat([features, ranking_features], axis=1)

                # Calculate momentum features
                momentum_features = self._calculate_momentum_features(
                    cs_data, asset, asset_data.index
                )
                features = pd.concat([features, momentum_features], axis=1)

                # Calculate mean reversion features
                reversion_features = self._calculate_reversion_features(
                    cs_data, asset, asset_data.index
                )
                features = pd.concat([features, reversion_features], axis=1)

                # Calculate relative performance
                relative_features = self._calculate_relative_performance(cs_data, asset, asset_data)
                features = pd.concat([features, relative_features], axis=1)

                # Calculate volatility features
                volatility_features = self._calculate_volatility_features(
                    cs_data, asset, asset_data.index
                )
                features = pd.concat([features, volatility_features], axis=1)

                # Calculate statistical arbitrage
                arbitrage_features = self._calculate_arbitrage_features(
                    cs_data, asset, asset_data.index
                )
                features = pd.concat([features, arbitrage_features], axis=1)

                # Calculate dispersion features
                dispersion_features = self._calculate_dispersion_features(
                    cs_data, asset, asset_data.index
                )
                features = pd.concat([features, dispersion_features], axis=1)

                # Calculate composite scores
                composite_features = self._calculate_composite_scores(features, cs_data, asset)
                features = pd.concat([features, composite_features], axis=1)

                all_features[asset] = features

            # Combine all asset features
            if all_features:
                # Stack all features with asset identifier
                combined_features = pd.DataFrame()
                for asset, features in all_features.items():
                    asset_features = features.copy()
                    asset_features["asset"] = asset
                    combined_features = pd.concat([combined_features, asset_features])

                return combined_features
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error calculating cross-sectional features: {e}")
            return pd.DataFrame()

    def _prepare_cross_sectional_data(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare aligned cross-sectional dataset."""
        returns_data = {}

        # Calculate returns for each asset
        for asset, df in data.items():
            if "close" in df.columns:
                returns = df["close"].pct_change()
                returns_data[asset] = returns

        # Create aligned DataFrame
        cs_data = pd.DataFrame(returns_data)

        # Forward fill missing values
        cs_data = cs_data.fillna(method="ffill").dropna()

        # Ensure sufficient data
        if len(cs_data) < 20:
            logger.warning("Insufficient historical data for cross-sectional analysis")
            return pd.DataFrame()

        return cs_data

    def _calculate_ranking_features(
        self, cs_data: pd.DataFrame, asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate ranking-based features."""
        features = pd.DataFrame(index=asset_index)

        if asset not in cs_data.columns:
            return features.fillna(0)

        for window in self.ranking_windows:
            # Calculate rolling returns
            rolling_returns = cs_data.rolling(window=window).apply(
                lambda x: (1 + x).prod() - 1, raw=True
            )

            # Calculate ranks
            returns_rank = rolling_returns.rank(axis=1, pct=True)

            # Asset-specific rankings
            asset_rank = returns_rank[asset].reindex(asset_index, method="ffill")
            features[f"rank_return_{window}d"] = asset_rank
            features[f"rank_percentile_{window}d"] = asset_rank * 100

            # Z-score ranking
            mean_return = rolling_returns.mean(axis=1)
            std_return = rolling_returns.std(axis=1)
            asset_zscore = safe_divide(rolling_returns[asset] - mean_return, std_return).reindex(
                asset_index, method="ffill"
            )
            features[f"rank_zscore_{window}d"] = asset_zscore

            # Decile ranking
            features[f"rank_decile_{window}d"] = (asset_rank * 10).astype(int).clip(1, 10)

        return features

    def _calculate_momentum_features(
        self, cs_data: pd.DataFrame, asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate cross-sectional momentum features."""
        features = pd.DataFrame(index=asset_index)

        if asset not in cs_data.columns:
            return features.fillna(0)

        for window in self.momentum_windows:
            # Momentum rank
            momentum = cs_data.rolling(window=window).mean()
            momentum_rank = momentum.rank(axis=1, pct=True)

            asset_momentum_rank = momentum_rank[asset].reindex(asset_index, method="ffill")
            features[f"momentum_rank_{window}d"] = asset_momentum_rank

            # Momentum strength (relative to cross-sectional mean)
            cs_mean = momentum.mean(axis=1)
            momentum_strength = safe_divide(momentum[asset] - cs_mean, cs_mean.abs()).reindex(
                asset_index, method="ffill"
            )
            features[f"momentum_strength_{window}d"] = momentum_strength

            # Momentum persistence (consistency of ranking)
            momentum_consistency = (
                momentum_rank[asset]
                .rolling(window=window)
                .std()
                .reindex(asset_index, method="ffill")
            )
            features[f"momentum_persistence_{window}d"] = 1 / (1 + momentum_consistency)

        return features

    def _calculate_reversion_features(
        self, cs_data: pd.DataFrame, asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate mean reversion features."""
        features = pd.DataFrame(index=asset_index)

        if asset not in cs_data.columns:
            return features.fillna(0)

        for window in self.reversion_windows:
            # Calculate z-scores
            rolling_mean = cs_data.rolling(window=window * 4).mean()
            rolling_std = cs_data.rolling(window=window * 4).std()

            zscore = safe_divide(cs_data - rolling_mean, rolling_std)

            # Reversion score (how extreme the current position)
            asset_zscore = zscore[asset].reindex(asset_index, method="ffill")
            features[f"reversion_score_{window}d"] = asset_zscore.abs()

            # Oversold/overbought ratios
            oversold_threshold = -2.0
            overbought_threshold = 2.0

            oversold_count = (zscore < oversold_threshold).sum(axis=1)
            overbought_count = (zscore > overbought_threshold).sum(axis=1)
            total_assets = zscore.count(axis=1)

            oversold_ratio = safe_divide(oversold_count, total_assets).reindex(
                asset_index, method="ffill"
            )
            overbought_ratio = safe_divide(overbought_count, total_assets).reindex(
                asset_index, method="ffill"
            )

            features[f"oversold_ratio_{window}d"] = oversold_ratio
            features[f"overbought_ratio_{window}d"] = overbought_ratio

        return features

    def _calculate_relative_performance(
        self, cs_data: pd.DataFrame, asset: str, asset_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate relative performance metrics."""
        features = pd.DataFrame(index=asset_data.index)

        if asset not in cs_data.columns:
            return features.fillna(0)

        # Market proxy (equal-weighted average)
        market_return = cs_data.mean(axis=1)

        # Relative strength index
        asset_return = cs_data[asset]
        rsi_periods = [14, 30]

        for period in rsi_periods:
            gains = asset_return.where(asset_return > 0, 0)
            losses = -asset_return.where(asset_return < 0, 0)

            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()

            rs = safe_divide(avg_gain, avg_loss)
            rsi = 100 - (100 / (1 + rs))

            features["relative_strength_index"] = rsi.reindex(asset_data.index, method="ffill")

        # Sector relative return (if sector info available)
        # Simplified: use market return
        relative_return = asset_return - market_return
        features["sector_relative_return"] = relative_return.reindex(
            asset_data.index, method="ffill"
        )
        features["market_relative_return"] = features["sector_relative_return"]

        # Beta calculation
        window = 60
        covariance = asset_return.rolling(window=window).cov(market_return)
        market_variance = market_return.rolling(window=window).var()
        beta = safe_divide(covariance, market_variance)

        # Beta-adjusted return
        beta_adjusted = asset_return - beta * market_return
        features["beta_adjusted_return"] = beta_adjusted.reindex(asset_data.index, method="ffill")

        # Alpha estimate
        features["alpha_estimate"] = (
            beta_adjusted.rolling(window=window).mean().reindex(asset_data.index, method="ffill")
        )

        return features

    def _calculate_volatility_features(
        self, cs_data: pd.DataFrame, asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate cross-sectional volatility features."""
        features = pd.DataFrame(index=asset_index)

        if asset not in cs_data.columns:
            return features.fillna(0)

        # Rolling volatilities
        vol_window = 20
        volatilities = cs_data.rolling(window=vol_window).std()

        # Volatility ranking
        vol_rank = volatilities.rank(axis=1, pct=True)
        features["relative_volatility_rank"] = vol_rank[asset].reindex(asset_index, method="ffill")

        # Volatility z-score
        vol_mean = volatilities.mean(axis=1)
        vol_std = volatilities.std(axis=1)
        vol_zscore = safe_divide(volatilities[asset] - vol_mean, vol_std).reindex(
            asset_index, method="ffill"
        )
        features["volatility_zscore"] = vol_zscore

        # Volatility-adjusted returns
        returns = cs_data[asset]
        vol_adjusted = safe_divide(returns, volatilities[asset])
        features["vol_adjusted_return"] = vol_adjusted.reindex(asset_index, method="ffill")

        # Sharpe ratio ranking
        returns_window = 60
        mean_returns = cs_data.rolling(window=returns_window).mean()
        sharpe_ratios = safe_divide(mean_returns, volatilities)
        sharpe_rank = sharpe_ratios.rank(axis=1, pct=True)

        features["sharpe_rank"] = sharpe_rank[asset].reindex(asset_index, method="ffill")

        return features

    def _calculate_arbitrage_features(
        self, cs_data: pd.DataFrame, asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate statistical arbitrage features."""
        features = pd.DataFrame(index=asset_index)

        if asset not in cs_data.columns or len(cs_data.columns) < 2:
            return features.fillna(0)

        # Find most correlated asset for pairs trading
        correlations = cs_data.corr()
        if asset in correlations.index:
            asset_corrs = correlations[asset].drop(asset)
            if not asset_corrs.empty:
                pair_asset = asset_corrs.abs().idxmax()

                # Calculate spread
                spread = cs_data[asset] - cs_data[pair_asset]

                # Z-score of spread
                spread_mean = spread.rolling(window=60).mean()
                spread_std = spread.rolling(window=60).std()
                spread_zscore = safe_divide(spread - spread_mean, spread_std).reindex(
                    asset_index, method="ffill"
                )

                features["pairs_zscore"] = spread_zscore

                # Cointegration score (simplified)
                cointegration_score = asset_corrs[pair_asset]
                features["cointegration_score"] = cointegration_score

                # Mean reversion indicator
                mean_reversion = 1 / (1 + spread_zscore.abs())
                features["spread_mean_reversion"] = mean_reversion

                # Arbitrage opportunity
                arbitrage_signal = (spread_zscore.abs() > self.zscore_threshold).astype(float)
                features["arbitrage_opportunity"] = arbitrage_signal

        return features

    def _calculate_dispersion_features(
        self, cs_data: pd.DataFrame, asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate cross-sectional dispersion features."""
        features = pd.DataFrame(index=asset_index)

        if asset not in cs_data.columns:
            return features.fillna(0)

        # Cross-sectional dispersion
        cs_std = cs_data.std(axis=1)
        features["cross_sectional_dispersion"] = cs_std.reindex(asset_index, method="ffill")

        # Return dispersion rank
        abs_deviations = cs_data.sub(cs_data.mean(axis=1), axis=0).abs()
        dispersion_rank = abs_deviations.rank(axis=1, pct=True)
        features["return_dispersion_rank"] = dispersion_rank[asset].reindex(
            asset_index, method="ffill"
        )

        # Correlation breakdown
        window = 60
        rolling_corr = cs_data.rolling(window=window).corr()

        # Average correlation with other assets
        if len(cs_data.columns) > 1:
            other_assets = [col for col in cs_data.columns if col != asset]
            asset_corrs = []

            for other in other_assets:
                corr_series = cs_data[asset].rolling(window=window).corr(cs_data[other])
                asset_corrs.append(corr_series)

            if asset_corrs:
                avg_correlation = pd.concat(asset_corrs, axis=1).mean(axis=1)
                features["correlation_breakdown"] = (1 - avg_correlation).reindex(
                    asset_index, method="ffill"
                )

        # Idiosyncratic risk
        market_return = cs_data.mean(axis=1)
        asset_return = cs_data[asset]

        # Simple regression to get residuals
        window = 60
        idiosyncratic_var = []

        for i in range(window, len(cs_data)):
            y = asset_return.iloc[i - window : i]
            x = market_return.iloc[i - window : i]

            if len(y) == len(x) and y.std() > 0 and x.std() > 0:
                correlation = y.corr(x)
                beta = correlation * (y.std() / x.std())
                residual = y.iloc[-1] - beta * x.iloc[-1]
                idiosyncratic_var.append(residual**2)
            else:
                idiosyncratic_var.append(0)

        idiosyncratic_series = pd.Series(idiosyncratic_var, index=cs_data.index[window:])
        features["idiosyncratic_risk"] = idiosyncratic_series.reindex(asset_index, method="ffill")

        return features

    def _calculate_composite_scores(
        self, features: pd.DataFrame, cs_data: pd.DataFrame, asset: str
    ) -> pd.DataFrame:
        """Calculate composite cross-sectional scores."""
        composite = pd.DataFrame(index=features.index)

        # Cross-sectional score (combination of ranking and momentum)
        ranking_score = features.get("rank_percentile_20d", 50) / 100
        momentum_score = features.get("momentum_rank_10d", 0.5)

        composite["cross_sectional_score"] = (ranking_score + momentum_score) / 2

        # Relative value score
        vol_rank = features.get("relative_volatility_rank", 0.5)
        sharpe_rank = features.get("sharpe_rank", 0.5)

        # High Sharpe, low vol is good value
        composite["relative_value_score"] = sharpe_rank - vol_rank + 0.5

        # Statistical edge (arbitrage + reversion)
        reversion_score = 1 / (1 + features.get("reversion_score_5d", 0).abs())
        arbitrage_score = features.get("arbitrage_opportunity", 0)

        composite["statistical_edge"] = (reversion_score + arbitrage_score) / 2

        # Cross momentum signal
        momentum_strength = features.get("momentum_strength_10d", 0)
        momentum_persistence = features.get("momentum_persistence_10d", 0)

        composite["cross_momentum_signal"] = momentum_strength * momentum_persistence

        return composite
