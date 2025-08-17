"""
Enhanced Cross-Sectional Calculator

Advanced cross-sectional analysis with multi-factor models, regime detection,
and sophisticated ranking methodologies for institutional-grade analytics.
"""

# Standard library imports
from typing import Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Local imports
from main.utils.core import AsyncCircuitBreaker, get_logger

from .base_calculator import BaseFeatureCalculator
from .cross_sectional import CrossSectionalCalculator
from .helpers import create_feature_dataframe, safe_divide

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class EnhancedCrossSectionalCalculator(BaseFeatureCalculator):
    """
    Enhanced cross-sectional calculator with advanced analytics.

    Features include:
    - Multi-factor ranking models
    - Regime-aware cross-sectional analysis
    - Dynamic factor loadings
    - Cross-sectional risk decomposition
    - Advanced momentum strategies
    - Sector rotation signals
    - Factor tilts and exposures
    - Cross-sectional machine learning features
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize enhanced cross-sectional calculator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Enhanced parameters
        self.factor_windows = config.get("factor_windows", [20, 60, 120])
        self.regime_windows = config.get("regime_windows", [60, 120])
        self.sector_rotation_window = config.get("sector_rotation_window", 20)

        # Factor model parameters
        self.n_factors = config.get("n_factors", 5)
        self.factor_methods = config.get("factor_methods", ["pca", "statistical"])

        # Advanced ranking parameters
        self.ranking_decay = config.get("ranking_decay", 0.94)
        self.momentum_halflife = config.get("momentum_halflife", 10)

        # Regime detection parameters
        self.regime_threshold = config.get("regime_threshold", 1.5)
        self.min_regime_length = config.get("min_regime_length", 10)

        # Circuit breaker for intensive computations
        self.circuit_breaker = AsyncCircuitBreaker(failure_threshold=3, recovery_timeout=30)

        # Initialize base cross-sectional calculator
        self.base_calculator = CrossSectionalCalculator(config)

        logger.info("Initialized EnhancedCrossSectionalCalculator")

    def get_feature_names(self) -> list[str]:
        """Get list of enhanced cross-sectional feature names."""
        features = []

        # Factor-based features
        for window in self.factor_windows:
            for factor_num in range(1, self.n_factors + 1):
                features.extend(
                    [
                        f"factor_{factor_num}_loading_{window}d",
                        f"factor_{factor_num}_score_{window}d",
                        f"factor_{factor_num}_rank_{window}d",
                    ]
                )

        # Regime-aware features
        features.extend(
            [
                "regime_indicator",
                "regime_momentum_score",
                "regime_adjusted_rank",
                "regime_transition_score",
                "volatility_regime",
            ]
        )

        # Advanced momentum
        features.extend(
            [
                "exponential_momentum_score",
                "momentum_acceleration",
                "momentum_regime_adjusted",
                "cross_sectional_alpha",
                "factor_adjusted_momentum",
            ]
        )

        # Risk decomposition
        features.extend(
            [
                "systematic_risk_loading",
                "idiosyncratic_risk_ratio",
                "factor_concentration_risk",
                "diversification_ratio",
                "risk_adjusted_score",
            ]
        )

        # Sector rotation
        features.extend(
            [
                "sector_momentum_rank",
                "sector_rotation_signal",
                "cross_sector_relative_strength",
                "sector_dispersion_score",
            ]
        )

        # Dynamic features
        features.extend(
            [
                "adaptive_ranking_score",
                "time_varying_beta",
                "dynamic_correlation_score",
                "regime_conditional_rank",
            ]
        )

        # Machine learning features
        features.extend(
            [
                "ml_momentum_probability",
                "ml_reversion_probability",
                "anomaly_detection_score",
                "pattern_recognition_signal",
            ]
        )

        # Advanced composites
        features.extend(
            [
                "enhanced_cross_sectional_score",
                "multi_factor_rank",
                "regime_weighted_composite",
                "institutional_flow_proxy",
            ]
        )

        return features

    def calculate(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate enhanced cross-sectional features.

        Args:
            data: Dictionary mapping asset names to DataFrames with OHLCV data

        Returns:
            DataFrame with enhanced cross-sectional features
        """
        try:
            if not data or len(data) < 5:
                logger.warning("Insufficient assets for enhanced cross-sectional analysis")
                return pd.DataFrame()

            # Prepare enhanced dataset
            enhanced_data = self._prepare_enhanced_data(data)
            if enhanced_data.empty:
                return pd.DataFrame()

            # Initialize results
            all_features = {}

            for asset in data:
                if asset not in enhanced_data["returns"].columns:
                    continue

                asset_data = data[asset]
                features = create_feature_dataframe(asset_data.index)

                # Calculate factor-based features
                factor_features = self._calculate_factor_features(
                    enhanced_data, asset, asset_data.index
                )
                features = pd.concat([features, factor_features], axis=1)

                # Calculate regime-aware features
                regime_features = self._calculate_regime_features(
                    enhanced_data, asset, asset_data.index
                )
                features = pd.concat([features, regime_features], axis=1)

                # Calculate advanced momentum
                momentum_features = self._calculate_advanced_momentum(
                    enhanced_data, asset, asset_data.index
                )
                features = pd.concat([features, momentum_features], axis=1)

                # Calculate risk decomposition
                risk_features = self._calculate_risk_decomposition(
                    enhanced_data, asset, asset_data.index
                )
                features = pd.concat([features, risk_features], axis=1)

                # Calculate sector features
                sector_features = self._calculate_sector_features(
                    enhanced_data, asset, asset_data.index
                )
                features = pd.concat([features, sector_features], axis=1)

                # Calculate dynamic features
                dynamic_features = self._calculate_dynamic_features(
                    enhanced_data, asset, asset_data.index
                )
                features = pd.concat([features, dynamic_features], axis=1)

                # Calculate ML features
                ml_features = self._calculate_ml_features(enhanced_data, asset, asset_data.index)
                features = pd.concat([features, ml_features], axis=1)

                # Calculate enhanced composites
                composite_features = self._calculate_enhanced_composites(
                    features, enhanced_data, asset
                )
                features = pd.concat([features, composite_features], axis=1)

                all_features[asset] = features

            # Combine all asset features
            if all_features:
                combined_features = pd.DataFrame()
                for asset, features in all_features.items():
                    asset_features = features.copy()
                    asset_features["asset"] = asset
                    combined_features = pd.concat([combined_features, asset_features])

                return combined_features
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error calculating enhanced cross-sectional features: {e}")
            return pd.DataFrame()

    def _prepare_enhanced_data(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Prepare enhanced dataset with multiple data types."""
        enhanced_data = {}

        # Returns data
        returns_data = {}
        volumes_data = {}
        volatilities_data = {}

        for asset, df in data.items():
            if "close" in df.columns:
                returns = df["close"].pct_change()
                returns_data[asset] = returns

                # Volatility
                volatility = returns.rolling(window=20).std()
                volatilities_data[asset] = volatility

                # Volume (if available)
                if "volume" in df.columns:
                    volumes_data[asset] = df["volume"]

        # Create aligned DataFrames
        enhanced_data["returns"] = pd.DataFrame(returns_data).fillna(method="ffill").dropna()
        enhanced_data["volatilities"] = pd.DataFrame(volatilities_data).fillna(method="ffill")

        if volumes_data:
            enhanced_data["volumes"] = pd.DataFrame(volumes_data).fillna(method="ffill")

        # Calculate cross-sectional features
        if not enhanced_data["returns"].empty:
            # Market factor
            enhanced_data["market_factor"] = enhanced_data["returns"].mean(axis=1)

            # Volatility factor
            enhanced_data["volatility_factor"] = enhanced_data["volatilities"].mean(axis=1)

            # Dispersion measure
            enhanced_data["dispersion"] = enhanced_data["returns"].std(axis=1)

        return enhanced_data

    def _calculate_factor_features(
        self, enhanced_data: dict[str, pd.DataFrame], asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate factor-based features using PCA and statistical factors."""
        features = pd.DataFrame(index=asset_index)

        returns_data = enhanced_data["returns"]
        if asset not in returns_data.columns:
            return features.fillna(0)

        for window in self.factor_windows:
            # PCA-based factors
            pca_features = self._calculate_pca_factors(returns_data, asset, window, asset_index)
            features = pd.concat([features, pca_features], axis=1)

        return features

    def _calculate_pca_factors(
        self, returns_data: pd.DataFrame, asset: str, window: int, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate PCA-based factor loadings and scores."""
        features = pd.DataFrame(index=asset_index)

        # Rolling PCA
        factor_loadings = []
        factor_scores = []

        for i in range(window, len(returns_data)):
            window_data = returns_data.iloc[i - window : i]

            # Remove assets with insufficient data
            clean_data = window_data.dropna(axis=1, thresh=window // 2)

            if len(clean_data.columns) >= self.n_factors and asset in clean_data.columns:
                # Standardize data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(clean_data.fillna(0))

                # PCA
                pca = PCA(n_components=min(self.n_factors, len(clean_data.columns) - 1))
                pca.fit(scaled_data)

                # Asset loadings
                asset_idx = clean_data.columns.get_loc(asset)
                loadings = pca.components_[:, asset_idx]

                # Factor scores (asset's projection onto factors)
                asset_returns = clean_data[asset].fillna(0).values
                scores = pca.transform(scaled_data)[-1]  # Latest observation

                factor_loadings.append(loadings)
                factor_scores.append(scores)
            else:
                factor_loadings.append(np.zeros(self.n_factors))
                factor_scores.append(np.zeros(self.n_factors))

        # Convert to DataFrame
        for factor_num in range(self.n_factors):
            loadings_series = pd.Series(
                [loadings[factor_num] for loadings in factor_loadings],
                index=returns_data.index[window:],
            )
            scores_series = pd.Series(
                [scores[factor_num] for scores in factor_scores], index=returns_data.index[window:]
            )

            features[f"factor_{factor_num+1}_loading_{window}d"] = loadings_series.reindex(
                asset_index, method="ffill"
            )
            features[f"factor_{factor_num+1}_score_{window}d"] = scores_series.reindex(
                asset_index, method="ffill"
            )

            # Factor rank
            factor_ranks = []
            for i in range(window, len(returns_data)):
                window_scores = [scores[factor_num] for scores in factor_scores[i - window : i]]
                if window_scores:
                    rank = stats.percentileofscore(window_scores, scores_series.iloc[i - window])
                    factor_ranks.append(rank / 100)
                else:
                    factor_ranks.append(0.5)

            rank_series = pd.Series(factor_ranks, index=returns_data.index[window:])
            features[f"factor_{factor_num+1}_rank_{window}d"] = rank_series.reindex(
                asset_index, method="ffill"
            )

        return features

    def _calculate_regime_features(
        self, enhanced_data: dict[str, pd.DataFrame], asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate regime-aware features."""
        features = pd.DataFrame(index=asset_index)

        returns_data = enhanced_data["returns"]
        if asset not in returns_data.columns:
            return features.fillna(0)

        # Volatility regime detection
        market_vol = enhanced_data.get("volatility_factor", pd.Series())
        if not market_vol.empty:
            vol_threshold = market_vol.rolling(window=120).quantile(0.7)
            vol_regime = (market_vol > vol_threshold).astype(int)
            features["volatility_regime"] = vol_regime.reindex(asset_index, method="ffill")

        # Market regime based on dispersion
        dispersion = enhanced_data.get("dispersion", pd.Series())
        if not dispersion.empty:
            dispersion_ma = dispersion.rolling(window=60).mean()
            dispersion_std = dispersion.rolling(window=60).std()

            regime_indicator = safe_divide(dispersion - dispersion_ma, dispersion_std)
            features["regime_indicator"] = regime_indicator.reindex(asset_index, method="ffill")

            # Regime momentum score
            regime_momentum = regime_indicator.rolling(window=20).mean()
            features["regime_momentum_score"] = regime_momentum.reindex(asset_index, method="ffill")

        # Regime-adjusted ranking
        asset_returns = returns_data[asset]
        base_rank = returns_data.rolling(window=20).apply(
            lambda x: stats.percentileofscore(x.iloc[-1], x.mean()), raw=False, axis=1
        )[asset]

        # Adjust rank based on regime
        regime_adjustment = features.get("regime_indicator", 0) * 0.1
        regime_adjusted_rank = base_rank + regime_adjustment
        features["regime_adjusted_rank"] = regime_adjusted_rank.reindex(asset_index, method="ffill")

        # Regime transition score
        regime_changes = features.get("regime_indicator", pd.Series()).diff().abs()
        features["regime_transition_score"] = regime_changes.reindex(asset_index, method="ffill")

        return features

    def _calculate_advanced_momentum(
        self, enhanced_data: dict[str, pd.DataFrame], asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate advanced momentum features."""
        features = pd.DataFrame(index=asset_index)

        returns_data = enhanced_data["returns"]
        if asset not in returns_data.columns:
            return features.fillna(0)

        asset_returns = returns_data[asset]

        # Exponential momentum score
        weights = np.exp(-np.log(2) * np.arange(self.momentum_halflife) / self.momentum_halflife)
        exp_momentum = asset_returns.rolling(window=self.momentum_halflife).apply(
            lambda x: np.average(x, weights=weights[: len(x)]), raw=True
        )
        features["exponential_momentum_score"] = exp_momentum.reindex(asset_index, method="ffill")

        # Momentum acceleration
        momentum_5d = asset_returns.rolling(window=5).mean()
        momentum_20d = asset_returns.rolling(window=20).mean()
        momentum_accel = momentum_5d - momentum_20d
        features["momentum_acceleration"] = momentum_accel.reindex(asset_index, method="ffill")

        # Regime-adjusted momentum
        regime_indicator = features.get("regime_indicator", 0)
        regime_adjusted_momentum = exp_momentum * (1 + regime_indicator * 0.2)
        features["momentum_regime_adjusted"] = regime_adjusted_momentum

        # Cross-sectional alpha
        market_return = enhanced_data.get("market_factor", pd.Series())
        if not market_return.empty:
            # Simple alpha calculation
            beta = (
                asset_returns.rolling(window=60).cov(market_return)
                / market_return.rolling(window=60).var()
            )
            alpha = asset_returns - beta * market_return
            features["cross_sectional_alpha"] = alpha.reindex(asset_index, method="ffill")

        # Factor-adjusted momentum
        # Use first factor as market proxy
        factor_loading = features.get("factor_1_loading_60d", 0)
        factor_score = features.get("factor_1_score_60d", 0)
        factor_adjusted = exp_momentum - factor_loading * factor_score
        features["factor_adjusted_momentum"] = factor_adjusted

        return features

    def _calculate_risk_decomposition(
        self, enhanced_data: dict[str, pd.DataFrame], asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate risk decomposition features."""
        features = pd.DataFrame(index=asset_index)

        returns_data = enhanced_data["returns"]
        if asset not in returns_data.columns:
            return features.fillna(0)

        asset_returns = returns_data[asset]
        market_return = enhanced_data.get("market_factor", pd.Series())

        if not market_return.empty:
            # Systematic risk loading
            rolling_beta = (
                asset_returns.rolling(window=60).cov(market_return)
                / market_return.rolling(window=60).var()
            )
            features["systematic_risk_loading"] = rolling_beta.reindex(asset_index, method="ffill")

            # Idiosyncratic risk ratio
            systematic_var = (rolling_beta**2) * market_return.rolling(window=60).var()
            total_var = asset_returns.rolling(window=60).var()
            idiosyncratic_ratio = 1 - safe_divide(systematic_var, total_var)
            features["idiosyncratic_risk_ratio"] = idiosyncratic_ratio.reindex(
                asset_index, method="ffill"
            )

        # Factor concentration risk
        factor_loadings = []
        for i in range(1, self.n_factors + 1):
            loading = features.get(f"factor_{i}_loading_60d", 0)
            factor_loadings.append(loading)

        if factor_loadings:
            factor_concentration = pd.concat(factor_loadings, axis=1).apply(
                lambda x: (x**2).sum() / len(x), axis=1
            )
            features["factor_concentration_risk"] = factor_concentration

        # Diversification ratio (simplified)
        correlation_sum = 0
        asset_vol = asset_returns.rolling(window=60).std()

        for other_asset in returns_data.columns:
            if other_asset != asset:
                correlation = asset_returns.rolling(window=60).corr(returns_data[other_asset])
                correlation_sum += correlation

        avg_correlation = correlation_sum / (len(returns_data.columns) - 1)
        diversification_ratio = 1 - avg_correlation
        features["diversification_ratio"] = diversification_ratio.reindex(
            asset_index, method="ffill"
        )

        # Risk-adjusted score
        risk_adjusted = safe_divide(features.get("exponential_momentum_score", 0), asset_vol)
        features["risk_adjusted_score"] = risk_adjusted

        return features

    def _calculate_sector_features(
        self, enhanced_data: dict[str, pd.DataFrame], asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate sector rotation features."""
        features = pd.DataFrame(index=asset_index)

        returns_data = enhanced_data["returns"]
        if asset not in returns_data.columns:
            return features.fillna(0)

        # Simplified sector analysis (in practice, you'd have sector mappings)
        asset_returns = returns_data[asset]

        # Sector momentum rank (using correlation with similar assets)
        correlations = returns_data.corrwith(asset_returns, axis=0)
        similar_assets = correlations.nlargest(5).index.tolist()

        if len(similar_assets) > 1:
            sector_returns = returns_data[similar_assets].mean(axis=1)
            sector_momentum = sector_returns.rolling(window=self.sector_rotation_window).mean()

            # Rank within sector
            sector_rank = (
                returns_data[similar_assets]
                .rolling(window=self.sector_rotation_window)
                .apply(
                    lambda x: stats.percentileofscore(x.iloc[-1], asset_returns.iloc[x.index[-1]]),
                    raw=False,
                    axis=1,
                )[asset]
            )

            features["sector_momentum_rank"] = (
                sector_rank.reindex(asset_index, method="ffill") / 100
            )

            # Sector rotation signal
            sector_vs_market = sector_momentum - enhanced_data.get("market_factor", 0)
            features["sector_rotation_signal"] = sector_vs_market.reindex(
                asset_index, method="ffill"
            )

            # Cross-sector relative strength
            features["cross_sector_relative_strength"] = safe_divide(
                asset_returns.rolling(window=20).mean(), sector_momentum
            ).reindex(asset_index, method="ffill")

        # Sector dispersion score
        sector_dispersion = returns_data.std(axis=1)
        features["sector_dispersion_score"] = sector_dispersion.reindex(asset_index, method="ffill")

        return features

    def _calculate_dynamic_features(
        self, enhanced_data: dict[str, pd.DataFrame], asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate dynamic adaptive features."""
        features = pd.DataFrame(index=asset_index)

        returns_data = enhanced_data["returns"]
        if asset not in returns_data.columns:
            return features.fillna(0)

        asset_returns = returns_data[asset]

        # Adaptive ranking score with decay
        ranking_scores = []
        decay_weights = [self.ranking_decay**i for i in range(20)]

        for i in range(20, len(returns_data)):
            window_returns = returns_data.iloc[i - 20 : i]
            if asset in window_returns.columns:
                daily_ranks = window_returns.rank(axis=1, pct=True)[asset]
                weighted_rank = np.average(daily_ranks, weights=decay_weights)
                ranking_scores.append(weighted_rank)
            else:
                ranking_scores.append(0.5)

        adaptive_ranking = pd.Series(ranking_scores, index=returns_data.index[20:])
        features["adaptive_ranking_score"] = adaptive_ranking.reindex(asset_index, method="ffill")

        # Time-varying beta
        market_return = enhanced_data.get("market_factor", pd.Series())
        if not market_return.empty:
            time_varying_beta = []
            for i in range(60, len(asset_returns)):
                recent_data = asset_returns.iloc[i - 60 : i]
                recent_market = market_return.iloc[i - 60 : i]

                # Weight recent observations more
                weights = np.exp(-0.1 * np.arange(59, -1, -1))

                # Weighted covariance and variance
                weighted_cov = np.average(
                    (recent_data - recent_data.mean()) * (recent_market - recent_market.mean()),
                    weights=weights,
                )
                weighted_var = np.average(
                    (recent_market - recent_market.mean()) ** 2, weights=weights
                )

                beta = safe_divide(weighted_cov, weighted_var, default_value=1.0)
                time_varying_beta.append(beta)

            tv_beta_series = pd.Series(time_varying_beta, index=asset_returns.index[60:])
            features["time_varying_beta"] = tv_beta_series.reindex(asset_index, method="ffill")

        # Dynamic correlation score
        avg_correlations = []
        for i in range(60, len(returns_data)):
            window_data = returns_data.iloc[i - 60 : i]
            if asset in window_data.columns:
                asset_corrs = window_data.corrwith(window_data[asset], axis=0)
                avg_corr = asset_corrs.drop(asset).mean()
                avg_correlations.append(avg_corr)
            else:
                avg_correlations.append(0)

        dynamic_corr = pd.Series(avg_correlations, index=returns_data.index[60:])
        features["dynamic_correlation_score"] = dynamic_corr.reindex(asset_index, method="ffill")

        # Regime conditional rank
        regime_indicator = features.get("regime_indicator", 0)
        base_rank = features.get("adaptive_ranking_score", 0.5)
        regime_conditional = base_rank * (1 + regime_indicator * 0.3)
        features["regime_conditional_rank"] = regime_conditional

        return features

    def _calculate_ml_features(
        self, enhanced_data: dict[str, pd.DataFrame], asset: str, asset_index: pd.Index
    ) -> pd.DataFrame:
        """Calculate machine learning-based features."""
        features = pd.DataFrame(index=asset_index)

        returns_data = enhanced_data["returns"]
        if asset not in returns_data.columns:
            return features.fillna(0)

        # Simplified ML features (in practice, you'd use actual ML models)
        asset_returns = returns_data[asset]

        # Momentum probability (based on recent performance patterns)
        momentum_indicators = []
        for i in range(20, len(asset_returns)):
            recent_returns = asset_returns.iloc[i - 20 : i]
            positive_ratio = (recent_returns > 0).mean()
            trend_strength = (
                abs(recent_returns.mean() / recent_returns.std()) if recent_returns.std() > 0 else 0
            )
            momentum_prob = min(positive_ratio * trend_strength, 1.0)
            momentum_indicators.append(momentum_prob)

        ml_momentum = pd.Series(momentum_indicators, index=asset_returns.index[20:])
        features["ml_momentum_probability"] = ml_momentum.reindex(asset_index, method="ffill")

        # Mean reversion probability
        reversion_indicators = []
        for i in range(60, len(asset_returns)):
            window_returns = asset_returns.iloc[i - 60 : i]
            current_position = (
                window_returns.iloc[-1] - window_returns.mean()
            ) / window_returns.std()
            reversion_prob = 1 / (1 + np.exp(-abs(current_position)))  # Sigmoid
            reversion_indicators.append(reversion_prob)

        ml_reversion = pd.Series(reversion_indicators, index=asset_returns.index[60:])
        features["ml_reversion_probability"] = ml_reversion.reindex(asset_index, method="ffill")

        # Anomaly detection score (based on statistical deviation)
        anomaly_scores = []
        for i in range(60, len(asset_returns)):
            window_data = returns_data.iloc[i - 60 : i]
            if asset in window_data.columns:
                asset_zscore = (
                    window_data[asset].iloc[-1] - window_data[asset].mean()
                ) / window_data[asset].std()
                anomaly_score = abs(asset_zscore) / 3  # Normalize
                anomaly_scores.append(min(anomaly_score, 1.0))
            else:
                anomaly_scores.append(0)

        anomaly_series = pd.Series(anomaly_scores, index=returns_data.index[60:])
        features["anomaly_detection_score"] = anomaly_series.reindex(asset_index, method="ffill")

        # Pattern recognition signal (simplified)
        pattern_scores = []
        for i in range(40, len(asset_returns)):
            recent_pattern = asset_returns.iloc[i - 20 : i]
            historical_patterns = []

            # Compare with historical 20-day patterns
            for j in range(40, i - 20, 5):
                hist_pattern = asset_returns.iloc[j - 20 : j]
                correlation = recent_pattern.corr(hist_pattern)
                if not np.isnan(correlation):
                    historical_patterns.append(correlation)

            if historical_patterns:
                pattern_score = max(historical_patterns)
                pattern_scores.append(pattern_score)
            else:
                pattern_scores.append(0)

        pattern_series = pd.Series(pattern_scores, index=asset_returns.index[40:])
        features["pattern_recognition_signal"] = pattern_series.reindex(asset_index, method="ffill")

        return features

    def _calculate_enhanced_composites(
        self, features: pd.DataFrame, enhanced_data: dict[str, pd.DataFrame], asset: str
    ) -> pd.DataFrame:
        """Calculate enhanced composite scores."""
        composite = pd.DataFrame(index=features.index)

        # Enhanced cross-sectional score
        components = [
            features.get("adaptive_ranking_score", 0.5) * 0.3,
            features.get("momentum_regime_adjusted", 0) * 0.25,
            features.get("risk_adjusted_score", 0) * 0.2,
            features.get("ml_momentum_probability", 0.5) * 0.25,
        ]
        composite["enhanced_cross_sectional_score"] = sum(components)

        # Multi-factor rank
        factor_ranks = []
        for i in range(1, min(4, self.n_factors + 1)):  # Use top 3 factors
            factor_rank = features.get(f"factor_{i}_rank_60d", 0.5)
            factor_ranks.append(factor_rank)

        if factor_ranks:
            composite["multi_factor_rank"] = pd.concat(factor_ranks, axis=1).mean(axis=1)
        else:
            composite["multi_factor_rank"] = 0.5

        # Regime-weighted composite
        regime_weight = 1 + features.get("regime_indicator", 0) * 0.2
        base_score = features.get("enhanced_cross_sectional_score", 0.5)
        composite["regime_weighted_composite"] = base_score * regime_weight

        # Institutional flow proxy
        volume_factor = 1  # Simplified - would use actual volume data
        momentum_strength = features.get("momentum_acceleration", 0)
        sector_strength = features.get("sector_rotation_signal", 0)

        institutional_proxy = (momentum_strength + sector_strength) * volume_factor
        composite["institutional_flow_proxy"] = institutional_proxy

        return composite
