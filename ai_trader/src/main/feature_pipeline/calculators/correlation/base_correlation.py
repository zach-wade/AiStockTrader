"""
Base Correlation Calculator

Base class for correlation-based feature calculators, providing shared utilities
for multi-asset correlation analysis, covariance calculations, and dependency measures.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from abc import abstractmethod
import warnings

from ..statistical.base_statistical import BaseStatisticalCalculator
from ..helpers import (
    calculate_correlation, calculate_covariance, safe_divide,
    normalize_series, create_feature_dataframe, align_time_series
)
from .correlation_config import CorrelationConfig

from main.utils.core import get_logger, ensure_utc

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class BaseCorrelationCalculator(BaseStatisticalCalculator):
    """
    Base class for correlation analysis calculators.
    
    Provides:
    - Multi-asset data handling and alignment
    - Correlation matrix calculations
    - Covariance computations
    - Distance and similarity measures
    - Rolling correlation utilities
    - Correlation stability metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base correlation calculator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Initialize correlation configuration
        corr_config = config.get('correlation', {}) if config else {}
        self.corr_config = CorrelationConfig(**corr_config)
        
        # Multi-asset data storage
        self.asset_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.sector_data: Optional[pd.DataFrame] = None
        
        # Common parameters
        self.correlation_windows = self.corr_config.correlation_windows
        self.correlation_methods = self.corr_config.correlation_methods
        self.min_correlation_periods = self.corr_config.min_correlation_periods
        
        logger.debug(f"Initialized {self.name} with correlation config")
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for correlation calculations."""
        return ['symbol', 'timestamp', 'close']
    
    def set_multi_asset_data(
        self,
        asset_data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame] = None,
        sector_data: Optional[pd.DataFrame] = None
    ):
        """
        Set multi-asset data for correlation analysis.
        
        Args:
            asset_data: Dictionary mapping symbol to price DataFrame
            benchmark_data: Optional benchmark price data
            sector_data: Optional sector price data
        """
        self.asset_data = asset_data
        self.benchmark_data = benchmark_data
        self.sector_data = sector_data
        
        # Validate and align data
        self._validate_multi_asset_data()
    
    def _validate_multi_asset_data(self):
        """Validate and align multi-asset data."""
        if not self.asset_data:
            logger.warning("No asset data provided")
            return
        
        # Check each asset has required columns
        for symbol, data in self.asset_data.items():
            if 'close' not in data.columns:
                logger.warning(f"Asset {symbol} missing 'close' column")
        
        # Align time series if multiple assets
        if len(self.asset_data) > 1:
            # Get all DataFrames
            dfs = list(self.asset_data.values())
            if self.benchmark_data is not None:
                dfs.append(self.benchmark_data)
            if self.sector_data is not None:
                dfs.append(self.sector_data)
            
            # Align indices
            aligned_dfs = align_time_series(*dfs, join='inner')
            
            # Update stored data
            for i, symbol in enumerate(self.asset_data.keys()):
                self.asset_data[symbol] = aligned_dfs[i]
            
            if self.benchmark_data is not None:
                self.benchmark_data = aligned_dfs[len(self.asset_data)]
            
            if self.sector_data is not None:
                self.sector_data = aligned_dfs[-1]
    
    def calculate_correlation_matrix(
        self,
        returns_data: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets.
        
        Args:
            returns_data: DataFrame with returns for multiple assets
            method: Correlation method
            
        Returns:
            Correlation matrix
        """
        if method == 'pearson':
            return returns_data.corr(method='pearson')
        elif method == 'spearman':
            return returns_data.corr(method='spearman')
        elif method == 'kendall':
            return returns_data.corr(method='kendall')
        else:
            logger.warning(f"Unknown correlation method: {method}")
            return returns_data.corr()
    
    def calculate_rolling_correlation_matrix(
        self,
        returns_data: pd.DataFrame,
        window: int,
        method: str = 'pearson'
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate rolling correlation matrices.
        
        Args:
            returns_data: DataFrame with returns for multiple assets
            window: Rolling window size
            method: Correlation method
            
        Returns:
            Dictionary of correlation matrices indexed by timestamp
        """
        rolling_correlations = {}
        
        for i in range(window, len(returns_data)):
            window_data = returns_data.iloc[i-window:i]
            timestamp = returns_data.index[i]
            
            corr_matrix = self.calculate_correlation_matrix(window_data, method)
            rolling_correlations[timestamp] = corr_matrix
        
        return rolling_correlations
    
    def calculate_pairwise_correlations(
        self,
        asset1_returns: pd.Series,
        asset2_returns: pd.Series,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate pairwise correlations for multiple windows.
        
        Args:
            asset1_returns: Returns for first asset
            asset2_returns: Returns for second asset
            windows: List of window sizes
            
        Returns:
            DataFrame with correlation features
        """
        if windows is None:
            windows = self.correlation_windows
        
        features = pd.DataFrame(index=asset1_returns.index)
        
        for window in windows:
            for method in self.correlation_methods:
                feature_name = f'correlation_{method}_{window}'
                
                # Calculate rolling correlation
                features[feature_name] = asset1_returns.rolling(
                    window=window,
                    min_periods=self.min_correlation_periods
                ).corr(asset2_returns)
        
        return features
    
    def calculate_correlation_stability(
        self,
        correlation_series: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate correlation stability metrics.
        
        Args:
            correlation_series: Time series of correlations
            window: Window for stability calculations
            
        Returns:
            DataFrame with stability metrics
        """
        features = pd.DataFrame(index=correlation_series.index)
        
        # Rolling statistics of correlation
        features[f'corr_mean_{window}'] = correlation_series.rolling(
            window=window, min_periods=window//2
        ).mean()
        
        features[f'corr_std_{window}'] = correlation_series.rolling(
            window=window, min_periods=window//2
        ).std()
        
        features[f'corr_cv_{window}'] = safe_divide(
            features[f'corr_std_{window}'],
            features[f'corr_mean_{window}'].abs(),
            default_value=0
        )
        
        # Correlation range
        features[f'corr_range_{window}'] = correlation_series.rolling(
            window=window, min_periods=window//2
        ).apply(lambda x: x.max() - x.min(), raw=True)
        
        # Correlation trend (slope)
        features[f'corr_trend_{window}'] = correlation_series.rolling(
            window=window, min_periods=window//2
        ).apply(self._calculate_trend, raw=True)
        
        return features
    
    def calculate_distance_metrics(
        self,
        returns_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate distance-based dependency metrics.
        
        Args:
            returns_data: DataFrame with returns for multiple assets
            
        Returns:
            DataFrame with distance metrics
        """
        n_assets = len(returns_data.columns)
        assets = returns_data.columns.tolist()
        
        # Initialize distance matrix
        distances = pd.DataFrame(
            index=assets,
            columns=assets,
            dtype=float
        )
        
        for i in range(n_assets):
            for j in range(i, n_assets):
                asset1 = assets[i]
                asset2 = assets[j]
                
                # Euclidean distance
                euclidean = np.sqrt(
                    ((returns_data[asset1] - returns_data[asset2]) ** 2).mean()
                )
                
                # Correlation distance
                corr = calculate_correlation(
                    returns_data[asset1],
                    returns_data[asset2]
                )
                corr_distance = np.sqrt(2 * (1 - corr))
                
                # Store symmetric values
                distances.loc[asset1, asset2] = euclidean
                distances.loc[asset2, asset1] = euclidean
        
        return distances
    
    def calculate_beta_features(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate beta and related features.
        
        Args:
            asset_returns: Asset returns
            market_returns: Market/benchmark returns
            windows: List of window sizes
            
        Returns:
            DataFrame with beta features
        """
        if windows is None:
            windows = self.correlation_windows
        
        features = pd.DataFrame(index=asset_returns.index)
        
        for window in windows:
            # Calculate rolling beta
            covariance = asset_returns.rolling(
                window=window,
                min_periods=self.min_correlation_periods
            ).cov(market_returns)
            
            market_variance = market_returns.rolling(
                window=window,
                min_periods=self.min_correlation_periods
            ).var()
            
            beta = safe_divide(covariance, market_variance, default_value=1.0)
            features[f'beta_{window}'] = beta
            
            # Beta stability
            features[f'beta_std_{window}'] = beta.rolling(
                window=window, min_periods=window//2
            ).std()
            
            # Correlation with market
            features[f'market_corr_{window}'] = asset_returns.rolling(
                window=window,
                min_periods=self.min_correlation_periods
            ).corr(market_returns)
        
        return features
    
    def calculate_partial_correlations(
        self,
        returns_data: pd.DataFrame,
        control_variables: List[str]
    ) -> pd.DataFrame:
        """
        Calculate partial correlations controlling for other variables.
        
        Args:
            returns_data: DataFrame with returns
            control_variables: Variables to control for
            
        Returns:
            Partial correlation matrix
        """
        try:
            from sklearn.linear_model import LinearRegression
            
            # Get variables
            all_vars = returns_data.columns.tolist()
            n_vars = len(all_vars)
            
            # Initialize partial correlation matrix
            partial_corr = pd.DataFrame(
                index=all_vars,
                columns=all_vars,
                dtype=float
            )
            
            for i in range(n_vars):
                for j in range(i, n_vars):
                    if i == j:
                        partial_corr.iloc[i, j] = 1.0
                        continue
                    
                    var1 = all_vars[i]
                    var2 = all_vars[j]
                    
                    # Remove effects of control variables
                    X_control = returns_data[control_variables]
                    
                    # Regress var1 on controls
                    model1 = LinearRegression()
                    model1.fit(X_control, returns_data[var1])
                    resid1 = returns_data[var1] - model1.predict(X_control)
                    
                    # Regress var2 on controls
                    model2 = LinearRegression()
                    model2.fit(X_control, returns_data[var2])
                    resid2 = returns_data[var2] - model2.predict(X_control)
                    
                    # Calculate correlation of residuals
                    pcorr = calculate_correlation(resid1, resid2)
                    
                    partial_corr.iloc[i, j] = pcorr
                    partial_corr.iloc[j, i] = pcorr
            
            return partial_corr
            
        except ImportError:
            logger.warning("sklearn not available for partial correlations")
            return pd.DataFrame()
    
    def calculate_tail_dependence(
        self,
        asset1_returns: pd.Series,
        asset2_returns: pd.Series,
        tail_threshold: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate tail dependence coefficients.
        
        Args:
            asset1_returns: Returns for first asset
            asset2_returns: Returns for second asset
            tail_threshold: Threshold for tail definition
            
        Returns:
            Dictionary with tail dependence metrics
        """
        # Align series
        aligned = pd.DataFrame({
            'asset1': asset1_returns,
            'asset2': asset2_returns
        }).dropna()
        
        # Get tail thresholds
        lower_q1 = aligned['asset1'].quantile(tail_threshold)
        upper_q1 = aligned['asset1'].quantile(1 - tail_threshold)
        lower_q2 = aligned['asset2'].quantile(tail_threshold)
        upper_q2 = aligned['asset2'].quantile(1 - tail_threshold)
        
        # Lower tail dependence
        lower_tail_mask = (aligned['asset1'] <= lower_q1) & (aligned['asset2'] <= lower_q2)
        lower_tail_prob = lower_tail_mask.sum() / len(aligned)
        lower_tail_coef = safe_divide(lower_tail_prob, tail_threshold, default_value=0)
        
        # Upper tail dependence
        upper_tail_mask = (aligned['asset1'] >= upper_q1) & (aligned['asset2'] >= upper_q2)
        upper_tail_prob = upper_tail_mask.sum() / len(aligned)
        upper_tail_coef = safe_divide(upper_tail_prob, tail_threshold, default_value=0)
        
        return {
            'lower_tail_dependence': lower_tail_coef,
            'upper_tail_dependence': upper_tail_coef,
            'tail_asymmetry': upper_tail_coef - lower_tail_coef
        }
    
    def _calculate_trend(self, x: np.ndarray) -> float:
        """
        Calculate linear trend (slope) of a series.
        
        Args:
            x: Input array
            
        Returns:
            Slope of linear fit
        """
        try:
            if len(x) < 2:
                return 0.0
            
            # Create time index
            t = np.arange(len(x))
            
            # Linear regression
            slope = np.polyfit(t, x, 1)[0]
            
            return slope
            
        except Exception:
            return 0.0
    
    def prepare_multi_asset_returns(
        self,
        lookback: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Prepare returns DataFrame from multi-asset data.
        
        Args:
            lookback: Optional lookback period
            
        Returns:
            DataFrame with returns for all assets
        """
        returns_dict = {}
        
        # Calculate returns for each asset
        for symbol, data in self.asset_data.items():
            if 'close' in data.columns:
                returns = self.calculate_returns(data['close'])
                returns_dict[symbol] = returns
        
        # Add benchmark if available
        if self.benchmark_data is not None and 'close' in self.benchmark_data.columns:
            benchmark_returns = self.calculate_returns(self.benchmark_data['close'])
            returns_dict['benchmark'] = benchmark_returns
        
        # Create combined DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Apply lookback if specified
        if lookback is not None and len(returns_df) > lookback:
            returns_df = returns_df.iloc[-lookback:]
        
        return returns_df
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation features - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get feature names - to be implemented by subclasses."""
        pass