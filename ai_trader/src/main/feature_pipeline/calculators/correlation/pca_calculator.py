"""
PCA Correlation Calculator

Specialized calculator for Principal Component Analysis of correlations including:
- Principal component loadings and exposures
- Explained variance ratios and decomposition
- Factor exposure analysis across time
- Idiosyncratic vs systematic risk decomposition
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base_correlation import BaseCorrelationCalculator

logger = logging.getLogger(__name__)


class PCACorrelationCalculator(BaseCorrelationCalculator):
    """Calculator for PCA-based correlation analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PCA correlation calculator."""
        super().__init__(config)
        
        # PCA-specific parameters
        self.pca_config = self.correlation_config.get_window_config('pca')
        self.pca_components = self.pca_config.get('pca_components', 5)
        self.variance_threshold = self.pca_config.get('variance_threshold', 0.95)
        
        # PCA analysis parameters
        self.pca_window = 60  # Days for rolling PCA
        self.min_assets = 5   # Minimum assets for PCA
        
        logger.debug(f"Initialized PCACorrelationCalculator with {self.pca_components} components")
    
    def get_feature_names(self) -> List[str]:
        """Return list of PCA correlation feature names."""
        feature_names = []
        
        # Principal component loadings and exposures
        for i in range(1, self.pca_components + 1):
            feature_names.extend([
                f'pc{i}_loading',
                f'pc{i}_exposure'
            ])
        
        # Explained variance features
        feature_names.extend([
            'explained_variance_ratio',
            'idiosyncratic_variance',
            'systematic_variance_ratio',
            'total_explained_variance'
        ])
        
        # Factor analysis features
        feature_names.extend([
            'first_pc_dominance',
            'factor_concentration',
            'pc_stability_score',
            'eigenvalue_dispersion'
        ])
        
        # Dynamic PCA features
        feature_names.extend([
            'pc1_trend_30d',
            'pc1_volatility_30d',
            'factor_rotation_speed',
            'pca_regime_indicator'
        ])
        
        # Risk decomposition features
        feature_names.extend([
            'common_factor_risk',
            'specific_risk_ratio',
            'diversification_ratio',
            'effective_dimension'
        ])
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate PCA correlation features.
        
        Args:
            data: DataFrame with symbol, timestamp, close columns
            
        Returns:
            DataFrame with PCA correlation features
        """
        try:
            # Validate and preprocess data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)
            
            processed_data = self.preprocess_data(data)
            
            if processed_data.empty:
                logger.warning("No data available after preprocessing")
                return self.create_empty_features(data.index)
            
            # Create features DataFrame
            unique_timestamps = processed_data['timestamp'].unique()
            features = self.create_empty_features(pd.Index(unique_timestamps))
            
            # Calculate principal component analysis
            features = self._calculate_pca_features(processed_data, features)
            
            # Calculate explained variance analysis
            features = self._calculate_variance_analysis(processed_data, features)
            
            # Calculate factor analysis features
            features = self._calculate_factor_analysis(processed_data, features)
            
            # Calculate dynamic PCA features
            features = self._calculate_dynamic_pca(processed_data, features)
            
            # Calculate risk decomposition
            features = self._calculate_risk_decomposition(processed_data, features)
            
            # Align features with original data
            if len(features) != len(data):
                features = self._align_features_with_data(features, data)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating PCA correlation features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_pca_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate principal component loadings and exposures."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty or len(returns_pivot.columns) < self.min_assets:
                logger.warning("Insufficient assets for PCA analysis")
                return features
            
            # Calculate PCA for the full dataset
            returns_clean = returns_pivot.dropna()
            
            if len(returns_clean) < 20:
                logger.warning("Insufficient data for PCA analysis")
                return features
            
            # Standardize returns
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_clean)
            
            # Fit PCA
            pca = PCA(n_components=min(self.pca_components, len(returns_clean.columns)))
            pca_result = pca.fit_transform(returns_scaled)
            
            # Store PC loadings (how much each asset loads on each PC)
            for i in range(pca.n_components_):
                pc_num = i + 1
                
                # Average absolute loading across assets
                avg_loading = np.mean(np.abs(pca.components_[i]))
                features[f'pc{pc_num}_loading'] = avg_loading
                
                # PC exposure (how much the portfolio is exposed to this PC)
                pc_exposure = np.std(pca_result[:, i])
                features[f'pc{pc_num}_exposure'] = pc_exposure
            
            # Rolling PCA for time-varying features
            features = self._calculate_rolling_pca(returns_pivot, features)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating PCA features: {e}")
            return features
    
    def _calculate_rolling_pca(self, returns_pivot: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling PCA features."""
        try:
            if len(returns_pivot) < self.pca_window:
                return features
            
            # Rolling PCA analysis
            pc1_loadings_series = []
            pc1_exposures_series = []
            explained_vars = []
            
            for i in range(self.pca_window, len(returns_pivot)):
                window_data = returns_pivot.iloc[i-self.pca_window:i]
                window_clean = window_data.dropna()
                
                if len(window_clean) < 20 or len(window_clean.columns) < self.min_assets:
                    pc1_loadings_series.append(np.nan)
                    pc1_exposures_series.append(np.nan)
                    explained_vars.append(np.nan)
                    continue
                
                try:
                    # Standardize and fit PCA
                    scaler = StandardScaler()
                    window_scaled = scaler.fit_transform(window_clean)
                    
                    pca = PCA(n_components=min(3, len(window_clean.columns)))
                    pca_result = pca.fit_transform(window_scaled)
                    
                    # Store first PC metrics
                    pc1_loading = np.mean(np.abs(pca.components_[0]))
                    pc1_exposure = np.std(pca_result[:, 0])
                    explained_var = pca.explained_variance_ratio_[0]
                    
                    pc1_loadings_series.append(pc1_loading)
                    pc1_exposures_series.append(pc1_exposure)
                    explained_vars.append(explained_var)
                    
                except Exception as e:
                    pc1_loadings_series.append(np.nan)
                    pc1_exposures_series.append(np.nan)
                    explained_vars.append(np.nan)
            
            # Store rolling results in features
            for i, timestamp in enumerate(returns_pivot.index[self.pca_window:]):
                if timestamp in features.index:
                    if i < len(pc1_loadings_series) and not np.isnan(pc1_loadings_series[i]):
                        features.loc[timestamp, 'pc1_loading'] = pc1_loadings_series[i]
                        features.loc[timestamp, 'pc1_exposure'] = pc1_exposures_series[i]
                        features.loc[timestamp, 'explained_variance_ratio'] = explained_vars[i]
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating rolling PCA: {e}")
            return features
    
    def _calculate_variance_analysis(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate variance decomposition analysis."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty or len(returns_pivot.columns) < self.min_assets:
                return features
            
            returns_clean = returns_pivot.dropna()
            
            if len(returns_clean) < 20:
                return features
            
            # Fit PCA for variance analysis
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_clean)
            
            pca = PCA()
            pca.fit(returns_scaled)
            
            # Explained variance ratios
            explained_variance_ratios = pca.explained_variance_ratio_
            
            # Total explained variance by first N components
            total_explained = np.sum(explained_variance_ratios[:self.pca_components])
            features['total_explained_variance'] = total_explained
            
            # Idiosyncratic variance (not explained by first PC)
            idiosyncratic_variance = 1.0 - explained_variance_ratios[0]
            features['idiosyncratic_variance'] = idiosyncratic_variance
            
            # Systematic variance ratio (explained by first PC)
            systematic_variance = explained_variance_ratios[0]
            features['systematic_variance_ratio'] = systematic_variance
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating variance analysis: {e}")
            return features
    
    def _calculate_factor_analysis(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor analysis features."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty or len(returns_pivot.columns) < self.min_assets:
                return features
            
            returns_clean = returns_pivot.dropna()
            
            if len(returns_clean) < 20:
                return features
            
            # Fit PCA for factor analysis
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_clean)
            
            pca = PCA()
            pca.fit(returns_scaled)
            
            # First PC dominance
            if len(pca.explained_variance_ratio_) > 0:
                first_pc_dominance = pca.explained_variance_ratio_[0]
                features['first_pc_dominance'] = first_pc_dominance
            
            # Factor concentration (Herfindahl index of explained variance)
            if len(pca.explained_variance_ratio_) > 1:
                factor_concentration = np.sum(pca.explained_variance_ratio_[:3]**2)
                features['factor_concentration'] = factor_concentration
            
            # Eigenvalue dispersion
            eigenvalues = pca.explained_variance_
            if len(eigenvalues) > 1:
                eigenvalue_dispersion = np.std(eigenvalues[:5]) / (np.mean(eigenvalues[:5]) + self.numerical_tolerance)
                features['eigenvalue_dispersion'] = eigenvalue_dispersion
            
            # PCA stability score (based on eigenvalue gaps)
            if len(eigenvalues) > 1:
                eigenvalue_gaps = np.diff(eigenvalues[:5])
                stability_score = np.max(eigenvalue_gaps) / (eigenvalues[0] + self.numerical_tolerance)
                features['pc_stability_score'] = stability_score
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating factor analysis: {e}")
            return features
    
    def _calculate_dynamic_pca(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate dynamic PCA features."""
        try:
            # Use PC1 exposure for dynamic analysis
            if 'pc1_exposure' not in features.columns:
                return features
            
            pc1_series = features['pc1_exposure']
            
            # PC1 trend
            def trend_function(values):
                if len(values) < 5:
                    return 0.0
                try:
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    return slope
                except (ValueError, TypeError, np.linalg.LinAlgError):
                    return 0.0
            
            pc1_trend = pc1_series.rolling(window=30, min_periods=5).apply(trend_function)
            features['pc1_trend_30d'] = pc1_trend
            
            # PC1 volatility
            pc1_volatility = pc1_series.rolling(window=30, min_periods=10).std()
            features['pc1_volatility_30d'] = pc1_volatility
            
            # Factor rotation speed (rate of change in PC structure)
            if 'pc1_loading' in features.columns:
                loading_changes = features['pc1_loading'].diff().abs()
                rotation_speed = loading_changes.rolling(window=20, min_periods=5).mean()
                features['factor_rotation_speed'] = rotation_speed
            
            # PCA regime indicator (based on first PC strength)
            if 'first_pc_dominance' in features.columns:
                dominance = features['first_pc_dominance'].iloc[0] if len(features) > 0 else 0.0
                
                if dominance > 0.5:
                    regime = 1  # Strong factor regime
                elif dominance > 0.3:
                    regime = 0  # Moderate factor regime
                else:
                    regime = -1  # Weak factor regime
                
                features['pca_regime_indicator'] = regime
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating dynamic PCA: {e}")
            return features
    
    def _calculate_risk_decomposition(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk decomposition using PCA."""
        try:
            returns_pivot = self.pivot_returns_data(data)
            
            if returns_pivot.empty or len(returns_pivot.columns) < self.min_assets:
                return features
            
            returns_clean = returns_pivot.dropna()
            
            if len(returns_clean) < 20:
                return features
            
            # Calculate portfolio-level risk decomposition
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_clean)
            
            pca = PCA()
            pca_result = pca.fit_transform(returns_scaled)
            
            # Common factor risk (risk from first few PCs)
            if len(pca.explained_variance_ratio_) >= 3:
                common_factor_risk = np.sum(pca.explained_variance_ratio_[:3])
                features['common_factor_risk'] = common_factor_risk
                
                # Specific risk ratio
                specific_risk = 1.0 - common_factor_risk
                features['specific_risk_ratio'] = specific_risk
            
            # Diversification ratio (based on eigenvalue distribution)
            eigenvalues = pca.explained_variance_
            if len(eigenvalues) > 1:
                # Effective number of factors
                eigenvalue_sum = np.sum(eigenvalues)
                eigenvalue_sum_sq = np.sum(eigenvalues**2)
                
                if eigenvalue_sum_sq > 0:
                    effective_dimension = eigenvalue_sum**2 / eigenvalue_sum_sq
                    features['effective_dimension'] = effective_dimension
                    
                    # Diversification ratio
                    max_dimension = len(eigenvalues)
                    diversification_ratio = effective_dimension / max_dimension
                    features['diversification_ratio'] = diversification_ratio
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating risk decomposition: {e}")
            return features
    
    def _align_features_with_data(self, features: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Align features DataFrame with original data structure."""
        try:
            if 'timestamp' in original_data.columns:
                expanded_features = original_data[['timestamp']].merge(
                    features.reset_index().rename(columns={'index': 'timestamp'}),
                    on='timestamp',
                    how='left'
                )
                
                expanded_features = expanded_features.drop('timestamp', axis=1)
                expanded_features.index = original_data.index
                expanded_features = expanded_features.fillna(0.0)
                
                return expanded_features
            else:
                return features.reindex(original_data.index, fill_value=0.0)
                
        except Exception as e:
            logger.warning(f"Error aligning features with data: {e}")
            return features