"""
Tail Risk Calculator

Specialized calculator for extreme value theory and tail risk analysis including:
- Extreme Value Theory (EVT) modeling
- Hill estimator for tail index
- Peak-over-threshold (POT) analysis
- Block maxima analysis
- Tail dependence measures
- Extreme quantile estimation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
import warnings

from .base_risk import BaseRiskCalculator
from ..helpers import safe_divide

logger = logging.getLogger(__name__)


class TailRiskCalculator(BaseRiskCalculator):
    """Calculator for extreme value theory and tail risk analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize tail risk calculator."""
        super().__init__(config)
        
        # Tail risk configuration
        self.tail_config = self.risk_config.get_tail_risk_config()
        self.threshold = self.tail_config['threshold']
        self.min_observations = self.tail_config['min_observations']
        self.confidence_levels = self.tail_config['confidence_levels']
        self.hill_fraction = self.tail_config['hill_estimator_fraction']
        self.hill_min_obs = self.tail_config['hill_estimator_min_observations']
        
        logger.debug(f"Initialized TailRiskCalculator with {self.threshold:.1%} threshold")
    
    def get_feature_names(self) -> List[str]:
        """Return list of tail risk feature names."""
        feature_names = [
            # Hill estimator results
            'hill_estimator_tail_index',
            'hill_estimator_alpha',
            'hill_estimator_standard_error',
            'hill_estimator_confidence_interval_lower',
            'hill_estimator_confidence_interval_upper',
            'hill_estimator_goodness_of_fit',
            
            # Peak-over-threshold analysis
            'pot_threshold',
            'pot_exceedances_count',
            'pot_exceedances_rate',
            'pot_scale_parameter',
            'pot_shape_parameter',
            'pot_location_parameter',
            'pot_mean_excess_function',
            
            # Block maxima analysis
            'block_maxima_location',
            'block_maxima_scale',
            'block_maxima_shape',
            'block_maxima_return_level_100',
            'block_maxima_return_level_250',
            'block_maxima_return_level_1000',
            
            # Extreme quantiles
            'extreme_quantile_99_9',
            'extreme_quantile_99_95',
            'extreme_quantile_99_99',
            'extreme_quantile_extrapolation_95',
            'extreme_quantile_extrapolation_99',
            
            # Tail dependence measures
            'tail_dependence_coefficient',
            'tail_dependence_upper',
            'tail_dependence_lower',
            'tail_correlation',
            'tail_beta',
            
            # Expected shortfall at extreme levels
            'expected_shortfall_99_5',
            'expected_shortfall_99_9',
            'expected_shortfall_99_95',
            'expected_shortfall_99_99',
            'coherent_risk_measure',
            
            # Tail risk ratios and metrics
            'tail_risk_ratio',
            'tail_expectation',
            'tail_variance',
            'tail_skewness',
            'tail_kurtosis',
            'tail_concentration_ratio',
            
            # EVT model diagnostics
            'evt_model_aic',
            'evt_model_bic',
            'evt_model_log_likelihood',
            'evt_qq_plot_correlation',
            'evt_pp_plot_correlation',
            'evt_goodness_of_fit_pvalue',
            
            # Tail risk forecasting
            'tail_risk_forecast_1d',
            'tail_risk_forecast_5d',
            'tail_risk_forecast_21d',
            'tail_risk_persistence',
            'tail_risk_clustering',
            
            # Advanced tail metrics
            'tail_conditional_expectation',
            'tail_spectral_risk_measure',
            'tail_distortion_risk_measure',
            'tail_coherent_risk_measure',
            'tail_robustness_measure',
        ]
        
        return feature_names
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate tail risk features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with tail risk features
        """
        try:
            # Validate input data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)
            
            # Calculate returns
            returns = self.calculate_returns(data)
            
            if len(returns) < self.min_observations:
                logger.warning("Insufficient data for tail risk calculation")
                return self.create_empty_features(data.index)
            
            # Create features DataFrame
            features = self.create_empty_features(data.index)
            
            # Calculate tail risk metrics
            features = self._calculate_hill_estimator(returns, features)
            features = self._calculate_peak_over_threshold(returns, features)
            features = self._calculate_block_maxima(returns, features)
            features = self._calculate_extreme_quantiles(returns, features)
            features = self._calculate_tail_dependence(returns, features)
            features = self._calculate_extreme_expected_shortfall(returns, features)
            features = self._calculate_tail_risk_metrics(returns, features)
            features = self._calculate_evt_diagnostics(returns, features)
            features = self._calculate_tail_risk_forecasting(returns, features)
            features = self._calculate_advanced_tail_metrics(returns, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating tail risk features: {e}")
            return self.create_empty_features(data.index)
    
    def _calculate_hill_estimator(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate Hill estimator for tail index."""
        try:
            # Sort returns in descending order (focus on positive tail)
            sorted_returns = np.sort(returns)
            n = len(sorted_returns)
            
            # Number of order statistics to use
            k = int(self.hill_fraction * n)
            k = max(k, self.hill_min_obs)
            k = min(k, n - 1)
            
            if k < 10:
                logger.warning("Insufficient data for Hill estimator")
                return features
            
            # Calculate Hill estimator
            # Focus on lower tail (negative returns)
            lower_tail = sorted_returns[:k]
            
            if len(lower_tail) > 0:
                # Hill estimator for lower tail
                log_ratios = []
                threshold_value = sorted_returns[k]
                
                for i in range(k):
                    if sorted_returns[i] < threshold_value and threshold_value != 0:
                        ratio = safe_divide(threshold_value, sorted_returns[i], default_value=0.0)
                        if ratio > 1:
                            log_ratios.append(np.log(ratio))
                
                if len(log_ratios) > 0:
                    hill_estimator = np.mean(log_ratios)
                    features['hill_estimator_tail_index'] = hill_estimator
                    
                    # Alpha parameter (inverse of tail index)
                    if hill_estimator > 0:
                        alpha = safe_divide(1, hill_estimator, default_value=0.0)
                        features['hill_estimator_alpha'] = alpha
                    
                    # Standard error
                    standard_error = safe_divide(hill_estimator, np.sqrt(k), default_value=0.0)
                    features['hill_estimator_standard_error'] = standard_error
                    
                    # Confidence interval (95%)
                    z_score = 1.96
                    ci_lower = hill_estimator - z_score * standard_error
                    ci_upper = hill_estimator + z_score * standard_error
                    
                    features['hill_estimator_confidence_interval_lower'] = ci_lower
                    features['hill_estimator_confidence_interval_upper'] = ci_upper
                    
                    # Goodness of fit (simplified)
                    goodness_of_fit = 1 - safe_divide(standard_error, hill_estimator, default_value=0.0)
                    features['hill_estimator_goodness_of_fit'] = max(0, goodness_of_fit)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating Hill estimator: {e}")
            return features
    
    def _calculate_peak_over_threshold(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate Peak-Over-Threshold (POT) analysis."""
        try:
            # Define threshold
            threshold = returns.quantile(self.threshold)
            features['pot_threshold'] = threshold
            
            # Get exceedances
            exceedances = returns[returns > threshold] - threshold
            features['pot_exceedances_count'] = len(exceedances)
            features['pot_exceedances_rate'] = safe_divide(len(exceedances), len(returns), default_value=0.0)
            
            if len(exceedances) < 20:
                logger.warning("Insufficient exceedances for POT analysis")
                return features
            
            # Fit Generalized Pareto Distribution
            try:
                # Simple exponential fit (shape parameter = 0)
                scale_param = exceedances.mean()
                features['pot_scale_parameter'] = scale_param
                features['pot_shape_parameter'] = 0  # Exponential distribution
                features['pot_location_parameter'] = threshold
                
                # Mean excess function
                mean_excess = exceedances.mean()
                features['pot_mean_excess_function'] = mean_excess
                
            except Exception as e:
                logger.debug(f"Error fitting GPD: {e}")
                # Use simple statistics
                features['pot_scale_parameter'] = exceedances.std()
                features['pot_shape_parameter'] = 0
                features['pot_location_parameter'] = threshold
                features['pot_mean_excess_function'] = exceedances.mean()
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating POT analysis: {e}")
            return features
    
    def _calculate_block_maxima(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate block maxima analysis."""
        try:
            # Block size (e.g., monthly blocks)
            block_size = 22  # Trading days per month
            
            if len(returns) < block_size * 6:  # Need at least 6 blocks
                logger.warning("Insufficient data for block maxima analysis")
                return features
            
            # Extract block maxima
            block_maxima = []
            for i in range(0, len(returns) - block_size + 1, block_size):
                block = returns.iloc[i:i + block_size]
                if len(block) == block_size:
                    block_maxima.append(block.max())
            
            if len(block_maxima) < 6:
                return features
            
            block_maxima = np.array(block_maxima)
            
            # Fit Generalized Extreme Value (GEV) distribution
            try:
                # Simple approach: use method of moments
                mean_bm = np.mean(block_maxima)
                std_bm = np.std(block_maxima)
                
                # Estimate parameters (simplified)
                features['block_maxima_location'] = mean_bm
                features['block_maxima_scale'] = std_bm
                features['block_maxima_shape'] = 0  # Gumbel distribution
                
                # Return levels
                return_periods = [100, 250, 1000]  # days
                for period in return_periods:
                    # Simplified return level calculation
                    return_prob = 1 / period
                    return_level = mean_bm - std_bm * np.log(-np.log(1 - return_prob))
                    features[f'block_maxima_return_level_{period}'] = return_level
                    
            except Exception as e:
                logger.debug(f"Error fitting GEV: {e}")
                features['block_maxima_location'] = np.mean(block_maxima)
                features['block_maxima_scale'] = np.std(block_maxima)
                features['block_maxima_shape'] = 0
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating block maxima: {e}")
            return features
    
    def _calculate_extreme_quantiles(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate extreme quantiles using EVT."""
        try:
            # Extreme quantiles
            extreme_levels = [0.999, 0.9995, 0.9999]
            extreme_labels = ['99_9', '99_95', '99_99']
            
            for level, label in zip(extreme_levels, extreme_labels):
                # Use empirical quantile if available
                if len(returns) >= 1000:
                    extreme_quantile = returns.quantile(level)
                else:
                    # Extrapolate using normal distribution
                    mean_return = returns.mean()
                    std_return = returns.std()
                    extreme_quantile = stats.norm.ppf(level, mean_return, std_return)
                
                features[f'extreme_quantile_{label}'] = extreme_quantile
            
            # Extrapolation beyond sample
            # Using Hill estimator if available
            hill_index = features.get('hill_estimator_tail_index', 0)
            
            if hill_index > 0:
                # Extrapolate using Hill estimator
                threshold = returns.quantile(0.95)
                n = len(returns)
                k = int(0.05 * n)  # Number of tail observations
                
                for confidence in [0.95, 0.99]:
                    # Extrapolated quantile
                    extrapolated_quantile = threshold * (safe_divide(n, k, default_value=1.0) * (1 - confidence)) ** (-hill_index)
                    conf_str = str(int(confidence * 100))
                    features[f'extreme_quantile_extrapolation_{conf_str}'] = extrapolated_quantile
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating extreme quantiles: {e}")
            return features
    
    def _calculate_tail_dependence(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate tail dependence measures."""
        try:
            # For single series, calculate tail dependence with lagged series
            if len(returns) < 100:
                return features
            
            # Create lagged series
            lagged_returns = returns.shift(1).dropna()
            aligned_returns = returns.iloc[1:]  # Align indices
            
            if len(aligned_returns) != len(lagged_returns):
                return features
            
            # Upper tail dependence
            high_threshold = 0.95
            threshold_value = aligned_returns.quantile(high_threshold)
            
            # Count joint exceedances
            joint_exceedances = ((aligned_returns > threshold_value) & 
                               (lagged_returns > threshold_value)).sum()
            marginal_exceedances = (aligned_returns > threshold_value).sum()
            
            if marginal_exceedances > 0:
                upper_tail_dependence = safe_divide(joint_exceedances, marginal_exceedances, default_value=0.0)
                features['tail_dependence_upper'] = upper_tail_dependence
            
            # Lower tail dependence
            low_threshold = 0.05
            threshold_value = aligned_returns.quantile(low_threshold)
            
            joint_exceedances = ((aligned_returns < threshold_value) & 
                               (lagged_returns < threshold_value)).sum()
            marginal_exceedances = (aligned_returns < threshold_value).sum()
            
            if marginal_exceedances > 0:
                lower_tail_dependence = safe_divide(joint_exceedances, marginal_exceedances, default_value=0.0)
                features['tail_dependence_lower'] = lower_tail_dependence
            
            # Overall tail dependence coefficient
            upper_td = features.get('tail_dependence_upper', 0)
            lower_td = features.get('tail_dependence_lower', 0)
            tail_dependence_coeff = safe_divide(upper_td + lower_td, 2, default_value=0.0)
            features['tail_dependence_coefficient'] = tail_dependence_coeff
            
            # Tail correlation (simplified)
            tail_data = aligned_returns[aligned_returns < aligned_returns.quantile(0.1)]
            lagged_tail_data = lagged_returns[aligned_returns < aligned_returns.quantile(0.1)]
            
            if len(tail_data) > 10:
                tail_correlation = tail_data.corr(lagged_tail_data)
                features['tail_correlation'] = tail_correlation if not np.isnan(tail_correlation) else 0
            
            # Tail beta (simplified)
            if len(tail_data) > 10:
                tail_beta = safe_divide(tail_data.std(), lagged_tail_data.std(), default_value=1.0)
                features['tail_beta'] = tail_beta
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating tail dependence: {e}")
            return features
    
    def _calculate_extreme_expected_shortfall(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate Expected Shortfall at extreme levels."""
        try:
            # Extreme confidence levels
            extreme_levels = [0.995, 0.999, 0.9995, 0.9999]
            extreme_labels = ['99_5', '99_9', '99_95', '99_99']
            
            for level, label in zip(extreme_levels, extreme_labels):
                # Calculate VaR at extreme level
                var_extreme = returns.quantile(1 - level)
                
                # Calculate Expected Shortfall
                tail_losses = returns[returns <= var_extreme]
                
                if len(tail_losses) > 0:
                    expected_shortfall = tail_losses.mean()
                else:
                    # Use theoretical approach
                    mean_return = returns.mean()
                    std_return = returns.std()
                    var_normal = stats.norm.ppf(1 - level, mean_return, std_return)
                    
                    # Expected shortfall for normal distribution
                    expected_shortfall = mean_return - std_return * stats.norm.pdf(stats.norm.ppf(1 - level)) / (1 - level)
                
                features[f'expected_shortfall_{label}'] = abs(expected_shortfall)
            
            # Coherent risk measure (average of ES at different levels)
            es_values = [features.get(f'expected_shortfall_{label}', 0) for label in extreme_labels]
            coherent_risk = np.mean(es_values)
            features['coherent_risk_measure'] = coherent_risk
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating extreme expected shortfall: {e}")
            return features
    
    def _calculate_tail_risk_metrics(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate tail risk ratios and metrics."""
        try:
            # Tail risk ratio
            tail_threshold = 0.05
            tail_returns = returns[returns < returns.quantile(tail_threshold)]
            
            if len(tail_returns) > 0:
                tail_mean = tail_returns.mean()
                overall_mean = returns.mean()
                
                if overall_mean != 0:
                    tail_risk_ratio = abs(safe_divide(tail_mean, overall_mean, default_value=0.0))
                    features['tail_risk_ratio'] = tail_risk_ratio
                
                # Tail expectation
                features['tail_expectation'] = abs(tail_mean)
                
                # Tail variance
                features['tail_variance'] = tail_returns.var()
                
                # Tail skewness and kurtosis
                if len(tail_returns) > 10:
                    features['tail_skewness'] = tail_returns.skew()
                    features['tail_kurtosis'] = tail_returns.kurtosis()
                
                # Tail concentration ratio
                tail_range = tail_returns.max() - tail_returns.min()
                overall_range = returns.max() - returns.min()
                
                if overall_range > 0:
                    concentration_ratio = safe_divide(tail_range, overall_range, default_value=0.0)
                    features['tail_concentration_ratio'] = concentration_ratio
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating tail risk metrics: {e}")
            return features
    
    def _calculate_evt_diagnostics(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate EVT model diagnostics."""
        try:
            # Simple diagnostics based on available data
            
            # Model fit quality (simplified)
            threshold = returns.quantile(0.95)
            exceedances = returns[returns > threshold] - threshold
            
            if len(exceedances) > 10:
                # Simple AIC/BIC approximation
                n = len(exceedances)
                log_likelihood = -n * np.log(exceedances.mean()) - n
                
                features['evt_model_log_likelihood'] = log_likelihood
                features['evt_model_aic'] = 2 * 2 - 2 * log_likelihood  # 2 parameters
                features['evt_model_bic'] = 2 * np.log(n) - 2 * log_likelihood
                
                # Goodness of fit (simplified)
                # Use Kolmogorov-Smirnov test approximation
                empirical_cdf = np.arange(1, len(exceedances) + 1) / len(exceedances)
                theoretical_cdf = 1 - np.exp(-exceedances / exceedances.mean())
                
                ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
                # Approximate p-value
                gof_pvalue = np.exp(-2 * len(exceedances) * ks_statistic**2)
                features['evt_goodness_of_fit_pvalue'] = gof_pvalue
                
                # QQ plot correlation (simplified)
                theoretical_quantiles = np.sort(theoretical_cdf)
                empirical_quantiles = np.sort(empirical_cdf)
                qq_correlation = np.corrcoef(theoretical_quantiles, empirical_quantiles)[0, 1]
                features['evt_qq_plot_correlation'] = qq_correlation if not np.isnan(qq_correlation) else 0
                
                # PP plot correlation
                features['evt_pp_plot_correlation'] = qq_correlation  # Same for this simplified version
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating EVT diagnostics: {e}")
            return features
    
    def _calculate_tail_risk_forecasting(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate tail risk forecasting metrics."""
        try:
            # Simple tail risk forecasting
            current_tail_risk = abs(returns.quantile(0.05))
            
            # Persistence of tail risk
            if len(returns) > 60:
                # Calculate rolling tail risk
                rolling_tail_risk = []
                window = 60
                
                for i in range(window, len(returns)):
                    window_returns = returns.iloc[i-window:i]
                    tail_risk = abs(window_returns.quantile(0.05))
                    rolling_tail_risk.append(tail_risk)
                
                if len(rolling_tail_risk) > 10:
                    rolling_series = pd.Series(rolling_tail_risk)
                    
                    # Persistence (autocorrelation)
                    persistence = rolling_series.autocorr(lag=1)
                    features['tail_risk_persistence'] = persistence if not np.isnan(persistence) else 0
                    
                    # Clustering
                    clustering = rolling_series.autocorr(lag=5)
                    features['tail_risk_clustering'] = clustering if not np.isnan(clustering) else 0
                    
                    # Simple forecasting
                    for horizon in [1, 5, 21]:
                        # Mean reversion forecast
                        long_term_avg = rolling_series.mean()
                        forecast = long_term_avg + (rolling_series.iloc[-1] - long_term_avg) * (persistence ** horizon)
                        features[f'tail_risk_forecast_{horizon}d'] = forecast
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating tail risk forecasting: {e}")
            return features
    
    def _calculate_advanced_tail_metrics(self, returns: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced tail risk metrics."""
        try:
            # Tail conditional expectation
            tail_threshold = 0.05
            tail_returns = returns[returns < returns.quantile(tail_threshold)]
            
            if len(tail_returns) > 0:
                tail_conditional_expectation = tail_returns.mean()
                features['tail_conditional_expectation'] = abs(tail_conditional_expectation)
            
            # Spectral risk measure (simplified)
            if len(returns) > 100:
                # Weight function for spectral risk measure
                quantile_levels = np.linspace(0.01, 0.99, 99)
                quantiles = [returns.quantile(q) for q in quantile_levels]
                
                # Simple spectral risk measure
                spectral_risk = np.mean(quantiles)
                features['tail_spectral_risk_measure'] = spectral_risk
            
            # Distortion risk measure
            # Using power distortion function
            alpha = 0.5  # Distortion parameter
            if len(returns) > 50:
                sorted_returns = np.sort(returns)
                n = len(sorted_returns)
                
                # Power distortion weights
                weights = np.array([safe_divide(i, n, default_value=0.0)**alpha - safe_divide(i-1, n, default_value=0.0)**alpha for i in range(1, n+1)])
                distorted_risk = np.sum(weights * sorted_returns)
                features['tail_distortion_risk_measure'] = distorted_risk
            
            # Coherent risk measure (combination of different measures)
            var_95 = abs(returns.quantile(0.05))
            es_95 = features.get('expected_shortfall_99_5', var_95)
            coherent_risk = 0.5 * var_95 + 0.5 * es_95
            features['tail_coherent_risk_measure'] = coherent_risk
            
            # Robustness measure
            # Measure of stability of tail risk estimates
            if len(returns) > 100:
                # Bootstrap-like approach
                n_bootstrap = 10
                tail_estimates = []
                
                for _ in range(n_bootstrap):
                    # Sample with replacement
                    bootstrap_sample = returns.sample(n=len(returns), replace=True)
                    tail_estimate = abs(bootstrap_sample.quantile(0.05))
                    tail_estimates.append(tail_estimate)
                
                # Robustness as inverse of coefficient of variation
                if len(tail_estimates) > 0:
                    cv = safe_divide(np.std(tail_estimates), np.mean(tail_estimates), default_value=0.0)
                    robustness = safe_divide(1, 1 + cv, default_value=1.0)
                    features['tail_robustness_measure'] = robustness
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating advanced tail metrics: {e}")
            return features