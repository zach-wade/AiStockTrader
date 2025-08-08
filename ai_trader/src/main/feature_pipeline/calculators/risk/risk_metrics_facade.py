"""
Risk Metrics Facade

Unified interface for all risk calculations, orchestrating multiple
specialized risk calculators to provide comprehensive risk analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_risk import BaseRiskCalculator
from .var_calculator import VaRCalculator
from .volatility_calculator import VolatilityCalculator
from .drawdown_calculator import DrawdownCalculator
from .performance_calculator import PerformanceCalculator
from .stress_test_calculator import StressTestCalculator
from .tail_risk_calculator import TailRiskCalculator

from ..helpers import create_feature_dataframe, safe_divide, aggregate_features

from main.utils.core import get_logger, process_in_batches

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class RiskMetricsFacade(BaseRiskCalculator):
    """
    Facade for comprehensive risk metric calculation.
    
    Orchestrates all risk calculators to provide:
    - Complete risk feature set (205+ features)
    - Parallel calculation for performance
    - Cross-calculator risk aggregation
    - Unified risk scoring
    - Risk limit monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk metrics facade.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Initialize all risk calculators
        self.calculators = {
            'var': VaRCalculator(config),
            'volatility': VolatilityCalculator(config),
            'drawdown': DrawdownCalculator(config),
            'performance': PerformanceCalculator(config),
            'stress_test': StressTestCalculator(config),
            'tail_risk': TailRiskCalculator(config)
        }
        
        # Parallel processing configuration
        self.max_workers = config.get('max_workers', 4)
        self.batch_size = config.get('batch_size', 1000)
        
        # Risk aggregation weights
        self.risk_weights = config.get('risk_weights', {
            'var': 0.25,
            'volatility': 0.20,
            'drawdown': 0.20,
            'tail_risk': 0.15,
            'stress_test': 0.20
        })
        
        # Risk limits from config
        self.risk_limits = config.get('risk_limits', {
            'max_var_95': 0.02,
            'max_volatility': 0.20,
            'max_drawdown': 0.15,
            'min_sharpe': 0.5,
            'max_tail_risk': 0.10
        })
        
        logger.info(f"Initialized RiskMetricsFacade with {len(self.calculators)} calculators")
    
    def get_feature_names(self) -> List[str]:
        """Get complete list of risk feature names."""
        features = []
        
        # Collect features from all calculators
        for name, calculator in self.calculators.items():
            calc_features = calculator.get_feature_names()
            # Prefix with calculator name for clarity
            features.extend([f"{name}_{feat}" for feat in calc_features])
        
        # Add facade-specific composite features
        features.extend([
            'risk_composite_score',
            'risk_adjusted_return',
            'risk_budget_utilization',
            'risk_concentration',
            'risk_diversification_ratio',
            'risk_regime_indicator',
            'downside_risk_score',
            'tail_risk_exposure',
            'risk_limit_proximity',
            'integrated_risk_measure'
        ])
        
        return features
    
    def get_required_columns(self) -> List[str]:
        """
        Get list of required input columns for risk calculations.
        
        Returns:
            List of required column names
        """
        # Risk calculations typically need OHLCV data
        return ['open', 'high', 'low', 'close', 'volume']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all risk metrics using parallel processing.
        
        Args:
            data: DataFrame with price/return data
            
        Returns:
            DataFrame with all risk features
        """
        try:
            # Validate input data
            if data.empty:
                return self._create_empty_features(data.index)
            
            # Prepare returns data if needed
            returns_data = self.prepare_returns_data(data)
            
            # Initialize features DataFrame
            features = create_feature_dataframe(returns_data.index)
            
            # Calculate features in parallel
            if self.max_workers > 1:
                calc_features = self._calculate_parallel(returns_data)
            else:
                calc_features = self._calculate_sequential(returns_data)
            
            # Merge calculator features
            for calc_name, calc_df in calc_features.items():
                # Prefix column names
                calc_df.columns = [f"{calc_name}_{col}" for col in calc_df.columns]
                features = pd.concat([features, calc_df], axis=1)
            
            # Calculate cross-risk metrics
            cross_risk_features = self._calculate_cross_risk_metrics(calc_features)
            features = pd.concat([features, cross_risk_features], axis=1)
            
            # Calculate composite risk scores
            composite_features = self._calculate_composite_scores(calc_features)
            features = pd.concat([features, composite_features], axis=1)
            
            # Monitor risk limits
            limit_features = self._monitor_risk_limits(calc_features)
            features = pd.concat([features, limit_features], axis=1)
            
            # Calculate integrated risk measures
            integrated_features = self._calculate_integrated_measures(
                calc_features, returns_data
            )
            features = pd.concat([features, integrated_features], axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in RiskMetricsFacade calculation: {e}")
            return self._create_empty_features(data.index)
    
    def _calculate_parallel(
        self,
        returns_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Calculate features using parallel processing."""
        calc_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit calculation tasks
            future_to_calc = {
                executor.submit(
                    self._safe_calculate,
                    calc_name,
                    calculator,
                    returns_data
                ): calc_name
                for calc_name, calculator in self.calculators.items()
            }
            
            # Collect results
            for future in as_completed(future_to_calc):
                calc_name = future_to_calc[future]
                try:
                    result = future.result()
                    if result is not None:
                        calc_results[calc_name] = result
                except Exception as e:
                    logger.error(f"Error in {calc_name} calculation: {e}")
                    calc_results[calc_name] = create_feature_dataframe(
                        returns_data.index
                    )
        
        return calc_results
    
    def _calculate_sequential(
        self,
        returns_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Calculate features sequentially."""
        calc_results = {}
        
        for calc_name, calculator in self.calculators.items():
            try:
                result = calculator.calculate(returns_data)
                calc_results[calc_name] = result
            except Exception as e:
                logger.error(f"Error in {calc_name} calculation: {e}")
                calc_results[calc_name] = create_feature_dataframe(
                    returns_data.index
                )
        
        return calc_results
    
    def _safe_calculate(
        self,
        calc_name: str,
        calculator: BaseRiskCalculator,
        returns_data: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Safely calculate features with error handling."""
        try:
            logger.debug(f"Calculating {calc_name} features")
            return calculator.calculate(returns_data)
        except Exception as e:
            logger.error(f"Failed to calculate {calc_name} features: {e}")
            return None
    
    def _calculate_cross_risk_metrics(
        self,
        calc_features: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate metrics that span multiple risk dimensions."""
        features = pd.DataFrame(
            index=next(iter(calc_features.values())).index
        )
        
        # Risk-adjusted return (using multiple risk measures)
        if 'performance' in calc_features and 'var' in calc_features:
            returns = calc_features.get('performance', {}).get('cumulative_return', 0)
            var_95 = calc_features.get('var', {}).get('var_95_historical', 1)
            
            features['risk_adjusted_return'] = safe_divide(returns, abs(var_95))
        
        # Risk concentration (how concentrated is risk across measures)
        risk_values = []
        if 'var' in calc_features:
            risk_values.append(calc_features['var'].get('var_95_historical', 0))
        if 'volatility' in calc_features:
            risk_values.append(calc_features['volatility'].get('volatility_ewma', 0))
        if 'drawdown' in calc_features:
            risk_values.append(calc_features['drawdown'].get('max_drawdown', 0))
        
        if risk_values:
            risk_array = pd.concat(risk_values, axis=1).abs()
            total_risk = risk_array.sum(axis=1)
            max_risk = risk_array.max(axis=1)
            features['risk_concentration'] = safe_divide(max_risk, total_risk)
        
        # Risk diversification ratio
        if len(risk_values) > 1:
            # Ratio of sum of individual risks to portfolio risk
            individual_sum = risk_array.sum(axis=1)
            portfolio_risk = risk_array.std(axis=1) * np.sqrt(len(risk_values))
            features['risk_diversification_ratio'] = safe_divide(
                individual_sum, portfolio_risk
            )
        
        return features
    
    def _calculate_composite_scores(
        self,
        calc_features: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate composite risk scores."""
        features = pd.DataFrame(
            index=next(iter(calc_features.values())).index
        )
        
        # Risk composite score (weighted average of normalized risks)
        risk_components = []
        
        for risk_type, weight in self.risk_weights.items():
            if risk_type in calc_features:
                # Get primary risk metric for each type
                if risk_type == 'var':
                    metric = calc_features[risk_type].get('var_95_historical', 0)
                elif risk_type == 'volatility':
                    metric = calc_features[risk_type].get('volatility_realized', 0)
                elif risk_type == 'drawdown':
                    metric = calc_features[risk_type].get('current_drawdown', 0)
                elif risk_type == 'tail_risk':
                    metric = calc_features[risk_type].get('expected_shortfall_95', 0)
                elif risk_type == 'stress_test':
                    metric = calc_features[risk_type].get('worst_case_loss', 0)
                else:
                    metric = 0
                
                # Normalize and weight
                normalized = abs(metric) / 0.1  # Normalize to 10% baseline
                risk_components.append(normalized * weight)
        
        features['risk_composite_score'] = sum(risk_components)
        
        # Risk regime indicator
        vol_regime = self._determine_volatility_regime(calc_features)
        drawdown_regime = self._determine_drawdown_regime(calc_features)
        
        features['risk_regime_indicator'] = (vol_regime + drawdown_regime) / 2
        
        # Downside risk score
        downside_components = []
        
        if 'var' in calc_features:
            downside_components.append(
                calc_features['var'].get('expected_shortfall_95', 0)
            )
        if 'drawdown' in calc_features:
            downside_components.append(
                calc_features['drawdown'].get('pain_index', 0)
            )
        if 'performance' in calc_features:
            downside_components.append(
                -calc_features['performance'].get('sortino_ratio', 0) / 10
            )
        
        if downside_components:
            features['downside_risk_score'] = np.mean(
                [abs(c) for c in downside_components]
            )
        
        # Tail risk exposure
        if 'tail_risk' in calc_features:
            tail_components = [
                calc_features['tail_risk'].get('hill_tail_index', 0),
                calc_features['tail_risk'].get('expected_tail_loss', 0),
                calc_features['tail_risk'].get('tail_risk_contribution', 0)
            ]
            features['tail_risk_exposure'] = np.mean(
                [abs(c) for c in tail_components if c != 0]
            )
        
        return features
    
    def _monitor_risk_limits(
        self,
        calc_features: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Monitor proximity to risk limits."""
        features = pd.DataFrame(
            index=next(iter(calc_features.values())).index
        )
        
        limit_distances = []
        
        # VaR limit
        if 'var' in calc_features and 'max_var_95' in self.risk_limits:
            var_95 = abs(calc_features['var'].get('var_95_historical', 0))
            var_limit = self.risk_limits['max_var_95']
            var_proximity = var_95 / var_limit
            limit_distances.append(var_proximity)
        
        # Volatility limit
        if 'volatility' in calc_features and 'max_volatility' in self.risk_limits:
            vol = calc_features['volatility'].get('volatility_annualized', 0)
            vol_limit = self.risk_limits['max_volatility']
            vol_proximity = vol / vol_limit
            limit_distances.append(vol_proximity)
        
        # Drawdown limit
        if 'drawdown' in calc_features and 'max_drawdown' in self.risk_limits:
            dd = abs(calc_features['drawdown'].get('current_drawdown', 0))
            dd_limit = self.risk_limits['max_drawdown']
            dd_proximity = dd / dd_limit
            limit_distances.append(dd_proximity)
        
        # Sharpe ratio limit (inverse - lower is worse)
        if 'performance' in calc_features and 'min_sharpe' in self.risk_limits:
            sharpe = calc_features['performance'].get('sharpe_ratio', 0)
            sharpe_limit = self.risk_limits['min_sharpe']
            sharpe_proximity = sharpe_limit / (sharpe + 0.1)  # Avoid division by zero
            limit_distances.append(sharpe_proximity)
        
        # Overall limit proximity (max = closest to breach)
        if limit_distances:
            features['risk_limit_proximity'] = pd.concat(
                [pd.Series(d, index=features.index) for d in limit_distances],
                axis=1
            ).max(axis=1)
        else:
            features['risk_limit_proximity'] = 0
        
        # Risk budget utilization
        features['risk_budget_utilization'] = features['risk_limit_proximity'].clip(0, 1)
        
        return features
    
    def _calculate_integrated_measures(
        self,
        calc_features: Dict[str, pd.DataFrame],
        returns_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate integrated risk measures combining multiple perspectives."""
        features = pd.DataFrame(index=returns_data.index)
        
        # Integrated Risk Measure (IRM)
        # Combines multiple risk metrics into single coherent measure
        irm_components = []
        
        # Market risk component
        if 'var' in calc_features:
            market_risk = calc_features['var'].get('var_95_parametric', 0)
            irm_components.append(abs(market_risk) * 0.3)
        
        # Extreme risk component
        if 'tail_risk' in calc_features:
            extreme_risk = calc_features['tail_risk'].get('expected_tail_loss', 0)
            irm_components.append(abs(extreme_risk) * 0.2)
        
        # Drawdown risk component
        if 'drawdown' in calc_features:
            dd_risk = calc_features['drawdown'].get('expected_drawdown', 0)
            irm_components.append(abs(dd_risk) * 0.25)
        
        # Volatility risk component
        if 'volatility' in calc_features:
            vol_risk = calc_features['volatility'].get('conditional_volatility', 0)
            irm_components.append(vol_risk * 0.25)
        
        if irm_components:
            features['integrated_risk_measure'] = sum(irm_components)
        else:
            features['integrated_risk_measure'] = 0
        
        return features
    
    def _determine_volatility_regime(
        self,
        calc_features: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Determine current volatility regime."""
        if 'volatility' not in calc_features:
            return 0
        
        vol_features = calc_features['volatility']
        current_vol = vol_features.get('volatility_ewma', 0)
        historical_vol = vol_features.get('volatility_historical', 0)
        
        # Regime classification
        # -1: Low vol, 0: Normal, 1: High vol, 2: Extreme vol
        vol_ratio = safe_divide(current_vol, historical_vol)
        
        regime = pd.Series(0, index=vol_features.index)
        regime[vol_ratio < 0.7] = -1
        regime[vol_ratio > 1.5] = 1
        regime[vol_ratio > 2.0] = 2
        
        return regime
    
    def _determine_drawdown_regime(
        self,
        calc_features: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Determine current drawdown regime."""
        if 'drawdown' not in calc_features:
            return 0
        
        dd_features = calc_features['drawdown']
        current_dd = abs(dd_features.get('current_drawdown', 0))
        
        # Regime classification based on drawdown severity
        regime = pd.Series(0, index=dd_features.index)
        regime[current_dd > 0.05] = 1   # Mild drawdown
        regime[current_dd > 0.10] = 2   # Moderate drawdown
        regime[current_dd > 0.20] = 3   # Severe drawdown
        
        return regime
    
    def prepare_returns_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare returns data from price data if needed."""
        if 'returns' in data.columns:
            return data
        elif 'close' in data.columns:
            returns = data['close'].pct_change()
            data = data.copy()
            data['returns'] = returns
            return data
        else:
            # Assume the data is already returns
            if len(data.columns) == 1:
                data = data.copy()
                data.columns = ['returns']
            return data
    
    def get_risk_summary(
        self,
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get summary of key risk metrics.
        
        Args:
            features: DataFrame with calculated risk features
            
        Returns:
            Dictionary with risk summary
        """
        summary = {}
        
        # Current risk levels
        summary['current_risk'] = {
            'composite_score': features['risk_composite_score'].iloc[-1],
            'var_95': features.get('var_var_95_historical', 0).iloc[-1],
            'volatility': features.get('volatility_volatility_annualized', 0).iloc[-1],
            'drawdown': features.get('drawdown_current_drawdown', 0).iloc[-1],
            'sharpe': features.get('performance_sharpe_ratio', 0).iloc[-1]
        }
        
        # Risk limits status
        summary['limit_status'] = {
            'proximity': features['risk_limit_proximity'].iloc[-1],
            'budget_used': features['risk_budget_utilization'].iloc[-1],
            'breaches': (features['risk_limit_proximity'] > 1).sum()
        }
        
        # Risk regime
        summary['regime'] = {
            'indicator': features['risk_regime_indicator'].iloc[-1],
            'description': self._describe_risk_regime(
                features['risk_regime_indicator'].iloc[-1]
            )
        }
        
        return summary
    
    def _describe_risk_regime(self, regime_value: float) -> str:
        """Describe risk regime based on indicator value."""
        if regime_value < 0:
            return "Low risk environment"
        elif regime_value < 1:
            return "Normal risk environment"
        elif regime_value < 2:
            return "Elevated risk environment"
        else:
            return "High risk environment"