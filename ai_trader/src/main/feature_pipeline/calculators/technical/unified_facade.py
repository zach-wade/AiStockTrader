"""
Unified Technical Indicators Facade

Provides backward compatibility with the original UnifiedTechnicalIndicatorsCalculator
by orchestrating calls to the specialized calculator modules. This facade maintains
the same interface while leveraging the refactored, single-responsibility calculators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from .base_technical import BaseTechnicalCalculator
from .trend_indicators import TrendIndicatorsCalculator
from .momentum_indicators import MomentumIndicatorsCalculator
from .volatility_indicators import VolatilityIndicatorsCalculator
from .volume_indicators import VolumeIndicatorsCalculator
from .adaptive_indicators import AdaptiveIndicatorsCalculator


class UnifiedTechnicalIndicatorsFacade(BaseTechnicalCalculator):
    """
    Facade that provides unified access to all technical indicators
    while maintaining backward compatibility with the original interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified facade with all specialized calculators."""
        super().__init__(config)
        
        # Initialize specialized calculators
        self.trend_calculator = TrendIndicatorsCalculator(config)
        self.momentum_calculator = MomentumIndicatorsCalculator(config)
        self.volatility_calculator = VolatilityIndicatorsCalculator(config)
        self.volume_calculator = VolumeIndicatorsCalculator(config)
        self.adaptive_calculator = AdaptiveIndicatorsCalculator(config)
        
        # Maintain list of calculators for easy iteration
        self.calculators = [
            self.trend_calculator,
            self.momentum_calculator,
            self.volatility_calculator,
            self.volume_calculator,
            self.adaptive_calculator
        ]
    
    def get_feature_names(self) -> List[str]:
        """Return combined list of all technical indicator feature names."""
        all_features = []
        for calculator in self.calculators:
            all_features.extend(calculator.get_feature_names())
        return all_features
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators using specialized calculators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicator features
        """
        try:
            # Create empty features DataFrame
            features = self.create_feature_dataframe(data.index)
            
            # Calculate features from each specialized calculator
            for calculator in self.calculators:
                try:
                    calculator_features = calculator.calculate(data)
                    if not calculator_features.empty:
                        features = self.combine_features(features, calculator_features)
                except Exception as e:
                    print(f"Error in {calculator.__class__.__name__}: {e}")
                    continue
            
            return features
            
        except Exception as e:
            print(f"Error in unified technical indicators calculation: {e}")
            return self.create_feature_dataframe(data.index)
    
    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Legacy method name for backward compatibility.
        Equivalent to calculate() method.
        """
        return self.calculate(data)
    
    def calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate only trend indicators."""
        return self.trend_calculator.calculate(data)
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate only momentum indicators."""
        return self.momentum_calculator.calculate(data)
    
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate only volatility indicators."""
        return self.volatility_calculator.calculate(data)
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate only volume indicators."""
        return self.volume_calculator.calculate(data)
    
    def calculate_adaptive_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate only adaptive indicators."""
        return self.adaptive_calculator.calculate(data)
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """Get combined calculation statistics from all calculators."""
        combined_stats = super().get_calculation_stats()
        
        calculator_stats = {}
        for calculator in self.calculators:
            calc_name = calculator.__class__.__name__
            calculator_stats[calc_name] = calculator.get_calculation_stats()
        
        combined_stats['calculator_breakdown'] = calculator_stats
        return combined_stats
    
    def reset_cache(self):
        """Reset cache for all calculators."""
        super().reset_cache()
        for calculator in self.calculators:
            calculator.reset_cache()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance from all calculators."""
        combined_importance = {}
        
        for calculator in self.calculators:
            calc_importance = calculator.get_feature_importance()
            combined_importance.update(calc_importance)
        
        return combined_importance
    
    def set_feature_importance(self, importance: Dict[str, float]):
        """Set feature importance across relevant calculators."""
        super().set_feature_importance(importance)
        
        # Distribute importance to appropriate calculators based on feature names
        for calculator in self.calculators:
            calc_features = calculator.get_feature_names()
            calc_importance = {
                feature: importance.get(feature, 0.0) 
                for feature in calc_features 
                if feature in importance
            }
            if calc_importance:
                calculator.set_feature_importance(calc_importance)