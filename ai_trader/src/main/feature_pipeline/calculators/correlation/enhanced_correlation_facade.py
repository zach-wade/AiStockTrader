"""
Enhanced Correlation Facade

Unified facade combining all specialized correlation calculators for backward compatibility.
Provides the same interface as the original EnhancedCorrelationCalculator while leveraging
the modular, specialized calculator architecture.

This facade instantiates and coordinates:
- RollingCorrelationCalculator: Benchmark correlations and rolling statistics
- BetaAnalysisCalculator: Dynamic beta and regime-dependent analysis  
- StabilityAnalysisCalculator: Correlation stability and breakdown detection
- LeadLagCalculator: Temporal relationship and timing analysis
- PCACorrelationCalculator: Principal component analysis and exposure
- RegimeCorrelationCalculator: Regime-dependent correlation analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base_correlation import BaseCorrelationCalculator
from .rolling_calculator import RollingCorrelationCalculator
from .beta_calculator import BetaAnalysisCalculator
from .stability_calculator import StabilityAnalysisCalculator
from .leadlag_calculator import LeadLagCalculator
from .pca_calculator import PCACorrelationCalculator
from .regime_calculator import RegimeCorrelationCalculator

logger = logging.getLogger(__name__)


class EnhancedCorrelationCalculator(BaseCorrelationCalculator):
    """
    Facade combining all specialized correlation calculators.
    
    Maintains 100% backward compatibility with the original EnhancedCorrelationCalculator
    while leveraging the new modular architecture for improved maintainability
    and performance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the enhanced correlation facade."""
        super().__init__(config)
        
        # Initialize all specialized calculators
        self.calculators = {
            'rolling': RollingCorrelationCalculator(config),
            'beta': BetaAnalysisCalculator(config),
            'stability': StabilityAnalysisCalculator(config),
            'leadlag': LeadLagCalculator(config),
            'pca': PCACorrelationCalculator(config),
            'regime': RegimeCorrelationCalculator(config)
        }
        
        logger.info(f"EnhancedCorrelationCalculator facade initialized with {len(self.calculators)} specialized calculators")
    
    def get_required_columns(self) -> List[str]:
        """Get the list of required columns for correlation calculations."""
        return ['symbol', 'timestamp', 'close']
    
    def get_feature_names(self) -> List[str]:
        """Return combined list of all feature names from specialized calculators."""
        all_features = []
        
        for calc_name, calculator in self.calculators.items():
            try:
                features = calculator.get_feature_names()
                all_features.extend(features)
                logger.debug(f"{calc_name} calculator: {len(features)} features")
            except Exception as e:
                logger.error(f"Error getting feature names from {calc_name} calculator: {e}")
        
        logger.info(f"Total features from all calculators: {len(all_features)}")
        return all_features
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all correlation features using specialized calculators.
        
        Args:
            data: DataFrame with symbol, timestamp, close columns
            
        Returns:
            DataFrame with all correlation features combined
        """
        try:
            # Validate input data
            if not self.validate_input_data(data):
                logger.warning("Input data validation failed")
                return self.create_empty_features(data.index)
            
            # Create base features DataFrame
            combined_features = self.create_empty_features(data.index)
            
            if len(data) < 50:  # Need minimum data for correlation analysis
                logger.warning("Insufficient data for correlation analysis")
                return combined_features
            
            # Calculate features from each specialized calculator
            calculation_results = {}
            
            for calc_name, calculator in self.calculators.items():
                try:
                    logger.debug(f"Calculating {calc_name} features...")
                    
                    # Calculate features
                    calc_features = calculator.calculate(data)
                    
                    if calc_features is not None and not calc_features.empty:
                        # Merge with combined features
                        for column in calc_features.columns:
                            if column not in combined_features.columns:
                                # Align calc_features with combined_features index
                                if len(calc_features) == len(combined_features):
                                    combined_features[column] = calc_features[column].values
                                else:
                                    # Try to align by index
                                    try:
                                        aligned_values = calc_features[column].reindex(combined_features.index, fill_value=0.0)
                                        combined_features[column] = aligned_values
                                    except (ValueError, TypeError, KeyError):
                                        # Fallback: use first value or zero
                                        fill_value = calc_features[column].iloc[0] if len(calc_features) > 0 else 0.0
                                        combined_features[column] = fill_value
                            else:
                                logger.warning(f"Duplicate feature name '{column}' from {calc_name} calculator")
                        
                        calculation_results[calc_name] = {
                            'features_count': len(calc_features.columns),
                            'status': 'success'
                        }
                        
                        logger.debug(f"{calc_name} calculator: {len(calc_features.columns)} features calculated")
                    else:
                        logger.warning(f"{calc_name} calculator returned empty features")
                        calculation_results[calc_name] = {
                            'features_count': 0,
                            'status': 'empty'
                        }
                
                except Exception as e:
                    logger.error(f"Error calculating {calc_name} features: {e}")
                    calculation_results[calc_name] = {
                        'error': str(e),
                        'status': 'failed'
                    }
            
            # Log summary
            total_features = len(combined_features.columns)
            successful_calcs = sum(1 for result in calculation_results.values() if result['status'] == 'success')
            
            logger.info(f"Feature calculation complete: {total_features} total features from {successful_calcs}/{len(self.calculators)} calculators")
            
            # Validate feature completeness
            self._validate_feature_completeness(combined_features, calculation_results)
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error in facade feature calculation: {e}")
            return self.create_empty_features(data.index)
    
    def _validate_feature_completeness(self, features: pd.DataFrame, results: Dict[str, Any]):
        """Validate that feature calculation is complete and consistent."""
        try:
            # Check expected feature counts
            expected_counts = {
                'rolling': 24,      # Rolling correlations and dynamics
                'beta': 19,         # Beta analysis features
                'stability': 17,    # Stability analysis features
                'leadlag': 19,      # Lead-lag timing features
                'pca': 18,          # PCA analysis features
                'regime': 17        # Regime correlation features
            }
            
            total_expected = sum(expected_counts.values())
            actual_total = len(features.columns)
            
            logger.info(f"Feature count validation: {actual_total}/{total_expected} expected features")
            
            # Validate individual calculator contributions
            for calc_name, expected_count in expected_counts.items():
                if calc_name in results and results[calc_name]['status'] == 'success':
                    actual_count = results[calc_name]['features_count']
                    if actual_count != expected_count:
                        logger.warning(f"{calc_name} calculator: {actual_count}/{expected_count} features (expected {expected_count})")
                else:
                    logger.warning(f"{calc_name} calculator: failed or missing")
            
            # Check for null features
            null_counts = features.isnull().sum()
            if null_counts.sum() > 0:
                high_null_features = null_counts[null_counts > len(features) * 0.5]
                if len(high_null_features) > 0:
                    logger.warning(f"High null count features: {list(high_null_features.index)}")
            
        except Exception as e:
            logger.error(f"Error validating feature completeness: {e}")
    
    def get_calculator_info(self) -> Dict[str, Any]:
        """Get information about all specialized calculators."""
        info = {
            'facade_version': '1.0.0',
            'total_calculators': len(self.calculators),
            'calculators': {}
        }
        
        for calc_name, calculator in self.calculators.items():
            try:
                feature_count = len(calculator.get_feature_names())
                info['calculators'][calc_name] = {
                    'class_name': calculator.__class__.__name__,
                    'feature_count': feature_count,
                    'status': 'initialized'
                }
            except Exception as e:
                info['calculators'][calc_name] = {
                    'class_name': calculator.__class__.__name__,
                    'error': str(e),
                    'status': 'error'
                }
        
        return info
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get features organized by calculator groups."""
        feature_groups = {}
        
        for calc_name, calculator in self.calculators.items():
            try:
                features = calculator.get_feature_names()
                feature_groups[calc_name] = features
            except Exception as e:
                logger.error(f"Error getting features from {calc_name}: {e}")
                feature_groups[calc_name] = []
        
        return feature_groups
    
    def calculate_individual(self, data: pd.DataFrame, calculator_name: str) -> pd.DataFrame:
        """Calculate features from a specific calculator only."""
        if calculator_name not in self.calculators:
            available = list(self.calculators.keys())
            raise ValueError(f"Calculator '{calculator_name}' not found. Available: {available}")
        
        calculator = self.calculators[calculator_name]
        return calculator.calculate(data)
    
    def get_calculator_status(self) -> Dict[str, str]:
        """Get status of all calculators."""
        status = {}
        
        for calc_name, calculator in self.calculators.items():
            try:
                # Simple health check - try to get feature names
                features = calculator.get_feature_names()
                status[calc_name] = f"healthy ({len(features)} features)"
            except Exception as e:
                status[calc_name] = f"error: {str(e)}"
        
        return status
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration across all calculators."""
        validation_results = {
            'facade': {'status': 'valid'},
            'calculators': {}
        }
        
        for calc_name, calculator in self.calculators.items():
            try:
                # Basic validation checks
                features = calculator.get_feature_names()
                
                validation_results['calculators'][calc_name] = {
                    'status': 'valid',
                    'feature_count': len(features),
                    'has_config': hasattr(calculator, 'config'),
                    'has_correlation_config': hasattr(calculator, 'correlation_config')
                }
                
                # Check for required attributes
                if not hasattr(calculator, 'calculate'):
                    validation_results['calculators'][calc_name]['status'] = 'missing_calculate_method'
                
            except Exception as e:
                validation_results['calculators'][calc_name] = {
                    'status': 'invalid',
                    'error': str(e)
                }
        
        return validation_results
    
    # Legacy compatibility methods
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Legacy method for input validation."""
        return self.validate_input_data(data)
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Legacy method for data preprocessing."""
        return super().preprocess_data(data)


# For backward compatibility - ensure the same interface
EnhancedCorrelationFacade = EnhancedCorrelationCalculator