"""
Advanced Statistical Calculator Facade

This facade provides 100% backward compatibility with the original monolithic
AdvancedStatisticalCalculator by coordinating all specialized statistical calculators.

The facade maintains the same interface and behavior while leveraging the new
modular architecture underneath for better maintainability and performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

from ..base_calculator import BaseFeatureCalculator
from .statistical_config import StatisticalConfig
from .moments_calculator import MomentsCalculator
from .entropy_calculator import EntropyCalculator
from .fractal_calculator import FractalCalculator
from .nonlinear_calculator import NonlinearCalculator
from .timeseries_calculator import TimeseriesCalculator
from .multivariate_calculator import MultivariateCalculator

logger = logging.getLogger(__name__)


class AdvancedStatisticalCalculator(BaseFeatureCalculator):
    """
    Facade for advanced statistical features calculation.
    
    This class maintains backward compatibility with the original monolithic
    AdvancedStatisticalCalculator while internally using specialized calculators
    for better code organization and maintainability.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the advanced statistical calculator facade."""
        super().__init__(config)
        
        # Create statistical configuration
        self.stat_config = StatisticalConfig(
            **config.get('statistical', {}) if config else {}
        )
        
        # Initialize all specialized calculators
        self.calculators = self._initialize_calculators(config)
        
        # Performance tracking
        self.performance_stats = {
            'total_features': 0,
            'calculator_times': {},
            'last_calculation_time': None
        }
        
        logger.debug(f"Initialized AdvancedStatisticalCalculator with {len(self.calculators)} specialized calculators")
    
    def _initialize_calculators(self, config: Optional[Dict] = None) -> Dict[str, object]:
        """Initialize all specialized statistical calculators."""
        calculators = {}
        
        try:
            calculators['moments'] = MomentsCalculator(config)
            calculators['entropy'] = EntropyCalculator(config)
            calculators['fractal'] = FractalCalculator(config)
            calculators['nonlinear'] = NonlinearCalculator(config)
            calculators['timeseries'] = TimeseriesCalculator(config)
            calculators['multivariate'] = MultivariateCalculator(config)
            
            logger.debug("Successfully initialized all statistical calculators")
            
        except Exception as e:
            logger.error(f"Error initializing statistical calculators: {e}")
            raise
        
        return calculators
    
    def get_feature_names(self) -> List[str]:
        """
        Get all feature names from all calculators.
        
        Returns:
            List of all statistical feature names
        """
        all_features = []
        
        for calc_name, calculator in self.calculators.items():
            try:
                features = calculator.get_feature_names()
                all_features.extend(features)
                logger.debug(f"{calc_name} calculator provides {len(features)} features")
            except Exception as e:
                logger.warning(f"Error getting feature names from {calc_name} calculator: {e}")
        
        # Remove duplicates while preserving order
        unique_features = []
        seen = set()
        for feature in all_features:
            if feature not in seen:
                unique_features.append(feature)
                seen.add(feature)
        
        self.performance_stats['total_features'] = len(unique_features)
        logger.info(f"Total statistical features available: {len(unique_features)}")
        
        return unique_features
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced statistical features using all specialized calculators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all statistical features combined
        """
        import time
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self.validate_inputs(data):
                logger.error("Invalid input data for statistical calculation")
                return self._create_empty_features(data.index if not data.empty else pd.DatetimeIndex([]))
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            if processed_data.empty:
                logger.error("Data preprocessing failed for statistical analysis")
                return self._create_empty_features(data.index)
            
            # Initialize combined features DataFrame
            combined_features = pd.DataFrame(index=processed_data.index)
            
            # Calculate features from each specialized calculator
            for calc_name, calculator in self.calculators.items():
                calc_start_time = time.time()
                
                try:
                    logger.debug(f"Calculating {calc_name} features...")
                    features = calculator.calculate(processed_data)
                    
                    if features is not None and not features.empty:
                        # Merge features (avoiding column name conflicts)
                        for column in features.columns:
                            if column not in combined_features.columns:
                                combined_features[column] = features[column]
                            else:
                                logger.warning(f"Duplicate feature name '{column}' from {calc_name} calculator")
                        
                        calc_time = time.time() - calc_start_time
                        self.performance_stats['calculator_times'][calc_name] = calc_time
                        
                        logger.debug(f"{calc_name} calculator: {len(features.columns)} features in {calc_time:.3f}s")
                    else:
                        logger.warning(f"{calc_name} calculator returned empty features")
                        
                except Exception as e:
                    logger.error(f"Error in {calc_name} calculator: {e}")
                    # Continue with other calculators even if one fails
                    continue
            
            # Apply final processing and validation
            combined_features = self._postprocess_features(combined_features)
            
            # Update performance statistics
            total_time = time.time() - start_time
            self.performance_stats['last_calculation_time'] = total_time
            
            logger.info(f"Statistical calculation completed: {len(combined_features.columns)} features in {total_time:.3f}s")
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Error in advanced statistical calculation: {e}")
            return self._create_empty_features(data.index)
    
    def _postprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply final processing to the combined features."""
        try:
            # Handle infinite values
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Apply feature validation and clipping where appropriate
            features = self._validate_and_clip_features(features)
            
            # Ensure proper data types
            for column in features.columns:
                if features[column].dtype == 'object':
                    try:
                        features[column] = pd.to_numeric(features[column], errors='coerce')
                    except (ValueError, TypeError):
                        pass  # Keep original if conversion fails
            
            logger.debug(f"Post-processed {len(features.columns)} statistical features")
            
            return features
            
        except Exception as e:
            logger.error(f"Error in feature post-processing: {e}")
            return features
    
    def _validate_and_clip_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Validate and clip features to reasonable ranges."""
        try:
            # Define reasonable ranges for different types of features
            feature_ranges = {
                # Entropy features should be non-negative
                'entropy': (0, 20),
                'shannon': (0, 20),
                
                # Correlation features should be [-1, 1]
                'correlation': (-1, 1),
                'autocorr': (-1, 1),
                
                # Probability features should be [0, 1]
                'prob': (0, 1),
                'pvalue': (0, 1),
                'explained_var': (0, 1),
                
                # Hurst exponent should be [0, 2]
                'hurst': (0, 2),
                
                # Fractal dimension should be [1, 3]
                'fractal_dimension': (1, 3),
                
                # Statistical moments (clip extreme values)
                'skewness': (-10, 10),
                'kurtosis': (0, 50),
                'moment': (-100, 100)
            }
            
            for column in features.columns:
                column_lower = column.lower()
                
                # Find appropriate range for this feature
                for pattern, (min_val, max_val) in feature_ranges.items():
                    if pattern in column_lower:
                        features[column] = features[column].clip(lower=min_val, upper=max_val)
                        break
            
            return features
            
        except Exception as e:
            logger.warning(f"Error in feature validation: {e}")
            return features
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from the last calculation."""
        return self.performance_stats.copy()
    
    def get_calculator_info(self) -> Dict[str, Dict]:
        """Get information about each specialized calculator."""
        info = {}
        
        for calc_name, calculator in self.calculators.items():
            try:
                feature_names = calculator.get_feature_names()
                info[calc_name] = {
                    'feature_count': len(feature_names),
                    'feature_names': feature_names,
                    'class_name': calculator.__class__.__name__
                }
            except Exception as e:
                info[calc_name] = {
                    'error': str(e),
                    'class_name': calculator.__class__.__name__ if calculator else 'None'
                }
        
        return info
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for statistical analysis."""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate input data for statistical calculations."""
        try:
            # Check required columns
            required_cols = self.get_required_columns()
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for sufficient data
            if len(data) < 10:
                logger.error("Insufficient data for statistical analysis (need at least 10 rows)")
                return False
            
            # Check for valid numeric data
            for col in required_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    logger.error(f"Column '{col}' must be numeric")
                    return False
                
                if data[col].isna().all():
                    logger.error(f"Column '{col}' contains all NaN values")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating input data: {e}")
            return False
    
    def _create_empty_features(self, index: pd.Index) -> pd.DataFrame:
        """Create empty features DataFrame with all expected columns."""
        try:
            # Get all feature names
            all_feature_names = self.get_feature_names()
            
            # Create empty DataFrame with NaN values
            empty_features = pd.DataFrame(
                index=index,
                columns=all_feature_names,
                dtype=float
            )
            empty_features[:] = np.nan
            
            logger.debug(f"Created empty features DataFrame with {len(all_feature_names)} columns")
            
            return empty_features
            
        except Exception as e:
            logger.error(f"Error creating empty features: {e}")
            # Fallback to basic empty DataFrame
            return pd.DataFrame(index=index)


# Maintain backward compatibility by providing the same class name
# that external code expects to import
__all__ = ['AdvancedStatisticalCalculator']