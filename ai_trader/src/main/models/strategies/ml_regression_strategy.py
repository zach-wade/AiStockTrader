"""
ML Regression Strategy for Return Prediction Models

This strategy is specifically designed for regression models that predict
future returns (like our trained XGBoost model predicting next_1d_return).
It includes enhanced position sizing, risk management, and signal generation
optimized for continuous return predictions.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy, Signal
from main.models.common import OrderSide, Order, OrderType
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine

logger = logging.getLogger(__name__)


class MLRegressionStrategy(BaseStrategy):
    """
    Advanced ML strategy optimized for regression models predicting returns.
    
    Features:
    - Dynamic position sizing based on predicted returns and volatility
    - Risk-adjusted signal generation with multiple thresholds
    - Position management with dynamic stops and targets
    - Real-time feature calculation and caching
    - Model confidence tracking and performance monitoring
    """
    
    def __init__(self, model_path: str, config: Dict[str, Any], feature_engine: Optional[UnifiedFeatureEngine] = None):
        """
        Initialize the ML regression strategy.
        
        Args:
            model_path: Path to the saved model directory
            config: Configuration dictionary
            feature_engine: Feature engine instance for real-time features
        """
        # Initialize base strategy
        super().__init__(config, feature_engine or self._create_dummy_engine())
        
        self.name = "ml_regression"
        self.model_path = Path(model_path)
        
        # Model artifacts
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_columns = []
        
        # Strategy configuration
        strategy_config = config.get('strategies', {}).get('ml_regression', {})
        
        # Return prediction thresholds (annualized)
        self.min_predicted_return = strategy_config.get('min_predicted_return', 0.001)  # 0.1% daily
        self.strong_signal_return = strategy_config.get('strong_signal_return', 0.003)  # 0.3% daily
        
        # Position sizing parameters
        self.base_position_size = strategy_config.get('base_position_size', 0.02)  # 2% of portfolio
        self.max_position_size = strategy_config.get('max_position_size', 0.05)  # 5% max
        self.use_kelly_sizing = strategy_config.get('use_kelly_sizing', True)
        self.kelly_fraction = strategy_config.get('kelly_fraction', 0.25)  # Conservative Kelly
        
        # Risk management
        self.max_positions = strategy_config.get('max_positions', 10)
        self.stop_loss_atr_mult = strategy_config.get('stop_loss_atr_mult', 2.0)
        self.profit_target_mult = strategy_config.get('profit_target_mult', 3.0)
        self.use_trailing_stop = strategy_config.get('use_trailing_stop', True)
        
        # Volatility adjustment
        self.volatility_lookback = strategy_config.get('volatility_lookback', 20)
        self.vol_target = strategy_config.get('vol_target', 0.15)  # 15% annual volatility target
        
        # Model confidence tracking
        self.recent_predictions = {}  # Track recent predictions for confidence
        self.prediction_window = strategy_config.get('prediction_window', 100)
        
        # Feature calculation cache
        self.feature_cache = {}
        self.cache_ttl = strategy_config.get('cache_ttl_seconds', 60)
        
        # Load model artifacts
        self._load_model()
        
        logger.info(f"MLRegressionStrategy initialized with model: {model_path}")
    
    def _create_dummy_engine(self):
        """Create a dummy feature engine for standalone usage."""
        from types import SimpleNamespace
        return SimpleNamespace(calculate_features=lambda *args, **kwargs: pd.DataFrame())
    
    def _load_model(self):
        """Load model artifacts from disk."""
        try:
            # Load model
            model_file = self.model_path / "model.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)
                logger.info(f"Loaded model from {model_file}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            # Load scaler
            scaler_file = self.model_path / "scaler.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                logger.info("Loaded feature scaler")
            
            # Load metadata
            metadata_file = self.model_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_columns = self.metadata.get('feature_columns', [])
                    logger.info(f"Loaded metadata with {len(self.feature_columns)} features")
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise
    
    def get_required_features(self) -> List[str]:
        """Return required feature sets based on model features."""
        # Analyze feature columns to determine requirements
        feature_sets = set(['technical'])  # Always need technical
        
        if self.feature_columns:
            # Check for various feature types
            feature_names = ' '.join(self.feature_columns).lower()
            
            if 'sentiment' in feature_names or 'news' in feature_names:
                feature_sets.add('sentiment')
            
            if 'volume' in feature_names or 'microstructure' in feature_names:
                feature_sets.add('microstructure')
            
            if 'regime' in feature_names or 'market' in feature_names:
                feature_sets.add('regime')
        
        return list(feature_sets)
    
    async def generate_signals(self, symbol: str, features: pd.DataFrame, 
                              current_position: Optional[Dict]) -> List[Signal]:
        """
        Generate trading signals using return predictions.
        
        Args:
            symbol: Trading symbol
            features: DataFrame with calculated features
            current_position: Current position information
            
        Returns:
            List of trading signals
        """
        try:
            if features.empty:
                logger.warning(f"No features available for {symbol}")
                return []
            
            # Calculate real-time features if needed
            features = await self._calculate_realtime_features(symbol, features)
            
            # Get prediction
            prediction_result = self._make_return_prediction(features)
            if prediction_result is None:
                return []
            
            predicted_return, prediction_std = prediction_result
            
            # Calculate volatility-adjusted position size
            position_size = self._calculate_position_size(
                predicted_return, prediction_std, features
            )
            
            # Generate signal based on prediction
            signal = self._generate_signal_from_prediction(
                symbol, predicted_return, prediction_std, 
                position_size, current_position
            )
            
            # Update prediction tracking
            self._update_prediction_tracking(symbol, predicted_return)
            
            return [signal] if signal else []
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}", exc_info=True)
            return []
    
    async def _calculate_realtime_features(self, symbol: str, features: pd.DataFrame) -> pd.DataFrame:
        """Calculate any missing real-time features."""
        cache_key = f"{symbol}_{datetime.now().minute}"
        
        # Check cache
        if cache_key in self.feature_cache:
            cached_time, cached_features = self.feature_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_features
        
        # Ensure we have all required features
        if self.feature_engine and hasattr(self.feature_engine, 'calculate_features'):
            try:
                # Calculate any missing features
                calculated_features = self.feature_engine.calculate_features(
                    data=features,
                    symbol=symbol,
                    calculators=['technical', 'microstructure', 'regime']
                )
                
                if not calculated_features.empty:
                    features = calculated_features
                    
            except Exception as e:
                logger.warning(f"Error calculating real-time features: {e}")
        
        # Cache the features
        self.feature_cache[cache_key] = (datetime.now(), features)
        
        return features
    
    def _make_return_prediction(self, features: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """
        Make return prediction with uncertainty estimation.
        
        Returns:
            Tuple of (predicted_return, prediction_std) or None
        """
        try:
            # Prepare features
            latest_features = self._prepare_model_features(features)
            if latest_features is None:
                return None
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(latest_features)
            else:
                features_scaled = latest_features.values
            
            # Make prediction
            predicted_return = self.model.predict(features_scaled)[0]
            
            # Estimate prediction uncertainty
            # For tree-based models, we can use the variance of predictions from trees
            prediction_std = self._estimate_prediction_uncertainty(
                self.model, features_scaled, predicted_return
            )
            
            logger.info(f"Predicted return: {predicted_return:.4f} ± {prediction_std:.4f}")
            
            return predicted_return, prediction_std
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def _prepare_model_features(self, features: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for model prediction."""
        try:
            # Get the latest row
            latest_row = features.iloc[-1:].copy()
            
            # Ensure all required features are present
            if self.feature_columns:
                missing_features = set(self.feature_columns) - set(latest_row.columns)
                if missing_features:
                    logger.warning(f"Missing {len(missing_features)} features, filling with zeros")
                    for feat in missing_features:
                        latest_row[feat] = 0
                
                # Select only the features used in training
                latest_row = latest_row[self.feature_columns]
            
            # Handle NaN values
            if latest_row.isna().any().any():
                logger.warning("Found NaN values in features, filling with 0")
                latest_row = latest_row.fillna(0)
            
            return latest_row
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _estimate_prediction_uncertainty(self, model, features_scaled, prediction):
        """Estimate uncertainty in prediction."""
        try:
            # For ensemble models like XGBoost, we can get individual tree predictions
            if hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'predict'):
                # Get predictions from individual trees
                booster = model.get_booster()
                # This is XGBoost specific - adjust for other models
                individual_preds = []
                for i in range(0, booster.num_boosted_rounds(), 10):  # Sample every 10th tree
                    pred = booster.predict(features_scaled, iteration_range=(0, i+1))
                    individual_preds.append(pred[0])
                
                if len(individual_preds) > 1:
                    return np.std(individual_preds)
            
            # Default: use a fraction of the absolute prediction as uncertainty
            return abs(prediction) * 0.3
            
        except Exception as e:
            logger.warning(f"Could not estimate prediction uncertainty: {e}")
            return abs(prediction) * 0.3
    
    def _calculate_position_size(self, predicted_return: float, 
                                prediction_std: float,
                                features: pd.DataFrame) -> float:
        """
        Calculate position size using Kelly Criterion and volatility adjustment.
        """
        # Get current volatility
        if 'volatility_20' in features.columns:
            current_vol = features.iloc[-1]['volatility_20']
        else:
            # Estimate from returns if available
            if 'returns' in features.columns:
                current_vol = features['returns'].tail(self.volatility_lookback).std()
            else:
                current_vol = 0.02  # Default 2% daily vol
        
        # Volatility adjustment
        vol_scalar = min(self.vol_target / (current_vol * np.sqrt(252)), 2.0) if current_vol > 0 else 1.0
        
        if self.use_kelly_sizing:
            # Kelly Criterion: f = (p*b - q) / b
            # Where p = probability of win, b = win/loss ratio, q = 1-p
            # For continuous returns: f = expected_return / variance
            
            if prediction_std > 0:
                kelly_size = (predicted_return / (prediction_std ** 2)) * self.kelly_fraction
                position_size = max(0, min(kelly_size, self.max_position_size))
            else:
                position_size = self.base_position_size
        else:
            # Simple scaling based on predicted return magnitude
            return_multiple = abs(predicted_return) / self.min_predicted_return
            position_size = min(
                self.base_position_size * return_multiple,
                self.max_position_size
            )
        
        # Apply volatility scaling
        position_size *= vol_scalar
        
        # Ensure within bounds
        position_size = max(0, min(position_size, self.max_position_size))
        
        return position_size
    
    def _generate_signal_from_prediction(self, symbol: str, 
                                       predicted_return: float,
                                       prediction_std: float,
                                       position_size: float,
                                       current_position: Optional[Dict]) -> Optional[Signal]:
        """Generate trading signal from return prediction."""
        # Calculate signal strength (z-score)
        if prediction_std > 0:
            z_score = predicted_return / prediction_std
        else:
            z_score = np.sign(predicted_return) * 2.0
        
        # Determine direction
        if abs(predicted_return) < self.min_predicted_return:
            direction = 'hold'
        elif predicted_return > 0:
            direction = 'buy'
        else:
            direction = 'sell'
        
        # Check if we should close existing position
        if current_position and current_position.get('quantity', 0) != 0:
            current_side = 'buy' if current_position['quantity'] > 0 else 'sell'
            
            # Exit if prediction opposes current position
            if (current_side == 'buy' and predicted_return < -self.min_predicted_return) or \
               (current_side == 'sell' and predicted_return > self.min_predicted_return):
                # Strong exit signal
                position_size = min(position_size * 1.5, self.max_position_size)
                logger.info(f"Exit signal for {symbol}: reversing from {current_side} to {direction}")
        
        # Calculate confidence based on z-score and prediction magnitude
        confidence = min(abs(z_score) / 3.0, 1.0)  # Normalize z-score to [0, 1]
        
        # Boost confidence for strong signals
        if abs(predicted_return) > self.strong_signal_return:
            confidence = min(confidence * 1.2, 1.0)
        
        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            size=position_size if direction != 'hold' else 0.0,
            metadata={
                'predicted_return': predicted_return,
                'prediction_std': prediction_std,
                'z_score': z_score,
                'volatility_adjusted': True,
                'model_path': str(self.model_path),
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _update_prediction_tracking(self, symbol: str, predicted_return: float):
        """Track recent predictions for confidence monitoring."""
        if symbol not in self.recent_predictions:
            self.recent_predictions[symbol] = []
        
        self.recent_predictions[symbol].append({
            'timestamp': datetime.now(),
            'prediction': predicted_return
        })
        
        # Keep only recent predictions
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.recent_predictions[symbol] = [
            p for p in self.recent_predictions[symbol] 
            if p['timestamp'] > cutoff_time
        ][-self.prediction_window:]
    
    def get_model_confidence(self, symbol: str) -> float:
        """Calculate model confidence based on recent prediction consistency."""
        if symbol not in self.recent_predictions or len(self.recent_predictions[symbol]) < 10:
            return 0.5  # Default confidence
        
        recent = self.recent_predictions[symbol]
        predictions = [p['prediction'] for p in recent]
        
        # Calculate consistency metrics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Lower confidence if predictions are inconsistent
        if std_pred > 0:
            consistency = 1.0 / (1.0 + std_pred / abs(mean_pred))
        else:
            consistency = 1.0
        
        return consistency
    
    def get_position_params(self, symbol: str, signal: Signal) -> Dict[str, Any]:
        """Get position management parameters for a signal."""
        # Get current price volatility for stop/target calculation
        atr_pct = 0.02  # Default 2% ATR
        
        # Try to get from features if available
        if hasattr(self, '_last_features') and self._last_features is not None:
            if 'atr_pct' in self._last_features.columns:
                atr_pct = self._last_features.iloc[-1]['atr_pct']
        
        stop_distance = atr_pct * self.stop_loss_atr_mult
        target_distance = atr_pct * self.profit_target_mult
        
        return {
            'stop_loss_pct': stop_distance,
            'profit_target_pct': target_distance,
            'trailing_stop': self.use_trailing_stop,
            'trailing_stop_pct': stop_distance * 0.5,  # Tighten as position moves favorable
            'time_stop_days': 5,  # Exit if position doesn't perform in 5 days
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model and strategy."""
        info = {
            'strategy_name': self.name,
            'model_path': str(self.model_path),
            'model_type': 'regression',
            'features_count': len(self.feature_columns),
            'has_scaler': self.scaler is not None,
            'position_sizing': {
                'method': 'kelly' if self.use_kelly_sizing else 'fixed',
                'base_size': self.base_position_size,
                'max_size': self.max_position_size,
                'volatility_target': self.vol_target
            },
            'thresholds': {
                'min_return': self.min_predicted_return,
                'strong_signal': self.strong_signal_return
            }
        }
        
        if self.metadata:
            info.update({
                'model_details': {
                    'type': self.metadata.get('model_type', 'unknown'),
                    'training_symbols': self.metadata.get('symbols', []),
                    'training_metrics': self.metadata.get('metrics', {}),
                    'training_date': self.metadata.get('training_timestamp', 'unknown'),
                    'target_variable': self.metadata.get('target_column', 'unknown')
                }
            })
        
        return info