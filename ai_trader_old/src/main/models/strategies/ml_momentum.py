"""
Machine Learning Momentum Strategy

Uses trained ML models to predict momentum opportunities by analyzing
technical indicators, market microstructure, and cross-sectional features.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import joblib
import numpy as np
import pandas as pd

# Local imports
from main.feature_pipeline.unified_feature_engine import UnifiedFeatureEngine
from main.models.inference.model_registry import ModelRegistry

from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MLMomentumStrategy(BaseStrategy):
    """
    ML-based momentum strategy that uses trained models to predict
    short-term price movements based on momentum features.
    """

    def __init__(self, config: Dict[str, Any], feature_engine: UnifiedFeatureEngine):
        super().__init__(config, feature_engine)
        self.name = "ml_momentum"

        # Load strategy configuration
        strategy_conf = self.config.get("strategies", {}).get(self.name, {})

        # Model settings
        self.model_name = strategy_conf.get("model_name", "momentum_classifier")
        self.model_version = strategy_conf.get("model_version", "latest")
        self.prediction_threshold = strategy_conf.get("prediction_threshold", 0.6)
        self.confidence_scaling = strategy_conf.get("confidence_scaling", 1.2)

        # Feature requirements
        self.lookback_periods = strategy_conf.get("lookback_periods", [5, 10, 20])
        self.momentum_features = strategy_conf.get(
            "momentum_features", ["price_momentum", "volume_momentum", "rsi_divergence"]
        )

        # Risk parameters
        self.max_positions = strategy_conf.get("max_positions", 5)
        self.position_timeout = strategy_conf.get("position_timeout_days", 5)

        # Initialize model registry
        self.model_registry = ModelRegistry(config)
        self.model = None
        self.feature_columns = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the trained ML model."""
        try:
            # Try to load from model registry
            model_info = self.model_registry.get_model(self.model_name, version=self.model_version)

            if model_info:
                self.model = model_info["model"]
                self.feature_columns = model_info.get("feature_columns", [])
                logger.info(f"Loaded model {self.model_name} version {self.model_version}")
            else:
                # Fallback to loading from file
                model_path = (
                    Path(self.config.get("models", {}).get("path", "models"))
                    / f"{self.model_name}.pkl"
                )
                if model_path.exists():
                    self.model = joblib.load(model_path)
                    logger.info(f"Loaded model from file: {model_path}")
                else:
                    logger.warning(f"No model found for {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.model = None

    def get_required_feature_sets(self) -> List[str]:
        """Specify required feature sets."""
        return ["technical", "microstructure", "cross_sectional", "market_regime"]

    async def generate_signals(
        self, symbol: str, features: pd.DataFrame, current_position: Optional[Dict]
    ) -> List[Signal]:
        """
        Generate trading signals using ML predictions.
        """
        if features.empty or self.model is None:
            return []

        try:
            # Prepare features for prediction
            ml_features = self._prepare_ml_features(features)
            if ml_features is None:
                return []

            # Make prediction
            prediction_proba = self._make_prediction(ml_features)
            if prediction_proba is None:
                return []

            # Generate signal based on prediction
            signal = self._generate_signal_from_prediction(
                symbol, prediction_proba, features.iloc[-1], current_position
            )

            return [signal] if signal else []

        except Exception as e:
            logger.error(f"Error generating ML momentum signal for {symbol}: {e}")
            return []

    def _prepare_ml_features(self, features: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for ML model."""
        try:
            latest_features = features.iloc[-1:].copy()

            # Calculate momentum-specific features
            for period in self.lookback_periods:
                if len(features) > period:
                    # Price momentum
                    latest_features[f"momentum_{period}"] = (
                        features["close"].iloc[-1] / features["close"].iloc[-period - 1] - 1
                    )

                    # Volume momentum
                    latest_features[f"volume_momentum_{period}"] = (
                        features["volume"].iloc[-period:].mean()
                        / features["volume"].iloc[-period * 2 : -period].mean()
                    )

                    # RSI divergence
                    if f"rsi_{period}" in features.columns:
                        price_change = (
                            features["close"].iloc[-1] / features["close"].iloc[-period - 1] - 1
                        )
                        rsi_change = (
                            features[f"rsi_{period}"].iloc[-1]
                            - features[f"rsi_{period}"].iloc[-period - 1]
                        )
                        latest_features[f"rsi_divergence_{period}"] = price_change - (
                            rsi_change / 100
                        )

            # Add microstructure features
            if "bid_ask_spread" in features.columns:
                latest_features["spread_zscore"] = (
                    features["bid_ask_spread"].iloc[-1] - features["bid_ask_spread"].mean()
                ) / features["bid_ask_spread"].std()

            # Add regime features
            if "market_regime" in features.columns:
                latest_features["regime_bullish"] = int(
                    features["market_regime"].iloc[-1] == "bullish"
                )
                latest_features["regime_bearish"] = int(
                    features["market_regime"].iloc[-1] == "bearish"
                )

            # Select features that match model training
            if self.feature_columns:
                # Ensure all required features exist
                for col in self.feature_columns:
                    if col not in latest_features.columns:
                        latest_features[col] = 0  # Default value

                return latest_features[self.feature_columns]
            else:
                # Use all numeric features
                numeric_features = latest_features.select_dtypes(include=[np.number])
                return numeric_features

        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None

    def _make_prediction(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """Make prediction using the ML model."""
        try:
            # Handle different model types
            if hasattr(self.model, "predict_proba"):
                # Classification model
                predictions = self.model.predict_proba(features)
                # Return probability of positive class (momentum opportunity)
                return predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]
            elif hasattr(self.model, "predict"):
                # Regression model - convert to probability
                predictions = self.model.predict(features)
                # Sigmoid transformation to get probability-like values
                return 1 / (1 + np.exp(-predictions))
            else:
                logger.error("Model does not have predict or predict_proba method")
                return None

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def _generate_signal_from_prediction(
        self,
        symbol: str,
        prediction_proba: np.ndarray,
        latest_features: pd.Series,
        current_position: Optional[Dict],
    ) -> Optional[Signal]:
        """Generate trading signal from ML prediction."""
        prob = prediction_proba[0] if isinstance(prediction_proba, np.ndarray) else prediction_proba

        # Determine signal direction based on prediction
        if prob > self.prediction_threshold:
            # Strong momentum signal
            direction = "buy"
            base_confidence = prob
        elif prob < (1 - self.prediction_threshold):
            # Strong reversal signal (anti-momentum)
            direction = "sell"
            base_confidence = 1 - prob
        else:
            # No clear signal
            return None

        # Check if we should act on the signal
        if current_position:
            # Already have a position
            if current_position.get("direction") == "long" and direction == "buy":
                return None  # Already long
            elif current_position.get("direction") == "short" and direction == "sell":
                return None  # Already short

        # Scale confidence based on additional factors
        confidence = self._calculate_signal_confidence(base_confidence, latest_features, direction)

        # Add metadata for execution and risk management
        metadata = {
            "strategy": self.name,
            "model_prediction": float(prob),
            "momentum_score": float(latest_features.get(f"momentum_{self.lookback_periods[0]}", 0)),
            "volume_ratio": float(latest_features.get("volume_ratio", 1.0)),
            "entry_reason": self._get_entry_reason(prob, latest_features),
            "suggested_holding_period": self.position_timeout,
            "model_version": self.model_version,
        }

        return Signal(symbol=symbol, direction=direction, confidence=confidence, metadata=metadata)

    def _calculate_signal_confidence(
        self, base_confidence: float, features: pd.Series, direction: str
    ) -> float:
        """Calculate adjusted signal confidence."""
        confidence = base_confidence

        # Adjust for market regime
        if "market_regime" in features:
            regime = features["market_regime"]
            if regime == "bullish" and direction == "buy":
                confidence *= 1.1
            elif regime == "bearish" and direction == "sell":
                confidence *= 1.1
            elif regime == "choppy":
                confidence *= 0.9

        # Adjust for volatility
        if "volatility_zscore" in features:
            vol_z = features["volatility_zscore"]
            if abs(vol_z) > 2:
                # High volatility - reduce confidence
                confidence *= 0.85

        # Adjust for volume
        if "volume_ratio" in features:
            vol_ratio = features["volume_ratio"]
            if vol_ratio > 2:
                # High volume - increase confidence
                confidence *= 1.05
            elif vol_ratio < 0.5:
                # Low volume - decrease confidence
                confidence *= 0.9

        # Apply scaling factor and ensure bounds
        confidence = confidence * self.confidence_scaling
        return max(0.0, min(1.0, confidence))

    def _get_entry_reason(self, prediction: float, features: pd.Series) -> str:
        """Generate human-readable entry reason."""
        reasons = []

        if prediction > self.prediction_threshold:
            reasons.append(f"Strong momentum signal ({prediction:.2%} probability)")
        else:
            reasons.append(f"Reversal signal ({(1-prediction):.2%} probability)")

        # Add supporting factors
        if "momentum_5" in features and features["momentum_5"] > 0.02:
            reasons.append(f"5-day momentum: {features['momentum_5']:.1%}")

        if "volume_ratio" in features and features["volume_ratio"] > 1.5:
            reasons.append(f"Volume surge: {features['volume_ratio']:.1f}x average")

        if "rsi_14" in features:
            rsi = features["rsi_14"]
            if rsi > 70:
                reasons.append("RSI overbought")
            elif rsi < 30:
                reasons.append("RSI oversold")

        return "; ".join(reasons)

    def get_position_size(self, signal: Signal, features: pd.DataFrame) -> float:
        """Calculate position size for the signal."""
        # Base size on confidence
        base_size = signal.confidence * 0.1  # Max 10% per position

        # Adjust for number of existing positions
        # This would need portfolio state in production
        max_size_per_position = 1.0 / self.max_positions

        return min(base_size, max_size_per_position)
