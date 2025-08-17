"""
ML Signal Adapter

Converts ML model predictions into trading signals compatible with the UnifiedSignalHandler.
Bridges the gap between ML predictions and the trading system.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

# Local imports
# Import ML components
from main.models.common import MLPrediction, OrderSide

# Import trading signal components
from main.trading_engine.signals.unified_signal import SignalPriority, SignalSource, UnifiedSignal

logger = logging.getLogger(__name__)


@dataclass
class MLSignalConfig:
    """Configuration for ML signal generation."""

    min_confidence: float = 0.6
    position_size_pct: float = 0.02  # 2% of portfolio
    use_dynamic_sizing: bool = True
    max_position_size_pct: float = 0.05  # 5% max
    signal_priority: SignalPriority = SignalPriority.MEDIUM
    stop_loss_pct: Optional[float] = 0.02  # 2% stop loss
    take_profit_pct: Optional[float] = 0.05  # 5% take profit


class MLSignalAdapter:
    """
    Adapter that converts ML predictions into trading signals.

    Features:
    - Confidence-based filtering
    - Dynamic position sizing
    - Risk management integration
    - Signal metadata enrichment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ML signal adapter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_config = MLSignalConfig()

        # Track signal history for analysis
        self.signal_history: List[Dict[str, Any]] = []
        self.max_history = 1000

        logger.info("ML Signal Adapter initialized")

    def convert_prediction_to_signal(
        self, prediction: MLPrediction, model_config: Optional[Dict[str, Any]] = None
    ) -> Optional[UnifiedSignal]:
        """
        Convert ML prediction to trading signal.

        Args:
            prediction: ML model prediction
            model_config: Model-specific configuration

        Returns:
            UnifiedSignal if criteria met, None otherwise
        """
        try:
            # Get configuration
            config = self._get_signal_config(model_config)

            # Check confidence threshold
            if prediction.confidence < config.min_confidence:
                logger.debug(
                    f"Prediction confidence {prediction.confidence:.2f} below threshold {config.min_confidence}"
                )
                return None

            # Determine signal direction
            signal_side = self._determine_signal_side(prediction)
            if not signal_side:
                logger.debug("No clear signal direction from prediction")
                return None

            # Calculate position size
            position_size = self._calculate_position_size(prediction, config)

            # Create signal metadata
            metadata = self._create_signal_metadata(prediction, config)

            # Create unified signal
            signal = UnifiedSignal(
                signal_id=f"ml_{prediction.model_id}_{uuid.uuid4().hex[:8]}",
                source=SignalSource.ML_MODEL,
                symbol=prediction.symbol,
                side=signal_side,
                strength=prediction.confidence,
                timestamp=prediction.timestamp or datetime.now(timezone.utc),
                priority=config.signal_priority,
                metadata=metadata,
            )

            # Add to history
            self._add_to_history(signal, prediction)

            logger.info(
                f"Created ML signal: {signal.symbol} {signal.side.value} "
                f"confidence={prediction.confidence:.2f} size={position_size:.2%}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error converting prediction to signal: {e}")
            return None

    def _get_signal_config(self, model_config: Optional[Dict[str, Any]]) -> MLSignalConfig:
        """Get signal configuration with model-specific overrides."""
        config = MLSignalConfig()

        # Apply global config
        if self.config:
            config.min_confidence = self.config.get("min_confidence", config.min_confidence)
            config.position_size_pct = self.config.get("position_size", config.position_size_pct)
            config.use_dynamic_sizing = self.config.get(
                "use_dynamic_sizing", config.use_dynamic_sizing
            )
            config.max_position_size_pct = self.config.get(
                "max_position_size", config.max_position_size_pct
            )
            config.stop_loss_pct = self.config.get("stop_loss_pct", config.stop_loss_pct)
            config.take_profit_pct = self.config.get("take_profit_pct", config.take_profit_pct)

        # Apply model-specific config
        if model_config:
            config.min_confidence = model_config.get("min_confidence", config.min_confidence)
            config.position_size_pct = model_config.get("position_size", config.position_size_pct)

        return config

    def _determine_signal_side(self, prediction: MLPrediction) -> Optional[OrderSide]:
        """Determine signal side from prediction."""
        # For regression models predicting returns
        if hasattr(prediction, "predicted_return") and prediction.predicted_return is not None:
            if prediction.predicted_return > 0.001:  # 0.1% threshold
                return OrderSide.BUY
            elif prediction.predicted_return < -0.001:
                return OrderSide.SELL

        # For classification models
        if hasattr(prediction, "predicted_class"):
            if prediction.predicted_class == 1:  # Bullish
                return OrderSide.BUY
            elif prediction.predicted_class == -1:  # Bearish
                return OrderSide.SELL

        # For directional predictions
        if hasattr(prediction, "direction"):
            if prediction.direction == "up":
                return OrderSide.BUY
            elif prediction.direction == "down":
                return OrderSide.SELL

        return None

    def _calculate_position_size(self, prediction: MLPrediction, config: MLSignalConfig) -> float:
        """Calculate position size based on prediction confidence."""
        base_size = config.position_size_pct

        if config.use_dynamic_sizing:
            # Scale position size with confidence
            # Map confidence [min_confidence, 1.0] to [0.5, 1.0] multiplier
            confidence_range = 1.0 - config.min_confidence
            confidence_normalized = (
                prediction.confidence - config.min_confidence
            ) / confidence_range
            size_multiplier = 0.5 + 0.5 * confidence_normalized

            position_size = base_size * size_multiplier

            # Apply max position size limit
            position_size = min(position_size, config.max_position_size_pct)
        else:
            position_size = base_size

        return position_size

    def _create_signal_metadata(
        self, prediction: MLPrediction, config: MLSignalConfig
    ) -> Dict[str, Any]:
        """Create comprehensive signal metadata."""
        metadata = {
            "ml_model_id": prediction.model_id,
            "ml_confidence": prediction.confidence,
            "ml_prediction_time": (
                prediction.timestamp.isoformat() if prediction.timestamp else None
            ),
            "position_size_pct": self._calculate_position_size(prediction, config),
        }

        # Add prediction details
        if hasattr(prediction, "predicted_return") and prediction.predicted_return is not None:
            metadata["predicted_return"] = prediction.predicted_return

        if hasattr(prediction, "prediction_horizon"):
            metadata["prediction_horizon"] = prediction.prediction_horizon

        # Add risk parameters
        if config.stop_loss_pct:
            metadata["stop_loss_pct"] = config.stop_loss_pct

        if config.take_profit_pct:
            metadata["take_profit_pct"] = config.take_profit_pct

        # Add feature importance if available
        if hasattr(prediction, "feature_importance") and prediction.feature_importance:
            # Get top 5 features
            sorted_features = sorted(
                prediction.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:5]
            metadata["top_features"] = dict(sorted_features)

        # Add any additional metadata from prediction
        if hasattr(prediction, "metadata") and prediction.metadata:
            metadata.update({f"ml_{k}": v for k, v in prediction.metadata.items()})

        return metadata

    def _add_to_history(self, signal: UnifiedSignal, prediction: MLPrediction):
        """Add signal to history for tracking."""
        history_entry = {
            "timestamp": datetime.now(timezone.utc),
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "side": signal.side.value,
            "confidence": prediction.confidence,
            "model_id": prediction.model_id,
        }

        self.signal_history.append(history_entry)

        # Maintain history size limit
        if len(self.signal_history) > self.max_history:
            self.signal_history = self.signal_history[-self.max_history :]

    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated signals."""
        if not self.signal_history:
            return {
                "total_signals": 0,
                "signals_by_side": {},
                "signals_by_symbol": {},
                "avg_confidence": 0.0,
            }

        total_signals = len(self.signal_history)

        # Count by side
        signals_by_side = {}
        for entry in self.signal_history:
            side = entry["side"]
            signals_by_side[side] = signals_by_side.get(side, 0) + 1

        # Count by symbol
        signals_by_symbol = {}
        for entry in self.signal_history:
            symbol = entry["symbol"]
            signals_by_symbol[symbol] = signals_by_symbol.get(symbol, 0) + 1

        # Average confidence
        avg_confidence = sum(entry["confidence"] for entry in self.signal_history) / total_signals

        return {
            "total_signals": total_signals,
            "signals_by_side": signals_by_side,
            "signals_by_symbol": signals_by_symbol,
            "avg_confidence": avg_confidence,
            "recent_signals": self.signal_history[-10:],  # Last 10 signals
        }

    def batch_convert_predictions(
        self,
        predictions: List[MLPrediction],
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[UnifiedSignal]:
        """
        Convert multiple predictions to signals.

        Args:
            predictions: List of ML predictions
            model_configs: Model-specific configurations by model_id

        Returns:
            List of valid signals
        """
        signals = []
        model_configs = model_configs or {}

        for prediction in predictions:
            model_config = model_configs.get(prediction.model_id)
            signal = self.convert_prediction_to_signal(prediction, model_config)

            if signal:
                signals.append(signal)

        logger.info(f"Converted {len(predictions)} predictions to {len(signals)} signals")
        return signals
