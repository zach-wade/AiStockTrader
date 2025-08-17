"""
ML Model Strategy for backtesting trained machine learning models.

This strategy wraps a trained ML model and converts its predictions
into trading signals for use with the backtesting engine.
"""

# Standard library imports
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import joblib
import numpy as np
import pandas as pd

# Local imports
from main.models.common import Order, OrderSide, OrderType

from .base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MLModelStrategy(BaseStrategy):
    """
    Strategy that uses a trained ML model to generate trading signals.

    This strategy loads a saved model and uses it to make predictions
    on market data, converting those predictions into buy/sell signals.
    """

    def __init__(self, model_path: str, config: Dict[str, Any], feature_engine=None):
        """
        Initialize the ML model strategy.

        Args:
            model_path: Path to the saved model directory
            config: Configuration dictionary
            feature_engine: Feature engine instance (optional)
        """
        # For ML models, we don't need the feature engine since features come from market data
        # Create a dummy feature engine to satisfy base class
        if feature_engine is None:
            # Standard library imports
            from types import SimpleNamespace

            feature_engine = SimpleNamespace(calculate_features=lambda: None)

        super().__init__(config, feature_engine)
        self.name = "ml_model"
        self.model_path = Path(model_path)

        # Model artifacts
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_columns = []

        # Trading parameters
        self.prediction_threshold = config.get("ml_strategy", {}).get("prediction_threshold", 0.5)
        self.confidence_scaling = config.get("ml_strategy", {}).get("confidence_scaling", True)
        self.min_confidence = config.get("ml_strategy", {}).get(
            "min_confidence", 0.3
        )  # Lower for testing
        self.max_position_size = config.get("ml_strategy", {}).get(
            "max_position_size", 100
        )  # 100 shares for testing

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load model artifacts from disk."""
        try:
            # Load model
            model_file = self.model_path / "model.pkl"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            self.model = joblib.load(model_file)
            logger.info(f"Loaded model from {model_file}")

            # Load scaler if exists
            scaler_file = self.model_path / "scaler.pkl"
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
                logger.info(f"Loaded scaler from {scaler_file}")

            # Load metadata
            metadata_file = self.model_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
                self.feature_columns = self.metadata.get("feature_columns", [])
                logger.info(f"Loaded metadata with {len(self.feature_columns)} features")
            else:
                logger.warning("No metadata file found, feature alignment may be incorrect")

            # Log model info
            model_type = self.metadata.get("model_type", "unknown") if self.metadata else "unknown"
            logger.info(f"ML Model Strategy initialized with {model_type} model")

        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

    def get_required_feature_sets(self) -> List[str]:
        """Return required feature sets based on model features."""
        # Analyze feature columns to determine which feature sets are needed
        feature_sets = set(["technical"])  # Always need basic technical features

        if self.feature_columns:
            # Check for sentiment features
            if any("sentiment" in col.lower() for col in self.feature_columns):
                feature_sets.add("sentiment")

            # Check for fundamental features
            if any(
                "fundamental" in col.lower() or "financial" in col.lower()
                for col in self.feature_columns
            ):
                feature_sets.add("fundamental")

        return list(feature_sets)

    async def generate_signals(
        self, symbol: str, features: pd.DataFrame, current_position: Optional[Dict]
    ) -> List[Signal]:
        """
        Generate trading signals using the ML model.

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

            # Get the latest features
            latest_features = self._prepare_model_features(features)
            if latest_features is None:
                return []

            # Make prediction
            prediction = self._make_prediction(latest_features)
            if prediction is None:
                return []

            # Convert prediction to signal
            signal = self._prediction_to_signal(symbol, prediction, current_position)

            return [signal] if signal else []

        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []

    def _prepare_model_features(self, features: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for model prediction."""
        try:
            # Get the latest row
            latest_row = features.iloc[-1:].copy()

            logger.debug(f"Preparing features. Available columns: {features.columns.tolist()}")
            logger.debug(f"Expected features: {self.feature_columns[:5]}...")  # Show first 5

            # If we have feature columns from metadata, use them
            if self.feature_columns:
                # Ensure all required features are present
                missing_features = set(self.feature_columns) - set(latest_row.columns)
                if missing_features:
                    logger.warning(f"Missing features: {missing_features}")
                    # Add missing features with zeros
                    for feat in missing_features:
                        latest_row[feat] = 0

                # Select only the features used in training
                latest_row = latest_row[self.feature_columns]

            # Remove any non-numeric columns
            numeric_cols = latest_row.select_dtypes(include=[np.number]).columns
            latest_row = latest_row[numeric_cols]

            # Handle any remaining NaN values
            if latest_row.isna().any().any():
                logger.warning("Found NaN values in features, filling with 0")
                latest_row = latest_row.fillna(0)

            return latest_row

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def _make_prediction(self, features: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Make prediction using the model."""
        try:
            # Scale features if scaler is available
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features.values

            # Make prediction
            if hasattr(self.model, "predict_proba"):
                # Classification model
                probabilities = self.model.predict_proba(features_scaled)[0]
                prediction = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                confidence = abs(prediction - 0.5) * 2  # Convert to confidence score
                logger.info(f"Classification prediction: {prediction}, confidence: {confidence}")
            else:
                # Regression model
                prediction = self.model.predict(features_scaled)[0]
                # Normalize prediction to [0, 1] range if needed
                if prediction < 0:
                    prediction = 0
                elif prediction > 1:
                    prediction = 1
                confidence = abs(prediction - 0.5) * 2
                logger.info(f"Regression prediction: {prediction}, confidence: {confidence}")

            return {"prediction": prediction, "confidence": confidence}

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

    def _prediction_to_signal(
        self, symbol: str, prediction: Dict[str, float], current_position: Optional[Dict]
    ) -> Optional[Signal]:
        """Convert model prediction to trading signal."""
        pred_value = prediction["prediction"]
        confidence = prediction["confidence"]

        # Check minimum confidence
        if confidence < self.min_confidence:
            return Signal(
                symbol=symbol,
                direction="hold",
                confidence=confidence,
                metadata={"prediction": pred_value, "reason": "low_confidence"},
            )

        # Determine signal direction
        if pred_value > self.prediction_threshold:
            direction = "buy"
        elif pred_value < (1 - self.prediction_threshold):
            direction = "sell"
        else:
            direction = "hold"

        # Check if we should close position
        if current_position and current_position.get("quantity", 0) != 0:
            current_side = "buy" if current_position["quantity"] > 0 else "sell"
            if direction != current_side and direction != "hold":
                # Signal to close and reverse position
                confidence = min(confidence * 1.2, 1.0)  # Boost confidence for reversals

        # Calculate position size
        if direction != "hold":
            if self.confidence_scaling:
                size = self.max_position_size * confidence
            else:
                size = self.max_position_size
        else:
            size = 0.0

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            size=size,
            metadata={
                "prediction": pred_value,
                "model_path": str(self.model_path),
                "timestamp": datetime.now().isoformat(),
            },
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_path": str(self.model_path),
            "model_type": "unknown",
            "features_count": len(self.feature_columns),
            "has_scaler": self.scaler is not None,
            "has_metadata": self.metadata is not None,
        }

        if self.metadata:
            info.update(
                {
                    "model_type": self.metadata.get("model_type", "unknown"),
                    "training_symbols": self.metadata.get("symbols", []),
                    "training_metrics": self.metadata.get("metrics", {}),
                    "training_date": self.metadata.get("training_timestamp", "unknown"),
                }
            )

        return info

    async def on_market_data(self, event) -> List[Order]:
        """
        Handle market data events from backtest engine.

        Args:
            event: MarketEvent with market data

        Returns:
            List of orders to execute
        """
        # Extract data from event
        symbol = event.data["symbol"]

        # For now, we'll use a simple heuristic since we don't have
        # the full feature engineering pipeline available during backtesting
        # This is a placeholder implementation - in production, you'd want
        # to maintain a rolling window of data and compute technical indicators

        # Create features matching what the model expects
        # These are the features from the model metadata:
        # ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions',
        #  'otc', 'returns', 'sma_20', 'volatility_20d', 'day_of_week',
        #  'month', 'trade_count', 'Dividends', 'Stock Splits', 'Adj Close']

        close_price = event.data["close"]
        features_df = pd.DataFrame(
            [
                {
                    "open": event.data["open"],
                    "high": event.data["high"],
                    "low": event.data["low"],
                    "close": close_price,
                    "volume": event.data["volume"],
                    "vwap": close_price,  # Simplified - use close as VWAP
                    "transactions": 1000,  # Placeholder
                    "otc": 0,  # Not OTC
                    "returns": 0.001,  # Placeholder - would compute from historical
                    "sma_20": close_price,  # Placeholder - would compute from historical
                    "volatility_20d": 0.02,  # Placeholder
                    "day_of_week": event.timestamp.weekday(),
                    "month": event.timestamp.month,
                    "trade_count": 1000,  # Placeholder
                    "Dividends": 0,  # Placeholder
                    "Stock Splits": 0,  # Placeholder
                    "Adj Close": close_price,  # Use close as adjusted close
                    "timestamp": event.timestamp,
                }
            ]
        )

        # Set timestamp as index
        features_df.set_index("timestamp", inplace=True)

        # Log to debug
        logger.debug(f"Processing market data for {symbol} at {event.timestamp}")

        # Generate signals
        signals = await self.generate_signals(symbol, features_df, None)

        if signals:
            logger.info(f"Generated {len(signals)} signals for {symbol}")

        # Convert signals to orders
        orders = []
        for signal in signals:
            if signal.action in ["buy", "sell"]:
                order = Order(
                    symbol=signal.symbol,
                    quantity=signal.quantity,
                    side=OrderSide.BUY if signal.action == "buy" else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    timestamp=event.timestamp,
                )
                orders.append(order)
                logger.info(
                    f"Created {signal.action} order for {signal.quantity} shares of {symbol}"
                )

        return orders
