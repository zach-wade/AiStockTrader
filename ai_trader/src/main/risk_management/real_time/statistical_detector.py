"""
Statistical anomaly detection for real-time market monitoring.

This module provides statistical methods for detecting anomalies in price,
volume, and volatility using z-scores, isolation forests, and other techniques.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import IsolationForest
from collections import deque
import warnings

from .anomaly_types import AnomalyType, AnomalySeverity
from .anomaly_models import AnomalyEvent, AnomalyDetectionConfig
from main.utils.core import ErrorHandlingMixin
from main.utils.monitoring import timer

warnings.filterwarnings('ignore', category=RuntimeWarning)
logger = logging.getLogger(__name__)


@dataclass
class StatisticalConfig:
    """Configuration for statistical anomaly detection."""
    z_score_threshold: float = 3.0
    min_samples: int = 30
    outlier_fraction: float = 0.05
    volatility_window: int = 20
    volume_window: int = 50
    price_window: int = 100
    use_rolling_stats: bool = True
    use_robust_estimators: bool = True
    enable_seasonality_adjustment: bool = True


class StatisticalAnomalyDetector(ErrorHandlingMixin):
    """
    Statistical anomaly detector using various statistical methods
    to identify outliers in market data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize statistical anomaly detector."""
        super().__init__()
        self.config = StatisticalConfig(**(config or {}))
        
        # Data buffers for each symbol
        self._price_buffers: Dict[str, deque] = {}
        self._volume_buffers: Dict[str, deque] = {}
        self._return_buffers: Dict[str, deque] = {}
        
        # Statistical models
        self._isolation_forests: Dict[str, IsolationForest] = {}
        
        # Cache for computed statistics
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        
        # Detection counters
        self._detection_count = 0
        self._anomaly_count = 0
        
        logger.info("Statistical anomaly detector initialized")
    
    @timer
    def detect_price_anomalies(self,
                             price_history: List[float],
                             current_price: float,
                             symbol: str) -> List[AnomalyEvent]:
        """Detect anomalies in price movements."""
        with self._handle_error("detecting price anomalies"):
            anomalies = []
            
            if len(price_history) < self.config.min_samples:
                return anomalies
            
            # Calculate returns
            prices = np.array(price_history + [current_price])
            returns = np.diff(np.log(prices))
            current_return = returns[-1]
            historical_returns = returns[:-1]
            
            # Z-score detection
            z_score_anomaly = self._detect_zscore_anomaly(
                historical_returns,
                current_return,
                "price_return"
            )
            
            if z_score_anomaly:
                anomaly_type = (AnomalyType.PRICE_SPIKE if current_return > 0 
                              else AnomalyType.PRICE_CRASH)
                
                anomaly = self._create_anomaly_event(
                    anomaly_type=anomaly_type,
                    symbol=symbol,
                    detected_value=current_price,
                    expected_value=price_history[-1],
                    deviation=z_score_anomaly['z_score'],
                    detection_method="z-score",
                    confidence=z_score_anomaly['confidence']
                )
                anomalies.append(anomaly)
            
            # Isolation Forest detection
            if self.config.use_rolling_stats and len(historical_returns) >= 50:
                iso_anomaly = self._detect_isolation_forest_anomaly(
                    historical_returns,
                    current_return,
                    symbol,
                    "price"
                )
                
                if iso_anomaly and not z_score_anomaly:  # Avoid duplicates
                    anomaly_type = (AnomalyType.PRICE_SPIKE if current_return > 0 
                                  else AnomalyType.PRICE_CRASH)
                    
                    anomaly = self._create_anomaly_event(
                        anomaly_type=anomaly_type,
                        symbol=symbol,
                        detected_value=current_price,
                        expected_value=price_history[-1],
                        deviation=iso_anomaly['anomaly_score'],
                        detection_method="isolation_forest",
                        confidence=iso_anomaly['confidence']
                    )
                    anomalies.append(anomaly)
            
            self._detection_count += 1
            self._anomaly_count += len(anomalies)
            
            return anomalies
    
    def detect_volume_anomalies(self,
                              volume_history: List[int],
                              current_volume: int,
                              symbol: str) -> List[AnomalyEvent]:
        """Detect anomalies in trading volume."""
        with self._handle_error("detecting volume anomalies"):
            anomalies = []
            
            if len(volume_history) < self.config.min_samples:
                return anomalies
            
            volumes = np.array(volume_history + [current_volume], dtype=float)
            
            # Log transform for volume (often log-normal distributed)
            log_volumes = np.log1p(volumes)  # log(1 + x) to handle zero volumes
            current_log_volume = log_volumes[-1]
            historical_log_volumes = log_volumes[:-1]
            
            # Z-score detection on log volumes
            z_score_anomaly = self._detect_zscore_anomaly(
                historical_log_volumes,
                current_log_volume,
                "volume",
                use_robust=True
            )
            
            if z_score_anomaly:
                anomaly = self._create_anomaly_event(
                    anomaly_type=AnomalyType.VOLUME_SURGE,
                    symbol=symbol,
                    detected_value=current_volume,
                    expected_value=np.median(volume_history),
                    deviation=z_score_anomaly['z_score'],
                    detection_method="z-score",
                    confidence=z_score_anomaly['confidence']
                )
                anomalies.append(anomaly)
            
            return anomalies
    
    def detect_volatility_anomalies(self,
                                  price_history: List[float],
                                  symbol: str) -> List[AnomalyEvent]:
        """Detect anomalies in price volatility."""
        with self._handle_error("detecting volatility anomalies"):
            anomalies = []
            
            if len(price_history) < self.config.volatility_window * 2:
                return anomalies
            
            prices = np.array(price_history)
            returns = np.diff(np.log(prices))
            
            # Calculate rolling volatility
            volatilities = []
            window = self.config.volatility_window
            
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                vol = np.std(window_returns) * np.sqrt(252)  # Annualized
                volatilities.append(vol)
            
            if len(volatilities) < 2:
                return anomalies
            
            current_vol = volatilities[-1]
            historical_vols = volatilities[:-1]
            
            # Detect volatility spike
            z_score_anomaly = self._detect_zscore_anomaly(
                np.array(historical_vols),
                current_vol,
                "volatility"
            )
            
            if z_score_anomaly:
                anomaly = self._create_anomaly_event(
                    anomaly_type=AnomalyType.VOLATILITY_SPIKE,
                    symbol=symbol,
                    detected_value=current_vol,
                    expected_value=np.mean(historical_vols),
                    deviation=z_score_anomaly['z_score'],
                    detection_method="z-score",
                    confidence=z_score_anomaly['confidence']
                )
                anomalies.append(anomaly)
            
            return anomalies
    
    def _detect_zscore_anomaly(self,
                             historical_data: np.ndarray,
                             current_value: float,
                             data_type: str,
                             use_robust: bool = None) -> Optional[Dict[str, Any]]:
        """Detect anomaly using z-score method."""
        if use_robust is None:
            use_robust = self.config.use_robust_estimators
        
        if use_robust:
            # Use median and MAD for robust estimation
            median = np.median(historical_data)
            mad = np.median(np.abs(historical_data - median))
            std = mad * 1.4826  # Scale MAD to standard deviation
            
            if std == 0:
                std = np.std(historical_data)
            
            z_score = (current_value - median) / std if std > 0 else 0
        else:
            # Standard z-score
            mean = np.mean(historical_data)
            std = np.std(historical_data)
            z_score = (current_value - mean) / std if std > 0 else 0
        
        if abs(z_score) > self.config.z_score_threshold:
            # Calculate confidence based on z-score magnitude
            confidence = 1 - (2 * stats.norm.cdf(-abs(z_score)))
            
            return {
                'z_score': z_score,
                'confidence': confidence,
                'threshold': self.config.z_score_threshold,
                'data_type': data_type
            }
        
        return None
    
    def _detect_isolation_forest_anomaly(self,
                                       historical_data: np.ndarray,
                                       current_value: float,
                                       symbol: str,
                                       data_type: str) -> Optional[Dict[str, Any]]:
        """Detect anomaly using Isolation Forest."""
        key = f"{symbol}_{data_type}"
        
        # Reshape data for sklearn
        X_historical = historical_data.reshape(-1, 1)
        X_current = np.array([[current_value]])
        
        # Train or retrieve model
        if key not in self._isolation_forests:
            self._isolation_forests[key] = IsolationForest(
                contamination=self.config.outlier_fraction,
                random_state=42,
                n_estimators=100
            )
            self._isolation_forests[key].fit(X_historical)
        
        model = self._isolation_forests[key]
        
        # Predict anomaly
        anomaly_prediction = model.predict(X_current)[0]
        anomaly_score = model.score_samples(X_current)[0]
        
        if anomaly_prediction == -1:  # Anomaly detected
            # Convert anomaly score to confidence (scores are negative, closer to 0 is more normal)
            confidence = 1 - np.exp(anomaly_score)
            
            return {
                'anomaly_score': -anomaly_score,  # Make positive for consistency
                'confidence': confidence,
                'data_type': data_type
            }
        
        return None
    
    def _create_anomaly_event(self,
                            anomaly_type: AnomalyType,
                            symbol: str,
                            detected_value: float,
                            expected_value: float,
                            deviation: float,
                            detection_method: str,
                            confidence: float) -> AnomalyEvent:
        """Create an anomaly event."""
        # Determine severity based on deviation
        severity = self._calculate_severity(deviation, anomaly_type)
        
        # Generate message
        message = self._generate_anomaly_message(
            anomaly_type, symbol, detected_value, expected_value, deviation
        )
        
        # Create event
        event = AnomalyEvent(
            event_id=f"STAT_{datetime.utcnow().timestamp():.0f}_{self._anomaly_count}",
            anomaly_type=anomaly_type,
            severity=severity,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            message=message,
            detected_value=detected_value,
            expected_value=expected_value,
            threshold=self.config.z_score_threshold,
            deviation=deviation,
            detection_method=detection_method,
            model_confidence=confidence,
            false_positive_probability=1 - confidence,
            suggested_actions=self._suggest_actions(anomaly_type, severity)
        )
        
        return event
    
    def _calculate_severity(self, deviation: float, anomaly_type: AnomalyType) -> AnomalySeverity:
        """Calculate anomaly severity based on deviation."""
        abs_deviation = abs(deviation)
        
        # Adjust thresholds based on anomaly type
        if anomaly_type in [AnomalyType.PRICE_CRASH, AnomalyType.CIRCUIT_BREAKER_RISK]:
            # More sensitive to negative events
            if abs_deviation >= 4.5:
                return AnomalySeverity.CRITICAL
            elif abs_deviation >= 3.5:
                return AnomalySeverity.HIGH
            elif abs_deviation >= 2.5:
                return AnomalySeverity.MEDIUM
            else:
                return AnomalySeverity.LOW
        else:
            # Standard thresholds
            if abs_deviation >= 5:
                return AnomalySeverity.CRITICAL
            elif abs_deviation >= 4:
                return AnomalySeverity.HIGH
            elif abs_deviation >= 3:
                return AnomalySeverity.MEDIUM
            else:
                return AnomalySeverity.LOW
    
    def _generate_anomaly_message(self,
                                anomaly_type: AnomalyType,
                                symbol: str,
                                detected_value: float,
                                expected_value: float,
                                deviation: float) -> str:
        """Generate descriptive message for anomaly."""
        messages = {
            AnomalyType.PRICE_SPIKE: f"{symbol} price spike detected: ${detected_value:.2f} ({deviation:.1f}� deviation)",
            AnomalyType.PRICE_CRASH: f"{symbol} price crash detected: ${detected_value:.2f} ({deviation:.1f}� deviation)",
            AnomalyType.VOLUME_SURGE: f"{symbol} volume surge: {detected_value:,.0f} ({deviation:.1f}� above normal)",
            AnomalyType.VOLATILITY_SPIKE: f"{symbol} volatility spike: {detected_value:.1%} annualized ({deviation:.1f}�)"
        }
        
        return messages.get(anomaly_type, f"{symbol} anomaly detected: {deviation:.1f}� deviation")
    
    def _suggest_actions(self, anomaly_type: AnomalyType, severity: AnomalySeverity) -> List[str]:
        """Suggest actions based on anomaly type and severity."""
        actions = []
        
        if severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
            actions.append("Review and adjust position sizes")
            actions.append("Check for news or market events")
            
            if anomaly_type == AnomalyType.PRICE_CRASH:
                actions.append("Consider stop-loss activation")
                actions.append("Evaluate hedging options")
            elif anomaly_type == AnomalyType.VOLATILITY_SPIKE:
                actions.append("Reduce leverage if applicable")
                actions.append("Widen stop-loss levels")
        
        return actions
    
    def update_buffers(self, symbol: str, price: float, volume: int):
        """Update internal data buffers."""
        # Initialize buffers if needed
        if symbol not in self._price_buffers:
            self._price_buffers[symbol] = deque(maxlen=self.config.price_window)
            self._volume_buffers[symbol] = deque(maxlen=self.config.volume_window)
            self._return_buffers[symbol] = deque(maxlen=self.config.price_window)
        
        # Update buffers
        self._price_buffers[symbol].append(price)
        self._volume_buffers[symbol].append(volume)
        
        # Calculate return if we have previous price
        if len(self._price_buffers[symbol]) > 1:
            prev_price = self._price_buffers[symbol][-2]
            log_return = np.log(price / prev_price)
            self._return_buffers[symbol].append(log_return)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            'detection_count': self._detection_count,
            'anomaly_count': self._anomaly_count,
            'detection_rate': self._anomaly_count / self._detection_count if self._detection_count > 0 else 0,
            'monitored_symbols': len(self._price_buffers),
            'models_trained': len(self._isolation_forests),
            'config': {
                'z_score_threshold': self.config.z_score_threshold,
                'min_samples': self.config.min_samples,
                'outlier_fraction': self.config.outlier_fraction
            }
        }


# Additional specialized detectors

class ZScoreDetector:
    """Specialized z-score based anomaly detector."""
    
    def __init__(self, threshold: float = 3.0, use_mad: bool = True):
        self.threshold = threshold
        self.use_mad = use_mad
    
    def detect(self, data: np.ndarray, value: float) -> Tuple[bool, float]:
        """Detect if value is anomalous compared to data."""
        if self.use_mad:
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            z_score = (value - median) / (1.4826 * mad) if mad > 0 else 0
        else:
            z_score = (value - np.mean(data)) / np.std(data) if np.std(data) > 0 else 0
        
        is_anomaly = abs(z_score) > self.threshold
        return is_anomaly, z_score


class IsolationForestDetector:
    """Specialized Isolation Forest detector."""
    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = None
    
    def fit(self, data: np.ndarray):
        """Fit the isolation forest model."""
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        self.model.fit(data.reshape(-1, 1))
    
    def detect(self, value: float) -> Tuple[bool, float]:
        """Detect if value is anomalous."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        prediction = self.model.predict([[value]])[0]
        score = self.model.score_samples([[value]])[0]
        
        is_anomaly = prediction == -1
        return is_anomaly, -score  # Return positive score