"""
Real-Time Anomaly Detection Orchestrator

This module provides the main orchestration layer for comprehensive real-time anomaly detection.
Coordinates multiple specialized detectors and provides a unified API for anomaly monitoring.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from collections import deque

from .anomaly_types import AnomalyType, AnomalySeverity
from .anomaly_models import AnomalyEvent, MarketRegime
from .statistical_detector import StatisticalAnomalyDetector
from .correlation_detector import CorrelationAnomalyDetector
from .regime_detector import MarketRegimeDetector

logger = logging.getLogger(__name__)


class RealTimeAnomalyDetector:
    """
    Comprehensive real-time anomaly detection system combining
    multiple detection methods for enhanced market safety.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize comprehensive anomaly detector."""
        self.config = config or {}
        
        # Sub-detectors
        self.statistical_detector = StatisticalAnomalyDetector(self.config.get('statistical', {}))
        self.correlation_detector = CorrelationAnomalyDetector(self.config.get('correlation', {}))
        self.regime_detector = MarketRegimeDetector(self.config.get('regime', {}))
        
        # Anomaly tracking
        self.recent_anomalies: deque = deque(maxlen=self.config.get('anomaly_history_size', 1000))
        self.anomaly_callbacks: List[Callable] = []
        
        # Data buffers
        self.price_buffers: Dict[str, deque] = {}
        self.volume_buffers: Dict[str, deque] = {}
        
        # Detection parameters
        self.max_buffer_size = self.config.get('max_buffer_size', 200)
        self.detection_enabled = True
        self.monitoring_interval = self.config.get('monitoring_interval', 30)
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("Real-time anomaly detector initialized")
    
    async def start_monitoring(self):
        """Start real-time anomaly monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Anomaly detection monitoring started")
    
    async def stop_monitoring(self):
        """Stop real-time anomaly monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Anomaly detection monitoring stopped")
    
    async def update_market_data(self, 
                               symbol: str,
                               price: float,
                               volume: int,
                               timestamp: Optional[datetime] = None):
        """Update market data and trigger anomaly detection."""
        
        if not self.detection_enabled:
            return
        
        # Initialize buffers if needed
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = deque(maxlen=self.max_buffer_size)
            self.volume_buffers[symbol] = deque(maxlen=self.max_buffer_size)
        
        # Add new data
        self.price_buffers[symbol].append(price)
        self.volume_buffers[symbol].append(volume)
        
        # Trigger anomaly detection
        await self._detect_anomalies_for_symbol(symbol)
    
    async def detect_portfolio_anomalies(self,
                                       portfolio_data: Dict[str, Any],
                                       correlation_matrix: Optional[np.ndarray] = None,
                                       symbols: Optional[List[str]] = None) -> List[AnomalyEvent]:
        """Detect portfolio-level anomalies."""
        
        anomalies = []
        
        # Correlation breakdown detection
        if correlation_matrix is not None and symbols:
            correlation_anomalies = self.correlation_detector.detect_correlation_breakdown(
                symbols, correlation_matrix
            )
            anomalies.extend(correlation_anomalies)
        
        # Market regime detection (using portfolio-level data)
        if 'portfolio_value_history' in portfolio_data:
            regime_anomalies = self.regime_detector.detect_regime_change(
                portfolio_data['portfolio_value_history'],
                portfolio_data.get('total_volume_history', [])
            )
            anomalies.extend(regime_anomalies)
        
        # Store anomalies
        for anomaly in anomalies:
            self.recent_anomalies.append(anomaly)
            
            # Log anomaly
            self._log_anomaly(anomaly)
            
            # Trigger callbacks
            for callback in self.anomaly_callbacks:
                try:
                    await callback(anomaly)
                except Exception as e:
                    logger.error(f"Error in anomaly callback: {e}")
        
        return anomalies
    
    async def _detect_anomalies_for_symbol(self, symbol: str):
        """Detect anomalies for a specific symbol."""
        
        if symbol not in self.price_buffers or len(self.price_buffers[symbol]) < 10:
            return  # Not enough data
        
        price_history = list(self.price_buffers[symbol])
        volume_history = list(self.volume_buffers[symbol])
        current_price = price_history[-1]
        current_volume = volume_history[-1]
        
        anomalies = []
        
        # Price anomaly detection
        price_anomalies = self.statistical_detector.detect_price_anomalies(
            price_history[:-1], current_price, symbol
        )
        anomalies.extend(price_anomalies)
        
        # Volume anomaly detection
        volume_anomalies = self.statistical_detector.detect_volume_anomalies(
            volume_history[:-1], current_volume, symbol
        )
        anomalies.extend(volume_anomalies)
        
        # Volatility anomaly detection
        volatility_anomalies = self.statistical_detector.detect_volatility_anomalies(
            price_history, symbol
        )
        anomalies.extend(volatility_anomalies)
        
        # Store and process anomalies
        for anomaly in anomalies:
            self.recent_anomalies.append(anomaly)
            
            # Log anomaly
            self._log_anomaly(anomaly)
            
            # Trigger callbacks
            for callback in self.anomaly_callbacks:
                try:
                    await callback(anomaly)
                except Exception as e:
                    logger.error(f"Error in anomaly callback: {e}")
    
    def _log_anomaly(self, anomaly: AnomalyEvent):
        """Log anomaly with appropriate severity level."""
        
        log_message = f"ANOMALY DETECTED [{anomaly.severity.value.upper()}]: {anomaly.message}"
        
        if anomaly.severity == AnomalySeverity.CRITICAL:
            logger.critical(log_message)
        elif anomaly.severity == AnomalySeverity.HIGH:
            logger.error(log_message)
        elif anomaly.severity == AnomalySeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    async def _monitoring_loop(self):
        """Background monitoring loop for continuous anomaly detection."""
        
        while True:
            try:
                # Perform periodic anomaly analysis
                await self._periodic_anomaly_analysis()
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                logger.info("Anomaly detection monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in anomaly detection monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _periodic_anomaly_analysis(self):
        """Perform periodic cross-symbol anomaly analysis."""
        
        # Analyze patterns across all symbols
        if len(self.price_buffers) >= 2:
            symbols = list(self.price_buffers.keys())
            
            # Calculate cross-symbol correlations
            if len(symbols) >= 2:
                # Build correlation matrix
                min_length = min(len(self.price_buffers[symbol]) for symbol in symbols)
                if min_length >= 30:  # Minimum data for correlation
                    price_matrix = []
                    for symbol in symbols:
                        prices = list(self.price_buffers[symbol])[-min_length:]
                        returns = np.diff(np.log(prices))
                        price_matrix.append(returns)
                    
                    if len(price_matrix) >= 2:
                        correlation_matrix = np.corrcoef(price_matrix)
                        
                        # Detect correlation anomalies
                        portfolio_anomalies = await self.detect_portfolio_anomalies(
                            portfolio_data={},
                            correlation_matrix=correlation_matrix,
                            symbols=symbols
                        )
    
    # Public API methods
    
    def add_anomaly_callback(self, callback: Callable):
        """Add callback for anomaly events."""
        self.anomaly_callbacks.append(callback)
        logger.info("Anomaly callback registered")
    
    def remove_anomaly_callback(self, callback: Callable):
        """Remove anomaly callback."""
        if callback in self.anomaly_callbacks:
            self.anomaly_callbacks.remove(callback)
            logger.info("Anomaly callback removed")
    
    def get_recent_anomalies(self, 
                           limit: Optional[int] = None,
                           severity_filter: Optional[AnomalySeverity] = None,
                           type_filter: Optional[AnomalyType] = None) -> List[AnomalyEvent]:
        """Get recent anomalies with optional filtering."""
        
        anomalies = list(self.recent_anomalies)
        
        if severity_filter:
            anomalies = [a for a in anomalies if a.severity == severity_filter]
        
        if type_filter:
            anomalies = [a for a in anomalies if a.anomaly_type == type_filter]
        
        if limit:
            anomalies = anomalies[-limit:]
        
        return anomalies
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics."""
        
        recent_24h = [
            a for a in self.recent_anomalies 
            if (datetime.now(timezone.utc) - a.timestamp).total_seconds() < 86400
        ]
        
        severity_counts = {}
        for severity in AnomalySeverity:
            severity_counts[severity.value] = len([a for a in recent_24h if a.severity == severity])
        
        type_counts = {}
        for anomaly_type in AnomalyType:
            type_counts[anomaly_type.value] = len([a for a in recent_24h if a.anomaly_type == anomaly_type])
        
        return {
            'total_anomalies_24h': len(recent_24h),
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'detection_enabled': self.detection_enabled,
            'monitored_symbols': len(self.price_buffers),
            'avg_detection_confidence': np.mean([a.model_confidence for a in recent_24h]) if recent_24h else 0,
            'monitoring_active': self._monitoring_task is not None and not self._monitoring_task.done()
        }
    
    def enable_detection(self):
        """Enable anomaly detection."""
        self.detection_enabled = True
        logger.info("Anomaly detection enabled")
    
    def disable_detection(self):
        """Disable anomaly detection."""
        self.detection_enabled = False
        logger.warning("Anomaly detection disabled")
    
    def clear_anomaly_history(self):
        """Clear anomaly history."""
        self.recent_anomalies.clear()
        logger.info("Anomaly history cleared")
    
    def clear_data_buffers(self):
        """Clear all data buffers."""
        self.price_buffers.clear()
        self.volume_buffers.clear()
        logger.info("Data buffers cleared")
    
    def get_current_market_regime(self) -> Optional[MarketRegime]:
        """Get current market regime."""
        return self.regime_detector.get_current_regime()
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get status of data buffers."""
        return {
            'total_symbols': len(self.price_buffers),
            'symbol_buffer_sizes': {
                symbol: len(self.price_buffers[symbol]) 
                for symbol in self.price_buffers
            },
            'max_buffer_size': self.max_buffer_size,
            'total_data_points': sum(len(buffer) for buffer in self.price_buffers.values())
        }
    
    def get_detector_status(self) -> Dict[str, Any]:
        """Get status of sub-detectors."""
        return {
            'statistical_detector': {
                'class': 'StatisticalAnomalyDetector',
                'config': self.statistical_detector.config
            },
            'correlation_detector': {
                'class': 'CorrelationAnomalyDetector',
                'baselines': len(self.correlation_detector.baseline_correlations),
                'history_size': len(self.correlation_detector.correlation_history)
            },
            'regime_detector': {
                'class': 'MarketRegimeDetector',
                'current_regime': self.regime_detector.get_current_regime().name if self.regime_detector.get_current_regime() else None,
                'regime_history': len(self.regime_detector.get_regime_history())
            }
        }


# Backward compatibility alias
AnomalyDetector = RealTimeAnomalyDetector