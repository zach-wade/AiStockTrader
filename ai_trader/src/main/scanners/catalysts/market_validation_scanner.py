# File: ai_trader/scanners/market_validation_scanner.py
"""
Market Validation Scanner

Validates catalyst signals from other scanners against real-time market data.
It acts as a confirmation layer, adding a "market validation score" to existing signals.

Now uses the repository pattern with hot/cold storage awareness to
efficiently access recent market data for validation.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import asyncio
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

from omegaconf import DictConfig
from main.interfaces.scanners import IScanner, IScannerRepository
from main.interfaces.events import IEventBus
from main.data_pipeline.storage.repositories.repository_types import QueryFilter
from main.utils.scanners import (
    ScannerDataAccess,
    ScannerCacheManager,
    ScannerMetricsCollector
)
from main.utils.core import timer
from ..catalyst_scanner_base import CatalystScannerBase
from main.events.types import AlertType, ScanAlert

logger = logging.getLogger(__name__)

class MarketValidationScanner(CatalystScannerBase):
    """
    Validates catalyst signals from other scanners against real-time market data.
    It acts as a confirmation layer, adding a "market validation score" to existing signals.
    
    Now uses the repository pattern with hot/cold storage awareness to
    efficiently access recent market data for validation.
    """

    def __init__(
        self,
        config: DictConfig,
        repository: IScannerRepository,
        event_bus: Optional[IEventBus] = None,
        metrics_collector: Optional[ScannerMetricsCollector] = None,
        cache_manager: Optional[ScannerCacheManager] = None
    ):
        """
        Initializes the MarketValidationScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "MarketValidationScanner",
            config,
            repository,
            event_bus,
            metrics_collector,
            cache_manager
        )
        
        self.params = self.config.get('scanners.market_validation', {})
        
        # Thresholds
        self.price_change_threshold = self.params.get('price_change_threshold_pct', 5.0)
        self.volume_spike_ratio = self.params.get('volume_spike_ratio', 3.0)
        self.correlation_threshold = self.params.get('correlation_threshold', 0.5)
        self.lookback_hours = self.params.get('lookback_hours', 48)  # 2 days
        self.use_cache = self.params.get('use_cache', True)
        
        # In-memory cache for recent price history to calculate correlation
        self.price_history: Dict[str, pd.Series] = {}
        
        # Track initialization
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize scanner resources."""
        if self._initialized:
            return
        
        logger.info(f"Initializing {self.name}")
        self._initialized = True
    
    async def cleanup(self) -> None:
        """Clean up scanner resources."""
        logger.info(f"Cleaning up {self.name}")
        self._initialized = False
        self.price_history.clear()

    async def scan(self, symbols: List[str], **kwargs) -> List[ScanAlert]:
        """
        Perform market validation on provided symbols or catalyst signals.
        
        Uses repository pattern for efficient market data access with hot storage
        for recent intraday data needed for validation.
        
        Args:
            symbols: List of stock symbols to validate
            **kwargs: Can include 'catalyst_signals' for validation mode
            
        Returns:
            List of ScanAlert objects for market validation signals
        """
        if not self._initialized:
            await self.initialize()
            
        with timer() as t:
            logger.info(f"📈 Market Validation Scanner: Analyzing {len(symbols)} symbols...")
            
            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(timezone.utc)
            
            try:
                # Check if we're in validation mode with catalyst signals
                catalyst_signals = kwargs.get('catalyst_signals', {})
                
                if catalyst_signals:
                    # Validate existing catalyst signals
                    validated_signals = await self._validate_catalyst_signals(catalyst_signals)
                    # Convert to ScanAlert format
                    alerts = self._convert_validated_signals_to_alerts(validated_signals)
                else:
                    # Direct market validation scan
                    # Check cache if enabled
                    if self.cache and self.use_cache:
                        cache_key = f"market_validation:{','.join(sorted(symbols[:10]))}:{self.lookback_hours}"
                        cached_alerts = await self.cache.get_cached_result(
                            self.name,
                            "batch",
                            cache_key
                        )
                        if cached_alerts is not None:
                            logger.info(f"Using cached results for market validation")
                            return cached_alerts
                    
                    # Build query filter for recent data
                    query_filter = QueryFilter(
                        symbols=symbols,
                        start_date=datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours),
                        end_date=datetime.now(timezone.utc)
                    )
                    
                    # Get market data from repository
                    # This will primarily use hot storage for recent data
                    market_data = await self.repository.get_market_data(
                        symbols=symbols,
                        query_filter=query_filter,
                        columns=['date', 'symbol', 'open', 'close', 'volume']
                    )
                    
                    alerts = []
                    
                    for symbol in symbols:
                        if symbol not in market_data:
                            logger.warning(f"No market data available for {symbol}")
                            continue
                        
                        price_data = market_data[symbol]
                        if price_data.empty:
                            continue
                        
                        # Store price history for correlation analysis
                        self.price_history[symbol] = price_data['close']
                        
                        # Calculate validation score
                        score, reasons = self._calculate_validation_score(symbol, price_data)
                        
                        # Create alert if score is significant
                        if score > 0:
                            normalized_score = min(score / 10.0, 1.0)  # Normalize to 0-1 range
                            alert = self.create_alert(
                                symbol=symbol,
                                alert_type=AlertType.MOMENTUM,  # Market validation typically indicates momentum
                                score=normalized_score,
                                metadata={
                                    'validation_reasons': reasons,
                                    'raw_score': score,
                                    'catalyst_type': 'market_validation'
                                }
                            )
                            alerts.append(alert)
                            
                            # Record metric
                            if self.metrics:
                                self.metrics.record_alert_generated(
                                    self.name,
                                    AlertType.MOMENTUM,
                                    symbol,
                                    normalized_score
                                )
                    
                    # Cache results if enabled
                    if self.cache and self.use_cache and alerts:
                        await self.cache.cache_result(
                            self.name,
                            "batch",
                            cache_key,
                            alerts,
                            ttl_seconds=300  # 5 minute TTL for validation
                        )
                    
                    alerts = self.deduplicate_alerts(alerts)
                
                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)
                
                logger.info(
                    f"✅ Market Validation Scanner: Found {len(alerts)} validated signals "
                    f"in {t.elapsed_ms:.2f}ms"
                )
                
                # Record metrics
                if self.metrics:
                    self.metrics.record_scan_duration(
                        self.name,
                        t.elapsed_ms,
                        len(symbols)
                    )
                
                return alerts
                
            except Exception as e:
                logger.error(f"❌ Error in Market Validation Scanner: {e}", exc_info=True)
                
                # Record error metric
                if self.metrics:
                    self.metrics.record_scan_error(
                        self.name,
                        type(e).__name__,
                        str(e)
                    )
                
                return []
    
    async def run(self, catalyst_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Legacy method for backward compatibility.
        Takes a dictionary of existing catalyst signals and adds a validation score.

        Args:
            catalyst_signals: The output from other Layer 2 scanners.
            
        Returns:
            The same dictionary, with signals enhanced with market validation data.
        """
        return await self._validate_catalyst_signals(catalyst_signals)
    
    async def _validate_catalyst_signals(self, catalyst_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Internal method to validate catalyst signals."""
        if not catalyst_signals:
            return {}
            
        symbols_to_validate = list(catalyst_signals.keys())
        logger.info(f"📈 Market Validation Scanner: Validating {len(symbols_to_validate)} catalyst signals...")

        # Build query filter for recent data
        query_filter = QueryFilter(
            symbols=symbols_to_validate,
            start_date=datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours),
            end_date=datetime.now(timezone.utc)
        )
        
        # Get market data from repository
        market_data = await self.repository.get_market_data(
            symbols=symbols_to_validate,
            query_filter=query_filter,
            columns=['date', 'symbol', 'open', 'close', 'volume']
        )

        validated_signals = catalyst_signals.copy()

        for symbol in symbols_to_validate:
            if symbol not in market_data:
                logger.warning(f"Could not fetch market data for {symbol}. Skipping validation.")
                continue
            
            price_data = market_data[symbol]
            if price_data.empty:
                continue

            # Calculate validation score
            validation_score, validation_reasons = self._calculate_validation_score(symbol, price_data)
            
            # Enhance the original signal with validation info
            if symbol in validated_signals:
                original_score = validated_signals[symbol].get('score', 0)
                # The final score is a blend of the original catalyst score and the market validation
                validated_signals[symbol]['final_score'] = (original_score * 0.6) + (validation_score * 0.4)
                validated_signals[symbol]['market_validation_score'] = validation_score
                validated_signals[symbol]['market_validation_reasons'] = validation_reasons
        
        return validated_signals
    
    def _convert_validated_signals_to_alerts(self, validated_signals: Dict[str, Dict[str, Any]]) -> List[ScanAlert]:
        """Convert validated signals dictionary to ScanAlert format."""
        alerts = []
        
        for symbol, signal in validated_signals.items():
            # Only create alerts for signals with significant validation
            if signal.get('market_validation_score', 0) > 0:
                # Normalize the validation score
                normalized_score = min(signal.get('final_score', 0) / 10.0, 1.0)
                
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.MOMENTUM,
                    score=normalized_score,
                    metadata={
                        'original_signal': signal,
                        'validation_score': signal.get('market_validation_score'),
                        'validation_reasons': signal.get('market_validation_reasons', []),
                        'final_score': signal.get('final_score'),
                        'catalyst_type': 'validated_signal'
                    }
                )
                alerts.append(alert)
                
                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name,
                        AlertType.MOMENTUM,
                        symbol,
                        normalized_score
                    )
        
        return alerts

    def _calculate_validation_score(self, symbol: str, price_data: pd.DataFrame) -> tuple[float, List[str]]:
        """Calculates a validation score (0-10) based on live price/volume action."""
        score = 0.0
        reasons = []

        # Ensure we have enough data
        if len(price_data) < 2:
            return score, reasons

        # 1. Price Change Anomaly
        # Use first available open and last close for the period
        first_open = price_data['open'].iloc[0]
        last_close = price_data['close'].iloc[-1]
        
        if first_open > 0:
            price_change_pct = (last_close / first_open - 1) * 100
            if abs(price_change_pct) >= self.price_change_threshold:
                score += 4.0
                reasons.append(f"Price Change: {price_change_pct:+.2f}%")

        # 2. Volume Spike Anomaly
        if 'volume' in price_data.columns and len(price_data) > 1:
            # Calculate average volume excluding the last data point
            avg_volume = price_data['volume'].iloc[:-1].mean()
            latest_volume = price_data['volume'].iloc[-1]
            
            if avg_volume > 0 and (latest_volume / avg_volume) >= self.volume_spike_ratio:
                score += 3.0
                reasons.append(f"Volume Spike: {latest_volume / avg_volume:.1f}x")
        
        # 3. Price Momentum Score
        if len(price_data) >= 10:
            # Calculate momentum as recent performance vs earlier performance
            mid_point = len(price_data) // 2
            early_avg = price_data['close'].iloc[:mid_point].mean()
            recent_avg = price_data['close'].iloc[mid_point:].mean()
            
            if early_avg > 0:
                momentum_pct = (recent_avg / early_avg - 1) * 100
                if abs(momentum_pct) > 2.0:  # 2% momentum threshold
                    score += min(abs(momentum_pct) / 5.0 * 3.0, 3.0)  # Max 3 points
                    reasons.append(f"Momentum: {momentum_pct:+.2f}%")

        return min(score, 10.0), reasons