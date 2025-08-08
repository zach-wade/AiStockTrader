"""
Intermarket Correlation Scanner

Detects cross-market relationships, correlations, and anomalies that
could signal trading opportunities or systemic risks.

Now uses the repository pattern with hot/cold storage awareness to
efficiently access cross-market data and historical patterns.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

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


class IntermarketScanner(CatalystScannerBase):
    """
    Scans for intermarket relationships and anomalies:
    - Cross-asset correlations (stocks, bonds, commodities, currencies)
    - Sector rotation patterns
    - Lead-lag relationships
    - Divergences and convergences
    - Risk-on/risk-off regime changes
    
    Now uses the repository pattern with hot/cold storage awareness to
    efficiently access cross-market data and historical patterns.
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
        Initializes the IntermarketScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "IntermarketScanner",
            config,
            repository,
            event_bus,
            metrics_collector,
            cache_manager
        )
        # Scanner-specific parameters
        self.params = self.config.get('scanners.intermarket', {})
        self.lookback_days = self.params.get('lookback_days', 60)
        self.correlation_window = self.params.get('correlation_window', 20)
        self.min_correlation = self.params.get('min_correlation', 0.7)
        self.divergence_threshold = self.params.get('divergence_threshold', 2.0)  # std devs
        self.use_cache = self.params.get('use_cache', True)
        
        # Key market indicators to track
        self.market_indicators = self.params.get('market_indicators', {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ',
            'IWM': 'Russell 2000',
            'TLT': '20Y Treasury',
            'DXY': 'Dollar Index',
            'GLD': 'Gold',
            'USO': 'Oil',
            'VIX': 'Volatility'
        })
        
        # Sector ETFs for rotation analysis
        self.sector_etfs = self.params.get('sector_etfs', {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        })
        
        # Alert thresholds
        self.alert_thresholds = self.params.get('alert_thresholds', {
            'correlation_break': 0.8,
            'divergence': 0.7,
            'regime_change': 0.75,
            'sector_rotation': 0.6
        })
        
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
    
    async def scan(self, symbols: List[str], **kwargs) -> List[ScanAlert]:
        """
        Scan for intermarket patterns and anomalies.
        
        Uses repository pattern for efficient cross-market data access with hot storage
        for recent correlations and cold storage for historical patterns.
        
        Args:
            symbols: List of stock symbols to analyze
            **kwargs: Additional scanner parameters
            
        Returns:
            List of ScanAlert objects for detected patterns
        """
        if not self._initialized:
            await self.initialize()
            
        with timer() as t:
            logger.info(f"🔄 Intermarket Scanner: Analyzing {len(symbols)} symbols...")
            
            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(timezone.utc)
            
            try:
                # Check cache if enabled
                if self.cache and self.use_cache:
                    cache_key = f"intermarket_scan:{','.join(sorted(symbols[:10]))}:{self.lookback_days}"
                    cached_alerts = await self.cache.get_cached_result(
                        self.name,
                        "batch",
                        cache_key
                    )
                    if cached_alerts is not None:
                        logger.info(f"Using cached results for intermarket scan")
                        return cached_alerts
                
                # Build query filter
                query_filter = QueryFilter(
                    symbols=symbols + list(self.market_indicators.keys()) + list(self.sector_etfs.keys()),
                    start_date=datetime.now(timezone.utc) - timedelta(days=self.lookback_days),
                    end_date=datetime.now(timezone.utc)
                )
                
                # Get market data from repository
                # This will use hot storage for recent data, cold for historical
                market_data = await self.repository.get_market_data(
                    symbols=query_filter.symbols,
                    query_filter=query_filter,
                    columns=['date', 'symbol', 'close', 'volume', 'returns']
                )
                
                if not market_data:
                    logger.warning("No market data available for intermarket analysis")
                    return []
                
                # Organize data by symbol
                symbol_data = self._organize_market_data(market_data)
                
                alerts = []
                
                # Batch process symbols for efficiency
                batch_size = 50
                for i in range(0, len(symbols), batch_size):
                    batch_symbols = symbols[i:i + batch_size]
                    batch_alerts = await self._analyze_symbol_batch(
                        batch_symbols,
                        symbol_data
                    )
                    alerts.extend(batch_alerts)
                
                # Check for broad market patterns
                market_alerts = self._analyze_market_conditions(symbol_data)
                alerts.extend(market_alerts)
                
                # Deduplicate alerts
                alerts = self.deduplicate_alerts(alerts)
                
                # Cache results if enabled
                if self.cache and self.use_cache and alerts:
                    await self.cache.cache_result(
                        self.name,
                        "batch",
                        cache_key,
                        alerts,
                        ttl_seconds=1800  # 30 minute TTL
                    )
                
                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)
                
                logger.info(
                    f"✅ Intermarket Scanner: Found {len(alerts)} alerts "
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
                logger.error(f"❌ Error in Intermarket Scanner: {e}", exc_info=True)
                
                # Record error metric
                if self.metrics:
                    self.metrics.record_scan_error(
                        self.name,
                        type(e).__name__,
                        str(e)
                    )
                
                return []
    
    async def run(self, universe: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Legacy method for backward compatibility.
        
        Args:
            universe: List of symbols to scan
            
        Returns:
            Dict mapping symbol to catalyst signal data
        """
        # Use the new scan method
        alerts = await self.scan(universe)
        
        # Convert to legacy format
        catalyst_signals = defaultdict(list)
        for alert in alerts:
            signal = {
                'score': alert.metadata.get('raw_score', alert.score * 5.0),  # Convert from 0-1 to 0-5 scale
                'reason': alert.metadata.get('reason', ''),
                'signal_type': 'intermarket',
                'metadata': {
                    'catalyst_type': alert.metadata.get('catalyst_type'),
                    'indicator': alert.metadata.get('indicator'),
                    'correlation': alert.metadata.get('recent_correlation', 
                                                     alert.metadata.get('correlation')),
                    'direction': alert.metadata.get('direction')
                }
            }
            catalyst_signals[alert.symbol].append(signal)
        
        return dict(catalyst_signals)
    
    def _organize_market_data(self, market_data: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Organize market data by symbol into DataFrames.
        """
        symbol_data = defaultdict(list)
        
        # Group data by symbol
        for record in market_data:
            symbol = record['symbol']
            symbol_data[symbol].append(record)
        
        # Convert to DataFrames
        result = {}
        for symbol, records in symbol_data.items():
            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate returns if not present
                if 'returns' not in df.columns and 'close' in df.columns:
                    df['returns'] = df['close'].pct_change()
                
                result[symbol] = df
        
        return result
    
    async def _analyze_symbol_batch(self,
                                   symbols: List[str],
                                   symbol_data: Dict[str, pd.DataFrame]) -> List[ScanAlert]:
        """
        Analyze a batch of symbols for intermarket patterns using concurrent processing.
        """
        async def analyze_single_symbol(symbol: str) -> List[ScanAlert]:
            if symbol not in symbol_data:
                return []
            
            symbol_alerts = []
            
            # Check correlations with market indicators
            correlation_alerts = self._check_correlation_patterns(
                symbol,
                symbol_data[symbol],
                symbol_data
            )
            symbol_alerts.extend(correlation_alerts)
            
            # Check for divergences
            divergence_alerts = self._check_divergences(
                symbol,
                symbol_data[symbol],
                symbol_data
            )
            symbol_alerts.extend(divergence_alerts)
            
            # Check sector relationships (limit to first 10 symbols)
            if symbols.index(symbol) < 10:
                sector_alerts = self._check_sector_relationships(
                    symbol,
                    symbol_data[symbol],
                    symbol_data
                )
                symbol_alerts.extend(sector_alerts)
            
            return symbol_alerts
        
        # Process all symbols concurrently
        symbol_results = await self.process_symbols_individually(
            symbols,
            analyze_single_symbol,
            max_concurrent=self.config.get('max_concurrent_symbols', 15)
        )
        
        # Flatten results
        alerts = []
        for symbol_alerts in symbol_results:
            alerts.extend(symbol_alerts)
        
        return alerts
    
    def _check_correlation_patterns(self,
                                   symbol: str,
                                   symbol_df: pd.DataFrame,
                                   market_data: Dict[str, pd.DataFrame]) -> List[ScanAlert]:
        """
        Check for unusual correlation patterns.
        """
        alerts = []
        
        # Calculate rolling correlations
        symbol_returns = symbol_df['returns'].dropna()
        
        for indicator, indicator_name in self.market_indicators.items():
            if indicator not in market_data:
                continue
            
            indicator_returns = market_data[indicator]['returns'].dropna()
            
            # Align data
            common_dates = symbol_returns.index.intersection(indicator_returns.index)
            if len(common_dates) < self.correlation_window:
                continue
            
            # Calculate correlation
            recent_correlation = symbol_returns.loc[common_dates[-self.correlation_window:]].corr(
                indicator_returns.loc[common_dates[-self.correlation_window:]]
            )
            
            historical_correlation = symbol_returns.loc[common_dates[:-self.correlation_window]].corr(
                indicator_returns.loc[common_dates[:-self.correlation_window]]
            )
            
            # Check for correlation breaks
            correlation_change = abs(recent_correlation - historical_correlation)
            
            if correlation_change > 0.5 and abs(historical_correlation) > self.min_correlation:
                score = min(correlation_change / 0.5, 1.0) * self.alert_thresholds['correlation_break']
                
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.CORRELATION_ANOMALY,
                    score=score,
                    metadata={
                        'indicator': indicator,
                        'indicator_name': indicator_name,
                        'recent_correlation': recent_correlation,
                        'historical_correlation': historical_correlation,
                        'correlation_change': correlation_change,
                        'reason': f"Correlation break with {indicator_name}: {historical_correlation:.2f} -> {recent_correlation:.2f}",
                        'catalyst_type': 'correlation_break',
                        'raw_score': score * 5.0  # Legacy scale
                    }
                )
                alerts.append(alert)
                
                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name,
                        AlertType.CORRELATION_ANOMALY,
                        symbol,
                        score
                    )
            
            # Check for new strong correlations
            elif abs(recent_correlation) > self.min_correlation and abs(historical_correlation) < 0.3:
                score = abs(recent_correlation) * self.alert_thresholds['correlation_break']
                
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.CORRELATION_ANOMALY,
                    score=score,
                    metadata={
                        'indicator': indicator,
                        'indicator_name': indicator_name,
                        'recent_correlation': recent_correlation,
                        'correlation_type': 'new_correlation',
                        'reason': f"New strong correlation with {indicator_name}: {recent_correlation:.2f}",
                        'catalyst_type': 'new_correlation',
                        'raw_score': score * 5.0  # Legacy scale
                    }
                )
                alerts.append(alert)
                
                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name,
                        AlertType.CORRELATION_ANOMALY,
                        symbol,
                        score
                    )
        
        return alerts
    
    def _check_divergences(self,
                          symbol: str,
                          symbol_df: pd.DataFrame,
                          market_data: Dict[str, pd.DataFrame]) -> List[ScanAlert]:
        """
        Check for price divergences with correlated assets.
        """
        alerts = []
        
        # Compare with S&P 500 as baseline
        if 'SPY' not in market_data:
            return alerts
        
        spy_data = market_data['SPY']
        
        # Ensure we have enough data
        if len(symbol_df) < 60 or len(spy_data) < 60:
            return alerts
        
        # Calculate relative performance
        symbol_perf = (symbol_df['close'].iloc[-1] / symbol_df['close'].iloc[-20] - 1)
        spy_perf = (spy_data['close'].iloc[-1] / spy_data['close'].iloc[-20] - 1)
        
        # Calculate historical correlation
        correlation = symbol_df['returns'].iloc[-60:].corr(spy_data['returns'].iloc[-60:])
        
        if abs(correlation) > self.min_correlation:
            # Calculate z-score of performance difference
            perf_diff = symbol_perf - spy_perf
            historical_diffs = []
            
            for i in range(20, 60):
                hist_symbol_perf = (symbol_df['close'].iloc[-i] / symbol_df['close'].iloc[-i-20] - 1)
                hist_spy_perf = (spy_data['close'].iloc[-i] / spy_data['close'].iloc[-i-20] - 1)
                historical_diffs.append(hist_symbol_perf - hist_spy_perf)
            
            mean_diff = np.mean(historical_diffs)
            std_diff = np.std(historical_diffs)
            
            if std_diff > 0:
                z_score = (perf_diff - mean_diff) / std_diff
                
                if abs(z_score) > self.divergence_threshold:
                    score = min(abs(z_score) / 3, 1.0) * self.alert_thresholds['divergence']
                    
                    alert = self.create_alert(
                        symbol=symbol,
                        alert_type=AlertType.DIVERGENCE,
                        score=score,
                        metadata={
                            'benchmark': 'SPY',
                            'correlation': correlation,
                            'symbol_performance': symbol_perf,
                            'spy_performance': spy_perf,
                            'z_score': z_score,
                            'direction': 'outperforming' if z_score > 0 else 'underperforming',
                            'reason': f"{'Outperforming' if z_score > 0 else 'Underperforming'} SPY by {abs(z_score):.1f} std devs",
                            'catalyst_type': 'performance_divergence',
                            'raw_score': score * 5.0  # Legacy scale
                        }
                    )
                    alerts.append(alert)
                    
                    # Record metric
                    if self.metrics:
                        self.metrics.record_alert_generated(
                            self.name,
                            AlertType.DIVERGENCE,
                            symbol,
                            score
                        )
        
        return alerts
    
    def _check_sector_relationships(self,
                                   symbol: str,
                                   symbol_df: pd.DataFrame,
                                   market_data: Dict[str, pd.DataFrame]) -> List[ScanAlert]:
        """
        Check for sector rotation patterns.
        """
        alerts = []
        
        # Find best correlated sector
        sector_correlations = {}
        symbol_returns = symbol_df['returns'].iloc[-60:]
        
        for sector_etf, sector_name in self.sector_etfs.items():
            if sector_etf in market_data:
                sector_returns = market_data[sector_etf]['returns'].iloc[-60:]
                
                # Align data
                common_dates = symbol_returns.index.intersection(sector_returns.index)
                if len(common_dates) >= 60:
                    correlation = symbol_returns.loc[common_dates].corr(sector_returns.loc[common_dates])
                    sector_correlations[sector_etf] = {
                        'name': sector_name,
                        'correlation': correlation
                    }
        
        if not sector_correlations:
            return alerts
        
        # Find primary sector
        primary_sector = max(sector_correlations.items(), key=lambda x: abs(x[1]['correlation']))
        sector_etf, sector_info = primary_sector
        
        if abs(sector_info['correlation']) > 0.6:
            # Check sector rotation
            sector_data = market_data[sector_etf]
            
            # Compare recent vs historical performance
            recent_sector_perf = (sector_data['close'].iloc[-5:].mean() / 
                                 sector_data['close'].iloc[-10:-5].mean() - 1)
            
            historical_sector_perf = (sector_data['close'].iloc[-30:-20].mean() / 
                                     sector_data['close'].iloc[-40:-30].mean() - 1)
            
            perf_change = recent_sector_perf - historical_sector_perf
            
            if abs(perf_change) > 0.02:  # 2% performance change
                score = min(abs(perf_change) / 0.05, 1.0) * self.alert_thresholds['sector_rotation']
                
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.SECTOR_ROTATION,
                    score=score,
                    metadata={
                        'sector': sector_info['name'],
                        'sector_etf': sector_etf,
                        'correlation': sector_info['correlation'],
                        'recent_performance': recent_sector_perf,
                        'performance_change': perf_change,
                        'rotation_direction': 'into' if perf_change > 0 else 'out of',
                        'reason': f"Sector rotation {'into' if perf_change > 0 else 'out of'} {sector_info['name']}",
                        'catalyst_type': 'sector_rotation',
                        'raw_score': score * 5.0  # Legacy scale
                    }
                )
                alerts.append(alert)
                
                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name,
                        AlertType.SECTOR_ROTATION,
                        symbol,
                        score
                    )
        
        return alerts
    
    def _analyze_market_conditions(self, market_data: Dict[str, pd.DataFrame]) -> List[ScanAlert]:
        """
        Analyze broad market conditions and regime.
        """
        alerts = []
        
        # Check VIX levels if available
        if 'VIX' in market_data:
            vix_data = market_data['VIX']
            if len(vix_data) >= 20:
                current_vix = vix_data['close'].iloc[-1]
                vix_ma = vix_data['close'].iloc[-20:].mean()
                
                if current_vix > vix_ma * 1.5:
                    score = min((current_vix / vix_ma - 1), 1.0) * self.alert_thresholds['regime_change']
                    
                    alert = self.create_alert(
                        symbol='MARKET',
                        alert_type=AlertType.REGIME_CHANGE,
                        score=score,
                        metadata={
                            'indicator': 'VIX',
                            'current_level': current_vix,
                            'average_level': vix_ma,
                            'regime': 'high_volatility',
                            'reason': f"VIX spike to {current_vix:.1f} ({current_vix/vix_ma:.1f}x average)",
                            'catalyst_type': 'volatility_regime',
                            'raw_score': score * 5.0  # Legacy scale
                        }
                    )
                    alerts.append(alert)
                    
                    # Record metric
                    if self.metrics:
                        self.metrics.record_alert_generated(
                            self.name,
                            AlertType.REGIME_CHANGE,
                            'MARKET',
                            score
                        )
        
        # Check bond-stock correlation
        if 'SPY' in market_data and 'TLT' in market_data:
            spy_data = market_data['SPY']
            tlt_data = market_data['TLT']
            
            if len(spy_data) >= 20 and len(tlt_data) >= 20:
                spy_returns = spy_data['returns'].iloc[-20:]
                tlt_returns = tlt_data['returns'].iloc[-20:]
                
                # Align data
                common_dates = spy_returns.index.intersection(tlt_returns.index)
                if len(common_dates) >= 20:
                    correlation = spy_returns.loc[common_dates].corr(tlt_returns.loc[common_dates])
                    
                    # Positive correlation unusual (normally negative)
                    if correlation > 0.3:
                        score = correlation * self.alert_thresholds['regime_change']
                        
                        alert = self.create_alert(
                            symbol='MARKET',
                            alert_type=AlertType.REGIME_CHANGE,
                            score=score,
                            metadata={
                                'relationship': 'stock_bond_correlation',
                                'correlation': correlation,
                                'regime': 'risk_off',
                                'reason': f"Unusual positive stock-bond correlation: {correlation:.2f}",
                                'catalyst_type': 'correlation_regime',
                                'raw_score': score * 5.0  # Legacy scale
                            }
                        )
                        alerts.append(alert)
                        
                        # Record metric
                        if self.metrics:
                            self.metrics.record_alert_generated(
                                self.name,
                                AlertType.REGIME_CHANGE,
                                'MARKET',
                                score
                            )
        
        return alerts