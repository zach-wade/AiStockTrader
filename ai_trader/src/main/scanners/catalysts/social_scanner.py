"""
Social Media Activity Scanner

Scans for unusual social media activity patterns that could signal
trading opportunities or risks.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np

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


class SocialScanner(CatalystScannerBase):
    """
    Scans social media platforms for unusual activity patterns:
    - Sentiment spikes (positive or negative)
    - Volume anomalies
    - Viral content detection
    - Influencer activity
    - Community growth patterns
    
    Now uses the repository pattern with hot/cold storage awareness to
    efficiently access social sentiment data and historical patterns.
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
        Initializes the SocialScanner with dependency injection.

        Args:
            config: Scanner configuration
            repository: Scanner data repository with hot/cold routing
            event_bus: Optional event bus for publishing alerts
            metrics_collector: Optional metrics collector
            cache_manager: Optional cache manager
        """
        super().__init__(
            "SocialScanner",
            config,
            repository,
            event_bus,
            metrics_collector,
            cache_manager
        )
        
        # Scanner-specific parameters
        self.params = self.config.get('scanners.social', {})
        self.lookback_hours = self.params.get('lookback_hours', 24)
        self.min_posts = self.params.get('min_posts', 10)
        self.sentiment_threshold = self.params.get('sentiment_threshold', 0.7)
        self.volume_spike_threshold = self.params.get('volume_spike_threshold', 3.0)
        self.viral_threshold = self.params.get('viral_threshold', 0.6)
        self.use_cache = self.params.get('use_cache', True)
        
        # Platform-specific settings
        self.platform_settings = self.params.get('platforms', {
            'reddit': {'weight': 0.4, 'min_score': 50},
            'twitter': {'weight': 0.3, 'min_engagement': 100},
            'stocktwits': {'weight': 0.2, 'min_volume': 20},
            'discord': {'weight': 0.1, 'min_activity': 10}
        })
        
        # Alert thresholds
        self.alert_thresholds = self.params.get('alert_thresholds', {
            'extreme': 0.9,
            'high': 0.7,
            'moderate': 0.5
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
        Scan symbols for social media activity patterns.
        
        Uses repository pattern for efficient social data access with hot storage
        for recent sentiment and cold storage for historical patterns.
        
        Args:
            symbols: List of stock symbols to scan
            **kwargs: Additional scanner parameters
            
        Returns:
            List of ScanAlert objects for detected patterns
        """
        if not self._initialized:
            await self.initialize()
            
        with timer() as t:
            logger.info(f"👥 Social Scanner: Analyzing {len(symbols)} symbols...")
            
            # Start metrics tracking
            if self.metrics:
                scan_start = datetime.now(timezone.utc)
            
            try:
                # Check cache if enabled
                if self.cache and self.use_cache:
                    cache_key = f"social_scan:{','.join(sorted(symbols[:10]))}:{self.lookback_hours}"
                    cached_alerts = await self.cache.get_cached_result(
                        self.name,
                        "batch",
                        cache_key
                    )
                    if cached_alerts is not None:
                        logger.info(f"Using cached results for social scan")
                        return cached_alerts
                
                # Build query filter for social data
                query_filter = QueryFilter(
                    symbols=symbols,
                    start_date=datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours),
                    end_date=datetime.now(timezone.utc)
                )
                
                # Get social sentiment data from repository
                # This will use hot storage for recent data, cold for historical
                social_data = await self.repository.get_social_sentiment(
                    symbols=symbols,
                    query_filter=query_filter
                )
                
                if not social_data:
                    logger.info("No social data available")
                    return []
                
                alerts = []
                for symbol, symbol_social_data in social_data.items():
                    if not symbol_social_data:
                        continue
                    
                    # Process social data for this symbol
                    symbol_alerts = await self._process_symbol_social_data(
                        symbol,
                        symbol_social_data
                    )
                    alerts.extend(symbol_alerts)
                
                # Deduplicate alerts
                alerts = self.deduplicate_alerts(alerts)
                
                # Cache results if enabled
                if self.cache and self.use_cache and alerts:
                    await self.cache.cache_result(
                        self.name,
                        "batch",
                        cache_key,
                        alerts,
                        ttl_seconds=1800  # 30 minute TTL for social data
                    )
                
                # Publish alerts to event bus
                await self.publish_alerts_to_event_bus(alerts, self.event_bus)
                
                logger.info(
                    f"✅ Social Scanner: Found {len(alerts)} alerts "
                    f"in {t.elapsed * 1000:.2f}ms"
                )
                
                # Record metrics
                if self.metrics:
                    self.metrics.record_scan_duration(
                        self.name,
                        t.elapsed * 1000,
                        len(symbols)
                    )
                
                return alerts
                
            except Exception as e:
                logger.error(f"❌ Error in Social Scanner: {e}", exc_info=True)
                
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
            universe: The list of symbols to scan.

        Returns:
            A dictionary mapping symbols to a list of their social catalyst signals.
        """
        # Use the new scan method
        alerts = await self.scan(universe)
        
        # Convert to legacy format
        catalyst_signals = defaultdict(list)
        for alert in alerts:
            signal = {
                'score': alert.metadata.get('raw_score', alert.score * 5.0),  # Convert from 0-1 to 0-5 scale
                'reason': alert.metadata.get('reason', ''),
                'signal_type': 'social',
                'metadata': {
                    'sentiment': alert.metadata.get('sentiment'),
                    'volume_spike': alert.metadata.get('volume_spike'),
                    'catalyst_type': alert.metadata.get('catalyst_type')
                }
            }
            catalyst_signals[alert.symbol].append(signal)
        
        return dict(catalyst_signals)

    async def _process_symbol_social_data(
        self,
        symbol: str,
        social_data: List[Dict[str, Any]]
    ) -> List[ScanAlert]:
        """
        Process social sentiment data for a symbol and generate alerts.
        
        Args:
            symbol: Stock symbol
            social_data: List of social sentiment data points
            
        Returns:
            List of alerts for this symbol
        """
        if not social_data:
            return []
        
        # Aggregate social data metrics
        aggregated_data = self._aggregate_social_data(social_data)
        
        # Calculate metrics
        metrics = self._calculate_social_metrics(aggregated_data)
        
        alerts = []
        
        # Check for sentiment extremes
        sentiment_alert = self._check_sentiment_extreme(symbol, metrics)
        if sentiment_alert:
            alerts.append(sentiment_alert)
        
        # Check for volume spikes
        volume_alert = self._check_volume_spike(symbol, metrics, aggregated_data)
        if volume_alert:
            alerts.append(volume_alert)
        
        # Check for viral patterns
        viral_alert = self._check_viral_pattern(symbol, metrics, aggregated_data)
        if viral_alert:
            alerts.append(viral_alert)
        
        # Check for coordinated activity
        coordinated_alert = self._check_coordinated_activity(symbol, aggregated_data)
        if coordinated_alert:
            alerts.append(coordinated_alert)
        
        return alerts
    
    def _aggregate_social_data(self, social_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate social data from repository format into internal format.
        """
        aggregated = {
            'sentiment_scores': [],
            'post_volumes': [],
            'engagement_metrics': [],
            'timestamps': [],
            'platforms': defaultdict(list)
        }
        
        for item in social_data:
            # Extract sentiment score
            sentiment = item.get('sentiment_score', 0.5)
            aggregated['sentiment_scores'].append(sentiment)
            
            # Extract engagement metrics
            engagement = item.get('engagement_score', 0)
            aggregated['engagement_metrics'].append(engagement)
            
            # Extract timestamp
            timestamp = item.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                aggregated['timestamps'].append(timestamp)
            
            # Group by platform
            platform = item.get('platform', 'unknown')
            aggregated['platforms'][platform].append(item)
        
        # Calculate hourly volumes
        if aggregated['timestamps']:
            aggregated['post_volumes'] = self._calculate_hourly_volumes(aggregated['timestamps'])
        
        return aggregated
    
    def _calculate_hourly_volumes(self, timestamps: List[datetime]) -> Dict[str, int]:
        """
        Calculate post volumes by hour.
        """
        hourly_volumes = defaultdict(int)
        
        for ts in timestamps:
            hour_key = ts.strftime('%Y-%m-%d %H:00')
            hourly_volumes[hour_key] += 1
        
        return dict(hourly_volumes)
    
    def _calculate_social_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate aggregate social metrics.
        """
        metrics = {}
        
        # Sentiment metrics
        if data['sentiment_scores']:
            metrics['avg_sentiment'] = np.mean(data['sentiment_scores'])
            metrics['sentiment_std'] = np.std(data['sentiment_scores'])
            metrics['sentiment_trend'] = self._calculate_trend(
                data['sentiment_scores']
            )
            
            # Recent vs historical sentiment
            recent_count = min(10, len(data['sentiment_scores']) // 4)
            if recent_count > 0:
                recent_sentiment = np.mean(data['sentiment_scores'][-recent_count:])
                historical_sentiment = np.mean(data['sentiment_scores'][:-recent_count])
                metrics['sentiment_delta'] = recent_sentiment - historical_sentiment
            else:
                metrics['sentiment_delta'] = 0
        
        # Engagement metrics
        if data['engagement_metrics']:
            metrics['total_engagement'] = sum(data['engagement_metrics'])
            metrics['avg_engagement'] = np.mean(data['engagement_metrics'])
            metrics['max_engagement'] = max(data['engagement_metrics'])
            
            # Viral coefficient (high engagement posts / total posts)
            high_engagement_threshold = np.percentile(data['engagement_metrics'], 75)
            viral_posts = sum(1 for e in data['engagement_metrics'] if e > high_engagement_threshold)
            metrics['viral_coefficient'] = viral_posts / len(data['engagement_metrics'])
        
        # Volume metrics
        if data['post_volumes']:
            volumes = list(data['post_volumes'].values())
            if len(volumes) > 1:
                metrics['volume_mean'] = np.mean(volumes)
                metrics['volume_std'] = np.std(volumes)
                metrics['recent_volume'] = volumes[-1] if volumes else 0
                metrics['volume_spike_ratio'] = (
                    metrics['recent_volume'] / metrics['volume_mean']
                    if metrics['volume_mean'] > 0 else 1.0
                )
            else:
                metrics['volume_spike_ratio'] = 1.0
        
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend direction (-1 to 1).
        """
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize to -1 to 1 range
        return np.tanh(slope * 10)
    
    def _check_sentiment_extreme(self,
                                symbol: str,
                                metrics: Dict[str, float]) -> Optional[ScanAlert]:
        """
        Check for extreme sentiment levels.
        """
        avg_sentiment = metrics.get('avg_sentiment', 0.5)
        sentiment_delta = metrics.get('sentiment_delta', 0)
        
        # Check for extreme positive sentiment
        if avg_sentiment > self.sentiment_threshold:
            score = avg_sentiment
            
            # Boost score for rapid sentiment increase
            if sentiment_delta > 0.2:
                score = min(score * 1.2, 1.0)
            
            if score > self.alert_thresholds['moderate']:
                alert_type = AlertType.SOCIAL_SENTIMENT
                
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=alert_type,
                    score=score,
                    metadata={
                        'sentiment': avg_sentiment,
                        'sentiment_delta': sentiment_delta,
                        'direction': 'bullish',
                        'reason': f"Extreme positive sentiment: {avg_sentiment:.2%}",
                        'catalyst_type': 'social_sentiment',
                        'raw_score': score * 5.0  # Legacy scale
                    }
                )
                
                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name,
                        alert_type,
                        symbol,
                        score
                    )
                
                return alert
        
        # Check for extreme negative sentiment
        elif avg_sentiment < (1 - self.sentiment_threshold):
            score = 1 - avg_sentiment
            
            # Boost score for rapid sentiment decrease
            if sentiment_delta < -0.2:
                score = min(score * 1.2, 1.0)
            
            if score > self.alert_thresholds['moderate']:
                alert_type = AlertType.SOCIAL_SENTIMENT
                
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=alert_type,
                    score=score,
                    metadata={
                        'sentiment': avg_sentiment,
                        'sentiment_delta': sentiment_delta,
                        'direction': 'bearish',
                        'reason': f"Extreme negative sentiment: {avg_sentiment:.2%}",
                        'catalyst_type': 'social_sentiment',
                        'raw_score': score * 5.0  # Legacy scale
                    }
                )
                
                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name,
                        alert_type,
                        symbol,
                        score
                    )
                
                return alert
        
        return None
    
    def _check_volume_spike(self,
                           symbol: str,
                           metrics: Dict[str, float],
                           data: Dict[str, Any]) -> Optional[ScanAlert]:
        """
        Check for unusual post volume spikes.
        """
        volume_spike_ratio = metrics.get('volume_spike_ratio', 1.0)
        
        if volume_spike_ratio > self.volume_spike_threshold:
            # Calculate alert score based on spike magnitude
            score = min(
                (volume_spike_ratio - 1) / (self.volume_spike_threshold * 2),
                1.0
            )
            
            if score > self.alert_thresholds['moderate']:
                # Determine sentiment context
                avg_sentiment = metrics.get('avg_sentiment', 0.5)
                if avg_sentiment > 0.6:
                    context = 'bullish'
                elif avg_sentiment < 0.4:
                    context = 'bearish'
                else:
                    context = 'mixed'
                
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.SOCIAL_VOLUME,
                    score=score,
                    metadata={
                        'volume_spike': volume_spike_ratio,
                        'post_count': len(data.get('sentiment_scores', [])),
                        'sentiment_context': context,
                        'reason': f"Social volume spike: {volume_spike_ratio:.1f}x normal",
                        'catalyst_type': 'social_volume',
                        'raw_score': score * 5.0  # Legacy scale
                    }
                )
                
                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name,
                        AlertType.SOCIAL_VOLUME,
                        symbol,
                        score
                    )
                
                return alert
        
        return None
    
    def _check_viral_pattern(self,
                            symbol: str,
                            metrics: Dict[str, float],
                            data: Dict[str, Any]) -> Optional[ScanAlert]:
        """
        Check for viral content patterns.
        """
        viral_coefficient = metrics.get('viral_coefficient', 0)
        max_engagement = metrics.get('max_engagement', 0)
        avg_engagement = metrics.get('avg_engagement', 1)
        
        if viral_coefficient > self.viral_threshold:
            # High percentage of posts going viral
            engagement_ratio = max_engagement / avg_engagement if avg_engagement > 0 else 1
            
            # Score based on viral coefficient and engagement ratio
            score = min(
                viral_coefficient * 0.6 + min(engagement_ratio / 10, 0.4),
                1.0
            )
            
            if score > self.alert_thresholds['moderate']:
                alert = self.create_alert(
                    symbol=symbol,
                    alert_type=AlertType.SOCIAL_VIRAL,
                    score=score,
                    metadata={
                        'viral_coefficient': viral_coefficient,
                        'max_engagement': max_engagement,
                        'engagement_ratio': engagement_ratio,
                        'reason': f"Viral pattern detected: {viral_coefficient:.1%} posts trending",
                        'catalyst_type': 'social_viral',
                        'raw_score': score * 5.0  # Legacy scale
                    }
                )
                
                # Record metric
                if self.metrics:
                    self.metrics.record_alert_generated(
                        self.name,
                        AlertType.SOCIAL_VIRAL,
                        symbol,
                        score
                    )
                
                return alert
        
        return None
    
    def _check_coordinated_activity(self,
                                   symbol: str,
                                   data: Dict[str, Any]) -> Optional[ScanAlert]:
        """
        Check for signs of coordinated social media activity.
        """
        if not data['timestamps'] or len(data['timestamps']) < self.min_posts:
            return None
        
        # Sort timestamps
        timestamps = sorted(data['timestamps'])
        
        # Calculate time gaps between posts
        time_gaps = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            time_gaps.append(gap)
        
        if not time_gaps:
            return None
        
        # Look for suspicious patterns
        avg_gap = np.mean(time_gaps)
        std_gap = np.std(time_gaps)
        
        # Check for burst patterns (many posts in short time)
        short_gaps = sum(1 for gap in time_gaps if gap < 60)  # Posts within 1 minute
        burst_ratio = short_gaps / len(time_gaps)
        
        if burst_ratio > 0.3:  # 30% of posts within 1 minute of each other
            score = min(burst_ratio * 1.5, 0.9)
            
            alert = self.create_alert(
                symbol=symbol,
                alert_type=AlertType.UNUSUAL_ACTIVITY,
                score=score,
                metadata={
                    'burst_ratio': burst_ratio,
                    'post_count': len(timestamps),
                    'avg_gap_seconds': avg_gap,
                    'reason': f"Coordinated activity suspected: {burst_ratio:.1%} burst posts",
                    'catalyst_type': 'coordinated_social',
                    'raw_score': score * 5.0  # Legacy scale
                }
            )
            
            # Record metric
            if self.metrics:
                self.metrics.record_alert_generated(
                    self.name,
                    AlertType.UNUSUAL_ACTIVITY,
                    symbol,
                    score
                )
            
            return alert
        
        return None