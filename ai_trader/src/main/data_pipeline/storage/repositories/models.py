"""
Typed dataclass models for repository data structures.

This module provides strongly-typed alternatives to Dict[str, Any] 
for better type safety and IDE support.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class DataInterval(str, Enum):
    """Market data interval types."""
    TICK = "tick"
    ONE_MIN = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    THIRTY_MIN = "30min"
    ONE_HOUR = "1hour"
    ONE_DAY = "1day"


@dataclass
class PriceData:
    """Market price data point."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trades: Optional[int] = None
    interval: Optional[DataInterval] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap,
            'trades': self.trades,
            'interval': self.interval.value if self.interval else None
        }


@dataclass
class VolumeStatistics:
    """Volume statistics for a symbol."""
    symbol: str
    avg_volume: float
    std_volume: float
    min_volume: int
    max_volume: int
    percentile_90: float
    lookback_days: int
    data_points: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'avg_volume': self.avg_volume,
            'std_volume': self.std_volume,
            'min_volume': self.min_volume,
            'max_volume': self.max_volume,
            'percentile_90': self.percentile_90,
            'lookback_days': self.lookback_days,
            'data_points': self.data_points
        }


@dataclass
class CompanyInfo:
    """Company information."""
    symbol: str
    name: str
    exchange: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    layer: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'exchange': self.exchange,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'layer': self.layer,
            'is_active': self.is_active,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


@dataclass
class NewsArticle:
    """News article data."""
    article_id: str
    symbol: str
    headline: str
    published_at: datetime
    source: str
    url: str
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'article_id': self.article_id,
            'symbol': self.symbol,
            'headline': self.headline,
            'published_at': self.published_at,
            'source': self.source,
            'url': self.url,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'summary': self.summary,
            'keywords': self.keywords
        }


@dataclass
class FinancialStatement:
    """Financial statement data."""
    symbol: str
    period: str  # e.g., "Q1 2024"
    period_end: datetime
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    eps_basic: Optional[float] = None
    eps_diluted: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'period': self.period,
            'period_end': self.period_end,
            'revenue': self.revenue,
            'net_income': self.net_income,
            'eps_basic': self.eps_basic,
            'eps_diluted': self.eps_diluted,
            'gross_margin': self.gross_margin,
            'operating_margin': self.operating_margin,
            'net_margin': self.net_margin,
            'roe': self.roe,
            'roa': self.roa,
            'debt_to_equity': self.debt_to_equity
        }


@dataclass
class ScannerResult:
    """Scanner operation result."""
    scanner_name: str
    symbols_scanned: int
    alerts_generated: int
    duration_ms: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.symbols_scanned == 0:
            return 0.0
        return (self.symbols_scanned - len(self.errors)) / self.symbols_scanned


@dataclass
class BatchOperationResult:
    """Result of a batch database operation."""
    success: bool
    records_processed: int
    records_created: int = 0
    records_updated: int = 0
    records_failed: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'records_processed': self.records_processed,
            'records_created': self.records_created,
            'records_updated': self.records_updated,
            'records_failed': self.records_failed,
            'duration_seconds': self.duration_seconds,
            'errors': self.errors
        }


@dataclass
class QueryParameters:
    """Parameters for database queries."""
    symbol: Optional[str] = None
    symbols: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    interval: Optional[DataInterval] = None
    limit: int = 1000
    offset: int = 0
    order_by: Optional[str] = None
    order_desc: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for query building."""
        params = {}
        if self.symbol:
            params['symbol'] = self.symbol
        if self.symbols:
            params['symbols'] = self.symbols
        if self.start_date:
            params['start_date'] = self.start_date
        if self.end_date:
            params['end_date'] = self.end_date
        if self.interval:
            params['interval'] = self.interval.value
        params['limit'] = self.limit
        params['offset'] = self.offset
        if self.order_by:
            params['order_by'] = self.order_by
        params['order_desc'] = self.order_desc
        return params


@dataclass
class ScannerSummary:
    """Summary of scanner operation."""
    scanner_name: str
    total_alerts: int
    alerts_by_type: Dict[str, int]  # AlertType.value -> count
    processing_time_ms: float
    symbols_scanned: int
    cache_hit: bool = False
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.symbols_scanned == 0:
            return 0.0
        error_symbols = len(self.errors)
        return (self.symbols_scanned - error_symbols) / self.symbols_scanned
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scanner_name': self.scanner_name,
            'total_alerts': self.total_alerts,
            'alerts_by_type': self.alerts_by_type,
            'processing_time_ms': self.processing_time_ms,
            'symbols_scanned': self.symbols_scanned,
            'cache_hit': self.cache_hit,
            'errors': self.errors,
            'success_rate': self.success_rate
        }


@dataclass
class ScannerMetrics:
    """Metrics for scanner performance."""
    scanner_name: str
    scan_count: int
    total_duration_ms: float
    average_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    error_count: int
    error_rate: float
    alerts_generated: int
    avg_alerts_per_scan: float
    last_scan_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scanner_name': self.scanner_name,
            'scan_count': self.scan_count,
            'total_duration_ms': self.total_duration_ms,
            'average_duration_ms': self.average_duration_ms,
            'min_duration_ms': self.min_duration_ms,
            'max_duration_ms': self.max_duration_ms,
            'error_count': self.error_count,
            'error_rate': self.error_rate,
            'alerts_generated': self.alerts_generated,
            'avg_alerts_per_scan': self.avg_alerts_per_scan,
            'last_scan_time': self.last_scan_time
        }


@dataclass
class AffinityScore:
    """Strategy affinity score for a symbol."""
    symbol: str
    momentum_score: float
    mean_reversion_score: float
    volatility_score: float
    breakout_score: float
    trend_score: float
    overall_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'momentum_score': self.momentum_score,
            'mean_reversion_score': self.mean_reversion_score,
            'volatility_score': self.volatility_score,
            'breakout_score': self.breakout_score,
            'trend_score': self.trend_score,
            'overall_score': self.overall_score,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class AlertMetadata:
    """Metadata for scan alerts."""
    volume_ratio: Optional[float] = None
    z_score: Optional[float] = None
    correlation: Optional[float] = None
    divergence_strength: Optional[float] = None
    news_sentiment: Optional[float] = None
    technical_pattern: Optional[str] = None
    sector_strength: Optional[float] = None
    reason: str = ""
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {'reason': self.reason}
        if self.volume_ratio is not None:
            result['volume_ratio'] = self.volume_ratio
        if self.z_score is not None:
            result['z_score'] = self.z_score
        if self.correlation is not None:
            result['correlation'] = self.correlation
        if self.divergence_strength is not None:
            result['divergence_strength'] = self.divergence_strength
        if self.news_sentiment is not None:
            result['news_sentiment'] = self.news_sentiment
        if self.technical_pattern:
            result['technical_pattern'] = self.technical_pattern
        if self.sector_strength is not None:
            result['sector_strength'] = self.sector_strength
        if self.timestamp:
            result['timestamp'] = self.timestamp
        return result