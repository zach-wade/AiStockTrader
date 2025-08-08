"""
Layer Metrics Collector

Provides monitoring and metrics collection for layer-based operations.
Tracks API calls, rate limit usage, and layer transitions.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone

from main.utils.core import get_logger
from main.utils.monitoring import record_metric, MetricType
from main.data_pipeline.core.enums import DataLayer


logger = get_logger(__name__)


class LayerMetricsCollector:
    """
    Collect and report metrics for layer-based operations.
    
    Provides centralized metrics collection for:
    - API calls per layer
    - Rate limit usage
    - Layer promotions/demotions
    - Performance metrics per layer
    """
    
    @staticmethod
    def record_api_call(
        layer: DataLayer, 
        endpoint: str, 
        success: bool,
        duration_ms: Optional[float] = None,
        symbol: Optional[str] = None
    ) -> None:
        """
        Record an API call for a specific layer.
        
        Args:
            layer: The data layer making the call
            endpoint: API endpoint being called
            success: Whether the call was successful
            duration_ms: Optional call duration in milliseconds
            symbol: Optional symbol being queried
        """
        tags = {
            "layer": layer.name,
            "layer_value": str(layer.value),
            "endpoint": endpoint,
            "success": str(success).lower()
        }
        
        if symbol:
            tags["symbol"] = symbol
        
        # Record the call count
        record_metric(
            "layer.api_calls",
            1,
            MetricType.COUNTER,
            tags=tags
        )
        
        # Record duration if provided
        if duration_ms is not None:
            record_metric(
                "layer.api_call_duration_ms",
                duration_ms,
                MetricType.HISTOGRAM,
                tags=tags
            )
        
        # Log for debugging
        logger.debug(
            f"API call recorded - Layer: {layer.name}, "
            f"Endpoint: {endpoint}, Success: {success}"
        )
    
    @staticmethod
    def record_rate_limit_usage(
        layer: DataLayer, 
        usage_percent: float,
        calls_made: int,
        calls_limit: int
    ) -> None:
        """
        Record rate limit usage percentage for a layer.
        
        Args:
            layer: The data layer
            usage_percent: Percentage of rate limit used (0-100)
            calls_made: Number of calls made
            calls_limit: Maximum calls allowed
        """
        tags = {
            "layer": layer.name,
            "layer_value": str(layer.value)
        }
        
        # Record usage percentage as gauge
        record_metric(
            "layer.rate_limit_usage_percent",
            usage_percent,
            MetricType.GAUGE,
            tags=tags
        )
        
        # Record actual vs limit
        record_metric(
            "layer.rate_limit_calls_made",
            calls_made,
            MetricType.GAUGE,
            tags=tags
        )
        
        record_metric(
            "layer.rate_limit_calls_limit",
            calls_limit,
            MetricType.GAUGE,
            tags=tags
        )
        
        # Warn if usage is high
        if usage_percent > 80:
            logger.warning(
                f"High rate limit usage for layer {layer.name}: "
                f"{usage_percent:.1f}% ({calls_made}/{calls_limit})"
            )
    
    @staticmethod
    def record_layer_promotion(
        symbol: str,
        from_layer: DataLayer,
        to_layer: DataLayer,
        reason: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a symbol promotion to a higher layer.
        
        Args:
            symbol: Symbol being promoted
            from_layer: Original layer
            to_layer: New layer
            reason: Reason for promotion
            metrics: Optional metrics that triggered promotion
        """
        tags = {
            "symbol": symbol,
            "from_layer": from_layer.name,
            "to_layer": to_layer.name,
            "from_value": str(from_layer.value),
            "to_value": str(to_layer.value)
        }
        
        # Record the promotion event
        record_metric(
            "layer.promotions",
            1,
            MetricType.COUNTER,
            tags=tags
        )
        
        # Record layer change magnitude
        layer_change = to_layer.value - from_layer.value
        record_metric(
            "layer.promotion_magnitude",
            layer_change,
            MetricType.HISTOGRAM,
            tags=tags
        )
        
        # Structured logging for analysis
        logger.info(
            "LAYER_PROMOTION",
            extra={
                "event_type": "layer_promotion",
                "symbol": symbol,
                "from_layer": from_layer.name,
                "from_layer_value": from_layer.value,
                "to_layer": to_layer.name,
                "to_layer_value": to_layer.value,
                "reason": reason,
                "metrics": metrics or {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @staticmethod
    def record_layer_demotion(
        symbol: str,
        from_layer: DataLayer,
        to_layer: DataLayer,
        reason: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a symbol demotion to a lower layer.
        
        Args:
            symbol: Symbol being demoted
            from_layer: Original layer
            to_layer: New layer
            reason: Reason for demotion
            metrics: Optional metrics that triggered demotion
        """
        tags = {
            "symbol": symbol,
            "from_layer": from_layer.name,
            "to_layer": to_layer.name,
            "from_value": str(from_layer.value),
            "to_value": str(to_layer.value)
        }
        
        # Record the demotion event
        record_metric(
            "layer.demotions",
            1,
            MetricType.COUNTER,
            tags=tags
        )
        
        # Record layer change magnitude (negative for demotion)
        layer_change = from_layer.value - to_layer.value
        record_metric(
            "layer.demotion_magnitude",
            layer_change,
            MetricType.HISTOGRAM,
            tags=tags
        )
        
        # Structured logging for analysis
        logger.warning(
            "LAYER_DEMOTION",
            extra={
                "event_type": "layer_demotion",
                "symbol": symbol,
                "from_layer": from_layer.name,
                "from_layer_value": from_layer.value,
                "to_layer": to_layer.name,
                "to_layer_value": to_layer.value,
                "reason": reason,
                "metrics": metrics or {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @staticmethod
    def record_layer_capacity(layer: DataLayer, current_symbols: int) -> None:
        """
        Record current capacity usage for a layer.
        
        Args:
            layer: The data layer
            current_symbols: Current number of symbols in the layer
        """
        max_symbols = layer.max_symbols
        usage_percent = (current_symbols / max_symbols * 100) if max_symbols > 0 else 0
        
        tags = {
            "layer": layer.name,
            "layer_value": str(layer.value)
        }
        
        # Record current symbol count
        record_metric(
            "layer.symbol_count",
            current_symbols,
            MetricType.GAUGE,
            tags=tags
        )
        
        # Record max capacity
        record_metric(
            "layer.symbol_capacity",
            max_symbols,
            MetricType.GAUGE,
            tags=tags
        )
        
        # Record usage percentage
        record_metric(
            "layer.capacity_usage_percent",
            usage_percent,
            MetricType.GAUGE,
            tags=tags
        )
        
        # Warn if near capacity
        if usage_percent > 90:
            logger.warning(
                f"Layer {layer.name} near capacity: "
                f"{current_symbols}/{max_symbols} ({usage_percent:.1f}%)"
            )
    
    @staticmethod
    def record_data_processing(
        layer: DataLayer,
        data_type: str,
        records_processed: int,
        duration_ms: float,
        success: bool = True
    ) -> None:
        """
        Record data processing metrics for a layer.
        
        Args:
            layer: The data layer
            data_type: Type of data processed (market_data, news, etc.)
            records_processed: Number of records processed
            duration_ms: Processing duration in milliseconds
            success: Whether processing was successful
        """
        tags = {
            "layer": layer.name,
            "layer_value": str(layer.value),
            "data_type": data_type,
            "success": str(success).lower()
        }
        
        # Record records processed
        record_metric(
            "layer.records_processed",
            records_processed,
            MetricType.COUNTER,
            tags=tags
        )
        
        # Record processing time
        record_metric(
            "layer.processing_duration_ms",
            duration_ms,
            MetricType.HISTOGRAM,
            tags=tags
        )
        
        # Calculate and record throughput
        if duration_ms > 0:
            throughput = (records_processed / duration_ms) * 1000  # records per second
            record_metric(
                "layer.processing_throughput",
                throughput,
                MetricType.GAUGE,
                tags=tags
            )
        
        logger.debug(
            f"Data processing recorded - Layer: {layer.name}, "
            f"Type: {data_type}, Records: {records_processed}, "
            f"Duration: {duration_ms:.2f}ms"
        )