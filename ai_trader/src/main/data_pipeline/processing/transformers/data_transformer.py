"""
Unified Data Transformer

Main transformer that leverages all utils for data transformation.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from main.utils.data import (
    ProcessingUtils as DataProcessingUtils,
    ValidationUtils,
    DataFrameStreamer,
    StreamingConfig,
    get_global_processor,
    get_global_validator
)
from main.utils.processing import StreamingConfig as ProcessingStreamConfig
from main.utils.core import get_logger, ensure_utc, ErrorHandlingMixin
from main.utils.monitoring import timer, record_metric, MetricType
from main.data_pipeline.core.enums import DataLayer, DataType
from .base_transformer import BaseTransformer


class DataTransformer(BaseTransformer):
    """
    Unified data transformer using utils for all operations.
    
    Handles market data, news, fundamentals, and corporate actions
    with layer-aware processing rules.
    """
    
    def __init__(self):
        """Initialize data transformer with utils."""
        super().__init__()
        self.processor = get_global_processor()
        self.validator = get_global_validator()
        self.streamer = DataFrameStreamer()
        
    async def transform(
        self,
        data: Any,
        source_format: str,
        target_format: str,
        layer: DataLayer,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Transform data with layer-aware processing.
        
        Args:
            data: Input data to transform
            source_format: Source data format
            target_format: Target data format
            layer: Data layer for processing rules
            context: Optional context (symbol, data_type, etc.)
        
        Returns:
            Transformed data
        """
        context = context or {}
        symbol = context.get('symbol', 'UNKNOWN')
        data_type = context.get('data_type', DataType.MARKET_DATA)
        
        with timer("transform.process", tags={
            "source": source_format,
            "target": target_format,
            "layer": layer.name,
            "data_type": data_type.value if hasattr(data_type, 'value') else str(data_type)
        }):
            # Convert to DataFrame if needed
            df = self._ensure_dataframe(data, source_format)
            
            if df is None or df.empty:
                self.logger.warning(f"Empty data for {symbol}")
                return pd.DataFrame()
            
            # Record input metrics
            # gauge("transform.input_rows", len(df), tags={"symbol": symbol})
            
            # For large datasets, use streaming
            if len(df) > 100000:
                self.logger.info(f"Using streaming for {len(df)} rows")
                config = StreamingConfig.get_large_dataset_config()
                result = await self._stream_transform(df, layer, context, config)
            else:
                result = await self._standard_transform(df, layer, context)
            
            # Record output metrics
            # gauge("transform.output_rows", len(result), tags={"symbol": symbol})
            record_metric("transform.reduction_rate", 
                         1 - (len(result) / len(df)) if len(df) > 0 else 0,
                         MetricType.GAUGE,
                         tags={"symbol": symbol})
            
            # Update stats
            self._transformation_stats['total_transformations'] += 1
            self._transformation_stats['successful'] += 1
            
            # Convert to target format if needed
            return self._convert_format(result, target_format)
    
    async def _standard_transform(
        self,
        df: pd.DataFrame,
        layer: DataLayer,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """Standard transformation pipeline using utils."""
        # Stage 1: Handle missing values
        df = DataProcessingUtils.handle_missing_values(
            df, 
            method='forward_fill' if layer >= DataLayer.LIQUID else 'drop'
        )
        
        # Stage 2: Remove outliers based on layer
        if layer >= DataLayer.CATALYST:
            # More aggressive outlier removal for higher layers
            df = DataProcessingUtils.remove_outliers(df, method='iqr', threshold=1.5)
        elif layer >= DataLayer.LIQUID:
            df = DataProcessingUtils.remove_outliers(df, method='iqr', threshold=3)
        
        # Stage 3: Normalize data if needed
        if context.get('normalize', False):
            df = DataProcessingUtils.normalize_data(
                df,
                method='robust' if layer >= DataLayer.CATALYST else 'zscore'
            )
        
        # Stage 4: Ensure UTC timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].apply(ensure_utc)
        elif df.index.name == 'timestamp':
            df.index = pd.DatetimeIndex([ensure_utc(ts) for ts in df.index])
        
        # Stage 5: Validate based on data type
        data_type = context.get('data_type', DataType.MARKET_DATA)
        if data_type == DataType.MARKET_DATA:
            is_valid, errors = ValidationUtils.validate_ohlcv_data(df)
            if not is_valid:
                self.logger.warning(f"Validation issues: {errors}")
                record_metric("transform.validation_errors", len(errors), 
                            MetricType.COUNTER, tags={"layer": layer.name})
        
        # Stage 6: Layer-specific enhancements
        if layer >= DataLayer.ACTIVE:
            df = await self._apply_advanced_features(df, context)
        
        return df
    
    async def _stream_transform(
        self,
        df: pd.DataFrame,
        layer: DataLayer,
        context: Dict[str, Any],
        config: StreamingConfig
    ) -> pd.DataFrame:
        """Stream transformation for large datasets."""
        results = []
        
        async def process_chunk(chunk):
            return await self._standard_transform(chunk, layer, context)
        
        # Process in chunks
        for chunk_df in self.streamer.stream(df, process_chunk):
            results.append(chunk_df)
            
            # Track progress
            progress = self.streamer.get_progress()
            if progress % 10 == 0:  # Log every 10%
                self.logger.info(f"Transform progress: {progress:.0f}%")
                # gauge("transform.progress", progress, tags={"symbol": context.get('symbol', 'UNKNOWN')})
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    async def _apply_advanced_features(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply advanced features for ACTIVE layer."""
        # Add derived columns for ACTIVE layer
        if 'close' in df.columns and 'open' in df.columns:
            # Price movement
            df['price_change'] = df['close'] - df['open']
            df['price_change_pct'] = (df['price_change'] / df['open'] * 100).round(2)
            
            # Volatility indicator
            if 'high' in df.columns and 'low' in df.columns:
                df['daily_range'] = df['high'] - df['low']
                df['range_pct'] = (df['daily_range'] / df['open'] * 100).round(2)
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_ma5'] = df['volume'].rolling(window=5, min_periods=1).mean()
            df['volume_ratio'] = (df['volume'] / df['volume_ma5']).round(2)
        
        return df
    
    def _ensure_dataframe(self, data: Any, source_format: str) -> Optional[pd.DataFrame]:
        """Ensure data is in DataFrame format."""
        if isinstance(data, pd.DataFrame):
            return data
        elif source_format == 'dict':
            return pd.DataFrame(data)
        elif source_format == 'list':
            return pd.DataFrame(data)
        elif source_format == 'json':
            import json
            if isinstance(data, str):
                data = json.loads(data)
            return pd.DataFrame(data)
        else:
            self.logger.error(f"Unsupported source format: {source_format}")
            return None
    
    def _convert_format(self, df: pd.DataFrame, target_format: str) -> Any:
        """Convert DataFrame to target format."""
        if target_format == 'pandas' or target_format == 'dataframe':
            return df
        elif target_format == 'dict':
            return df.to_dict('records')
        elif target_format == 'list':
            return df.values.tolist()
        elif target_format == 'json':
            return df.to_json(orient='records')
        else:
            self.logger.warning(f"Unknown target format {target_format}, returning DataFrame")
            return df