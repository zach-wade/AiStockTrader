"""
Data Pipeline Processing Interfaces

Interfaces for data processing components including transformation,
standardization, cleaning, and feature building with layer awareness.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd

from main.data_pipeline.core.enums import DataLayer, DataType, ProcessingPriority


class IDataTransformer(ABC):
    """Interface for data transformation operations."""
    
    @abstractmethod
    async def transform(
        self,
        data: Any,
        source_format: str,
        target_format: str,
        layer: DataLayer,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Transform data from source to target format."""
        pass
    
    @abstractmethod
    async def validate_transformation(
        self,
        original_data: Any,
        transformed_data: Any,
        layer: DataLayer
    ) -> Dict[str, Any]:
        """Validate transformation results."""
        pass
    
    @abstractmethod
    async def get_supported_transformations(self) -> Dict[str, List[str]]:
        """Get supported transformation mappings."""
        pass
    
    @abstractmethod
    async def get_transformation_stats(self) -> Dict[str, Any]:
        """Get transformation statistics."""
        pass


class IDataStandardizer(ABC):
    """Interface for data standardization operations."""
    
    @abstractmethod
    async def standardize(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        source: str,
        layer: DataLayer,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Standardize data according to layer-specific rules."""
        pass
    
    @abstractmethod
    async def standardize_columns(
        self,
        data: pd.DataFrame,
        source: str,
        layer: DataLayer
    ) -> pd.DataFrame:
        """Standardize column names and types."""
        pass
    
    @abstractmethod
    async def standardize_timestamps(
        self,
        data: pd.DataFrame,
        layer: DataLayer
    ) -> pd.DataFrame:
        """Standardize timestamp formats."""
        pass
    
    @abstractmethod
    async def standardize_symbols(
        self,
        data: pd.DataFrame,
        layer: DataLayer
    ) -> pd.DataFrame:
        """Standardize symbol formats."""
        pass
    
    @abstractmethod
    async def get_standardization_rules(self, layer: DataLayer) -> Dict[str, Any]:
        """Get standardization rules for a layer."""
        pass


class IDataCleaner(ABC):
    """Interface for data cleaning operations."""
    
    @abstractmethod
    async def clean(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        layer: DataLayer,
        cleaning_profile: Optional[str] = None
    ) -> pd.DataFrame:
        """Clean data according to layer-specific rules."""
        pass
    
    @abstractmethod
    async def remove_duplicates(
        self,
        data: pd.DataFrame,
        layer: DataLayer,
        dedup_strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """Remove duplicate records."""
        pass
    
    @abstractmethod
    async def handle_missing_values(
        self,
        data: pd.DataFrame,
        layer: DataLayer,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """Handle missing values."""
        pass
    
    @abstractmethod
    async def detect_outliers(
        self,
        data: pd.DataFrame,
        layer: DataLayer,
        method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect outliers in data."""
        pass
    
    @abstractmethod
    async def clean_outliers(
        self,
        data: pd.DataFrame,
        outlier_info: Dict[str, Any],
        layer: DataLayer,
        action: str = "flag"
    ) -> pd.DataFrame:
        """Clean outliers from data."""
        pass
    
    @abstractmethod
    async def get_cleaning_stats(self) -> Dict[str, Any]:
        """Get data cleaning statistics."""
        pass


class IFeatureBuilder(ABC):
    """Interface for feature building operations."""
    
    @abstractmethod
    async def build_features(
        self,
        data: pd.DataFrame,
        feature_set: str,
        layer: DataLayer,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Build features for a specific feature set."""
        pass
    
    @abstractmethod
    async def build_technical_features(
        self,
        market_data: pd.DataFrame,
        layer: DataLayer,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Build technical analysis features."""
        pass
    
    @abstractmethod
    async def build_fundamental_features(
        self,
        fundamental_data: pd.DataFrame,
        layer: DataLayer,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Build fundamental analysis features."""
        pass
    
    @abstractmethod  
    async def build_sentiment_features(
        self,
        text_data: pd.DataFrame,
        layer: DataLayer,
        sentiment_models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Build sentiment analysis features."""
        pass
    
    @abstractmethod
    async def validate_features(
        self,
        features: pd.DataFrame,
        feature_set: str,
        layer: DataLayer
    ) -> Dict[str, Any]:
        """Validate built features."""
        pass
    
    @abstractmethod
    async def get_available_features(self, layer: DataLayer) -> Dict[str, List[str]]:
        """Get available features for a layer."""
        pass
    
    @abstractmethod
    async def get_feature_dependencies(self, feature_set: str) -> Dict[str, List[str]]:
        """Get feature dependencies."""
        pass


class IDataAnalyzer(ABC):
    """Interface for data analysis operations."""
    
    @abstractmethod
    async def analyze_data_quality(
        self,
        data: pd.DataFrame,
        data_type: DataType,
        layer: DataLayer
    ) -> Dict[str, Any]:
        """Analyze data quality metrics."""
        pass
    
    @abstractmethod
    async def analyze_data_completeness(
        self,
        data: pd.DataFrame,
        expected_schema: Dict[str, Any],
        layer: DataLayer
    ) -> Dict[str, Any]:
        """Analyze data completeness."""
        pass
    
    @abstractmethod
    async def analyze_data_freshness(
        self,
        data: pd.DataFrame,
        layer: DataLayer
    ) -> Dict[str, Any]:
        """Analyze data freshness."""
        pass
    
    @abstractmethod
    async def generate_data_profile(
        self,
        data: pd.DataFrame,
        layer: DataLayer
    ) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        pass
    
    @abstractmethod
    async def compare_data_profiles(
        self,
        profile1: Dict[str, Any],
        profile2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two data profiles."""
        pass


class IStreamProcessor(ABC):
    """Interface for stream processing operations."""
    
    @abstractmethod
    async def start_stream(
        self,
        stream_config: Dict[str, Any],
        layer: DataLayer
    ) -> str:
        """Start a data stream."""
        pass
    
    @abstractmethod
    async def stop_stream(self, stream_id: str) -> bool:
        """Stop a data stream."""
        pass
    
    @abstractmethod
    async def process_stream_batch(
        self,
        stream_id: str,
        batch_data: Any,
        layer: DataLayer
    ) -> Dict[str, Any]:
        """Process a batch from a stream."""
        pass
    
    @abstractmethod
    async def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """Get status of a stream."""
        pass
    
    @abstractmethod
    async def get_active_streams(self) -> List[Dict[str, Any]]:
        """Get all active streams."""
        pass


class IBatchProcessor(ABC):
    """Interface for batch processing operations."""
    
    @abstractmethod
    async def process_batch(
        self,
        batch_data: Any,
        batch_config: Dict[str, Any],
        layer: DataLayer,
        priority: ProcessingPriority = ProcessingPriority.NORMAL
    ) -> Dict[str, Any]:
        """Process a data batch."""
        pass
    
    @abstractmethod
    async def schedule_batch(
        self,
        batch_config: Dict[str, Any],
        layer: DataLayer,
        schedule_time: Optional[datetime] = None
    ) -> str:
        """Schedule a batch for processing."""
        pass
    
    @abstractmethod
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status of a batch."""
        pass
    
    @abstractmethod
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a scheduled batch."""
        pass
    
    @abstractmethod
    async def get_batch_queue(self, layer: Optional[DataLayer] = None) -> List[Dict[str, Any]]:
        """Get the batch processing queue."""
        pass