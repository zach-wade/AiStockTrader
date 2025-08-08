"""
Validation Framework - Validator Interfaces

Specific validator interfaces for different data validation types.
These interfaces extend the base IValidator interface for specialized use cases.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd

from main.data_pipeline.core.enums import DataLayer, DataType
from main.interfaces.data_pipeline.validation import IValidator, IValidationResult, IValidationContext


class IRecordValidator(IValidator):
    """Interface for record-level validation."""
    
    @abstractmethod
    async def validate_single_record(
        self,
        record: Dict[str, Any],
        record_index: int,
        context: IValidationContext
    ) -> IValidationResult:
        """Validate a single record."""
        pass
    
    @abstractmethod
    async def validate_record_batch(
        self,
        records: List[Dict[str, Any]],
        context: IValidationContext
    ) -> List[IValidationResult]:
        """Validate a batch of records."""
        pass
    
    @abstractmethod
    async def validate_required_fields(
        self,
        record: Dict[str, Any],
        required_fields: List[str],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate required fields are present and valid."""
        pass
    
    @abstractmethod
    async def validate_field_types(
        self,
        record: Dict[str, Any],
        field_types: Dict[str, type],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate field data types."""
        pass
    
    @abstractmethod
    async def validate_field_ranges(
        self,
        record: Dict[str, Any],
        field_ranges: Dict[str, Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate field values are within acceptable ranges."""
        pass
    
    @abstractmethod
    async def apply_field_mapping(
        self,
        record: Dict[str, Any],
        source: str,
        field_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Apply source-specific field mapping to record."""
        pass


class IFeatureValidator(IValidator):
    """Interface for feature-level validation."""
    
    @abstractmethod
    async def validate_feature_dataframe(
        self,
        features: pd.DataFrame,
        context: IValidationContext
    ) -> IValidationResult:
        """Validate a feature DataFrame."""
        pass
    
    @abstractmethod
    async def validate_feature_completeness(
        self,
        features: pd.DataFrame,
        required_features: List[str],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate feature completeness."""
        pass
    
    @abstractmethod
    async def validate_feature_distributions(
        self,
        features: pd.DataFrame,
        distribution_constraints: Dict[str, Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate feature distributions are within expected bounds."""
        pass
    
    @abstractmethod
    async def validate_feature_correlations(
        self,
        features: pd.DataFrame,
        correlation_constraints: Dict[str, float],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate feature correlations."""
        pass
    
    @abstractmethod
    async def detect_feature_drift(
        self,
        current_features: pd.DataFrame,
        reference_features: pd.DataFrame,
        context: IValidationContext
    ) -> Dict[str, Any]:
        """Detect feature drift compared to reference data."""
        pass
    
    @abstractmethod
    async def validate_feature_engineering(
        self,
        source_data: pd.DataFrame,
        engineered_features: pd.DataFrame,
        context: IValidationContext
    ) -> IValidationResult:
        """Validate feature engineering results."""
        pass


class IMarketDataValidator(IRecordValidator):
    """Interface for market data validation."""
    
    @abstractmethod
    async def validate_ohlcv_data(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        symbol: Optional[str] = None,
        context: Optional[IValidationContext] = None
    ) -> IValidationResult:
        """Validate OHLCV market data."""
        pass
    
    @abstractmethod
    async def validate_price_consistency(
        self,
        data: pd.DataFrame,
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate price consistency (OHLC relationships)."""
        pass
    
    @abstractmethod
    async def validate_volume_data(
        self,
        data: pd.DataFrame,
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate volume data."""
        pass
    
    @abstractmethod
    async def validate_trading_hours(
        self,
        data: pd.DataFrame,
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate data falls within trading hours."""
        pass
    
    @abstractmethod
    async def validate_corporate_actions(
        self,
        market_data: pd.DataFrame,
        corporate_actions: List[Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate market data against corporate actions."""
        pass


class INewsValidator(IRecordValidator):
    """Interface for news data validation."""
    
    @abstractmethod
    async def validate_news_content(
        self,
        news_records: List[Dict[str, Any]],
        context: IValidationContext
    ) -> IValidationResult:
        """Validate news content and metadata."""
        pass
    
    @abstractmethod
    async def validate_sentiment_scores(
        self,
        news_records: List[Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate sentiment scores are within valid ranges."""
        pass
    
    @abstractmethod
    async def validate_news_duplicates(
        self,
        news_records: List[Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Detect and validate news duplicate handling."""
        pass
    
    @abstractmethod
    async def validate_source_credibility(
        self,
        news_records: List[Dict[str, Any]],
        credibility_scores: Dict[str, float],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate news source credibility."""
        pass


class IFundamentalsValidator(IRecordValidator):
    """Interface for fundamental data validation."""
    
    @abstractmethod
    async def validate_earnings_data(
        self,
        earnings_records: List[Dict[str, Any]],
        context: IValidationContext
    ) -> IValidationResult:
        """Validate earnings data."""
        pass
    
    @abstractmethod
    async def validate_financial_ratios(
        self,
        financial_data: List[Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate financial ratios are reasonable."""
        pass
    
    @abstractmethod
    async def validate_balance_sheet_consistency(
        self,
        balance_sheet_data: List[Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate balance sheet consistency."""
        pass
    
    @abstractmethod
    async def validate_income_statement_consistency(
        self,
        income_data: List[Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate income statement consistency."""
        pass


class ITimeSeriesValidator(IValidator):
    """Interface for time series data validation."""
    
    @abstractmethod
    async def validate_time_series_continuity(
        self,
        data: pd.DataFrame,
        timestamp_column: str,
        context: IValidationContext
    ) -> IValidationResult:
        """Validate time series data continuity."""
        pass
    
    @abstractmethod
    async def validate_frequency_consistency(
        self,
        data: pd.DataFrame,
        expected_frequency: str,
        timestamp_column: str,
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate data frequency consistency."""
        pass
    
    @abstractmethod
    async def detect_time_series_outliers(
        self,
        data: pd.DataFrame,
        value_columns: List[str],
        context: IValidationContext
    ) -> Dict[str, Any]:
        """Detect outliers in time series data."""
        pass
    
    @abstractmethod
    async def validate_seasonality_patterns(
        self,
        data: pd.DataFrame,
        value_column: str,
        context: IValidationContext
    ) -> Dict[str, Any]:
        """Validate expected seasonality patterns."""
        pass


class ISchemaValidator(IValidator):
    """Interface for schema validation."""
    
    @abstractmethod
    async def validate_dataframe_schema(
        self,
        data: pd.DataFrame,
        expected_schema: Dict[str, Any],
        context: IValidationContext
    ) -> IValidationResult:
        """Validate DataFrame against expected schema."""
        pass
    
    @abstractmethod
    async def validate_column_types(
        self,
        data: pd.DataFrame,
        expected_types: Dict[str, type],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate column data types."""
        pass
    
    @abstractmethod
    async def validate_column_constraints(
        self,
        data: pd.DataFrame,
        constraints: Dict[str, Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate column constraints (nullable, unique, etc.)."""
        pass
    
    @abstractmethod
    async def validate_foreign_key_constraints(
        self,
        data: pd.DataFrame,
        foreign_keys: Dict[str, Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate foreign key constraints."""
        pass