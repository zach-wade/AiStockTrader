"""
Record Level Validator - Interface Implementation

Performs granular, record-by-record validation for various data types.
Implements IRecordValidator and IMarketDataValidator interfaces.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Interface imports
from main.interfaces.validation import (
    IValidationResult,
    IValidationContext,
    ValidationSeverity,
    ValidationStage
)
from main.interfaces.validation.validators import (
    IRecordValidator,
    IMarketDataValidator
)

# Core imports
from main.data_pipeline.core.enums import DataLayer, DataType
from main.data_pipeline.validation.core.validation_types import ValidationResult, ValidationContext
from main.utils.core import get_logger

# Config imports
from main.config.validation_models import DataPipelineConfig

logger = get_logger(__name__)


class RecordValidator:
    """
    Record-level validator implementation.
    
    Implements IRecordValidator and IMarketDataValidator interfaces
    for granular record validation.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], DataPipelineConfig.ValidationConfig]] = None):
        """
        Initialize the record validator.
        
        Args:
            config: Configuration dictionary or ValidationConfig object
        """
        # Handle different config types
        if isinstance(config, DataPipelineConfig.ValidationConfig):
            self.validation_config = config
            self.config = {}  # Keep for backward compatibility
            # Extract settings from ValidationConfig
            # Map cleaning settings to validation rules
            self.default_required_fields = {
                'market_data': ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'],
                'news': ['symbol', 'timestamp', 'title'],
                'fundamentals': ['symbol', 'period', 'fiscal_date']
            }
            self.field_mappings = config.cleaning.field_mappings
            self.validation_rules = {
                'field_ranges': {},
                'field_types': {}
            }
            self.max_price_deviation = config.market_data.max_price_deviation
        else:
            # Legacy dict-based configuration
            self.config = config or {}
            self.validation_config = None
            self.default_required_fields = self.config.get('required_fields', {})
            self.field_mappings = self.config.get('field_mappings', {})
            self.validation_rules = self.config.get('validation_rules', {})
            self.max_price_deviation = self.config.get('max_price_deviation', 0.5)
        
        logger.info("Initialized RecordValidator with interface-based architecture")
    
    # IValidator interface methods
    async def validate(
        self,
        data: Any,
        context: IValidationContext
    ) -> IValidationResult:
        """Validate data with given context."""
        start_time = datetime.now()
        
        try:
            if isinstance(data, list):
                # Batch validation
                results = await self.validate_record_batch(data, context)
                
                # Aggregate results
                all_errors = []
                all_warnings = []
                metrics = {'record_count': len(data), 'results': []}
                
                for result in results:
                    all_errors.extend(result.errors)
                    all_warnings.extend(result.warnings)
                    metrics['results'].append(result.metrics)
                
                passed = len(all_errors) == 0
                
            else:
                # Single record validation
                result = await self.validate_single_record(data, 0, context)
                return result
            
        except Exception as e:
            logger.error(f"Record validation error: {e}", exc_info=True)
            all_errors = [f"Validation error: {str(e)}"]
            all_warnings = []
            metrics = {}
            passed = False
        
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return ValidationResult(
            stage=context.stage,
            passed=passed,
            errors=all_errors,
            warnings=all_warnings,
            metadata=metrics,  # Map metrics to metadata field
            timestamp=start_time,
            duration_ms=duration_ms
        )
    
    async def get_validation_rules(
        self,
        context: IValidationContext
    ) -> List[str]:
        """Get applicable validation rules for context."""
        rules = []
        
        # Add data type specific rules
        if context.data_type == DataType.MARKET_DATA:
            rules.extend([
                'required_fields_check',
                'ohlc_validation',
                'volume_validation',
                'price_relationships'
            ])
        elif context.data_type == DataType.NEWS:
            rules.extend([
                'required_fields_check',
                'content_validation',
                'sentiment_range_check'
            ])
        elif context.data_type == DataType.FINANCIALS:
            rules.extend([
                'required_fields_check',
                'financial_ratios_check',
                'earnings_validation'
            ])
        
        return rules
    
    async def is_applicable(
        self,
        context: IValidationContext
    ) -> bool:
        """Check if validator applies to given context."""
        # Record validator applies to all contexts
        return True
    
    # IRecordValidator interface methods
    async def validate_single_record(
        self,
        record: Dict[str, Any],
        record_index: int,
        context: IValidationContext
    ) -> IValidationResult:
        """Validate a single record."""
        start_time = datetime.now()
        errors = []
        warnings = []
        metrics = {'record_index': record_index}
        
        try:
            # Apply field mapping if available
            if context.source and context.source in self.field_mappings:
                record = await self.apply_field_mapping(
                    record, context.source, self.field_mappings[context.source]
                )
            
            # Get required fields for this data type
            data_type_key = context.data_type.value if hasattr(context.data_type, 'value') else str(context.data_type)
            required_fields = self.default_required_fields.get(data_type_key, [])
            
            # Validate required fields
            passed, field_errors = await self.validate_required_fields(
                record, required_fields, context
            )
            errors.extend(field_errors)
            
            # Validate field types if configured
            if 'field_types' in self.validation_rules:
                type_passed, type_errors = await self.validate_field_types(
                    record, self.validation_rules['field_types'], context
                )
                errors.extend(type_errors)
            
            # Validate field ranges if configured
            if 'field_ranges' in self.validation_rules:
                range_passed, range_errors = await self.validate_field_ranges(
                    record, self.validation_rules['field_ranges'], context
                )
                errors.extend(range_errors)
            
            # Data type specific validation
            if context.data_type == DataType.MARKET_DATA:
                market_errors, market_warnings = await self._validate_market_data_record(record)
                errors.extend(market_errors)
                warnings.extend(market_warnings)
            
            metrics['validation_checks_performed'] = len([
                'required_fields', 'field_types', 'field_ranges', 'data_type_specific'
            ])
            
        except Exception as e:
            logger.error(f"Single record validation error: {e}", exc_info=True)
            errors.append(f"Record validation error: {str(e)}")
        
        passed = len(errors) == 0
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        return ValidationResult(
            stage=context.stage,
            passed=passed,
            errors=errors,
            warnings=warnings,
            metadata=metrics,  # Map metrics to metadata field
            timestamp=start_time,
            duration_ms=duration_ms
        )
    
    async def validate_record_batch(
        self,
        records: List[Dict[str, Any]],
        context: IValidationContext
    ) -> List[IValidationResult]:
        """Validate a batch of records."""
        results = []
        
        for i, record in enumerate(records):
            result = await self.validate_single_record(record, i, context)
            results.append(result)
        
        return results
    
    async def validate_required_fields(
        self,
        record: Dict[str, Any],
        required_fields: List[str],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate required fields are present and valid."""
        errors = []
        
        missing_fields = []
        for field in required_fields:
            if not self._check_field_exists(record, field):
                missing_fields.append(field)
            elif record[field] is None:
                missing_fields.append(f"{field} (null)")
            elif isinstance(record[field], str) and not record[field].strip():
                missing_fields.append(f"{field} (empty)")
        
        if missing_fields:
            errors.append(f"Missing or invalid required fields: {missing_fields}")
        
        return len(errors) == 0, errors
    
    async def validate_field_types(
        self,
        record: Dict[str, Any],
        field_types: Dict[str, type],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate field data types."""
        errors = []
        
        for field, expected_type in field_types.items():
            if field in record and record[field] is not None:
                if not isinstance(record[field], expected_type):
                    actual_type = type(record[field]).__name__
                    expected_name = expected_type.__name__
                    errors.append(
                        f"Field '{field}' has type {actual_type}, expected {expected_name}"
                    )
        
        return len(errors) == 0, errors
    
    async def validate_field_ranges(
        self,
        record: Dict[str, Any],
        field_ranges: Dict[str, Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate field values are within acceptable ranges."""
        errors = []
        
        for field, range_config in field_ranges.items():
            if field in record and record[field] is not None:
                value = record[field]
                
                # Check minimum value
                if 'min' in range_config and value < range_config['min']:
                    errors.append(
                        f"Field '{field}' value {value} below minimum {range_config['min']}"
                    )
                
                # Check maximum value
                if 'max' in range_config and value > range_config['max']:
                    errors.append(
                        f"Field '{field}' value {value} above maximum {range_config['max']}"
                    )
                
                # Check allowed values
                if 'allowed' in range_config and value not in range_config['allowed']:
                    errors.append(
                        f"Field '{field}' value {value} not in allowed values: {range_config['allowed']}"
                    )
        
        return len(errors) == 0, errors
    
    async def apply_field_mapping(
        self,
        record: Dict[str, Any],
        source: str,
        field_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Apply source-specific field mapping to record."""
        if not field_mapping:
            return record
        
        normalized = {}
        
        # Map known fields
        for standard_field, source_field in field_mapping.items():
            if source_field in record:
                normalized[standard_field] = record[source_field]
        
        # Include unmapped fields as-is
        for field, value in record.items():
            if field not in field_mapping.values():
                normalized[field] = value
        
        return normalized
    
    # IMarketDataValidator interface methods (delegated to record validation)
    async def validate_ohlcv_data(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        symbol: Optional[str] = None,
        context: Optional[IValidationContext] = None
    ) -> IValidationResult:
        """Validate OHLCV market data."""
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        
        # Use the record validation with market data context
        if context is None:
            from main.data_pipeline.validation.core.validation_types import ValidationContext
            context = ValidationContext(
                stage=ValidationStage.INGEST,
                layer=DataLayer.RAW,
                data_type=DataType.MARKET_DATA,
                symbol=symbol
            )
        
        return await self.validate(data, context)
    
    async def validate_price_consistency(
        self,
        data: pd.DataFrame,
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate price consistency (OHLC relationships)."""
        errors = []
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            errors.append(f"Missing required OHLC columns. Need: {required_cols}")
            return False, errors
        
        # Check OHLC relationships for each row
        for idx, row in data.iterrows():
            row_errors = await self._validate_ohlc_relationships(row.to_dict())
            if row_errors:
                errors.extend([f"Row {idx}: {error}" for error in row_errors])
        
        return len(errors) == 0, errors
    
    async def validate_volume_data(
        self,
        data: pd.DataFrame,
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate volume data."""
        errors = []
        
        if 'volume' not in data.columns:
            errors.append("Volume column missing")
            return False, errors
        
        # Check for negative volumes
        negative_volumes = data[data['volume'] < 0]
        if not negative_volumes.empty:
            errors.append(f"Found {len(negative_volumes)} rows with negative volume")
        
        # Check for null volumes
        null_volumes = data[data['volume'].isnull()]
        if not null_volumes.empty:
            errors.append(f"Found {len(null_volumes)} rows with null volume")
        
        return len(errors) == 0, errors
    
    async def validate_trading_hours(
        self,
        data: pd.DataFrame,
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate data falls within trading hours."""
        # This is a placeholder - would need market-specific trading hours
        # For now, just check that we have timestamp data
        errors = []
        
        if data.index.name not in ['timestamp', 'date'] and 'timestamp' not in data.columns:
            errors.append("No timestamp information found for trading hours validation")
        
        return len(errors) == 0, errors
    
    async def validate_corporate_actions(
        self,
        market_data: pd.DataFrame,
        corporate_actions: List[Dict[str, Any]],
        context: IValidationContext
    ) -> Tuple[bool, List[str]]:
        """Validate market data against corporate actions."""
        # This would implement corporate action validation logic
        # For now, return success as a placeholder
        return True, []
    
    # Helper methods
    async def _validate_market_data_record(
        self,
        record: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Validate market data specific fields."""
        errors = []
        warnings = []
        
        # Validate OHLC relationships
        ohlc_errors = await self._validate_ohlc_relationships(record)
        errors.extend(ohlc_errors)
        
        # Validate volume
        volume = record.get('volume')
        if volume is not None:
            if not isinstance(volume, (int, float)) or pd.isna(volume):
                errors.append(f"Invalid volume type or NaN: {volume}")
            elif volume < 0:
                errors.append(f"Negative volume: {volume}")
            elif volume == 0:
                warnings.append("Zero volume detected")
        
        return errors, warnings
    
    async def _validate_ohlc_relationships(
        self,
        record: Dict[str, Any]
    ) -> List[str]:
        """Validate OHLC price relationships."""
        errors = []
        
        try:
            ohlc_fields = ['open', 'high', 'low', 'close']
            if not all(field in record for field in ohlc_fields):
                return errors  # Skip if not all OHLC fields present
            
            o = float(record['open']) if pd.notna(record['open']) else np.nan
            h = float(record['high']) if pd.notna(record['high']) else np.nan
            l = float(record['low']) if pd.notna(record['low']) else np.nan
            c = float(record['close']) if pd.notna(record['close']) else np.nan
            
            if any(pd.isna([o, h, l, c])):
                errors.append("OHLC validation skipped due to NaN values")
                return errors
            
            # Basic OHLC relationship checks
            if h < l:
                errors.append(f"High ({h}) < Low ({l})")
            if h < o:
                errors.append(f"High ({h}) < Open ({o})")
            if h < c:
                errors.append(f"High ({h}) < Close ({c})")
            if l > o:
                errors.append(f"Low ({l}) > Open ({o})")
            if l > c:
                errors.append(f"Low ({l}) > Close ({c})")
            
            # Check for extreme price deviations
            prices = [o, h, l, c]
            median_price = np.median(prices)
            max_deviation = self.max_price_deviation  # Use instance attribute
            
            if median_price > 0:
                for price, name in zip(prices, ohlc_fields):
                    deviation = abs(price - median_price) / median_price
                    if deviation > max_deviation:
                        errors.append(
                            f"{name} deviates {deviation:.1%} from median "
                            f"(max allowed: {max_deviation:.1%})"
                        )
        
        except Exception as e:
            errors.append(f"Error validating OHLC relationships: {e}")
            logger.warning(f"OHLC validation error: {e}", exc_info=True)
        
        return errors
    
    def _check_field_exists(self, record: Dict[str, Any], field: str) -> bool:
        """Check if field exists, considering common variations."""
        if field in record and record[field] is not None:
            return True
        
        # Check common field variations
        variations = {
            'date': ['Date', 'ex_date', 'Ex-Date', 'published_date'],
            'amount': ['Amount', 'Dividends', 'dividend'],
            'timestamp': ['date', 'time', 'datetime'],
            'symbol': ['ticker', 'Symbol'],
            'firm': ['firm_name', 'brokerage']
        }
        
        if field in variations:
            return any(
                var in record and record[var] is not None 
                for var in variations[field]
            )
        
        return False