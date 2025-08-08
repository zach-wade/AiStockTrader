"""
Record Validator Helper

Validates data records based on model requirements and business rules.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from main.interfaces.repositories.base import ValidationLevel
from main.utils.core import get_logger

logger = get_logger(__name__)


class RecordValidator:
    """
    Validates records according to specified rules and validation levels.
    
    Supports different validation levels (NONE, BASIC, STRICT) and
    custom validation functions.
    """
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.BASIC,
        required_fields: Optional[List[str]] = None,
        field_types: Optional[Dict[str, type]] = None,
        custom_validators: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the RecordValidator.
        
        Args:
            validation_level: Level of validation to perform
            required_fields: List of required field names
            field_types: Expected types for fields
            custom_validators: Custom validation functions by field name
        """
        self.validation_level = validation_level
        self.required_fields = required_fields or []
        self.field_types = field_types or {}
        self.custom_validators = custom_validators or {}
        
        logger.debug(f"RecordValidator initialized with level: {validation_level.value}")
    
    def validate(self, record: Dict[str, Any]) -> List[str]:
        """
        Validate a single record.
        
        Args:
            record: Record to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        if self.validation_level == ValidationLevel.NONE:
            return []
        
        errors = []
        
        # Basic validation
        if self.validation_level >= ValidationLevel.BASIC:
            errors.extend(self._validate_required_fields(record))
            errors.extend(self._validate_field_types(record))
        
        # Strict validation
        if self.validation_level >= ValidationLevel.STRICT:
            errors.extend(self._validate_field_values(record))
            errors.extend(self._run_custom_validators(record))
        
        return errors
    
    def validate_batch(self, records: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Validate multiple records.
        
        Args:
            records: List of records to validate
            
        Returns:
            Dictionary mapping record index to validation errors
        """
        errors_by_index = {}
        
        for i, record in enumerate(records):
            errors = self.validate(record)
            if errors:
                errors_by_index[i] = errors
        
        return errors_by_index
    
    def _validate_required_fields(self, record: Dict[str, Any]) -> List[str]:
        """Check that all required fields are present and non-null."""
        errors = []
        
        for field in self.required_fields:
            if field not in record:
                errors.append(f"Missing required field: {field}")
            elif record[field] is None:
                errors.append(f"Required field '{field}' cannot be null")
        
        return errors
    
    def _validate_field_types(self, record: Dict[str, Any]) -> List[str]:
        """Check that fields have the expected types."""
        errors = []
        
        for field, expected_type in self.field_types.items():
            if field in record and record[field] is not None:
                value = record[field]
                
                # Special handling for datetime
                if expected_type == datetime:
                    if not isinstance(value, (datetime, str)):
                        errors.append(
                            f"Field '{field}' must be datetime or string, got {type(value).__name__}"
                        )
                # Special handling for numeric types
                elif expected_type in (int, float):
                    if not isinstance(value, (int, float)):
                        errors.append(
                            f"Field '{field}' must be numeric, got {type(value).__name__}"
                        )
                # Regular type check
                elif not isinstance(value, expected_type):
                    errors.append(
                        f"Field '{field}' must be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        
        return errors
    
    def _validate_field_values(self, record: Dict[str, Any]) -> List[str]:
        """Validate field values against business rules."""
        errors = []
        
        # Validate numeric ranges
        if 'price' in record and record['price'] is not None:
            if record['price'] < 0:
                errors.append("Price cannot be negative")
            if record['price'] > 1000000:
                errors.append("Price exceeds maximum allowed value")
        
        if 'volume' in record and record['volume'] is not None:
            if record['volume'] < 0:
                errors.append("Volume cannot be negative")
        
        # Validate OHLC relationships
        if all(k in record for k in ['open', 'high', 'low', 'close']):
            o, h, l, c = record['open'], record['high'], record['low'], record['close']
            
            if None not in (o, h, l, c):
                if l > h:
                    errors.append("Low price cannot exceed high price")
                if o > h or o < l:
                    errors.append("Open price must be between low and high")
                if c > h or c < l:
                    errors.append("Close price must be between low and high")
        
        # Validate dates
        if 'timestamp' in record and record['timestamp'] is not None:
            try:
                if isinstance(record['timestamp'], str):
                    timestamp = datetime.fromisoformat(record['timestamp'])
                else:
                    timestamp = record['timestamp']
                
                # Check for future dates
                if timestamp > datetime.now():
                    errors.append("Timestamp cannot be in the future")
                
                # Check for very old dates
                min_date = datetime(1900, 1, 1)
                if timestamp < min_date:
                    errors.append(f"Timestamp cannot be before {min_date}")
                    
            except (ValueError, TypeError) as e:
                errors.append(f"Invalid timestamp format: {e}")
        
        return errors
    
    def _run_custom_validators(self, record: Dict[str, Any]) -> List[str]:
        """Run custom validation functions."""
        errors = []
        
        for field, validator_func in self.custom_validators.items():
            if field in record:
                try:
                    # Custom validators should return error message or None
                    error = validator_func(record[field], record)
                    if error:
                        errors.append(error)
                except Exception as e:
                    logger.warning(f"Custom validator for '{field}' failed: {e}")
                    errors.append(f"Validation error for field '{field}': {e}")
        
        return errors
    
    def add_custom_validator(self, field: str, validator: Callable) -> None:
        """
        Add a custom validator for a field.
        
        Args:
            field: Field name to validate
            validator: Function that takes (value, record) and returns error or None
        """
        self.custom_validators[field] = validator
    
    def set_validation_level(self, level: ValidationLevel) -> None:
        """
        Change the validation level.
        
        Args:
            level: New validation level
        """
        self.validation_level = level
        logger.debug(f"Validation level changed to: {level.value}")