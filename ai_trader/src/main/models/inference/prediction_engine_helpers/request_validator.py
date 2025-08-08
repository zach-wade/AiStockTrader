"""
Request Validator for Prediction Engine

Provides validation for prediction requests including:
- Input data validation
- Feature validation
- Request parameter validation
- Security checks
- Rate limiting validation
"""

import logging
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    LENIENT = "lenient"


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


class RequestValidator:
    """Validates prediction engine requests."""
    
    def __init__(self,
                 required_features: Optional[List[str]] = None,
                 feature_types: Optional[Dict[str, type]] = None,
                 feature_ranges: Optional[Dict[str, tuple]] = None,
                 max_request_size: int = 1000,
                 validation_level: ValidationLevel = ValidationLevel.NORMAL):
        """
        Initialize request validator.
        
        Args:
            required_features: List of required feature names
            feature_types: Expected types for each feature
            feature_ranges: Valid ranges for numeric features
            max_request_size: Maximum number of samples per request
            validation_level: Strictness of validation
        """
        self.required_features = set(required_features or [])
        self.feature_types = feature_types or {}
        self.feature_ranges = feature_ranges or {}
        self.max_request_size = max_request_size
        self.validation_level = validation_level
        
        # Request tracking for rate limiting
        self._request_history: List[datetime] = []
        self._request_history_limit = 1000
    
    def validate_request(self, 
                        request_data: Union[pd.DataFrame, Dict[str, Any], List[Dict]],
                        request_metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a prediction request.
        
        Args:
            request_data: Input data for prediction
            request_metadata: Optional metadata about the request
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Convert to DataFrame if needed
            df = self._convert_to_dataframe(request_data)
            metadata['sample_count'] = len(df)
            metadata['feature_count'] = len(df.columns)
            
            # Basic validation
            if self.validation_level != ValidationLevel.LENIENT:
                self._validate_basic_constraints(df, errors, warnings)
            
            # Feature validation
            self._validate_features(df, errors, warnings)
            
            # Data type validation
            if self.validation_level == ValidationLevel.STRICT:
                self._validate_data_types(df, errors, warnings)
            
            # Range validation
            self._validate_ranges(df, errors, warnings)
            
            # Security validation
            self._validate_security(df, errors, warnings)
            
            # Rate limiting validation
            if request_metadata:
                self._validate_rate_limits(request_metadata, errors, warnings)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def validate_features(self, features: pd.DataFrame) -> ValidationResult:
        """
        Validate feature DataFrame.
        
        Args:
            features: Features to validate
            
        Returns:
            ValidationResult
        """
        return self.validate_request(features)
    
    def validate_batch(self, batch: List[Dict[str, Any]]) -> List[ValidationResult]:
        """
        Validate a batch of requests.
        
        Args:
            batch: List of request dictionaries
            
        Returns:
            List of ValidationResults
        """
        results = []
        for i, request in enumerate(batch):
            try:
                result = self.validate_request(request)
                results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    is_valid=False,
                    errors=[f"Failed to validate request {i}: {str(e)}"],
                    warnings=[],
                    metadata={'index': i}
                ))
        
        return results
    
    def _convert_to_dataframe(self, data: Union[pd.DataFrame, Dict, List]) -> pd.DataFrame:
        """Convert input data to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            # Single sample
            return pd.DataFrame([data])
        elif isinstance(data, list):
            # Multiple samples
            return pd.DataFrame(data)
        else:
            raise ValidationError(f"Unsupported data type: {type(data)}")
    
    def _validate_basic_constraints(self, df: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Validate basic constraints."""
        # Check if empty
        if df.empty:
            errors.append("Input data is empty")
            return
        
        # Check size limit
        if len(df) > self.max_request_size:
            errors.append(f"Request size ({len(df)}) exceeds maximum ({self.max_request_size})")
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            null_features = null_counts[null_counts > 0].index.tolist()
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Null values found in features: {null_features}")
            else:
                warnings.append(f"Null values found in features: {null_features}")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df[numeric_cols]).sum()
        if inf_counts.any():
            inf_features = inf_counts[inf_counts > 0].index.tolist()
            errors.append(f"Infinite values found in features: {inf_features}")
    
    def _validate_features(self, df: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Validate required features are present."""
        if not self.required_features:
            return
        
        present_features = set(df.columns)
        missing_features = self.required_features - present_features
        
        if missing_features:
            errors.append(f"Missing required features: {sorted(missing_features)}")
        
        # Check for extra features
        extra_features = present_features - self.required_features
        if extra_features and self.validation_level == ValidationLevel.STRICT:
            warnings.append(f"Extra features found: {sorted(extra_features)}")
    
    def _validate_data_types(self, df: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Validate data types."""
        if not self.feature_types:
            return
        
        for feature, expected_type in self.feature_types.items():
            if feature not in df.columns:
                continue
            
            actual_type = df[feature].dtype
            
            # Check numeric types
            if expected_type in (int, float, np.number):
                if not pd.api.types.is_numeric_dtype(actual_type):
                    errors.append(f"Feature '{feature}' should be numeric, got {actual_type}")
            
            # Check string types
            elif expected_type in (str, object):
                if not pd.api.types.is_object_dtype(actual_type):
                    warnings.append(f"Feature '{feature}' should be string/object, got {actual_type}")
            
            # Check datetime types
            elif expected_type == datetime:
                if not pd.api.types.is_datetime64_any_dtype(actual_type):
                    errors.append(f"Feature '{feature}' should be datetime, got {actual_type}")
    
    def _validate_ranges(self, df: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Validate value ranges."""
        if not self.feature_ranges:
            return
        
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature not in df.columns:
                continue
            
            # Skip non-numeric features
            if not pd.api.types.is_numeric_dtype(df[feature]):
                continue
            
            # Check min
            if min_val is not None:
                below_min = df[feature] < min_val
                if below_min.any():
                    count = below_min.sum()
                    min_found = df[feature].min()
                    msg = f"Feature '{feature}' has {count} values below minimum ({min_val}), min found: {min_found}"
                    if self.validation_level == ValidationLevel.STRICT:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
            
            # Check max
            if max_val is not None:
                above_max = df[feature] > max_val
                if above_max.any():
                    count = above_max.sum()
                    max_found = df[feature].max()
                    msg = f"Feature '{feature}' has {count} values above maximum ({max_val}), max found: {max_found}"
                    if self.validation_level == ValidationLevel.STRICT:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
    
    def _validate_security(self, df: pd.DataFrame, errors: List[str], warnings: List[str]):
        """Validate for security issues."""
        # Check for suspiciously large values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            max_abs = df[col].abs().max()
            if max_abs > 1e10:  # Arbitrary large number
                warnings.append(f"Feature '{col}' has suspiciously large values (max abs: {max_abs:.2e})")
        
        # Check for potential injection in string columns
        string_cols = df.select_dtypes(include=[object]).columns
        dangerous_patterns = ['<script', 'javascript:', 'onclick', 'onerror', 'eval(']
        
        for col in string_cols:
            for pattern in dangerous_patterns:
                if df[col].astype(str).str.contains(pattern, case=False, na=False).any():
                    errors.append(f"Potential injection detected in feature '{col}' (pattern: {pattern})")
    
    def _validate_rate_limits(self, metadata: Dict[str, Any], errors: List[str], warnings: List[str]):
        """Validate rate limiting constraints."""
        # Track request
        now = datetime.now()
        self._request_history.append(now)
        
        # Clean old entries
        cutoff = now - timedelta(hours=1)
        self._request_history = [t for t in self._request_history[-self._request_history_limit:] 
                                if t > cutoff]
        
        # Check rate limits
        requests_per_minute = sum(1 for t in self._request_history 
                                 if t > now - timedelta(minutes=1))
        requests_per_hour = len(self._request_history)
        
        # Define limits based on validation level
        limits = {
            ValidationLevel.STRICT: (10, 100),    # 10/min, 100/hour
            ValidationLevel.NORMAL: (60, 1000),   # 60/min, 1000/hour
            ValidationLevel.LENIENT: (300, 5000)  # 300/min, 5000/hour
        }
        
        min_limit, hour_limit = limits[self.validation_level]
        
        if requests_per_minute > min_limit:
            warnings.append(f"High request rate: {requests_per_minute} requests/minute")
        
        if requests_per_hour > hour_limit:
            errors.append(f"Rate limit exceeded: {requests_per_hour} requests/hour (limit: {hour_limit})")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation statistics report."""
        return {
            'required_features': sorted(self.required_features),
            'feature_count': len(self.required_features),
            'type_constraints': len(self.feature_types),
            'range_constraints': len(self.feature_ranges),
            'validation_level': self.validation_level.value,
            'max_request_size': self.max_request_size,
            'recent_requests': len(self._request_history)
        }


# Convenience validators
def create_strict_validator(required_features: List[str], **kwargs) -> RequestValidator:
    """Create a strict validator."""
    return RequestValidator(
        required_features=required_features,
        validation_level=ValidationLevel.STRICT,
        **kwargs
    )


def create_lenient_validator(required_features: Optional[List[str]] = None, **kwargs) -> RequestValidator:
    """Create a lenient validator."""
    return RequestValidator(
        required_features=required_features,
        validation_level=ValidationLevel.LENIENT,
        **kwargs
    )