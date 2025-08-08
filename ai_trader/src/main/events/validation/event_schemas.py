"""
Event Schema Definitions and Validation

Provides JSON schema definitions for all event types and validation utilities.
This ensures data integrity and catches malformed events at publish time.
"""

from typing import Dict, Any, Optional
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator

from main.utils.core import get_logger

logger = get_logger(__name__)


# Schema definitions for event types
SCHEMAS = {
    "BackfillRequested": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["backfill_id", "symbol", "layer", "data_types", "start_date", "end_date"],
        "properties": {
            "backfill_id": {
                "type": "string",
                "minLength": 1,
                "description": "Unique identifier for the backfill request"
            },
            "symbol": {
                "type": "string",
                "pattern": "^[A-Z]{1,5}$",
                "description": "Stock symbol (1-5 uppercase letters)"
            },
            "layer": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
                "description": "Data layer (0-3)"
            },
            "data_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["market_data", "news", "corporate_actions", "fundamentals", "options"]
                },
                "minItems": 1,
                "description": "Types of data to backfill"
            },
            "start_date": {
                "type": "string",
                "format": "date-time",
                "description": "Start date for backfill (ISO 8601)"
            },
            "end_date": {
                "type": "string",
                "format": "date-time",
                "description": "End date for backfill (ISO 8601)"
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "critical"],
                "default": "normal",
                "description": "Backfill priority"
            }
        },
        "additionalProperties": False
    },
    
    "SymbolQualified": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["symbol", "layer", "qualification_reason"],
        "properties": {
            "symbol": {
                "type": "string",
                "pattern": "^[A-Z]{1,5}$",
                "description": "Stock symbol"
            },
            "layer": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
                "description": "Qualified layer"
            },
            "qualification_reason": {
                "type": "string",
                "minLength": 1,
                "description": "Reason for qualification"
            },
            "metrics": {
                "type": "object",
                "description": "Metrics that led to qualification",
                "additionalProperties": True
            },
            "source": {
                "type": "string",
                "description": "Source system"
            }
        }
    },
    
    "SymbolPromoted": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["symbol", "from_layer", "to_layer", "promotion_reason"],
        "properties": {
            "symbol": {
                "type": "string",
                "pattern": "^[A-Z]{1,5}$"
            },
            "from_layer": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3
            },
            "to_layer": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3
            },
            "promotion_reason": {
                "type": "string",
                "minLength": 1
            },
            "metrics": {
                "type": "object",
                "additionalProperties": True
            }
        },
        "additionalProperties": False
    },
    
    "BackfillCompleted": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["backfill_id", "symbol", "success"],
        "properties": {
            "backfill_id": {
                "type": "string",
                "minLength": 1
            },
            "symbol": {
                "type": "string",
                "pattern": "^[A-Z]{1,5}$"
            },
            "layer": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3
            },
            "success": {
                "type": "boolean"
            },
            "records_processed": {
                "type": "integer",
                "minimum": 0
            },
            "duration_seconds": {
                "type": "number",
                "minimum": 0
            },
            "errors": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        }
    },
    
    "DataGapDetected": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["symbol", "data_type", "gap_start", "gap_end"],
        "properties": {
            "symbol": {
                "type": "string",
                "pattern": "^[A-Z]{1,5}$"
            },
            "data_type": {
                "type": "string",
                "enum": ["market_data", "news", "corporate_actions", "fundamentals", "options"]
            },
            "gap_start": {
                "type": "string",
                "format": "date-time"
            },
            "gap_end": {
                "type": "string",
                "format": "date-time"
            },
            "gap_size_hours": {
                "type": "number",
                "minimum": 0
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "default": "normal"
            }
        }
    }
}


class EventSchemaValidator:
    """
    Validates events against their schemas.
    
    Features:
    - Schema validation with detailed error messages
    - Performance optimizations with compiled validators
    - Optional strict mode vs warning mode
    - Schema versioning support
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the schema validator.
        
        Args:
            strict_mode: If True, raise exceptions on validation failure.
                        If False, log warnings but allow events through.
        """
        self.strict_mode = strict_mode
        self._validators: Dict[str, Draft7Validator] = {}
        self._compile_validators()
        
        # Statistics
        self._stats = {
            'validated': 0,
            'passed': 0,
            'failed': 0,
            'unknown_types': 0
        }
    
    def _compile_validators(self):
        """Pre-compile validators for better performance."""
        for event_type, schema in SCHEMAS.items():
            self._validators[event_type] = Draft7Validator(schema)
    
    def validate_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        Validate an event against its schema.
        
        Args:
            event_type: Type of event (e.g., 'BackfillRequested')
            event_data: Event data to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If strict_mode is True and validation fails
        """
        self._stats['validated'] += 1
        
        # Get validator for event type
        validator = self._validators.get(event_type)
        
        if not validator:
            self._stats['unknown_types'] += 1
            if self.strict_mode:
                raise ValueError(f"No schema defined for event type: {event_type}")
            else:
                logger.warning(f"No schema defined for event type: {event_type}")
                return True  # Allow unknown types in non-strict mode
        
        # Validate
        try:
            validator.validate(event_data)
            self._stats['passed'] += 1
            return True
            
        except ValidationError as e:
            self._stats['failed'] += 1
            error_msg = self._format_validation_error(event_type, e)
            
            if self.strict_mode:
                raise ValidationError(error_msg) from e
            else:
                logger.warning(f"Event validation failed: {error_msg}")
                return False
    
    def _format_validation_error(self, event_type: str, error: ValidationError) -> str:
        """Format validation error with helpful context."""
        path = " -> ".join(str(p) for p in error.path) if error.path else "root"
        return (
            f"Event type '{event_type}' validation failed at '{path}': "
            f"{error.message}"
        )
    
    def get_statistics(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self._stats.copy()
    
    def reset_statistics(self):
        """Reset validation statistics."""
        self._stats = {
            'validated': 0,
            'passed': 0,
            'failed': 0,
            'unknown_types': 0
        }
    
    def get_schema(self, event_type: str) -> Optional[Dict[str, Any]]:
        """Get schema for an event type."""
        return SCHEMAS.get(event_type)
    
    def list_event_types(self) -> list:
        """List all event types with schemas."""
        return list(SCHEMAS.keys())
    
    def add_schema(self, event_type: str, schema: Dict[str, Any]):
        """
        Add or update a schema for an event type.
        
        Args:
            event_type: Event type name
            schema: JSON schema definition
        """
        # Validate the schema itself
        Draft7Validator.check_schema(schema)
        
        # Add to schemas and compile validator
        SCHEMAS[event_type] = schema
        self._validators[event_type] = Draft7Validator(schema)
        
        logger.info(f"Added/updated schema for event type: {event_type}")


# Global validator instance (non-strict by default for backwards compatibility)
_default_validator = EventSchemaValidator(strict_mode=False)


def validate_event(event_type: str, event_data: Dict[str, Any]) -> bool:
    """
    Convenience function to validate an event using the default validator.
    
    Args:
        event_type: Type of event
        event_data: Event data to validate
        
    Returns:
        True if valid, False otherwise
    """
    return _default_validator.validate_event(event_type, event_data)


def get_validator() -> EventSchemaValidator:
    """Get the default validator instance."""
    return _default_validator


def create_strict_validator() -> EventSchemaValidator:
    """Create a new validator in strict mode."""
    return EventSchemaValidator(strict_mode=True)