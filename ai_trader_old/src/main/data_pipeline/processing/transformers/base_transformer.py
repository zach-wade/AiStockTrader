"""
Base Transformer Implementation

Thin wrapper implementing IDataTransformer interface.
"""

# Standard library imports
from typing import Any

# Third-party imports
import pandas as pd

# Local imports
from main.data_pipeline.core.enums import DataLayer
from main.interfaces.data_pipeline.processing import IDataTransformer
from main.utils.core import ErrorHandlingMixin, get_logger


class BaseTransformer(IDataTransformer, ErrorHandlingMixin):
    """Base implementation of IDataTransformer interface."""

    def __init__(self):
        """Initialize base transformer."""
        self.logger = get_logger(__name__)
        self._transformation_stats = {"total_transformations": 0, "successful": 0, "failed": 0}

    async def transform(
        self,
        data: Any,
        source_format: str,
        target_format: str,
        layer: DataLayer,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Transform data from source to target format."""
        raise NotImplementedError("Subclasses must implement transform method")

    async def validate_transformation(
        self, original_data: Any, transformed_data: Any, layer: DataLayer
    ) -> dict[str, Any]:
        """Validate transformation results."""
        validation_result = {"is_valid": True, "errors": [], "warnings": [], "metrics": {}}

        # Basic validation
        if transformed_data is None:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Transformation resulted in None")
            return validation_result

        # DataFrame-specific validation
        if isinstance(original_data, pd.DataFrame) and isinstance(transformed_data, pd.DataFrame):
            # Check for data loss
            if len(transformed_data) < len(original_data) * 0.5:
                validation_result["warnings"].append(
                    f"Significant data reduction: {len(original_data)} -> {len(transformed_data)} rows"
                )

            # Check columns
            validation_result["metrics"]["original_columns"] = len(original_data.columns)
            validation_result["metrics"]["transformed_columns"] = len(transformed_data.columns)
            validation_result["metrics"]["rows_retained"] = (
                len(transformed_data) / len(original_data) if len(original_data) > 0 else 0
            )

        return validation_result

    async def get_supported_transformations(self) -> dict[str, list[str]]:
        """Get supported transformation mappings."""
        return {
            "pandas": ["dict", "list", "json"],
            "dict": ["pandas", "json"],
            "list": ["pandas", "dict"],
            "json": ["dict", "pandas"],
        }

    async def get_transformation_stats(self) -> dict[str, Any]:
        """Get transformation statistics."""
        return self._transformation_stats.copy()
