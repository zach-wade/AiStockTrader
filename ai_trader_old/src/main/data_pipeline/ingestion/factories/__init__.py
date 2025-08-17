"""
Ingestion Factories Module

Provides factory classes for creating appropriate handlers and processors
based on data characteristics.

Note: Import factories directly from their modules to avoid circular imports.
E.g.: from main.data_pipeline.ingestion.factories.bulk_loader_factory import BulkLoaderFactory
"""

# Only export the fundamentals format factory by default to avoid circular imports
from .fundamentals_format_factory import FundamentalsFormatFactory

__all__ = ["FundamentalsFormatFactory"]
