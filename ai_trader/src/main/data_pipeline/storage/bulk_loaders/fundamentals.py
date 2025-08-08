"""
Fundamentals Bulk Loader - Compatibility Module

This module provides backward compatibility by re-exporting the bulk loader
from its new location in the ingestion layer.
"""

from main.data_pipeline.ingestion.loaders.fundamentals import (
    FundamentalsBulkLoader,
    __all__ as original_all
)

# Re-export everything from the original module
__all__ = ['FundamentalsBulkLoader'] if 'original_all' not in locals() else original_all