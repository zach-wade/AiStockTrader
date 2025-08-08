"""
Corporate Actions Bulk Loader - Compatibility Module

This module provides backward compatibility by re-exporting the bulk loader
from its new location in the ingestion layer.
"""

from main.data_pipeline.ingestion.loaders.corporate_actions import (
    CorporateActionsBulkLoader,
    __all__ as original_all
)

# Re-export everything from the original module
__all__ = ['CorporateActionsBulkLoader'] if 'original_all' not in locals() else original_all