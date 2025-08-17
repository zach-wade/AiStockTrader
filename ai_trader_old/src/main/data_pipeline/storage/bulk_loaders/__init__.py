"""
Bulk Loaders Module

This module provides compatibility imports for bulk loaders that are now located
in the ingestion layer. This maintains backward compatibility with existing code
while following the new architecture where loaders belong in ingestion.
"""

# Local imports
# Import all bulk loaders from the ingestion layer
from main.data_pipeline.ingestion.loaders import (
    BaseBulkLoader,
    CorporateActionsBulkLoader,
    FundamentalsBulkLoader,
    MarketDataBulkLoader,
    MarketDataSplitBulkLoader,
    NewsBulkLoader,
)

# Re-export for backward compatibility
__all__ = [
    "BaseBulkLoader",
    "MarketDataBulkLoader",
    "MarketDataSplitBulkLoader",
    "NewsBulkLoader",
    "FundamentalsBulkLoader",
    "CorporateActionsBulkLoader",
]
