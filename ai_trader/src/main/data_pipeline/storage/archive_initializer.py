"""
Archive Initializer - Compatibility Module

Provides backward compatibility for code expecting archive at old location.
"""

from main.data_pipeline.storage.archive import (
    DataArchive,
    RawDataRecord,
    get_archive
)

# Re-export for backward compatibility
__all__ = [
    'DataArchive',
    'RawDataRecord', 
    'get_archive'
]