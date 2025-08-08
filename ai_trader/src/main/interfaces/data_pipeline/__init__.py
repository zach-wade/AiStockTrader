"""
Data Pipeline Interfaces

Comprehensive interface definitions for all data pipeline components
supporting layer-based architecture, event-driven processing, and
clean separation of concerns.
"""

# Orchestration Interfaces
from .orchestration import (
    ILayerManager,
    IRetentionManager,
    IUnifiedOrchestrator,
    IDataProcessor,
    IBackfillOrchestrator,
    IProgressTracker,
    IEventCoordinator,
    OrchestrationStatus
)

# Processing Interfaces
from .processing import (
    IDataTransformer,
    IDataStandardizer,
    IDataCleaner,
    IFeatureBuilder,
    IDataAnalyzer,
    IStreamProcessor,
    IBatchProcessor
)

# Historical Interfaces
from .historical import (
    IGapDetector,
    IHistoricalManager,
    IDataFetcher,
    IDataRouter,
    IHistoricalAnalyzer,
    IArchiveManager,
    IMaintenanceScheduler
)

# Ingestion Interfaces
from .ingestion import (
    IDataSource,
    IDataClient,
    IIngestionClient,
    IAssetClient,
    IIngestionProcessor,
    IIngestionCoordinator,
    IRateLimiter,
    IIngestionValidator,
    IIngestionMonitor
)

# Monitoring Interfaces
from .monitoring import (
    IHealthMonitor,
    IMetricsCollector,
    IPerformanceMonitor,
    IAlertManager,
    IDashboardProvider,
    ILogAggregator,
    IServiceMonitor,
    HealthStatus,
    AlertSeverity
)

__all__ = [
    # Orchestration
    'ILayerManager',
    'IRetentionManager', 
    'IUnifiedOrchestrator',
    'IDataProcessor',
    'IBackfillOrchestrator',
    'IProgressTracker',
    'IEventCoordinator',
    'OrchestrationStatus',
    
    # Processing
    'IDataTransformer',
    'IDataStandardizer',
    'IDataCleaner',
    'IFeatureBuilder',
    'IDataAnalyzer',
    'IStreamProcessor',
    'IBatchProcessor',
    
    # Historical
    'IGapDetector',
    'IHistoricalManager',
    'IDataFetcher',
    'IDataRouter',
    'IHistoricalAnalyzer',
    'IArchiveManager',
    'IMaintenanceScheduler',
    
    # Ingestion
    'IDataSource',
    'IDataClient',
    'IIngestionClient',
    'IAssetClient',
    'IIngestionProcessor',
    'IIngestionCoordinator',
    'IRateLimiter',
    'IIngestionValidator',
    'IIngestionMonitor',
    
    # Monitoring
    'IHealthMonitor',
    'IMetricsCollector',
    'IPerformanceMonitor',
    'IAlertManager',
    'IDashboardProvider',
    'ILogAggregator',
    'IServiceMonitor',
    'HealthStatus',
    'AlertSeverity'
]