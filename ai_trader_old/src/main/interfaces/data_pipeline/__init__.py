"""
Data Pipeline Interfaces

Comprehensive interface definitions for all data pipeline components
supporting layer-based architecture, event-driven processing, and
clean separation of concerns.
"""

# Orchestration Interfaces
# Historical Interfaces
from .historical import (
    IArchiveManager,
    IDataFetcher,
    IDataRouter,
    IGapDetector,
    IHistoricalAnalyzer,
    IHistoricalManager,
    IMaintenanceScheduler,
)

# Ingestion Interfaces
from .ingestion import (
    IAssetClient,
    IDataClient,
    IDataSource,
    IIngestionClient,
    IIngestionCoordinator,
    IIngestionMonitor,
    IIngestionProcessor,
    IIngestionValidator,
    IRateLimiter,
)

# Monitoring Interfaces
from .monitoring import (
    AlertSeverity,
    HealthStatus,
    IAlertManager,
    IDashboardProvider,
    IHealthMonitor,
    ILogAggregator,
    IMetricsCollector,
    IPerformanceMonitor,
    IServiceMonitor,
)
from .orchestration import (
    IBackfillOrchestrator,
    IDataProcessor,
    IEventCoordinator,
    ILayerManager,
    IProgressTracker,
    IRetentionManager,
    IUnifiedOrchestrator,
    OrchestrationStatus,
)

# Processing Interfaces
from .processing import (
    IBatchProcessor,
    IDataAnalyzer,
    IDataCleaner,
    IDataStandardizer,
    IDataTransformer,
    IFeatureBuilder,
    IStreamProcessor,
)

__all__ = [
    # Orchestration
    "ILayerManager",
    "IRetentionManager",
    "IUnifiedOrchestrator",
    "IDataProcessor",
    "IBackfillOrchestrator",
    "IProgressTracker",
    "IEventCoordinator",
    "OrchestrationStatus",
    # Processing
    "IDataTransformer",
    "IDataStandardizer",
    "IDataCleaner",
    "IFeatureBuilder",
    "IDataAnalyzer",
    "IStreamProcessor",
    "IBatchProcessor",
    # Historical
    "IGapDetector",
    "IHistoricalManager",
    "IDataFetcher",
    "IDataRouter",
    "IHistoricalAnalyzer",
    "IArchiveManager",
    "IMaintenanceScheduler",
    # Ingestion
    "IDataSource",
    "IDataClient",
    "IIngestionClient",
    "IAssetClient",
    "IIngestionProcessor",
    "IIngestionCoordinator",
    "IRateLimiter",
    "IIngestionValidator",
    "IIngestionMonitor",
    # Monitoring
    "IHealthMonitor",
    "IMetricsCollector",
    "IPerformanceMonitor",
    "IAlertManager",
    "IDashboardProvider",
    "ILogAggregator",
    "IServiceMonitor",
    "HealthStatus",
    "AlertSeverity",
]
