"""
Unified Pipeline Orchestrator

Single entry point for all data pipeline operations.
Coordinates layer-based processing using existing services.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum

from main.utils.core import get_logger, timer
from main.interfaces.events import IEventBus, Event, EventType
from main.data_pipeline.services.container import SimpleServiceContainer
from main.data_pipeline.orchestration.coordinators import (
    LayerCoordinator,
    DataFetchCoordinator,
    StorageCoordinator,
    DataStage,
    FetchRequest,
    StorageRequest
)
from main.data_pipeline.core.enums import DataLayer


class PipelineMode(Enum):
    """Pipeline execution modes."""
    BACKFILL = "backfill"
    REALTIME = "realtime"
    INCREMENTAL = "incremental"
    TEST = "test"


@dataclass
class PipelineRequest:
    """Unified request for pipeline operations."""
    mode: PipelineMode
    layers: List[DataLayer]
    stages: List[DataStage]
    symbols: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    dry_run: bool = False


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    mode: PipelineMode
    layers_processed: int
    total_symbols: int
    total_records: int
    duration_seconds: float
    success: bool
    errors: List[str]


class UnifiedPipeline:
    """
    Unified pipeline orchestrator.
    
    Single entry point that coordinates all data pipeline operations
    using the focused coordinators and services.
    """
    
    def __init__(
        self,
        container: SimpleServiceContainer,
        event_bus: Optional[IEventBus] = None
    ):
        """
        Initialize the unified pipeline.
        
        Args:
            container: Service container with all dependencies
            event_bus: Optional event bus for notifications
        """
        self.container = container
        self.event_bus = event_bus
        self.logger = get_logger(__name__)
        
        # Resolve coordinators
        self.layer_coordinator = container.resolve(LayerCoordinator)
        self.fetch_coordinator = container.resolve(DataFetchCoordinator)
        self.storage_coordinator = container.resolve(StorageCoordinator)
        
        if not all([self.layer_coordinator, self.fetch_coordinator, self.storage_coordinator]):
            raise ValueError("Required coordinators not found in container")
    
    @timer
    async def execute(self, request: PipelineRequest) -> PipelineResult:
        """
        Execute pipeline request.
        
        Args:
            request: Pipeline execution request
            
        Returns:
            Pipeline execution result
        """
        start_time = datetime.now(timezone.utc)
        errors = []
        total_symbols = 0
        total_records = 0
        
        self.logger.info(
            f"Starting {request.mode.value} pipeline for "
            f"{len(request.layers)} layers, {len(request.stages)} stages"
        )
        
        try:
            # Emit start event
            await self._emit_event("pipeline_started", request)
            
            # Get symbols for each layer
            layer_symbols = await self.layer_coordinator.get_layer_symbols()
            
            # Process each layer
            for layer in request.layers:
                symbols = request.symbols or layer_symbols.get_layer(layer)
                
                if not symbols:
                    self.logger.warning(f"No symbols for {layer.value}")
                    continue
                
                total_symbols += len(symbols)
                
                # Skip if dry run
                if request.dry_run:
                    self.logger.info(f"[DRY RUN] Would process {len(symbols)} symbols in {layer.value}")
                    continue
                
                # Process layer
                layer_records = await self._process_layer(
                    layer=layer,
                    symbols=symbols,
                    stages=request.stages,
                    start_date=request.start_date,
                    end_date=request.end_date
                )
                
                total_records += layer_records
                
                # Emit progress event
                await self._emit_event("layer_completed", {
                    "layer": layer.value,
                    "symbols": len(symbols),
                    "records": layer_records
                })
            
            # Flush all storage
            await self.storage_coordinator.flush_all()
            
            # Calculate duration
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Emit completion event
            await self._emit_event("pipeline_completed", {
                "duration": duration,
                "records": total_records
            })
            
            return PipelineResult(
                mode=request.mode,
                layers_processed=len(request.layers),
                total_symbols=total_symbols,
                total_records=total_records,
                duration_seconds=duration,
                success=True,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            errors.append(str(e))
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return PipelineResult(
                mode=request.mode,
                layers_processed=0,
                total_symbols=total_symbols,
                total_records=total_records,
                duration_seconds=duration,
                success=False,
                errors=errors
            )
    
    async def _process_layer(
        self,
        layer: DataLayer,
        symbols: List[str],
        stages: List[DataStage],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> int:
        """
        Process a single layer.
        
        Args:
            layer: Layer to process
            symbols: Symbols to process
            stages: Stages to execute
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of records processed
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        self.logger.info(f"Processing {layer.value} with {len(symbols)} symbols")
        
        # Create fetch request
        fetch_request = FetchRequest(
            symbols=symbols,
            stages=stages,
            start_date=start_date,
            end_date=end_date
        )
        
        # Fetch data
        fetch_results = await self.fetch_coordinator.fetch_data(fetch_request)
        
        # Count total records
        total_records = sum(r.records_fetched for r in fetch_results)
        
        self.logger.info(f"Fetched {total_records} records for {layer.value}")
        
        return total_records
    
    async def _emit_event(self, event_type: str, data: Any):
        """Emit event if event bus is available."""
        if self.event_bus:
            event = Event(
                type=EventType.DATA_PIPELINE,
                data={
                    "event": event_type,
                    **( data if isinstance(data, dict) else {"data": data})
                }
            )
            await self.event_bus.publish(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "storage": {},
            "coordinators": {
                "layer": "active",
                "fetch": "active",
                "storage": "active"
            }
        }
        
        # Get storage statistics if available
        if hasattr(self.storage_coordinator, 'get_statistics'):
            stats["storage"] = self.storage_coordinator.get_statistics()
        
        return stats