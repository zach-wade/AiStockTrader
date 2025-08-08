"""
Validation Framework - Pipeline Interfaces

Pipeline-specific validation interfaces for orchestrating validation
across different stages and managing validation workflows.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Callable
from datetime import datetime
from enum import Enum
import pandas as pd

from main.data_pipeline.core.enums import DataLayer, DataType
from main.interfaces.data_pipeline.validation import (
    ValidationStage, 
    IValidationPipeline, 
    IValidationResult, 
    IValidationContext,
    IValidator
)


class ValidationMode(Enum):
    """Validation execution modes."""
    STRICT = "strict"          # Fail fast on first error
    LENIENT = "lenient"        # Continue on non-critical errors
    ADVISORY = "advisory"      # Report issues but don't fail
    DIAGNOSTIC = "diagnostic"  # Comprehensive validation for debugging


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IValidationWorkflow(ABC):
    """Interface for validation workflow management."""
    
    @abstractmethod
    async def create_workflow(
        self,
        workflow_name: str,
        stages: List[ValidationStage],
        validators_per_stage: Dict[ValidationStage, List[str]],
        workflow_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create validation workflow."""
        pass
    
    @abstractmethod
    async def execute_workflow(
        self,
        workflow_id: str,
        data: Any,
        context: IValidationContext,
        mode: ValidationMode = ValidationMode.STRICT
    ) -> Dict[str, Any]:
        """Execute validation workflow."""
        pass
    
    @abstractmethod
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status."""
        pass
    
    @abstractmethod
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel running workflow."""
        pass
    
    @abstractmethod
    async def get_workflow_results(
        self,
        workflow_id: str,
        include_details: bool = True
    ) -> Dict[str, Any]:
        """Get workflow execution results."""
        pass


class IValidationOrchestrator(ABC):
    """Interface for validation orchestration across multiple data sources."""
    
    @abstractmethod
    async def orchestrate_validation(
        self,
        data_sources: Dict[str, Any],
        validation_plan: Dict[str, Any],
        orchestration_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Orchestrate validation across multiple data sources."""
        pass
    
    @abstractmethod
    async def create_validation_plan(
        self,
        data_sources: List[str],
        target_layers: List[DataLayer],
        data_types: List[DataType],
        dependencies: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """Create validation execution plan."""
        pass
    
    @abstractmethod
    async def execute_parallel_validation(
        self,
        validation_tasks: List[Dict[str, Any]],
        max_concurrency: int = 5
    ) -> List[Dict[str, Any]]:
        """Execute validation tasks in parallel."""
        pass
    
    @abstractmethod
    async def handle_validation_dependencies(
        self,
        validation_results: Dict[str, Any],
        dependency_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle validation dependencies and cross-validations."""
        pass


class IValidationScheduler(ABC):
    """Interface for scheduled validation operations."""
    
    @abstractmethod
    async def schedule_validation(
        self,
        schedule_name: str,
        validation_config: Dict[str, Any],
        schedule_expression: str,  # cron expression
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """Schedule recurring validation."""
        pass
    
    @abstractmethod
    async def trigger_validation(
        self,
        trigger_name: str,
        data_sources: List[str],
        validation_config: Dict[str, Any]
    ) -> str:
        """Trigger ad-hoc validation."""
        pass
    
    @abstractmethod
    async def get_scheduled_validations(self) -> List[Dict[str, Any]]:
        """Get all scheduled validations."""
        pass
    
    @abstractmethod
    async def pause_schedule(self, schedule_id: str) -> bool:
        """Pause scheduled validation."""
        pass
    
    @abstractmethod
    async def resume_schedule(self, schedule_id: str) -> bool:
        """Resume scheduled validation."""
        pass
    
    @abstractmethod
    async def delete_schedule(self, schedule_id: str) -> bool:
        """Delete scheduled validation."""
        pass


class IValidationCircuitBreaker(ABC):
    """Interface for validation circuit breaker pattern."""
    
    @abstractmethod
    async def should_break_circuit(
        self,
        validation_history: List[IValidationResult],
        failure_threshold: float = 0.5,
        time_window_minutes: int = 15
    ) -> Tuple[bool, str]:
        """Check if circuit should be broken."""
        pass
    
    @abstractmethod
    async def get_circuit_status(self, circuit_name: str) -> Dict[str, Any]:
        """Get circuit breaker status."""
        pass
    
    @abstractmethod
    async def reset_circuit(self, circuit_name: str) -> bool:
        """Reset circuit breaker."""
        pass
    
    @abstractmethod
    async def configure_circuit(
        self,
        circuit_name: str,
        failure_threshold: float,
        recovery_timeout_minutes: int,
        half_open_max_calls: int
    ) -> None:
        """Configure circuit breaker parameters."""
        pass


class IValidationRetryManager(ABC):
    """Interface for validation retry management."""
    
    @abstractmethod
    async def retry_validation(
        self,
        validation_function: Callable,
        context: IValidationContext,
        max_retries: int = 3,
        backoff_multiplier: float = 2.0,
        max_backoff_seconds: int = 60
    ) -> IValidationResult:
        """Retry validation with exponential backoff."""
        pass
    
    @abstractmethod
    async def should_retry(
        self,
        validation_result: IValidationResult,
        attempt_number: int,
        retry_config: Dict[str, Any]
    ) -> bool:
        """Determine if validation should be retried."""
        pass
    
    @abstractmethod
    async def get_retry_delay(
        self,
        attempt_number: int,
        base_delay: float,
        backoff_multiplier: float,
        max_delay: float
    ) -> float:
        """Calculate retry delay."""
        pass


class IBatchValidationProcessor(ABC):
    """Interface for batch validation processing."""
    
    @abstractmethod
    async def process_validation_batch(
        self,
        batch_data: List[Any],
        batch_size: int,
        validation_config: Dict[str, Any],
        context: IValidationContext
    ) -> AsyncIterator[List[IValidationResult]]:
        """Process validation in batches."""
        pass
    
    @abstractmethod
    async def optimize_batch_size(
        self,
        data_characteristics: Dict[str, Any],
        performance_targets: Dict[str, Any]
    ) -> int:
        """Optimize batch size based on data and performance requirements."""
        pass
    
    @abstractmethod
    async def handle_batch_failures(
        self,
        failed_batch: List[Any],
        batch_results: List[IValidationResult],
        retry_config: Dict[str, Any]
    ) -> List[IValidationResult]:
        """Handle failed batch processing."""
        pass


class IValidationStateManager(ABC):
    """Interface for validation state management."""
    
    @abstractmethod
    async def save_validation_state(
        self,
        validation_id: str,
        state: Dict[str, Any],
        checkpoint_name: Optional[str] = None
    ) -> None:
        """Save validation state for recovery."""
        pass
    
    @abstractmethod
    async def load_validation_state(
        self,
        validation_id: str,
        checkpoint_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load validation state for recovery."""
        pass
    
    @abstractmethod
    async def create_checkpoint(
        self,
        validation_id: str,
        checkpoint_name: str,
        state: Dict[str, Any]
    ) -> None:
        """Create validation checkpoint."""
        pass
    
    @abstractmethod
    async def restore_from_checkpoint(
        self,
        validation_id: str,
        checkpoint_name: str
    ) -> Dict[str, Any]:
        """Restore validation from checkpoint."""
        pass
    
    @abstractmethod
    async def cleanup_validation_state(
        self,
        validation_id: str,
        retention_days: int = 7
    ) -> None:
        """Cleanup old validation state."""
        pass


class IValidationProgressTracker(ABC):
    """Interface for validation progress tracking."""
    
    @abstractmethod
    async def start_progress_tracking(
        self,
        validation_id: str,
        total_steps: int,
        description: str
    ) -> None:
        """Start tracking validation progress."""
        pass
    
    @abstractmethod
    async def update_progress(
        self,
        validation_id: str,
        completed_steps: int,
        current_step_description: Optional[str] = None
    ) -> None:
        """Update validation progress."""
        pass
    
    @abstractmethod
    async def get_progress_status(
        self,
        validation_id: str
    ) -> Dict[str, Any]:
        """Get current progress status."""
        pass
    
    @abstractmethod
    async def complete_progress_tracking(
        self,
        validation_id: str,
        final_status: str,
        summary: Optional[Dict[str, Any]] = None
    ) -> None:
        """Complete progress tracking."""
        pass


class IAdvancedValidationPipeline(IValidationPipeline):
    """Extended validation pipeline with advanced features."""
    
    @abstractmethod
    async def validate_with_workflow(
        self,
        workflow_id: str,
        data: Any,
        context: IValidationContext,
        mode: ValidationMode = ValidationMode.STRICT
    ) -> Dict[str, Any]:
        """Validate using predefined workflow."""
        pass
    
    @abstractmethod
    async def validate_with_dependencies(
        self,
        primary_data: Any,
        dependent_data: Dict[str, Any],
        dependency_rules: Dict[str, Any],
        context: IValidationContext
    ) -> Dict[str, Any]:
        """Validate with cross-data dependencies."""
        pass
    
    @abstractmethod
    async def validate_incrementally(
        self,
        new_data: Any,
        existing_validation_state: Dict[str, Any],
        context: IValidationContext
    ) -> Dict[str, Any]:
        """Perform incremental validation."""
        pass
    
    @abstractmethod
    async def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        pass
    
    @abstractmethod
    async def configure_pipeline(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Configure pipeline settings."""
        pass