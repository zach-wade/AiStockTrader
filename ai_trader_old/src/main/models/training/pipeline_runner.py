# Standard library imports
import argparse
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional
import uuid

# Local imports
from main.config.config_manager import get_config

# REFACTOR: Import the orchestrators that the runner will now create.
from main.data_pipeline.processing.orchestrator import ProcessingOrchestrator
from main.feature_pipeline.feature_orchestrator import FeatureOrchestrator

from .pipeline_args import PipelineArgs, create_parser
from .pipeline_results import PipelineResults, ResultsPrinter
from .pipeline_stages import PipelineStages
from .training_orchestrator import ModelTrainingOrchestrator

logger = logging.getLogger(__name__)


class TrainingPipelineRunner:
    """
    Coordinates the training pipeline by initializing all necessary components
    and delegating stage execution to the PipelineStages class.
    """

    def __init__(self, args: Optional[argparse.Namespace] = None):
        """Initialize the pipeline runner."""
        self.args = args or create_parser().parse_args([])
        self.pipeline_args = PipelineArgs.from_namespace(self.args)
        self.system_config = get_config()
        self.results = PipelineResults(
            run_id=f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}",
            start_time=datetime.now(),
            config_snapshot={
                "cli_args": self.pipeline_args.__dict__,
                "system_config_version": self.system_config.get("version", "unknown"),
            },
        )
        self.stages: Optional[PipelineStages] = None
        self.printer = ResultsPrinter()

    async def run(self) -> int:
        """Run the complete training pipeline."""
        try:
            # REFACTOR: Initialization is now done here.
            await self._initialize_system_components()

            # Run pipeline stages conditionally
            if not self.pipeline_args.skip_download:
                await self.stages.run_data_collection(self.pipeline_args, self.results)

            if not self.pipeline_args.skip_features:
                await self.stages.run_feature_engineering(self.pipeline_args, self.results)

            if not self.pipeline_args.skip_hyperopt:
                await self.stages.run_hyperparameter_optimization(self.pipeline_args, self.results)

            if not self.pipeline_args.skip_training:
                await self.stages.run_model_training(self.pipeline_args, self.results)

            self.results.update_status(
                "completed" if not self.results.errors else "completed_with_errors"
            )
            return 0 if not self.results.errors else 1

        except Exception as e:
            logger.error(f"❌ Pipeline failed with a critical error: {e}", exc_info=True)
            self.results.add_error(f"Critical pipeline failure: {e}")
            self.results.update_status("failed")
            return 1
        finally:
            self.results.end_time = datetime.now()
            self.printer.print_summary(self.results)

            # Save final results
            reports_dir = Path(
                self.system_config.get("paths", {}).get("reports", "reports/training")
            )
            run_report_dir = reports_dir / self.results.run_id
            run_report_dir.mkdir(parents=True, exist_ok=True)
            self.printer.save_to_json(self.results, str(run_report_dir / "run_results.json"))

    async def _initialize_system_components(self):
        """
        REFACTOR: This method now creates all necessary orchestrator instances
        and injects them into the PipelineStages object.
        """
        logger.info("--- Initializing System Components ---")
        try:
            data_orchestrator = ProcessingOrchestrator(self.system_config)
            feature_integration = FeatureOrchestrator(config=self.system_config)
            training_orchestrator = ModelTrainingOrchestrator(self.system_config)

            self.stages = PipelineStages(
                data_orchestrator=data_orchestrator,
                feature_integration=feature_integration,
                training_orchestrator=training_orchestrator,
            )
            logger.info("✅ All system components initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize system components: {e}", exc_info=True)
            raise RuntimeError("Critical component initialization failed.") from e
