"""
Individual pipeline stage implementations.
This refactored version uses dependency injection to receive its orchestrators,
making it a stateless, procedural coordinator.
"""
import logging
from typing import Dict, Any
from datetime import datetime

# All imports can now be at the top level, as circular dependencies are resolved.
from main.data_pipeline.processing.orchestrator import ProcessingOrchestrator
from main.feature_pipeline.feature_orchestrator import FeatureOrchestrator
from .training_orchestrator import ModelTrainingOrchestrator
from .pipeline_args import PipelineArgs
from .pipeline_results import PipelineResults

logger = logging.getLogger(__name__)


class PipelineStages:
    """Handles execution of individual pipeline stages by delegating to specialized orchestrators."""
    
    def __init__(self,
                 data_orchestrator: ProcessingOrchestrator,
                 feature_integration: FeatureOrchestrator,
                 training_orchestrator: ModelTrainingOrchestrator):
        """
        Initialize pipeline stages with pre-built orchestrator components.
        """
        self.data_orchestrator = data_orchestrator
        self.feature_integration = feature_integration
        self.training_orchestrator = training_orchestrator
        logger.info("PipelineStages initialized with all necessary orchestrators.")
    
    async def run_data_collection(self, args: PipelineArgs, results: PipelineResults):
        """Run data collection phase."""
        logger.info("--- STAGE: DATA COLLECTION ---")
        start_time = datetime.now()
        try:
            # The logic for running the stage is preserved.
            await self.data_orchestrator.run_stage(
                stage_name='daily_and_intraday',
                symbols=args.symbols,
                lookback_days=args.lookback_days
            )
            results.metrics['data_collection'] = {"status": "success", "symbols_processed": len(args.symbols)}
        except Exception as e:
            logger.error(f"Data collection stage failed: {e}", exc_info=True)
            results.add_error(f"Data collection failed: {e}")
        finally:
            results.metrics.setdefault('timing', {})['data_collection'] = (datetime.now() - start_time).total_seconds()
    
    async def run_feature_engineering(self, args: PipelineArgs, results: PipelineResults):
        """Run feature engineering phase."""
        logger.info("--- STAGE: FEATURE ENGINEERING ---")
        start_time = datetime.now()
        try:
            features_summary = await self.feature_integration.calculate_features_for_symbols(
                symbols=args.symbols,
                lookback_days=args.lookback_days,
                calculators=args.feature_calculators
            )
            results.metrics['feature_engineering'] = features_summary
        except Exception as e:
            logger.error(f"Feature engineering stage failed: {e}", exc_info=True)
            results.add_error(f"Feature engineering failed: {e}")
        finally:
            results.metrics.setdefault('timing', {})['feature_engineering'] = (datetime.now() - start_time).total_seconds()
    
    async def run_hyperparameter_optimization(self, args: PipelineArgs, results: PipelineResults):
        """Run hyperparameter optimization phase."""
        logger.info("--- STAGE: HYPERPARAMETER OPTIMIZATION ---")
        start_time = datetime.now()
        try:
            # The training orchestrator now handles the logic for fast mode vs. full optimization.
            best_params = await self.training_orchestrator.run_hyperparameter_optimization(
                symbols=args.symbols,
                model_types=args.model_types,
                fast_mode=args.fast_mode
            )
            results.metrics['hyperparameter_optimization'] = best_params
        except Exception as e:
            logger.error(f"Hyperparameter optimization stage failed: {e}", exc_info=True)
            results.add_error(f"Hyperparameter optimization failed: {e}")
        finally:
            results.metrics.setdefault('timing', {})['hyperopt'] = (datetime.now() - start_time).total_seconds()
    
    async def run_model_training(self, args: PipelineArgs, results: PipelineResults):
        """Run model training phase."""
        logger.info("--- STAGE: MODEL TRAINING ---")
        start_time = datetime.now()
        try:
            # The training orchestrator uses the results from the previous stage.
            best_params = results.metrics.get('hyperparameter_optimization', {})
            training_summary = await self.training_orchestrator.run_training(
                symbols=args.symbols,
                model_types=args.model_types,
                best_params=best_params
            )
            results.model_summaries = training_summary
        except Exception as e:
            logger.error(f"Model training stage failed: {e}", exc_info=True)
            results.add_error(f"Model training failed: {e}")
        finally:
            results.metrics.setdefault('timing', {})['training'] = (datetime.now() - start_time).total_seconds()