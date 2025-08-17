"""
Orchestrates the high-level model training, optimization, and validation workflow.
This class is a lean coordinator that delegates tasks to specialized components.
"""

# Standard library imports
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import joblib
import pandas as pd

# Local imports
from main.config.config_manager import get_config
from main.feature_pipeline.feature_store_compat import FeatureStore

# Use interfaces to avoid circular dependency
from main.interfaces.backtesting import (
    BacktestConfig,
    BacktestMode,
    BacktestResult,
    IBacktestEngineFactory,
)
from main.utils.core import get_logger

# from .hyperparameter_search import HyperparameterSearch  # Requires optuna
from .pipeline_results import PipelineResults  # Assuming this is the standard results object
from .train_pipeline import ModelTrainingPipeline

logger = get_logger(__name__)


class ModelTrainingOrchestrator:
    """Orchestrates the complete model training process."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        backtest_factory: Optional[IBacktestEngineFactory] = None,
    ):
        """
        Initializes the orchestrator and its required, specialized components.

        Args:
            config: Configuration dictionary
            backtest_factory: Factory for creating backtest engines (DI)
        """
        self.config = config or get_config()
        self.backtest_factory = backtest_factory

        # Instantiate all necessary components via dependency injection
        # This makes the orchestrator testable and breaks circular dependencies
        feature_store_path = self.config.get("paths", {}).get("feature_store", "data_lake/features")
        self.feature_store = FeatureStore(feature_store_path, self.config)

        # Note: Only create these if needed to avoid importing issues
        # self.hyperopt_runner = HyperparameterSearch(self.config)
        self.training_pipeline = ModelTrainingPipeline(self.config)

        # Initialize backtesting components
        self.backtest_config_template = BacktestConfig(
            start_date=None,  # Will be set dynamically
            end_date=None,  # Will be set dynamically
            initial_cash=self.config.get("backtesting", {}).get("initial_cash", 100000),
            symbols=[],  # Will be set dynamically
            mode=BacktestMode.PORTFOLIO,
            use_adjusted_prices=True,
            include_dividends=True,
            include_splits=True,
        )

        logger.info("ModelTrainingOrchestrator initialized with all components.")

    async def run_full_workflow(
        self, symbols: List[str], model_types: List[str], fast_mode: bool = False
    ) -> Dict:
        """
        Runs the entire training workflow from optimization to validation.
        """
        logger.info(f"--- Starting Full Training Workflow for models: {model_types} ---")

        # 1. Hyperparameter Optimization
        best_params = {}
        if not fast_mode:
            best_params = await self.run_hyperparameter_optimization(symbols, model_types)
        else:
            logger.info("Fast mode enabled, skipping hyperparameter optimization.")
            # In fast mode, the training pipeline will use default params from config

        # 2. Model Training
        training_results = await self.run_training(symbols, model_types, best_params)

        # 3. Save trained models
        await self.save_trained_models(training_results, symbols)

        # 4. Backtest Validation - TEMPORARILY DISABLED
        # TODO: Re-enable once standalone backtest system is implemented
        # trained_models = {name: result['model_artifact'] for name, result in training_results.items() if 'model_artifact' in result}
        # if trained_models:
        #     backtest_results = await self.run_backtest_validation(
        #         models=trained_models,
        #         symbols=symbols,
        #         lookback_days=self.config.get('backtesting', {}).get('lookback_days', 90)
        #     )
        #     # Integrate backtest metrics into final results
        #     for model_type, b_results in backtest_results.items():
        #         if model_type in training_results:
        #             training_results[model_type]['backtest_metrics'] = b_results

        logger.info("--- Full Training Workflow Completed ---")
        return training_results

    async def run_hyperparameter_optimization(
        self, symbols: List[str], model_types: List[str]
    ) -> Dict[str, Dict]:
        """Delegates hyperparameter optimization to the HyperparameterSearch utility."""
        logger.info("Orchestrator: Delegating to HyperparameterSearch...")
        all_best_params = {}

        # Prepare data once for all optimization runs
        lookback = self.config.get("hyperopt", {}).get("lookback_days", 180)
        request = {"feature_sets": ["all"], "symbols": symbols, "lookback_days": lookback}
        features_data = await self.feature_store.get_features(request)

        # Combine into a single DataFrame for training
        # This assumes your feature store returns a dict of DataFrames
        combined_df = pd.concat(features_data.values(), ignore_index=True)
        X = combined_df.drop(["target"], axis=1, errors="ignore")
        y = combined_df["target"]

        for model_type in model_types:
            study = self.hyperopt_runner.run_study(model_type, X, y)
            all_best_params[model_type] = study.best_params

        return all_best_params

    async def run_training(
        self, symbols: List[str], model_types: List[str], best_params: Dict
    ) -> Dict:
        """Delegates model training to the ModelTrainingPipeline."""
        logger.info("Orchestrator: Delegating to ModelTrainingPipeline...")
        all_training_results = {}

        # Prepare data once for all models
        lookback = self.config.get("training", {}).get("lookback_days", 365)
        request = {"feature_sets": ["all"], "symbols": symbols, "lookback_days": lookback}
        logger.info(f"Loading features with request: {request}")
        training_data = await self.feature_store.get_features(request)
        logger.info(f"Got training data keys: {training_data.keys()}")

        # Check if we have data
        if not training_data or all(df.empty for df in training_data.values()):
            logger.error("No training data loaded!")
            return {}

        combined_df = pd.concat(training_data.values(), ignore_index=True)
        logger.info(f"Combined dataframe shape: {combined_df.shape}")
        logger.info(f"Combined dataframe columns: {combined_df.columns.tolist()}")

        # Identify target and feature columns
        target_cols = [
            col
            for col in combined_df.columns
            if col.startswith(("next_", "up_down_", "directional_"))
        ]
        non_feature_cols = ["symbol", "timestamp", "feature_set", "interval"] + target_cols

        # Only include numeric columns as features
        feature_cols = []
        for col in combined_df.columns:
            if col not in non_feature_cols:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(combined_df[col]):
                    feature_cols.append(col)
                else:
                    logger.debug(
                        f"Excluding non-numeric column: {col} (dtype: {combined_df[col].dtype})"
                    )

        logger.info(f"Found {len(target_cols)} target columns: {target_cols[:5]}...")
        logger.info(f"Found {len(feature_cols)} feature columns: {feature_cols[:10]}...")

        # Use next-day return as default target for regression
        default_target = (
            "next_1d_return"
            if "next_1d_return" in target_cols
            else target_cols[0] if target_cols else None
        )

        if default_target is None:
            logger.error("No target columns found in data!")
            return {}

        logger.info(f"Using target column: {default_target}")

        for model_type in model_types:
            params = best_params.get(
                model_type, {}
            )  # Use optimized params or empty dict for defaults
            result = self.training_pipeline.train_model(
                model_type=model_type,
                training_data=combined_df,
                feature_columns=feature_cols,
                target_column=default_target,
                hyperparameters=params,
            )
            all_training_results[model_type] = result

        return all_training_results

    async def run_backtest_validation(
        self, models: Dict[str, Any], symbols: List[str], lookback_days: int = 90
    ) -> Dict[str, Dict]:
        """
        Validate trained models using backtesting.

        Args:
            models: Dictionary of model_type -> trained model
            symbols: List of symbols to backtest
            lookback_days: Number of days to look back for backtesting

        Returns:
            Dictionary of model_type -> backtest metrics
        """
        logger.info("Starting backtest validation for trained models")

        # Standard library imports
        from datetime import datetime, timedelta

        # Configure backtest period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        backtest_results = {}

        for model_type, model in models.items():
            logger.info(f"Backtesting {model_type} model")

            try:
                # Create strategy wrapper for the model
                # This assumes models have a predict method
                # Local imports
                from main.models.strategies.base_strategy import BaseStrategy

                class ModelStrategy(BaseStrategy):
                    def __init__(self, trained_model, config):
                        super().__init__(config)
                        self.model = trained_model

                    async def on_market_data(self, market_event):
                        # Simple strategy: predict and trade based on model output
                        symbol = market_event.data["symbol"]

                        # Get features for prediction
                        # In a real implementation, would get proper features
                        features = self._extract_features(market_event)

                        # Make prediction
                        prediction = self.model.predict(features)

                        # Generate orders based on prediction
                        if prediction > 0.5:  # Buy signal
                            return [self._create_buy_order(symbol, 100)]
                        elif prediction < -0.5:  # Sell signal
                            return [self._create_sell_order(symbol, 100)]

                        return []

                # Create backtest config
                backtest_config = BacktestConfig(
                    start_date=start_date,
                    end_date=end_date,
                    initial_cash=self.backtest_config_template.initial_cash,
                    symbols=symbols,
                    mode=self.backtest_config_template.mode,
                    use_adjusted_prices=self.backtest_config_template.use_adjusted_prices,
                )

                # Create strategy instance
                strategy = ModelStrategy(model, self.config)

                # Create backtest engine using factory
                if self.backtest_factory is None:
                    logger.warning(
                        f"No backtest factory provided, skipping backtest for {model_type}"
                    )
                    continue

                backtest_engine = self.backtest_factory.create(
                    config=backtest_config, strategy=strategy, data_source=self.feature_store
                )

                # Run backtest
                result = await backtest_engine.run()

                # Extract key metrics
                backtest_results[model_type] = {
                    "total_return": result.metrics.get("total_return", 0),
                    "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                    "max_drawdown": result.metrics.get("max_drawdown", 0),
                    "win_rate": result.metrics.get("win_rate", 0),
                    "total_trades": result.metrics.get("total_trades", 0),
                    "final_equity": result.metrics.get("final_equity", 0),
                }

                logger.info(
                    f"Backtest completed for {model_type}: Return={backtest_results[model_type]['total_return']:.2%}"
                )

            except Exception as e:
                logger.error(f"Backtest failed for {model_type}: {str(e)}")
                backtest_results[model_type] = {
                    "error": str(e),
                    "total_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                }

        return backtest_results

    async def save_trained_models(
        self, training_results: Dict[str, Dict], symbols: List[str]
    ) -> None:
        """
        Save trained models with metadata to disk.

        Args:
            training_results: Dictionary of model_type -> training result
            symbols: List of symbols used for training
        """
        try:
            # Get model storage path from config
            models_base_path = Path(self.config.get("ml.model_storage.path", "models"))
            models_base_path.mkdir(parents=True, exist_ok=True)

            # Create timestamp for this training run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for model_type, result in training_results.items():
                if "error" in result:
                    logger.warning(f"Skipping save for {model_type} due to training error")
                    continue

                # Create model-specific directory
                model_dir = models_base_path / model_type / timestamp
                model_dir.mkdir(parents=True, exist_ok=True)

                # Save model artifact
                model_path = model_dir / "model.pkl"
                joblib.dump(result["model_artifact"], model_path)
                logger.info(f"Saved {model_type} model to {model_path}")

                # Save scaler if present
                if "scaler_artifact" in result:
                    scaler_path = model_dir / "scaler.pkl"
                    joblib.dump(result["scaler_artifact"], scaler_path)
                    logger.info(f"Saved {model_type} scaler to {scaler_path}")

                # Save metadata
                metadata = {
                    "model_type": model_type,
                    "training_timestamp": timestamp,
                    "symbols": symbols,
                    "metrics": result.get("metrics", {}),
                    "feature_columns": result.get("feature_columns", []),
                    "training_metadata": result.get("training_metadata", {}),
                    "config": {
                        "lookback_days": self.config.get("training.lookback_days", 365),
                        "test_size": self.config.get("ml.training.train_split", 0.8),
                        "model_config": self.config.get(f"ml.models.{model_type}", {}),
                    },
                }

                metadata_path = model_dir / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved {model_type} metadata to {metadata_path}")

                # Update the training results with save paths
                result["model_path"] = str(model_path)
                result["metadata_path"] = str(metadata_path)

        except Exception as e:
            logger.error(f"Error saving trained models: {e}", exc_info=True)
