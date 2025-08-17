"""
Training Pipeline Arguments

Defines command-line arguments and configuration for the training pipeline.
Provides a clean interface for configuring pipeline runs with validation.
"""

# Standard library imports
import argparse
from dataclasses import dataclass, field
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PipelineArgs:
    """
    Data class containing all training pipeline arguments.
    Provides type safety and validation for pipeline configuration.
    """

    # Pipeline control flags
    skip_download: bool = False
    skip_features: bool = False
    skip_hyperopt: bool = False
    skip_training: bool = False
    skip_validation: bool = False
    skip_deployment: bool = False

    # Data collection parameters
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    data_sources: List[str] = field(default_factory=lambda: ["yahoo", "alpaca"])

    # Feature engineering parameters
    feature_sets: List[str] = field(
        default_factory=lambda: ["technical", "sentiment", "microstructure"]
    )
    feature_lookback: int = 20
    feature_cache: bool = True

    # Model training parameters
    models: List[str] = field(default_factory=lambda: ["xgboost", "lightgbm", "random_forest"])
    target: str = "next_day_return"
    train_split: float = 0.8
    validation_split: float = 0.1
    random_seed: int = 42

    # Hyperparameter optimization
    hyperopt_trials: int = 100
    hyperopt_algorithm: str = "tpe"  # Tree-structured Parzen Estimator
    hyperopt_metric: str = "sharpe_ratio"
    hyperopt_direction: str = "maximize"

    # Advanced options
    parallel_jobs: int = -1  # Use all CPU cores
    memory_limit: Optional[int] = None  # MB
    gpu_enabled: bool = False
    debug_mode: bool = False

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("models/training_runs"))
    experiment_name: Optional[str] = None
    save_artifacts: bool = True

    # Ensemble configuration
    ensemble_enabled: bool = True
    ensemble_method: str = "voting"  # voting, stacking, blending
    ensemble_weights: Optional[Dict[str, float]] = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "PipelineArgs":
        """Create PipelineArgs from argparse namespace."""
        # Convert namespace to dict and filter out None values
        args_dict = {k: v for k, v in vars(args).items() if v is not None}

        # Handle special conversions
        if "symbols" in args_dict and isinstance(args_dict["symbols"], str):
            args_dict["symbols"] = [s.strip() for s in args_dict["symbols"].split(",")]

        if "feature_sets" in args_dict and isinstance(args_dict["feature_sets"], str):
            args_dict["feature_sets"] = [s.strip() for s in args_dict["feature_sets"].split(",")]

        if "models" in args_dict and isinstance(args_dict["models"], str):
            args_dict["models"] = [m.strip() for m in args_dict["models"].split(",")]

        if "data_sources" in args_dict and isinstance(args_dict["data_sources"], str):
            args_dict["data_sources"] = [s.strip() for s in args_dict["data_sources"].split(",")]

        if "output_dir" in args_dict and isinstance(args_dict["output_dir"], str):
            args_dict["output_dir"] = Path(args_dict["output_dir"])

        # Parse dates
        if "start_date" in args_dict and isinstance(args_dict["start_date"], str):
            args_dict["start_date"] = datetime.strptime(args_dict["start_date"], "%Y-%m-%d").date()

        if "end_date" in args_dict and isinstance(args_dict["end_date"], str):
            args_dict["end_date"] = datetime.strptime(args_dict["end_date"], "%Y-%m-%d").date()

        return cls(**args_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, date):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    def validate(self) -> List[str]:
        """Validate arguments and return list of errors."""
        errors = []

        # Validate dates
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            errors.append("start_date must be before end_date")

        # Validate splits
        if self.train_split <= 0 or self.train_split >= 1:
            errors.append("train_split must be between 0 and 1")

        if self.validation_split < 0 or self.validation_split >= 1:
            errors.append("validation_split must be between 0 and 1")

        if self.train_split + self.validation_split >= 1:
            errors.append("train_split + validation_split must be less than 1")

        # Validate hyperopt parameters
        if self.hyperopt_trials < 1:
            errors.append("hyperopt_trials must be at least 1")

        if self.hyperopt_algorithm not in ["tpe", "random", "anneal"]:
            errors.append(f"Unknown hyperopt_algorithm: {self.hyperopt_algorithm}")

        if self.hyperopt_direction not in ["maximize", "minimize"]:
            errors.append(f"hyperopt_direction must be 'maximize' or 'minimize'")

        # Validate ensemble
        if self.ensemble_method not in ["voting", "stacking", "blending"]:
            errors.append(f"Unknown ensemble_method: {self.ensemble_method}")

        # Validate models
        valid_models = ["xgboost", "lightgbm", "random_forest", "gradient_boosting", "neural_net"]
        for model in self.models:
            if model not in valid_models:
                errors.append(f"Unknown model: {model}")

        # Validate feature sets
        valid_features = [
            "technical",
            "sentiment",
            "news",
            "microstructure",
            "cross_sectional",
            "options",
        ]
        for feature_set in self.feature_sets:
            if feature_set not in valid_features:
                errors.append(f"Unknown feature set: {feature_set}")

        return errors


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for training pipeline."""
    parser = argparse.ArgumentParser(
        description="AI Trader Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline control
    control_group = parser.add_argument_group("Pipeline Control")
    control_group.add_argument(
        "--skip-download", action="store_true", help="Skip data download stage"
    )
    control_group.add_argument(
        "--skip-features", action="store_true", help="Skip feature engineering stage"
    )
    control_group.add_argument(
        "--skip-hyperopt", action="store_true", help="Skip hyperparameter optimization"
    )
    control_group.add_argument(
        "--skip-training", action="store_true", help="Skip model training stage"
    )
    control_group.add_argument(
        "--skip-validation", action="store_true", help="Skip model validation stage"
    )
    control_group.add_argument(
        "--skip-deployment", action="store_true", help="Skip model deployment stage"
    )

    # Data parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument(
        "--symbols",
        type=str,
        default="AAPL,GOOGL,MSFT,AMZN,TSLA",
        help="Comma-separated list of symbols",
    )
    data_group.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    data_group.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    data_group.add_argument(
        "--data-sources",
        type=str,
        default="yahoo,alpaca",
        help="Comma-separated list of data sources",
    )

    # Feature parameters
    feature_group = parser.add_argument_group("Feature Engineering")
    feature_group.add_argument(
        "--feature-sets",
        type=str,
        default="technical,sentiment,microstructure",
        help="Comma-separated list of feature sets",
    )
    feature_group.add_argument(
        "--feature-lookback", type=int, default=20, help="Lookback period for features"
    )
    feature_group.add_argument(
        "--no-feature-cache",
        dest="feature_cache",
        action="store_false",
        help="Disable feature caching",
    )

    # Model parameters
    model_group = parser.add_argument_group("Model Training")
    model_group.add_argument(
        "--models",
        type=str,
        default="xgboost,lightgbm,random_forest",
        help="Comma-separated list of models to train",
    )
    model_group.add_argument(
        "--target", type=str, default="next_day_return", help="Target variable for prediction"
    )
    model_group.add_argument(
        "--train-split", type=float, default=0.8, help="Training data split ratio"
    )
    model_group.add_argument(
        "--validation-split", type=float, default=0.1, help="Validation data split ratio"
    )
    model_group.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Hyperparameter optimization
    hyperopt_group = parser.add_argument_group("Hyperparameter Optimization")
    hyperopt_group.add_argument(
        "--hyperopt-trials",
        type=int,
        default=100,
        help="Number of hyperparameter optimization trials",
    )
    hyperopt_group.add_argument(
        "--hyperopt-algorithm",
        type=str,
        default="tpe",
        choices=["tpe", "random", "anneal"],
        help="Hyperparameter optimization algorithm",
    )
    hyperopt_group.add_argument(
        "--hyperopt-metric", type=str, default="sharpe_ratio", help="Metric to optimize"
    )
    hyperopt_group.add_argument(
        "--hyperopt-direction",
        type=str,
        default="maximize",
        choices=["maximize", "minimize"],
        help="Optimization direction",
    )

    # Advanced options
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--parallel-jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores)"
    )
    advanced_group.add_argument("--memory-limit", type=int, help="Memory limit in MB")
    advanced_group.add_argument(
        "--gpu", dest="gpu_enabled", action="store_true", help="Enable GPU acceleration"
    )
    advanced_group.add_argument(
        "--debug",
        dest="debug_mode",
        action="store_true",
        help="Enable debug mode with verbose output",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="models/training_runs",
        help="Output directory for results",
    )
    output_group.add_argument("--experiment-name", type=str, help="Name for this experiment")
    output_group.add_argument(
        "--no-artifacts",
        dest="save_artifacts",
        action="store_false",
        help="Do not save training artifacts",
    )

    # Ensemble configuration
    ensemble_group = parser.add_argument_group("Ensemble Configuration")
    ensemble_group.add_argument(
        "--no-ensemble",
        dest="ensemble_enabled",
        action="store_false",
        help="Disable ensemble models",
    )
    ensemble_group.add_argument(
        "--ensemble-method",
        type=str,
        default="voting",
        choices=["voting", "stacking", "blending"],
        help="Ensemble method to use",
    )
    ensemble_group.add_argument(
        "--ensemble-weights", type=str, help="JSON string of model weights for ensemble"
    )

    return parser


def parse_args(args_list: Optional[List[str]] = None) -> PipelineArgs:
    """Parse command line arguments and return PipelineArgs."""
    parser = create_parser()
    args = parser.parse_args(args_list)

    # Parse ensemble weights if provided
    if args.ensemble_weights:
        try:
            args.ensemble_weights = json.loads(args.ensemble_weights)
        except json.JSONDecodeError:
            parser.error(f"Invalid JSON for ensemble_weights: {args.ensemble_weights}")

    # Convert to PipelineArgs
    pipeline_args = PipelineArgs.from_namespace(args)

    # Validate
    errors = pipeline_args.validate()
    if errors:
        for error in errors:
            parser.error(error)

    return pipeline_args


if __name__ == "__main__":
    # Example usage
    args = parse_args()
    print("Pipeline Arguments:")
    print(json.dumps(args.to_dict(), indent=2))
