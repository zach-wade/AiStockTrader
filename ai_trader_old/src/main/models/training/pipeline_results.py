"""
Training Pipeline Results

Tracks and reports results from training pipeline execution including
metrics, errors, timings, and artifacts for each stage.
"""

# Standard library imports
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import pandas as pd
from tabulate import tabulate

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result from a single pipeline stage."""

    stage_name: str
    status: str  # 'success', 'failed', 'skipped'
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def complete(self, status: str = "success"):
        """Mark stage as complete."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.status = status


@dataclass
class ModelResult:
    """Result from training a single model."""

    model_name: str
    model_type: str
    training_time: float
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResults:
    """Complete results from a training pipeline run."""

    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, completed_with_errors

    # Configuration snapshot
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Stage results
    stages: Dict[str, StageResult] = field(default_factory=dict)

    # Model results
    models: Dict[str, ModelResult] = field(default_factory=dict)

    # Overall metrics
    overall_metrics: Dict[str, Any] = field(default_factory=dict)

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Data statistics
    data_stats: Dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage_name: str) -> StageResult:
        """Add a new stage to track."""
        stage = StageResult(stage_name=stage_name, status="running", start_time=datetime.now())
        self.stages[stage_name] = stage
        return stage

    def add_model(self, model_name: str, model_type: str) -> ModelResult:
        """Add a new model result."""
        model = ModelResult(model_name=model_name, model_type=model_type, training_time=0.0)
        self.models[model_name] = model
        return model

    def add_error(self, error: str, stage: Optional[str] = None):
        """Add an error message."""
        self.errors.append(error)
        if stage and stage in self.stages:
            self.stages[stage].errors.append(error)

    def add_warning(self, warning: str, stage: Optional[str] = None):
        """Add a warning message."""
        self.warnings.append(warning)
        if stage and stage in self.stages:
            self.stages[stage].warnings.append(warning)

    def update_status(self, status: str):
        """Update overall pipeline status."""
        self.status = status
        if status in ["completed", "failed", "completed_with_errors"]:
            self.end_time = datetime.now()

    def get_duration(self) -> Optional[timedelta]:
        """Get total pipeline duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def get_successful_models(self) -> List[str]:
        """Get list of successfully trained models."""
        return [name for name, model in self.models.items() if model.model_path is not None]

    def get_best_model(
        self, metric: str = "sharpe_ratio", higher_better: bool = True
    ) -> Optional[Tuple[str, ModelResult]]:
        """Get the best performing model by a specific metric."""
        best_model = None
        best_value = None

        for name, model in self.models.items():
            if metric in model.metrics:
                value = model.metrics[metric]
                if best_value is None:
                    best_value = value
                    best_model = (name, model)
                elif higher_better and value > best_value:
                    best_value = value
                    best_model = (name, model)
                elif not higher_better and value < best_value:
                    best_value = value
                    best_model = (name, model)

        return best_model

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert model results to DataFrame for analysis."""
        if not self.models:
            return pd.DataFrame()

        rows = []
        for name, model in self.models.items():
            row = {
                "model_name": name,
                "model_type": model.model_type,
                "training_time": model.training_time,
                **model.metrics,
            }
            rows.append(row)

        return pd.DataFrame(rows)


class ResultsPrinter:
    """Utility class for printing and saving pipeline results."""

    @staticmethod
    def print_summary(results: PipelineResults):
        """Print a formatted summary of results."""
        print("\n" + "=" * 80)
        print(f"Training Pipeline Results - {results.run_id}")
        print("=" * 80)

        # Overall status
        duration = results.get_duration()
        duration_str = str(duration).split(".")[0] if duration else "N/A"

        print(f"\nStatus: {results.status.upper()}")
        print(f"Duration: {duration_str}")
        print(f"Models Trained: {len(results.get_successful_models())}/{len(results.models)}")

        # Stage summary
        if results.stages:
            print("\nStage Summary:")
            stage_data = []
            for name, stage in results.stages.items():
                duration = f"{stage.duration_seconds:.1f}s" if stage.duration_seconds else "N/A"
                status_symbol = {
                    "success": "",
                    "failed": "",
                    "skipped": "-",
                    "running": "...",
                }.get(stage.status, "?")

                stage_data.append(
                    [
                        f"{status_symbol} {name}",
                        stage.status,
                        duration,
                        len(stage.errors),
                        len(stage.warnings),
                    ]
                )

            print(
                tabulate(
                    stage_data,
                    headers=["Stage", "Status", "Duration", "Errors", "Warnings"],
                    tablefmt="simple",
                )
            )

        # Model results
        if results.models:
            print("\nModel Performance:")
            df = results.to_dataframe()

            # Select key metrics to display
            display_cols = ["model_name", "model_type"]
            metric_cols = [
                col for col in df.columns if col not in display_cols and col != "training_time"
            ]

            # Limit to important metrics
            important_metrics = ["sharpe_ratio", "accuracy", "precision", "recall", "auc"]
            metric_cols = [col for col in important_metrics if col in metric_cols]

            if metric_cols:
                display_df = df[display_cols + metric_cols].round(4)
                print(tabulate(display_df, headers="keys", tablefmt="simple", showindex=False))

            # Best model
            best = results.get_best_model()
            if best:
                name, model = best
                print(
                    f"\nBest Model: {name} (Sharpe Ratio: {model.metrics.get('sharpe_ratio', 'N/A'):.4f})"
                )

        # Data statistics
        if results.data_stats:
            print("\nData Statistics:")
            for key, value in results.data_stats.items():
                print(f"  {key}: {value}")

        # Errors and warnings
        if results.errors:
            print(f"\nErrors ({len(results.errors)}):")
            for error in results.errors[:5]:  # Show first 5
                print(f"  - {error}")
            if len(results.errors) > 5:
                print(f"  ... and {len(results.errors) - 5} more")

        if results.warnings:
            print(f"\nWarnings ({len(results.warnings)}):")
            for warning in results.warnings[:5]:  # Show first 5
                print(f"  - {warning}")
            if len(results.warnings) > 5:
                print(f"  ... and {len(results.warnings) - 5} more")

        print("\n" + "=" * 80)

    @staticmethod
    def save_to_json(results: PipelineResults, filepath: str):
        """Save results to JSON file."""
        try:
            # Convert to dict and handle datetime serialization
            data = results.to_dict()

            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, Path):
                    return str(obj)
                return str(obj)

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=json_serializer)

            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    @staticmethod
    def save_to_html(results: PipelineResults, filepath: str):
        """Save results as HTML report."""
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Pipeline Results - {results.run_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: green; }}
        .failed {{ color: red; }}
        .warning {{ color: orange; }}
        .metric {{ font-family: monospace; }}
    </style>
</head>
<body>
    <h1>Training Pipeline Results</h1>
    <p><strong>Run ID:</strong> {results.run_id}</p>
    <p><strong>Status:</strong> <span class="{results.status}">{results.status.upper()}</span></p>
    <p><strong>Duration:</strong> {str(results.get_duration()).split('.')[0] if results.get_duration() else 'N/A'}</p>

    <h2>Model Performance</h2>
"""

            if results.models:
                df = results.to_dataframe()
                html_content += df.to_html(classes="metrics", index=False)
            else:
                html_content += "<p>No models trained.</p>"

            # Add errors if any
            if results.errors:
                html_content += f"""
    <h2>Errors</h2>
    <ul class="failed">
"""
                for error in results.errors:
                    html_content += f"        <li>{error}</li>\n"
                html_content += "    </ul>\n"

            html_content += """
</body>
</html>
"""

            with open(filepath, "w") as f:
                f.write(html_content)

            logger.info(f"HTML report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save HTML report: {e}")

    @staticmethod
    def generate_model_comparison_plot(results: PipelineResults, output_path: str):
        """Generate a plot comparing model performance."""
        try:
            # Third-party imports
            import matplotlib.pyplot as plt
            import seaborn as sns

            df = results.to_dataframe()
            if df.empty:
                return

            # Create figure with subplots for different metrics
            metrics = ["sharpe_ratio", "accuracy", "precision", "recall"]
            available_metrics = [m for m in metrics if m in df.columns]

            if not available_metrics:
                return

            fig, axes = plt.subplots(
                1, len(available_metrics), figsize=(5 * len(available_metrics), 5)
            )
            if len(available_metrics) == 1:
                axes = [axes]

            for idx, metric in enumerate(available_metrics):
                ax = axes[idx]
                df_sorted = df.sort_values(metric, ascending=False)
                sns.barplot(data=df_sorted, x="model_name", y=metric, ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel("Model")
                ax.set_ylabel("Value")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

            logger.info(f"Model comparison plot saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")
