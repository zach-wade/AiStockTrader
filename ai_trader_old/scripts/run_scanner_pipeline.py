#!/usr/bin/env python3
"""
Scanner Pipeline Execution Script

Runs the complete scanner pipeline from Layer 0 to Layer 3.
Supports various execution modes and command-line options.
"""

# Standard library imports
import argparse
import asyncio
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sys

# Third-party imports
import yaml

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party imports
from omegaconf import DictConfig, OmegaConf

# Local imports
from src.main.config.config_manager import get_config
from src.main.scanners.scanner_pipeline import PipelineResult, ScannerPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/scanner_pipeline_{datetime.now().strftime("%Y%m%d")}.log'),
    ],
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Scanner Pipeline from Layer 0 to Layer 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_scanner_pipeline.py

  # Run in test mode (limited symbols)
  python scripts/run_scanner_pipeline.py --test

  # Run with custom config
  python scripts/run_scanner_pipeline.py --config config/scanner_pipeline_custom.yaml

  # Run specific layers only
  python scripts/run_scanner_pipeline.py --start-layer 1 --end-layer 2

  # Run with verbose output
  python scripts/run_scanner_pipeline.py --verbose

  # Run without saving results
  python scripts/run_scanner_pipeline.py --no-save
        """,
    )

    # Execution modes
    parser.add_argument("--test", action="store_true", help="Run in test mode with limited symbols")

    parser.add_argument("--dry-run", action="store_true", help="Run without updating database")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to custom configuration file")

    parser.add_argument(
        "--symbols", type=str, nargs="+", help="Specific symbols to scan (overrides Layer 0)"
    )

    # Layer control
    parser.add_argument(
        "--start-layer",
        type=str,
        choices=["0", "1", "1.5", "2", "3"],
        default="0",
        help="Starting layer (default: 0)",
    )

    parser.add_argument(
        "--end-layer",
        type=str,
        choices=["0", "1", "1.5", "2", "3"],
        default="3",
        help="Ending layer (default: 3)",
    )

    # Output control
    parser.add_argument("--no-save", action="store_true", help="Do not save intermediate results")

    parser.add_argument("--output-dir", type=str, help="Custom output directory for results")

    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "both"],
        default="json",
        help="Output format for results (default: json)",
    )

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument("--quiet", action="store_true", help="Suppress non-error output")

    # Performance
    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel execution where possible"
    )

    parser.add_argument("--timeout", type=int, help="Overall pipeline timeout in seconds")

    return parser.parse_args()


def load_custom_config(config_path: str | None) -> DictConfig | None:
    """Load custom configuration from file."""
    if not config_path:
        return None

    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return None

    try:
        with open(config_file) as f:
            if config_file.suffix == ".yaml" or config_file.suffix == ".yml":
                config_dict = yaml.safe_load(f)
            elif config_file.suffix == ".json":
                config_dict = json.load(f)
            else:
                logger.error(f"Unsupported config format: {config_file.suffix}")
                return None

        return OmegaConf.create(config_dict)

    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return None


def merge_configs(
    base_config: DictConfig, custom_config: DictConfig | None, args: argparse.Namespace
) -> DictConfig:
    """Merge configurations with command-line overrides."""
    # Start with base config
    config = OmegaConf.create(base_config)

    # Merge custom config if provided
    if custom_config:
        config = OmegaConf.merge(config, custom_config)

    # Apply command-line overrides
    if args.test:
        config.scanner_pipeline.test_mode.symbols_limit = 100

    if args.no_save:
        config.scanner_pipeline.save_intermediate_results = False

    if args.output_dir:
        config.scanner_pipeline.output_dir = args.output_dir

    if args.parallel:
        config.scanner_pipeline.performance.parallel_execution = True

    if args.timeout:
        config.scanner_pipeline.performance.timeout_seconds = args.timeout

    return config


def format_results_summary(result: PipelineResult) -> str:
    """Format pipeline results for display."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("Scanner Pipeline Results Summary")
    lines.append("=" * 80)

    # Overall status
    status_emoji = "✅" if result.success else "❌"
    lines.append(f"\nStatus: {status_emoji} {'SUCCESS' if result.success else 'FAILED'}")
    lines.append(f"Duration: {result.total_duration:.2f} seconds")
    lines.append(f"Final Opportunities: {len(result.final_opportunities)}")

    # Layer progression
    lines.append("\nLayer Progression:")
    lines.append("-" * 40)

    for layer in result.layer_results:
        reduction = ""
        if layer.input_count > 0:
            reduction_pct = (1 - layer.output_count / layer.input_count) * 100
            reduction = f" (-{reduction_pct:.1f}%)"

        status = "✓" if not layer.errors else "✗"
        lines.append(
            f"  {status} Layer {layer.layer_number}: {layer.input_count:,} → "
            f"{layer.output_count:,}{reduction} ({layer.execution_time:.1f}s)"
        )

    # Funnel metrics
    funnel = result.metadata.get("funnel_reduction", {})
    if funnel:
        lines.append("\nFunnel Metrics:")
        lines.append("-" * 40)
        lines.append(f"  Total Reduction: {funnel.get('total_reduction', 0) * 100:.1f}%")
        lines.append(f"  Selection Rate: {funnel.get('final_selection_rate', 0) * 100:.3f}%")

    # Top opportunities
    if result.final_opportunities:
        lines.append("\nTop 5 Trading Opportunities:")
        lines.append("-" * 40)

        for i, opp in enumerate(result.final_opportunities[:5], 1):
            symbol = opp.get("symbol", "N/A")
            score = opp.get("score", 0)
            rvol = opp.get("rvol", 0)
            price_change = opp.get("price_change_pct", 0)

            lines.append(
                f"  {i}. {symbol:<6} | Score: {score:5.2f} | "
                f"RVOL: {rvol:4.1f}x | Change: {price_change:+5.2f}%"
            )

    # Errors if any
    if result.errors:
        lines.append("\nErrors:")
        lines.append("-" * 40)
        for error in result.errors:
            lines.append(f"  ⚠️  {error}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


async def save_results(result: PipelineResult, format_type: str, output_dir: str | None):
    """Save results in specified format."""
    if not output_dir:
        output_dir = "data/scanner_pipeline"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON
    if format_type in ["json", "both"]:
        json_file = output_path / f"pipeline_results_{timestamp}.json"

        # Convert to dict
        result_dict = {
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "total_duration": result.total_duration,
            "success": result.success,
            "errors": result.errors,
            "metadata": result.metadata,
            "final_symbols": result.final_symbols,
            "final_opportunities": result.final_opportunities,
            "layer_results": [
                {
                    "layer_name": layer.layer_name,
                    "layer_number": layer.layer_number,
                    "input_count": layer.input_count,
                    "output_count": layer.output_count,
                    "execution_time": layer.execution_time,
                    "errors": layer.errors,
                }
                for layer in result.layer_results
            ],
        }

        with open(json_file, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Results saved to: {json_file}")

    # Save as CSV
    if format_type in ["csv", "both"]:
        # Save opportunities as CSV
        if result.final_opportunities:
            # Third-party imports
            import pandas as pd

            csv_file = output_path / f"opportunities_{timestamp}.csv"
            df = pd.DataFrame(result.final_opportunities)
            df.to_csv(csv_file, index=False)

            logger.info(f"Opportunities saved to: {csv_file}")


async def run_pipeline(args: argparse.Namespace) -> int:
    """Run the scanner pipeline with given arguments."""
    try:
        # Load configurations
        base_config = get_config()
        custom_config = load_custom_config(args.config)
        config = merge_configs(base_config, custom_config, args)

        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        elif args.quiet:
            logging.getLogger().setLevel(logging.ERROR)

        # Create pipeline instance
        logger.info("Initializing Scanner Pipeline...")
        pipeline = ScannerPipeline(config=config, test_mode=args.test)

        # Run pipeline
        logger.info("Starting Scanner Pipeline execution...")
        start_time = datetime.now(UTC)

        result = await pipeline.run()

        # Display results
        if not args.quiet:
            print(format_results_summary(result))

        # Save results
        if not args.no_save:
            await save_results(result, args.format, args.output_dir)

        # Return exit code
        return 0 if result.success else 1

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    args = parse_arguments()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Run async pipeline
    exit_code = asyncio.run(run_pipeline(args))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
