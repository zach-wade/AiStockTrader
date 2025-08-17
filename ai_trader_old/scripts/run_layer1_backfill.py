#!/usr/bin/env python3

"""
Layer 1 Backfill Runner
Reads configuration from config/layer1_backfill.yaml and executes backfill stages
"""

# Standard library imports
import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import subprocess
import sys

# Third-party imports
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging first (before imports that might fail)
log_dir = Path("logs/backfill")
log_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

log_file = log_dir / f'layer1_backfill_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
logger = logging.getLogger(__name__)

# Now import project modules with error handling
try:
    # Local imports
    from main.config import get_config
    from main.universe.universe_manager import UniverseManager
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error(
        "Make sure you're running from the project root and virtual environment is activated"
    )
    print("\nTo run this script:")
    print("  cd /Users/zachwade/StockMonitoring/ai_trader")
    print("  source venv/bin/activate")
    print("  python scripts/run_layer1_backfill.py")
    sys.exit(1)


class Layer1BackfillRunner:
    """Orchestrates Layer 1 backfill based on configuration"""

    def __init__(self, config_path: str = "config/layer1_backfill.yaml"):
        """Initialize with configuration file"""
        self.config_path = Path(config_path)
        self.load_config()
        self.failed_stages = []
        self.completed_stages = []

    def load_config(self):
        """Load backfill configuration from YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        self.backfill_config = self.config.get("layer1_backfill", {})
        self.stages = self.backfill_config.get("stages", {})
        self.execution_config = self.backfill_config.get("execution", {})

    async def get_layer1_symbols_count(self) -> int:
        """Get count of Layer 1 qualified symbols"""
        config = get_config()
        um = UniverseManager(config)
        try:
            symbols = await um.get_qualified_symbols("1")
            return len(symbols)
        finally:
            await um.close()

    def check_disk_space(self, required_gb: int = 150) -> bool:
        """Check if sufficient disk space is available"""
        # Standard library imports
        import shutil

        stat = shutil.disk_usage(".")
        available_gb = stat.free // (1024**3)

        if available_gb < required_gb:
            logger.error(
                f"Insufficient disk space. Required: {required_gb}GB, Available: {available_gb}GB"
            )
            return False
        else:
            logger.info(f"Disk space check passed. Available: {available_gb}GB")
            return True

    def run_backfill_stage(self, stage_name: str, stage_config: dict) -> bool:
        """Run a single backfill stage"""
        if not stage_config.get("enabled", True):
            logger.info(f"Stage '{stage_name}' is disabled in configuration")
            return True

        description = stage_config.get("description", "")
        lookback_days = stage_config.get("lookback_days", 365)
        rationale = stage_config.get("rationale", "")

        logger.info("\n" + "=" * 70)
        logger.info(f"📊 STAGE: {stage_name.upper()}")
        logger.info("=" * 70)
        logger.info(f"Description: {description}")
        logger.info(f"Lookback days: {lookback_days} (~{lookback_days/365:.1f} years)")
        if rationale:
            logger.info(f"Rationale: {rationale.strip()}")
        logger.info("-" * 70)

        # Build command
        cmd = [
            sys.executable,
            "ai_trader.py",
            "backfill",
            "--stage",
            stage_name,
            "--symbols",
            "layer1",  # Always use Layer 1, not Layer 0!
            "--days",
            str(lookback_days),
        ]

        if self.backfill_config.get("force_refresh", False):
            cmd.append("--force")

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("Starting backfill... (this may take several minutes)")

        try:
            # Run the command with real-time output
            start_time = datetime.now()

            # Run with subprocess and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Stream output in real-time
            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)

                # Only print important lines to console
                if any(
                    keyword in line
                    for keyword in ["✅", "❌", "ERROR", "Downloaded", "records", "symbols"]
                ):
                    print(f"  {line}")

            process.wait()

            duration = (datetime.now() - start_time).total_seconds()

            if process.returncode == 0:
                logger.info(
                    f"\n✅ Stage '{stage_name}' completed successfully in {duration:.1f} seconds"
                )

                # Extract summary info from output
                for line in output_lines[-20:]:  # Check last 20 lines for summary
                    if "Downloaded" in line and "records" in line:
                        logger.info(f"   {line.strip()}")

                self.completed_stages.append(stage_name)
                return True
            else:
                raise subprocess.CalledProcessError(process.returncode, cmd)

        except subprocess.CalledProcessError as e:
            logger.error(f"\n❌ Stage '{stage_name}' failed with exit code {e.returncode}")

            # Show last few lines of output for debugging
            if output_lines:
                logger.error("Last output lines:")
                for line in output_lines[-10:]:
                    logger.error(f"  {line}")

            self.failed_stages.append(stage_name)

            # Continue on error if configured
            if self.backfill_config.get("continue_on_error", True):
                logger.warning("Continuing with next stage due to continue_on_error=true")
                return False
            else:
                raise

    def run_all_stages(self):
        """Run all enabled stages in priority order"""
        # Sort stages by priority
        sorted_stages = sorted(self.stages.items(), key=lambda x: x[1].get("priority", 999))

        for stage_name, stage_config in sorted_stages:
            self.run_backfill_stage(stage_name, stage_config)

    def print_summary(self):
        """Print execution summary"""
        logger.info("\n" + "=" * 70)
        logger.info("BACKFILL SUMMARY")
        logger.info("=" * 70)

        if self.completed_stages:
            logger.info(f"✅ Completed stages ({len(self.completed_stages)}):")
            for stage in self.completed_stages:
                logger.info(f"   - {stage}")

        if self.failed_stages:
            logger.info(f"\n❌ Failed stages ({len(self.failed_stages)}):")
            for stage in self.failed_stages:
                logger.info(f"   - {stage}")

            logger.info("\nTo retry failed stages:")
            for stage in self.failed_stages:
                days = self.stages[stage].get("lookback_days", 365)
                logger.info(
                    f"   python ai_trader.py backfill --stage {stage} --symbols layer1 --days {days}"
                )

        # Storage estimates
        storage_estimates = self.backfill_config.get("storage_estimates", {})
        logger.info("\nEstimated storage usage:")
        for stage in self.completed_stages:
            if stage in storage_estimates:
                logger.info(f"   {stage}: {storage_estimates[stage]}")


async def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("🚀 LAYER 1 COMPREHENSIVE BACKFILL")
    print("=" * 70)

    runner = Layer1BackfillRunner()

    # Show configuration summary
    logger.info("Configuration loaded from: config/layer1_backfill.yaml")
    logger.info(f"Log file: {log_file.name}")

    # Check prerequisites
    logger.info("\n📋 Checking prerequisites...")

    # Check disk space
    if not runner.check_disk_space(150):
        logger.error("Please free up disk space before running backfill")
        logger.error("This backfill requires approximately 60-120GB of storage")
        return 1

    # Check Layer 1 symbols
    logger.info("\n🔍 Checking Layer 1 qualified symbols...")
    try:
        symbol_count = await runner.get_layer1_symbols_count()
    except Exception as e:
        logger.error(f"Failed to get Layer 1 symbols: {e}")
        logger.error("Make sure the database is running and Layer 1 scan has been completed")
        return 1

    if symbol_count == 0:
        logger.error("❌ No Layer 1 qualified symbols found in database")
        logger.info("\nTo populate Layer 1 symbols:")
        logger.info("  1. Run full scanner pipeline: python ai_trader.py scan --full")
        logger.info("  2. Or just universe scan: python ai_trader.py universe --populate")
        return 1

    logger.info(f"✅ Found {symbol_count} Layer 1 qualified symbols")

    # Show what will be downloaded
    logger.info("\n📦 Stages to be executed:")
    enabled_stages = [
        (name, config) for name, config in runner.stages.items() if config.get("enabled", True)
    ]

    total_days = 0
    for i, (stage_name, stage_config) in enumerate(
        sorted(enabled_stages, key=lambda x: x[1].get("priority", 999))
    ):
        days = stage_config.get("lookback_days", 365)
        total_days += days
        logger.info(
            f"  {i+1}. {stage_name}: {days} days of {stage_config.get('description', 'data')}"
        )

    # Confirm before proceeding
    if os.environ.get("SKIP_CONFIRM") != "true":
        print("\n⚠️  WARNING: Large Data Download")
        print(f"  - Symbols: {symbol_count} Layer 1 qualified stocks")
        print(f"  - Stages: {len(enabled_stages)} data types")
        print("  - Storage: ~60-120GB estimated")
        print("  - Time: 4-8 hours (depends on API limits)")
        print("  - Note: Uses Layer 1 symbols (most liquid ~2,000), NOT Layer 0 (all ~10,000)")

        response = input("\n🤔 Continue with backfill? (y/N) ")
        if response.lower() != "y":
            logger.info("Backfill cancelled by user")
            return 0

    # Run backfill
    logger.info("\n🏃 Starting backfill process...")
    start_time = datetime.now()
    runner.run_all_stages()

    # Print summary
    runner.print_summary()

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"\n⏱️  Total duration: {duration:.1f} seconds ({duration/3600:.1f} hours)")

    return 1 if runner.failed_stages else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
