#!/usr/bin/env python3
"""
Integration Test Runner for AI Trading System

This script runs comprehensive integration tests covering:
1. Data Pipeline (Backfill, Streaming, Storage)
2. Feature Engineering Pipeline
3. Strategy Signal Generation
4. Risk Management Checks
5. Trade Execution Flow
6. Performance Monitoring
"""

# Standard library imports
import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Third-party imports
import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Manages the execution of integration tests."""

    def __init__(self, test_config_path: str = None):
        """Initialize the test runner."""
        self.test_config_path = test_config_path or "tests/integration/test_config.yaml"
        self.test_results = {}

    def run_all_tests(self, verbose: bool = False) -> int:
        """Run all integration tests."""
        logger.info("🚀 Starting AI Trading System Integration Tests")
        logger.info(f"Using test config: {self.test_config_path}")

        # Set environment variables for testing
        os.environ["AI_TRADER_CONFIG"] = self.test_config_path
        os.environ["AI_TRADER_ENV"] = "test"

        # Define test suites
        test_suites = {
            "data_pipeline": "tests/integration/test_data_pipeline.py",
            "end_to_end": "tests/integration/test_end_to_end_pipeline.py",
            "unified_system": "tests/integration/test_unified_system.py",
        }

        total_failed = 0

        for suite_name, test_file in test_suites.items():
            logger.info(f"\n📋 Running {suite_name} tests...")

            # Run pytest for each test file
            args = [test_file]
            if verbose:
                args.extend(["-v", "-s"])
            else:
                args.append("-q")

            # Add coverage if available
            args.extend(["--cov=.", "--cov-report=term-missing"])

            result = pytest.main(args)

            self.test_results[suite_name] = {
                "status": "passed" if result == 0 else "failed",
                "exit_code": result,
            }

            if result != 0:
                total_failed += 1
                logger.error(f"❌ {suite_name} tests failed with exit code {result}")
            else:
                logger.info(f"✅ {suite_name} tests passed")

        # Generate summary report
        self._generate_summary_report()

        return total_failed

    def run_specific_test(self, test_name: str, verbose: bool = False) -> int:
        """Run a specific test suite."""
        logger.info(f"🎯 Running specific test: {test_name}")

        # Map test names to files
        test_mapping = {
            "data": "tests/integration/test_data_pipeline.py::TestEndToEndPipeline::test_complete_pipeline_flow",
            "pipeline": "tests/integration/test_end_to_end_pipeline.py",
            "system": "tests/integration/test_unified_system.py",
            "quality": "tests/integration/test_end_to_end_pipeline.py::TestDataQualityValidation",
            "attribution": "tests/integration/test_end_to_end_pipeline.py::TestStrategyAttribution",
        }

        if test_name not in test_mapping:
            logger.error(f"Unknown test: {test_name}")
            logger.info(f"Available tests: {', '.join(test_mapping.keys())}")
            return 1

        args = [test_mapping[test_name]]
        if verbose:
            args.extend(["-v", "-s"])

        return pytest.main(args)

    def run_performance_tests(self) -> int:
        """Run performance-focused integration tests."""
        logger.info("⚡ Running performance integration tests")

        # These tests focus on system performance under load
        performance_tests = [
            "tests/integration/test_end_to_end_pipeline.py::test_pipeline_with_multiple_symbols",
            "tests/integration/test_end_to_end_pipeline.py::test_pipeline_performance_metrics",
        ]

        args = performance_tests + ["-v", "--benchmark-only"]
        return pytest.main(args)

    def _generate_summary_report(self):
        """Generate a summary report of test results."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 INTEGRATION TEST SUMMARY")
        logger.info("=" * 60)

        total_suites = len(self.test_results)
        passed_suites = sum(1 for r in self.test_results.values() if r["status"] == "passed")

        logger.info(f"Total Test Suites: {total_suites}")
        logger.info(f"Passed: {passed_suites}")
        logger.info(f"Failed: {total_suites - passed_suites}")

        logger.info("\nDetailed Results:")
        for suite_name, result in self.test_results.items():
            status_emoji = "✅" if result["status"] == "passed" else "❌"
            logger.info(f"  {status_emoji} {suite_name}: {result['status']}")

        logger.info("=" * 60)

        # Save report to file
        report_path = (
            f"test_results/integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        os.makedirs("test_results", exist_ok=True)

        with open(report_path, "w") as f:
            f.write("Integration Test Report\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            for suite_name, result in self.test_results.items():
                f.write(f"{suite_name}: {result['status']}\n")

        logger.info(f"Report saved to: {report_path}")


async def test_live_data_flow():
    """Test with live data flow (requires API keys)."""
    logger.info("🔴 Testing live data flow...")

    # This would test with real API connections
    # Only run if specifically requested and API keys are available
    pass


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run AI Trading System Integration Tests")
    parser.add_argument(
        "--test",
        "-t",
        help="Run specific test suite",
        choices=["data", "pipeline", "system", "quality", "attribution", "all"],
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--performance", "-p", action="store_true", help="Run performance tests")
    parser.add_argument("--config", "-c", help="Path to test configuration file")

    args = parser.parse_args()

    # Initialize test runner
    runner = IntegrationTestRunner(test_config_path=args.config)

    # Run tests based on arguments
    if args.performance:
        exit_code = runner.run_performance_tests()
    elif args.test and args.test != "all":
        exit_code = runner.run_specific_test(args.test, verbose=args.verbose)
    else:
        exit_code = runner.run_all_tests(verbose=args.verbose)

    # Exit with appropriate code
    if exit_code == 0:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.error(f"\n💔 {exit_code} test(s) failed")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
