#!/usr/bin/env python3
"""
Run all tests for AI Trader project.

Executes unit tests, integration tests, and generates coverage report.
"""

# Standard library imports
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


class TestRunner:
    """Manages test execution and reporting."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        self.src_dir = project_root / "src" / "main"

    def run_unit_tests(self, verbose: bool = False) -> int:
        """Run unit tests."""
        print("\n" + "=" * 60)
        print("Running Unit Tests")
        print("=" * 60)

        cmd = ["pytest", str(self.tests_dir / "unit"), "-m", "unit", "--tb=short"]

        if verbose:
            cmd.append("-v")

        result = subprocess.run(cmd, cwd=self.project_root, check=False)
        return result.returncode

    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests."""
        print("\n" + "=" * 60)
        print("Running Integration Tests")
        print("=" * 60)

        cmd = ["pytest", str(self.tests_dir / "integration"), "-m", "integration", "--tb=short"]

        if verbose:
            cmd.append("-v")

        result = subprocess.run(cmd, cwd=self.project_root, check=False)
        return result.returncode

    def run_with_coverage(self, test_type: str = "all", verbose: bool = False) -> int:
        """Run tests with coverage analysis."""
        print("\n" + "=" * 60)
        print(f"Running {test_type.title()} Tests with Coverage")
        print("=" * 60)

        if test_type == "all":
            test_path = str(self.tests_dir)
            markers = ""
        elif test_type == "unit":
            test_path = str(self.tests_dir / "unit")
            markers = "-m unit"
        elif test_type == "integration":
            test_path = str(self.tests_dir / "integration")
            markers = "-m integration"
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        cmd = [
            "pytest",
            test_path,
            f"--cov={self.src_dir}",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--tb=short",
        ]

        if markers:
            cmd.extend(markers.split())

        if verbose:
            cmd.append("-v")

        # Set environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "src")
        env["AI_TRADER_ENV"] = "test"

        result = subprocess.run(cmd, cwd=self.project_root, env=env, check=False)

        if result.returncode == 0:
            self._print_coverage_summary()

        return result.returncode

    def run_specific_test(self, test_path: str, verbose: bool = False) -> int:
        """Run a specific test file or test case."""
        print("\n" + "=" * 60)
        print(f"Running: {test_path}")
        print("=" * 60)

        cmd = ["pytest", test_path, "--tb=short"]

        if verbose:
            cmd.append("-v")

        result = subprocess.run(cmd, cwd=self.project_root, check=False)
        return result.returncode

    def run_by_marker(self, marker: str, verbose: bool = False) -> int:
        """Run tests by marker (e.g., slow, requires_db)."""
        print("\n" + "=" * 60)
        print(f"Running Tests with Marker: {marker}")
        print("=" * 60)

        cmd = ["pytest", str(self.tests_dir), "-m", marker, "--tb=short"]

        if verbose:
            cmd.append("-v")

        result = subprocess.run(cmd, cwd=self.project_root, check=False)
        return result.returncode

    def run_failed_tests(self, verbose: bool = False) -> int:
        """Re-run only failed tests from last run."""
        print("\n" + "=" * 60)
        print("Re-running Failed Tests")
        print("=" * 60)

        cmd = ["pytest", "--lf", "--tb=short"]

        if verbose:
            cmd.append("-v")

        result = subprocess.run(cmd, cwd=self.project_root, check=False)
        return result.returncode

    def _print_coverage_summary(self):
        """Print coverage summary from XML report."""
        coverage_file = self.project_root / "coverage.xml"
        if not coverage_file.exists():
            return

        try:
            # Standard library imports
            import xml.etree.ElementTree as ET

            tree = ET.parse(coverage_file)
            root = tree.getroot()

            line_rate = float(root.attrib.get("line-rate", 0))
            coverage_pct = line_rate * 100

            print("\n" + "=" * 60)
            print(f"Coverage Summary: {coverage_pct:.1f}%")
            print("=" * 60)

            if coverage_pct < 80:
                print("⚠️  Warning: Coverage below 80% threshold!")
            else:
                print("✅ Coverage meets minimum threshold!")

        except Exception as e:
            print(f"Could not parse coverage report: {e}")

    def generate_test_report(self, output_file: str = "test_report.txt"):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("Generating Test Report")
        print("=" * 60)

        report = []
        report.append("AI Trader Test Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Count tests
        unit_tests = self._count_tests(self.tests_dir / "unit")
        integration_tests = self._count_tests(self.tests_dir / "integration")

        report.append("Test Summary:")
        report.append(f"- Unit Tests: {unit_tests}")
        report.append(f"- Integration Tests: {integration_tests}")
        report.append(f"- Total Tests: {unit_tests + integration_tests}")
        report.append("")

        # List test files
        report.append("Test Files:")
        report.append("-" * 30)

        for test_file in sorted(self.tests_dir.rglob("test_*.py")):
            rel_path = test_file.relative_to(self.tests_dir)
            report.append(f"  {rel_path}")

        # Save report
        output_path = self.project_root / output_file
        output_path.write_text("\n".join(report))
        print(f"Report saved to: {output_path}")

    def _count_tests(self, directory: Path) -> int:
        """Count number of test functions in directory."""
        count = 0
        for test_file in directory.rglob("test_*.py"):
            content = test_file.read_text()
            count += content.count("def test_")
            count += content.count("async def test_")
        return count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run AI Trader tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all tests with coverage
  %(prog)s -t unit           # Run only unit tests
  %(prog)s -t integration    # Run only integration tests
  %(prog)s -m slow           # Run tests marked as slow
  %(prog)s -f                # Re-run failed tests
  %(prog)s -s tests/unit/test_order_manager.py  # Run specific test
        """,
    )

    parser.add_argument(
        "-t",
        "--type",
        choices=["all", "unit", "integration"],
        default="all",
        help="Type of tests to run",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    parser.add_argument(
        "-c",
        "--coverage",
        action="store_true",
        default=True,
        help="Run with coverage (default: True)",
    )

    parser.add_argument("-m", "--marker", help="Run tests with specific marker")

    parser.add_argument("-f", "--failed", action="store_true", help="Re-run only failed tests")

    parser.add_argument("-s", "--specific", help="Run specific test file or test")

    parser.add_argument("-r", "--report", action="store_true", help="Generate test report")

    parser.add_argument("--no-coverage", action="store_true", help="Run without coverage analysis")

    args = parser.parse_args()

    # Find project root
    project_root = Path(__file__).parent.parent
    runner = TestRunner(project_root)

    exit_code = 0

    try:
        if args.report:
            runner.generate_test_report()
            return

        if args.failed:
            exit_code = runner.run_failed_tests(args.verbose)
        elif args.specific:
            exit_code = runner.run_specific_test(args.specific, args.verbose)
        elif args.marker:
            exit_code = runner.run_by_marker(args.marker, args.verbose)
        elif args.no_coverage:
            if args.type == "unit":
                exit_code = runner.run_unit_tests(args.verbose)
            elif args.type == "integration":
                exit_code = runner.run_integration_tests(args.verbose)
            else:
                exit_code = runner.run_unit_tests(args.verbose)
                if exit_code == 0:
                    exit_code = runner.run_integration_tests(args.verbose)
        else:
            exit_code = runner.run_with_coverage(args.type, args.verbose)

    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\nError running tests: {e}")
        exit_code = 1

    # Print summary
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
