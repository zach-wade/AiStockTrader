#!/usr/bin/env python3
"""
Events-specific test runner for AI Trader.

Focused test runner for the events module with specialized reporting
and debugging capabilities.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class EventsTestRunner:
    """Specialized test runner for events module."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        self.events_unit_dir = self.tests_dir / "unit" / "events"
        self.events_integration_dir = self.tests_dir / "integration" / "events"
        self.src_events_dir = project_root / "src" / "main" / "events"
    
    def run_events_tests(
        self,
        test_type: str = "all",
        component: Optional[str] = None,
        verbose: bool = False,
        coverage: bool = True,
        parallel: bool = False,
        debug: bool = False
    ) -> int:
        """Run events module tests with specialized options."""
        
        print("\n" + "🎯" + "=" * 58)
        print("🎯 AI TRADER EVENTS TEST RUNNER")
        print("🎯" + "=" * 58)
        
        # Environment setup
        self._setup_test_environment()
        
        # Determine test paths
        test_paths = self._get_test_paths(test_type, component)
        
        if not test_paths:
            print(f"❌ No test paths found for type '{test_type}' and component '{component}'")
            return 1
        
        # Build pytest command
        cmd = self._build_pytest_command(
            test_paths, verbose, coverage, parallel, debug
        )
        
        # Display test plan
        self._display_test_plan(test_type, component, test_paths)
        
        # Run tests
        print(f"\n🚀 Running tests...")
        print(f"📝 Command: {' '.join(cmd)}")
        print("-" * 60)
        
        start_time = datetime.now()
        result = subprocess.run(cmd, cwd=self.project_root)
        end_time = datetime.now()
        
        # Display results
        self._display_results(result.returncode, start_time, end_time, coverage)
        
        return result.returncode
    
    def run_component_tests(self, component: str, **kwargs) -> int:
        """Run tests for specific events component."""
        component_map = {
            'bus': ['event_bus', 'event_bus_initializer'],
            'bridge': ['scanner_feature_bridge', 'scanner_feature_bridge_initializer'],
            'pipeline': ['feature_pipeline_handler'],
            'engine': ['event_driven_engine'], 
            'types': ['event_types'],
            'helpers': ['*helpers*'],
            'bus_helpers': ['event_bus_helpers'],
            'bridge_helpers': ['scanner_bridge_helpers'],
            'pipeline_helpers': ['feature_pipeline_helpers'],
        }
        
        if component not in component_map:
            print(f"❌ Unknown component '{component}'")
            print(f"Available components: {', '.join(component_map.keys())}")
            return 1
        
        return self.run_events_tests(component=component, **kwargs)
    
    def run_performance_tests(self, **kwargs) -> int:
        """Run performance-focused tests."""
        cmd = [
            "pytest",
            str(self.events_integration_dir / "test_performance_integration.py"),
            "-m", "performance",
            "-v",
            "--benchmark-only" if self._has_package("pytest-benchmark") else "",
        ]
        
        cmd = [c for c in cmd if c]  # Remove empty strings
        
        print("\n⚡ Running Events Performance Tests")
        print("=" * 50)
        
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode
    
    def run_smoke_tests(self, **kwargs) -> int:
        """Run smoke tests for basic functionality."""
        smoke_tests = [
            self.events_unit_dir / "test_event_types.py",
            self.events_unit_dir / "test_event_bus.py", 
            self.events_integration_dir / "test_event_coordination_integration.py"
        ]
        
        cmd = [
            "pytest",
            *[str(p) for p in smoke_tests if p.exists()],
            "-m", "not slow",
            "-x",  # Stop on first failure
            "--tb=short"
        ]
        
        print("\n💨 Running Events Smoke Tests")
        print("=" * 50)
        
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode
    
    def analyze_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage for events module."""
        print("\n📊 Analyzing Events Module Coverage")
        print("=" * 50)
        
        # Run coverage analysis
        cmd = [
            "pytest",
            str(self.events_unit_dir),
            str(self.events_integration_dir),
            f"--cov={self.src_events_dir}",
            "--cov-report=term-missing",
            "--cov-report=json:coverage_events.json",
            "-q"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Coverage analysis completed")
            return self._parse_coverage_report()
        else:
            print(f"❌ Coverage analysis failed: {result.stderr}")
            return {}
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report for events module."""
        print("\n📋 Generating Events Test Report")
        print("=" * 50)
        
        report = []
        report.append("# Events Module Test Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Test file counts
        unit_tests = list(self.events_unit_dir.glob("test_*.py"))
        integration_tests = list(self.events_integration_dir.glob("test_*.py"))
        
        report.append("## Test Coverage Summary")
        report.append(f"- Unit test files: {len(unit_tests)}")
        report.append(f"- Integration test files: {len(integration_tests)}")
        report.append(f"- Total test files: {len(unit_tests) + len(integration_tests)}")
        report.append("")
        
        # Component breakdown
        components = self._analyze_test_components()
        report.append("## Component Test Coverage")
        for component, info in components.items():
            report.append(f"- {component}: {info['unit']} unit + {info['integration']} integration")
        
        report.append("")
        
        # Test files list
        report.append("## Unit Test Files")
        for test_file in sorted(unit_tests):
            report.append(f"  - {test_file.name}")
        
        report.append("")
        report.append("## Integration Test Files") 
        for test_file in sorted(integration_tests):
            report.append(f"  - {test_file.name}")
        
        # Save report
        report_file = self.tests_dir / "reports" / f"events_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_file.parent.mkdir(exist_ok=True)
        report_file.write_text("\n".join(report))
        
        print(f"✅ Report saved to: {report_file}")
        return str(report_file)
    
    def _setup_test_environment(self):
        """Setup environment for testing."""
        env_vars = {
            "AI_TRADER_ENV": "test",
            "TESTING": "1", 
            "PYTHONPATH": str(self.project_root / "src"),
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
    
    def _get_test_paths(self, test_type: str, component: Optional[str]) -> List[Path]:
        """Get test paths based on type and component."""
        paths = []
        
        if test_type in ["all", "unit"]:
            if component:
                unit_files = list(self.events_unit_dir.glob(f"*{component}*.py"))
                paths.extend(unit_files)
            else:
                paths.append(self.events_unit_dir)
        
        if test_type in ["all", "integration"]:
            if component:
                integration_files = list(self.events_integration_dir.glob(f"*{component}*.py"))
                paths.extend(integration_files)
            else:
                paths.append(self.events_integration_dir)
        
        return paths
    
    def _build_pytest_command(
        self,
        test_paths: List[Path],
        verbose: bool,
        coverage: bool,
        parallel: bool,
        debug: bool
    ) -> List[str]:
        """Build pytest command with options."""
        cmd = ["pytest"]
        
        # Add test paths
        cmd.extend([str(p) for p in test_paths])
        
        # Add markers for events
        cmd.extend(["-m", "events"])
        
        # Coverage options
        if coverage:
            cmd.extend([
                f"--cov={self.src_events_dir}",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/events",
            ])
        
        # Parallel execution
        if parallel and self._has_package("pytest-xdist"):
            cmd.extend(["-n", "auto"])
        
        # Verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Debug mode
        if debug:
            cmd.extend(["-s", "--tb=long", "--capture=no"])
        else:
            cmd.extend(["--tb=short"])
        
        return cmd
    
    def _display_test_plan(self, test_type: str, component: Optional[str], test_paths: List[Path]):
        """Display what tests will be run."""
        print(f"\n📋 Test Plan:")
        print(f"   Type: {test_type}")
        print(f"   Component: {component or 'all'}")
        print(f"   Paths: {len(test_paths)}")
        
        for path in test_paths:
            if path.is_file():
                print(f"     📄 {path.relative_to(self.tests_dir)}")
            else:
                file_count = len(list(path.glob("test_*.py")))
                print(f"     📁 {path.relative_to(self.tests_dir)} ({file_count} files)")
    
    def _display_results(self, return_code: int, start_time: datetime, end_time: datetime, coverage: bool):
        """Display test results summary."""
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        if return_code == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Tests failed (exit code: {return_code})")
        
        print(f"⏱️  Duration: {duration.total_seconds():.2f} seconds")
        
        if coverage:
            print(f"📊 Coverage report: htmlcov/events/index.html")
        
        print("=" * 60)
    
    def _analyze_test_components(self) -> Dict[str, Dict[str, int]]:
        """Analyze test coverage by component."""
        components = {}
        
        # Map file patterns to components
        component_patterns = {
            'event_bus': ['event_bus'],
            'scanner_bridge': ['scanner_feature_bridge', 'scanner_bridge'],
            'feature_pipeline': ['feature_pipeline', 'pipeline'],
            'event_types': ['event_types'],
            'helpers': ['helpers'],
            'engine': ['engine'],
        }
        
        for component, patterns in component_patterns.items():
            unit_count = 0
            integration_count = 0
            
            for pattern in patterns:
                unit_count += len(list(self.events_unit_dir.glob(f"*{pattern}*.py")))
                integration_count += len(list(self.events_integration_dir.glob(f"*{pattern}*.py")))
            
            components[component] = {
                'unit': unit_count,
                'integration': integration_count
            }
        
        return components
    
    def _parse_coverage_report(self) -> Dict[str, Any]:
        """Parse coverage report from JSON."""
        coverage_file = self.project_root / "coverage_events.json"
        
        if not coverage_file.exists():
            return {}
        
        try:
            import json
            with open(coverage_file) as f:
                data = json.load(f)
            
            return {
                'total_coverage': data.get('totals', {}).get('percent_covered', 0),
                'files': len(data.get('files', {})),
                'summary': data.get('totals', {})
            }
        except Exception as e:
            print(f"Error parsing coverage report: {e}")
            return {}
    
    def _has_package(self, package: str) -> bool:
        """Check if package is available."""
        try:
            __import__(package.replace("-", "_"))
            return True
        except ImportError:
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run events module tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run all events tests
  %(prog)s -t unit                   # Run only unit tests
  %(prog)s -t integration            # Run only integration tests  
  %(prog)s -c bus                    # Run event bus tests
  %(prog)s -c bridge -v              # Run scanner bridge tests (verbose)
  %(prog)s --performance             # Run performance tests
  %(prog)s --smoke                   # Run smoke tests
  %(prog)s --coverage-report         # Generate coverage analysis
  %(prog)s --test-report             # Generate test report
        """
    )
    
    parser.add_argument(
        "-t", "--type",
        choices=["all", "unit", "integration"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "-c", "--component",
        choices=["bus", "bridge", "pipeline", "engine", "types", "helpers", "bus_helpers", "bridge_helpers", "pipeline_helpers"],
        help="Specific component to test"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (no capture, full tracebacks)"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance tests only"
    )
    
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke tests only"
    )
    
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="Generate coverage analysis report"
    )
    
    parser.add_argument(
        "--test-report",
        action="store_true",
        help="Generate comprehensive test report"
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    runner = EventsTestRunner(project_root)
    
    try:
        if args.test_report:
            runner.generate_test_report()
            return
        
        if args.coverage_report:
            runner.analyze_coverage()
            return
        
        if args.performance:
            exit_code = runner.run_performance_tests()
        elif args.smoke:
            exit_code = runner.run_smoke_tests()
        elif args.component:
            exit_code = runner.run_component_tests(
                args.component,
                test_type=args.type,
                verbose=args.verbose,
                coverage=not args.no_coverage,
                parallel=args.parallel,
                debug=args.debug
            )
        else:
            exit_code = runner.run_events_tests(
                test_type=args.type,
                component=args.component,
                verbose=args.verbose,
                coverage=not args.no_coverage,
                parallel=args.parallel,
                debug=args.debug
            )
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()