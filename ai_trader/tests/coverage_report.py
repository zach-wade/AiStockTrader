#!/usr/bin/env python3
"""
Test Coverage Analysis and Reporting

Generates detailed coverage reports for the AI Trader project,
identifies gaps, and creates coverage badges.
"""

import os
import sys
from pathlib import Path
from pathlib import Path
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from datetime import datetime


class CoverageAnalyzer:
    """Analyzes test coverage and generates reports."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src" / "main"
        self.tests_path = project_root / "tests"
        self.coverage_file = project_root / "coverage.xml"
        self.htmlcov_path = project_root / "htmlcov"
        
    def run_coverage(self, verbose: bool = False) -> int:
        """Run tests with coverage."""
        cmd = [
            "pytest",
            f"--cov={self.src_path}",
            "--cov-report=xml",
            "--cov-report=html",
            "--cov-report=term-missing",
            "-v" if verbose else "-q",
            str(self.tests_path)
        ]
        
        print(f"Running coverage analysis...")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode
    
    def parse_coverage_xml(self) -> Dict[str, float]:
        """Parse coverage.xml to get detailed metrics."""
        if not self.coverage_file.exists():
            print(f"Coverage file not found: {self.coverage_file}")
            print("Run tests with coverage first: python tests/coverage_report.py --run")
            return {'overall': 0, 'packages': {}}
        
        tree = ET.parse(self.coverage_file)
        root = tree.getroot()
        
        # Get overall coverage
        overall_coverage = float(root.attrib.get('line-rate', 0)) * 100
        
        # Get package-level coverage
        package_coverage = {}
        for package in root.findall('.//package'):
            name = package.attrib['name'].replace('.', '/')
            line_rate = float(package.attrib.get('line-rate', 0)) * 100
            package_coverage[name] = line_rate
        
        return {
            'overall': overall_coverage,
            'packages': package_coverage
        }
    
    def identify_untested_components(self) -> Dict[str, List[str]]:
        """Identify components without tests."""
        untested = {
            'modules': [],
            'classes': [],
            'functions': []
        }
        
        if not self.coverage_file.exists():
            return untested
        
        tree = ET.parse(self.coverage_file)
        root = tree.getroot()
        
        # Find untested files
        for class_elem in root.findall('.//class'):
            filename = class_elem.attrib['filename']
            line_rate = float(class_elem.attrib.get('line-rate', 0))
            
            if line_rate == 0:
                rel_path = Path(filename).relative_to(self.src_path)
                untested['modules'].append(str(rel_path))
        
        return untested
    
    def generate_coverage_badge(self, coverage: float) -> str:
        """Generate coverage badge SVG."""
        if coverage >= 90:
            color = "brightgreen"
        elif coverage >= 80:
            color = "green"
        elif coverage >= 70:
            color = "yellow"
        elif coverage >= 60:
            color = "orange"
        else:
            color = "red"
        
        badge_url = f"https://img.shields.io/badge/coverage-{coverage:.1f}%25-{color}"
        return badge_url
    
    def generate_report(self) -> str:
        """Generate comprehensive coverage report."""
        coverage_data = self.parse_coverage_xml()
        untested = self.identify_untested_components()
        
        report = f"""
# AI Trader Test Coverage Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Coverage: {coverage_data.get('overall', 0):.1f}%

![Coverage Badge]({self.generate_coverage_badge(coverage_data.get('overall', 0))})

## Package Coverage

| Package | Coverage |
|---------|----------|
"""
        
        # Add package coverage
        for package, coverage in sorted(coverage_data.get('packages', {}).items()):
            report += f"| {package} | {coverage:.1f}% |\n"
        
        # Add untested components
        report += "\n## Untested Components\n\n"
        
        if untested['modules']:
            report += "### Completely Untested Modules\n"
            for module in sorted(untested['modules']):
                report += f"- {module}\n"
        
        # Add priority recommendations
        report += "\n## Test Priority Recommendations\n\n"
        report += self._generate_priority_recommendations(coverage_data, untested)
        
        return report
    
    def _generate_priority_recommendations(self, coverage_data: Dict, untested: Dict) -> str:
        """Generate testing priority recommendations."""
        recommendations = []
        
        # Critical components that need testing
        critical_paths = [
            'trading_engine/risk',
            'events',
            'monitoring/alert_manager',
            'orchestration/system_coordinator',
            'models/training'
        ]
        
        for path in critical_paths:
            pkg_coverage = coverage_data.get('packages', {}).get(path, 0)
            if pkg_coverage < 80:
                recommendations.append(
                    f"- **{path}** (Current: {pkg_coverage:.1f}%) - Critical component needs coverage"
                )
        
        return "\n".join(recommendations[:10])  # Top 10 recommendations
    
    def save_report(self, report: str, output_path: Optional[Path] = None):
        """Save report to file."""
        if output_path is None:
            output_path = self.project_root / "coverage_report.md"
        
        output_path.write_text(report)
        print(f"Coverage report saved to: {output_path}")
    
    def generate_github_actions_summary(self) -> str:
        """Generate summary for GitHub Actions."""
        coverage_data = self.parse_coverage_xml()
        overall = coverage_data.get('overall', 0)
        
        summary = f"""### Test Coverage Summary

**Overall Coverage:** {overall:.1f}%

"""
        
        if overall < 80:
            summary += "⚠️ **Warning:** Coverage is below 80% threshold\n\n"
        else:
            summary += "✅ **Success:** Coverage meets minimum threshold\n\n"
        
        # Add top 5 least covered packages
        packages = coverage_data.get('packages', {})
        if packages:
            sorted_packages = sorted(packages.items(), key=lambda x: x[1])[:5]
            summary += "**Least Covered Packages:**\n"
            for pkg, cov in sorted_packages:
                summary += f"- {pkg}: {cov:.1f}%\n"
        
        return summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze test coverage")
    parser.add_argument(
        "--run", 
        action="store_true",
        help="Run tests with coverage before analysis"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="coverage_report.md",
        help="Output report filename"
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Generate GitHub Actions summary"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    analyzer = CoverageAnalyzer(project_root)
    
    # Run coverage if requested
    if args.run:
        exit_code = analyzer.run_coverage(verbose=args.verbose)
        if exit_code != 0:
            print("Tests failed!")
            sys.exit(exit_code)
    
    # Generate report
    if args.github_summary:
        summary = analyzer.generate_github_actions_summary()
        print(summary)
    else:
        report = analyzer.generate_report()
        analyzer.save_report(report, Path(args.report))
        print(report)


if __name__ == "__main__":
    main()