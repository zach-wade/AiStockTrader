#!/usr/bin/env python3
"""
Pattern Checker - Automated Code Anti-Pattern Detection

This script scans Python files for common anti-patterns and code quality issues.
Part of the standardized code review process.
"""

# Standard library imports
import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
import sys


class IssueSeverity(Enum):
    """Issue severity levels."""

    CRITICAL = "🔴 Critical"
    MAJOR = "🟡 Major"
    MINOR = "🔵 Minor"


@dataclass
class Issue:
    """Represents a code issue found during pattern checking."""

    file_path: str
    line_number: int
    severity: IssueSeverity
    category: str
    description: str
    code_snippet: str | None = None
    fix_suggestion: str | None = None


class PatternChecker:
    """Checks Python code for anti-patterns and quality issues."""

    def __init__(self, root_path: Path):
        """Initialize the pattern checker.

        Args:
            root_path: Root directory to scan
        """
        self.root_path = root_path
        self.issues: list[Issue] = []

        # Patterns to detect
        self.sql_injection_patterns = [
            r'f".*SELECT.*FROM.*{.*}',
            r"f'.*SELECT.*FROM.*{.*}",
            r'f".*INSERT.*INTO.*{.*}',
            r"f'.*INSERT.*INTO.*{.*}",
            r'f".*UPDATE.*SET.*{.*}',
            r"f'.*UPDATE.*SET.*{.*}",
            r'f".*DELETE.*FROM.*{.*}',
            r"f'.*DELETE.*FROM.*{.*}",
        ]

        self.direct_instantiation_patterns = [
            r"CompanyRepository\s*\(",
            r"MarketDataRepository\s*\(",
            r"NewsRepository\s*\(",
            r"FundamentalsRepository\s*\(",
            r"AsyncDatabaseAdapter\s*\(",
            r"SyncDatabaseAdapter\s*\(",
        ]

        self.hardcoded_secrets_patterns = [
            r'api_key\s*=\s*["\'][\w]{20,}',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][\w]{20,}',
            r'AWS_[A-Z_]+\s*=\s*["\']',
            r'POLYGON_API_KEY\s*=\s*["\']',
        ]

    def check_file(self, file_path: Path) -> list[Issue]:
        """Check a single Python file for patterns.

        Args:
            file_path: Path to the Python file

        Returns:
            List of issues found
        """
        issues = []

        try:
            content = file_path.read_text()
            lines = content.split("\n")

            # Parse AST for structural checks
            tree = ast.parse(content)

            # Check for bare except clauses
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    if node.type is None:  # Bare except
                        issues.append(
                            Issue(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                severity=IssueSeverity.MAJOR,
                                category="Exception Handling",
                                description="Bare except clause found",
                                code_snippet=(
                                    lines[node.lineno - 1] if node.lineno <= len(lines) else None
                                ),
                                fix_suggestion="Specify exception type: except Exception as e:",
                            )
                        )

                # Check for star imports
                if isinstance(node, ast.ImportFrom):
                    if any(alias.name == "*" for alias in node.names):
                        issues.append(
                            Issue(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                severity=IssueSeverity.MINOR,
                                category="Imports",
                                description=f"Star import from {node.module}",
                                code_snippet=(
                                    lines[node.lineno - 1] if node.lineno <= len(lines) else None
                                ),
                                fix_suggestion="Import specific items instead of using *",
                            )
                        )

                # Check for functions returning None on error
                if isinstance(node, ast.FunctionDef):
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.Return):
                            if isinstance(stmt.value, ast.Constant) and stmt.value.value is None:
                                # Check if this is in an except block
                                for parent in ast.walk(tree):
                                    if hasattr(parent, "orelse") and stmt in ast.walk(parent):
                                        if isinstance(parent, ast.Try):
                                            issues.append(
                                                Issue(
                                                    file_path=str(file_path),
                                                    line_number=stmt.lineno,
                                                    severity=IssueSeverity.MAJOR,
                                                    category="Error Handling",
                                                    description="Function returns None on error",
                                                    code_snippet=(
                                                        lines[stmt.lineno - 1]
                                                        if stmt.lineno <= len(lines)
                                                        else None
                                                    ),
                                                    fix_suggestion="Raise exception instead of returning None",
                                                )
                                            )

            # Check for regex patterns
            for i, line in enumerate(lines, 1):
                # SQL injection patterns
                for pattern in self.sql_injection_patterns:
                    if re.search(pattern, line):
                        issues.append(
                            Issue(
                                file_path=str(file_path),
                                line_number=i,
                                severity=IssueSeverity.CRITICAL,
                                category="Security",
                                description="Potential SQL injection vulnerability",
                                code_snippet=line.strip(),
                                fix_suggestion="Use parameterized queries or validate_table_name()",
                            )
                        )

                # Direct instantiation patterns
                for pattern in self.direct_instantiation_patterns:
                    if re.search(pattern, line):
                        issues.append(
                            Issue(
                                file_path=str(file_path),
                                line_number=i,
                                severity=IssueSeverity.MAJOR,
                                category="Design Pattern",
                                description="Direct instantiation of repository/adapter",
                                code_snippet=line.strip(),
                                fix_suggestion="Use factory pattern for instantiation",
                            )
                        )

                # Hardcoded secrets
                for pattern in self.hardcoded_secrets_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(
                            Issue(
                                file_path=str(file_path),
                                line_number=i,
                                severity=IssueSeverity.CRITICAL,
                                category="Security",
                                description="Hardcoded secret or API key",
                                code_snippet=line.strip()[:50] + "...",  # Truncate for security
                                fix_suggestion="Use environment variables or secure config",
                            )
                        )

                # Check for long lines
                if len(line) > 120:
                    issues.append(
                        Issue(
                            file_path=str(file_path),
                            line_number=i,
                            severity=IssueSeverity.MINOR,
                            category="Code Style",
                            description=f"Line too long ({len(line)} characters)",
                            fix_suggestion="Break line to stay under 120 characters",
                        )
                    )

            # Check file size
            if len(lines) > 500:
                issues.append(
                    Issue(
                        file_path=str(file_path),
                        line_number=1,
                        severity=IssueSeverity.MINOR,
                        category="File Organization",
                        description=f"File too large ({len(lines)} lines)",
                        fix_suggestion="Consider splitting into smaller modules",
                    )
                )

        except Exception as e:
            print(f"Error checking {file_path}: {e}")

        return issues

    def check_directory(
        self, directory: Path, exclude_patterns: list[str] | None = None
    ) -> list[Issue]:
        """Check all Python files in a directory.

        Args:
            directory: Directory to scan
            exclude_patterns: Patterns to exclude from scanning

        Returns:
            List of all issues found
        """
        exclude_patterns = exclude_patterns or ["__pycache__", ".git", "venv", ".pytest_cache"]
        all_issues = []

        for py_file in directory.rglob("*.py"):
            # Skip excluded paths
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            issues = self.check_file(py_file)
            all_issues.extend(issues)

        return all_issues

    def generate_report(self, issues: list[Issue]) -> str:
        """Generate a formatted report of issues.

        Args:
            issues: List of issues to report

        Returns:
            Formatted report string
        """
        if not issues:
            return "✅ No anti-patterns detected!"

        # Group by severity
        critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        major = [i for i in issues if i.severity == IssueSeverity.MAJOR]
        minor = [i for i in issues if i.severity == IssueSeverity.MINOR]

        report = ["=== PATTERN CHECK REPORT ===\n"]
        report.append(f"Total Issues: {len(issues)}")
        report.append(f"Critical: {len(critical)}, Major: {len(major)}, Minor: {len(minor)}\n")

        # Report by severity
        for severity_list, severity_name in [
            (critical, "CRITICAL"),
            (major, "MAJOR"),
            (minor, "MINOR"),
        ]:
            if severity_list:
                report.append(f"\n{getattr(IssueSeverity, severity_name).value} Issues:")
                report.append("-" * 50)

                # Group by category
                by_category: dict[str, list[Issue]] = {}
                for issue in severity_list:
                    if issue.category not in by_category:
                        by_category[issue.category] = []
                    by_category[issue.category].append(issue)

                for category, category_issues in by_category.items():
                    report.append(f"\n{category}:")
                    for issue in category_issues[:5]:  # Limit to 5 per category
                        report.append(f"  {issue.file_path}:{issue.line_number}")
                        report.append(f"    {issue.description}")
                        if issue.fix_suggestion:
                            report.append(f"    Fix: {issue.fix_suggestion}")

                    if len(category_issues) > 5:
                        report.append(f"  ... and {len(category_issues) - 5} more")

        return "\n".join(report)


def main():
    """Main entry point for the pattern checker."""
    # Standard library imports
    import argparse

    parser = argparse.ArgumentParser(description="Check Python code for anti-patterns")
    parser.add_argument("path", help="File or directory to check")
    parser.add_argument(
        "--severity",
        choices=["critical", "major", "minor", "all"],
        default="all",
        help="Minimum severity to report",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path {path} does not exist")
        sys.exit(1)

    checker = PatternChecker(path.parent if path.is_file() else path)

    if path.is_file():
        issues = checker.check_file(path)
    else:
        issues = checker.check_directory(path)

    # Filter by severity if requested
    if args.severity != "all":
        severity_map = {
            "critical": IssueSeverity.CRITICAL,
            "major": IssueSeverity.MAJOR,
            "minor": IssueSeverity.MINOR,
        }
        min_severity = severity_map[args.severity]

        if min_severity == IssueSeverity.CRITICAL:
            issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        elif min_severity == IssueSeverity.MAJOR:
            issues = [
                i for i in issues if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.MAJOR]
            ]

    if args.json:
        # Standard library imports
        import json

        output = {
            "total": len(issues),
            "issues": [
                {
                    "file": i.file_path,
                    "line": i.line_number,
                    "severity": i.severity.name,
                    "category": i.category,
                    "description": i.description,
                    "fix": i.fix_suggestion,
                }
                for i in issues
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(checker.generate_report(issues))

    # Exit with error code if critical issues found
    if any(i.severity == IssueSeverity.CRITICAL for i in issues):
        sys.exit(1)


if __name__ == "__main__":
    main()
