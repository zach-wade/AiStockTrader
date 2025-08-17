#!/usr/bin/env python3
"""
Syntax Checker - Automated Python Syntax and Import Validation

This script validates Python syntax, imports, and type hints.
Part of the standardized code review process.
"""

# Standard library imports
import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


@dataclass
class SyntaxIssue:
    """Represents a syntax or import issue."""

    file_path: str
    line_number: int
    issue_type: str
    message: str
    severity: str  # "error" or "warning"


class SyntaxChecker:
    """Checks Python files for syntax errors and import issues."""

    def __init__(self, project_root: Path):
        """Initialize the syntax checker.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.issues: list[SyntaxIssue] = []

        # Add project root to sys.path for import checking
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Common async functions that require await
        self.async_functions = {
            "fetch_one",
            "fetch_all",
            "execute",
            "execute_query",
            "save",
            "load",
            "create",
            "update",
            "delete",
            "get",
            "post",
            "put",
            "patch",
            "connect",
            "disconnect",
            "close",
            "acquire",
            "release",
            "run",
            "start",
            "stop",
            "publish",
            "subscribe",
            "process",
            "handle",
        }

    def check_syntax(self, file_path: Path) -> list[SyntaxIssue]:
        """Check a file for syntax errors.

        Args:
            file_path: Path to the Python file

        Returns:
            List of syntax issues found
        """
        issues = []

        try:
            content = file_path.read_text()
            compile(content, str(file_path), "exec")

            # Parse AST for more detailed checks
            tree = ast.parse(content)

            # Check for undefined variables
            issues.extend(self._check_undefined_variables(tree, file_path))

            # Check for missing awaits
            issues.extend(self._check_missing_awaits(tree, file_path))

            # Check for type hints
            issues.extend(self._check_type_hints(tree, file_path))

            # Check imports
            issues.extend(self._check_imports(tree, file_path))

        except SyntaxError as e:
            issues.append(
                SyntaxIssue(
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    issue_type="Syntax Error",
                    message=str(e.msg),
                    severity="error",
                )
            )
        except Exception as e:
            issues.append(
                SyntaxIssue(
                    file_path=str(file_path),
                    line_number=0,
                    issue_type="Parse Error",
                    message=str(e),
                    severity="error",
                )
            )

        return issues

    def _check_undefined_variables(self, tree: ast.AST, file_path: Path) -> list[SyntaxIssue]:
        """Check for potentially undefined variables.

        Args:
            tree: AST tree of the file
            file_path: Path to the file

        Returns:
            List of undefined variable issues
        """
        issues = []

        class NameVisitor(ast.NodeVisitor):
            def __init__(self):
                self.defined: set[str] = set()
                self.used: set[tuple[str, int]] = set()
                self.builtins = set(dir(__builtins__))

                # Add common global names
                self.defined.update(["__name__", "__file__", "__doc__", "logger"])

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.defined.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    if node.id not in self.defined and node.id not in self.builtins:
                        self.used.add((node.id, node.lineno))
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                self.defined.add(node.name)
                # Add function arguments to defined
                for arg in node.args.args:
                    self.defined.add(arg.arg)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node):
                self.defined.add(node.name)
                self.generic_visit(node)

            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    self.defined.add(name.split(".")[0])

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    if alias.name == "*":
                        # Can't track star imports
                        continue
                    name = alias.asname if alias.asname else alias.name
                    self.defined.add(name)

        visitor = NameVisitor()
        visitor.visit(tree)

        # Check for variables used before definition
        for name, lineno in visitor.used:
            if name not in visitor.defined:
                issues.append(
                    SyntaxIssue(
                        file_path=str(file_path),
                        line_number=lineno,
                        issue_type="Undefined Variable",
                        message=f"Name '{name}' may be undefined",
                        severity="warning",
                    )
                )

        return issues

    def _check_missing_awaits(self, tree: ast.AST, file_path: Path) -> list[SyntaxIssue]:
        """Check for missing await keywords in async contexts.

        Args:
            tree: AST tree of the file
            file_path: Path to the file

        Returns:
            List of missing await issues
        """
        issues = []

        class AsyncVisitor(ast.NodeVisitor):
            def __init__(self):
                self.in_async = False
                self.issues = []

            def visit_AsyncFunctionDef(self, node):
                old_async = self.in_async
                self.in_async = True
                self.generic_visit(node)
                self.in_async = old_async

            def visit_Call(self, node):
                if self.in_async:
                    # Check if this looks like an async call
                    func_name = None
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr

                    if func_name and any(
                        async_func in func_name.lower()
                        for async_func in ["async", "await"] + list(self.async_functions)
                    ):
                        # Check if it's awaited
                        parent = getattr(node, "parent", None)
                        if not isinstance(parent, ast.Await):
                            self.issues.append((node.lineno, func_name))

                self.generic_visit(node)

        # Add parent references
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent

        visitor = AsyncVisitor()
        visitor.async_functions = self.async_functions
        visitor.visit(tree)

        for lineno, func_name in visitor.issues:
            issues.append(
                SyntaxIssue(
                    file_path=str(file_path),
                    line_number=lineno,
                    issue_type="Missing Await",
                    message=f"Possible missing 'await' for async call to '{func_name}'",
                    severity="warning",
                )
            )

        return issues

    def _check_type_hints(self, tree: ast.AST, file_path: Path) -> list[SyntaxIssue]:
        """Check for missing type hints in function signatures.

        Args:
            tree: AST tree of the file
            file_path: Path to the file

        Returns:
            List of type hint issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private methods and special methods
                if node.name.startswith("_"):
                    continue

                # Check return type hint
                if node.returns is None and node.name != "__init__":
                    issues.append(
                        SyntaxIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            issue_type="Missing Type Hint",
                            message=f"Function '{node.name}' missing return type hint",
                            severity="warning",
                        )
                    )

                # Check parameter type hints
                for arg in node.args.args:
                    if arg.arg != "self" and arg.arg != "cls" and arg.annotation is None:
                        issues.append(
                            SyntaxIssue(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                issue_type="Missing Type Hint",
                                message=f"Parameter '{arg.arg}' in '{node.name}' missing type hint",
                                severity="warning",
                            )
                        )

        return issues

    def _check_imports(self, tree: ast.AST, file_path: Path) -> list[SyntaxIssue]:
        """Check for import issues.

        Args:
            tree: AST tree of the file
            file_path: Path to the file

        Returns:
            List of import issues
        """
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    try:
                        # Try to find the module
                        spec = importlib.util.find_spec(module_name)
                        if spec is None:
                            issues.append(
                                SyntaxIssue(
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    issue_type="Import Error",
                                    message=f"Module '{module_name}' not found",
                                    severity="error",
                                )
                            )
                    except (ImportError, ModuleNotFoundError, ValueError):
                        issues.append(
                            SyntaxIssue(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                issue_type="Import Error",
                                message=f"Cannot import '{module_name}'",
                                severity="error",
                            )
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    try:
                        # Check if it's a relative import
                        if node.level > 0:
                            # Relative imports are harder to check statically
                            continue

                        spec = importlib.util.find_spec(node.module)
                        if spec is None:
                            issues.append(
                                SyntaxIssue(
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    issue_type="Import Error",
                                    message=f"Module '{node.module}' not found",
                                    severity="error",
                                )
                            )
                    except (ImportError, ModuleNotFoundError, ValueError):
                        # Skip imports that can't be checked
                        pass

        return issues

    def check_directory(
        self, directory: Path, exclude_patterns: list[str] | None = None
    ) -> list[SyntaxIssue]:
        """Check all Python files in a directory.

        Args:
            directory: Directory to scan
            exclude_patterns: Patterns to exclude from scanning

        Returns:
            List of all issues found
        """
        exclude_patterns = exclude_patterns or [
            "__pycache__",
            ".git",
            "venv",
            ".pytest_cache",
            "tests",
        ]
        all_issues = []

        for py_file in directory.rglob("*.py"):
            # Skip excluded paths
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            issues = self.check_syntax(py_file)
            all_issues.extend(issues)

        return all_issues

    def generate_report(self, issues: list[SyntaxIssue]) -> str:
        """Generate a formatted report of syntax issues.

        Args:
            issues: List of issues to report

        Returns:
            Formatted report string
        """
        if not issues:
            return "✅ No syntax issues detected!"

        # Separate errors and warnings
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]

        report = ["=== SYNTAX CHECK REPORT ===\n"]
        report.append(f"Total Issues: {len(issues)}")
        report.append(f"Errors: {len(errors)}, Warnings: {len(warnings)}\n")

        # Group by issue type
        by_type: dict[str, list[SyntaxIssue]] = {}

        if errors:
            report.append("\n🔴 ERRORS:")
            report.append("-" * 50)
            for issue in errors:
                if issue.issue_type not in by_type:
                    by_type[issue.issue_type] = []
                by_type[issue.issue_type].append(issue)

            for issue_type, type_issues in by_type.items():
                report.append(f"\n{issue_type}:")
                for issue in type_issues[:10]:  # Limit to 10 per type
                    report.append(f"  {issue.file_path}:{issue.line_number}")
                    report.append(f"    {issue.message}")

                if len(type_issues) > 10:
                    report.append(f"  ... and {len(type_issues) - 10} more")

        by_type.clear()

        if warnings:
            report.append("\n⚠️ WARNINGS:")
            report.append("-" * 50)
            for issue in warnings:
                if issue.issue_type not in by_type:
                    by_type[issue.issue_type] = []
                by_type[issue.issue_type].append(issue)

            for issue_type, type_issues in by_type.items():
                report.append(f"\n{issue_type}:")
                for issue in type_issues[:5]:  # Limit to 5 per type for warnings
                    report.append(f"  {issue.file_path}:{issue.line_number}")
                    report.append(f"    {issue.message}")

                if len(type_issues) > 5:
                    report.append(f"  ... and {len(type_issues) - 5} more")

        return "\n".join(report)


def main():
    """Main entry point for the syntax checker."""
    # Standard library imports
    import argparse

    parser = argparse.ArgumentParser(description="Check Python syntax and imports")
    parser.add_argument("path", help="File or directory to check")
    parser.add_argument("--errors-only", action="store_true", help="Only show errors, not warnings")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path {path} does not exist")
        sys.exit(1)

    # Find project root (look for src/ or setup.py)
    project_root = path
    while project_root.parent != project_root:
        if (project_root / "src").exists() or (project_root / "setup.py").exists():
            break
        project_root = project_root.parent

    checker = SyntaxChecker(project_root)

    if path.is_file():
        issues = checker.check_syntax(path)
    else:
        issues = checker.check_directory(path)

    # Filter warnings if requested
    if args.errors_only:
        issues = [i for i in issues if i.severity == "error"]

    if args.json:
        # Standard library imports
        import json

        output = {
            "total": len(issues),
            "errors": len([i for i in issues if i.severity == "error"]),
            "warnings": len([i for i in issues if i.severity == "warning"]),
            "issues": [
                {
                    "file": i.file_path,
                    "line": i.line_number,
                    "type": i.issue_type,
                    "message": i.message,
                    "severity": i.severity,
                }
                for i in issues
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(checker.generate_report(issues))

    # Exit with error code if errors found
    if any(i.severity == "error" for i in issues):
        sys.exit(1)


if __name__ == "__main__":
    main()
