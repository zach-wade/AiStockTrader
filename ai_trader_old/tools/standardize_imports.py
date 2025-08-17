#!/usr/bin/env python3
"""
Import standardization tool for AI Trader.

This script helps standardize import patterns across the codebase by:
1. Replacing ai_trader imports with main imports
2. Organizing imports according to the style guide
3. Checking for common import issues
"""

# Standard library imports
import argparse
import ast
from pathlib import Path
import re
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class ImportStandardizer:
    """Standardize imports across the codebase."""

    def __init__(self, dry_run: bool = True, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.issues_found = []
        self.files_modified = 0

    def process_directory(self, directory: Path) -> None:
        """Process all Python files in a directory."""
        python_files = list(directory.rglob("*.py"))

        print(f"Found {len(python_files)} Python files to process")

        for file_path in python_files:
            if self.should_skip_file(file_path):
                continue

            self.process_file(file_path)

        # Print summary
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"Files analyzed: {len(python_files)}")
        print(f"Files with issues: {len(self.issues_found)}")
        print(f"Files modified: {self.files_modified}")

        if self.issues_found:
            print("\nIssues found:")
            for issue in self.issues_found[:10]:  # Show first 10 issues
                print(f"  - {issue}")

            if len(self.issues_found) > 10:
                print(f"  ... and {len(self.issues_found) - 10} more")

    def should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_dirs = {
            "__pycache__",
            ".git",
            "venv",
            ".venv",
            "migrations",
            "tests",  # May want to process tests separately
        }

        for part in file_path.parts:
            if part in skip_dirs:
                return True

        return False

    def process_file(self, file_path: Path) -> None:
        """Process a single Python file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Apply standardizations
            content = self.fix_module_prefix(content, file_path)
            content = self.check_import_order(content, file_path)
            content = self.check_star_imports(content, file_path)
            content = self.check_relative_imports(content, file_path)

            # Write back if changed
            if content != original_content:
                self.files_modified += 1

                if not self.dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    if self.verbose:
                        print(f"Modified: {file_path}")
                elif self.verbose:
                    print(f"Would modify: {file_path}")

        except Exception as e:
            self.issues_found.append(f"{file_path}: Error processing - {e}")

    def fix_module_prefix(self, content: str, file_path: Path) -> str:
        """Replace ai_trader imports with main imports."""
        # Pattern to match ai_trader imports
        pattern = r"from ai_trader\."
        replacement = r"from main."

        if re.search(pattern, content):
            self.issues_found.append(f"{file_path}: Found ai_trader import prefix")
            content = re.sub(pattern, replacement, content)

        return content

    def check_import_order(self, content: str, file_path: Path) -> str:
        """Check and fix import ordering."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content

        # Extract imports
        imports = []
        other_statements = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(node)
            else:
                other_statements.append(node)

        if not imports:
            return content

        # Categorize imports
        stdlib_imports = []
        third_party_imports = []
        local_imports = []

        for imp in imports:
            if isinstance(imp, ast.ImportFrom):
                module = imp.module or ""
                if module.startswith("main.") or imp.level > 0:
                    local_imports.append(imp)
                elif self.is_stdlib_module(module):
                    stdlib_imports.append(imp)
                else:
                    third_party_imports.append(imp)
            else:
                # ast.Import
                for alias in imp.names:
                    if self.is_stdlib_module(alias.name):
                        stdlib_imports.append(imp)
                    else:
                        third_party_imports.append(imp)

        # Check if ordering is correct
        all_imports = stdlib_imports + third_party_imports + local_imports
        if imports != all_imports:
            self.issues_found.append(f"{file_path}: Import ordering needs adjustment")

        return content

    def is_stdlib_module(self, module_name: str) -> bool:
        """Check if module is from standard library."""
        stdlib_modules = {
            "asyncio",
            "collections",
            "datetime",
            "json",
            "logging",
            "os",
            "pathlib",
            "sys",
            "typing",
            "functools",
            "itertools",
            "math",
            "re",
            "time",
            "uuid",
            "warnings",
            "abc",
            "enum",
            "dataclasses",
            "contextlib",
            "copy",
            "pickle",
            "threading",
            "multiprocessing",
            "concurrent",
            "queue",
            "weakref",
        }

        if not module_name:
            return False

        top_level = module_name.split(".")[0]
        return top_level in stdlib_modules

    def check_star_imports(self, content: str, file_path: Path) -> str:
        """Check for star imports outside of __init__.py."""
        if file_path.name != "__init__.py":
            if re.search(r"from .+ import \*", content):
                self.issues_found.append(f"{file_path}: Star import found in non-__init__ file")

        return content

    def check_relative_imports(self, content: str, file_path: Path) -> str:
        """Check for relative imports in non-__init__ files."""
        if file_path.name != "__init__.py":
            # Check for relative imports (from . or from ..)
            if re.search(r"from \.+\s+import", content):
                self.issues_found.append(f"{file_path}: Relative import found in non-__init__ file")

        return content


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Standardize imports across the AI Trader codebase"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "main",
        help="Directory to process (default: src/main)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually modify files (disables dry-run)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    # If --apply is set, disable dry-run
    if args.apply:
        args.dry_run = False

    print("Import Standardization Tool")
    print(f"{'='*60}")
    print(f"Directory: {args.directory}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLYING CHANGES'}")
    print(f"{'='*60}\n")

    standardizer = ImportStandardizer(dry_run=args.dry_run, verbose=args.verbose)

    standardizer.process_directory(args.directory)

    if args.dry_run:
        print("\nThis was a dry run. Use --apply to actually modify files.")
    else:
        print("\nFiles have been modified. Remember to:")
        print("1. Review the changes: git diff")
        print("2. Run tests: ./tools/run_tests.sh")
        print("3. Run linting: ./tools/lint.sh")


if __name__ == "__main__":
    main()
