#!/usr/bin/env python3
"""
Automated Code Analysis Tool for AI Trading System Audit

This script analyzes the codebase to identify:
- Large files that need refactoring
- Circular import patterns
- Duplicate code blocks
- Dead code candidates
- Missing docstrings
- Empty modules

Created: 2025-08-08
Part of Phase 4 Project Audit
"""

# Standard library imports
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path


class CodeAnalyzer:
    """Analyzes Python codebase for quality issues."""

    def __init__(self, base_path: str = "src/main"):
        self.base_path = Path(base_path)
        self.results = {
            "large_files": [],
            "circular_imports": [],
            "duplicate_code": [],
            "missing_docstrings": [],
            "empty_modules": [],
            "import_graph": defaultdict(set),
            "statistics": {},
        }

    def analyze(self) -> dict:
        """Run all analyses."""
        print("🔍 Starting code analysis...")

        # Collect all Python files
        py_files = list(self.base_path.rglob("*.py"))
        print(f"Found {len(py_files)} Python files")

        # Run analyses
        self._analyze_file_sizes(py_files)
        self._analyze_imports(py_files)
        self._analyze_docstrings(py_files)
        self._find_empty_modules()
        self._detect_circular_imports()
        self._find_duplicate_code(py_files)

        # Calculate statistics
        self._calculate_statistics(py_files)

        return self.results

    def _analyze_file_sizes(self, files: list[Path]):
        """Find files over 500 lines."""
        print("📏 Analyzing file sizes...")

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    line_count = len(lines)

                    if line_count > 500:
                        self.results["large_files"].append(
                            {
                                "file": str(file_path.relative_to(self.base_path.parent)),
                                "lines": line_count,
                                "size_kb": file_path.stat().st_size / 1024,
                            }
                        )
            except Exception:
                pass

        # Sort by size
        self.results["large_files"].sort(key=lambda x: x["lines"], reverse=True)

    def _analyze_imports(self, files: list[Path]):
        """Build import graph for circular detection."""
        print("🔄 Analyzing imports...")

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                module_name = self._path_to_module(file_path)
                imports = set()

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith("main."):
                                imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith("main."):
                            imports.add(node.module)

                self.results["import_graph"][module_name] = imports

            except Exception:
                pass

    def _detect_circular_imports(self):
        """Detect circular import patterns."""
        print("🔁 Detecting circular imports...")

        def find_cycles(graph, start, current, path, visited, cycles):
            """DFS to find cycles."""
            visited.add(current)
            path.append(current)

            for neighbor in graph.get(current, []):
                if neighbor == start and len(path) > 1:
                    # Found a cycle
                    cycle = path + [neighbor]
                    # Normalize cycle to start with smallest element
                    min_idx = cycle.index(min(cycle))
                    normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
                    cycles.add(normalized)
                elif neighbor not in visited:
                    find_cycles(graph, start, neighbor, path[:], visited.copy(), cycles)

        cycles = set()
        for module in self.results["import_graph"]:
            find_cycles(self.results["import_graph"], module, module, [], set(), cycles)

        # Convert to list and limit to unique cycles
        unique_cycles = []
        seen = set()
        for cycle in cycles:
            if cycle not in seen and cycle[::-1] not in seen:
                unique_cycles.append(list(cycle))
                seen.add(cycle)

        self.results["circular_imports"] = unique_cycles[:10]  # Limit to 10

    def _analyze_docstrings(self, files: list[Path]):
        """Find classes and functions missing docstrings."""
        print("📝 Analyzing docstrings...")

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read())

                module_name = str(file_path.relative_to(self.base_path.parent))

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not ast.get_docstring(node):
                            # Skip private methods and test files
                            if not node.name.startswith("_") and "test" not in module_name:
                                self.results["missing_docstrings"].append(
                                    {
                                        "file": module_name,
                                        "type": (
                                            "function"
                                            if isinstance(node, ast.FunctionDef)
                                            else "class"
                                        ),
                                        "name": node.name,
                                        "line": node.lineno,
                                    }
                                )
            except Exception:
                pass

        # Limit to 20 examples
        self.results["missing_docstrings"] = self.results["missing_docstrings"][:20]

    def _find_empty_modules(self):
        """Find empty or near-empty modules."""
        print("📭 Finding empty modules...")

        for dir_path in self.base_path.rglob("*"):
            if dir_path.is_dir():
                py_files = list(dir_path.glob("*.py"))

                # Check if directory has only __init__.py
                if len(py_files) == 1 and py_files[0].name == "__init__.py":
                    init_size = py_files[0].stat().st_size
                    if init_size < 100:  # Less than 100 bytes
                        self.results["empty_modules"].append(
                            str(dir_path.relative_to(self.base_path.parent))
                        )

    def _find_duplicate_code(self, files: list[Path]):
        """Find duplicate code blocks using hashing."""
        print("🔍 Finding duplicate code blocks...")

        code_blocks = defaultdict(list)

        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()

                # Check for duplicate blocks of 10+ lines
                for i in range(len(lines) - 10):
                    block = "".join(lines[i : i + 10])
                    # Skip blocks that are mostly comments or whitespace
                    if block.count("#") < 5 and len(block.strip()) > 100:
                        block_hash = hashlib.md5(block.encode()).hexdigest()
                        code_blocks[block_hash].append(
                            {
                                "file": str(file_path.relative_to(self.base_path.parent)),
                                "line": i + 1,
                            }
                        )
            except Exception:
                pass

        # Find duplicates
        duplicates = []
        for block_hash, locations in code_blocks.items():
            if len(locations) > 1:
                duplicates.append({"locations": locations, "count": len(locations)})

        # Sort by count and limit
        duplicates.sort(key=lambda x: x["count"], reverse=True)
        self.results["duplicate_code"] = duplicates[:10]

    def _calculate_statistics(self, files: list[Path]):
        """Calculate overall statistics."""
        print("📊 Calculating statistics...")

        total_lines = 0
        total_files = len(files)

        for file_path in files:
            try:
                with open(file_path) as f:
                    total_lines += len(f.readlines())
            except:
                pass

        self.results["statistics"] = {
            "total_files": total_files,
            "total_lines": total_lines,
            "large_files_count": len(self.results["large_files"]),
            "circular_imports_count": len(self.results["circular_imports"]),
            "duplicate_blocks_count": len(self.results["duplicate_code"]),
            "empty_modules_count": len(self.results["empty_modules"]),
        }

    def _path_to_module(self, path: Path) -> str:
        """Convert file path to module name."""
        relative = path.relative_to(self.base_path.parent)
        module = str(relative).replace(os.sep, ".").replace(".py", "")
        return f"main.{module}" if not module.startswith("main.") else module

    def print_report(self):
        """Print a formatted report."""
        print("\n" + "=" * 60)
        print("CODE ANALYSIS REPORT")
        print("=" * 60)

        # Statistics
        stats = self.results["statistics"]
        print("\n📊 Overall Statistics:")
        print(f"  Total Files: {stats['total_files']}")
        print(f"  Total Lines: {stats['total_lines']:,}")
        print(f"  Large Files (>500 lines): {stats['large_files_count']}")
        print(f"  Circular Import Patterns: {stats['circular_imports_count']}")
        print(f"  Duplicate Code Blocks: {stats['duplicate_blocks_count']}")
        print(f"  Empty Modules: {stats['empty_modules_count']}")

        # Large files
        if self.results["large_files"]:
            print("\n📏 Top 10 Largest Files:")
            for i, file_info in enumerate(self.results["large_files"][:10], 1):
                print(f"  {i}. {file_info['file']} ({file_info['lines']} lines)")

        # Circular imports
        if self.results["circular_imports"]:
            print("\n🔁 Circular Import Patterns Found:")
            for i, cycle in enumerate(self.results["circular_imports"][:5], 1):
                cycle_str = " → ".join(cycle[-3:]) if len(cycle) > 3 else " → ".join(cycle)
                print(f"  {i}. {cycle_str}")

        # Duplicate code
        if self.results["duplicate_code"]:
            print("\n📋 Top Duplicate Code Blocks:")
            for i, dup in enumerate(self.results["duplicate_code"][:5], 1):
                locs = dup["locations"][:2]  # Show first 2 locations
                print(f"  {i}. Found in {dup['count']} locations:")
                for loc in locs:
                    print(f"     - {loc['file']}:{loc['line']}")

        # Empty modules
        if self.results["empty_modules"]:
            print("\n📭 Empty Modules:")
            for module in self.results["empty_modules"][:10]:
                print(f"  - {module}/")

        print("\n" + "=" * 60)

    def save_results(self, output_file: str = "code_analysis_results.json"):
        """Save results to JSON file."""
        # Remove the import graph from saved results (too large)
        save_data = {k: v for k, v in self.results.items() if k != "import_graph"}

        with open(output_file, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"\n💾 Results saved to {output_file}")


def main():
    """Run the code analysis."""
    analyzer = CodeAnalyzer()
    analyzer.analyze()
    analyzer.print_report()
    analyzer.save_results()


if __name__ == "__main__":
    main()
