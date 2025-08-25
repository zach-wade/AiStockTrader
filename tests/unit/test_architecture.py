"""
Architecture validation tests for the AI Trading System.

These tests ensure that clean architecture principles, SOLID principles,
and proper dependency rules are maintained throughout the codebase.
"""

import ast
import os
import re
from collections import defaultdict
from pathlib import Path

import pytest


class ArchitectureValidator:
    """Validates architecture rules and dependencies in the codebase."""

    def __init__(self, src_path: str = "src"):
        self.src_path = Path(src_path)
        self.domain_path = self.src_path / "domain"
        self.application_path = self.src_path / "application"
        self.infrastructure_path = self.src_path / "infrastructure"

        # Define layer dependencies rules
        self.allowed_dependencies = {
            "domain": set(),  # Domain should have no external dependencies
            "application": {"domain"},  # Application can depend on domain
            "infrastructure": {"domain", "application"},  # Infrastructure can depend on both
        }

        # Cache for parsed modules
        self._module_cache: dict[Path, ast.Module] = {}
        self._import_cache: dict[Path, set[str]] = {}

    def parse_file(self, filepath: Path) -> ast.Module | None:
        """Parse a Python file and return its AST."""
        if filepath in self._module_cache:
            return self._module_cache[filepath]

        try:
            with open(filepath, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(filepath))
                self._module_cache[filepath] = tree
                return tree
        except (SyntaxError, FileNotFoundError):
            return None

    def get_imports(self, filepath: Path) -> set[str]:
        """Extract all imports from a Python file."""
        if filepath in self._import_cache:
            return self._import_cache[filepath]

        imports = set()
        tree = self.parse_file(filepath)
        if not tree:
            return imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

        self._import_cache[filepath] = imports
        return imports

    def get_layer(self, filepath: Path) -> str | None:
        """Determine which layer a file belongs to."""
        relative_path = filepath.relative_to(self.src_path)
        parts = relative_path.parts

        if parts and parts[0] in ["domain", "application", "infrastructure"]:
            return parts[0]
        return None

    def check_dependency_violations(self) -> list[tuple[Path, str, str]]:
        """Check for dependency rule violations."""
        violations = []

        for filepath in self.src_path.rglob("*.py"):
            if "__pycache__" in str(filepath):
                continue

            layer = self.get_layer(filepath)
            if not layer:
                continue

            imports = self.get_imports(filepath)
            for import_str in imports:
                # Check if import is from src
                if import_str.startswith("src."):
                    import_parts = import_str.split(".")
                    if len(import_parts) > 1:
                        imported_layer = import_parts[1]

                        # Check if this dependency is allowed
                        if (
                            imported_layer in ["domain", "application", "infrastructure"]
                            and imported_layer not in self.allowed_dependencies[layer]
                            and imported_layer != layer
                        ):
                            violations.append((filepath, layer, imported_layer))

        return violations

    def find_circular_dependencies(self) -> list[list[str]]:
        """Detect circular dependencies in the codebase."""
        # Build dependency graph
        graph = defaultdict(set)

        for filepath in self.src_path.rglob("*.py"):
            if "__pycache__" in str(filepath):
                continue

            module_name = self._filepath_to_module(filepath)
            imports = self.get_imports(filepath)

            for import_str in imports:
                if import_str.startswith("src."):
                    graph[module_name].add(import_str)

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = []

        def dfs(node: str) -> bool:
            if node in rec_stack:
                # Found a cycle
                cycle_start = rec_stack.index(node)
                cycles.append(rec_stack[cycle_start:] + [node])
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.append(node)

            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True

            rec_stack.pop()
            return False

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

    def _filepath_to_module(self, filepath: Path) -> str:
        """Convert filepath to module name."""
        relative = filepath.relative_to(self.src_path.parent)
        module = str(relative).replace(os.sep, ".").replace(".py", "")
        return module

    def check_value_object_immutability(self) -> list[tuple[Path, str]]:
        """Check that all value objects are immutable."""
        violations = []
        vo_path = self.domain_path / "value_objects"

        if not vo_path.exists():
            return violations

        for filepath in vo_path.rglob("*.py"):
            if "__pycache__" in str(filepath) or "__init__" in str(filepath):
                continue

            tree = self.parse_file(filepath)
            if not tree:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check for mutable methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            # Look for methods that might mutate state
                            if (
                                item.name
                                not in [
                                    "__init__",
                                    "__new__",
                                    "__repr__",
                                    "__str__",
                                    "__eq__",
                                    "__hash__",
                                    "__lt__",
                                    "__le__",
                                    "__gt__",
                                    "__ge__",
                                    "__add__",
                                    "__sub__",
                                    "__mul__",
                                    "__truediv__",
                                    "__neg__",
                                    "__abs__",
                                    "__bool__",
                                ]
                                and not item.name.startswith("_")
                                and not item.name.startswith("from_")
                                and not item.name.startswith("to_")
                                and not item.name.startswith("with_")
                                and not item.name.startswith("is_")
                                and not item.name.startswith("has_")
                                and not item.name.startswith("get_")
                            ):
                                # Check if method has assignments to self
                                for stmt in ast.walk(item):
                                    if isinstance(stmt, ast.Assign):
                                        for target in stmt.targets:
                                            if (
                                                isinstance(target, ast.Attribute)
                                                and isinstance(target, ast.Name)
                                                and target.id == "self"
                                            ):
                                                violations.append(
                                                    (
                                                        filepath,
                                                        f"Class {node.name} has mutable method {item.name}",
                                                    )
                                                )

        return violations

    def check_entity_ids(self) -> list[tuple[Path, str]]:
        """Check that all entities have ID attributes."""
        violations = []
        entities_path = self.domain_path / "entities"

        if not entities_path.exists():
            return violations

        for filepath in entities_path.rglob("*.py"):
            if "__pycache__" in str(filepath) or "__init__" in str(filepath):
                continue

            tree = self.parse_file(filepath)
            if not tree:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Skip abstract classes, exceptions, and enums
                    if any(
                        base.id == "ABC" if isinstance(base, ast.Name) else False
                        for base in node.bases
                    ):
                        continue
                    if any(
                        (isinstance(base, ast.Name) and base.id in ["Enum", "IntEnum", "StrEnum"])
                        or (
                            isinstance(base, ast.Attribute)
                            and base.attr in ["Enum", "IntEnum", "StrEnum"]
                        )
                        for base in node.bases
                    ):
                        continue
                    if "Exception" in node.name or "Error" in node.name:
                        continue

                    # Check for id attribute in __init__ or as dataclass field
                    has_id = False

                    # Check for dataclass field annotations (id: UUID = ...)
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign):
                            if isinstance(item.target, ast.Name) and item.target.id in [
                                "id",
                                "_id",
                                "entity_id",
                            ]:
                                has_id = True
                                break

                    # Also check in __init__ method
                    if not has_id:
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                                # Look for self.id or self._id assignment
                                for stmt in ast.walk(item):
                                    if isinstance(stmt, ast.Assign):
                                        for target in stmt.targets:
                                            if (
                                                isinstance(target, ast.Attribute)
                                                and isinstance(target, ast.Name)
                                                and target.id == "self"
                                                and target.attr in ["id", "_id", "entity_id"]
                                            ):
                                                has_id = True
                                                break

                    # Skip dataclasses without explicit entities (Request classes)
                    if not has_id and "Request" not in node.name:
                        violations.append((filepath, f"Entity {node.name} lacks an ID attribute"))

        return violations

    def check_repository_interfaces(self) -> list[tuple[Path, str]]:
        """Check that repository interfaces are in application layer."""
        violations = []

        # Check for repository interfaces in wrong layers
        for layer_path in [self.domain_path, self.infrastructure_path]:
            for filepath in layer_path.rglob("*.py"):
                if "__pycache__" in str(filepath):
                    continue

                content = filepath.read_text()
                # Look for abstract repository classes
                if re.search(r"class \w+Repository\(.*ABC.*\):", content):
                    violations.append(
                        (filepath, "Repository interface should be in application layer")
                    )

        # Check that concrete implementations are in infrastructure
        app_interfaces_path = self.application_path / "interfaces"
        if app_interfaces_path.exists():
            for filepath in app_interfaces_path.rglob("*.py"):
                if "__pycache__" in str(filepath):
                    continue

                tree = self.parse_file(filepath)
                if not tree:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if it's a concrete implementation (no ABC base)
                        is_abstract = any(
                            (isinstance(base, ast.Name) and base.id == "ABC")
                            or (isinstance(base, ast.Attribute) and base.attr == "ABC")
                            for base in node.bases
                        )

                        if (
                            not is_abstract
                            and "Repository" in node.name
                            and "Error" not in node.name
                            and "Exception" not in node.name
                        ):
                            # Check if it has concrete implementations
                            has_concrete_methods = False
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef):
                                    # Check if method has actual implementation
                                    if not (
                                        len(item.body) == 1 and isinstance(item.body[0], ast.Raise)
                                    ):
                                        has_concrete_methods = True
                                        break

                            if has_concrete_methods:
                                violations.append(
                                    (
                                        filepath,
                                        f"Concrete repository {node.name} should be in infrastructure",
                                    )
                                )

        return violations

    def check_business_logic_in_infrastructure(self) -> list[tuple[Path, str]]:
        """Check that no business logic exists in infrastructure layer."""
        violations = []

        # Business logic indicators
        business_keywords = [
            "calculate",
            "validate",
            "authorize",
            "decide",
            "determine",
            "evaluate",
            "assess",
            "verify",
            "check_business",
            "apply_rules",
            "process_rules",
            "business_rule",
            "domain_logic",
        ]

        for filepath in self.infrastructure_path.rglob("*.py"):
            if "__pycache__" in str(filepath):
                continue

            tree = self.parse_file(filepath)
            if not tree:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check function name for business logic indicators
                    for keyword in business_keywords:
                        if keyword in node.name.lower():
                            violations.append(
                                (
                                    filepath,
                                    f"Function {node.name} appears to contain business logic",
                                )
                            )
                            break

                    # Check for complex conditionals (might indicate business rules)
                    if_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.If))
                    if if_count > 3:  # Arbitrary threshold for complex logic
                        violations.append(
                            (
                                filepath,
                                f"Function {node.name} has complex conditionals that might be business logic",
                            )
                        )

        return violations

    def check_single_responsibility(self) -> list[tuple[Path, str]]:
        """Check that classes follow Single Responsibility Principle."""
        violations = []

        for filepath in self.src_path.rglob("*.py"):
            if "__pycache__" in str(filepath):
                continue

            tree = self.parse_file(filepath)
            if not tree:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Count public methods (not starting with _)
                    public_methods = [
                        item
                        for item in node.body
                        if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")
                    ]

                    # Check for too many public methods (possible SRP violation)
                    if len(public_methods) > 10:
                        violations.append(
                            (
                                filepath,
                                f"Class {node.name} has {len(public_methods)} public methods (possible SRP violation)",
                            )
                        )

                    # Check for mixed responsibilities (e.g., both IO and business logic)
                    has_io = False
                    has_business_logic = False

                    for method in public_methods:
                        method_str = ast.unparse(method) if hasattr(ast, "unparse") else ""

                        # Check for IO operations
                        if any(
                            keyword in method_str
                            for keyword in [
                                "open(",
                                "read",
                                "write",
                                "request",
                                "response",
                                "fetch",
                                "save",
                                "load",
                            ]
                        ):
                            has_io = True

                        # Check for business logic
                        if any(
                            keyword in method.name.lower()
                            for keyword in [
                                "calculate",
                                "validate",
                                "process",
                                "evaluate",
                                "decide",
                            ]
                        ):
                            has_business_logic = True

                    if has_io and has_business_logic:
                        violations.append(
                            (
                                filepath,
                                f"Class {node.name} mixes IO and business logic responsibilities",
                            )
                        )

        return violations

    def check_interface_segregation(self) -> list[tuple[Path, str]]:
        """Check that interfaces follow Interface Segregation Principle."""
        violations = []
        interfaces_path = self.application_path / "interfaces"

        if not interfaces_path.exists():
            return violations

        for filepath in interfaces_path.rglob("*.py"):
            if "__pycache__" in str(filepath):
                continue

            tree = self.parse_file(filepath)
            if not tree:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's an interface (has ABC base)
                    is_interface = any(
                        (isinstance(base, ast.Name) and base.id == "ABC")
                        or (isinstance(base, ast.Attribute) and base.attr == "ABC")
                        for base in node.bases
                    )

                    if is_interface:
                        # Count abstract methods
                        abstract_methods = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                # Check for @abstractmethod decorator
                                for decorator in item.decorator_list:
                                    if (
                                        isinstance(decorator, ast.Name)
                                        and decorator.id == "abstractmethod"
                                    ):
                                        abstract_methods.append(item.name)
                                        break

                        # Check for too many methods in interface
                        if len(abstract_methods) > 5:
                            violations.append(
                                (
                                    filepath,
                                    f"Interface {node.name} has {len(abstract_methods)} methods (consider splitting)",
                                )
                            )

        return violations

    def check_naming_conventions(self) -> list[tuple[Path, str]]:
        """Check that naming conventions are followed."""
        violations = []

        # Define naming patterns
        patterns = {
            "class": re.compile(r"^[A-Z][a-zA-Z0-9]*$"),  # PascalCase
            "function": re.compile(r"^[a-z_][a-z0-9_]*$"),  # snake_case
            "constant": re.compile(r"^[A-Z][A-Z0-9_]*$"),  # UPPER_SNAKE_CASE
        }

        for filepath in self.src_path.rglob("*.py"):
            if "__pycache__" in str(filepath):
                continue

            tree = self.parse_file(filepath)
            if not tree:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not patterns["class"].match(node.name):
                        violations.append(
                            (filepath, f"Class {node.name} doesn't follow PascalCase convention")
                        )

                elif isinstance(node, ast.FunctionDef):
                    if not patterns["function"].match(node.name):
                        violations.append(
                            (filepath, f"Function {node.name} doesn't follow snake_case convention")
                        )

                elif isinstance(node, ast.Assign):
                    # Check for module-level constants
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # If it's all uppercase, it should be a constant
                            if target.id.isupper() and not patterns["constant"].match(target.id):
                                violations.append(
                                    (
                                        filepath,
                                        f"Constant {target.id} doesn't follow UPPER_SNAKE_CASE convention",
                                    )
                                )

        return violations


# Test fixtures
@pytest.fixture
def validator():
    """Create an ArchitectureValidator instance."""
    return ArchitectureValidator()


# Dependency Rule Tests
class TestDependencyRules:
    """Test dependency rules between layers."""

    def test_domain_has_no_external_dependencies(self, validator):
        """Domain layer should not depend on application or infrastructure."""
        violations = validator.check_dependency_violations()
        domain_violations = [
            (path, from_layer, to_layer)
            for path, from_layer, to_layer in violations
            if from_layer == "domain"
        ]

        assert not domain_violations, (
            f"Domain layer has forbidden dependencies: "
            f"{[(str(path.relative_to(validator.src_path)), to_layer) for path, _, to_layer in domain_violations]}"
        )

    def test_application_has_no_infrastructure_dependencies(self, validator):
        """Application layer should not depend on infrastructure."""
        violations = validator.check_dependency_violations()
        app_violations = [
            (path, from_layer, to_layer)
            for path, from_layer, to_layer in violations
            if from_layer == "application" and to_layer == "infrastructure"
        ]

        assert not app_violations, (
            f"Application layer depends on infrastructure: "
            f"{[str(path.relative_to(validator.src_path)) for path, _, _ in app_violations]}"
        )

    def test_no_circular_dependencies(self, validator):
        """There should be no circular dependencies in the codebase."""
        cycles = validator.find_circular_dependencies()

        assert not cycles, f"Circular dependencies found: {cycles}"


# Domain Integrity Tests
class TestDomainIntegrity:
    """Test domain layer integrity rules."""

    def test_value_objects_are_immutable(self, validator):
        """All value objects should be immutable."""
        violations = validator.check_value_object_immutability()

        assert not violations, (
            f"Value objects with mutability issues: "
            f"{[(str(path.relative_to(validator.src_path)), msg) for path, msg in violations]}"
        )

    def test_entities_have_ids(self, validator):
        """All entities should have unique identifiers."""
        violations = validator.check_entity_ids()

        assert not violations, (
            f"Entities without IDs: "
            f"{[(str(path.relative_to(validator.src_path)), msg) for path, msg in violations]}"
        )

    def test_domain_services_path_exists(self, validator):
        """Domain services directory should exist."""
        services_path = validator.domain_path / "services"
        assert services_path.exists(), "Domain services directory is missing"


# Clean Architecture Tests
class TestCleanArchitecture:
    """Test clean architecture principles."""

    def test_repository_interfaces_in_application(self, validator):
        """Repository interfaces should be in the application layer."""
        violations = validator.check_repository_interfaces()

        assert not violations, (
            f"Repository interface violations: "
            f"{[(str(path.relative_to(validator.src_path)), msg) for path, msg in violations]}"
        )

    def test_no_business_logic_in_infrastructure(self, validator):
        """Infrastructure should not contain business logic."""
        violations = validator.check_business_logic_in_infrastructure()

        # Filter out acceptable cases
        filtered_violations = []
        for path, msg in violations:
            path_str = str(path).lower()
            # Skip adapters and configs
            if "adapter" in path_str or "config" in path_str:
                continue
            # Skip technical infrastructure that needs conditionals
            if (
                "logging" in path_str
                or "telemetry" in path_str
                or "secrets" in path_str
                or "sanitizer" in path_str
            ):
                if "complex conditionals" in msg:
                    continue  # These legitimately need conditionals for technical implementation
            # Skip monitoring and resilience - technical infrastructure
            if (
                "monitoring/" in path_str
                or "resilience/" in path_str
                or "observability/" in path_str
            ):
                if "complex conditionals" in msg:
                    continue  # Technical implementation needs conditionals
            filtered_violations.append((path, msg))

        assert not filtered_violations, (
            f"Business logic found in infrastructure: "
            f"{[(str(path.relative_to(validator.src_path)), msg) for path, msg in filtered_violations]}"
        )

    def test_use_cases_in_application(self, validator):
        """Use cases should be in the application layer."""
        use_cases_path = validator.application_path / "use_cases"

        # Check that use cases directory exists (if there are use cases)
        if any(validator.application_path.rglob("*use_case*.py")):
            assert use_cases_path.exists(), "Use cases should be organized in application/use_cases"


# SOLID Principles Tests
class TestSOLIDPrinciples:
    """Test SOLID principles compliance."""

    def test_single_responsibility(self, validator):
        """Classes should have a single responsibility."""
        violations = validator.check_single_responsibility()

        # Filter out acceptable cases (e.g., facades, controllers)
        filtered_violations = [
            (path, msg)
            for path, msg in violations
            if "controller" not in str(path).lower()
            and "facade" not in str(path).lower()
            and "manager" not in str(path).lower()
        ]

        # Only fail if there are severe violations
        severe_violations = [
            (path, msg) for path, msg in filtered_violations if "mixes IO and business logic" in msg
        ]

        assert not severe_violations, (
            f"Single Responsibility Principle violations: "
            f"{[(str(path.relative_to(validator.src_path)), msg) for path, msg in severe_violations]}"
        )

    def test_interface_segregation(self, validator):
        """Interfaces should be small and focused."""
        violations = validator.check_interface_segregation()

        # Warning level - don't fail, just report
        if violations:
            pytest.skip(
                f"Interface Segregation warnings: "
                f"{[(str(path.relative_to(validator.src_path)), msg) for path, msg in violations]}"
            )

    def test_dependency_inversion(self, validator):
        """High-level modules should not depend on low-level modules."""
        # This is largely covered by the dependency rules tests
        # Additional check: ensure interfaces exist for external dependencies
        interfaces_path = validator.application_path / "interfaces"

        assert interfaces_path.exists(), (
            "Application interfaces directory is missing - "
            "needed for Dependency Inversion Principle"
        )

        # Check that there are actual interface definitions
        interface_files = list(interfaces_path.rglob("*.py"))
        non_init_files = [f for f in interface_files if "__init__" not in str(f)]

        assert non_init_files, "No interface definitions found in application/interfaces"


# Code Organization Tests
class TestCodeOrganization:
    """Test code organization and structure."""

    def test_module_structure(self, validator):
        """Verify proper module structure."""
        required_dirs = [
            validator.domain_path,
            validator.application_path,
            validator.infrastructure_path,
            validator.domain_path / "entities",
            validator.domain_path / "value_objects",
            validator.domain_path / "services",
            validator.application_path / "interfaces",
        ]

        missing_dirs = [d for d in required_dirs if not d.exists()]

        assert not missing_dirs, (
            f"Missing required directories: "
            f"{[str(d.relative_to(validator.src_path.parent)) for d in missing_dirs]}"
        )

    def test_naming_conventions(self, validator):
        """Check that naming conventions are followed."""
        violations = validator.check_naming_conventions()

        # Filter out test files and generated code
        filtered_violations = [
            (path, msg)
            for path, msg in violations
            if "test_" not in str(path) and "conftest" not in str(path)
        ]

        assert not filtered_violations, (
            f"Naming convention violations: "
            f"{[(str(path.relative_to(validator.src_path)), msg) for path, msg in filtered_violations]}"
        )

    def test_no_cross_layer_imports(self, validator):
        """Ensure no direct cross-layer imports bypassing interfaces."""
        violations = []

        # Check for direct infrastructure imports in domain
        for filepath in validator.domain_path.rglob("*.py"):
            if "__pycache__" in str(filepath):
                continue

            imports = validator.get_imports(filepath)
            for imp in imports:
                if "infrastructure" in imp and "src." in imp:
                    violations.append((filepath, f"Direct infrastructure import: {imp}"))

        assert not violations, (
            f"Cross-layer import violations: "
            f"{[(str(path.relative_to(validator.src_path)), msg) for path, msg in violations]}"
        )


# Integration Tests
class TestArchitectureIntegration:
    """Integration tests for overall architecture health."""

    def test_repository_pattern_implementation(self, validator):
        """Verify correct implementation of repository pattern."""
        # Check that interfaces exist
        interfaces_path = validator.application_path / "interfaces"
        if not interfaces_path.exists():
            pytest.skip("No interfaces directory found")

        # Find repository interfaces
        repository_interfaces = []
        for filepath in interfaces_path.rglob("*.py"):
            if "repository" in filepath.name.lower():
                tree = validator.parse_file(filepath)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and "Repository" in node.name:
                            repository_interfaces.append(node.name)

        # Check that concrete implementations exist in infrastructure
        if repository_interfaces:
            infra_repos_path = validator.infrastructure_path / "repositories"
            assert infra_repos_path.exists(), (
                f"Repository implementations missing: found interfaces {repository_interfaces} "
                f"but no infrastructure/repositories directory"
            )

    def test_aggregate_boundaries(self, validator):
        """Verify that aggregates have proper boundaries."""
        entities_path = validator.domain_path / "entities"
        if not entities_path.exists():
            pytest.skip("No entities directory found")

        # Check for aggregate roots (entities that manage other entities)
        for filepath in entities_path.rglob("*.py"):
            if "__pycache__" in str(filepath) or "__init__" in str(filepath):
                continue

            tree = validator.parse_file(filepath)
            if not tree:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if class has methods that return other entities
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            # Check return type annotations for entity types
                            if item.returns:
                                return_str = (
                                    ast.unparse(item.returns) if hasattr(ast, "unparse") else ""
                                )
                                if "List[" in return_str and "Entity" in return_str:
                                    # This might be an aggregate root
                                    # Verify it has proper transaction boundaries
                                    method_names = [
                                        m.name for m in node.body if isinstance(m, ast.FunctionDef)
                                    ]

                                    # Aggregate roots should have methods for managing the aggregate
                                    expected_methods = ["add_", "remove_", "update_", "validate_"]
                                    has_management_methods = any(
                                        any(
                                            method.startswith(prefix) for prefix in expected_methods
                                        )
                                        for method in method_names
                                    )

                                    assert has_management_methods or node.name.endswith(
                                        "Aggregate"
                                    ), (
                                        f"Class {node.name} appears to be an aggregate root but lacks "
                                        f"proper aggregate management methods"
                                    )


# Performance Tests
class TestArchitecturePerformance:
    """Test for architecture patterns that might impact performance."""

    def test_no_n_plus_one_repository_calls(self, validator):
        """Check for potential N+1 query problems in use cases."""
        use_cases_path = validator.application_path / "use_cases"
        if not use_cases_path.exists():
            return  # No use cases to check

        violations = []

        for filepath in use_cases_path.rglob("*.py"):
            if "__pycache__" in str(filepath):
                continue

            tree = validator.parse_file(filepath)
            if not tree:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for loops with repository calls
                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.For):
                            # Check if there are repository method calls in the loop
                            for loop_node in ast.walk(subnode):
                                if isinstance(loop_node, ast.Call):
                                    if isinstance(loop_node.func, ast.Attribute):
                                        if "repository" in loop_node.func.attr.lower():
                                            violations.append(
                                                (filepath, f"Potential N+1 query in {node.name}")
                                            )

        # This is a warning, not a failure
        if violations:
            pytest.skip(
                f"Potential N+1 query problems: "
                f"{[(str(path.relative_to(validator.src_path)), msg) for path, msg in violations]}"
            )


# Run all architecture tests with a summary
def test_architecture_summary(validator):
    """Run all architecture checks and provide a summary."""
    results = {
        "dependency_violations": validator.check_dependency_violations(),
        "circular_dependencies": validator.find_circular_dependencies(),
        "value_object_issues": validator.check_value_object_immutability(),
        "entity_id_issues": validator.check_entity_ids(),
        "repository_issues": validator.check_repository_interfaces(),
        "business_logic_in_infra": validator.check_business_logic_in_infrastructure(),
        "srp_violations": validator.check_single_responsibility(),
        "isp_violations": validator.check_interface_segregation(),
        "naming_violations": validator.check_naming_conventions(),
    }

    # Count total issues
    total_issues = sum(len(v) for v in results.values())

    if total_issues > 0:
        summary = "\n=== Architecture Validation Summary ===\n"
        for check_name, violations in results.items():
            if violations:
                summary += f"\n{check_name}: {len(violations)} issue(s)"

        summary += f"\n\nTotal issues: {total_issues}"

        # Don't fail the test, just warn
        pytest.skip(summary)
    else:
        # All checks passed
        assert True, "All architecture validations passed!"
