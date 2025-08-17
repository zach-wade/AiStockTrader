"""
Validation Rule Registry

Manages and organizes validation rules by type, stage, and profile.
"""

# Standard library imports
from collections import defaultdict

# Local imports
from main.interfaces.data_pipeline.validation import ValidationSeverity, ValidationStage
from main.utils.core import get_logger

from .rule_definitions import RuleProfile, ValidationRule

logger = get_logger(__name__)


class RuleRegistry:
    """Registry for managing validation rules."""

    def __init__(self):
        """Initialize rule registry."""
        self.rules: dict[str, ValidationRule] = {}
        self.profiles: dict[str, RuleProfile] = {}

        # Indexes for efficient lookups
        self.rules_by_stage: dict[ValidationStage, list[ValidationRule]] = defaultdict(list)
        self.rules_by_data_type: dict[str, list[ValidationRule]] = defaultdict(list)
        self.rules_by_profile: dict[str, list[ValidationRule]] = defaultdict(list)
        self.rules_by_severity: dict[ValidationSeverity, list[ValidationRule]] = defaultdict(list)

    def register_rule(self, rule: ValidationRule) -> None:
        """
        Register a validation rule.

        Args:
            rule: ValidationRule to register
        """
        if rule.name in self.rules:
            logger.warning(f"Overwriting existing rule: {rule.name}")

        self.rules[rule.name] = rule

        # Update indexes
        for stage in rule.stages:
            self.rules_by_stage[stage].append(rule)

        for data_type in rule.data_types:
            self.rules_by_data_type[data_type].append(rule)

        for profile in rule.applies_to_profiles:
            self.rules_by_profile[profile].append(rule)

        self.rules_by_severity[rule.severity].append(rule)

        logger.debug(f"Registered rule: {rule.name}")

    def register_rules(self, rules: list[ValidationRule]) -> None:
        """Register multiple rules."""
        for rule in rules:
            self.register_rule(rule)

    def register_profile(self, profile: RuleProfile) -> None:
        """
        Register a validation profile.

        Args:
            profile: RuleProfile to register
        """
        self.profiles[profile.name] = profile
        logger.debug(f"Registered profile: {profile.name}")

    def register_profiles(self, profiles: dict[str, RuleProfile]) -> None:
        """Register multiple profiles."""
        for profile in profiles.values():
            self.register_profile(profile)

    def get_rule(self, rule_name: str) -> ValidationRule | None:
        """Get rule by name."""
        return self.rules.get(rule_name)

    def get_profile(self, profile_name: str) -> RuleProfile | None:
        """Get profile by name."""
        return self.profiles.get(profile_name)

    def get_rules_for_stage(
        self, stage: ValidationStage, profile: str | None = None
    ) -> list[ValidationRule]:
        """
        Get rules for a specific validation stage.

        Args:
            stage: Validation stage
            profile: Optional profile filter

        Returns:
            List of applicable rules
        """
        rules = self.rules_by_stage.get(stage, [])

        if profile:
            rules = [
                r
                for r in rules
                if profile in r.applies_to_profiles or "all" in r.applies_to_profiles
            ]

        return sorted(rules, key=lambda r: r.priority)

    def get_rules_for_data_type(
        self, data_type: str, profile: str | None = None, stage: ValidationStage | None = None
    ) -> list[ValidationRule]:
        """
        Get rules for a specific data type.

        Args:
            data_type: Data type (e.g., 'market_data', 'news')
            profile: Optional profile filter
            stage: Optional stage filter

        Returns:
            List of applicable rules
        """
        rules = self.rules_by_data_type.get(data_type, [])

        if profile:
            rules = [
                r
                for r in rules
                if profile in r.applies_to_profiles or "all" in r.applies_to_profiles
            ]

        if stage:
            rules = [r for r in rules if stage in r.stages]

        return sorted(rules, key=lambda r: r.priority)

    def get_rules_for_profile(
        self, profile_name: str, stage: ValidationStage | None = None, data_type: str | None = None
    ) -> list[ValidationRule]:
        """
        Get rules for a specific profile.

        Args:
            profile_name: Profile name
            stage: Optional stage filter
            data_type: Optional data type filter

        Returns:
            List of applicable rules
        """
        # Get rules that apply to this profile or "all"
        rules = self.rules_by_profile.get(profile_name, [])
        all_rules = self.rules_by_profile.get("all", [])
        rules = list(set(rules + all_rules))

        if stage:
            rules = [r for r in rules if stage in r.stages]

        if data_type:
            rules = [r for r in rules if data_type in r.data_types]

        # Filter by enabled checks in profile
        profile = self.profiles.get(profile_name)
        if profile:
            if "all" not in profile.enabled_checks:
                rules = [
                    r
                    for r in rules
                    if r.name in profile.enabled_checks
                    or r.severity.value in profile.enabled_checks
                ]

        return sorted(rules, key=lambda r: r.priority)

    def get_critical_rules(self) -> list[ValidationRule]:
        """Get all critical severity rules."""
        return self.rules_by_severity.get(ValidationSeverity.CRITICAL, [])

    def get_required_fields(self, data_type: str) -> set[str]:
        """
        Get required fields for a data type based on rules.

        Args:
            data_type: Data type

        Returns:
            Set of required field names
        """
        required_fields = set()
        rules = self.rules_by_data_type.get(data_type, [])

        for rule in rules:
            # Parse expression to find field references
            # This is simplified - could be enhanced with proper parsing
            if "df[" in rule.expression:
                # Standard library imports
                import re

                pattern = r"df\['([^']+)'\]|df\[\"([^\"]+)\"\]"
                matches = re.findall(pattern, rule.expression)
                for match in matches:
                    field = match[0] or match[1]
                    if field:
                        required_fields.add(field)

        return required_fields

    def get_rule_count(self) -> dict[str, int]:
        """Get counts of rules by various categories."""
        return {
            "total": len(self.rules),
            "by_severity": {
                severity.value: len(rules) for severity, rules in self.rules_by_severity.items()
            },
            "by_stage": {stage.value: len(rules) for stage, rules in self.rules_by_stage.items()},
            "by_data_type": {
                data_type: len(rules) for data_type, rules in self.rules_by_data_type.items()
            },
            "profiles": len(self.profiles),
        }

    def clear(self) -> None:
        """Clear all registered rules and profiles."""
        self.rules.clear()
        self.profiles.clear()
        self.rules_by_stage.clear()
        self.rules_by_data_type.clear()
        self.rules_by_profile.clear()
        self.rules_by_severity.clear()
        logger.info("Cleared rule registry")
