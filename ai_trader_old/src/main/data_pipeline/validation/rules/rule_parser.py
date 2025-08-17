"""
Validation Rule Parser

Loads and parses validation rules from YAML configurations.
"""

# Standard library imports
from pathlib import Path
from typing import Any

# Third-party imports
import yaml

# Local imports
from main.interfaces.data_pipeline.validation import ValidationSeverity, ValidationStage
from main.utils.core import get_logger

from .rule_definitions import (
    DEFAULT_FUNDAMENTALS_RULES,
    DEFAULT_MARKET_DATA_RULES,
    DEFAULT_NEWS_RULES,
    FailureAction,
    RuleProfile,
    ValidationRule,
)

logger = get_logger(__name__)


class RuleParser:
    """Parses validation rules from configuration files."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize rule parser.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.rules_config = {}

    def _get_default_config_path(self) -> Path:
        """Get default configuration path."""
        # Try to find validation_rules.yaml in various locations
        possible_paths = [
            Path("config/validation_rules.yaml"),
            Path("ai_trader/config/validation_rules.yaml"),
            Path(__file__).parent.parent / "config" / "validation_rules.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        logger.warning("No validation rules config file found, using defaults")
        return Path("validation_rules.yaml")  # Non-existent, will use defaults

    def load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    self.rules_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded validation rules from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load validation rules config: {e}")
                self.rules_config = {}
        else:
            logger.info("Using default validation rules configuration")
            self.rules_config = self._get_default_config()

        return self.rules_config

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration when no file is available."""
        return {
            "profiles": {
                "strict": {
                    "description": "Strict validation profile",
                    "enabled_checks": ["all"],
                    "thresholds": {
                        "max_nan_ratio": 0.1,
                        "max_price_deviation": 0.5,
                        "min_quality_score": 80,
                    },
                    "failure_actions": {
                        "critical": "STOP_PROCESSING",
                        "error": "FLAG_ROW",
                        "warning": "CONTINUE_WITH_WARNING",
                    },
                },
                "lenient": {
                    "description": "Lenient validation profile",
                    "enabled_checks": ["critical", "error"],
                    "thresholds": {
                        "max_nan_ratio": 0.3,
                        "max_price_deviation": 1.0,
                        "min_quality_score": 60,
                    },
                    "failure_actions": {
                        "critical": "FLAG_ROW",
                        "error": "CONTINUE_WITH_WARNING",
                        "warning": "CONTINUE_WITH_WARNING",
                    },
                },
            },
            "market_data_rules": DEFAULT_MARKET_DATA_RULES,
            "news_rules": DEFAULT_NEWS_RULES,
            "fundamentals_rules": DEFAULT_FUNDAMENTALS_RULES,
            "error_handling": {
                "collect_all_errors": False,
                "fail_fast": True,
                "max_errors_per_batch": 100,
            },
        }

    def parse_profiles(self) -> dict[str, RuleProfile]:
        """Parse validation profiles from configuration."""
        profiles = {}
        profiles_config = self.rules_config.get("profiles", {})

        for name, config in profiles_config.items():
            profiles[name] = RuleProfile(
                name=name,
                description=config.get("description", ""),
                enabled_checks=config.get("enabled_checks", []),
                thresholds=config.get("thresholds", {}),
                failure_actions={
                    k: FailureAction[v] if isinstance(v, str) else v
                    for k, v in config.get("failure_actions", {}).items()
                },
            )

        return profiles

    def parse_rules(self, rule_type: str) -> list[ValidationRule]:
        """
        Parse rules of a specific type.

        Args:
            rule_type: Type of rules to parse (market_data_rules, news_rules, etc.)

        Returns:
            List of parsed validation rules
        """
        rules = []
        rules_config = self.rules_config.get(rule_type, {})

        for name, config in rules_config.items():
            try:
                rule = self._create_rule_from_config(name, config)
                rules.append(rule)
            except Exception as e:
                logger.error(f"Failed to parse rule '{name}': {e}")

        return sorted(rules, key=lambda r: r.priority)

    def _create_rule_from_config(self, name: str, config: dict[str, Any]) -> ValidationRule:
        """Create ValidationRule from configuration dictionary."""
        # Parse severity
        severity_str = config.get("severity", "WARNING")
        if isinstance(severity_str, str):
            severity = ValidationSeverity[severity_str.upper()]
        else:
            severity = severity_str

        # Parse stages
        stages_raw = config.get("stages", ["INGEST"])
        stages = []
        for stage in stages_raw:
            if isinstance(stage, str):
                stages.append(ValidationStage[stage.upper()])
            else:
                stages.append(stage)

        # Parse failure action
        action_str = config.get("failure_action", "FLAG_AND_CONTINUE")
        if isinstance(action_str, str):
            failure_action = FailureAction[action_str.upper()]
        else:
            failure_action = action_str

        return ValidationRule(
            name=name,
            expression=config.get("expression", ""),
            error_message=config.get("error_message", f"Validation failed for {name}"),
            severity=severity,
            applies_to_profiles=config.get("applies_to_profiles", ["all"]),
            data_types=config.get("data_types", []),
            stages=stages,
            failure_action=failure_action,
            enabled=config.get("enabled", True),
            priority=config.get("priority", 0),
        )

    def validate_rule_definition(self, rule_def: dict[str, Any]) -> bool:
        """
        Validate a rule definition for completeness.

        Args:
            rule_def: Rule definition dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["expression", "error_message", "severity"]

        for field in required_fields:
            if field not in rule_def:
                logger.error(f"Rule definition missing required field: {field}")
                return False

        # Validate severity
        severity = rule_def.get("severity", "")
        if isinstance(severity, str):
            try:
                ValidationSeverity[severity.upper()]
            except KeyError:
                logger.error(f"Invalid severity: {severity}")
                return False

        # Validate stages if present
        if "stages" in rule_def:
            stages = rule_def["stages"]
            if not isinstance(stages, list):
                logger.error("Stages must be a list")
                return False

            for stage in stages:
                if isinstance(stage, str):
                    try:
                        ValidationStage[stage.upper()]
                    except KeyError:
                        logger.error(f"Invalid stage: {stage}")
                        return False

        return True
