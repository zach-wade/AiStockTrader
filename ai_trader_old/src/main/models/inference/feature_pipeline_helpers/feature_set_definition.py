"""
Feature set definitions for model inference.

This module defines and manages feature sets used by different models,
including feature metadata, transformations, and validation rules.
"""

# Standard library imports
from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Local imports
from main.utils.core import ErrorHandlingMixin, get_logger

logger = get_logger(__name__)


class FeatureType(Enum):
    """Types of features."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TEXT = "text"
    EMBEDDING = "embedding"
    TIMESERIES = "timeseries"


class FeatureTransform(Enum):
    """Standard feature transformations."""

    NONE = "none"
    NORMALIZE = "normalize"
    STANDARDIZE = "standardize"
    LOG = "log"
    SQRT = "sqrt"
    RANK = "rank"
    CLIP = "clip"
    FILLNA = "fillna"


@dataclass
class FeatureDefinition:
    """Definition of a single feature."""

    name: str
    feature_type: FeatureType
    description: str = ""

    # Source information
    source_table: Optional[str] = None
    source_column: Optional[str] = None
    calculator: Optional[str] = None

    # Transformations
    transforms: List[FeatureTransform] = field(default_factory=list)
    transform_params: Dict[str, Any] = field(default_factory=dict)

    # Validation
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = True

    # Metadata
    importance: float = 0.0
    update_frequency: str = "daily"
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class FeatureSetDefinition:
    """Definition of a complete feature set for a model."""

    name: str
    version: str
    description: str

    # Features
    features: List[FeatureDefinition] = field(default_factory=list)

    # Model metadata
    model_type: str = ""
    target_variable: Optional[str] = None

    # Feature groups
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)

    # Validation rules
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        return [f.name for f in self.features]

    def get_required_features(self) -> List[str]:
        """Get required feature names."""
        return [f.name for f in self.features if f.required]

    def get_features_by_type(self, feature_type: FeatureType) -> List[FeatureDefinition]:
        """Get features of specific type."""
        return [f for f in self.features if f.feature_type == feature_type]

    def get_feature_by_name(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name."""
        for feature in self.features:
            if feature.name == name:
                return feature
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "features": [
                {
                    "name": f.name,
                    "type": f.feature_type.value,
                    "description": f.description,
                    "transforms": [t.value for t in f.transforms],
                    "required": f.required,
                    "importance": f.importance,
                }
                for f in self.features
            ],
            "model_type": self.model_type,
            "target_variable": self.target_variable,
            "feature_groups": self.feature_groups,
            "validation_rules": self.validation_rules,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSetDefinition":
        """Create from dictionary."""
        features = []
        for f_data in data.get("features", []):
            feature = FeatureDefinition(
                name=f_data["name"],
                feature_type=FeatureType(f_data["type"]),
                description=f_data.get("description", ""),
                transforms=[FeatureTransform(t) for t in f_data.get("transforms", [])],
                required=f_data.get("required", True),
                importance=f_data.get("importance", 0.0),
            )
            features.append(feature)

        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            features=features,
            model_type=data.get("model_type", ""),
            target_variable=data.get("target_variable"),
            feature_groups=data.get("feature_groups", {}),
            validation_rules=data.get("validation_rules", []),
            tags=data.get("tags", []),
        )


class FeatureSetManager(ErrorHandlingMixin):
    """
    Manages feature set definitions for models.

    Features:
    - Feature set registration and versioning
    - Feature validation
    - Transformation pipeline generation
    - Feature importance tracking
    """

    def __init__(self):
        """Initialize feature set manager."""
        super().__init__()
        self._feature_sets: Dict[str, FeatureSetDefinition] = {}
        self._validators: Dict[str, Callable] = {}

        # Register default validators
        self._register_default_validators()

    def register_feature_set(self, feature_set: FeatureSetDefinition) -> None:
        """Register a feature set."""
        key = f"{feature_set.name}:{feature_set.version}"
        self._feature_sets[key] = feature_set

        logger.info(
            f"Registered feature set: {feature_set.name} v{feature_set.version} "
            f"with {len(feature_set.features)} features"
        )

    def get_feature_set(
        self, name: str, version: Optional[str] = None
    ) -> Optional[FeatureSetDefinition]:
        """Get a feature set by name and version."""
        if version:
            key = f"{name}:{version}"
            return self._feature_sets.get(key)

        # Get latest version
        matching = [(k, v) for k, v in self._feature_sets.items() if k.startswith(f"{name}:")]

        if not matching:
            return None

        # Sort by version and return latest
        matching.sort(key=lambda x: x[0])
        return matching[-1][1]

    def validate_features(
        self, feature_set_name: str, features: Dict[str, Any], version: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate features against a feature set definition.

        Args:
            feature_set_name: Name of feature set
            features: Feature values to validate
            version: Optional version

        Returns:
            Tuple of (is_valid, error_messages)
        """
        feature_set = self.get_feature_set(feature_set_name, version)

        if not feature_set:
            return False, [f"Feature set not found: {feature_set_name}"]

        errors = []

        # Check required features
        required = feature_set.get_required_features()
        missing = [f for f in required if f not in features]

        if missing:
            errors.append(f"Missing required features: {missing}")

        # Validate each feature
        for feature_def in feature_set.features:
            if feature_def.name in features:
                value = features[feature_def.name]

                # Type validation
                if not self._validate_feature_type(value, feature_def.feature_type):
                    errors.append(
                        f"Invalid type for {feature_def.name}: "
                        f"expected {feature_def.feature_type.value}"
                    )

                # Range validation
                if feature_def.min_value is not None and value < feature_def.min_value:
                    errors.append(
                        f"{feature_def.name} below minimum: " f"{value} < {feature_def.min_value}"
                    )

                if feature_def.max_value is not None and value > feature_def.max_value:
                    errors.append(
                        f"{feature_def.name} above maximum: " f"{value} > {feature_def.max_value}"
                    )

                # Allowed values validation
                if feature_def.allowed_values and value not in feature_def.allowed_values:
                    errors.append(f"{feature_def.name} not in allowed values: {value}")

        # Run custom validation rules
        for rule in feature_set.validation_rules:
            validator = self._validators.get(rule.get("type"))
            if validator:
                valid, error = validator(features, rule)
                if not valid:
                    errors.append(error)

        return len(errors) == 0, errors

    def get_transformation_pipeline(
        self, feature_set_name: str, version: Optional[str] = None
    ) -> List[Tuple[str, FeatureTransform, Dict[str, Any]]]:
        """
        Get transformation pipeline for a feature set.

        Returns:
            List of (feature_name, transform, params) tuples
        """
        feature_set = self.get_feature_set(feature_set_name, version)

        if not feature_set:
            return []

        pipeline = []

        for feature in feature_set.features:
            for transform in feature.transforms:
                params = feature.transform_params.get(transform.value, {})
                pipeline.append((feature.name, transform, params))

        return pipeline

    def get_feature_importance(
        self, feature_set_name: str, version: Optional[str] = None
    ) -> Dict[str, float]:
        """Get feature importance scores."""
        feature_set = self.get_feature_set(feature_set_name, version)

        if not feature_set:
            return {}

        return {f.name: f.importance for f in feature_set.features}

    def _validate_feature_type(self, value: Any, expected_type: FeatureType) -> bool:
        """Validate feature type."""
        if expected_type == FeatureType.NUMERIC:
            return isinstance(value, (int, float))
        elif expected_type == FeatureType.CATEGORICAL:
            return isinstance(value, str)
        elif expected_type == FeatureType.BINARY:
            return isinstance(value, (bool, int)) and value in [0, 1, True, False]
        elif expected_type == FeatureType.TEXT:
            return isinstance(value, str)
        elif expected_type == FeatureType.EMBEDDING:
            return isinstance(value, (list, tuple)) and all(
                isinstance(x, (int, float)) for x in value
            )
        elif expected_type == FeatureType.TIMESERIES:
            return isinstance(value, (list, tuple))

        return True

    def _register_default_validators(self):
        """Register default validation rules."""

        def cross_feature_validator(
            features: Dict[str, Any], rule: Dict[str, Any]
        ) -> Tuple[bool, str]:
            """Validate relationships between features."""
            feature_a = rule.get("feature_a")
            feature_b = rule.get("feature_b")
            relation = rule.get("relation")

            if feature_a not in features or feature_b not in features:
                return True, ""  # Skip if features not present

            val_a = features[feature_a]
            val_b = features[feature_b]

            if relation == "less_than" and val_a >= val_b:
                return False, f"{feature_a} must be less than {feature_b}"
            elif relation == "greater_than" and val_a <= val_b:
                return False, f"{feature_a} must be greater than {feature_b}"

            return True, ""

        def sum_constraint_validator(
            features: Dict[str, Any], rule: Dict[str, Any]
        ) -> Tuple[bool, str]:
            """Validate sum constraints."""
            feature_names = rule.get("features", [])
            target_sum = rule.get("sum", 1.0)
            tolerance = rule.get("tolerance", 0.01)

            total = sum(features.get(f, 0) for f in feature_names)

            if abs(total - target_sum) > tolerance:
                return False, f"Sum of {feature_names} must equal {target_sum} (got {total})"

            return True, ""

        self._validators["cross_feature"] = cross_feature_validator
        self._validators["sum_constraint"] = sum_constraint_validator

    def create_default_feature_sets(self) -> None:
        """Create default feature set definitions."""
        # Technical features set
        technical_set = FeatureSetDefinition(
            name="technical_features",
            version="1.0",
            description="Technical analysis features",
            features=[
                FeatureDefinition(
                    name="returns",
                    feature_type=FeatureType.NUMERIC,
                    description="Simple returns",
                    transforms=[FeatureTransform.CLIP],
                    transform_params={"clip": {"min": -0.1, "max": 0.1}},
                    min_value=-1.0,
                    max_value=1.0,
                ),
                FeatureDefinition(
                    name="volume_ratio",
                    feature_type=FeatureType.NUMERIC,
                    description="Volume relative to average",
                    transforms=[FeatureTransform.LOG],
                    min_value=0.0,
                ),
                FeatureDefinition(
                    name="rsi",
                    feature_type=FeatureType.NUMERIC,
                    description="Relative Strength Index",
                    min_value=0.0,
                    max_value=100.0,
                ),
            ],
            feature_groups={"momentum": ["returns", "rsi"], "volume": ["volume_ratio"]},
        )

        self.register_feature_set(technical_set)

        # ML features set
        ml_set = FeatureSetDefinition(
            name="ml_features",
            version="1.0",
            description="Machine learning features",
            features=[
                FeatureDefinition(
                    name="price_embedding",
                    feature_type=FeatureType.EMBEDDING,
                    description="Price pattern embedding",
                    calculator="embedding_calculator",
                ),
                FeatureDefinition(
                    name="sentiment_score",
                    feature_type=FeatureType.NUMERIC,
                    description="Aggregate sentiment",
                    transforms=[FeatureTransform.STANDARDIZE],
                    min_value=-1.0,
                    max_value=1.0,
                ),
            ],
        )

        self.register_feature_set(ml_set)
