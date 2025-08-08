"""
Model version control and lifecycle management.

This module provides comprehensive version control for models including:
- Semantic versioning
- Version lifecycle management
- Dependencies and compatibility
- Version comparison and rollback
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    timer
)
from main.utils.database import DatabasePool
from main.utils.monitoring import record_metric

from main.models.inference.model_registry_types import (
    ModelVersion,
    DeploymentStatus
)

logger = get_logger(__name__)


class VersionStatus(Enum):
    """Model version status."""
    DRAFT = "draft"
    TESTING = "testing"
    STAGED = "staged"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class VersionType(Enum):
    """Version increment types."""
    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features, backward compatible
    PATCH = "patch"  # Bug fixes, backward compatible
    HOTFIX = "hotfix"  # Emergency fixes


@dataclass
class VersionDependency:
    """Version dependency specification."""
    model_name: str
    version_constraint: str  # e.g., ">=1.2.0,<2.0.0"
    required: bool = True
    description: str = ""


@dataclass
class VersionCompatibility:
    """Version compatibility information."""
    compatible_versions: List[str]
    breaking_changes: List[str] = field(default_factory=list)
    migration_notes: str = ""
    deprecation_warnings: List[str] = field(default_factory=list)


@dataclass
class VersionLifecycle:
    """Version lifecycle tracking."""
    version: str
    status: VersionStatus
    created_at: datetime
    updated_at: datetime
    deployed_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    lifecycle_events: List[Dict[str, Any]] = field(default_factory=list)


class SemanticVersion:
    """Semantic version parser and comparator."""
    
    def __init__(self, version: str):
        """Parse semantic version."""
        self.original = version
        self.major, self.minor, self.patch, self.prerelease, self.build = \
            self._parse_version(version)
    
    def _parse_version(self, version: str) -> Tuple[int, int, int, str, str]:
        """Parse version string into components."""
        # Remove 'v' prefix if present
        version = version.lstrip('v')
        
        # Split on '+' for build metadata
        if '+' in version:
            version, build = version.split('+', 1)
        else:
            build = ""
        
        # Split on '-' for prerelease
        if '-' in version:
            version, prerelease = version.split('-', 1)
        else:
            prerelease = ""
        
        # Parse major.minor.patch
        parts = version.split('.')
        
        if len(parts) < 3:
            # Pad with zeros
            parts.extend(['0'] * (3 - len(parts)))
        
        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])
        except ValueError:
            raise ValueError(f"Invalid version format: {self.original}")
        
        return major, minor, patch, prerelease, build
    
    def __str__(self) -> str:
        """String representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.prerelease:
            version += f"-{self.prerelease}"
        
        if self.build:
            version += f"+{self.build}"
        
        return version
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, SemanticVersion):
            other = SemanticVersion(str(other))
        
        return (self.major == other.major and
                self.minor == other.minor and
                self.patch == other.patch and
                self.prerelease == other.prerelease)
    
    def __lt__(self, other) -> bool:
        """Less than comparison."""
        if not isinstance(other, SemanticVersion):
            other = SemanticVersion(str(other))
        
        # Compare major.minor.patch
        if (self.major, self.minor, self.patch) != \
           (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < \
                   (other.major, other.minor, other.patch)
        
        # Handle prerelease
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        
        return False
    
    def __le__(self, other) -> bool:
        return self == other or self < other
    
    def __gt__(self, other) -> bool:
        return not self <= other
    
    def __ge__(self, other) -> bool:
        return not self < other
    
    def increment(
        self,
        version_type: VersionType,
        prerelease: Optional[str] = None
    ) -> 'SemanticVersion':
        """Increment version based on type."""
        major, minor, patch = self.major, self.minor, self.patch
        
        if version_type == VersionType.MAJOR:
            major += 1
            minor = 0
            patch = 0
        elif version_type == VersionType.MINOR:
            minor += 1
            patch = 0
        elif version_type in [VersionType.PATCH, VersionType.HOTFIX]:
            patch += 1
        
        new_version = f"{major}.{minor}.{patch}"
        
        if prerelease:
            new_version += f"-{prerelease}"
        
        return SemanticVersion(new_version)
    
    def satisfies_constraint(self, constraint: str) -> bool:
        """Check if version satisfies constraint."""
        # Parse constraint (simplified implementation)
        constraints = constraint.split(',')
        
        for c in constraints:
            c = c.strip()
            
            if c.startswith('>='): 
                min_version = SemanticVersion(c[2:].strip())
                if not self >= min_version:
                    return False
            elif c.startswith('>'):
                min_version = SemanticVersion(c[1:].strip())
                if not self > min_version:
                    return False
            elif c.startswith('<='): 
                max_version = SemanticVersion(c[2:].strip())
                if not self <= max_version:
                    return False
            elif c.startswith('<'):
                max_version = SemanticVersion(c[1:].strip())
                if not self < max_version:
                    return False
            elif c.startswith('='):
                exact_version = SemanticVersion(c[1:].strip())
                if not self == exact_version:
                    return False
            else:
                # Assume exact match
                exact_version = SemanticVersion(c)
                if not self == exact_version:
                    return False
        
        return True
    
    def is_compatible_with(self, other: 'SemanticVersion') -> bool:
        """Check if versions are compatible (same major version)."""
        return self.major == other.major


class VersionController(ErrorHandlingMixin):
    """
    Controls model version lifecycle and management.
    
    Features:
    - Semantic versioning
    - Version lifecycle management
    - Dependency tracking
    - Compatibility checking
    - Rollback capabilities
    """
    
    def __init__(self, db_pool: DatabasePool):
        """Initialize version controller."""
        super().__init__()
        self.db_pool = db_pool
        
        # Version cache
        self._version_cache: Dict[str, List[ModelVersion]] = {}
        self._lifecycle_cache: Dict[str, VersionLifecycle] = {}
        
    async def create_version(
        self,
        model_name: str,
        version_type: VersionType,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[VersionDependency]] = None,
        prerelease: Optional[str] = None
    ) -> ModelVersion:
        """
        Create a new model version.
        
        Args:
            model_name: Name of the model
            version_type: Type of version increment
            metadata: Optional version metadata
            dependencies: Optional version dependencies
            prerelease: Optional prerelease identifier
            
        Returns:
            Created model version
        """
        with self._handle_error("creating model version"):
            # Get latest version
            latest_version = await self.get_latest_version(model_name)
            
            if latest_version:
                # Increment from latest
                current_semver = SemanticVersion(latest_version.version)
                new_semver = current_semver.increment(version_type, prerelease)
            else:
                # First version
                if prerelease:
                    new_semver = SemanticVersion(f"1.0.0-{prerelease}")
                else:
                    new_semver = SemanticVersion("1.0.0")
            
            new_version = str(new_semver)
            
            # Validate dependencies
            if dependencies:
                await self._validate_dependencies(dependencies)
            
            # Create version
            model_version = ModelVersion(
                id=None,  # Will be set by database
                model_name=model_name,
                version=new_version,
                created_at=datetime.utcnow(),
                status=VersionStatus.DRAFT.value,
                metadata=metadata or {}
            )
            
            # Store in database
            version_id = await self._store_version(
                model_version, dependencies
            )
            model_version.id = version_id
            
            # Initialize lifecycle
            await self._initialize_lifecycle(model_name, new_version)
            
            # Clear cache
            self._clear_version_cache(model_name)
            
            logger.info(
                f"Created version {new_version} for model {model_name} "
                f"(type: {version_type.value})"
            )
            
            record_metric(
                'version_controller.version_created',
                1,
                tags={
                    'model': model_name,
                    'version': new_version,
                    'type': version_type.value
                }
            )
            
            return model_version
    
    async def promote_version(
        self,
        model_name: str,
        version: str,
        target_status: VersionStatus,
        validation_results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Promote a version to a new status.
        
        Args:
            model_name: Name of the model
            version: Version to promote
            target_status: Target status
            validation_results: Optional validation results
            
        Returns:
            True if promotion succeeded
        """
        with self._handle_error("promoting version"):
            # Get current version
            model_version = await self.get_version(model_name, version)
            
            if not model_version:
                raise ValueError(f"Version {version} not found for {model_name}")
            
            current_status = VersionStatus(model_version.status)
            
            # Validate promotion path
            if not self._is_valid_promotion(current_status, target_status):
                raise ValueError(
                    f"Invalid promotion from {current_status.value} "
                    f"to {target_status.value}"
                )
            
            # Check dependencies if promoting to production
            if target_status == VersionStatus.PRODUCTION:
                await self._validate_production_requirements(
                    model_name, version
                )
            
            # Update version status
            await self._update_version_status(
                model_name, version, target_status
            )
            
            # Update lifecycle
            await self._update_lifecycle(
                model_name, version, target_status, validation_results
            )
            
            # Handle status-specific actions
            if target_status == VersionStatus.DEPRECATED:
                await self._handle_deprecation(model_name, version)
            elif target_status == VersionStatus.ARCHIVED:
                await self._handle_archival(model_name, version)
            
            # Clear cache
            self._clear_version_cache(model_name)
            
            logger.info(
                f"Promoted {model_name} v{version} from "
                f"{current_status.value} to {target_status.value}"
            )
            
            record_metric(
                'version_controller.version_promoted',
                1,
                tags={
                    'model': model_name,
                    'version': version,
                    'from_status': current_status.value,
                    'to_status': target_status.value
                }
            )
            
            return True
    
    async def rollback_version(
        self,
        model_name: str,
        target_version: str,
        reason: str = "manual_rollback"
    ) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            model_name: Name of the model
            target_version: Version to rollback to
            reason: Reason for rollback
            
        Returns:
            True if rollback succeeded
        """
        with self._handle_error("rolling back version"):
            # Validate target version
            target = await self.get_version(model_name, target_version)
            
            if not target:
                raise ValueError(f"Target version {target_version} not found")
            
            if target.status not in [VersionStatus.PRODUCTION.value, 
                                   VersionStatus.STAGED.value]:
                raise ValueError(
                    f"Cannot rollback to version with status {target.status}"
                )
            
            # Get current production version
            current = await self.get_production_version(model_name)
            
            if not current:
                raise ValueError(f"No production version found for {model_name}")
            
            # Check compatibility
            is_compatible = await self._check_rollback_compatibility(
                current, target
            )
            
            if not is_compatible:
                logger.warning(
                    f"Rollback from {current.version} to {target_version} "
                    f"may have compatibility issues"
                )
            
            # Perform rollback
            # 1. Demote current version
            await self._update_version_status(
                model_name, current.version, VersionStatus.STAGED
            )
            
            # 2. Promote target version
            await self._update_version_status(
                model_name, target_version, VersionStatus.PRODUCTION
            )
            
            # 3. Record rollback event
            await self._record_rollback(
                model_name, current.version, target_version, reason
            )
            
            # Clear cache
            self._clear_version_cache(model_name)
            
            logger.info(
                f"Rolled back {model_name} from v{current.version} "
                f"to v{target_version} (reason: {reason})"
            )
            
            record_metric(
                'version_controller.version_rollback',
                1,
                tags={
                    'model': model_name,
                    'from_version': current.version,
                    'to_version': target_version,
                    'reason': reason
                }
            )
            
            return True
    
    async def get_version(
        self,
        model_name: str,
        version: str
    ) -> Optional[ModelVersion]:
        """Get a specific model version."""
        versions = await self.list_versions(model_name)
        
        for v in versions:
            if v.version == version:
                return v
        
        return None
    
    async def get_latest_version(
        self,
        model_name: str,
        status: Optional[VersionStatus] = None
    ) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        versions = await self.list_versions(model_name)
        
        if status:
            versions = [
                v for v in versions 
                if v.status == status.value
            ]
        
        if not versions:
            return None
        
        # Sort by semantic version
        version_pairs = [
            (SemanticVersion(v.version), v) for v in versions
        ]
        version_pairs.sort(key=lambda x: x[0], reverse=True)
        
        return version_pairs[0][1]
    
    async def get_production_version(
        self,
        model_name: str
    ) -> Optional[ModelVersion]:
        """Get the current production version."""
        return await self.get_latest_version(
            model_name, VersionStatus.PRODUCTION
        )
    
    async def list_versions(
        self,
        model_name: str,
        limit: Optional[int] = None
    ) -> List[ModelVersion]:
        """List all versions for a model."""
        # Check cache first
        if model_name in self._version_cache:
            versions = self._version_cache[model_name]
        else:
            # Load from database
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT * FROM model_versions
                    WHERE model_name = $1
                    ORDER BY created_at DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"
                
                rows = await conn.fetch(query, model_name)
                
                versions = [
                    ModelVersion(
                        id=row['id'],
                        model_name=row['model_name'],
                        version=row['version'],
                        created_at=row['created_at'],
                        status=row['status'],
                        metadata=json.loads(row['metadata']) 
                                 if row['metadata'] else {}
                    )
                    for row in rows
                ]
                
                # Cache results
                self._version_cache[model_name] = versions
        
        if limit:
            return versions[:limit]
        
        return versions
    
    async def check_compatibility(
        self,
        model_name: str,
        version_a: str,
        version_b: str
    ) -> VersionCompatibility:
        """Check compatibility between two versions."""
        semver_a = SemanticVersion(version_a)
        semver_b = SemanticVersion(version_b)
        
        # Get version metadata
        ver_a = await self.get_version(model_name, version_a)
        ver_b = await self.get_version(model_name, version_b)
        
        if not ver_a or not ver_b:
            raise ValueError("One or both versions not found")
        
        # Check basic compatibility
        compatible = semver_a.is_compatible_with(semver_b)
        
        # Analyze breaking changes
        breaking_changes = []
        if semver_a.major != semver_b.major:
            breaking_changes.append("Major version change")
        
        # Get compatibility metadata
        breaking_changes.extend(
            ver_b.metadata.get('breaking_changes', [])
        )
        
        deprecation_warnings = ver_b.metadata.get(
            'deprecation_warnings', []
        )
        
        migration_notes = ver_b.metadata.get('migration_notes', '')
        
        return VersionCompatibility(
            compatible_versions=[version_a, version_b] if compatible else [],
            breaking_changes=breaking_changes,
            migration_notes=migration_notes,
            deprecation_warnings=deprecation_warnings
        )
    
    def _is_valid_promotion(
        self,
        current: VersionStatus,
        target: VersionStatus
    ) -> bool:
        """Check if promotion path is valid."""
        valid_transitions = {
            VersionStatus.DRAFT: [VersionStatus.TESTING, VersionStatus.ARCHIVED],
            VersionStatus.TESTING: [VersionStatus.STAGED, VersionStatus.ARCHIVED],
            VersionStatus.STAGED: [VersionStatus.PRODUCTION, VersionStatus.ARCHIVED],
            VersionStatus.PRODUCTION: [VersionStatus.DEPRECATED],
            VersionStatus.DEPRECATED: [VersionStatus.ARCHIVED],
            VersionStatus.ARCHIVED: []  # Terminal state
        }
        
        return target in valid_transitions.get(current, [])
    
    async def _validate_dependencies(
        self,
        dependencies: List[VersionDependency]
    ) -> None:
        """Validate version dependencies."""
        for dep in dependencies:
            if dep.required:
                # Check if dependency model exists
                versions = await self.list_versions(dep.model_name)
                
                if not versions:
                    raise ValueError(
                        f"Required dependency {dep.model_name} not found"
                    )
                
                # Check if any version satisfies constraint
                satisfied = False
                for version in versions:
                    semver = SemanticVersion(version.version)
                    if semver.satisfies_constraint(dep.version_constraint):
                        satisfied = True
                        break
                
                if not satisfied:
                    raise ValueError(
                        f"No version of {dep.model_name} satisfies "
                        f"constraint {dep.version_constraint}"
                    )
    
    async def _validate_production_requirements(
        self,
        model_name: str,
        version: str
    ) -> None:
        """Validate requirements for production promotion."""
        # Check if version has been properly tested
        lifecycle = await self._get_lifecycle(model_name, version)
        
        if not lifecycle or lifecycle.status not in [
            VersionStatus.STAGED, VersionStatus.TESTING
        ]:
            raise ValueError(
                f"Version {version} must be staged or tested "
                f"before production promotion"
            )
        
        # Additional production checks could go here
        # (e.g., performance benchmarks, security scans)
    
    async def _store_version(
        self,
        version: ModelVersion,
        dependencies: Optional[List[VersionDependency]] = None
    ) -> int:
        """Store version in database."""
        async with self.db_pool.acquire() as conn:
            # Insert version
            version_id = await conn.fetchval(
                """
                INSERT INTO model_versions (
                    model_name, version, created_at, status, metadata
                ) VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                version.model_name,
                version.version,
                version.created_at,
                version.status,
                json.dumps(version.metadata)
            )
            
            # Store dependencies
            if dependencies:
                for dep in dependencies:
                    await conn.execute(
                        """
                        INSERT INTO version_dependencies (
                            version_id, model_name, version_constraint,
                            required, description
                        ) VALUES ($1, $2, $3, $4, $5)
                        """,
                        version_id,
                        dep.model_name,
                        dep.version_constraint,
                        dep.required,
                        dep.description
                    )
            
            return version_id
    
    async def _initialize_lifecycle(
        self,
        model_name: str,
        version: str
    ) -> None:
        """Initialize version lifecycle."""
        lifecycle = VersionLifecycle(
            version=version,
            status=VersionStatus.DRAFT,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            lifecycle_events=[
                {
                    'event': 'created',
                    'timestamp': datetime.utcnow().isoformat(),
                    'status': VersionStatus.DRAFT.value
                }
            ]
        )
        
        # Store in database
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO version_lifecycle (
                    model_name, version, status, created_at,
                    updated_at, events
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                model_name,
                version,
                lifecycle.status.value,
                lifecycle.created_at,
                lifecycle.updated_at,
                json.dumps(lifecycle.lifecycle_events)
            )
        
        # Cache
        lifecycle_key = f"{model_name}:{version}"
        self._lifecycle_cache[lifecycle_key] = lifecycle
    
    async def _update_version_status(
        self,
        model_name: str,
        version: str,
        status: VersionStatus
    ) -> None:
        """Update version status."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE model_versions
                SET status = $1
                WHERE model_name = $2 AND version = $3
                """,
                status.value,
                model_name,
                version
            )
    
    async def _update_lifecycle(
        self,
        model_name: str,
        version: str,
        status: VersionStatus,
        validation_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update version lifecycle."""
        lifecycle_key = f"{model_name}:{version}"
        
        # Get current lifecycle
        lifecycle = await self._get_lifecycle(model_name, version)
        
        if lifecycle:
            # Update status
            lifecycle.status = status
            lifecycle.updated_at = datetime.utcnow()
            
            # Add lifecycle event
            event = {
                'event': 'status_change',
                'timestamp': datetime.utcnow().isoformat(),
                'status': status.value
            }
            
            if validation_results:
                event['validation_results'] = validation_results
            
            lifecycle.lifecycle_events.append(event)
            
            # Update timestamps
            if status == VersionStatus.PRODUCTION:
                lifecycle.deployed_at = datetime.utcnow()
            elif status == VersionStatus.DEPRECATED:
                lifecycle.deprecated_at = datetime.utcnow()
            elif status == VersionStatus.ARCHIVED:
                lifecycle.archived_at = datetime.utcnow()
            
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE version_lifecycle
                    SET status = $1, updated_at = $2,
                        deployed_at = $3, deprecated_at = $4,
                        archived_at = $5, events = $6
                    WHERE model_name = $7 AND version = $8
                    """,
                    lifecycle.status.value,
                    lifecycle.updated_at,
                    lifecycle.deployed_at,
                    lifecycle.deprecated_at,
                    lifecycle.archived_at,
                    json.dumps(lifecycle.lifecycle_events),
                    model_name,
                    version
                )
            
            # Update cache
            self._lifecycle_cache[lifecycle_key] = lifecycle
    
    async def _get_lifecycle(
        self,
        model_name: str,
        version: str
    ) -> Optional[VersionLifecycle]:
        """Get version lifecycle."""
        lifecycle_key = f"{model_name}:{version}"
        
        # Check cache first
        if lifecycle_key in self._lifecycle_cache:
            return self._lifecycle_cache[lifecycle_key]
        
        # Load from database
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM version_lifecycle
                WHERE model_name = $1 AND version = $2
                """,
                model_name,
                version
            )
            
            if row:
                lifecycle = VersionLifecycle(
                    version=row['version'],
                    status=VersionStatus(row['status']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    deployed_at=row['deployed_at'],
                    deprecated_at=row['deprecated_at'],
                    archived_at=row['archived_at'],
                    lifecycle_events=json.loads(row['events'])
                )
                
                # Cache
                self._lifecycle_cache[lifecycle_key] = lifecycle
                
                return lifecycle
        
        return None
    
    async def _handle_deprecation(
        self,
        model_name: str,
        version: str
    ) -> None:
        """Handle version deprecation."""
        logger.info(f"Deprecated {model_name} v{version}")
        
        # Could trigger notifications, migration reminders, etc.
    
    async def _handle_archival(
        self,
        model_name: str,
        version: str
    ) -> None:
        """Handle version archival."""
        logger.info(f"Archived {model_name} v{version}")
        
        # Could trigger cleanup, artifact removal, etc.
    
    async def _check_rollback_compatibility(
        self,
        current: ModelVersion,
        target: ModelVersion
    ) -> bool:
        """Check if rollback is compatible."""
        current_semver = SemanticVersion(current.version)
        target_semver = SemanticVersion(target.version)
        
        # Generally safe to rollback to same major version
        return current_semver.is_compatible_with(target_semver)
    
    async def _record_rollback(
        self,
        model_name: str,
        from_version: str,
        to_version: str,
        reason: str
    ) -> None:
        """Record rollback event."""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO version_rollbacks (
                    model_name, from_version, to_version,
                    reason, timestamp
                ) VALUES ($1, $2, $3, $4, $5)
                """,
                model_name,
                from_version,
                to_version,
                reason,
                datetime.utcnow()
            )
    
    def _clear_version_cache(self, model_name: str) -> None:
        """Clear version cache for a model."""
        if model_name in self._version_cache:
            del self._version_cache[model_name]
        
        # Clear lifecycle cache for this model
        to_remove = [
            key for key in self._lifecycle_cache.keys()
            if key.startswith(f"{model_name}:")
        ]
        
        for key in to_remove:
            del self._lifecycle_cache[key]