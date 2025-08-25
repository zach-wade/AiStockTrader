"""
Audit configuration and retention policies for regulatory compliance.

This module provides comprehensive configuration management for audit logging,
including retention policies, compliance settings, performance tuning, and
security configurations.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .events import ComplianceRegulation
from .exceptions import AuditConfigError


class StorageBackend(Enum):
    """Supported audit storage backends."""

    FILE = "file"
    DATABASE = "database"
    EXTERNAL = "external"
    MEMORY = "memory"  # For testing only


class CompressionType(Enum):
    """Supported compression types for audit logs."""

    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


class EncryptionType(Enum):
    """Supported encryption types for audit logs."""

    NONE = "none"
    AES256 = "aes256"
    FERNET = "fernet"
    GPG = "gpg"


@dataclass
class RetentionPolicy:
    """
    Audit log retention policy configuration.

    Defines how long audit logs are retained based on regulation requirements,
    event types, and business needs.
    """

    # Base retention periods
    default_retention_days: int = 2555  # 7 years (SOX requirement)
    critical_event_retention_days: int = 3650  # 10 years
    authentication_retention_days: int = 1095  # 3 years
    configuration_retention_days: int = 2555  # 7 years

    # Regulation-specific retention
    regulation_retention: dict[ComplianceRegulation, int] = field(
        default_factory=lambda: {
            ComplianceRegulation.SOX: 2555,  # 7 years
            ComplianceRegulation.SEC: 1825,  # 5 years
            ComplianceRegulation.FINRA: 1095,  # 3 years
            ComplianceRegulation.CFTC: 1825,  # 5 years
            ComplianceRegulation.MIFID_II: 1825,  # 5 years
            ComplianceRegulation.GDPR: 365,  # 1 year (unless legally required longer)
            ComplianceRegulation.BASEL_III: 2190,  # 6 years
        }
    )

    # Archive settings
    archive_after_days: int = 365  # Archive old logs after 1 year
    compress_archived_logs: bool = True
    compression_type: CompressionType = CompressionType.GZIP

    # Cleanup settings
    auto_cleanup_enabled: bool = True
    cleanup_check_interval_hours: int = 24

    def get_retention_days(
        self, event_type: str, regulations: list[ComplianceRegulation], is_critical: bool = False
    ) -> int:
        """
        Get retention period for a specific event.

        Args:
            event_type: Type of audit event
            regulations: Applicable compliance regulations
            is_critical: Whether the event is critical

        Returns:
            Retention period in days
        """
        retention_days = self.default_retention_days

        # Apply critical event retention
        if is_critical:
            retention_days = max(retention_days, self.critical_event_retention_days)

        # Apply event-type specific retention
        if event_type == "authentication":
            retention_days = max(retention_days, self.authentication_retention_days)
        elif event_type == "configuration":
            retention_days = max(retention_days, self.configuration_retention_days)

        # Apply regulation-specific retention (use longest required period)
        for regulation in regulations:
            if regulation in self.regulation_retention:
                regulation_days = self.regulation_retention[regulation]
                retention_days = max(retention_days, regulation_days)

        return retention_days

    def validate(self) -> None:
        """Validate retention policy configuration."""
        if self.default_retention_days < 1:
            raise AuditConfigError(
                "Default retention days must be at least 1",
                config_key="default_retention_days",
                config_value=self.default_retention_days,
            )

        if self.archive_after_days < 0:
            raise AuditConfigError(
                "Archive after days cannot be negative",
                config_key="archive_after_days",
                config_value=self.archive_after_days,
            )

        if self.cleanup_check_interval_hours < 1:
            raise AuditConfigError(
                "Cleanup check interval must be at least 1 hour",
                config_key="cleanup_check_interval_hours",
                config_value=self.cleanup_check_interval_hours,
            )


@dataclass
class SecurityConfig:
    """Security configuration for audit logging."""

    # Encryption settings
    encryption_enabled: bool = True
    encryption_type: EncryptionType = EncryptionType.AES256
    encryption_key_file: str | None = None
    key_rotation_days: int = 90

    # Digital signature settings
    digital_signatures_enabled: bool = True
    signature_algorithm: str = "RSA-SHA256"
    signing_key_file: str | None = None

    # Access control
    access_control_enabled: bool = True
    required_roles: list[str] = field(default_factory=lambda: ["audit_viewer"])
    admin_roles: list[str] = field(default_factory=lambda: ["audit_admin"])

    # Integrity verification
    integrity_checks_enabled: bool = True
    integrity_check_interval_hours: int = 1
    tamper_detection_enabled: bool = True

    # Network security
    tls_enabled: bool = True
    tls_min_version: str = "1.2"
    certificate_validation: bool = True

    def validate(self) -> None:
        """Validate security configuration."""
        if self.encryption_enabled and not self.encryption_key_file:
            raise AuditConfigError(
                "Encryption key file must be specified when encryption is enabled",
                config_key="encryption_key_file",
            )

        if self.digital_signatures_enabled and not self.signing_key_file:
            raise AuditConfigError(
                "Signing key file must be specified when digital signatures are enabled",
                config_key="signing_key_file",
            )

        if self.key_rotation_days < 1:
            raise AuditConfigError(
                "Key rotation days must be at least 1",
                config_key="key_rotation_days",
                config_value=self.key_rotation_days,
            )


@dataclass
class PerformanceConfig:
    """Performance configuration for audit logging."""

    # Threading and concurrency
    max_worker_threads: int = 4
    async_processing_enabled: bool = True
    batch_processing_enabled: bool = True
    batch_size: int = 100
    batch_timeout_seconds: int = 5

    # Queue settings
    event_queue_size: int = 10000
    priority_queue_enabled: bool = True

    # Caching settings
    cache_enabled: bool = True
    cache_size_mb: int = 100
    cache_ttl_seconds: int = 300

    # Buffer settings
    buffer_size_mb: int = 50
    flush_interval_seconds: int = 1
    force_flush_on_critical: bool = True

    # Connection pooling
    connection_pool_size: int = 10
    connection_timeout_seconds: int = 30

    # Performance monitoring
    metrics_enabled: bool = True
    performance_alerts_enabled: bool = True
    max_latency_ms: int = 1  # Alert if logging takes longer than 1ms
    max_throughput_events_per_second: int = 10000

    def validate(self) -> None:
        """Validate performance configuration."""
        if self.max_worker_threads < 1:
            raise AuditConfigError(
                "Max worker threads must be at least 1",
                config_key="max_worker_threads",
                config_value=self.max_worker_threads,
            )

        if self.batch_size < 1:
            raise AuditConfigError(
                "Batch size must be at least 1",
                config_key="batch_size",
                config_value=self.batch_size,
            )

        if self.event_queue_size < 100:
            raise AuditConfigError(
                "Event queue size must be at least 100",
                config_key="event_queue_size",
                config_value=self.event_queue_size,
            )


@dataclass
class ComplianceConfig:
    """Compliance-specific configuration."""

    # Enabled regulations
    enabled_regulations: list[ComplianceRegulation] = field(
        default_factory=lambda: [
            ComplianceRegulation.SOX,
            ComplianceRegulation.SEC,
            ComplianceRegulation.FINRA,
        ]
    )

    # Reporting settings
    automated_reporting_enabled: bool = True
    report_generation_schedule: str = "daily"  # "hourly", "daily", "weekly", "monthly"
    report_formats: list[str] = field(default_factory=lambda: ["json", "csv", "xml"])

    # Jurisdiction settings
    primary_jurisdiction: str = "US"
    additional_jurisdictions: list[str] = field(default_factory=list)

    # Regulatory reporting
    regulatory_reporting_enabled: bool = True
    regulatory_endpoints: dict[str, str] = field(default_factory=dict)

    # Data privacy
    pii_anonymization_enabled: bool = True
    data_minimization_enabled: bool = True
    consent_tracking_enabled: bool = True

    # Compliance checks
    real_time_compliance_checks: bool = True
    compliance_violation_alerts: bool = True

    def validate(self) -> None:
        """Validate compliance configuration."""
        if not self.enabled_regulations:
            raise AuditConfigError(
                "At least one compliance regulation must be enabled",
                config_key="enabled_regulations",
            )

        valid_schedules = ["hourly", "daily", "weekly", "monthly"]
        if self.report_generation_schedule not in valid_schedules:
            raise AuditConfigError(
                f"Report generation schedule must be one of: {valid_schedules}",
                config_key="report_generation_schedule",
                config_value=self.report_generation_schedule,
            )


@dataclass
class StorageConfig:
    """Storage configuration for different backends."""

    # Primary storage backend
    primary_backend: StorageBackend = StorageBackend.DATABASE

    # Backup storage backends
    backup_backends: list[StorageBackend] = field(default_factory=lambda: [StorageBackend.FILE])

    # File storage settings
    file_storage_path: str = "/var/log/audit"
    file_rotation_enabled: bool = True
    file_rotation_size_mb: int = 100
    file_rotation_count: int = 10

    # Database storage settings
    database_url: str | None = None
    database_table_name: str = "audit_logs"
    database_connection_pool_size: int = 10
    database_batch_insert_size: int = 1000

    # External storage settings
    external_storage_type: str | None = None  # "s3", "azure", "gcp"
    external_storage_config: dict[str, Any] = field(default_factory=dict)
    external_storage_retry_attempts: int = 3

    # Storage optimization
    compression_enabled: bool = True
    compression_level: int = 6
    deduplication_enabled: bool = True

    def validate(self) -> None:
        """Validate storage configuration."""
        if self.primary_backend == StorageBackend.DATABASE and not self.database_url:
            raise AuditConfigError(
                "Database URL must be specified when using database storage",
                config_key="database_url",
            )

        if self.file_rotation_size_mb < 1:
            raise AuditConfigError(
                "File rotation size must be at least 1 MB",
                config_key="file_rotation_size_mb",
                config_value=self.file_rotation_size_mb,
            )

        if not os.access(os.path.dirname(self.file_storage_path), os.W_OK):
            raise AuditConfigError(
                f"File storage path is not writable: {self.file_storage_path}",
                config_key="file_storage_path",
                config_value=self.file_storage_path,
            )


@dataclass
class AuditConfig:
    """
    Comprehensive audit logging configuration.

    Provides centralized configuration for all aspects of audit logging,
    including retention policies, security settings, performance tuning,
    and compliance requirements.
    """

    # System information
    system_name: str = "AI Trading System"
    system_version: str = "1.0.0"
    environment: str = "production"  # "development", "staging", "production"

    # Core audit settings
    audit_enabled: bool = True
    strict_validation: bool = True
    fail_on_storage_error: bool = True

    # Configuration components
    retention_policy: RetentionPolicy = field(default_factory=RetentionPolicy)
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    compliance_config: ComplianceConfig = field(default_factory=ComplianceConfig)
    storage_config: StorageConfig = field(default_factory=StorageConfig)

    # Monitoring and alerting
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    health_check_enabled: bool = True

    def __post_init__(self) -> None:
        """Post-initialization validation."""
        self.validate()

    @property
    def batch_size(self) -> int:
        """Get batch size for async processing."""
        return self.performance_config.batch_size

    @property
    def batch_timeout(self) -> int:
        """Get batch timeout for async processing."""
        return self.performance_config.batch_timeout_seconds

    def validate(self) -> None:
        """Validate entire audit configuration."""
        if not self.system_name:
            raise AuditConfigError("System name is required", config_key="system_name")

        if not self.system_version:
            raise AuditConfigError("System version is required", config_key="system_version")

        if self.environment not in ["development", "staging", "production"]:
            raise AuditConfigError(
                "Environment must be one of: development, staging, production",
                config_key="environment",
                config_value=self.environment,
            )

        # Validate component configurations
        self.retention_policy.validate()
        self.security_config.validate()
        self.performance_config.validate()
        self.compliance_config.validate()
        self.storage_config.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "system_name": self.system_name,
            "system_version": self.system_version,
            "environment": self.environment,
            "audit_enabled": self.audit_enabled,
            "strict_validation": self.strict_validation,
            "fail_on_storage_error": self.fail_on_storage_error,
            "monitoring_enabled": self.monitoring_enabled,
            "alerting_enabled": self.alerting_enabled,
            "health_check_enabled": self.health_check_enabled,
            "retention_policy": {
                "default_retention_days": self.retention_policy.default_retention_days,
                "critical_event_retention_days": self.retention_policy.critical_event_retention_days,
                "archive_after_days": self.retention_policy.archive_after_days,
                "auto_cleanup_enabled": self.retention_policy.auto_cleanup_enabled,
            },
            "security_config": {
                "encryption_enabled": self.security_config.encryption_enabled,
                "digital_signatures_enabled": self.security_config.digital_signatures_enabled,
                "access_control_enabled": self.security_config.access_control_enabled,
                "integrity_checks_enabled": self.security_config.integrity_checks_enabled,
            },
            "performance_config": {
                "async_processing_enabled": self.performance_config.async_processing_enabled,
                "batch_processing_enabled": self.performance_config.batch_processing_enabled,
                "batch_size": self.performance_config.batch_size,
                "max_worker_threads": self.performance_config.max_worker_threads,
            },
            "storage_config": {
                "primary_backend": self.storage_config.primary_backend.value,
                "backup_backends": [b.value for b in self.storage_config.backup_backends],
                "compression_enabled": self.storage_config.compression_enabled,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "AuditConfig":
        """Create configuration from dictionary."""
        # This is a simplified implementation - in production, you'd want
        # more sophisticated configuration loading with proper type conversion
        config = cls()

        # Update basic properties
        for key in [
            "system_name",
            "system_version",
            "environment",
            "audit_enabled",
            "strict_validation",
            "fail_on_storage_error",
        ]:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config

    @classmethod
    def from_file(cls, config_path: str) -> "AuditConfig":
        """Load configuration from JSON file."""
        try:
            with open(config_path) as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            raise AuditConfigError(
                f"Configuration file not found: {config_path}",
                config_key="config_path",
                config_value=config_path,
            )
        except json.JSONDecodeError as e:
            raise AuditConfigError(
                f"Invalid JSON in configuration file: {e}", config_key="config_format"
            )

    @classmethod
    def from_env(cls) -> "AuditConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Load from environment variables with AUDIT_ prefix
        system_name = os.getenv("AUDIT_SYSTEM_NAME")
        if system_name:
            config.system_name = system_name

        environment = os.getenv("AUDIT_ENVIRONMENT")
        if environment:
            config.environment = environment

        audit_enabled = os.getenv("AUDIT_ENABLED")
        if audit_enabled:
            config.audit_enabled = audit_enabled.lower() == "true"

        # Storage configuration
        if os.getenv("AUDIT_DATABASE_URL"):
            config.storage_config.database_url = os.getenv("AUDIT_DATABASE_URL")

        storage_path = os.getenv("AUDIT_STORAGE_PATH")
        if storage_path:
            config.storage_config.file_storage_path = storage_path

        # Security configuration
        encryption_enabled = os.getenv("AUDIT_ENCRYPTION_ENABLED")
        if encryption_enabled:
            config.security_config.encryption_enabled = encryption_enabled.lower() == "true"

        if os.getenv("AUDIT_ENCRYPTION_KEY_FILE"):
            config.security_config.encryption_key_file = os.getenv("AUDIT_ENCRYPTION_KEY_FILE")

        return config

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        try:
            with open(config_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        except Exception as e:
            raise AuditConfigError(
                f"Failed to save configuration to file: {e}",
                config_key="config_path",
                config_value=config_path,
            )
