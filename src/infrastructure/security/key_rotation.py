"""
Automated key rotation service for the AI Trading System.

Provides scheduled key rotation, monitoring, and emergency rotation capabilities
for maintaining cryptographic security in production environments.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .key_management import RSAKeyManager

logger = logging.getLogger(__name__)


class RotationStatus(Enum):
    """Status of key rotation operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RotationTrigger(Enum):
    """Triggers for key rotation."""

    SCHEDULED = "scheduled"
    MANUAL = "manual"
    EMERGENCY = "emergency"
    EXPIRY_WARNING = "expiry_warning"
    SECURITY_INCIDENT = "security_incident"


@dataclass
class RotationJob:
    """Represents a key rotation job."""

    job_id: str
    key_id: str
    trigger: RotationTrigger
    scheduled_at: datetime
    status: RotationStatus = RotationStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    new_key_id: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_due(self) -> bool:
        """Check if rotation job is due."""
        return datetime.utcnow() >= self.scheduled_at

    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "key_id": self.key_id,
            "trigger": self.trigger.value,
            "scheduled_at": self.scheduled_at.isoformat(),
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "new_key_id": self.new_key_id,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }


class KeyRotationService:
    """
    Automated key rotation service.

    Features:
    - Scheduled key rotation based on expiry dates
    - Manual rotation triggers
    - Emergency rotation capabilities
    - Rotation monitoring and alerting
    - Retry logic for failed rotations
    - Integration hooks for application updates
    """

    def __init__(
        self,
        key_manager: RSAKeyManager,
        rotation_schedule_hours: int = 24,  # Check every 24 hours
        emergency_rotation_enabled: bool = True,
        max_concurrent_rotations: int = 3,
        notification_callbacks: list[Callable[[str, dict[str, Any]], None]] | None = None,
    ):
        self.key_manager = key_manager
        self.rotation_schedule_hours = rotation_schedule_hours
        self.emergency_rotation_enabled = emergency_rotation_enabled
        self.max_concurrent_rotations = max_concurrent_rotations
        self.notification_callbacks = notification_callbacks or []

        # Job management
        self.rotation_jobs: dict[str, RotationJob] = {}
        self.active_rotations: dict[str, RotationJob] = {}

        # Service control
        self._running = False
        self._rotation_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # Statistics
        self._stats: dict[str, Any] = {
            "total_rotations": 0,
            "successful_rotations": 0,
            "failed_rotations": 0,
            "emergency_rotations": 0,
            "last_rotation": None,
            "service_start_time": None,
        }

        logger.info("Key Rotation Service initialized")

    def start(self) -> None:
        """Start the rotation service."""
        with self._lock:
            if self._running:
                logger.warning("Key rotation service already running")
                return

            self._running = True
            self._stop_event.clear()
            self._stats["service_start_time"] = datetime.utcnow()

            # Start rotation thread
            self._rotation_thread = threading.Thread(
                target=self._rotation_worker, name="KeyRotationService", daemon=True
            )
            self._rotation_thread.start()

            logger.info("Key rotation service started")

    def stop(self) -> None:
        """Stop the rotation service."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._stop_event.set()

            # Wait for rotation thread to finish
            if self._rotation_thread and self._rotation_thread.is_alive():
                self._rotation_thread.join(timeout=30)
                if self._rotation_thread.is_alive():
                    logger.warning("Rotation thread did not stop gracefully")

            logger.info("Key rotation service stopped")

    def schedule_rotation(
        self,
        key_id: str,
        trigger: RotationTrigger,
        scheduled_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Schedule a key rotation.

        Returns:
            Job ID for the scheduled rotation
        """
        if not scheduled_at:
            scheduled_at = datetime.utcnow()

        job_id = f"rotation_{key_id}_{int(time.time())}"

        job = RotationJob(
            job_id=job_id,
            key_id=key_id,
            trigger=trigger,
            scheduled_at=scheduled_at,
            metadata=metadata or {},
        )

        with self._lock:
            self.rotation_jobs[job_id] = job

        logger.info(f"Scheduled key rotation: {key_id} at {scheduled_at}")

        # Send notification
        self._notify(
            "rotation_scheduled",
            {
                "job_id": job_id,
                "key_id": key_id,
                "trigger": trigger.value,
                "scheduled_at": scheduled_at.isoformat(),
            },
        )

        return job_id

    def emergency_rotation(self, key_id: str, reason: str) -> str:
        """
        Trigger emergency key rotation.

        Returns:
            Job ID for emergency rotation
        """
        if not self.emergency_rotation_enabled:
            raise RuntimeError("Emergency rotation is disabled")

        logger.warning(f"Emergency key rotation triggered: {key_id} - {reason}")

        job_id = self.schedule_rotation(
            key_id=key_id,
            trigger=RotationTrigger.EMERGENCY,
            scheduled_at=datetime.utcnow(),  # Immediate
            metadata={"emergency_reason": reason},
        )

        # Update stats
        self._stats["emergency_rotations"] = (self._stats.get("emergency_rotations") or 0) + 1

        # Send urgent notification
        self._notify(
            "emergency_rotation",
            {
                "job_id": job_id,
                "key_id": key_id,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        return job_id

    def cancel_rotation(self, job_id: str) -> bool:
        """Cancel a pending rotation job."""
        with self._lock:
            if job_id not in self.rotation_jobs:
                return False

            job = self.rotation_jobs[job_id]

            # Can only cancel pending jobs
            if job.status != RotationStatus.PENDING:
                return False

            job.status = RotationStatus.CANCELLED

            logger.info(f"Cancelled rotation job: {job_id}")

            self._notify(
                "rotation_cancelled",
                {
                    "job_id": job_id,
                    "key_id": job.key_id,
                },
            )

            return True

    def get_rotation_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of a rotation job."""
        with self._lock:
            if job_id not in self.rotation_jobs:
                return None

            return self.rotation_jobs[job_id].to_dict()

    def list_pending_rotations(self) -> list[dict[str, Any]]:
        """List all pending rotation jobs."""
        with self._lock:
            pending_jobs = []
            for job in self.rotation_jobs.values():
                if job.status == RotationStatus.PENDING:
                    pending_jobs.append(job.to_dict())

            return sorted(pending_jobs, key=lambda j: j["scheduled_at"])

    def _rotation_worker(self) -> None:
        """Main rotation worker thread."""
        logger.info("Rotation worker thread started")

        while self._running and not self._stop_event.is_set():
            try:
                # Check for keys needing rotation
                self._check_expiring_keys()

                # Process pending rotation jobs
                self._process_rotation_jobs()

                # Cleanup completed jobs
                self._cleanup_old_jobs()

                # Wait before next cycle
                self._stop_event.wait(timeout=self.rotation_schedule_hours * 3600)

            except Exception as e:
                logger.error(f"Error in rotation worker: {e}")
                # Wait a bit before retrying
                self._stop_event.wait(timeout=300)  # 5 minutes

        logger.info("Rotation worker thread stopped")

    def _check_expiring_keys(self) -> None:
        """Check for keys that need rotation due to expiry."""
        try:
            keys_needing_rotation = self.key_manager.get_keys_needing_rotation()

            for metadata in keys_needing_rotation:
                # Check if rotation already scheduled
                existing_job = None
                with self._lock:
                    for job in self.rotation_jobs.values():
                        if job.key_id == metadata.key_id and job.status in [
                            RotationStatus.PENDING,
                            RotationStatus.IN_PROGRESS,
                        ]:
                            existing_job = job
                            break

                if not existing_job:
                    # Schedule rotation
                    days_until_expiry = metadata.days_until_expiry()
                    if days_until_expiry is not None and days_until_expiry <= 7:
                        # Urgent - schedule immediately
                        scheduled_at = datetime.utcnow()
                    else:
                        # Schedule for optimal time (off-hours)
                        scheduled_at = self._get_optimal_rotation_time()

                    self.schedule_rotation(
                        key_id=metadata.key_id,
                        trigger=RotationTrigger.EXPIRY_WARNING,
                        scheduled_at=scheduled_at,
                        metadata={
                            "days_until_expiry": days_until_expiry,
                            "usage": metadata.usage.value,
                        },
                    )

        except Exception as e:
            logger.error(f"Error checking expiring keys: {e}")

    def _process_rotation_jobs(self) -> None:
        """Process pending rotation jobs."""
        with self._lock:
            # Get due jobs
            due_jobs = []
            for job in self.rotation_jobs.values():
                if (
                    job.status == RotationStatus.PENDING
                    and job.is_due()
                    and len(self.active_rotations) < self.max_concurrent_rotations
                ):
                    due_jobs.append(job)

            # Sort by priority (emergency first)
            due_jobs.sort(
                key=lambda j: (0 if j.trigger == RotationTrigger.EMERGENCY else 1, j.scheduled_at)
            )

        # Process due jobs
        for job in due_jobs:
            if len(self.active_rotations) >= self.max_concurrent_rotations:
                break

            self._execute_rotation(job)

    def _execute_rotation(self, job: RotationJob) -> None:
        """Execute a rotation job."""
        with self._lock:
            if job.job_id in self.active_rotations:
                return

            job.status = RotationStatus.IN_PROGRESS
            job.started_at = datetime.utcnow()
            self.active_rotations[job.job_id] = job

        logger.info(f"Starting key rotation: {job.key_id} (job: {job.job_id})")

        def rotation_task() -> None:
            try:
                # Perform the rotation
                new_key_id = self.key_manager.rotate_key(job.key_id)

                # Update job status
                with self._lock:
                    job.status = RotationStatus.COMPLETED
                    job.completed_at = datetime.utcnow()
                    job.new_key_id = new_key_id

                    # Remove from active rotations
                    if job.job_id in self.active_rotations:
                        del self.active_rotations[job.job_id]

                # Update stats
                self._stats["total_rotations"] = (self._stats.get("total_rotations") or 0) + 1
                self._stats["successful_rotations"] = (
                    self._stats.get("successful_rotations") or 0
                ) + 1
                self._stats["last_rotation"] = datetime.utcnow()

                logger.info(f"Completed key rotation: {job.key_id} -> {new_key_id}")

                # Send notification
                self._notify(
                    "rotation_completed",
                    {
                        "job_id": job.job_id,
                        "key_id": job.key_id,
                        "new_key_id": new_key_id,
                        "trigger": job.trigger.value,
                        "duration": (
                            (job.completed_at - job.started_at).total_seconds()
                            if job.completed_at and job.started_at
                            else 0
                        ),
                    },
                )

            except Exception as e:
                logger.error(f"Key rotation failed: {job.key_id} - {e}")

                with self._lock:
                    job.status = RotationStatus.FAILED
                    job.error_message = str(e)
                    job.retry_count += 1

                    # Remove from active rotations
                    if job.job_id in self.active_rotations:
                        del self.active_rotations[job.job_id]

                # Update stats
                self._stats["total_rotations"] = (self._stats.get("total_rotations") or 0) + 1
                self._stats["failed_rotations"] = (self._stats.get("failed_rotations") or 0) + 1

                # Schedule retry if possible
                if job.can_retry() and job.trigger != RotationTrigger.EMERGENCY:
                    retry_delay = min(300 * (2**job.retry_count), 3600)  # Exponential backoff
                    retry_time = datetime.utcnow() + timedelta(seconds=retry_delay)

                    retry_job_id = self.schedule_rotation(
                        key_id=job.key_id,
                        trigger=job.trigger,
                        scheduled_at=retry_time,
                        metadata={"retry_of": job.job_id, "retry_count": job.retry_count},
                    )

                    logger.info(f"Scheduled retry for failed rotation: {retry_job_id}")

                # Send error notification
                self._notify(
                    "rotation_failed",
                    {
                        "job_id": job.job_id,
                        "key_id": job.key_id,
                        "error": str(e),
                        "retry_count": job.retry_count,
                        "can_retry": job.can_retry(),
                    },
                )

        # Run rotation in separate thread to avoid blocking
        rotation_thread = threading.Thread(
            target=rotation_task, name=f"Rotation-{job.job_id}", daemon=True
        )
        rotation_thread.start()

    def _cleanup_old_jobs(self) -> None:
        """Clean up old completed/failed jobs."""
        cutoff_time = datetime.utcnow() - timedelta(days=7)  # Keep for 7 days

        with self._lock:
            jobs_to_remove = []
            for job_id, job in self.rotation_jobs.items():
                if (
                    job.status
                    in [RotationStatus.COMPLETED, RotationStatus.FAILED, RotationStatus.CANCELLED]
                    and (job.completed_at or job.started_at or job.scheduled_at) < cutoff_time
                ):
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self.rotation_jobs[job_id]

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old rotation jobs")

    def _get_optimal_rotation_time(self) -> datetime:
        """Get optimal time for key rotation (off-hours)."""
        now = datetime.utcnow()

        # Schedule for 2 AM UTC (typical off-hours)
        optimal_time = now.replace(hour=2, minute=0, second=0, microsecond=0)

        # If 2 AM today has passed, schedule for tomorrow
        if optimal_time <= now:
            optimal_time += timedelta(days=1)

        return optimal_time

    def _notify(self, event_type: str, data: dict[str, Any]) -> None:
        """Send notification to registered callbacks."""
        for callback in self.notification_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get rotation service statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_rotations": len(self.active_rotations),
                "pending_jobs": len(
                    [j for j in self.rotation_jobs.values() if j.status == RotationStatus.PENDING]
                ),
                "is_running": self._running,
            }

    def health_check(self) -> dict[str, Any]:
        """Perform health check on rotation service."""
        health: dict[str, Any] = {
            "healthy": True,
            "service_running": self._running,
            "checks": {},
            "warnings": [],
            "errors": [],
        }

        try:
            with self._lock:
                # Check if service is running
                if not self._running:
                    warnings = health["warnings"]
                    if isinstance(warnings, list):
                        warnings.append("Rotation service is not running")

                # Check for stuck rotations
                stuck_rotations = []
                for job in self.active_rotations.values():
                    if job.started_at and datetime.utcnow() - job.started_at > timedelta(hours=1):
                        stuck_rotations.append(job.job_id)

                if stuck_rotations:
                    warnings = health["warnings"]
                    if isinstance(warnings, list):
                        warnings.extend([f"Stuck rotation: {job_id}" for job_id in stuck_rotations])

                # Check for overdue rotations
                overdue_jobs = []
                for job in self.rotation_jobs.values():
                    if (
                        job.status == RotationStatus.PENDING
                        and job.scheduled_at < datetime.utcnow() - timedelta(hours=6)
                    ):
                        overdue_jobs.append(job.job_id)

                if overdue_jobs:
                    warnings = health["warnings"]
                    if isinstance(warnings, list):
                        warnings.extend([f"Overdue rotation: {job_id}" for job_id in overdue_jobs])

                checks = health["checks"]
                if isinstance(checks, dict):
                    checks["active_rotations"] = len(self.active_rotations)
                    checks["pending_jobs"] = len(
                        [
                            j
                            for j in self.rotation_jobs.values()
                            if j.status == RotationStatus.PENDING
                        ]
                    )
                    checks["stuck_rotations"] = len(stuck_rotations)
                    checks["overdue_jobs"] = len(overdue_jobs)

        except Exception as e:
            health["healthy"] = False
            errors = health["errors"]
            if isinstance(errors, list):
                errors.append(f"Health check failed: {e}")

        return health


def create_production_rotation_service(
    key_manager: RSAKeyManager,
    notification_callbacks: list[Callable[[str, dict[str, Any]], None]] | None = None,
) -> KeyRotationService:
    """Create a production-ready key rotation service."""
    return KeyRotationService(
        key_manager=key_manager,
        rotation_schedule_hours=24,  # Check daily
        emergency_rotation_enabled=True,
        max_concurrent_rotations=2,  # Conservative limit
        notification_callbacks=notification_callbacks,
    )
