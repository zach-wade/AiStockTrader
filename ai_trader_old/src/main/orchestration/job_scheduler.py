"""
Job Scheduler for AI Trading System

This module provides enterprise-grade job scheduling with dependency management,
error recovery, market hours awareness, and resource limit enforcement.

Originally located in scripts/scheduler/, moved to proper location in orchestration module.
"""

# Standard library imports
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

# Third-party imports
import psutil
import pytz
import schedule
import yaml

# Local imports
from main.utils.core import get_logger


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class JobExecution:
    """Represents a job execution instance."""

    job_id: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    status: JobStatus = JobStatus.PENDING
    return_code: int | None = None
    output: str = ""
    error: str = ""
    pid: int | None = None
    attempt: int = 1
    resource_usage: dict[str, Any] = field(default_factory=dict)


class JobScheduler:
    """Master job scheduler for AI Trading System."""

    def __init__(self, config_path: str = None):
        """Initialize the job scheduler."""
        # Default to scripts location for backward compatibility
        if config_path is None:
            scripts_path = Path(__file__).parent.parent.parent.parent / "scripts" / "scheduler"
            config_path = str(scripts_path / "job_definitions.yaml")

        self.config_path = config_path

        # Load configuration
        self.config = self._load_config()
        self.global_config = self.config.get("global", {})
        self.jobs_config = self.config.get("jobs", {})
        self.market_schedule = self.config.get("market_schedule", {})
        self.holidays = set(self.config.get("holidays", []))

        # Initialize timezone
        self.timezone = pytz.timezone(self.global_config.get("timezone", "US/Eastern"))

        # Setup logging
        self.logger = get_logger(__name__)

        # Job tracking
        self.running_jobs: dict[str, JobExecution] = {}
        self.job_history: list[JobExecution] = []
        self.dependency_graph: dict[str, set[str]] = {}

        # Resource management
        self.max_concurrent_jobs = self.global_config.get("max_concurrent_jobs", 5)
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_jobs)

        # Monitoring integration (optional)
        self.health_reporter = None
        try:
            # Local imports
            from main.monitoring.health_reporter import get_health_reporter

            self.health_reporter = get_health_reporter()
        except ImportError:
            pass

        # Build dependency graph
        self._build_dependency_graph()

        self.logger.info("Job Scheduler initialized successfully")
        self.logger.info(f"Loaded {len(self.jobs_config)} job definitions")

    def _load_config(self) -> dict[str, Any]:
        """Load job configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load job configuration: {e}")

    def _build_dependency_graph(self):
        """Build job dependency graph for execution ordering."""
        for job_id, job_config in self.jobs_config.items():
            dependencies = job_config.get("dependencies", [])
            self.dependency_graph[job_id] = set(dependencies)

    def _is_trading_day(self, date: datetime = None) -> bool:
        """Check if given date is a trading day."""
        if date is None:
            date = datetime.now(self.timezone)

        # Check if weekend
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if holiday
        date_str = date.strftime("%Y-%m-%d")
        if date_str in self.holidays:
            return False

        return True

    def _is_market_hours(self, dt: datetime = None) -> bool:
        """Check if current time is during market hours."""
        if dt is None:
            dt = datetime.now(self.timezone)

        if not self._is_trading_day(dt):
            return False

        market_open = self.market_schedule.get("market_open", "09:30")
        market_close = self.market_schedule.get("market_close", "16:00")

        current_time = dt.strftime("%H:%M")
        return market_open <= current_time <= market_close

    def _should_run_job(self, job_id: str, job_config: dict[str, Any]) -> bool:
        """Determine if a job should run based on current conditions."""
        # Check if job is enabled
        if not job_config.get("enabled", True):
            return False

        # Check if manual trigger only
        if job_config.get("manual_trigger", False):
            return False

        # Check trading day requirement
        if not self._is_trading_day():
            # Only allow jobs that don't require trading days
            if job_config.get("require_trading_day", True):
                return False

        # Check market hours requirement
        if job_config.get("run_during_market_hours", False):
            if not self._is_market_hours():
                return False

        return True

    def _check_dependencies(self, job_id: str) -> bool:
        """Check if all job dependencies are satisfied."""
        dependencies = self.dependency_graph.get(job_id, set())

        for dep_job_id in dependencies:
            # Check if dependency completed successfully recently
            recent_executions = [
                ex
                for ex in self.job_history
                if ex.job_id == dep_job_id
                and ex.start_time
                and ex.start_time.date() == datetime.now().date()
                and ex.status == JobStatus.COMPLETED
            ]

            if not recent_executions:
                self.logger.debug(f"Job {job_id} waiting for dependency {dep_job_id}")
                return False

        return True

    def _check_resource_limits(self, job_config: dict[str, Any]) -> bool:
        """Check if resources are available to run the job."""
        # Check concurrent job limit
        if len(self.running_jobs) >= self.max_concurrent_jobs:
            # Allow critical jobs to override
            if job_config.get("priority") != "critical":
                return False

        # Check memory availability
        memory_limit_gb = self.global_config.get("max_memory_per_job_gb", 8)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        if available_memory_gb < memory_limit_gb:
            if job_config.get("priority") != "critical":
                return False

        # Check CPU availability
        cpu_limit_percent = self.global_config.get("max_cpu_per_job_percent", 50)
        current_cpu_percent = psutil.cpu_percent(interval=1)

        if current_cpu_percent > (100 - cpu_limit_percent):
            if job_config.get("priority") != "critical":
                return False

        return True

    def _execute_job(self, job_id: str, job_config: dict[str, Any]) -> JobExecution:
        """Execute a single job."""
        execution = JobExecution(job_id=job_id)
        execution.start_time = datetime.now(self.timezone)
        execution.status = JobStatus.RUNNING

        self.logger.info(f"Starting job: {job_id}")

        try:
            # Build command
            script_path = job_config["script"]
            args = job_config.get("args", [])
            timeout_minutes = job_config.get(
                "timeout_minutes", self.global_config.get("max_job_runtime_minutes", 120)
            )

            # Prepare environment
            env = os.environ.copy()
            env["AI_TRADER_JOB_ID"] = job_id
            env["AI_TRADER_JOB_START"] = execution.start_time.isoformat()

            # Build full command
            base_dir = self.global_config.get("base_directory", os.getcwd())
            python_path = self.global_config.get("python_path", sys.executable)

            if script_path.endswith(".py"):
                cmd = [python_path, os.path.join(base_dir, script_path)] + args
            else:
                cmd = [os.path.join(base_dir, script_path)] + args

            # Execute with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=base_dir,
                text=True,
            )

            execution.pid = process.pid

            try:
                stdout, stderr = process.communicate(timeout=timeout_minutes * 60)
                execution.return_code = process.returncode
                execution.output = stdout
                execution.error = stderr

                if process.returncode == 0:
                    execution.status = JobStatus.COMPLETED
                    self.logger.info(f"Job {job_id} completed successfully")
                else:
                    execution.status = JobStatus.FAILED
                    self.logger.error(f"Job {job_id} failed with return code {process.returncode}")

            except subprocess.TimeoutExpired:
                process.kill()
                execution.status = JobStatus.TIMEOUT
                execution.error = f"Job timed out after {timeout_minutes} minutes"
                self.logger.error(f"Job {job_id} timed out")

        except Exception as e:
            execution.status = JobStatus.FAILED
            execution.error = str(e)
            self.logger.error(f"Job {job_id} failed with exception: {e}")

        finally:
            execution.end_time = datetime.now(self.timezone)

            # Capture resource usage
            if execution.pid:
                try:
                    proc = psutil.Process(execution.pid)
                    execution.resource_usage = {
                        "cpu_percent": proc.cpu_percent(),
                        "memory_mb": proc.memory_info().rss / (1024 * 1024),
                        "runtime_seconds": (
                            execution.end_time - execution.start_time
                        ).total_seconds(),
                    }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        return execution

    def _retry_job(
        self, job_id: str, job_config: dict[str, Any], failed_execution: JobExecution
    ) -> JobExecution | None:
        """Retry a failed job if retry attempts are configured."""
        max_retries = job_config.get("retry_attempts", 0)

        if failed_execution.attempt < max_retries:
            self.logger.info(f"Retrying job {job_id}, attempt {failed_execution.attempt + 1}")

            # Wait before retry (exponential backoff)
            wait_seconds = min(300, 30 * (2 ** (failed_execution.attempt - 1)))
            time.sleep(wait_seconds)

            # Create new execution
            retry_execution = self._execute_job(job_id, job_config)
            retry_execution.attempt = failed_execution.attempt + 1

            return retry_execution

        return None

    def _send_job_alert(self, job_id: str, job_config: dict[str, Any], execution: JobExecution):
        """Send alerts for job failures or timeouts."""
        alert_config = job_config.get("alerts", {})

        should_alert = (
            execution.status == JobStatus.FAILED and alert_config.get("on_failure", False)
        ) or (execution.status == JobStatus.TIMEOUT and alert_config.get("on_timeout", False))

        if should_alert and self.health_reporter:
            priority = job_config.get("priority", "medium")

            alert_message = (
                f"Job Alert: {job_id}\n"
                f"Status: {execution.status.value}\n"
                f"Duration: {(execution.end_time - execution.start_time).total_seconds():.1f}s\n"
                f"Error: {execution.error[:500] if execution.error else 'None'}"
            )

            try:
                self.health_reporter.send_alert(
                    title=f"Job {execution.status.value.upper()}: {job_id}",
                    message=alert_message,
                    severity=priority,
                    component="job_scheduler",
                )
            except Exception as e:
                self.logger.error(f"Failed to send alert for job {job_id}: {e}")

    def run_job(self, job_id: str) -> bool:
        """Run a single job with all checks and error handling."""
        if job_id not in self.jobs_config:
            self.logger.error(f"Job {job_id} not found in configuration")
            return False

        job_config = self.jobs_config[job_id]

        # Pre-execution checks
        if not self._should_run_job(job_id, job_config):
            self.logger.debug(f"Job {job_id} skipped - conditions not met")
            return False

        if not self._check_dependencies(job_id):
            self.logger.debug(f"Job {job_id} skipped - dependencies not satisfied")
            return False

        if not self._check_resource_limits(job_config):
            self.logger.debug(f"Job {job_id} skipped - resource limits exceeded")
            return False

        # Execute job
        execution = self._execute_job(job_id, job_config)

        # Handle retries
        if execution.status == JobStatus.FAILED:
            retry_execution = self._retry_job(job_id, job_config, execution)
            if retry_execution:
                execution = retry_execution

        # Store execution record
        if job_id in self.running_jobs:
            del self.running_jobs[job_id]

        self.job_history.append(execution)

        # Send alerts if configured
        self._send_job_alert(job_id, job_config, execution)

        # Log job completion
        runtime = (execution.end_time - execution.start_time).total_seconds()
        self.logger.info(
            f"Job {job_id} finished: {execution.status.value} "
            f"(attempt {execution.attempt}, {runtime:.1f}s)"
        )

        return execution.status == JobStatus.COMPLETED

    def setup_schedule(self):
        """Setup all scheduled jobs using the schedule library."""
        for job_id, job_config in self.jobs_config.items():
            schedule_spec = job_config.get("schedule")

            if not schedule_spec or schedule_spec == "manual":
                continue

            # Parse cron-like schedule (simplified)
            try:
                self._add_scheduled_job(job_id, schedule_spec)
                self.logger.info(f"Scheduled job {job_id}: {schedule_spec}")
            except Exception as e:
                self.logger.error(f"Failed to schedule job {job_id}: {e}")

    def _add_scheduled_job(self, job_id: str, schedule_spec: str):
        """Add a job to the schedule based on cron-like specification."""
        # This is a simplified scheduler - in production, consider using APScheduler
        # For now, handle basic patterns

        if schedule_spec.startswith("*/"):
            # Handle interval patterns like "*/5 9-16 * * 1-5"
            parts = schedule_spec.split()
            if len(parts) >= 2:
                minute_spec = parts[0]
                hour_spec = parts[1]

                if minute_spec.startswith("*/"):
                    interval = int(minute_spec[2:])

                    if "-" in hour_spec:
                        start_hour, end_hour = map(int, hour_spec.split("-"))

                        # Schedule every N minutes during specified hours
                        for hour in range(start_hour, end_hour + 1):
                            for minute in range(0, 60, interval):
                                time_str = f"{hour:02d}:{minute:02d}"
                                schedule.every().day.at(time_str).do(self.run_job, job_id)

        elif " " in schedule_spec:
            # Handle specific time patterns like "0 4 * * 1-5"
            parts = schedule_spec.split()
            if len(parts) >= 2:
                minute = parts[0]
                hour = parts[1]

                if minute.isdigit() and hour.isdigit():
                    time_str = f"{hour.zfill(2)}:{minute.zfill(2)}"
                    schedule.every().day.at(time_str).do(self.run_job, job_id)

    def run_scheduler(self):
        """Main scheduler loop."""
        self.logger.info("Starting job scheduler...")

        # Setup scheduled jobs
        self.setup_schedule()

        try:
            while True:
                # Run pending scheduled jobs
                schedule.run_pending()

                # Health reporting
                if self.health_reporter:
                    self._report_scheduler_health()

                # Sleep for a minute
                time.sleep(60)

        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
        finally:
            self.cleanup()

    def _report_scheduler_health(self):
        """Report scheduler health to monitoring system."""
        try:
            running_count = len(self.running_jobs)
            recent_failures = len(
                [
                    ex
                    for ex in self.job_history[-100:]  # Last 100 executions
                    if ex.status == JobStatus.FAILED
                    and ex.start_time
                    and (datetime.now(self.timezone) - ex.start_time) < timedelta(hours=1)
                ]
            )

            metrics = {
                "running_jobs": running_count,
                "recent_failures": recent_failures,
                "total_jobs_configured": len(self.jobs_config),
                "scheduler_uptime": time.time(),  # Will be calculated differently in practice
            }

            self.health_reporter.update_custom_metrics("job_scheduler", metrics)

        except Exception as e:
            self.logger.error(f"Failed to report scheduler health: {e}")

    def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up job scheduler...")

        # Wait for running jobs to complete (with timeout)
        if self.running_jobs:
            self.logger.info(f"Waiting for {len(self.running_jobs)} running jobs to complete...")

            for _ in range(30):  # Wait up to 30 seconds
                if not self.running_jobs:
                    break
                time.sleep(1)

        # Shutdown executor
        self.executor.shutdown(wait=True)

        self.logger.info("Job scheduler cleanup complete")

    def get_job_status(self, job_id: str = None) -> dict[str, Any]:
        """Get status of jobs."""
        if job_id:
            if job_id in self.running_jobs:
                return {"status": "running", "execution": self.running_jobs[job_id]}

            recent_executions = [ex for ex in self.job_history if ex.job_id == job_id]

            if recent_executions:
                latest = max(recent_executions, key=lambda x: x.start_time or datetime.min)
                return {"status": latest.status.value, "execution": latest}

            return {"status": "never_run"}

        else:
            return {
                "running_jobs": len(self.running_jobs),
                "total_jobs": len(self.jobs_config),
                "recent_executions": len(
                    [
                        ex
                        for ex in self.job_history
                        if ex.start_time
                        and (datetime.now(self.timezone) - ex.start_time) < timedelta(hours=24)
                    ]
                ),
            }
