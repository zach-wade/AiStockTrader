"""
Error and exception logging
Created: 2025-06-16
"""

"""Comprehensive error logging and tracking system."""
# Standard library imports
import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Any

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error categories for classification."""

    DATA = "data"
    TRADING = "trading"
    RISK = "risk"
    SYSTEM = "system"
    NETWORK = "network"
    BROKER = "broker"
    STRATEGY = "strategy"
    DATABASE = "database"


@dataclass
class ErrorEvent:
    """Error event with full context."""

    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    traceback: str
    timestamp: datetime
    module: str
    function: str
    line_number: int
    context: dict[str, Any]
    user_impact: str
    recovery_action: str


class ErrorLogger:
    """Advanced error logging with analysis and alerting."""

    def __init__(self, config: dict[str, Any], alert_system=None):
        self.config = config
        self.alert_system = alert_system

        # Logging configuration
        self.log_dir = Path(config.get("log_dir", "logs/errors"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Error storage
        self.error_history: deque = deque(maxlen=10000)
        self.error_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.error_patterns: dict[str, list[ErrorEvent]] = defaultdict(list)

        # Rate limiting
        self.rate_limits = {
            ErrorSeverity.DEBUG: 100,  # per minute
            ErrorSeverity.INFO: 50,
            ErrorSeverity.WARNING: 20,
            ErrorSeverity.ERROR: 10,
            ErrorSeverity.CRITICAL: 5,
            ErrorSeverity.FATAL: 1,
        }
        self.rate_counters: dict[ErrorSeverity, deque] = {
            severity: deque(maxlen=limit) for severity, limit in self.rate_limits.items()
        }

        # Error handlers
        self.error_handlers: dict[ErrorCategory, list[Callable]] = defaultdict(list)

        # Setup logging
        self._setup_logging()

        # Metrics
        self.metrics = {
            "total_errors": 0,
            "errors_by_severity": defaultdict(int),
            "errors_by_category": defaultdict(int),
            "error_rate": deque(maxlen=60),  # Last 60 minutes
            "recovery_success_rate": 0.0,
        }

        # Background tasks
        self._running = False
        self._tasks = []

    def _setup_logging(self):
        """Setup specialized error logging."""
        # Main error log
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log", maxBytes=50 * 1024 * 1024, backupCount=10  # 50MB
        )
        error_handler.setLevel(logging.WARNING)
        error_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        error_handler.setFormatter(error_formatter)

        # Critical error log
        critical_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "critical_errors.log", maxBytes=10 * 1024 * 1024, backupCount=20  # 10MB
        )
        critical_handler.setLevel(logging.CRITICAL)
        critical_handler.setFormatter(error_formatter)

        # JSON structured log for analysis
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.json", maxBytes=50 * 1024 * 1024, backupCount=5
        )
        json_handler.setLevel(logging.WARNING)

        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(error_handler)
        root_logger.addHandler(critical_handler)
        root_logger.addHandler(json_handler)

        # Set custom exception handler
        sys.excepthook = self._exception_handler

    def _exception_handler(self, exc_type, exc_value, exc_traceback):
        """Global exception handler."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the exception
        self.log_exception(
            exc_value,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            context={"uncaught": True},
        )

    async def start(self):
        """Start error logger background tasks."""
        self._running = True
        self._tasks = [
            asyncio.create_task(self._monitor_error_patterns()),
            asyncio.create_task(self._cleanup_old_logs()),
            asyncio.create_task(self._analyze_error_trends()),
        ]

    async def stop(self):
        """Stop error logger."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def log_error(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: dict[str, Any] | None = None,
        user_impact: str = "Unknown",
        recovery_action: str = "Manual intervention required",
    ) -> str:
        """Log an error with full context."""
        # Check rate limits
        if not self._check_rate_limit(severity):
            return None

        # Get caller information
        frame = sys._getframe(1)
        module = frame.f_globals.get("__name__", "unknown")
        function = frame.f_code.co_name
        line_number = frame.f_lineno

        # Create error event
        error_id = f"ERR_{datetime.now().timestamp():.0f}"

        error_event = ErrorEvent(
            error_id=error_id,
            severity=severity,
            category=category,
            message=message,
            exception_type="",
            traceback="",
            timestamp=datetime.now(),
            module=module,
            function=function,
            line_number=line_number,
            context=context or {},
            user_impact=user_impact,
            recovery_action=recovery_action,
        )

        # Process error
        self._process_error(error_event)

        return error_id

    def log_exception(
        self,
        exception: Exception,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: dict[str, Any] | None = None,
        user_impact: str = "Unknown",
        recovery_action: str = "Manual intervention required",
    ) -> str:
        """Log an exception with full traceback."""
        # Check rate limits
        if not self._check_rate_limit(severity):
            return None

        # Get caller information
        frame = sys._getframe(1)
        module = frame.f_globals.get("__name__", "unknown")
        function = frame.f_code.co_name
        line_number = frame.f_lineno

        # Get traceback
        tb_str = "".join(
            traceback.format_exception(type(exception), exception, exception.__traceback__)
        )

        # Create error event
        error_id = f"EXC_{datetime.now().timestamp():.0f}"

        error_event = ErrorEvent(
            error_id=error_id,
            severity=severity,
            category=category,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback=tb_str,
            timestamp=datetime.now(),
            module=module,
            function=function,
            line_number=line_number,
            context=context or {},
            user_impact=user_impact,
            recovery_action=recovery_action,
        )

        # Process error
        self._process_error(error_event)

        return error_id

    def _check_rate_limit(self, severity: ErrorSeverity) -> bool:
        """Check if error can be logged based on rate limits."""
        current_time = datetime.now()
        minute_ago = current_time - timedelta(minutes=1)

        # Clean old entries
        counter = self.rate_counters[severity]
        while counter and counter[0] < minute_ago:
            counter.popleft()

        # Check limit
        if len(counter) >= self.rate_limits[severity]:
            return False

        # Add timestamp
        counter.append(current_time)
        return True

    def _process_error(self, error_event: ErrorEvent):
        """Process and store error event."""
        # Store in history
        self.error_history.append(error_event)

        # Update counts
        self.error_counts[error_event.category.value][error_event.severity.value] += 1

        # Update metrics
        self.metrics["total_errors"] += 1
        self.metrics["errors_by_severity"][error_event.severity] += 1
        self.metrics["errors_by_category"][error_event.category] += 1

        # Log to file
        self._log_to_file(error_event)

        # Log to JSON
        self._log_to_json(error_event)

        # Check for patterns
        self._check_error_patterns(error_event)

        # Trigger handlers
        self._trigger_handlers(error_event)

        # Send alerts if needed
        if error_event.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            asyncio.create_task(self._send_alert(error_event))

        # Log to standard logger
        log_message = f"[{error_event.category.value}] {error_event.message}"

        if error_event.severity == ErrorSeverity.DEBUG:
            logger.debug(log_message)
        elif error_event.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif error_event.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif error_event.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error_event.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            logger.critical(log_message)

    def _log_to_file(self, error_event: ErrorEvent):
        """Log error to appropriate file."""
        # Determine log file
        if error_event.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            log_file = self.log_dir / "critical_errors.log"
        else:
            log_file = self.log_dir / f"errors_{error_event.category.value}.log"

        # Format log entry
        log_entry = f"""
================================================================================
Error ID: {error_event.error_id}
Time: {error_event.timestamp}
Severity: {error_event.severity.value}
Category: {error_event.category.value}
Module: {error_event.module}
Function: {error_event.function}
Line: {error_event.line_number}

Message: {error_event.message}

User Impact: {error_event.user_impact}
Recovery Action: {error_event.recovery_action}

Context:
{json.dumps(error_event.context, indent=2)}

{error_event.traceback if error_event.traceback else ''}
================================================================================
"""

        # Append to file
        with open(log_file, "a") as f:
            f.write(log_entry)

    def _log_to_json(self, error_event: ErrorEvent):
        """Log error to JSON file for analysis."""
        json_file = self.log_dir / f'errors_{datetime.now().strftime("%Y%m%d")}.json'

        # Convert to dict
        error_dict = asdict(error_event)
        error_dict["timestamp"] = error_event.timestamp.isoformat()
        error_dict["severity"] = error_event.severity.value
        error_dict["category"] = error_event.category.value

        # Append to JSON file
        with open(json_file, "a") as f:
            json.dump(error_dict, f)
            f.write("\n")

    def _check_error_patterns(self, error_event: ErrorEvent):
        """Check for error patterns and correlations."""
        # Group similar errors
        pattern_key = (
            f"{error_event.category.value}:{error_event.exception_type}:{error_event.module}"
        )
        self.error_patterns[pattern_key].append(error_event)

        # Check for rapid occurrence
        recent_errors = [
            e
            for e in self.error_patterns[pattern_key]
            if e.timestamp > datetime.now() - timedelta(minutes=5)
        ]

        if len(recent_errors) > 5:
            # Pattern detected
            logger.warning(
                f"Error pattern detected: {pattern_key} occurred {len(recent_errors)} times in 5 minutes"
            )

            # Trigger pattern alert
            if self.alert_system:
                asyncio.create_task(
                    self.alert_system.send_alert(
                        {
                            "type": "error_pattern",
                            "pattern": pattern_key,
                            "count": len(recent_errors),
                            "severity": "warning",
                        }
                    )
                )

    def _trigger_handlers(self, error_event: ErrorEvent):
        """Trigger registered error handlers."""
        handlers = self.error_handlers.get(error_event.category, [])

        for handler in handlers:
            try:
                handler(error_event)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")

    async def _send_alert(self, error_event: ErrorEvent):
        """Send alert for critical errors."""
        if not self.alert_system:
            return

        await self.alert_system.alert_system_error(
            error_type=error_event.category.value,
            error_message=error_event.message,
            traceback=error_event.traceback,
        )

    async def _monitor_error_patterns(self):
        """Monitor for error patterns and trends."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Calculate error rate
                recent_errors = [
                    e
                    for e in self.error_history
                    if e.timestamp > datetime.now() - timedelta(hours=1)
                ]

                error_rate = len(recent_errors) / 60  # Errors per minute
                self.metrics["error_rate"].append((datetime.now(), error_rate))

                # Check for anomalies
                if error_rate > 1.0:  # More than 1 error per minute
                    logger.warning(f"High error rate detected: {error_rate:.2f} errors/minute")

                # Check for specific patterns
                self._analyze_patterns()

            except Exception as e:
                logger.error(f"Error in pattern monitoring: {e}")

    def _analyze_patterns(self):
        """Analyze error patterns for insights."""
        # Find most common errors
        error_frequencies = defaultdict(int)

        for error in self.error_history:
            key = f"{error.category.value}:{error.message[:50]}"
            error_frequencies[key] += 1

        # Log top errors
        top_errors = sorted(error_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]

        if top_errors:
            logger.info("Top error patterns:")
            for error_key, count in top_errors:
                logger.info(f"  {error_key}: {count} occurrences")

    async def _cleanup_old_logs(self):
        """Clean up old log files."""
        while self._running:
            try:
                await asyncio.sleep(86400)  # Daily

                # Remove logs older than configured retention
                retention_days = self.config.get("log_retention_days", 30)
                cutoff_date = datetime.now() - timedelta(days=retention_days)

                for log_file in self.log_dir.glob("*.log"):
                    if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                        log_file.unlink()
                        logger.info(f"Removed old log file: {log_file}")

            except Exception as e:
                logger.error(f"Error in log cleanup: {e}")

    async def _analyze_error_trends(self):
        """Analyze error trends for reporting."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Hourly

                # Generate error report
                report = self.generate_error_report()

                # Save report
                report_file = (
                    self.log_dir / f'error_report_{datetime.now().strftime("%Y%m%d_%H")}.json'
                )
                with open(report_file, "w") as f:
                    json.dump(report, f, indent=2, default=str)

            except Exception as e:
                logger.error(f"Error in trend analysis: {e}")

    def register_handler(self, category: ErrorCategory, handler: Callable):
        """Register error handler for specific category."""
        self.error_handlers[category].append(handler)

    def get_recent_errors(
        self,
        minutes: int = 60,
        severity: ErrorSeverity | None = None,
        category: ErrorCategory | None = None,
    ) -> list[ErrorEvent]:
        """Get recent errors with optional filtering."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        errors = [e for e in self.error_history if e.timestamp > cutoff_time]

        if severity:
            errors = [e for e in errors if e.severity == severity]

        if category:
            errors = [e for e in errors if e.category == category]

        return errors

    def generate_error_report(self) -> dict[str, Any]:
        """Generate comprehensive error report."""
        report = {
            "generated_at": datetime.now(),
            "total_errors": self.metrics["total_errors"],
            "errors_by_severity": dict(self.metrics["errors_by_severity"]),
            "errors_by_category": dict(self.metrics["errors_by_category"]),
            "error_rate": {
                "current": self.metrics["error_rate"][-1][1] if self.metrics["error_rate"] else 0,
                "average": (
                    np.mean([r[1] for r in self.metrics["error_rate"]])
                    if self.metrics["error_rate"]
                    else 0
                ),
                "max": (
                    max([r[1] for r in self.metrics["error_rate"]])
                    if self.metrics["error_rate"]
                    else 0
                ),
            },
            "top_errors": [],
            "critical_errors": [],
            "patterns": [],
        }

        # Add top errors
        error_counts = defaultdict(int)
        for error in self.error_history:
            key = f"{error.category.value}:{error.message[:100]}"
            error_counts[key] += 1

        report["top_errors"] = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        # Add critical errors
        report["critical_errors"] = [
            {
                "error_id": e.error_id,
                "timestamp": e.timestamp,
                "message": e.message,
                "impact": e.user_impact,
            }
            for e in self.error_history
            if e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]
        ][
            -10:
        ]  # Last 10 critical errors

        # Add patterns
        for pattern_key, errors in self.error_patterns.items():
            if len(errors) > 5:
                report["patterns"].append(
                    {
                        "pattern": pattern_key,
                        "count": len(errors),
                        "first_seen": min(e.timestamp for e in errors),
                        "last_seen": max(e.timestamp for e in errors),
                    }
                )

        return report

    # Convenience methods for common error types

    def log_data_error(self, message: str, symbol: str | None = None, **kwargs):
        """Log data-related error."""
        context = {"symbol": symbol} if symbol else {}
        context.update(kwargs)

        return self.log_error(
            message=message,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.DATA,
            context=context,
            user_impact="Data quality may be compromised",
            recovery_action="Verify data source and retry",
        )

    def log_trading_error(self, message: str, order_id: str | None = None, **kwargs):
        """Log trading-related error."""
        context = {"order_id": order_id} if order_id else {}
        context.update(kwargs)

        return self.log_error(
            message=message,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.TRADING,
            context=context,
            user_impact="Trade execution may be affected",
            recovery_action="Check order status and positions",
        )

    def log_risk_error(self, message: str, risk_type: str, **kwargs):
        """Log risk-related error."""
        context = {"risk_type": risk_type}
        context.update(kwargs)

        return self.log_error(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.RISK,
            context=context,
            user_impact="Risk limits may be breached",
            recovery_action="Review positions and reduce exposure",
        )
