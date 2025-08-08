"""
Alerting Service for Validation Failures

This module provides integration with external alerting services like Slack, Email, and PagerDuty
for critical validation failures and system issues.
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class AlertChannel(Enum):
    """Supported alert channels."""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    LOG_ONLY = "log_only"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertingService:
    """
    Service for sending alerts through various channels.
    
    Supports Slack webhooks, email via SMTP, and PagerDuty integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alerting service with configuration.
        
        Args:
            config: Configuration dictionary (required)
        """
        if not config:
            raise ValueError("Configuration is required for AlertingService")
        self.config = config
        
        # Load alerting configuration
        alerting_config = self.config.get('alerting', {})
        self.enabled_channels = alerting_config.get('enabled_channels', ['log_only'])
        
        # Slack configuration
        self.slack_webhook_url = alerting_config.get('slack.webhook_url')
        self.slack_channel = alerting_config.get('slack.channel', '#alerts')
        self.slack_username = alerting_config.get('slack.username', 'AI Trader Alert')
        
        # Email configuration
        self.smtp_host = alerting_config.get('email.smtp_host')
        self.smtp_port = alerting_config.get('email.smtp_port', 587)
        self.smtp_username = alerting_config.get('email.smtp_username')
        self.smtp_password = alerting_config.get('email.smtp_password')
        self.email_from = alerting_config.get('email.from_address')
        self.email_recipients = alerting_config.get('email.recipients', [])
        
        # PagerDuty configuration
        self.pagerduty_api_key = alerting_config.get('pagerduty.api_key')
        self.pagerduty_service_id = alerting_config.get('pagerduty.service_id')
        
        # Alert throttling
        self.throttle_seconds = alerting_config.get('throttle_seconds', 300)  # 5 minutes
        self._alert_history: Dict[str, datetime] = {}
        
        logger.info(f"AlertingService initialized with channels: {self.enabled_channels}")
    
    async def send_alert(
        self,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        channels: Optional[List[AlertChannel]] = None
    ) -> Dict[str, Any]:
        """
        Send an alert through configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            priority: Alert priority
            context: Additional context data
            channels: Specific channels to use (defaults to all enabled)
            
        Returns:
            Dictionary with send results for each channel
        """
        # Check throttling
        alert_key = f"{title}:{message}"
        if self._is_throttled(alert_key):
            logger.debug(f"Alert throttled: {title}")
            return {'throttled': True}
        
        # Use specified channels or all enabled channels
        channels_to_use = channels or [AlertChannel(ch) for ch in self.enabled_channels]
        
        results = {}
        
        for channel in channels_to_use:
            try:
                if channel == AlertChannel.SLACK and self.slack_webhook_url:
                    results['slack'] = await self._send_slack_alert(title, message, priority, context)
                elif channel == AlertChannel.EMAIL and self.smtp_host:
                    results['email'] = await self._send_email_alert(title, message, priority, context)
                elif channel == AlertChannel.PAGERDUTY and self.pagerduty_api_key:
                    results['pagerduty'] = await self._send_pagerduty_alert(title, message, priority, context)
                elif channel == AlertChannel.LOG_ONLY:
                    results['log'] = self._log_alert(title, message, priority, context)
                else:
                    logger.warning(f"Channel {channel.value} not configured or not supported")
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
                results[channel.value] = {'success': False, 'error': str(e)}
        
        # Update throttle history
        self._alert_history[alert_key] = datetime.utcnow()
        
        return results
    
    async def _send_slack_alert(
        self,
        title: str,
        message: str,
        priority: AlertPriority,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send alert via Slack webhook."""
        try:
            # Map priority to emoji and color
            priority_config = {
                AlertPriority.LOW: {"emoji": ":information_source:", "color": "#36a64f"},
                AlertPriority.MEDIUM: {"emoji": ":warning:", "color": "#ff9800"},
                AlertPriority.HIGH: {"emoji": ":exclamation:", "color": "#ff5722"},
                AlertPriority.CRITICAL: {"emoji": ":rotating_light:", "color": "#f44336"}
            }
            
            config = priority_config[priority]
            
            # Build Slack message
            slack_message = {
                "channel": self.slack_channel,
                "username": self.slack_username,
                "icon_emoji": config["emoji"],
                "attachments": [
                    {
                        "color": config["color"],
                        "title": f"{config['emoji']} {title}",
                        "text": message,
                        "fields": [],
                        "footer": "AI Trader Alerting System",
                        "ts": int(datetime.utcnow().timestamp())
                    }
                ]
            }
            
            # Add context fields if provided
            if context:
                for key, value in context.items():
                    slack_message["attachments"][0]["fields"].append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_webhook_url,
                    json=slack_message,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent: {title}")
                        return {'success': True}
                    else:
                        error_msg = f"Slack webhook returned {response.status}"
                        logger.error(error_msg)
                        return {'success': False, 'error': error_msg}
                        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _send_email_alert(
        self,
        title: str,
        message: str,
        priority: AlertPriority,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send alert via email."""
        try:
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{priority.value.upper()}] {title}"
            msg['From'] = self.email_from
            msg['To'] = ', '.join(self.email_recipients)
            
            # Build HTML body
            html_body = f"""
            <html>
                <head></head>
                <body>
                    <h2>{title}</h2>
                    <p><strong>Priority:</strong> {priority.value.upper()}</p>
                    <p><strong>Time:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    <hr>
                    <p>{message}</p>
            """
            
            if context:
                html_body += "<hr><h3>Additional Context:</h3><ul>"
                for key, value in context.items():
                    html_body += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
                html_body += "</ul>"
            
            html_body += """
                    <hr>
                    <p><em>This is an automated alert from AI Trader.</em></p>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_email_sync, msg)
            
            logger.info(f"Email alert sent: {title}")
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_email_sync(self, msg: MIMEMultipart):
        """Synchronous email sending (called in thread pool)."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
    
    async def _send_pagerduty_alert(
        self,
        title: str,
        message: str,
        priority: AlertPriority,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send alert via PagerDuty."""
        try:
            # Map priority to PagerDuty severity
            severity_map = {
                AlertPriority.LOW: "info",
                AlertPriority.MEDIUM: "warning",
                AlertPriority.HIGH: "error",
                AlertPriority.CRITICAL: "critical"
            }
            
            # Build PagerDuty event
            event = {
                "routing_key": self.pagerduty_service_id,
                "event_action": "trigger",
                "payload": {
                    "summary": title,
                    "source": "ai-trader",
                    "severity": severity_map[priority],
                    "custom_details": {
                        "message": message,
                        **(context or {})
                    }
                }
            }
            
            # Send to PagerDuty
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=event,
                    headers={
                        "Authorization": f"Token token={self.pagerduty_api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=10
                ) as response:
                    if response.status == 202:
                        logger.info(f"PagerDuty alert sent: {title}")
                        return {'success': True}
                    else:
                        error_msg = f"PagerDuty API returned {response.status}"
                        logger.error(error_msg)
                        return {'success': False, 'error': error_msg}
                        
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return {'success': False, 'error': str(e)}
    
    def _log_alert(
        self,
        title: str,
        message: str,
        priority: AlertPriority,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Log alert (fallback when no external channels configured)."""
        log_level = {
            AlertPriority.LOW: logging.INFO,
            AlertPriority.MEDIUM: logging.WARNING,
            AlertPriority.HIGH: logging.ERROR,
            AlertPriority.CRITICAL: logging.CRITICAL
        }[priority]
        
        log_message = f"ALERT [{priority.value.upper()}] {title}: {message}"
        if context:
            log_message += f" | Context: {json.dumps(context)}"
        
        logger.log(log_level, log_message)
        return {'success': True}
    
    def _is_throttled(self, alert_key: str) -> bool:
        """Check if an alert should be throttled."""
        if alert_key not in self._alert_history:
            return False
        
        last_sent = self._alert_history[alert_key]
        elapsed = (datetime.utcnow() - last_sent).total_seconds()
        
        return elapsed < self.throttle_seconds
    
    async def test_alerting(self) -> Dict[str, Any]:
        """Test all configured alerting channels."""
        results = await self.send_alert(
            title="Test Alert",
            message="This is a test alert from AI Trader to verify alerting configuration.",
            priority=AlertPriority.LOW,
            context={
                "test_time": datetime.utcnow().isoformat(),
                "configured_channels": ', '.join(self.enabled_channels)
            }
        )
        
        return results


# Global alerting service instance
_alerting_service: Optional[AlertingService] = None


def get_alerting_service(config: Optional[Dict[str, Any]] = None) -> AlertingService:
    """
    Get or create global alerting service instance.
    
    Args:
        config: Configuration dictionary (required for first initialization)
        
    Returns:
        AlertingService instance
        
    Raises:
        ValueError: If config is not provided for first initialization
    """
    global _alerting_service
    if _alerting_service is None:
        if config is None:
            raise ValueError("Configuration is required for first AlertingService initialization")
        _alerting_service = AlertingService(config)
    return _alerting_service