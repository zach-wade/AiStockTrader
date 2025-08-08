"""
Model monitoring report generation.

This module generates comprehensive monitoring reports for ML models including:
- Performance summaries
- Data drift analysis
- Alert summaries
- Health status reports
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path

from main.utils.core import (
    get_logger,
    ErrorHandlingMixin,
    timer
)
from main.utils.database import DatabasePool
from main.utils.monitoring import record_metric

logger = get_logger(__name__)


@dataclass
class ModelHealthStatus:
    """Model health status summary."""
    model_name: str
    version: str
    overall_health: str  # 'healthy', 'warning', 'critical'
    health_score: float  # 0.0 to 1.0
    last_checked: datetime
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Model performance report."""
    model_name: str
    version: str
    period_start: datetime
    period_end: datetime
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    
    # Trading-specific metrics
    profit_loss: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Data quality
    data_drift_score: Optional[float] = None
    feature_drift_count: int = 0


@dataclass
class DriftReport:
    """Data drift analysis report."""
    model_name: str
    version: str
    analysis_date: datetime
    
    # Overall drift
    overall_drift_score: float
    drift_threshold: float
    is_drifting: bool
    
    # Feature-level drift
    feature_drift_scores: Dict[str, float] = field(default_factory=dict)
    drifting_features: List[str] = field(default_factory=list)
    
    # Statistical tests
    ks_test_results: Dict[str, float] = field(default_factory=dict)
    chi_square_results: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MonitoringReport:
    """Comprehensive monitoring report."""
    report_id: str
    generated_at: datetime
    report_period: Tuple[datetime, datetime]
    
    # Model summaries
    model_health_statuses: List[ModelHealthStatus] = field(default_factory=list)
    performance_reports: List[PerformanceReport] = field(default_factory=list)
    drift_reports: List[DriftReport] = field(default_factory=list)
    
    # System-wide metrics
    total_models: int = 0
    healthy_models: int = 0
    models_with_warnings: int = 0
    critical_models: int = 0
    
    # Alert summary
    alert_count: int = 0
    critical_alerts: int = 0
    warning_alerts: int = 0
    
    # Recommendations
    action_items: List[str] = field(default_factory=list)


class MonitorReporter(ErrorHandlingMixin):
    """
    Generates comprehensive monitoring reports for ML models.
    
    Features:
    - Health status reports
    - Performance analysis
    - Data drift detection
    - Alert summaries
    - Automated recommendations
    """
    
    def __init__(self, db_pool: DatabasePool, report_storage_path: str = "./reports"):
        """Initialize monitor reporter."""
        super().__init__()
        self.db_pool = db_pool
        self.report_storage_path = Path(report_storage_path)
        self.report_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Report templates
        self._health_thresholds = {
            'accuracy': 0.8,
            'error_rate': 0.05,
            'latency_ms': 1000,
            'drift_score': 0.3
        }
        
    @timer
    async def generate_comprehensive_report(
        self,
        models: Optional[List[Tuple[str, str]]] = None,
        period_hours: int = 24
    ) -> MonitoringReport:
        """
        Generate comprehensive monitoring report.
        
        Args:
            models: Optional list of (model_name, version) tuples
            period_hours: Report period in hours
            
        Returns:
            Comprehensive monitoring report
        """
        with self._handle_error("generating comprehensive report"):
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=period_hours)
            
            # Get models to report on
            if models is None:
                models = await self._get_active_models()
            
            # Generate report ID
            report_id = f"monitoring_report_{end_time.strftime('%Y%m%d_%H%M%S')}"
            
            # Collect data
            health_statuses = await self._collect_health_statuses(models)
            performance_reports = await self._collect_performance_reports(
                models, start_time, end_time
            )
            drift_reports = await self._collect_drift_reports(models)
            
            # Calculate system-wide metrics
            total_models = len(models)
            healthy_models = sum(
                1 for h in health_statuses 
                if h.overall_health == 'healthy'
            )
            models_with_warnings = sum(
                1 for h in health_statuses 
                if h.overall_health == 'warning'
            )
            critical_models = sum(
                1 for h in health_statuses 
                if h.overall_health == 'critical'
            )
            
            # Get alert summary
            alert_summary = await self._get_alert_summary(
                start_time, end_time
            )
            
            # Generate action items
            action_items = await self._generate_action_items(
                health_statuses, performance_reports, drift_reports
            )
            
            # Create report
            report = MonitoringReport(
                report_id=report_id,
                generated_at=end_time,
                report_period=(start_time, end_time),
                model_health_statuses=health_statuses,
                performance_reports=performance_reports,
                drift_reports=drift_reports,
                total_models=total_models,
                healthy_models=healthy_models,
                models_with_warnings=models_with_warnings,
                critical_models=critical_models,
                alert_count=alert_summary['total'],
                critical_alerts=alert_summary['critical'],
                warning_alerts=alert_summary['warning'],
                action_items=action_items
            )
            
            # Store report
            await self._store_report(report)
            
            # Record metrics
            record_metric(
                'monitor_reporter.report_generated',
                1,
                tags={
                    'models': total_models,
                    'healthy': healthy_models,
                    'warnings': models_with_warnings,
                    'critical': critical_models
                }
            )
            
            logger.info(
                f"Generated monitoring report {report_id} for "
                f"{total_models} models"
            )
            
            return report
    
    async def generate_model_health_report(
        self,
        model_name: str,
        version: str
    ) -> ModelHealthStatus:
        """Generate health status report for a specific model."""
        with self._handle_error("generating model health report"):
            # Collect health metrics
            metrics = await self._collect_model_metrics(
                model_name, version
            )
            
            # Assess health
            health_assessment = self._assess_model_health(metrics)
            
            # Create health status
            health_status = ModelHealthStatus(
                model_name=model_name,
                version=version,
                overall_health=health_assessment['status'],
                health_score=health_assessment['score'],
                last_checked=datetime.utcnow(),
                issues=health_assessment['issues'],
                metrics=metrics
            )
            
            return health_status
    
    async def generate_performance_report(
        self,
        model_name: str,
        version: str,
        period_hours: int = 24
    ) -> PerformanceReport:
        """Generate performance report for a model."""
        with self._handle_error("generating performance report"):
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=period_hours)
            
            # Query performance metrics
            async with self.db_pool.acquire() as conn:
                # Get ML metrics
                ml_metrics_query = """
                    SELECT 
                        AVG(CASE WHEN metric_name = 'accuracy' THEN metric_value END) as accuracy,
                        AVG(CASE WHEN metric_name = 'precision' THEN metric_value END) as precision,
                        AVG(CASE WHEN metric_name = 'recall' THEN metric_value END) as recall,
                        AVG(CASE WHEN metric_name = 'f1_score' THEN metric_value END) as f1_score,
                        AVG(CASE WHEN metric_name = 'auc' THEN metric_value END) as auc
                    FROM model_metrics
                    WHERE model_name = $1 AND version = $2
                    AND timestamp BETWEEN $3 AND $4
                """
                
                ml_row = await conn.fetchrow(
                    ml_metrics_query, model_name, version, start_time, end_time
                )
                
                # Get trading metrics
                trading_metrics_query = """
                    SELECT 
                        AVG(CASE WHEN metric_name = 'profit_loss' THEN metric_value END) as profit_loss,
                        AVG(CASE WHEN metric_name = 'sharpe_ratio' THEN metric_value END) as sharpe_ratio,
                        AVG(CASE WHEN metric_name = 'max_drawdown' THEN metric_value END) as max_drawdown,
                        AVG(CASE WHEN metric_name = 'win_rate' THEN metric_value END) as win_rate
                    FROM model_metrics
                    WHERE model_name = $1 AND version = $2
                    AND timestamp BETWEEN $3 AND $4
                """
                
                trading_row = await conn.fetchrow(
                    trading_metrics_query, model_name, version, start_time, end_time
                )
                
                # Get request metrics
                request_metrics_query = """
                    SELECT 
                        COUNT(*) as total_requests,
                        COUNT(CASE WHEN success = true THEN 1 END) as successful_requests,
                        AVG(latency_ms) as avg_latency_ms
                    FROM model_requests
                    WHERE model_name = $1 AND version = $2
                    AND timestamp BETWEEN $3 AND $4
                """
                
                request_row = await conn.fetchrow(
                    request_metrics_query, model_name, version, start_time, end_time
                )
                
                # Get drift metrics
                drift_query = """
                    SELECT 
                        drift_score,
                        feature_drift_count
                    FROM model_drift_analysis
                    WHERE model_name = $1 AND version = $2
                    ORDER BY analysis_date DESC
                    LIMIT 1
                """
                
                drift_row = await conn.fetchrow(
                    drift_query, model_name, version
                )
            
            # Calculate error rate
            total_requests = request_row['total_requests'] or 0
            successful_requests = request_row['successful_requests'] or 0
            error_rate = (
                (total_requests - successful_requests) / max(total_requests, 1)
            )
            
            # Create performance report
            report = PerformanceReport(
                model_name=model_name,
                version=version,
                period_start=start_time,
                period_end=end_time,
                accuracy=ml_row['accuracy'],
                precision=ml_row['precision'],
                recall=ml_row['recall'],
                f1_score=ml_row['f1_score'],
                auc=ml_row['auc'],
                profit_loss=trading_row['profit_loss'],
                sharpe_ratio=trading_row['sharpe_ratio'],
                max_drawdown=trading_row['max_drawdown'],
                win_rate=trading_row['win_rate'],
                total_requests=total_requests,
                successful_requests=successful_requests,
                error_rate=error_rate,
                avg_latency_ms=request_row['avg_latency_ms'] or 0.0,
                data_drift_score=drift_row['drift_score'] if drift_row else None,
                feature_drift_count=drift_row['feature_drift_count'] if drift_row else 0
            )
            
            return report
    
    async def generate_drift_report(
        self,
        model_name: str,
        version: str
    ) -> Optional[DriftReport]:
        """Generate data drift report for a model."""
        with self._handle_error("generating drift report"):
            async with self.db_pool.acquire() as conn:
                # Get latest drift analysis
                query = """
                    SELECT *
                    FROM model_drift_analysis
                    WHERE model_name = $1 AND version = $2
                    ORDER BY analysis_date DESC
                    LIMIT 1
                """
                
                row = await conn.fetchrow(query, model_name, version)
                
                if not row:
                    return None
                
                # Parse JSON fields
                feature_drift_scores = json.loads(
                    row['feature_drift_scores'] or '{}'
                )
                ks_test_results = json.loads(
                    row['ks_test_results'] or '{}'
                )
                chi_square_results = json.loads(
                    row['chi_square_results'] or '{}'
                )
                
                # Identify drifting features
                drift_threshold = row['drift_threshold']
                drifting_features = [
                    feature for feature, score in feature_drift_scores.items()
                    if score > drift_threshold
                ]
                
                # Generate recommendations
                recommendations = self._generate_drift_recommendations(
                    row['drift_score'], drifting_features
                )
                
                # Create drift report
                drift_report = DriftReport(
                    model_name=model_name,
                    version=version,
                    analysis_date=row['analysis_date'],
                    overall_drift_score=row['drift_score'],
                    drift_threshold=drift_threshold,
                    is_drifting=row['drift_score'] > drift_threshold,
                    feature_drift_scores=feature_drift_scores,
                    drifting_features=drifting_features,
                    ks_test_results=ks_test_results,
                    chi_square_results=chi_square_results,
                    recommendations=recommendations
                )
                
                return drift_report
    
    async def _get_active_models(self) -> List[Tuple[str, str]]:
        """Get list of active models."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT DISTINCT model_name, version
                FROM model_versions
                WHERE status IN ('production', 'staged')
                ORDER BY model_name, version
            """
            
            rows = await conn.fetch(query)
            return [(row['model_name'], row['version']) for row in rows]
    
    async def _collect_health_statuses(
        self,
        models: List[Tuple[str, str]]
    ) -> List[ModelHealthStatus]:
        """Collect health statuses for models."""
        health_statuses = []
        
        for model_name, version in models:
            health_status = await self.generate_model_health_report(
                model_name, version
            )
            health_statuses.append(health_status)
        
        return health_statuses
    
    async def _collect_performance_reports(
        self,
        models: List[Tuple[str, str]],
        start_time: datetime,
        end_time: datetime
    ) -> List[PerformanceReport]:
        """Collect performance reports for models."""
        performance_reports = []
        
        period_hours = int((end_time - start_time).total_seconds() / 3600)
        
        for model_name, version in models:
            report = await self.generate_performance_report(
                model_name, version, period_hours
            )
            performance_reports.append(report)
        
        return performance_reports
    
    async def _collect_drift_reports(
        self,
        models: List[Tuple[str, str]]
    ) -> List[DriftReport]:
        """Collect drift reports for models."""
        drift_reports = []
        
        for model_name, version in models:
            drift_report = await self.generate_drift_report(
                model_name, version
            )
            if drift_report:
                drift_reports.append(drift_report)
        
        return drift_reports
    
    async def _collect_model_metrics(
        self,
        model_name: str,
        version: str
    ) -> Dict[str, float]:
        """Collect current metrics for a model."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT metric_name, metric_value
                FROM model_metrics
                WHERE model_name = $1 AND version = $2
                AND timestamp > $3
                ORDER BY timestamp DESC
            """
            
            # Get metrics from last hour
            cutoff = datetime.utcnow() - timedelta(hours=1)
            rows = await conn.fetch(query, model_name, version, cutoff)
            
            # Get latest value for each metric
            metrics = {}
            seen_metrics = set()
            
            for row in rows:
                metric_name = row['metric_name']
                if metric_name not in seen_metrics:
                    metrics[metric_name] = row['metric_value']
                    seen_metrics.add(metric_name)
            
            return metrics
    
    def _assess_model_health(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Assess overall model health based on metrics."""
        issues = []
        health_score = 1.0
        
        # Check accuracy
        accuracy = metrics.get('accuracy')
        if accuracy and accuracy < self._health_thresholds['accuracy']:
            issues.append(f"Low accuracy: {accuracy:.2%}")
            health_score -= 0.3
        
        # Check error rate
        error_rate = metrics.get('error_rate')
        if error_rate and error_rate > self._health_thresholds['error_rate']:
            issues.append(f"High error rate: {error_rate:.2%}")
            health_score -= 0.2
        
        # Check latency
        latency = metrics.get('avg_latency_ms')
        if latency and latency > self._health_thresholds['latency_ms']:
            issues.append(f"High latency: {latency:.0f}ms")
            health_score -= 0.1
        
        # Check drift
        drift_score = metrics.get('data_drift_score')
        if drift_score and drift_score > self._health_thresholds['drift_score']:
            issues.append(f"Data drift detected: {drift_score:.2f}")
            health_score -= 0.4
        
        # Determine overall status
        health_score = max(0.0, health_score)
        
        if health_score >= 0.8:
            status = 'healthy'
        elif health_score >= 0.5:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'score': health_score,
            'issues': issues
        }
    
    async def _get_alert_summary(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, int]:
        """Get alert summary for the period."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical,
                    COUNT(CASE WHEN severity = 'warning' THEN 1 END) as warning
                FROM model_alerts
                WHERE timestamp BETWEEN $1 AND $2
            """
            
            row = await conn.fetchrow(query, start_time, end_time)
            
            return {
                'total': row['total'] or 0,
                'critical': row['critical'] or 0,
                'warning': row['warning'] or 0
            }
    
    async def _generate_action_items(
        self,
        health_statuses: List[ModelHealthStatus],
        performance_reports: List[PerformanceReport],
        drift_reports: List[DriftReport]
    ) -> List[str]:
        """Generate action items based on analysis."""
        action_items = []
        
        # Critical health issues
        critical_models = [
            h for h in health_statuses 
            if h.overall_health == 'critical'
        ]
        
        if critical_models:
            action_items.append(
                f"URGENT: {len(critical_models)} models in critical state require immediate attention"
            )
        
        # Data drift issues
        drifting_models = [d for d in drift_reports if d.is_drifting]
        
        if drifting_models:
            action_items.append(
                f"Data drift detected in {len(drifting_models)} models - consider retraining"
            )
        
        # Performance degradation
        poor_performance = [
            p for p in performance_reports
            if p.accuracy and p.accuracy < 0.7
        ]
        
        if poor_performance:
            action_items.append(
                f"{len(poor_performance)} models showing poor accuracy - investigate and retrain"
            )
        
        # High error rates
        high_error_models = [
            p for p in performance_reports
            if p.error_rate > 0.1
        ]
        
        if high_error_models:
            action_items.append(
                f"{len(high_error_models)} models have high error rates - check infrastructure"
            )
        
        return action_items
    
    def _generate_drift_recommendations(
        self,
        drift_score: float,
        drifting_features: List[str]
    ) -> List[str]:
        """Generate recommendations for drift issues."""
        recommendations = []
        
        if drift_score > 0.5:
            recommendations.append("Immediate retraining recommended")
        elif drift_score > 0.3:
            recommendations.append("Schedule retraining within 1 week")
        
        if len(drifting_features) > 5:
            recommendations.append(
                "Consider feature selection - many features drifting"
            )
        
        if drifting_features:
            recommendations.append(
                f"Focus on these drifting features: {', '.join(drifting_features[:5])}"
            )
        
        return recommendations
    
    async def _store_report(
        self,
        report: MonitoringReport
    ) -> None:
        """Store report to file and database."""
        # Store to file
        report_file = self.report_storage_path / f"{report.report_id}.json"
        
        # Convert to JSON-serializable format
        report_dict = {
            'report_id': report.report_id,
            'generated_at': report.generated_at.isoformat(),
            'report_period': [
                report.report_period[0].isoformat(),
                report.report_period[1].isoformat()
            ],
            'summary': {
                'total_models': report.total_models,
                'healthy_models': report.healthy_models,
                'models_with_warnings': report.models_with_warnings,
                'critical_models': report.critical_models,
                'alert_count': report.alert_count,
                'critical_alerts': report.critical_alerts,
                'warning_alerts': report.warning_alerts
            },
            'action_items': report.action_items
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Store summary in database
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO monitoring_reports (
                    report_id, generated_at, period_start, period_end,
                    total_models, healthy_models, models_with_warnings,
                    critical_models, alert_count, action_items
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                report.report_id,
                report.generated_at,
                report.report_period[0],
                report.report_period[1],
                report.total_models,
                report.healthy_models,
                report.models_with_warnings,
                report.critical_models,
                report.alert_count,
                json.dumps(report.action_items)
            )
        
        logger.info(f"Stored monitoring report: {report_file}")
    
    def export_report_to_html(
        self,
        report: MonitoringReport
    ) -> str:
        """Export report to HTML format."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Monitoring Report - {report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; }}
                .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 10px; }}
                .healthy {{ color: green; }}
                .warning {{ color: orange; }}
                .critical {{ color: red; }}
                .action-items {{ background-color: #fff3cd; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Monitoring Report</h1>
                <p>Generated: {generated_at}</p>
                <p>Period: {period_start} to {period_end}</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>Total Models</h3>
                    <div class="value">{total_models}</div>
                </div>
                <div class="metric healthy">
                    <h3>Healthy</h3>
                    <div class="value">{healthy_models}</div>
                </div>
                <div class="metric warning">
                    <h3>Warnings</h3>
                    <div class="value">{models_with_warnings}</div>
                </div>
                <div class="metric critical">
                    <h3>Critical</h3>
                    <div class="value">{critical_models}</div>
                </div>
            </div>
            
            <div class="action-items">
                <h3>Action Items</h3>
                <ul>
                    {action_items_html}
                </ul>
            </div>
        </body>
        </html>
        """
        
        action_items_html = '\n'.join([
            f"<li>{item}</li>" for item in report.action_items
        ])
        
        return html_template.format(
            report_id=report.report_id,
            generated_at=report.generated_at.strftime('%Y-%m-%d %H:%M:%S'),
            period_start=report.report_period[0].strftime('%Y-%m-%d %H:%M'),
            period_end=report.report_period[1].strftime('%Y-%m-%d %H:%M'),
            total_models=report.total_models,
            healthy_models=report.healthy_models,
            models_with_warnings=report.models_with_warnings,
            critical_models=report.critical_models,
            action_items_html=action_items_html
        )