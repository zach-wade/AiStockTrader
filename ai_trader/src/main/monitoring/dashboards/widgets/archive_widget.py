"""
Archive Dashboard Widget

Provides real-time monitoring and visualization of archive system metrics.
Integrates with existing dashboard infrastructure to display archive statistics,
performance metrics, and health indicators.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
import asyncio

# Try to import ArchiveMetricsCollector, but make it optional
try:
    from main.monitoring.metrics.archive_metrics_collector import ArchiveMetricsCollector
except ImportError:
    ArchiveMetricsCollector = None  # Will work without metrics collector

# Try to import UnifiedMetrics, but make it optional
try:
    from main.monitoring.metrics.unified_metrics import UnifiedMetrics
except ImportError:
    UnifiedMetrics = None  # Will work without unified metrics

from main.utils.core import get_logger

logger = get_logger(__name__)


class ArchiveWidget:
    """
    Dashboard widget for archive system monitoring.
    
    Displays:
    - Storage statistics and usage
    - Operation throughput (reads/writes per second)
    - Compression efficiency metrics
    - Performance latency graphs
    - Health status indicators
    """
    
    def __init__(
        self,
        metrics_collector: Optional[Any] = None,  # ArchiveMetricsCollector when available
        config: Optional[Dict[str, Any]] = None,
        unified_metrics: Optional[Any] = None  # UnifiedMetrics when available
    ):
        """
        Initialize the archive widget.
        
        Args:
            metrics_collector: Archive metrics collector instance
            config: Widget configuration
        """
        self.metrics_collector = metrics_collector
        self.config = config or {}
        self.unified_metrics = unified_metrics
        
        # Widget settings
        self.refresh_interval = self.config.get('refresh_interval', 5)  # seconds
        self.time_window = self.config.get('time_window', 3600)  # 1 hour window
        self.panel_height = self.config.get('panel_height', 8)
        self.panel_width = self.config.get('panel_width', 12)
        
        # Cache for metrics
        self._cached_metrics = None
        self._last_update = None
        
        logger.info("ArchiveWidget initialized")
    
    async def render(self) -> Dict[str, Any]:
        """
        Render the widget as dashboard panels.
        
        Returns:
            Dictionary containing panel configurations
        """
        # Update metrics if needed
        await self.update_data()
        
        panels = {
            'archive_overview': self._create_overview_panel(),
            'archive_operations': self._create_operations_panel(),
            'archive_storage': self._create_storage_panel(),
            'archive_performance': self._create_performance_panel(),
            'archive_health': self._create_health_panel()
        }
        
        return {
            'title': 'Archive System Monitoring',
            'panels': panels,
            'layout': self._get_layout_config(),
            'refresh_interval': self.refresh_interval
        }
    
    async def update_data(self) -> Dict[str, Any]:
        """
        Fetch and update latest metrics from collector.
        
        Returns:
            Updated metrics dictionary
        """
        # Check cache validity
        if self._last_update:
            cache_age = (datetime.now(timezone.utc) - self._last_update).total_seconds()
            if cache_age < self.refresh_interval:
                return self._cached_metrics
        
        # Fetch new metrics
        if self.metrics_collector:
            try:
                storage_metrics = await self.metrics_collector.get_storage_metrics()
                operation_stats = self.metrics_collector.get_operation_stats()
                performance_metrics = self.metrics_collector.get_performance_metrics()
                
                self._cached_metrics = {
                    'storage': {
                        'total_files': storage_metrics.total_files,
                        'total_size_mb': storage_metrics.total_size_mb,
                        'avg_file_size_mb': storage_metrics.avg_file_size_mb,
                        'compression_ratio': storage_metrics.compression_ratio,
                        'by_type': storage_metrics.by_type,
                        'by_source': storage_metrics.by_source
                    },
                    'operations': operation_stats,
                    'performance': performance_metrics,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Query UnifiedMetrics if available
                if self.unified_metrics and UnifiedMetrics:
                    await self._enrich_with_unified_metrics()
                
                self._last_update = datetime.now(timezone.utc)
                
            except Exception as e:
                logger.error(f"Failed to update archive metrics: {e}")
                # Use default/empty metrics if collector fails
                self._cached_metrics = self._get_default_metrics()
        else:
            # No metrics collector available - use defaults
            self._cached_metrics = self._get_default_metrics()
        
        return self._cached_metrics
    
    def _create_overview_panel(self) -> Dict[str, Any]:
        """Create overview statistics panel."""
        metrics = self._cached_metrics or self._get_default_metrics()
        
        return {
            'type': 'stat',
            'title': 'Archive Overview',
            'targets': [
                {
                    'name': 'Total Files',
                    'value': metrics['storage']['total_files'],
                    'unit': 'files'
                },
                {
                    'name': 'Storage Used',
                    'value': f"{metrics['storage']['total_size_mb']:.2f}",
                    'unit': 'MB'
                },
                {
                    'name': 'Compression Ratio',
                    'value': f"{metrics['storage']['compression_ratio']:.2f}",
                    'unit': 'x'
                },
                {
                    'name': 'Avg File Size',
                    'value': f"{metrics['storage']['avg_file_size_mb']:.2f}",
                    'unit': 'MB'
                }
            ],
            'gridPos': {'h': self.panel_height, 'w': self.panel_width, 'x': 0, 'y': 0}
        }
    
    def _create_operations_panel(self) -> Dict[str, Any]:
        """Create operations throughput panel."""
        metrics = self._cached_metrics or self._get_default_metrics()
        ops = metrics['operations']
        
        # Calculate rates
        total_writes = ops.get('write', {}).get('total', 0)
        total_reads = ops.get('read', {}).get('total', 0)
        
        # Get time-based metrics if available
        if self.metrics_collector:
            recent_writes = self.metrics_collector._get_recent_operations('write', 60)
            recent_reads = self.metrics_collector._get_recent_operations('read', 60)
            write_rate = len(recent_writes) / 60.0 if recent_writes else 0
            read_rate = len(recent_reads) / 60.0 if recent_reads else 0
        else:
            write_rate = 0
            read_rate = 0
        
        return {
            'type': 'graph',
            'title': 'Operations Throughput',
            'targets': [
                {
                    'name': 'Write Rate',
                    'datapoints': self._generate_time_series(write_rate, 'writes/sec'),
                    'color': 'green'
                },
                {
                    'name': 'Read Rate',
                    'datapoints': self._generate_time_series(read_rate, 'reads/sec'),
                    'color': 'blue'
                }
            ],
            'yAxis': {'label': 'Operations/sec', 'min': 0},
            'xAxis': {'mode': 'time'},
            'gridPos': {'h': self.panel_height, 'w': self.panel_width, 'x': 12, 'y': 0}
        }
    
    def _create_storage_panel(self) -> Dict[str, Any]:
        """Create storage usage panel."""
        metrics = self._cached_metrics or self._get_default_metrics()
        storage = metrics['storage']
        
        # Prepare data for pie chart
        by_type_data = []
        for file_type, stats in storage.get('by_type', {}).items():
            by_type_data.append({
                'name': file_type,
                'value': stats.get('size_mb', 0),
                'y': stats.get('size_mb', 0)
            })
        
        return {
            'type': 'piechart',
            'title': 'Storage by Type',
            'data': by_type_data,
            'options': {
                'pieType': 'donut',
                'legendDisplayMode': 'table',
                'legendPlacement': 'right',
                'unit': 'MB'
            },
            'gridPos': {'h': self.panel_height, 'w': self.panel_width // 2, 'x': 0, 'y': self.panel_height}
        }
    
    def _create_performance_panel(self) -> Dict[str, Any]:
        """Create performance metrics panel."""
        metrics = self._cached_metrics or self._get_default_metrics()
        perf = metrics.get('performance', {})
        
        # Extract latency metrics
        write_latency = perf.get('write', {}).get('avg_latency_ms', 0)
        read_latency = perf.get('read', {}).get('avg_latency_ms', 0)
        
        return {
            'type': 'gauge',
            'title': 'Operation Latency',
            'targets': [
                {
                    'name': 'Write Latency',
                    'value': write_latency,
                    'unit': 'ms',
                    'thresholds': {
                        'mode': 'absolute',
                        'steps': [
                            {'value': 0, 'color': 'green'},
                            {'value': 100, 'color': 'yellow'},
                            {'value': 500, 'color': 'red'}
                        ]
                    }
                },
                {
                    'name': 'Read Latency',
                    'value': read_latency,
                    'unit': 'ms',
                    'thresholds': {
                        'mode': 'absolute',
                        'steps': [
                            {'value': 0, 'color': 'green'},
                            {'value': 50, 'color': 'yellow'},
                            {'value': 200, 'color': 'red'}
                        ]
                    }
                }
            ],
            'options': {
                'orientation': 'horizontal',
                'showThresholdLabels': True,
                'showThresholdMarkers': True
            },
            'gridPos': {'h': self.panel_height, 'w': self.panel_width // 2, 'x': 6, 'y': self.panel_height}
        }
    
    def _create_health_panel(self) -> Dict[str, Any]:
        """Create health status panel."""
        metrics = self._cached_metrics or self._get_default_metrics()
        
        # Calculate health indicators
        health_items = []
        
        # Storage health
        storage_mb = metrics['storage']['total_size_mb']
        storage_status = 'OK' if storage_mb < 10000 else 'WARNING' if storage_mb < 50000 else 'CRITICAL'
        health_items.append({
            'name': 'Storage',
            'status': storage_status,
            'value': f"{storage_mb:.1f} MB",
            'color': self._get_status_color(storage_status)
        })
        
        # Error rate
        ops = metrics['operations']
        total_ops = ops.get('write', {}).get('total', 0) + ops.get('read', {}).get('total', 0)
        failed_ops = ops.get('write', {}).get('failed', 0) + ops.get('read', {}).get('failed', 0)
        error_rate = (failed_ops / total_ops * 100) if total_ops > 0 else 0
        error_status = 'OK' if error_rate < 1 else 'WARNING' if error_rate < 5 else 'CRITICAL'
        health_items.append({
            'name': 'Error Rate',
            'status': error_status,
            'value': f"{error_rate:.2f}%",
            'color': self._get_status_color(error_status)
        })
        
        # Compression efficiency
        compression = metrics['storage']['compression_ratio']
        compression_status = 'OK' if compression > 1.5 else 'WARNING' if compression > 1.0 else 'CRITICAL'
        health_items.append({
            'name': 'Compression',
            'status': compression_status,
            'value': f"{compression:.2f}x",
            'color': self._get_status_color(compression_status)
        })
        
        return {
            'type': 'table',
            'title': 'System Health',
            'columns': [
                {'text': 'Component', 'type': 'string'},
                {'text': 'Status', 'type': 'string'},
                {'text': 'Value', 'type': 'string'}
            ],
            'rows': health_items,
            'options': {
                'showHeader': True,
                'cellHeight': 'sm'
            },
            'gridPos': {'h': self.panel_height, 'w': self.panel_width, 'x': 12, 'y': self.panel_height}
        }
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get critical alerts for the archive system.
        
        Returns:
            List of alert configurations
        """
        alerts = []
        
        if not self._cached_metrics:
            return alerts
        
        # High error rate alert
        ops = self._cached_metrics['operations']
        total_ops = ops.get('write', {}).get('total', 0) + ops.get('read', {}).get('total', 0)
        failed_ops = ops.get('write', {}).get('failed', 0) + ops.get('read', {}).get('failed', 0)
        error_rate = (failed_ops / total_ops * 100) if total_ops > 0 else 0
        
        if error_rate > 5:
            alerts.append({
                'name': 'High Archive Error Rate',
                'severity': 'critical',
                'condition': f"error_rate > 5%",
                'value': f"{error_rate:.2f}%",
                'message': f"Archive error rate is {error_rate:.2f}%, exceeding 5% threshold"
            })
        
        # Low compression alert
        compression = self._cached_metrics['storage']['compression_ratio']
        if compression < 1.0:
            alerts.append({
                'name': 'Poor Compression Ratio',
                'severity': 'warning',
                'condition': "compression_ratio < 1.0",
                'value': f"{compression:.2f}x",
                'message': f"Archive compression ratio is {compression:.2f}x, data is expanding!"
            })
        
        # High storage usage alert (example threshold)
        storage_mb = self._cached_metrics['storage']['total_size_mb']
        if storage_mb > 50000:  # 50GB threshold
            alerts.append({
                'name': 'High Storage Usage',
                'severity': 'warning',
                'condition': "storage > 50GB",
                'value': f"{storage_mb/1000:.1f} GB",
                'message': f"Archive storage usage is {storage_mb/1000:.1f} GB"
            })
        
        # Add any UnifiedMetrics alerts
        if self._cached_metrics and 'unified_alerts' in self._cached_metrics:
            for unified_alert in self._cached_metrics['unified_alerts']:
                alerts.append({
                    'name': f"UnifiedMetrics: {unified_alert['metric']}",
                    'severity': unified_alert['severity'],
                    'condition': f"threshold: {unified_alert['threshold']}",
                    'value': '',
                    'message': unified_alert['message']
                })
        
        return alerts
    
    def _get_layout_config(self) -> Dict[str, Any]:
        """Get layout configuration for panels."""
        return {
            'grid': {
                'columns': 24,
                'rows': 'auto',
                'gap': 1
            },
            'responsive': True,
            'theme': self.config.get('theme', 'dark')
        }
    
    def _generate_time_series(self, value: float, unit: str) -> List[Tuple[float, float]]:
        """Generate time series data points for graphs."""
        # Generate dummy time series for visualization
        # In production, this would come from actual metrics history
        now = datetime.now(timezone.utc)
        points = []
        for i in range(60):  # Last 60 data points
            timestamp = now - timedelta(seconds=i * 60)
            # Add some variation to make it look realistic
            variation = value * (0.8 + 0.4 * (i % 10) / 10)
            points.append((timestamp.timestamp() * 1000, variation))
        return points[::-1]  # Reverse to get chronological order
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status indicator."""
        colors = {
            'OK': 'green',
            'WARNING': 'yellow',
            'CRITICAL': 'red',
            'UNKNOWN': 'gray'
        }
        return colors.get(status, 'gray')
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics when collector is unavailable."""
        return {
            'storage': {
                'total_files': 0,
                'total_size_mb': 0.0,
                'avg_file_size_mb': 0.0,
                'compression_ratio': 1.0,
                'by_type': {},
                'by_source': {}
            },
            'operations': {
                'write': {'total': 0, 'success': 0, 'failed': 0},
                'read': {'total': 0, 'success': 0, 'failed': 0}
            },
            'performance': {
                'write': {'avg_latency_ms': 0, 'p95_latency_ms': 0},
                'read': {'avg_latency_ms': 0, 'p95_latency_ms': 0}
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def export_config(self, format: str = 'grafana') -> str:
        """
        Export widget configuration for external dashboard systems.
        
        Args:
            format: Export format ('grafana', 'json', 'prometheus')
            
        Returns:
            Configuration string in requested format
        """
        panels = await self.render()
        
        if format == 'grafana':
            # Convert to Grafana dashboard JSON
            grafana_config = {
                'dashboard': {
                    'title': 'Archive System Monitoring',
                    'panels': self._convert_to_grafana_panels(panels['panels']),
                    'refresh': f"{self.refresh_interval}s",
                    'time': {'from': 'now-1h', 'to': 'now'},
                    'timezone': 'browser',
                    'schemaVersion': 30
                }
            }
            return json.dumps(grafana_config, indent=2)
        
        elif format == 'prometheus':
            # Generate Prometheus queries
            queries = [
                'archive_storage_total_files',
                'archive_storage_size_mb',
                'archive_compression_ratio',
                'archive_operations_total{operation="write"}',
                'archive_operations_total{operation="read"}',
                'archive_latency_ms{operation="write",quantile="0.5"}',
                'archive_latency_ms{operation="read",quantile="0.5"}'
            ]
            return '\n'.join(queries)
        
        else:
            # Default JSON format
            return json.dumps(panels, indent=2, default=str)
    
    def _convert_to_grafana_panels(self, panels: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert internal panel format to Grafana format."""
        grafana_panels = []
        panel_id = 1
        
        for panel_name, panel_config in panels.items():
            grafana_panel = {
                'id': panel_id,
                'title': panel_config.get('title', panel_name),
                'type': self._map_panel_type(panel_config['type']),
                'datasource': 'Prometheus',
                'gridPos': panel_config.get('gridPos', {'h': 8, 'w': 12, 'x': 0, 'y': 0})
            }
            
            # Add type-specific configuration
            if panel_config['type'] == 'graph':
                grafana_panel['targets'] = [
                    {
                        'expr': f"archive_{target['name'].lower().replace(' ', '_')}",
                        'legendFormat': target['name']
                    }
                    for target in panel_config.get('targets', [])
                ]
            
            grafana_panels.append(grafana_panel)
            panel_id += 1
        
        return grafana_panels
    
    def _map_panel_type(self, internal_type: str) -> str:
        """Map internal panel type to Grafana panel type."""
        type_map = {
            'stat': 'stat',
            'graph': 'timeseries',
            'gauge': 'gauge',
            'piechart': 'piechart',
            'table': 'table'
        }
        return type_map.get(internal_type, 'graph')
    
    async def _enrich_with_unified_metrics(self):
        """Enrich cached metrics with data from UnifiedMetrics."""
        if not self.unified_metrics:
            return
        
        try:
            # Get recent archive metrics from UnifiedMetrics
            archive_metrics = await self.unified_metrics.aggregate_metrics(
                names=[
                    'archive_storage_total_files',
                    'archive_storage_size_mb',
                    'archive_compression_ratio',
                    'archive_operations_total',
                    'archive_operation_duration_ms',
                    'archive_throughput_records_per_sec'
                ],
                aggregation='avg',
                period_minutes=60
            )
            
            # Update cached metrics with UnifiedMetrics data
            if archive_metrics:
                unified_data = {}
                for metric in archive_metrics:
                    unified_data[metric.name] = {
                        'value': metric.value,
                        'samples': metric.sample_count
                    }
                
                self._cached_metrics['unified'] = unified_data
                
                # Update storage metrics if newer data available
                if 'archive_storage_total_files' in unified_data:
                    latest_files = unified_data['archive_storage_total_files']['value']
                    if latest_files > 0:
                        self._cached_metrics['storage']['total_files'] = int(latest_files)
                
                if 'archive_storage_size_mb' in unified_data:
                    latest_size = unified_data['archive_storage_size_mb']['value']
                    if latest_size > 0:
                        self._cached_metrics['storage']['total_size_mb'] = latest_size
                
                if 'archive_compression_ratio' in unified_data:
                    latest_ratio = unified_data['archive_compression_ratio']['value']
                    if latest_ratio > 0:
                        self._cached_metrics['storage']['compression_ratio'] = latest_ratio
            
            # Get recent alerts from UnifiedMetrics
            active_alerts = await self.unified_metrics.get_active_alerts()
            archive_alerts = [
                alert for alert in active_alerts 
                if 'archive' in alert.metric_name.lower()
            ]
            
            if archive_alerts:
                self._cached_metrics['unified_alerts'] = [
                    {
                        'metric': alert.metric_name,
                        'severity': alert.severity.value,
                        'threshold': alert.threshold,
                        'message': alert.message,
                        'triggered_at': alert.triggered_at.isoformat()
                    }
                    for alert in archive_alerts
                ]
            
        except Exception as e:
            logger.error(f"Failed to enrich with UnifiedMetrics: {e}")