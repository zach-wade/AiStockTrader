"""
Metrics Exporter

Provides functionality to export monitoring metrics to various formats
and destinations including files, databases, and monitoring systems.
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Exports metrics to various formats and destinations."""
    
    def __init__(self, export_dir: Optional[str] = None):
        """
        Initialize metrics exporter.
        
        Args:
            export_dir: Directory for exported files
        """
        self.export_dir = Path(export_dir) if export_dir else Path.cwd() / 'exports'
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_json(self, metrics: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.export_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            logger.info(f"Exported metrics to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting metrics to JSON: {e}")
            raise
    
    def export_to_csv(self, metrics: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export metrics to CSV file.
        
        Args:
            metrics: Metrics dictionary
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"metrics_{timestamp}.csv"
        
        filepath = self.export_dir / filename
        
        try:
            # Flatten nested metrics
            flat_metrics = self._flatten_dict(metrics)
            
            # Convert to DataFrame for easy CSV export
            df = pd.DataFrame([flat_metrics])
            df.to_csv(filepath, index=False)
            
            logger.info(f"Exported metrics to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting metrics to CSV: {e}")
            raise
    
    def export_to_html(self, metrics: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export metrics to HTML file.
        
        Args:
            metrics: Metrics dictionary
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"metrics_{timestamp}.html"
        
        filepath = self.export_dir / filename
        
        try:
            html_content = self._generate_html_report(metrics)
            
            with open(filepath, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Exported metrics to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Error exporting metrics to HTML: {e}")
            raise
    
    def export_to_prometheus(self, metrics: Dict[str, Any]) -> str:
        """
        Export metrics in Prometheus format.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            Prometheus formatted string
        """
        lines = []
        flat_metrics = self._flatten_dict(metrics)
        
        for key, value in flat_metrics.items():
            if isinstance(value, (int, float)):
                # Convert key to Prometheus format (replace dots with underscores)
                metric_name = key.replace('.', '_').replace('-', '_')
                lines.append(f"{metric_name} {value}")
        
        return '\n'.join(lines)
    
    def export_to_influxdb(self, metrics: Dict[str, Any], measurement: str = 'ai_trader') -> Dict[str, Any]:
        """
        Format metrics for InfluxDB.
        
        Args:
            metrics: Metrics dictionary
            measurement: InfluxDB measurement name
            
        Returns:
            InfluxDB formatted data
        """
        timestamp = datetime.utcnow().isoformat()
        flat_metrics = self._flatten_dict(metrics)
        
        # Separate tags and fields
        tags = {}
        fields = {}
        
        for key, value in flat_metrics.items():
            if isinstance(value, str):
                tags[key] = value
            elif isinstance(value, (int, float)):
                fields[key] = value
        
        return {
            'measurement': measurement,
            'tags': tags,
            'fields': fields,
            'time': timestamp
        }
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to comma-separated strings or take length
                if v and isinstance(v[0], (int, float)):
                    items.append((f"{new_key}_mean", sum(v) / len(v) if v else 0))
                    items.append((f"{new_key}_count", len(v)))
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _generate_html_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate HTML report from metrics.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            HTML content
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Trader Metrics Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #333;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                }}
                .section {{
                    background-color: white;
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    text-align: left;
                    padding: 8px;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .metric-value {{
                    font-weight: bold;
                    color: #2196F3;
                }}
                .timestamp {{
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Trader Metrics Report</h1>
                <p class="timestamp">Generated: {timestamp}</p>
            </div>
        """
        
        # Group metrics by category
        categories = defaultdict(dict)
        flat_metrics = self._flatten_dict(metrics)
        
        for key, value in flat_metrics.items():
            parts = key.split('.')
            category = parts[0] if len(parts) > 1 else 'General'
            metric_name = '.'.join(parts[1:]) if len(parts) > 1 else key
            categories[category][metric_name] = value
        
        # Generate sections for each category
        for category, category_metrics in sorted(categories.items()):
            html += f"""
            <div class="section">
                <h2>{category.replace('_', ' ').title()}</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """
            
            for metric_name, value in sorted(category_metrics.items()):
                formatted_value = self._format_value(value)
                html += f"""
                    <tr>
                        <td>{metric_name.replace('_', ' ').title()}</td>
                        <td class="metric-value">{formatted_value}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _format_value(self, value: Any) -> str:
        """Format value for display."""
        if isinstance(value, float):
            return f"{value:.4f}"
        elif isinstance(value, (list, dict)):
            return json.dumps(value, indent=2)
        else:
            return str(value)


class BatchMetricsExporter:
    """Exports batches of metrics efficiently."""
    
    def __init__(self, exporter: Optional[MetricsExporter] = None):
        """
        Initialize batch exporter.
        
        Args:
            exporter: MetricsExporter instance
        """
        self.exporter = exporter or MetricsExporter()
        self.batch = []
        self.batch_size = 100
    
    def add_metrics(self, metrics: Dict[str, Any], timestamp: Optional[datetime] = None):
        """Add metrics to batch."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.batch.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
        if len(self.batch) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Export all batched metrics."""
        if not self.batch:
            return
        
        # Create DataFrame from batch
        df = pd.DataFrame(self.batch)
        
        # Export to CSV with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"batch_metrics_{timestamp}.csv"
        filepath = self.exporter.export_dir / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(self.batch)} metrics to {filepath}")
        
        # Clear batch
        self.batch = []


# Convenience functions
def export_metrics(metrics: Dict[str, Any], format: str = 'json', **kwargs) -> str:
    """
    Export metrics in specified format.
    
    Args:
        metrics: Metrics to export
        format: Export format (json, csv, html, prometheus)
        **kwargs: Additional arguments for exporter
        
    Returns:
        Path to exported file or formatted string
    """
    exporter = MetricsExporter()
    
    if format == 'json':
        return exporter.export_to_json(metrics, **kwargs)
    elif format == 'csv':
        return exporter.export_to_csv(metrics, **kwargs)
    elif format == 'html':
        return exporter.export_to_html(metrics, **kwargs)
    elif format == 'prometheus':
        return exporter.export_to_prometheus(metrics)
    else:
        raise ValueError(f"Unsupported format: {format}")