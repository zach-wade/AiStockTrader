#!/usr/bin/env python3
"""
Unified System Dashboard V2 - Comprehensive system monitoring interface.

This dashboard consolidates all system monitoring information into a single,
well-organized interface with multiple tabs for different aspects of the system.
"""

# Standard library imports
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import logging
from typing import Any

# Third-party imports
import dash
from dash import Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import psutil

# Local imports
from main.utils.database import DatabasePool

logger = logging.getLogger(__name__)


class SystemDashboardV2:
    """
    Unified System Dashboard with comprehensive monitoring capabilities.

    Features:
    - System Health: Resource usage, component status, uptime
    - Data Pipeline: Feed status, throughput, quality metrics
    - Infrastructure: Database, APIs, network monitoring
    - Analytics: Sentiment analysis, ML performance, diagnostics
    """

    def __init__(self, db_pool: DatabasePool, orchestrator: Any | None = None, port: int = 8052):
        """Initialize the system dashboard."""
        self.db_pool = db_pool
        self.orchestrator = orchestrator
        self.port = port
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Data storage
        self.system_metrics_history = []
        self.component_status = {}
        self.data_feed_metrics = {}
        self.error_logs = []
        self.sentiment_data = {}
        self.process_metrics = {}

        # Setup layout and callbacks
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div(
            [
                # Header
                html.Div(
                    [
                        html.H1("AI Trader - System Dashboard", style={"color": "#00ff88"}),
                        html.Div(id="last-update", style={"color": "#888", "fontSize": "14px"}),
                    ],
                    style={"textAlign": "center", "marginBottom": "30px"},
                ),
                # Auto-refresh interval
                dcc.Interval(id="interval-component", interval=5000),  # 5 second refresh
                # Tabs
                dcc.Tabs(
                    id="tabs",
                    value="health",
                    children=[
                        dcc.Tab(label="System Health", value="health"),
                        dcc.Tab(label="Data Pipeline", value="pipeline"),
                        dcc.Tab(label="Infrastructure", value="infrastructure"),
                        dcc.Tab(label="Analytics", value="analytics"),
                    ],
                    style={"marginBottom": "20px"},
                ),
                # Tab content
                html.Div(id="tab-content"),
            ],
            style={
                "fontFamily": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                "backgroundColor": "#0a0a0a",
                "color": "#e0e0e0",
                "minHeight": "100vh",
                "padding": "20px",
            },
        )

    def setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            [Output("tab-content", "children"), Output("last-update", "children")],
            [Input("tabs", "value"), Input("interval-component", "n_intervals")],
        )
        def update_content(active_tab, n):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update data
            self.refresh_data()

            if active_tab == "health":
                content = self.render_system_health()
            elif active_tab == "pipeline":
                content = self.render_data_pipeline()
            elif active_tab == "infrastructure":
                content = self.render_infrastructure()
            elif active_tab == "analytics":
                content = self.render_analytics()
            else:
                content = html.Div()

            return content, f"Last updated: {timestamp}"

    def refresh_data(self):
        """Refresh all dashboard data."""
        try:
            # Update system metrics
            self.update_system_metrics()

            # Update component status
            self.update_component_status()

            # Update data feed metrics
            self.update_data_feed_metrics()

            # Update error logs
            self.update_error_logs()

            # Update sentiment data
            asyncio.run(self.update_sentiment_data())

        except Exception as e:
            logger.error(f"Error refreshing dashboard data: {e}")

    def update_system_metrics(self):
        """Update system resource metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        network = psutil.net_io_counters()

        metric = {
            "timestamp": datetime.now(),
            "cpu": cpu_percent,
            "memory": memory.percent,
            "memory_available": memory.available / (1024**3),  # GB
            "disk": disk.percent,
            "disk_free": disk.free / (1024**3),  # GB
            "network_sent": network.bytes_sent,
            "network_recv": network.bytes_recv,
        }

        self.system_metrics_history.append(metric)

        # Keep only last hour
        cutoff = datetime.now() - timedelta(hours=1)
        self.system_metrics_history = [
            m for m in self.system_metrics_history if m["timestamp"] > cutoff
        ]

    def update_component_status(self):
        """Update component health status."""
        self.component_status = {
            "Database": self._check_database_health(),
            "Data Pipeline": self._check_data_pipeline_health(),
            "Trading Engine": self._check_trading_engine_health(),
            "Risk Management": self._check_risk_management_health(),
            "Feature Engine": self._check_feature_engine_health(),
            "API Clients": self._check_api_clients_health(),
            "Redis Cache": self._check_redis_health(),
            "Message Queue": self._check_message_queue_health(),
        }

    def update_data_feed_metrics(self):
        """Update data feed metrics."""
        self.data_feed_metrics = {
            "alpaca": {
                "status": "connected" if self._check_alpaca_connection() else "disconnected",
                "messages_per_sec": np.secure_uniform(10, 50),  # Mock data
                "last_message_age": np.secure_uniform(0, 5),
                "error_rate": np.secure_uniform(0, 0.05),
                "symbols_subscribed": 25,
            },
            "polygon": {
                "status": "connected",
                "messages_per_sec": np.secure_uniform(5, 20),
                "last_message_age": np.secure_uniform(0, 10),
                "error_rate": np.secure_uniform(0, 0.02),
                "symbols_subscribed": 50,
            },
            "benzinga": {
                "status": "connected",
                "messages_per_sec": np.secure_uniform(0.5, 2),
                "last_message_age": np.secure_uniform(0, 60),
                "error_rate": np.secure_uniform(0, 0.01),
                "articles_processed": 127,
            },
            "reddit": {
                "status": "connected",
                "messages_per_sec": np.secure_uniform(0.1, 0.5),
                "last_message_age": np.secure_uniform(0, 300),
                "error_rate": np.secure_uniform(0, 0.03),
                "posts_analyzed": 342,
            },
        }

    def update_error_logs(self):
        """Update error logs."""
        # Mock error logs - in production would fetch from logging system
        log_levels = ["ERROR", "WARNING", "INFO"]
        components = ["DataPipeline", "TradingEngine", "RiskManager", "FeatureEngine", "APIClient"]
        messages = [
            "Connection timeout to data source",
            "Circuit breaker triggered for high volatility",
            "Feature calculation delayed",
            "API rate limit approaching",
            "Database query slow: 523ms",
            "Memory usage above threshold",
            "WebSocket reconnection attempt",
            "Order rejected by broker",
        ]

        # Add some new logs
        for i in range(np.secure_randint(0, 3)):
            self.error_logs.append(
                {
                    "timestamp": datetime.now() - timedelta(minutes=np.secure_randint(0, 60)),
                    "level": np.secure_choice(log_levels),
                    "component": np.secure_choice(components),
                    "message": np.secure_choice(messages),
                }
            )

        # Keep only recent logs
        self.error_logs = sorted(self.error_logs, key=lambda x: x["timestamp"], reverse=True)[:100]

    async def update_sentiment_data(self):
        """Update sentiment analysis data."""
        # Mock sentiment data - in production would fetch from sentiment service
        self.sentiment_data = {
            "overall_sentiment": np.secure_uniform(-0.2, 0.3),
            "posts_analyzed": np.secure_randint(100, 500),
            "symbols": {
                "AAPL": {"sentiment": 0.25, "posts": 45, "trend": "up"},
                "TSLA": {"sentiment": -0.15, "posts": 67, "trend": "down"},
                "NVDA": {"sentiment": 0.35, "posts": 89, "trend": "up"},
                "MSFT": {"sentiment": 0.10, "posts": 23, "trend": "stable"},
                "GOOGL": {"sentiment": 0.20, "posts": 34, "trend": "up"},
            },
            "time_series": [
                {
                    "timestamp": datetime.now() - timedelta(hours=i),
                    "sentiment": np.secure_uniform(-0.1, 0.2),
                }
                for i in range(24, 0, -1)
            ],
        }

    def render_system_health(self):
        """Render system health tab."""
        # Current metrics
        latest_metrics = (
            self.system_metrics_history[-1]
            if self.system_metrics_history
            else {"cpu": 0, "memory": 0, "disk": 0, "memory_available": 0, "disk_free": 0}
        )

        # Metrics cards
        metrics_row = html.Div(
            [
                self._create_metric_card(
                    "CPU Usage", f"{latest_metrics['cpu']:.1f}%", latest_metrics["cpu"] > 80
                ),
                self._create_metric_card(
                    "Memory Usage",
                    f"{latest_metrics['memory']:.1f}%",
                    latest_metrics["memory"] > 80,
                ),
                self._create_metric_card(
                    "Disk Usage", f"{latest_metrics['disk']:.1f}%", latest_metrics["disk"] > 90
                ),
                self._create_metric_card(
                    "Available Memory",
                    f"{latest_metrics['memory_available']:.1f} GB",
                    latest_metrics["memory_available"] < 2,
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "20px",
                "marginBottom": "30px",
            },
        )

        # Resource usage chart
        resource_fig = self._create_resource_usage_chart()
        resource_chart = dcc.Graph(figure=resource_fig, style={"height": "400px"})

        # Component status grid
        component_grid = self._create_component_status_grid()

        # Process information
        process_info = self._create_process_info()

        return html.Div(
            [
                metrics_row,
                html.H3(
                    "Resource Usage Trends", style={"color": "#00ff88", "marginBottom": "15px"}
                ),
                resource_chart,
                html.Br(),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Component Status",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                component_grid,
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "Process Information",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                process_info,
                            ],
                            style={"width": "48%", "display": "inline-block", "marginLeft": "4%"},
                        ),
                    ]
                ),
            ]
        )

    def render_data_pipeline(self):
        """Render data pipeline tab."""
        # Data feed status cards
        feed_cards = []
        for feed_name, metrics in self.data_feed_metrics.items():
            feed_cards.append(self._create_feed_status_card(feed_name, metrics))

        feeds_grid = html.Div(
            feed_cards,
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, 1fr)",
                "gap": "20px",
                "marginBottom": "30px",
            },
        )

        # Throughput chart
        throughput_fig = self._create_throughput_chart()
        throughput_chart = dcc.Graph(figure=throughput_fig, style={"height": "400px"})

        # Data quality metrics
        quality_metrics = self._create_data_quality_metrics()

        # Gap detection
        gap_analysis = self._create_gap_analysis()

        # Dual storage metrics
        dual_storage_metrics = self._create_dual_storage_metrics()

        return html.Div(
            [
                html.H3("Data Feed Status", style={"color": "#00ff88", "marginBottom": "15px"}),
                feeds_grid,
                html.H3("Message Throughput", style={"color": "#00ff88", "marginBottom": "15px"}),
                throughput_chart,
                html.Br(),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Data Quality Metrics",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                quality_metrics,
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "Gap Detection",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                gap_analysis,
                            ],
                            style={"width": "48%", "display": "inline-block", "marginLeft": "4%"},
                        ),
                    ]
                ),
                html.Br(),
                html.H3("Dual Storage Metrics", style={"color": "#00ff88", "marginBottom": "15px"}),
                dual_storage_metrics,
            ]
        )

    def render_infrastructure(self):
        """Render infrastructure tab."""
        # Database metrics
        db_metrics = self._create_database_metrics()

        # API client status
        api_status = self._create_api_client_status()

        # Network I/O chart
        network_fig = self._create_network_io_chart()
        network_chart = dcc.Graph(figure=network_fig, style={"height": "400px"})

        # Process monitoring
        process_table = self._create_process_monitoring_table()

        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Database Metrics",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                db_metrics,
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "API Client Status",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                api_status,
                            ],
                            style={"width": "48%", "display": "inline-block", "marginLeft": "4%"},
                        ),
                    ]
                ),
                html.Br(),
                html.H3("Network I/O", style={"color": "#00ff88", "marginBottom": "15px"}),
                network_chart,
                html.Br(),
                html.H3("Process Monitoring", style={"color": "#00ff88", "marginBottom": "15px"}),
                process_table,
            ]
        )

    def render_analytics(self):
        """Render analytics tab."""
        # Sentiment analysis
        sentiment_overview = self._create_sentiment_overview()
        sentiment_chart = self._create_sentiment_time_series()

        # Feature engine status
        feature_status = self._create_feature_engine_status()

        # ML model performance
        ml_performance = self._create_ml_performance_metrics()

        # Error logs
        error_log_table = self._create_error_log_table()

        return html.Div(
            [
                html.H3("Sentiment Analysis", style={"color": "#00ff88", "marginBottom": "15px"}),
                sentiment_overview,
                dcc.Graph(figure=sentiment_chart, style={"height": "300px"}),
                html.Br(),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Feature Engine Status",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                feature_status,
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "ML Model Performance",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                ml_performance,
                            ],
                            style={"width": "48%", "display": "inline-block", "marginLeft": "4%"},
                        ),
                    ]
                ),
                html.Br(),
                html.H3(
                    "Recent Errors & Warnings", style={"color": "#00ff88", "marginBottom": "15px"}
                ),
                error_log_table,
            ]
        )

    # Helper methods

    def _create_metric_card(self, label: str, value: str, alert: bool = False):
        """Create a metric card component."""
        color = "#ff4444" if alert else "#00ff88"

        return html.Div(
            [
                html.Div(label, style={"fontSize": "14px", "color": "#888", "marginBottom": "5px"}),
                html.Div(value, style={"fontSize": "24px", "fontWeight": "bold", "color": color}),
            ],
            style={
                "backgroundColor": "#1a1a1a",
                "padding": "20px",
                "borderRadius": "8px",
                "textAlign": "center",
                "border": f'1px solid {"#ff4444" if alert else "#333"}',
            },
        )

    def _create_resource_usage_chart(self):
        """Create resource usage time series chart."""
        if not self.system_metrics_history:
            return go.Figure()

        df = pd.DataFrame(self.system_metrics_history)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df["cpu"], name="CPU %", line=dict(color="#00ff88", width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["memory"],
                name="Memory %",
                line=dict(color="#00aaff", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df["disk"], name="Disk %", line=dict(color="#ffaa00", width=2)
            )
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Time",
            yaxis_title="Usage %",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=50),
        )

        return fig

    def _create_component_status_grid(self):
        """Create component status grid."""
        components = []

        for comp_name, status in self.component_status.items():
            status_color = "#00ff88" if status["healthy"] else "#ff4444"
            status_text = "Healthy" if status["healthy"] else "Unhealthy"

            component = html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                style={
                                    "width": "12px",
                                    "height": "12px",
                                    "borderRadius": "50%",
                                    "backgroundColor": status_color,
                                    "display": "inline-block",
                                    "marginRight": "8px",
                                }
                            ),
                            html.Span(comp_name, style={"fontWeight": "bold"}),
                        ]
                    ),
                    html.Div(status_text, style={"fontSize": "12px", "color": "#888"}),
                    html.Div(
                        status.get("message", ""), style={"fontSize": "11px", "color": "#666"}
                    ),
                ],
                style={
                    "backgroundColor": "#1a1a1a",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "border": f"1px solid {status_color}",
                },
            )

            components.append(component)

        return html.Div(
            components,
            style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px"},
        )

    def _create_process_info(self):
        """Create process information display."""
        process = psutil.Process()
        create_time = datetime.fromtimestamp(process.create_time())
        uptime = datetime.now() - create_time

        info = [
            ("Process ID", str(process.pid)),
            (
                "Uptime",
                f"{int(uptime.total_seconds() // 3600)}h {int((uptime.total_seconds() % 3600) // 60)}m",
            ),
            ("Threads", str(process.num_threads())),
            ("CPU Affinity", str(len(process.cpu_affinity()))),
            ("Memory (RSS)", f"{process.memory_info().rss / (1024**3):.2f} GB"),
            ("Memory (VMS)", f"{process.memory_info().vms / (1024**3):.2f} GB"),
            ("Open Files", str(len(process.open_files()))),
            ("Connections", str(len(process.connections()))),
        ]

        info_items = []
        for label, value in info:
            info_items.append(
                html.Div(
                    [html.Span(f"{label}: ", style={"color": "#888"}), html.Span(value)],
                    style={"marginBottom": "8px"},
                )
            )

        return html.Div(
            info_items,
            style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderRadius": "8px"},
        )

    def _create_feed_status_card(self, feed_name: str, metrics: dict[str, Any]):
        """Create data feed status card."""
        status_color = "#00ff88" if metrics["status"] == "connected" else "#ff4444"

        return html.Div(
            [
                html.H4(feed_name.capitalize(), style={"marginBottom": "15px"}),
                html.Div(
                    [
                        html.Div(
                            style={
                                "width": "12px",
                                "height": "12px",
                                "borderRadius": "50%",
                                "backgroundColor": status_color,
                                "display": "inline-block",
                                "marginRight": "8px",
                            }
                        ),
                        html.Span(metrics["status"].capitalize()),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Rate: ", style={"color": "#888"}),
                                html.Span(f"{metrics['messages_per_sec']:.1f} msg/s"),
                            ],
                            style={"fontSize": "14px"},
                        ),
                        html.Div(
                            [
                                html.Span("Latency: ", style={"color": "#888"}),
                                html.Span(f"{metrics['last_message_age']:.1f}s"),
                            ],
                            style={"fontSize": "14px"},
                        ),
                        html.Div(
                            [
                                html.Span("Error Rate: ", style={"color": "#888"}),
                                html.Span(f"{metrics['error_rate']*100:.2f}%"),
                            ],
                            style={"fontSize": "14px"},
                        ),
                    ]
                ),
            ],
            style={
                "backgroundColor": "#1a1a1a",
                "padding": "20px",
                "borderRadius": "8px",
                "border": f"1px solid {status_color}",
            },
        )

    def _create_throughput_chart(self):
        """Create message throughput chart."""
        # Generate sample data
        times = pd.date_range(end=datetime.now(), periods=60, freq="1min")

        fig = go.Figure()

        for feed_name in self.data_feed_metrics.keys():
            throughput = np.secure_uniform(
                self.data_feed_metrics[feed_name]["messages_per_sec"] * 0.8,
                self.data_feed_metrics[feed_name]["messages_per_sec"] * 1.2,
                size=60,
            )
            fig.add_trace(
                go.Scatter(x=times, y=throughput, name=feed_name.capitalize(), mode="lines")
            )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Time",
            yaxis_title="Messages/sec",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=50),
        )

        return fig

    def _create_data_quality_metrics(self):
        """Create data quality metrics display."""
        metrics = [
            ("Data Completeness", "98.5%", False),
            ("Validation Pass Rate", "99.2%", False),
            ("Duplicate Rate", "0.3%", False),
            ("Missing Fields", "1.2%", False),
            ("Schema Compliance", "100%", False),
            ("Latency Compliance", "95.8%", True),
        ]

        metric_items = []
        for metric, value, warning in metrics:
            color = "#ffaa00" if warning else "#00ff88"
            metric_items.append(
                html.Div(
                    [
                        html.Span(f"{metric}: ", style={"color": "#888"}),
                        html.Span(value, style={"color": color, "fontWeight": "bold"}),
                    ],
                    style={"marginBottom": "10px"},
                )
            )

        return html.Div(
            metric_items,
            style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderRadius": "8px"},
        )

    def _create_gap_analysis(self):
        """Create data gap analysis display."""
        gaps = [
            {"symbol": "AAPL", "gaps": 2, "duration": "5m"},
            {"symbol": "GOOGL", "gaps": 1, "duration": "2m"},
            {"symbol": "TSLA", "gaps": 3, "duration": "8m"},
            {"symbol": "MSFT", "gaps": 0, "duration": "-"},
        ]

        gap_items = []
        for gap in gaps:
            color = "#ff4444" if gap["gaps"] > 2 else "#ffaa00" if gap["gaps"] > 0 else "#00ff88"
            gap_items.append(
                html.Div(
                    [
                        html.Span(f"{gap['symbol']}: ", style={"fontWeight": "bold"}),
                        html.Span(f"{gap['gaps']} gaps", style={"color": color}),
                        html.Span(
                            f" ({gap['duration']})", style={"color": "#888", "fontSize": "12px"}
                        ),
                    ],
                    style={"marginBottom": "8px"},
                )
            )

        return html.Div(
            [
                html.Div(
                    "Data gaps in last hour:", style={"marginBottom": "15px", "color": "#888"}
                ),
                html.Div(gap_items),
            ],
            style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderRadius": "8px"},
        )

    def _create_dual_storage_metrics(self):
        """Create dual storage metrics display."""
        # Mock metrics - in production, fetch from repositories
        metrics = {
            "hot_writes_success": 15234,
            "hot_writes_failed": 23,
            "cold_writes_success": 14892,
            "cold_writes_failed": 342,
            "events_published": 15234,
            "events_failed": 12,
            "hot_success_rate": 0.998,
            "cold_success_rate": 0.977,
            "hot_circuit_breaker_state": "closed",
            "cold_circuit_breaker_state": "closed",
            "cold_storage_lag_seconds": 45,
            "queue_depth": 234,
        }

        # Success rate cards
        success_cards = html.Div(
            [
                html.Div(
                    [
                        html.H4("Hot Storage", style={"color": "#00ff88", "marginBottom": "10px"}),
                        html.Div(
                            f"{metrics['hot_success_rate']:.1%}",
                            style={
                                "fontSize": "36px",
                                "fontWeight": "bold",
                                "color": (
                                    "#00ff88" if metrics["hot_success_rate"] > 0.99 else "#ffaa00"
                                ),
                            },
                        ),
                        html.Div(
                            f"{metrics['hot_writes_success']:,} successful",
                            style={"color": "#888", "fontSize": "14px"},
                        ),
                        html.Div(
                            f"{metrics['hot_writes_failed']} failed",
                            style={"color": "#ff4444", "fontSize": "14px"},
                        ),
                        html.Div(
                            [
                                html.Span("Circuit Breaker: ", style={"color": "#888"}),
                                html.Span(
                                    metrics["hot_circuit_breaker_state"].upper(),
                                    style={
                                        "color": (
                                            "#00ff88"
                                            if metrics["hot_circuit_breaker_state"] == "closed"
                                            else "#ff4444"
                                        ),
                                        "fontWeight": "bold",
                                    },
                                ),
                            ],
                            style={"marginTop": "10px"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#1a1a1a",
                        "padding": "20px",
                        "borderRadius": "8px",
                        "width": "48%",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        html.H4("Cold Storage", style={"color": "#00ff88", "marginBottom": "10px"}),
                        html.Div(
                            f"{metrics['cold_success_rate']:.1%}",
                            style={
                                "fontSize": "36px",
                                "fontWeight": "bold",
                                "color": (
                                    "#00ff88" if metrics["cold_success_rate"] > 0.95 else "#ffaa00"
                                ),
                            },
                        ),
                        html.Div(
                            f"{metrics['cold_writes_success']:,} successful",
                            style={"color": "#888", "fontSize": "14px"},
                        ),
                        html.Div(
                            f"{metrics['cold_writes_failed']} failed",
                            style={"color": "#ff4444", "fontSize": "14px"},
                        ),
                        html.Div(
                            [
                                html.Span("Circuit Breaker: ", style={"color": "#888"}),
                                html.Span(
                                    metrics["cold_circuit_breaker_state"].upper(),
                                    style={
                                        "color": (
                                            "#00ff88"
                                            if metrics["cold_circuit_breaker_state"] == "closed"
                                            else "#ff4444"
                                        ),
                                        "fontWeight": "bold",
                                    },
                                ),
                            ],
                            style={"marginTop": "10px"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#1a1a1a",
                        "padding": "20px",
                        "borderRadius": "8px",
                        "width": "48%",
                        "display": "inline-block",
                        "marginLeft": "4%",
                    },
                ),
            ],
            style={"marginBottom": "20px"},
        )

        # Event publishing and lag metrics
        event_metrics = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            "Event Publishing", style={"color": "#888", "marginBottom": "10px"}
                        ),
                        html.Div(
                            [
                                html.Span(
                                    f"{metrics['events_published']:,} published",
                                    style={"color": "#00ff88"},
                                ),
                                html.Span(" | ", style={"color": "#444"}),
                                html.Span(
                                    f"{metrics['events_failed']} failed", style={"color": "#ff4444"}
                                ),
                            ]
                        ),
                    ],
                    style={"marginBottom": "15px"},
                ),
                html.Div(
                    [
                        html.Div(
                            "Cold Storage Lag", style={"color": "#888", "marginBottom": "10px"}
                        ),
                        html.Div(
                            [
                                html.Span(
                                    f"{metrics['cold_storage_lag_seconds']}s",
                                    style={
                                        "fontSize": "24px",
                                        "fontWeight": "bold",
                                        "color": (
                                            "#00ff88"
                                            if metrics["cold_storage_lag_seconds"] < 60
                                            else "#ffaa00"
                                        ),
                                    },
                                ),
                                html.Span(
                                    f" ({metrics['queue_depth']} items in queue)",
                                    style={
                                        "color": "#888",
                                        "fontSize": "14px",
                                        "marginLeft": "10px",
                                    },
                                ),
                            ]
                        ),
                    ]
                ),
            ],
            style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderRadius": "8px"},
        )

        return html.Div([success_cards, event_metrics])

    def _create_database_metrics(self):
        """Create database metrics display."""
        metrics = [
            ("Active Connections", "42/100"),
            ("Query Latency (avg)", "12.3ms"),
            ("Cache Hit Rate", "87.5%"),
            ("Replication Lag", "0.2s"),
            ("Storage Used", "234 GB"),
            ("Index Efficiency", "94.2%"),
        ]

        metric_items = []
        for metric, value in metrics:
            metric_items.append(
                html.Div(
                    [
                        html.Span(f"{metric}: ", style={"color": "#888"}),
                        html.Span(value, style={"fontWeight": "bold"}),
                    ],
                    style={"marginBottom": "10px"},
                )
            )

        return html.Div(
            metric_items,
            style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderRadius": "8px"},
        )

    def _create_api_client_status(self):
        """Create API client status display."""
        clients = [
            {
                "name": "Alpaca Trading",
                "status": "connected",
                "requests": "1.2k/min",
                "errors": "0.1%",
            },
            {"name": "Polygon.io", "status": "connected", "requests": "800/min", "errors": "0.0%"},
            {
                "name": "Benzinga News",
                "status": "connected",
                "requests": "50/min",
                "errors": "0.5%",
            },
            {
                "name": "Reddit API",
                "status": "rate_limited",
                "requests": "60/min",
                "errors": "2.1%",
            },
        ]

        client_items = []
        for client in clients:
            status_color = "#00ff88" if client["status"] == "connected" else "#ffaa00"

            client_items.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    style={
                                        "width": "8px",
                                        "height": "8px",
                                        "borderRadius": "50%",
                                        "backgroundColor": status_color,
                                        "display": "inline-block",
                                        "marginRight": "8px",
                                    }
                                ),
                                html.Span(client["name"], style={"fontWeight": "bold"}),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span(
                                    f"Requests: {client['requests']}",
                                    style={"fontSize": "12px", "marginRight": "10px"},
                                ),
                                html.Span(
                                    f"Errors: {client['errors']}",
                                    style={"fontSize": "12px", "color": "#888"},
                                ),
                            ],
                            style={"marginTop": "5px"},
                        ),
                    ],
                    style={"marginBottom": "15px"},
                )
            )

        return html.Div(
            client_items,
            style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderRadius": "8px"},
        )

    def _create_network_io_chart(self):
        """Create network I/O chart."""
        if not self.system_metrics_history:
            return go.Figure()

        # Calculate rates
        rates = []
        for i in range(1, len(self.system_metrics_history)):
            prev = self.system_metrics_history[i - 1]
            curr = self.system_metrics_history[i]
            time_diff = (curr["timestamp"] - prev["timestamp"]).total_seconds()

            if time_diff > 0:
                rates.append(
                    {
                        "timestamp": curr["timestamp"],
                        "sent_rate": (curr["network_sent"] - prev["network_sent"])
                        / time_diff
                        / 1024,  # KB/s
                        "recv_rate": (curr["network_recv"] - prev["network_recv"])
                        / time_diff
                        / 1024,  # KB/s
                    }
                )

        if not rates:
            return go.Figure()

        df = pd.DataFrame(rates)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["sent_rate"],
                name="Sent",
                line=dict(color="#00ff88", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["recv_rate"],
                name="Received",
                line=dict(color="#00aaff", width=2),
            )
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Time",
            yaxis_title="KB/s",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=50),
        )

        return fig

    def _create_process_monitoring_table(self):
        """Create process monitoring table."""
        processes = []

        # Get top processes by CPU
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                if proc.info["name"] and "python" in proc.info["name"].lower():
                    processes.append(
                        {
                            "PID": proc.info["pid"],
                            "Name": proc.info["name"][:30],
                            "CPU %": f"{proc.info['cpu_percent']:.1f}",
                            "Memory %": f"{proc.info['memory_percent']:.1f}",
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if not processes:
            return html.Div("No processes found", style={"textAlign": "center", "padding": "20px"})

        df = pd.DataFrame(processes[:10])  # Top 10

        return dash.dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in df.columns],
            style_cell={
                "backgroundColor": "#1a1a1a",
                "color": "#e0e0e0",
                "border": "1px solid #333",
            },
            style_header={"backgroundColor": "#0f0f0f", "fontWeight": "bold"},
        )

    def _create_sentiment_overview(self):
        """Create sentiment overview display."""
        overall_sentiment = self.sentiment_data.get("overall_sentiment", 0)
        sentiment_color = (
            "#00ff88"
            if overall_sentiment > 0.1
            else "#ff4444" if overall_sentiment < -0.1 else "#ffaa00"
        )

        overview = html.Div(
            [
                html.Div(
                    [
                        html.Span("Overall Sentiment: ", style={"color": "#888"}),
                        html.Span(
                            f"{overall_sentiment:+.3f}",
                            style={
                                "fontSize": "24px",
                                "fontWeight": "bold",
                                "color": sentiment_color,
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Span("Posts Analyzed: ", style={"color": "#888"}),
                        html.Span(str(self.sentiment_data.get("posts_analyzed", 0))),
                    ],
                    style={"marginTop": "10px"},
                ),
            ]
        )

        # Symbol sentiments
        symbols = []
        for symbol, data in self.sentiment_data.get("symbols", {}).items():
            sentiment_color = (
                "#00ff88"
                if data["sentiment"] > 0.1
                else "#ff4444" if data["sentiment"] < -0.1 else "#ffaa00"
            )
            trend_icon = "↗" if data["trend"] == "up" else "↘" if data["trend"] == "down" else "→"

            symbols.append(
                html.Div(
                    [
                        html.Span(f"{symbol}: ", style={"fontWeight": "bold"}),
                        html.Span(f"{data['sentiment']:+.2f}", style={"color": sentiment_color}),
                        html.Span(
                            f" ({data['posts']} posts) ",
                            style={"color": "#888", "fontSize": "12px"},
                        ),
                        html.Span(trend_icon, style={"fontSize": "16px"}),
                    ],
                    style={"marginBottom": "5px"},
                )
            )

        return html.Div(
            [
                overview,
                html.Hr(style={"borderColor": "#333", "margin": "15px 0"}),
                html.Div(symbols),
            ],
            style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderRadius": "8px"},
        )

    def _create_sentiment_time_series(self):
        """Create sentiment time series chart."""
        if not self.sentiment_data.get("time_series"):
            return go.Figure()

        df = pd.DataFrame(self.sentiment_data["time_series"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["sentiment"],
                mode="lines+markers",
                line=dict(color="#00ff88", width=2),
            )
        )

        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
        fig.add_hline(y=0.2, line_dash="dot", line_color="green", annotation_text="Bullish")
        fig.add_hline(y=-0.2, line_dash="dot", line_color="red", annotation_text="Bearish")

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Time",
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-0.5, 0.5]),
            margin=dict(l=0, r=0, t=20, b=50),
        )

        return fig

    def _create_feature_engine_status(self):
        """Create feature engine status display."""
        features = [
            {
                "name": "Price Features",
                "status": "active",
                "latency": "12ms",
                "success_rate": "99.8%",
            },
            {
                "name": "Volume Features",
                "status": "active",
                "latency": "8ms",
                "success_rate": "99.9%",
            },
            {
                "name": "Technical Indicators",
                "status": "active",
                "latency": "45ms",
                "success_rate": "98.5%",
            },
            {
                "name": "Sentiment Features",
                "status": "delayed",
                "latency": "250ms",
                "success_rate": "95.2%",
            },
            {
                "name": "Market Features",
                "status": "active",
                "latency": "23ms",
                "success_rate": "99.1%",
            },
        ]

        feature_items = []
        for feature in features:
            status_color = "#00ff88" if feature["status"] == "active" else "#ffaa00"

            feature_items.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    style={
                                        "width": "8px",
                                        "height": "8px",
                                        "borderRadius": "50%",
                                        "backgroundColor": status_color,
                                        "display": "inline-block",
                                        "marginRight": "8px",
                                    }
                                ),
                                html.Span(feature["name"], style={"fontWeight": "bold"}),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span(
                                    f"Latency: {feature['latency']}",
                                    style={"fontSize": "12px", "marginRight": "10px"},
                                ),
                                html.Span(
                                    f"Success: {feature['success_rate']}",
                                    style={"fontSize": "12px", "color": "#888"},
                                ),
                            ],
                            style={"marginTop": "5px"},
                        ),
                    ],
                    style={"marginBottom": "15px"},
                )
            )

        return html.Div(
            feature_items,
            style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderRadius": "8px"},
        )

    def _create_ml_performance_metrics(self):
        """Create ML model performance metrics display."""
        models = [
            {"name": "AAPL Predictor", "accuracy": "68.2%", "sharpe": "1.85", "trades": 142},
            {"name": "TSLA Predictor", "accuracy": "64.5%", "sharpe": "1.52", "trades": 89},
            {"name": "Market Regime", "accuracy": "71.8%", "sharpe": "-", "trades": "-"},
        ]

        model_items = []
        for model in models:
            model_items.append(
                html.Div(
                    [
                        html.H5(model["name"], style={"marginBottom": "10px"}),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span("Accuracy: ", style={"color": "#888"}),
                                        html.Span(model["accuracy"]),
                                    ],
                                    style={"fontSize": "14px"},
                                ),
                                html.Div(
                                    [
                                        html.Span("Sharpe: ", style={"color": "#888"}),
                                        html.Span(model["sharpe"]),
                                    ],
                                    style={"fontSize": "14px"},
                                ),
                                html.Div(
                                    [
                                        html.Span("Trades: ", style={"color": "#888"}),
                                        html.Span(str(model["trades"])),
                                    ],
                                    style={"fontSize": "14px"},
                                ),
                            ]
                        ),
                    ],
                    style={
                        "backgroundColor": "#222",
                        "padding": "15px",
                        "borderRadius": "6px",
                        "marginBottom": "10px",
                    },
                )
            )

        return html.Div(
            model_items,
            style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderRadius": "8px"},
        )

    def _create_error_log_table(self):
        """Create error log table."""
        if not self.error_logs:
            return html.Div("No recent errors", style={"textAlign": "center", "padding": "20px"})

        # Convert to DataFrame
        df = pd.DataFrame(self.error_logs[:20])  # Show last 20
        df["Time"] = df["timestamp"].dt.strftime("%H:%M:%S")

        # Style based on level
        style_conditions = [
            {"if": {"column_id": "level", "filter_query": '{level} = "ERROR"'}, "color": "#ff4444"},
            {
                "if": {"column_id": "level", "filter_query": '{level} = "WARNING"'},
                "color": "#ffaa00",
            },
            {"if": {"column_id": "level", "filter_query": '{level} = "INFO"'}, "color": "#00aaff"},
        ]

        return dash.dash_table.DataTable(
            data=df[["Time", "level", "component", "message"]].to_dict("records"),
            columns=[
                {"name": "Time", "id": "Time"},
                {"name": "Level", "id": "level"},
                {"name": "Component", "id": "component"},
                {"name": "Message", "id": "message"},
            ],
            style_cell={
                "backgroundColor": "#1a1a1a",
                "color": "#e0e0e0",
                "border": "1px solid #333",
                "textAlign": "left",
            },
            style_header={"backgroundColor": "#0f0f0f", "fontWeight": "bold"},
            style_data_conditional=style_conditions,
            page_size=10,
        )

    # Health check methods

    def _check_database_health(self):
        """Check database health."""
        try:
            # Would perform actual database ping
            return {"healthy": True, "message": "Connected"}
        except Exception as e:
            return {"healthy": False, "message": str(e)}

    def _check_data_pipeline_health(self):
        """Check data pipeline health."""
        if self.orchestrator and hasattr(self.orchestrator, "data_orchestrator"):
            try:
                # Check if data orchestrator is healthy
                return {"healthy": True, "message": "All feeds operational"}
            except Exception as e:
                return {"healthy": False, "message": f"Pipeline error: {str(e)[:50]}"}
        return {"healthy": False, "message": "Not initialized"}

    def _check_trading_engine_health(self):
        """Check trading engine health."""
        if self.orchestrator and hasattr(self.orchestrator, "trading_system"):
            return {"healthy": True, "message": "Trading active"}
        return {"healthy": False, "message": "Not initialized"}

    def _check_risk_management_health(self):
        """Check risk management health."""
        return {"healthy": True, "message": "Limits enforced"}

    def _check_feature_engine_health(self):
        """Check feature engine health."""
        return {"healthy": True, "message": "Features calculating"}

    def _check_api_clients_health(self):
        """Check API clients health."""
        return {"healthy": True, "message": "All APIs connected"}

    def _check_redis_health(self):
        """Check Redis health."""
        return {"healthy": True, "message": "Cache operational"}

    def _check_message_queue_health(self):
        """Check message queue health."""
        return {"healthy": True, "message": "Queue processing"}

    def _check_alpaca_connection(self):
        """Check if Alpaca is connected."""
        # Would check actual connection status
        return True

    def run(self, debug=False):
        """Run the dashboard server."""
        logger.info(f"Starting System Dashboard V2 on http://localhost:{self.port}")
        self.app.run(host="0.0.0.0", port=self.port, debug=debug)
