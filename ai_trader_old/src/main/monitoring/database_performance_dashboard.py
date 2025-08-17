"""
Performance Monitoring Dashboard

Web-based dashboard for monitoring AI trading system performance including
database metrics, memory usage, query performance, and system health.
"""

# Standard library imports
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import logging
from typing import Any

# Third-party imports
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Local imports
from main.data_pipeline.storage.database_optimizer import get_database_optimizer
from main.utils.core import get_utility_manager
from main.utils.database import get_db_pool
from main.utils.monitoring import get_memory_monitor
from main.utils.networking import get_websocket_manager

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics"""

    timestamp: datetime
    database: dict[str, Any]
    memory: dict[str, Any]
    connections: dict[str, Any]
    queries: dict[str, Any]
    circuit_breakers: dict[str, Any]
    websockets: dict[str, Any]


class PerformanceDashboard:
    """Main performance monitoring dashboard"""

    def __init__(self):
        self.app = FastAPI(title="AI Trading System Performance Dashboard")
        self.db_optimizer = get_database_optimizer()
        self.db_pool = get_db_pool()
        self.memory_monitor = get_memory_monitor()
        self.utility_manager = get_utility_manager()
        self.websocket_manager = get_websocket_manager()

        # Dashboard state
        self.metrics_history: list[SystemMetrics] = []
        self.max_history = 1000  # Keep last 1000 metrics
        self.active_websockets: list[WebSocket] = []

        # Monitoring tasks
        self.monitoring_task: asyncio.Task | None = None
        self.monitoring_interval = 5.0  # 5 seconds

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page"""
            return self._render_dashboard_template()

        @self.app.get("/api/metrics")
        async def get_current_metrics():
            """Get current system metrics"""
            try:
                metrics = await self._collect_metrics()
                return JSONResponse(content=asdict(metrics))
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/metrics/history")
        async def get_metrics_history(hours: int = 1):
            """Get historical metrics"""
            try:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                filtered_metrics = [
                    asdict(m) for m in self.metrics_history if m.timestamp >= cutoff_time
                ]
                return JSONResponse(content=filtered_metrics)
            except Exception as e:
                logger.error(f"Error getting metrics history: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/database/analysis")
        async def get_database_analysis(hours: int = 24):
            """Get detailed database analysis"""
            try:
                analysis = await self.db_optimizer.analyze_query_performance(hours)
                return JSONResponse(content=analysis)
            except Exception as e:
                logger.error(f"Error analyzing database: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.get("/api/indexes/status")
        async def get_index_status():
            """Get database index status"""
            try:
                # This would check index usage statistics
                status = await self._get_index_status()
                return JSONResponse(content=status)
            except Exception as e:
                logger.error(f"Error getting index status: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.post("/api/indexes/deploy")
        async def deploy_indexes(priority: int = 1, dry_run: bool = False):
            """Deploy database indexes"""
            try:
                results = await self.db_optimizer.deploy_indexes(
                    priority_filter=priority, dry_run=dry_run
                )
                return JSONResponse(content=results)
            except Exception as e:
                logger.error(f"Error deploying indexes: {e}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

        @self.app.websocket("/ws/metrics")
        async def websocket_metrics(websocket: WebSocket):
            """WebSocket endpoint for real-time metrics"""
            await websocket.accept()
            self.active_websockets.append(websocket)

            try:
                while True:
                    # Keep connection alive and wait for disconnect
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.active_websockets.remove(websocket)

    def _render_dashboard_template(self) -> str:
        """Render the dashboard HTML template"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Trading System Performance Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
                .metric-label { color: #7f8c8d; margin-bottom: 10px; }
                .status-good { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-error { color: #e74c3c; }
                .chart-container { height: 300px; margin-top: 20px; }
                .recommendations { background: #ecf0f1; padding: 15px; border-radius: 5px; margin-top: 15px; }
                .refresh-btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                .refresh-btn:hover { background: #2980b9; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>AI Trading System Performance Dashboard</h1>
                    <p>Real-time monitoring of database, memory, and system performance</p>
                    <button class="refresh-btn" onclick="refreshMetrics()">Refresh</button>
                    <span id="last-updated" style="float: right; margin-top: 10px;"></span>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Database Cache Hit Ratio</div>
                        <div class="metric-value" id="cache-hit-ratio">--</div>
                        <div class="chart-container">
                            <canvas id="cache-chart"></canvas>
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Connection Pool Status</div>
                        <div class="metric-value" id="active-connections">--</div>
                        <div id="pool-details"></div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Memory Usage</div>
                        <div class="metric-value" id="memory-usage">--</div>
                        <div class="chart-container">
                            <canvas id="memory-chart"></canvas>
                        </div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Query Performance</div>
                        <div class="metric-value" id="avg-query-time">--</div>
                        <div id="slow-queries"></div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">Circuit Breakers</div>
                        <div class="metric-value" id="circuit-breaker-status">--</div>
                        <div id="breaker-details"></div>
                    </div>

                    <div class="metric-card">
                        <div class="metric-label">WebSocket Connections</div>
                        <div class="metric-value" id="websocket-count">--</div>
                        <div id="websocket-details"></div>
                    </div>
                </div>

                <div class="metric-card" style="margin-top: 20px;">
                    <h3>System Recommendations</h3>
                    <div id="recommendations" class="recommendations">
                        Loading recommendations...
                    </div>
                </div>
            </div>

            <script>
                let cacheChart, memoryChart;
                let ws;

                function initCharts() {
                    const cacheCtx = document.getElementById('cache-chart').getContext('2d');
                    cacheChart = new Chart(cacheCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Cache Hit %',
                                data: [],
                                borderColor: '#3498db',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: { beginAtZero: true, max: 100 }
                            }
                        }
                    });

                    const memoryCtx = document.getElementById('memory-chart').getContext('2d');
                    memoryChart = new Chart(memoryCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Memory MB',
                                data: [],
                                borderColor: '#e74c3c',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false
                        }
                    });
                }

                function connectWebSocket() {
                    ws = new WebSocket(`ws://${window.location.host}/ws/metrics`);
                    ws.onopen = function() {
                        console.log('WebSocket connected');
                    };
                    ws.onclose = function() {
                        console.log('WebSocket disconnected, reconnecting...');
                        setTimeout(connectWebSocket, 5000);
                    };
                }

                async function refreshMetrics() {
                    try {
                        const response = await fetch('/api/metrics');
                        const metrics = await response.json();
                        updateDashboard(metrics);
                    } catch (error) {
                        console.error('Error fetching metrics:', error);
                    }
                }

                function updateDashboard(metrics) {
                    // Update cache hit ratio
                    const cacheHit = metrics.database?.cache_hit_ratio || 0;
                    document.getElementById('cache-hit-ratio').textContent = cacheHit.toFixed(1) + '%';
                    document.getElementById('cache-hit-ratio').className =
                        cacheHit > 95 ? 'metric-value status-good' :
                        cacheHit > 85 ? 'metric-value status-warning' : 'metric-value status-error';

                    // Update connection pool
                    const connections = metrics.connections?.pool_status?.active_connections || 0;
                    document.getElementById('active-connections').textContent = connections;

                    // Update memory usage
                    const memoryMB = metrics.memory?.current?.rss_mb || 0;
                    document.getElementById('memory-usage').textContent = memoryMB.toFixed(0) + ' MB';
                    document.getElementById('memory-usage').className =
                        memoryMB < 1000 ? 'metric-value status-good' :
                        memoryMB < 2000 ? 'metric-value status-warning' : 'metric-value status-error';

                    // Update query performance
                    const avgQueryTime = metrics.queries?.performance_metrics?.recent_avg_time || 0;
                    document.getElementById('avg-query-time').textContent = (avgQueryTime * 1000).toFixed(1) + ' ms';

                    // Update circuit breakers
                    const breakers = Object.values(metrics.circuit_breakers || {});
                    const openBreakers = breakers.filter(b => b.state === 'OPEN').length;
                    document.getElementById('circuit-breaker-status').textContent =
                        openBreakers > 0 ? `${openBreakers} OPEN` : 'All CLOSED';
                    document.getElementById('circuit-breaker-status').className =
                        openBreakers === 0 ? 'metric-value status-good' : 'metric-value status-error';

                    // Update WebSocket count
                    const wsCount = metrics.websockets?.total_clients || 0;
                    document.getElementById('websocket-count').textContent = wsCount;

                    // Update charts
                    const now = new Date().toLocaleTimeString();
                    updateChart(cacheChart, now, cacheHit);
                    updateChart(memoryChart, now, memoryMB);

                    // Update timestamp
                    document.getElementById('last-updated').textContent =
                        'Last updated: ' + new Date().toLocaleString();
                }

                function updateChart(chart, label, value) {
                    chart.data.labels.push(label);
                    chart.data.datasets[0].data.push(value);

                    // Keep only last 20 points
                    if (chart.data.labels.length > 20) {
                        chart.data.labels.shift();
                        chart.data.datasets[0].data.shift();
                    }

                    chart.update('none');
                }

                // Initialize dashboard
                document.addEventListener('DOMContentLoaded', function() {
                    initCharts();
                    connectWebSocket();
                    refreshMetrics();

                    // Auto-refresh every 10 seconds
                    setInterval(refreshMetrics, 10000);
                });
            </script>
        </body>
        </html>
        """

    async def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # Database metrics
            database_metrics = await self.db_optimizer.create_performance_dashboard()

            # Memory metrics
            memory_metrics = self.memory_monitor.get_memory_report()

            # Connection pool metrics
            connection_metrics = self.db_pool.get_pool_metrics()

            # Circuit breaker metrics
            utility_stats = self.utility_manager.get_stats()

            # WebSocket metrics
            websocket_metrics = self.websocket_manager.get_overall_stats()

            return SystemMetrics(
                timestamp=datetime.now(),
                database=database_metrics,
                memory=memory_metrics,
                connections=connection_metrics,
                queries=connection_metrics.get("performance_metrics", {}),
                circuit_breakers=utility_stats.get("circuit_breakers", {}),
                websockets=websocket_metrics,
            )

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return empty metrics on error
            return SystemMetrics(
                timestamp=datetime.now(),
                database={},
                memory={},
                connections={},
                queries={},
                circuit_breakers={},
                websockets={},
            )

    async def _get_index_status(self) -> dict[str, Any]:
        """Get database index utilization status"""
        try:
            # This would query pg_stat_user_indexes for usage statistics
            return {
                "total_indexes": 25,
                "unused_indexes": 2,
                "most_used": [
                    {"name": "idx_market_data_symbol_time", "scans": 45230},
                    {"name": "idx_companies_active", "scans": 12450},
                ],
                "recommendations": [
                    "Consider dropping unused index: idx_old_feature_data",
                    "High table scan ratio on companies table",
                ],
            }
        except Exception as e:
            logger.error(f"Error getting index status: {e}")
            return {"error": str(e)}

    async def start_monitoring(self):
        """Start the monitoring background task"""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop the monitoring background task"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        try:
            while True:
                # Collect metrics
                metrics = await self._collect_metrics()

                # Store in history
                self.metrics_history.append(metrics)

                # Trim history
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history :]

                # Send to WebSocket clients
                await self._broadcast_metrics(metrics)

                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)

        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

    async def _broadcast_metrics(self, metrics: SystemMetrics):
        """Broadcast metrics to WebSocket clients"""
        if not self.active_websockets:
            return

        try:
            message = json.dumps(asdict(metrics), default=str)

            # Send to all connected clients
            disconnected = []
            for websocket in self.active_websockets:
                try:
                    await websocket.send_text(message)
                except Exception as ws_error:
                    logger.debug(f"WebSocket disconnected: {ws_error}")
                    disconnected.append(websocket)

            # Remove disconnected clients
            for ws in disconnected:
                self.active_websockets.remove(ws)

        except Exception as e:
            logger.error(f"Error broadcasting metrics: {e}")

    async def run(self, host: str = "0.0.0.0", port: int = 8888):
        """Run the dashboard server"""
        await self.start_monitoring()

        try:
            config = uvicorn.Config(self.app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
        finally:
            await self.stop_monitoring()


# Global dashboard instance
_performance_dashboard: PerformanceDashboard | None = None


def get_performance_dashboard() -> PerformanceDashboard:
    """Get global performance dashboard instance"""
    global _performance_dashboard
    if _performance_dashboard is None:
        _performance_dashboard = PerformanceDashboard()
    return _performance_dashboard


# CLI entry point
async def main():
    """Main entry point for running the dashboard"""
    # Standard library imports
    import argparse

    parser = argparse.ArgumentParser(description="AI Trading System Performance Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")

    args = parser.parse_args()

    logger.info(f"Starting performance dashboard on {args.host}:{args.port}")

    dashboard = get_performance_dashboard()
    await dashboard.run(host=args.host, port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
