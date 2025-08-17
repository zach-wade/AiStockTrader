#!/usr/bin/env python3
"""
Unified Trading Dashboard V2 - Comprehensive trading monitoring interface.

This dashboard consolidates all trading-related information into a single,
well-organized interface with multiple tabs for different aspects of trading.
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
import plotly.express as px
import plotly.graph_objs as go

# Local imports
from main.interfaces.metrics import IMetricsRecorder
from main.utils.database import DatabasePool

logger = logging.getLogger(__name__)


class TradingDashboardV2:
    """
    Unified Trading Dashboard with comprehensive monitoring capabilities.

    Features:
    - Trading Overview: P&L, positions, activity, performance
    - Market Analysis: VIX, sectors, breadth, movers
    - Portfolio Analytics: Detailed breakdowns and risk metrics
    - Alerts & Activity: Real-time notifications and logs
    """

    def __init__(
        self,
        db_pool: DatabasePool,
        metrics_recorder: IMetricsRecorder | None = None,
        port: int = 8080,
    ):
        """Initialize the trading dashboard."""
        self.db_pool = db_pool
        self.metrics_recorder = metrics_recorder
        self.port = port
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # Data storage
        self.positions_data = []
        self.pnl_history = []
        self.market_data = {}
        self.alerts = []
        self.activity_log = []

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._loop = None

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
                        html.H1("AI Trader - Trading Dashboard", style={"color": "#00ff88"}),
                        html.Div(id="last-update", style={"color": "#888", "fontSize": "14px"}),
                    ],
                    style={"textAlign": "center", "marginBottom": "30px"},
                ),
                # Auto-refresh interval
                dcc.Interval(id="interval-component", interval=5000),  # 5 second refresh
                # Tabs
                dcc.Tabs(
                    id="tabs",
                    value="overview",
                    children=[
                        dcc.Tab(label="Trading Overview", value="overview"),
                        dcc.Tab(label="Market Analysis", value="market"),
                        dcc.Tab(label="Portfolio Analytics", value="portfolio"),
                        dcc.Tab(label="Alerts & Activity", value="alerts"),
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

            # Update data in thread pool to avoid event loop conflicts
            self._run_async(self.refresh_data())

            if active_tab == "overview":
                content = self.render_trading_overview()
            elif active_tab == "market":
                content = self.render_market_analysis()
            elif active_tab == "portfolio":
                content = self.render_portfolio_analytics()
            elif active_tab == "alerts":
                content = self.render_alerts_activity()
            else:
                content = html.Div()

            return content, f"Last updated: {timestamp}"

    async def refresh_data(self):
        """Refresh all dashboard data."""
        try:
            # Fetch positions
            self.positions_data = await self.fetch_positions()

            # Fetch P&L history
            self.pnl_history = await self.fetch_pnl_history()

            # Fetch market data
            self.market_data = await self.fetch_market_data()

            # Fetch alerts
            self.alerts = await self.fetch_alerts()

            # Fetch activity
            self.activity_log = await self.fetch_activity()

        except Exception as e:
            logger.error(f"Error refreshing dashboard data: {e}")

    async def fetch_positions(self) -> list[dict[str, Any]]:
        """Fetch current positions."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT
                    p.symbol,
                    p.quantity,
                    p.entry_price as avg_price,
                    p.realized_pnl,
                    p.unrealized_pnl,
                    p.entry_timestamp as opened_at,
                    md.close as current_price
                FROM positions p
                LEFT JOIN LATERAL (
                    SELECT close
                    FROM market_data
                    WHERE symbol = p.symbol
                    ORDER BY timestamp DESC
                    LIMIT 1
                ) md ON true
                WHERE p.status = 'open'
                ORDER BY ABS(p.quantity * COALESCE(md.close, p.entry_price)) DESC
            """

            rows = await conn.fetch(query)

            positions = []
            total_value = 0

            # First pass to calculate total portfolio value
            for row in rows:
                if row["quantity"] and row["current_price"]:
                    market_value = abs(float(row["quantity"]) * float(row["current_price"]))
                    total_value += market_value

            # Second pass to create position data
            for row in rows:
                if row["quantity"] and row["current_price"]:
                    quantity = float(row["quantity"])
                    current_price = float(row["current_price"])
                    avg_price = float(row["avg_price"])

                    market_value = abs(quantity * current_price)
                    unrealized_pnl = quantity * (current_price - avg_price)
                    total_pnl = unrealized_pnl + float(row["realized_pnl"] or 0)

                    positions.append(
                        {
                            "symbol": row["symbol"],
                            "quantity": quantity,
                            "avg_price": avg_price,
                            "current_price": current_price,
                            "market_value": market_value,
                            "unrealized_pnl": unrealized_pnl,
                            "realized_pnl": float(row["realized_pnl"] or 0),
                            "total_pnl": total_pnl,
                            "pnl_percent": (
                                ((current_price - avg_price) / avg_price * 100)
                                if avg_price > 0
                                else 0
                            ),
                            "portfolio_percent": (
                                (market_value / total_value * 100) if total_value > 0 else 0
                            ),
                            "side": "long" if quantity > 0 else "short",
                            "opened_at": row["opened_at"],
                        }
                    )

            return positions

    async def fetch_pnl_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Fetch P&L history."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT
                    DATE(entry_timestamp) as date,
                    SUM(realized_pnl) as daily_realized,
                    SUM(unrealized_pnl) as daily_unrealized,
                    COUNT(*) as trades
                FROM positions
                WHERE entry_timestamp >= $1
                GROUP BY DATE(entry_timestamp)
                ORDER BY date
            """

            start_date = datetime.now() - timedelta(days=days)
            rows = await conn.fetch(query, start_date)

            history = []
            cumulative_pnl = 0

            for row in rows:
                daily_pnl = float(row["daily_realized"] or 0) + float(row["daily_unrealized"] or 0)
                cumulative_pnl += daily_pnl

                history.append(
                    {
                        "date": row["date"],
                        "daily_pnl": daily_pnl,
                        "cumulative_pnl": cumulative_pnl,
                        "trades": row["trades"],
                    }
                )

            return history

    async def fetch_market_data(self) -> dict[str, Any]:
        """Fetch market overview data."""
        async with self.db_pool.acquire() as conn:
            # Get VIX data
            vix_query = """
                SELECT close, open
                FROM market_data
                WHERE symbol = '^VIX'
                ORDER BY timestamp DESC
                LIMIT 2
            """
            vix_rows = await conn.fetch(vix_query)

            vix_data = {}
            if len(vix_rows) >= 1:
                current_vix = float(vix_rows[0]["close"])
                prev_vix = float(vix_rows[1]["close"]) if len(vix_rows) > 1 else current_vix
                vix_data = {
                    "current": current_vix,
                    "change": current_vix - prev_vix,
                    "change_pct": (
                        ((current_vix - prev_vix) / prev_vix * 100) if prev_vix > 0 else 0
                    ),
                    "level": self._get_vix_level(current_vix),
                }

            # Get sector ETFs
            sector_query = """
                SELECT
                    symbol,
                    close,
                    open,
                    volume
                FROM market_data
                WHERE symbol IN ('XLK', 'XLV', 'XLF', 'XLY', 'XLC', 'XLI', 'XLP', 'XLE', 'XLU', 'XLRE', 'XLB')
                  AND timestamp >= $1
                ORDER BY symbol, timestamp DESC
            """

            sector_rows = await conn.fetch(sector_query, datetime.now() - timedelta(hours=24))

            # Process sector data
            sectors = self._process_sector_data(sector_rows)

            # Get market indices
            indices_query = """
                SELECT symbol, last_price, change_percent, volume
                FROM market_indices
                WHERE symbol IN ('SPY', 'QQQ', 'IWM', 'DIA')
                ORDER BY symbol
            """

            indices_rows = await conn.fetch(indices_query)
            indices = [dict(row) for row in indices_rows]

            return {
                "vix": vix_data,
                "sectors": sectors,
                "indices": indices,
                "market_status": self._get_market_status(),
            }

    async def fetch_alerts(self) -> list[dict[str, Any]]:
        """Fetch active alerts."""
        # In a real implementation, this would fetch from alerts table
        # For now, return sample data
        return [
            {
                "id": 1,
                "timestamp": datetime.now() - timedelta(minutes=5),
                "type": "risk",
                "severity": "warning",
                "message": "Portfolio concentration in AAPL exceeds 30%",
                "symbol": "AAPL",
            },
            {
                "id": 2,
                "timestamp": datetime.now() - timedelta(minutes=15),
                "type": "market",
                "severity": "info",
                "message": "VIX spike detected - increased volatility",
                "symbol": "^VIX",
            },
        ]

    async def fetch_activity(self) -> list[dict[str, Any]]:
        """Fetch recent trading activity."""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT
                    created_at as timestamp,
                    symbol,
                    side,
                    quantity,
                    limit_price as price,
                    order_type,
                    status
                FROM orders
                ORDER BY created_at DESC
                LIMIT 50
            """

            rows = await conn.fetch(query)

            activity = []
            for row in rows:
                activity.append(
                    {
                        "timestamp": row["timestamp"],
                        "symbol": row["symbol"],
                        "side": row["side"],
                        "quantity": float(row["quantity"]),
                        "price": float(row["price"]) if row["price"] else 0,
                        "order_type": row["order_type"],
                        "status": row["status"],
                    }
                )

            return activity

    def render_trading_overview(self):
        """Render trading overview tab."""
        # Calculate summary metrics
        total_pnl = sum(p["total_pnl"] for p in self.positions_data)
        unrealized_pnl = sum(p["unrealized_pnl"] for p in self.positions_data)
        realized_pnl = sum(p["realized_pnl"] for p in self.positions_data)

        # Daily P&L
        today_pnl = 0
        if self.pnl_history:
            today_data = [h for h in self.pnl_history if h["date"] == datetime.now().date()]
            if today_data:
                today_pnl = today_data[0]["daily_pnl"]

        # Create metrics cards
        metrics_row = html.Div(
            [
                self._create_metric_card("Total P&L", f"${total_pnl:,.2f}", total_pnl < 0),
                self._create_metric_card("Today's P&L", f"${today_pnl:,.2f}", today_pnl < 0),
                self._create_metric_card(
                    "Unrealized P&L", f"${unrealized_pnl:,.2f}", unrealized_pnl < 0
                ),
                self._create_metric_card("Realized P&L", f"${realized_pnl:,.2f}", realized_pnl < 0),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "20px",
                "marginBottom": "30px",
            },
        )

        # Positions heatmap
        positions_fig = self._create_positions_heatmap()
        positions_chart = dcc.Graph(figure=positions_fig, style={"height": "400px"})

        # P&L chart
        pnl_fig = self._create_pnl_chart()
        pnl_chart = dcc.Graph(figure=pnl_fig, style={"height": "400px"})

        # Activity feed
        activity_feed = self._create_activity_feed()

        # Risk gauge with VIX adjustment
        risk_gauge = self._create_risk_gauge()

        return html.Div(
            [
                metrics_row,
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Positions Heatmap",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                positions_chart,
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "P&L History",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                pnl_chart,
                            ],
                            style={"width": "48%", "display": "inline-block", "marginLeft": "4%"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Recent Activity",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                activity_feed,
                            ],
                            style={"width": "70%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "Risk Status",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                risk_gauge,
                            ],
                            style={"width": "28%", "display": "inline-block", "marginLeft": "2%"},
                        ),
                    ]
                ),
            ]
        )

    def render_market_analysis(self):
        """Render market analysis tab."""
        vix_data = self.market_data.get("vix", {})
        sectors = self.market_data.get("sectors", [])
        indices = self.market_data.get("indices", [])

        # VIX card
        vix_card = html.Div(
            [
                html.H3(
                    "VIX - Volatility Index", style={"color": "#00ff88", "marginBottom": "15px"}
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span("Current: ", style={"color": "#888"}),
                                html.Span(
                                    f"{vix_data.get('current', 0):.2f}",
                                    style={"fontSize": "24px", "fontWeight": "bold"},
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("Change: ", style={"color": "#888"}),
                                html.Span(
                                    f"{vix_data.get('change', 0):+.2f} ({vix_data.get('change_pct', 0):+.1f}%)",
                                    style={
                                        "color": (
                                            "#00ff88"
                                            if vix_data.get("change", 0) < 0
                                            else "#ff4444"
                                        )
                                    },
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Span("Level: ", style={"color": "#888"}),
                                html.Span(
                                    vix_data.get("level", "Unknown"), style={"fontWeight": "bold"}
                                ),
                            ]
                        ),
                    ],
                    style={"backgroundColor": "#1a1a1a", "padding": "20px", "borderRadius": "8px"},
                ),
            ]
        )

        # Sector performance chart
        sector_fig = self._create_sector_performance_chart(sectors)
        sector_chart = dcc.Graph(figure=sector_fig, style={"height": "400px"})

        # Market indices
        indices_cards = html.Div(
            [self._create_index_card(idx) for idx in indices],
            style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "20px"},
        )

        # Market breadth indicators
        breadth_fig = self._create_market_breadth_chart()
        breadth_chart = dcc.Graph(figure=breadth_fig, style={"height": "300px"})

        return html.Div(
            [
                vix_card,
                html.Br(),
                indices_cards,
                html.Br(),
                html.H3("Sector Performance", style={"color": "#00ff88", "marginBottom": "15px"}),
                sector_chart,
                html.Br(),
                html.H3("Market Breadth", style={"color": "#00ff88", "marginBottom": "15px"}),
                breadth_chart,
            ]
        )

    def render_portfolio_analytics(self):
        """Render portfolio analytics tab."""
        # Calculate portfolio metrics
        portfolio_value = sum(p["market_value"] for p in self.positions_data)
        position_count = len(self.positions_data)
        avg_position_size = portfolio_value / position_count if position_count > 0 else 0

        # Sector exposure
        sector_exposure = self._calculate_sector_exposure()
        sector_fig = px.pie(
            values=list(sector_exposure.values()),
            names=list(sector_exposure.keys()),
            title="Sector Exposure",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        sector_fig.update_layout(template="plotly_dark", height=400)

        # Position details table
        positions_df = pd.DataFrame(self.positions_data)
        if not positions_df.empty:
            positions_table = self._create_positions_table(positions_df)
        else:
            positions_table = html.Div(
                "No positions", style={"textAlign": "center", "padding": "50px"}
            )

        # Risk metrics
        risk_metrics = self._calculate_risk_metrics()
        risk_cards = html.Div(
            [
                self._create_metric_card("Portfolio Value", f"${portfolio_value:,.2f}", False),
                self._create_metric_card("VaR (95%)", f"${risk_metrics['var_95']:,.2f}", True),
                self._create_metric_card(
                    "Max Concentration",
                    f"{risk_metrics['max_concentration']:.1f}%",
                    risk_metrics["max_concentration"] > 30,
                ),
                self._create_metric_card(
                    "Leverage", f"{risk_metrics['leverage']:.2f}x", risk_metrics["leverage"] > 2
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "20px",
                "marginBottom": "30px",
            },
        )

        # Performance attribution
        attribution_fig = self._create_performance_attribution()

        return html.Div(
            [
                risk_cards,
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Sector Exposure",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                dcc.Graph(figure=sector_fig),
                            ],
                            style={"width": "48%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "Performance Attribution",
                                    style={"color": "#00ff88", "marginBottom": "15px"},
                                ),
                                dcc.Graph(figure=attribution_fig, style={"height": "400px"}),
                            ],
                            style={"width": "48%", "display": "inline-block", "marginLeft": "4%"},
                        ),
                    ]
                ),
                html.Br(),
                html.H3("Position Details", style={"color": "#00ff88", "marginBottom": "15px"}),
                positions_table,
            ]
        )

    def render_alerts_activity(self):
        """Render alerts and activity tab."""
        # Active alerts
        alert_cards = []
        for alert in self.alerts:
            severity_color = {
                "info": "#00aaff",
                "warning": "#ffaa00",
                "error": "#ff4444",
                "critical": "#ff0000",
            }.get(alert["severity"], "#888")

            card = html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                alert["timestamp"].strftime("%H:%M:%S"),
                                style={"marginRight": "10px"},
                            ),
                            html.Span(
                                alert["type"].upper(),
                                style={"color": severity_color, "fontWeight": "bold"},
                            ),
                        ],
                        style={"marginBottom": "5px"},
                    ),
                    html.Div(alert["message"], style={"fontSize": "14px"}),
                    html.Div(
                        f"Symbol: {alert['symbol']}", style={"fontSize": "12px", "color": "#888"}
                    ),
                ],
                style={
                    "backgroundColor": "#1a1a1a",
                    "padding": "15px",
                    "marginBottom": "10px",
                    "borderRadius": "4px",
                    "borderLeft": f"3px solid {severity_color}",
                },
            )
            alert_cards.append(card)

        alerts_section = html.Div(
            [
                html.H3("Active Alerts", style={"color": "#00ff88", "marginBottom": "15px"}),
                (
                    html.Div(alert_cards)
                    if alert_cards
                    else html.Div(
                        "No active alerts", style={"textAlign": "center", "padding": "20px"}
                    )
                ),
            ]
        )

        # Trading logs
        logs_data = []
        for activity in self.activity_log[:20]:  # Last 20 activities
            logs_data.append(
                {
                    "Time": activity["timestamp"].strftime("%H:%M:%S"),
                    "Symbol": activity["symbol"],
                    "Side": activity["side"],
                    "Quantity": activity["quantity"],
                    "Price": f"${activity['price']:.2f}" if activity["price"] > 0 else "-",
                    "Type": activity["order_type"],
                    "Status": activity["status"],
                }
            )

        if logs_data:
            logs_df = pd.DataFrame(logs_data)
            logs_table = dash.dash_table.DataTable(
                data=logs_df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in logs_df.columns],
                style_cell={
                    "backgroundColor": "#1a1a1a",
                    "color": "#e0e0e0",
                    "border": "1px solid #333",
                },
                style_header={"backgroundColor": "#0f0f0f", "fontWeight": "bold"},
                style_data_conditional=[
                    {
                        "if": {"column_id": "Side", "filter_query": '{Side} = "buy"'},
                        "color": "#00ff88",
                    },
                    {
                        "if": {"column_id": "Side", "filter_query": '{Side} = "sell"'},
                        "color": "#ff4444",
                    },
                ],
            )
        else:
            logs_table = html.Div(
                "No trading activity", style={"textAlign": "center", "padding": "50px"}
            )

        return html.Div(
            [
                alerts_section,
                html.Br(),
                html.H3("Trading Activity Log", style={"color": "#00ff88", "marginBottom": "15px"}),
                logs_table,
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

    def _create_positions_heatmap(self):
        """Create positions heatmap visualization."""
        if not self.positions_data:
            return go.Figure().add_annotation(
                text="No positions to display",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        # Sort by absolute P&L
        positions = sorted(self.positions_data, key=lambda x: abs(x["total_pnl"]), reverse=True)[
            :20
        ]

        symbols = [p["symbol"] for p in positions]
        pnl_values = [p["total_pnl"] for p in positions]
        sizes = [p["market_value"] for p in positions]

        # Create treemap
        fig = go.Figure(
            go.Treemap(
                labels=symbols,
                values=sizes,
                parents=[""] * len(symbols),
                text=[
                    f"{s}<br>${pnl:.2f}<br>{pnl_pct:.1f}%"
                    for s, pnl, pnl_pct in zip(
                        symbols, pnl_values, [p["pnl_percent"] for p in positions]
                    )
                ],
                textposition="middle center",
                marker=dict(
                    colorscale="RdYlGn", cmid=0, colorbar=dict(title="P&L ($)"), line=dict(width=2)
                ),
                customdata=pnl_values,
                hovertemplate="<b>%{label}</b><br>P&L: $%{customdata:.2f}<br>Size: $%{value:,.0f}<extra></extra>",
            )
        )

        fig.update_traces(marker_colors=pnl_values)
        fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))

        return fig

    def _create_pnl_chart(self):
        """Create P&L history chart."""
        if not self.pnl_history:
            return go.Figure().add_annotation(
                text="No P&L history available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

        df = pd.DataFrame(self.pnl_history)

        fig = go.Figure()

        # Cumulative P&L line
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["cumulative_pnl"],
                mode="lines",
                name="Cumulative P&L",
                line=dict(color="#00ff88", width=2),
            )
        )

        # Daily P&L bars
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["daily_pnl"],
                name="Daily P&L",
                marker_color=df["daily_pnl"].apply(lambda x: "#00ff88" if x >= 0 else "#ff4444"),
            )
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return fig

    def _create_activity_feed(self):
        """Create recent activity feed."""
        if not self.activity_log:
            return html.Div("No recent activity", style={"textAlign": "center", "padding": "20px"})

        activity_items = []
        for activity in self.activity_log[:10]:  # Show last 10
            side_color = "#00ff88" if activity["side"] == "buy" else "#ff4444"

            item = html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                activity["timestamp"].strftime("%H:%M:%S"),
                                style={"marginRight": "10px", "color": "#888"},
                            ),
                            html.Span(
                                activity["symbol"],
                                style={"fontWeight": "bold", "marginRight": "10px"},
                            ),
                            html.Span(
                                activity["side"].upper(),
                                style={"color": side_color, "marginRight": "10px"},
                            ),
                            html.Span(
                                f"{activity['quantity']} @ ${activity['price']:.2f}"
                                if activity["price"] > 0
                                else f"{activity['quantity']} shares"
                            ),
                            html.Span(
                                f" - {activity['status']}",
                                style={"color": "#888", "fontSize": "12px"},
                            ),
                        ]
                    )
                ],
                style={"padding": "10px", "borderBottom": "1px solid #333"},
            )

            activity_items.append(item)

        return html.Div(
            activity_items,
            style={
                "backgroundColor": "#1a1a1a",
                "borderRadius": "8px",
                "maxHeight": "300px",
                "overflowY": "auto",
            },
        )

    def _create_risk_gauge(self):
        """Create risk gauge with VIX adjustment."""
        # Calculate base risk (placeholder - would use real risk metrics)
        position_concentration = (
            max([p["portfolio_percent"] for p in self.positions_data]) if self.positions_data else 0
        )
        base_risk = min(position_concentration * 2, 100)  # Simple risk calculation

        # Adjust for VIX
        vix_current = self.market_data.get("vix", {}).get("current", 20)
        vix_multiplier = 1.0

        if vix_current > 30:
            vix_multiplier = 1.3
        elif vix_current > 25:
            vix_multiplier = 1.15
        elif vix_current < 15:
            vix_multiplier = 0.9

        adjusted_risk = min(base_risk * vix_multiplier, 100)

        # Create gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=adjusted_risk,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Risk Level<br><sub>VIX: {vix_current:.1f}</sub>"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "green"},
                        {"range": [50, 75], "color": "yellow"},
                        {"range": [75, 100], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )

        fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=40, b=20))

        return dcc.Graph(figure=fig)

    def _get_vix_level(self, vix_value: float) -> str:
        """Get VIX level description."""
        if vix_value < 15:
            return "Low volatility"
        elif vix_value < 25:
            return "Normal volatility"
        elif vix_value < 35:
            return "High volatility"
        else:
            return "Extreme volatility"

    def _get_market_status(self) -> str:
        """Get current market status."""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return "closed"

        # Simple US market hours check
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        if market_open <= now <= market_close:
            return "open"
        elif now < market_open:
            return "pre-market"
        else:
            return "after-hours"

    def _process_sector_data(self, rows):
        """Process sector ETF data."""
        sector_map = {
            "XLK": "Technology",
            "XLV": "Healthcare",
            "XLF": "Financial",
            "XLY": "Consumer Disc",
            "XLC": "Communication",
            "XLI": "Industrial",
            "XLP": "Consumer Staples",
            "XLE": "Energy",
            "XLU": "Utilities",
            "XLRE": "Real Estate",
            "XLB": "Materials",
        }

        sectors = []
        for symbol, name in sector_map.items():
            symbol_data = [r for r in rows if r["symbol"] == symbol]
            if symbol_data:
                latest = symbol_data[0]
                close = float(latest["close"])
                open_price = float(latest["open"])
                change_pct = ((close - open_price) / open_price * 100) if open_price > 0 else 0

                sectors.append(
                    {
                        "name": name,
                        "symbol": symbol,
                        "change_pct": change_pct,
                        "volume": latest["volume"],
                    }
                )

        return sorted(sectors, key=lambda x: x["change_pct"], reverse=True)

    def _create_sector_performance_chart(self, sectors):
        """Create sector performance bar chart."""
        if not sectors:
            return go.Figure()

        df = pd.DataFrame(sectors)

        fig = go.Figure(
            go.Bar(
                x=df["change_pct"],
                y=df["name"],
                orientation="h",
                marker_color=df["change_pct"].apply(lambda x: "#00ff88" if x >= 0 else "#ff4444"),
                text=df["change_pct"].apply(lambda x: f"{x:+.2f}%"),
                textposition="outside",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Change %",
            yaxis_title="",
            margin=dict(l=100, r=50, t=20, b=50),
        )

        return fig

    def _create_index_card(self, index_data):
        """Create market index card."""
        change_pct = float(index_data.get("change_percent", 0))
        color = "#00ff88" if change_pct >= 0 else "#ff4444"

        return html.Div(
            [
                html.Div(index_data["symbol"], style={"fontSize": "16px", "fontWeight": "bold"}),
                html.Div(
                    f"${float(index_data.get('last_price', 0)):.2f}", style={"fontSize": "20px"}
                ),
                html.Div(f"{change_pct:+.2f}%", style={"color": color}),
            ],
            style={
                "backgroundColor": "#1a1a1a",
                "padding": "15px",
                "borderRadius": "8px",
                "textAlign": "center",
            },
        )

    def _create_market_breadth_chart(self):
        """Create market breadth visualization."""
        # Placeholder implementation
        fig = go.Figure()

        # Add advance/decline line
        dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
        advances = np.secure_randint(1000, 2000, size=30)
        declines = np.secure_randint(800, 1800, size=30)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=advances - declines,
                mode="lines",
                name="Advance-Decline Line",
                line=dict(color="#00ff88", width=2),
            )
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Net Advances",
            margin=dict(l=0, r=0, t=20, b=50),
        )

        return fig

    def _calculate_sector_exposure(self):
        """Calculate portfolio sector exposure."""
        # Simplified sector mapping
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "META": "Technology",
            "NVDA": "Technology",
            "AMD": "Technology",
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
            "UNH": "Healthcare",
            "JPM": "Financial",
            "BAC": "Financial",
            "WFC": "Financial",
            "AMZN": "Consumer",
            "WMT": "Consumer",
            "HD": "Consumer",
            "XOM": "Energy",
            "CVX": "Energy",
        }

        sector_values = {}
        for position in self.positions_data:
            sector = sector_map.get(position["symbol"], "Other")
            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += position["market_value"]

        return sector_values

    def _create_positions_table(self, df):
        """Create positions detail table."""
        # Format columns
        df["P&L"] = df["total_pnl"].apply(lambda x: f"${x:,.2f}")
        df["P&L %"] = df["pnl_percent"].apply(lambda x: f"{x:.1f}%")
        df["Value"] = df["market_value"].apply(lambda x: f"${x:,.2f}")
        df["Portfolio %"] = df["portfolio_percent"].apply(lambda x: f"{x:.1f}%")

        columns_to_show = [
            "symbol",
            "side",
            "quantity",
            "avg_price",
            "current_price",
            "P&L",
            "P&L %",
            "Value",
            "Portfolio %",
        ]

        return dash.dash_table.DataTable(
            data=df[columns_to_show].to_dict("records"),
            columns=[{"name": i, "id": i} for i in columns_to_show],
            style_cell={
                "backgroundColor": "#1a1a1a",
                "color": "#e0e0e0",
                "border": "1px solid #333",
            },
            style_header={"backgroundColor": "#0f0f0f", "fontWeight": "bold"},
            style_data_conditional=[
                {"if": {"column_id": "P&L", "filter_query": "{total_pnl} < 0"}, "color": "#ff4444"},
                {
                    "if": {"column_id": "P&L", "filter_query": "{total_pnl} >= 0"},
                    "color": "#00ff88",
                },
            ],
            sort_action="native",
        )

    def _calculate_risk_metrics(self):
        """Calculate portfolio risk metrics."""
        portfolio_value = sum(p["market_value"] for p in self.positions_data)

        # Simple VaR calculation (5% of portfolio value)
        var_95 = portfolio_value * 0.05

        # Max concentration
        max_concentration = (
            max([p["portfolio_percent"] for p in self.positions_data]) if self.positions_data else 0
        )

        # Leverage (placeholder)
        leverage = 1.5

        return {"var_95": var_95, "max_concentration": max_concentration, "leverage": leverage}

    def _create_performance_attribution(self):
        """Create performance attribution chart."""
        # Get top contributing positions
        top_positions = sorted(
            self.positions_data, key=lambda x: abs(x["total_pnl"]), reverse=True
        )[:10]

        symbols = [p["symbol"] for p in top_positions]
        pnl_values = [p["total_pnl"] for p in top_positions]

        fig = go.Figure(
            go.Waterfall(
                x=symbols,
                y=pnl_values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": "#00ff88"}},
                decreasing={"marker": {"color": "#ff4444"}},
                totals={"marker": {"color": "#00aaff"}},
            )
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Symbol",
            yaxis_title="P&L Contribution ($)",
            margin=dict(l=0, r=0, t=20, b=50),
        )

        return fig

    def _run_async(self, coro):
        """Run async coroutine in thread pool."""

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        future = self._executor.submit(run_in_thread)
        return future.result()

    def run(self, debug=False):
        """Run the dashboard server."""
        logger.info(f"Starting Trading Dashboard V2 on http://localhost:{self.port}")
        try:
            self.app.run(host="0.0.0.0", port=self.port, debug=debug)
        except Exception as e:
            logger.error(f"Dashboard startup error: {e}", exc_info=True)
            raise
