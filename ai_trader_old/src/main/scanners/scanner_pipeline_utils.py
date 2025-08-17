"""
Scanner Pipeline Utilities

Helper functions and utilities for the scanner pipeline.
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
from typing import Any

# Third-party imports
import pandas as pd
from sqlalchemy import text

# Local imports
from main.interfaces.database import IAsyncDatabase

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """
    Real-time monitoring for scanner pipeline execution.
    Tracks performance metrics and provides alerts.
    """

    def __init__(self, alert_threshold_seconds: int = 60):
        self.layer_times: dict[str, float] = {}
        self.layer_counts: dict[str, tuple[int, int]] = {}  # (input, output)
        self.start_time: datetime | None = None
        self.alert_threshold = alert_threshold_seconds
        self.alerts: list[dict[str, Any]] = []

    def start_pipeline(self):
        """Mark pipeline start."""
        self.start_time = datetime.now(UTC)
        logger.info("Pipeline monitoring started")

    def record_layer_start(self, layer_name: str, input_count: int):
        """Record layer execution start."""
        self.layer_times[f"{layer_name}_start"] = datetime.now(UTC).timestamp()
        if layer_name in self.layer_counts:
            _, output = self.layer_counts[layer_name]
            self.layer_counts[layer_name] = (input_count, output)
        else:
            self.layer_counts[layer_name] = (input_count, 0)

    def record_layer_end(self, layer_name: str, output_count: int, errors: list[str] = None):
        """Record layer execution end."""
        end_time = datetime.now(UTC).timestamp()
        start_time = self.layer_times.get(f"{layer_name}_start")

        if start_time:
            duration = end_time - start_time
            self.layer_times[f"{layer_name}_duration"] = duration

            # Update counts
            input_count, _ = self.layer_counts.get(layer_name, (0, 0))
            self.layer_counts[layer_name] = (input_count, output_count)

            # Check for alerts
            if duration > self.alert_threshold:
                self.alerts.append(
                    {
                        "type": "slow_layer",
                        "layer": layer_name,
                        "duration": duration,
                        "threshold": self.alert_threshold,
                        "timestamp": datetime.now(UTC),
                    }
                )

            # Check for excessive reduction
            if input_count > 0:
                reduction_rate = 1 - (output_count / input_count)
                if reduction_rate > 0.95:  # >95% reduction
                    self.alerts.append(
                        {
                            "type": "excessive_reduction",
                            "layer": layer_name,
                            "reduction_rate": reduction_rate,
                            "input_count": input_count,
                            "output_count": output_count,
                            "timestamp": datetime.now(UTC),
                        }
                    )

            # Check for errors
            if errors:
                self.alerts.append(
                    {
                        "type": "layer_errors",
                        "layer": layer_name,
                        "errors": errors,
                        "timestamp": datetime.now(UTC),
                    }
                )

    def get_summary(self) -> dict[str, Any]:
        """Get monitoring summary."""
        total_duration = None
        if self.start_time:
            total_duration = (datetime.now(UTC) - self.start_time).total_seconds()

        return {
            "total_duration": total_duration,
            "layer_durations": {
                k: v for k, v in self.layer_times.items() if k.endswith("_duration")
            },
            "layer_counts": self.layer_counts,
            "alerts": self.alerts,
            "alert_count": len(self.alerts),
        }


class SymbolValidator:
    """
    Validates symbols at each layer of the pipeline.
    """

    @staticmethod
    def validate_symbols(symbols: list[str], layer_name: str) -> tuple[list[str], list[str]]:
        """Validate symbol list and return valid/invalid symbols."""
        valid_symbols = []
        invalid_symbols = []

        for symbol in symbols:
            if SymbolValidator.is_valid_symbol(symbol):
                valid_symbols.append(symbol)
            else:
                invalid_symbols.append(symbol)

        if invalid_symbols:
            logger.warning(
                f"Layer {layer_name}: Found {len(invalid_symbols)} invalid symbols: "
                f"{invalid_symbols[:5]}{'...' if len(invalid_symbols) > 5 else ''}"
            )

        return valid_symbols, invalid_symbols

    @staticmethod
    def is_valid_symbol(symbol: str) -> bool:
        """Check if symbol is valid."""
        if not symbol or not isinstance(symbol, str):
            return False

        # Basic validation rules
        if len(symbol) < 1 or len(symbol) > 10:
            return False

        # Must contain only letters, numbers, dots, or dashes
        if not all(c.isalnum() or c in ".-" for c in symbol):
            return False

        # Must start with a letter
        if not symbol[0].isalpha():
            return False

        return True


class PerformanceAnalyzer:
    """
    Analyzes pipeline performance and provides optimization suggestions.
    """

    def __init__(self, db_adapter: IAsyncDatabase):
        self.db_adapter = db_adapter

    async def analyze_historical_performance(self, days: int = 7) -> dict[str, Any]:
        """Analyze historical pipeline performance."""
        query = text(
            """
            SELECT
                execution_date,
                layer_name,
                input_count,
                output_count,
                execution_time_seconds,
                success
            FROM pipeline_execution_history
            WHERE execution_date >= CURRENT_DATE - INTERVAL :days
            ORDER BY execution_date DESC, layer_name
        """
        )

        def execute_query(session):
            result = session.execute(query, {"days": f"{days} days"})
            return [dict(row._mapping) for row in result]

        try:
            history = await self.db_adapter.run_sync(execute_query)

            if not history:
                return {"message": "No historical data available"}

            # Convert to DataFrame for analysis
            df = pd.DataFrame(history)

            # Calculate metrics by layer
            layer_metrics = {}
            for layer in df["layer_name"].unique():
                layer_df = df[df["layer_name"] == layer]

                layer_metrics[layer] = {
                    "avg_execution_time": layer_df["execution_time_seconds"].mean(),
                    "max_execution_time": layer_df["execution_time_seconds"].max(),
                    "avg_reduction_rate": (
                        1 - (layer_df["output_count"] / layer_df["input_count"]).mean()
                    ),
                    "success_rate": layer_df["success"].mean(),
                    "execution_count": len(layer_df),
                }

            # Identify bottlenecks
            bottlenecks = []
            for layer, metrics in layer_metrics.items():
                if metrics["avg_execution_time"] > 60:  # >1 minute average
                    bottlenecks.append(
                        {
                            "layer": layer,
                            "avg_time": metrics["avg_execution_time"],
                            "suggestion": f"Consider optimizing {layer} - average time {metrics['avg_execution_time']:.1f}s",
                        }
                    )

            return {
                "layer_metrics": layer_metrics,
                "bottlenecks": bottlenecks,
                "total_executions": len(df["execution_date"].unique()),
                "overall_success_rate": df["success"].mean(),
            }

        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e)}

    def suggest_optimizations(self, pipeline_result: dict[str, Any]) -> list[str]:
        """Suggest optimizations based on pipeline results."""
        suggestions = []

        # Check layer execution times
        for layer in pipeline_result.get("layer_results", []):
            if layer["execution_time"] > 120:  # >2 minutes
                suggestions.append(
                    f"Layer {layer['layer_number']} took {layer['execution_time']:.1f}s. "
                    f"Consider enabling parallel processing or caching."
                )

            # Check reduction rates
            if layer["input_count"] > 0:
                reduction = 1 - (layer["output_count"] / layer["input_count"])
                if reduction > 0.9:  # >90% reduction
                    suggestions.append(
                        f"Layer {layer['layer_number']} filtered out {reduction*100:.1f}% of symbols. "
                        f"Review filtering criteria to ensure it's not too restrictive."
                    )

        # Check final output
        final_count = len(pipeline_result.get("final_opportunities", []))
        if final_count == 0:
            suggestions.append(
                "No final opportunities found. Consider relaxing filtering criteria "
                "or checking market conditions."
            )
        elif final_count > 50:
            suggestions.append(
                f"Found {final_count} opportunities. Consider tightening criteria "
                f"to focus on higher-quality candidates."
            )

        return suggestions


class DataQualityChecker:
    """
    Checks data quality throughout the pipeline.
    """

    @staticmethod
    async def check_market_data_freshness(
        db_adapter: IAsyncDatabase, symbols: list[str], max_age_hours: int = 24
    ) -> dict[str, Any]:
        """Check if market data is fresh for given symbols."""
        if not symbols:
            return {"fresh_count": 0, "stale_count": 0, "missing_count": 0}

        query = text(
            """
            WITH latest_data AS (
                SELECT
                    symbol,
                    MAX(timestamp) as last_update
                FROM market_data
                WHERE symbol = ANY(:symbols)
                GROUP BY symbol
            )
            SELECT
                symbol,
                last_update,
                CASE
                    WHEN last_update > NOW() - INTERVAL :max_age
                    THEN 'fresh'
                    ELSE 'stale'
                END as status
            FROM latest_data
        """
        )

        def execute_query(session):
            result = session.execute(
                query, {"symbols": symbols, "max_age": f"{max_age_hours} hours"}
            )
            return [dict(row._mapping) for row in result]

        try:
            results = await db_adapter.run_sync(execute_query)

            # Count by status
            fresh_count = sum(1 for r in results if r["status"] == "fresh")
            stale_count = sum(1 for r in results if r["status"] == "stale")

            # Find missing symbols
            found_symbols = {r["symbol"] for r in results}
            missing_symbols = set(symbols) - found_symbols

            return {
                "fresh_count": fresh_count,
                "stale_count": stale_count,
                "missing_count": len(missing_symbols),
                "total_symbols": len(symbols),
                "freshness_rate": fresh_count / len(symbols) if symbols else 0,
                "stale_symbols": [r["symbol"] for r in results if r["status"] == "stale"][:10],
                "missing_symbols": list(missing_symbols)[:10],
            }

        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return {"error": str(e)}


class PipelineReporter:
    """
    Generates detailed reports from pipeline execution.
    """

    @staticmethod
    def generate_html_report(pipeline_result: dict[str, Any], output_path: Path) -> str:
        """Generate HTML report from pipeline results."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Scanner Pipeline Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .status-success {{ color: green; font-weight: bold; }}
        .status-failed {{ color: red; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .layer-progression {{ margin: 20px 0; }}
        .opportunity {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0;
                        border-left: 4px solid #4CAF50; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Scanner Pipeline Report</h1>
        <p>Generated: {timestamp}</p>
        <p>Status: <span class="{status_class}">{status}</span></p>
        <p>Duration: {duration:.2f} seconds</p>
    </div>

    <h2>Summary Metrics</h2>
    <div>
        <div class="metric">
            <div class="metric-value">{initial_symbols:,}</div>
            <div class="metric-label">Initial Symbols</div>
        </div>
        <div class="metric">
            <div class="metric-value">{final_opportunities}</div>
            <div class="metric-label">Final Opportunities</div>
        </div>
        <div class="metric">
            <div class="metric-value">{reduction_rate:.1f}%</div>
            <div class="metric-label">Total Reduction</div>
        </div>
        <div class="metric">
            <div class="metric-value">{selection_rate:.3f}%</div>
            <div class="metric-label">Selection Rate</div>
        </div>
    </div>

    <h2>Layer Progression</h2>
    <table class="layer-progression">
        <tr>
            <th>Layer</th>
            <th>Name</th>
            <th>Input</th>
            <th>Output</th>
            <th>Reduction</th>
            <th>Time (s)</th>
            <th>Status</th>
        </tr>
        {layer_rows}
    </table>

    <h2>Top Opportunities</h2>
    {opportunities_html}

    {errors_section}
</body>
</html>
        """

        # Prepare data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "SUCCESS" if pipeline_result.get("success") else "FAILED"
        status_class = "status-success" if pipeline_result.get("success") else "status-failed"

        # Layer rows
        layer_rows = []
        for layer in pipeline_result.get("layer_results", []):
            reduction = ""
            if layer["input_count"] > 0:
                reduction_pct = (1 - layer["output_count"] / layer["input_count"]) * 100
                reduction = f"{reduction_pct:.1f}%"

            status = "✓" if not layer.get("errors") else "✗"

            layer_rows.append(
                f"""<tr>
                    <td>{layer['layer_number']}</td>
                    <td>{layer['layer_name']}</td>
                    <td>{layer['input_count']:,}</td>
                    <td>{layer['output_count']:,}</td>
                    <td>{reduction}</td>
                    <td>{layer['execution_time']:.1f}</td>
                    <td>{status}</td>
                </tr>"""
            )

        # Opportunities
        opportunities_html = ""
        for i, opp in enumerate(pipeline_result.get("final_opportunities", [])[:10], 1):
            opportunities_html += f"""
            <div class="opportunity">
                <strong>{i}. {opp.get('symbol', 'N/A')}</strong><br>
                Score: {opp.get('score', 0):.2f} |
                RVOL: {opp.get('rvol', 0):.1f}x |
                Price Change: {opp.get('price_change_pct', 0):.2f}%
            </div>
            """

        # Errors section
        errors_section = ""
        if pipeline_result.get("errors"):
            errors_section = "<h2>Errors</h2><ul>"
            for error in pipeline_result["errors"]:
                errors_section += f"<li>{error}</li>"
            errors_section += "</ul>"

        # Calculate metrics
        initial_symbols = (
            pipeline_result["layer_results"][0]["output_count"]
            if pipeline_result.get("layer_results")
            else 0
        )
        final_opportunities = len(pipeline_result.get("final_opportunities", []))

        funnel = pipeline_result.get("metadata", {}).get("funnel_reduction", {})
        reduction_rate = funnel.get("total_reduction", 0) * 100
        selection_rate = funnel.get("final_selection_rate", 0) * 100

        # Format HTML
        html = html_template.format(
            timestamp=timestamp,
            status=status,
            status_class=status_class,
            duration=pipeline_result.get("total_duration", 0),
            initial_symbols=initial_symbols,
            final_opportunities=final_opportunities,
            reduction_rate=reduction_rate,
            selection_rate=selection_rate,
            layer_rows="\n".join(layer_rows),
            opportunities_html=opportunities_html,
            errors_section=errors_section,
        )

        # Save to file
        output_file = (
            output_path / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        with open(output_file, "w") as f:
            f.write(html)

        return str(output_file)


# Utility functions for common pipeline operations


async def batch_process_symbols(
    symbols: list[str], process_func, batch_size: int = 100, max_concurrent: int = 5
) -> list[Any]:
    """
    Process symbols in batches with concurrency control.

    Args:
        symbols: List of symbols to process
        process_func: Async function to process each batch
        batch_size: Size of each batch
        max_concurrent: Maximum concurrent batches

    Returns:
        List of results from all batches
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch_with_semaphore(batch):
        async with semaphore:
            return await process_func(batch)

    # Create batches
    batches = [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]

    # Process batches
    tasks = [process_batch_with_semaphore(batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect results
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.error(f"Batch {i} failed: {result}")
        else:
            results.extend(result)

    return results


def calculate_market_hours_elapsed() -> float:
    """
    Calculate hours elapsed in current trading day.
    Returns 0 if market is closed.
    """
    now = datetime.now(UTC)
    et_offset = timedelta(hours=-5)  # ET timezone
    et_now = now + et_offset

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et_now.replace(hour=16, minute=0, second=0, microsecond=0)

    if et_now < market_open:
        return 0.0  # Pre-market
    elif et_now > market_close:
        return 6.5  # Full day
    else:
        elapsed = (et_now - market_open).total_seconds() / 3600
        return min(elapsed, 6.5)


def estimate_remaining_pipeline_time(
    completed_layers: list[str], historical_times: dict[str, float]
) -> float:
    """
    Estimate remaining pipeline execution time based on historical data.

    Args:
        completed_layers: List of completed layer names
        historical_times: Dict of layer_name -> average_seconds

    Returns:
        Estimated seconds remaining
    """
    all_layers = ["Layer 0", "Layer 1", "Layer 1.5", "Layer 2", "Layer 3"]
    remaining_layers = [l for l in all_layers if l not in completed_layers]

    estimated_time = 0.0
    for layer in remaining_layers:
        estimated_time += historical_times.get(layer, 60.0)  # Default 60s if no history

    return estimated_time
