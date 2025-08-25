"""
Comprehensive Monitoring Example for AI Trading System

Demonstrates the complete observability and monitoring setup including:
- OpenTelemetry distributed tracing
- Structured logging with correlation
- Business and technical metrics
- Health checks with market hours
- Performance monitoring
- Integration with resilience infrastructure
"""

import asyncio
import time
from decimal import Decimal

# Import monitoring components
from src.infrastructure.monitoring import (  # Performance; Metrics; Telemetry; Logging
    TradingHealthChecker,
    correlation_context,
    initialize_performance_monitor,
    initialize_trading_metrics,
    initialize_trading_telemetry,
    log_order_filled,
    log_order_submitted,
    log_risk_breach,
    profile_performance,
    setup_structured_logging,
    trace_market_data_operation,
    trace_order_execution,
    trace_risk_calculation,
    track_trading_metric,
)

# Import resilience integration
from src.infrastructure.monitoring.integration import initialize_resilience_integration

# Import observability components
from src.infrastructure.observability import (
    create_file_exporter,
    create_multi_exporter,
    create_prometheus_exporter,
    initialize_observability_collector,
    initialize_trading_intelligence,
)


# Mock trading system components for demonstration
class MockBrokerClient:
    """Mock broker client for demonstration."""

    def __init__(self, name: str = "mock_broker"):
        self.name = name
        self.connected = True

    async def get_account(self):
        """Mock account info."""
        return type(
            "Account", (), {"status": "active", "buying_power": 50000.0, "account_blocked": False}
        )()

    async def get_connection_status(self):
        """Mock connection status."""
        return "connected" if self.connected else "disconnected"


class MockMarketDataClient:
    """Mock market data client for demonstration."""

    def __init__(self, name: str = "mock_market_data"):
        self.name = name
        self.connected = True

    async def get_latest_quote(self, symbol: str):
        """Mock quote data."""
        if not self.connected:
            raise ConnectionError("Market data disconnected")

        # Return mock quote
        return type("Quote", (), {"bid": 150.25, "ask": 150.30, "timestamp": time.time()})()

    async def get_connection_status(self):
        """Mock connection status."""
        return "connected" if self.connected else "disconnected"


class MockDatabaseConnection:
    """Mock database connection for demonstration."""

    def __init__(self):
        self.connected = True

    async def get_connection(self):
        """Mock connection factory."""
        return self

    async def acquire(self):
        """Mock connection acquisition."""
        return self

    async def cursor(self):
        """Mock cursor."""
        return MockCursor()

    async def get_pool_stats(self):
        """Mock pool statistics."""
        return {"size": 10, "checked_in": 8, "checked_out": 2, "overflow": 0}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockCursor:
    """Mock database cursor."""

    async def execute(self, query: str):
        """Mock query execution."""
        if "SELECT 1" in query:
            self._result = [(1,)]
        elif "COUNT(*)" in query:
            self._result = [(42,)]
        else:
            self._result = []

    async def fetchone(self):
        """Mock fetch one result."""
        return self._result[0] if self._result else None

    async def fetchall(self):
        """Mock fetch all results."""
        return self._result

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# Trading operation examples with monitoring
class MonitoredTradingSystem:
    """Example trading system with comprehensive monitoring."""

    def __init__(self):
        self.broker = MockBrokerClient()
        self.market_data = MockMarketDataClient()
        self.database = MockDatabaseConnection()

        # Initialize all monitoring components
        self._setup_monitoring()

    def _setup_monitoring(self):
        """Setup comprehensive monitoring infrastructure."""
        print("Setting up monitoring infrastructure...")

        # 1. Initialize OpenTelemetry
        self.telemetry = initialize_trading_telemetry(
            service_name="demo-trading-system",
            service_version="1.0.0",
            # In production: endpoint="http://jaeger:4317"
        )

        # 2. Setup structured logging
        setup_structured_logging(level="INFO", format_type="json", enable_sampling=True)

        # 3. Initialize metrics collection
        self.metrics = initialize_trading_metrics()

        # 4. Initialize performance monitoring
        self.performance_monitor = initialize_performance_monitor()

        # 5. Setup observability collector
        self.observability_collector = initialize_observability_collector(
            buffer_size=1000, flush_interval=10.0, enable_auto_export=True, export_interval=30.0
        )

        # 6. Setup exporters
        prometheus_exporter = create_prometheus_exporter(metrics_file="demo_metrics.prom")
        file_exporter = create_file_exporter(file_path="demo_observability.jsonl")
        multi_exporter = create_multi_exporter([prometheus_exporter, file_exporter])

        self.observability_collector.add_exporter(multi_exporter.export)

        # 7. Initialize business intelligence
        self.trading_intelligence = initialize_trading_intelligence(
            max_history_days=30, performance_calculation_interval=60.0
        )

        # 8. Setup health checking
        self.health_checker = TradingHealthChecker(check_interval=30.0, history_size=100)

        # 9. Initialize resilience integration
        self.resilience_integration = initialize_resilience_integration()

        print("Monitoring infrastructure setup complete!")

    @trace_order_execution
    @profile_performance("order_submission")
    @track_trading_metric("orders_submitted", "counter", {"system": "demo"})
    async def submit_order(
        self,
        symbol: str,
        quantity: int,
        price: Decimal | None = None,
        strategy: str = "demo_strategy",
    ) -> str:
        """Submit trading order with full monitoring."""
        order_id = f"order_{int(time.time() * 1000)}"

        with correlation_context() as correlation_id:
            # Log order submission
            log_order_submitted(
                order_id=order_id,
                symbol=symbol,
                side="buy",
                quantity=quantity,
                price=price,
                strategy=strategy,
            )

            # Record in observability collector
            self.observability_collector.collect_order_event(
                operation="submitted",
                status="success",
                order_id=order_id,
                symbol=symbol,
                strategy=strategy,
                quantity=quantity,
                price=float(price) if price else None,
            )

            # Record metrics
            self.metrics.record_order_submitted(
                order_id, symbol, quantity, float(price) if price else None
            )

            # Simulate processing time
            await asyncio.sleep(0.01)

            return order_id

    @trace_order_execution
    @profile_performance("order_execution")
    async def execute_order(
        self,
        order_id: str,
        symbol: str,
        quantity: int,
        price: Decimal,
        strategy: str = "demo_strategy",
    ) -> Dict[str, Any]:
        """Execute trading order with monitoring."""
        start_time = time.perf_counter()

        with correlation_context():
            try:
                # Simulate order execution
                await asyncio.sleep(0.05)  # Simulate network latency

                execution_time_ms = (time.perf_counter() - start_time) * 1000

                # Log successful execution
                log_order_filled(order_id=order_id, symbol=symbol, quantity=quantity, price=price)

                # Record in observability collector
                self.observability_collector.collect_order_event(
                    operation="filled",
                    status="success",
                    order_id=order_id,
                    symbol=symbol,
                    duration_ms=execution_time_ms,
                    strategy=strategy,
                    quantity=quantity,
                    price=float(price),
                )

                # Record metrics
                self.metrics.record_order_filled(
                    order_id, symbol, quantity, float(price), execution_time_ms
                )

                # Process for business intelligence
                self.trading_intelligence.process_trading_event(
                    type(
                        "Event",
                        (),
                        {
                            "event_type": "order",
                            "operation": "filled",
                            "order_id": order_id,
                            "symbol": symbol,
                            "strategy": strategy,
                            "timestamp": time.time(),
                            "context": {
                                "quantity": quantity,
                                "price": float(price),
                                "pnl": float(quantity * price * 0.001),  # Mock P&L
                            },
                            "portfolio_id": "demo_portfolio",
                        },
                    )()
                )

                return {
                    "order_id": order_id,
                    "status": "filled",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": float(price),
                    "execution_time_ms": execution_time_ms,
                }

            except Exception as e:
                # Record failed execution
                self.observability_collector.collect_error_event(
                    event_type="order",
                    operation="execution_failed",
                    error=e,
                    order_id=order_id,
                    symbol=symbol,
                )

                self.metrics.record_order_rejected(order_id, str(e))
                raise

    @trace_market_data_operation
    @profile_performance("market_data_fetch")
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data with monitoring."""
        start_time = time.perf_counter()

        try:
            quote = await self.market_data.get_latest_quote(symbol)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Record market data event
            self.observability_collector.collect_market_data_event(
                operation="quote_fetch",
                status="success",
                symbol=symbol,
                duration_ms=duration_ms,
                bid=quote.bid,
                ask=quote.ask,
            )

            return {
                "symbol": symbol,
                "bid": quote.bid,
                "ask": quote.ask,
                "spread": quote.ask - quote.bid,
                "timestamp": quote.timestamp,
            }

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            self.observability_collector.collect_error_event(
                event_type="market_data",
                operation="quote_fetch",
                error=e,
                duration_ms=duration_ms,
                symbol=symbol,
            )
            raise

    @trace_risk_calculation
    @profile_performance("risk_calculation")
    async def calculate_portfolio_risk(
        self, portfolio_id: str = "demo_portfolio"
    ) -> Dict[str, float]:
        """Calculate portfolio risk with monitoring."""
        start_time = time.perf_counter()

        try:
            # Simulate risk calculations
            await asyncio.sleep(0.02)

            # Mock risk metrics
            risk_metrics = {
                "var_95": 5000.0,
                "max_drawdown": 2.5,
                "beta": 1.2,
                "sharpe_ratio": 1.8,
                "volatility": 15.0,
            }

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Record risk calculation event
            self.observability_collector.collect_risk_event(
                operation="portfolio_risk_calculation",
                status="success",
                portfolio_id=portfolio_id,
                duration_ms=duration_ms,
                **risk_metrics,
            )

            # Check for risk breaches
            if risk_metrics["max_drawdown"] > 2.0:
                log_risk_breach(
                    risk_type="max_drawdown",
                    current_value=risk_metrics["max_drawdown"],
                    limit_value=2.0,
                    portfolio_id=portfolio_id,
                )

            # Process for business intelligence
            for risk_type, value in risk_metrics.items():
                self.trading_intelligence.process_trading_event(
                    type(
                        "Event",
                        (),
                        {
                            "event_type": "risk",
                            "operation": "risk_calculation",
                            "portfolio_id": portfolio_id,
                            "timestamp": time.time(),
                            "metrics": {risk_type: value},
                            "context": {"calculation_duration_ms": duration_ms},
                        },
                    )()
                )

            return risk_metrics

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            self.observability_collector.collect_error_event(
                event_type="risk",
                operation="portfolio_risk_calculation",
                error=e,
                duration_ms=duration_ms,
                portfolio_id=portfolio_id,
            )
            raise

    async def start_monitoring(self):
        """Start all background monitoring tasks."""
        print("Starting background monitoring...")

        # Start background tasks
        await self.metrics.start_background_collection(interval=30.0)
        await self.performance_monitor.start_background_monitoring(interval=60.0)
        await self.observability_collector.start_background_collection()
        await self.trading_intelligence.start_background_analysis()
        await self.health_checker.start_monitoring()

        print("Background monitoring started!")

    async def stop_monitoring(self):
        """Stop all background monitoring tasks."""
        print("Stopping background monitoring...")

        await self.metrics.stop_background_collection()
        await self.performance_monitor.stop_background_monitoring()
        await self.observability_collector.stop_background_collection()
        await self.trading_intelligence.stop_background_analysis()
        await self.health_checker.stop_monitoring()

        print("Background monitoring stopped!")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            "timestamp": time.time(),
            "metrics_summary": self.metrics.get_metric_snapshots()[-10:],  # Last 10 metrics
            "health_summary": self.health_checker.get_health_summary(),
            "performance_report": self.performance_monitor.get_performance_report(),
            "observability_health": self.observability_collector.get_health_summary(),
            "trading_intelligence": self.trading_intelligence.generate_comprehensive_report(),
            "resilience_status": self.resilience_integration.get_resilience_summary(),
        }


async def run_monitoring_demo():
    """Run comprehensive monitoring demonstration."""
    print("=" * 60)
    print("AI Trading System - Comprehensive Monitoring Demo")
    print("=" * 60)

    # Initialize trading system
    trading_system = MonitoredTradingSystem()

    try:
        # Start monitoring
        await trading_system.start_monitoring()

        print("\n📊 Executing trading operations with monitoring...")

        # Simulate trading operations
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

        for i, symbol in enumerate(symbols):
            print(f"\n🔄 Processing {symbol} (operation {i+1}/{len(symbols)})")

            # Get market data
            market_data = await trading_system.get_market_data(symbol)
            print(f"   📈 Market Data: Bid={market_data['bid']}, Ask={market_data['ask']}")

            # Submit order
            order_id = await trading_system.submit_order(
                symbol=symbol,
                quantity=100,
                price=Decimal(str(market_data["bid"])),
                strategy="momentum_strategy",
            )
            print(f"   📋 Order Submitted: {order_id}")

            # Execute order
            execution_result = await trading_system.execute_order(
                order_id=order_id,
                symbol=symbol,
                quantity=100,
                price=Decimal(str(market_data["bid"])),
                strategy="momentum_strategy",
            )
            print(f"   ✅ Order Executed: {execution_result['execution_time_ms']:.2f}ms")

            # Calculate risk
            risk_metrics = await trading_system.calculate_portfolio_risk()
            print(
                f"   ⚠️  Risk Metrics: VaR={risk_metrics['var_95']}, Sharpe={risk_metrics['sharpe_ratio']}"
            )

            # Small delay between operations
            await asyncio.sleep(0.5)

        print("\n📊 Generating monitoring reports...")

        # Wait for metrics collection
        await asyncio.sleep(2)

        # Get comprehensive monitoring summary
        monitoring_summary = trading_system.get_monitoring_summary()

        print("\n📈 Monitoring Summary:")
        print(f"   • Health Status: {monitoring_summary['health_summary']['overall_status']}")
        print(f"   • Metrics Collected: {len(monitoring_summary['metrics_summary'])}")
        print(
            f"   • Observability Events: {monitoring_summary['observability_health']['events_collected']}"
        )
        print(
            f"   • Performance Report: {monitoring_summary['performance_report'].operation if hasattr(monitoring_summary['performance_report'], 'operation') else 'Available'}"
        )

        # Show trading intelligence
        intelligence = monitoring_summary["trading_intelligence"]
        if "trading_activity" in intelligence:
            activity = intelligence["trading_activity"]
            print(
                f"   • Trading Activity: {activity.get('total_trades_30d', 0)} trades, {activity.get('active_orders', 0)} active orders"
            )

        print("\n📊 Performance Bottlenecks:")
        performance_monitor = trading_system.performance_monitor
        bottlenecks = performance_monitor.get_bottlenecks(top_n=3)
        for i, bottleneck in enumerate(bottlenecks[:3], 1):
            print(
                f"   {i}. {bottleneck.get('description', 'Unknown bottleneck')} (Severity: {bottleneck.get('severity', 'unknown')})"
            )

        if not bottlenecks:
            print("   ✅ No significant bottlenecks detected")

        print("\n📋 Files Generated:")
        print("   • Prometheus metrics: demo_metrics.prom")
        print("   • Observability events: demo_observability.jsonl")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        await trading_system.stop_monitoring()

        # Shutdown telemetry
        trading_system.telemetry.shutdown()


if __name__ == "__main__":
    # Run the monitoring demo
    asyncio.run(run_monitoring_demo())
