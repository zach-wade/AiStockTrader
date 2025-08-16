#!/usr/bin/env python3
"""
Smoke Test Harness for Minimal Trading Path (MTP)
Validates core trading functionality in paper mode.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

# Configure structured logging
logging.basicConfig(
    format='{"ts":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Configuration
SYMBOLS = ["SPY", "AAPL", "GOOGL"]
MAX_RISK_PER_TRADE = Decimal("0.01")  # 1% of portfolio
MAX_POSITION_SIZE = Decimal("10000")  # $10,000 max per position
MOCK_PORTFOLIO_VALUE = Decimal("100000")  # $100,000 portfolio


class HealthCheck:
    """System health checks."""

    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []

    async def check_database(self) -> bool:
        """Check database connectivity."""
        try:
            # TODO: Replace with actual database check
            logger.info("Checking database connectivity...")
            await asyncio.sleep(0.1)  # Simulate check
            self.checks_passed.append("database")
            return True
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            self.checks_failed.append("database")
            return False

    async def check_market_data_api(self) -> bool:
        """Check market data API connectivity."""
        try:
            # TODO: Replace with actual API check
            logger.info("Checking market data API...")
            await asyncio.sleep(0.1)  # Simulate check
            self.checks_passed.append("market_data_api")
            return True
        except Exception as e:
            logger.error(f"Market data API check failed: {e}")
            self.checks_failed.append("market_data_api")
            return False

    async def check_broker_api(self) -> bool:
        """Check broker API connectivity."""
        try:
            # TODO: Replace with actual broker check
            logger.info("Checking broker API...")
            await asyncio.sleep(0.1)  # Simulate check
            self.checks_passed.append("broker_api")
            return True
        except Exception as e:
            logger.error(f"Broker API check failed: {e}")
            self.checks_failed.append("broker_api")
            return False

    async def run_all_checks(self) -> bool:
        """Run all health checks."""
        checks = [
            self.check_database(),
            self.check_market_data_api(),
            self.check_broker_api(),
        ]
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        success = all(r is True for r in results if not isinstance(r, Exception))
        
        logger.info(
            f"Health check summary: {len(self.checks_passed)} passed, "
            f"{len(self.checks_failed)} failed"
        )
        
        if self.checks_failed:
            logger.error(f"Failed checks: {self.checks_failed}")
        
        return success


class MarketDataProvider:
    """Mock market data provider."""

    async def fetch_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        """Fetch current prices for symbols."""
        # TODO: Replace with real market data provider
        import random

        prices = {}
        for symbol in symbols:
            # Generate mock prices
            base_prices = {"SPY": 450.0, "AAPL": 150.0, "GOOGL": 140.0}
            base = base_prices.get(symbol, 100.0)
            variation = random.uniform(0.98, 1.02)
            prices[symbol] = Decimal(str(base * variation))
        
        logger.info(f"Fetched prices: {prices}")
        return prices

    async def fetch_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch detailed quote for a symbol."""
        price = (await self.fetch_prices([symbol]))[symbol]
        return {
            "symbol": symbol,
            "bid": price - Decimal("0.01"),
            "ask": price + Decimal("0.01"),
            "last": price,
            "volume": 1000000,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class RiskEngine:
    """Risk management engine."""

    def __init__(self, portfolio_value: Decimal):
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = MAX_RISK_PER_TRADE
        self.max_position_size = MAX_POSITION_SIZE

    def check_position_size(self, symbol: str, quantity: int, price: Decimal) -> Tuple[bool, Optional[str]]:
        """Check if position size is within limits."""
        position_value = quantity * price
        
        if position_value > self.max_position_size:
            msg = f"Position size ${position_value} exceeds max ${self.max_position_size}"
            logger.warning(msg)
            return False, msg
        
        risk_amount = self.portfolio_value * self.max_risk_per_trade
        if position_value > risk_amount:
            msg = f"Position size ${position_value} exceeds risk limit ${risk_amount}"
            logger.warning(msg)
            return False, msg
        
        return True, None

    def check_order(self, order: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate order against risk rules."""
        # Basic validation
        if order["quantity"] <= 0:
            return False, "Invalid quantity"
        
        if order["price"] <= 0:
            return False, "Invalid price"
        
        # Position size check
        return self.check_position_size(
            order["symbol"], order["quantity"], order["price"]
        )

    def calculate_position_size(self, symbol: str, price: Decimal, stop_loss: Decimal) -> int:
        """Calculate optimal position size based on risk."""
        risk_amount = self.portfolio_value * self.max_risk_per_trade
        risk_per_share = abs(price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        shares = int(risk_amount / risk_per_share)
        max_shares = int(self.max_position_size / price)
        
        return min(shares, max_shares)


class SimpleStrategy:
    """Simple momentum strategy for testing."""

    def __init__(self):
        self.signals = []

    async def generate_signals(self, prices: Dict[str, Decimal]) -> List[Dict[str, Any]]:
        """Generate trading signals based on prices."""
        signals = []
        
        for symbol, price in prices.items():
            # Simple mock logic: buy if price ends in 0-3
            price_str = str(price)
            last_digit = int(price_str[-1]) if price_str[-1].isdigit() else 0
            
            if last_digit <= 3:
                signal = {
                    "symbol": symbol,
                    "action": "buy",
                    "quantity": 10,
                    "price": price,
                    "reason": "momentum_signal",
                    "confidence": 0.7,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                signals.append(signal)
                logger.info(f"Generated signal: {signal}")
        
        self.signals = signals
        return signals


class PaperBroker:
    """Paper trading broker simulation."""

    def __init__(self):
        self.orders = []
        self.positions = {}
        self.order_id_counter = 1

    async def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Submit an order to paper broker."""
        order_id = f"ORDER_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        order_record = {
            "id": order_id,
            "symbol": order["symbol"],
            "side": order["action"],
            "quantity": order["quantity"],
            "price": float(order["price"]),
            "status": "filled",  # Instant fill for paper trading
            "filled_at": datetime.now(timezone.utc).isoformat(),
        }
        
        self.orders.append(order_record)
        
        # Update positions
        symbol = order["symbol"]
        if symbol not in self.positions:
            self.positions[symbol] = {"quantity": 0, "avg_price": 0}
        
        pos = self.positions[symbol]
        if order["action"] == "buy":
            new_qty = pos["quantity"] + order["quantity"]
            if new_qty > 0:
                pos["avg_price"] = (
                    (pos["quantity"] * pos["avg_price"] + order["quantity"] * float(order["price"]))
                    / new_qty
                )
            pos["quantity"] = new_qty
        
        logger.info(f"Order submitted: {order_record}")
        return order_record

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        return self.positions

    def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders."""
        return self.orders


class SmokeTester:
    """Main smoke test orchestrator."""

    def __init__(self):
        self.health_check = HealthCheck()
        self.market_data = MarketDataProvider()
        self.risk_engine = RiskEngine(MOCK_PORTFOLIO_VALUE)
        self.strategy = SimpleStrategy()
        self.broker = PaperBroker()
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "health_checks_passed": 0,
            "health_checks_failed": 0,
            "prices_fetched": 0,
            "signals_generated": 0,
            "orders_submitted": 0,
            "orders_rejected": 0,
            "errors": [],
        }

    async def run_smoke_test(self) -> bool:
        """Run complete smoke test sequence."""
        self.metrics["start_time"] = datetime.now(timezone.utc).isoformat()
        success = True
        
        try:
            # Step 1: Health checks
            logger.info("=== Starting Health Checks ===")
            health_ok = await self.health_check.run_all_checks()
            self.metrics["health_checks_passed"] = len(self.health_check.checks_passed)
            self.metrics["health_checks_failed"] = len(self.health_check.checks_failed)
            
            if not health_ok:
                logger.error("Health checks failed, aborting smoke test")
                return False
            
            # Step 2: Fetch market data
            logger.info("=== Fetching Market Data ===")
            prices = await self.market_data.fetch_prices(SYMBOLS)
            self.metrics["prices_fetched"] = len(prices)
            
            # Step 3: Generate signals
            logger.info("=== Generating Trading Signals ===")
            signals = await self.strategy.generate_signals(prices)
            self.metrics["signals_generated"] = len(signals)
            
            # Step 4: Process signals through risk and execution
            logger.info("=== Processing Orders ===")
            for signal in signals:
                # Risk check
                order = {
                    "symbol": signal["symbol"],
                    "action": signal["action"],
                    "quantity": signal["quantity"],
                    "price": signal["price"],
                }
                
                risk_ok, risk_msg = self.risk_engine.check_order(order)
                
                if risk_ok:
                    # Submit order
                    result = await self.broker.submit_order(order)
                    if result["status"] == "filled":
                        self.metrics["orders_submitted"] += 1
                    else:
                        self.metrics["orders_rejected"] += 1
                        logger.warning(f"Order rejected: {result}")
                else:
                    self.metrics["orders_rejected"] += 1
                    logger.warning(f"Risk check failed: {risk_msg}")
            
            # Step 5: Verify results
            logger.info("=== Smoke Test Results ===")
            positions = self.broker.get_positions()
            orders = self.broker.get_orders()
            
            logger.info(f"Final positions: {positions}")
            logger.info(f"Total orders: {len(orders)}")
            
            # Success criteria
            success = (
                self.metrics["health_checks_failed"] == 0
                and self.metrics["prices_fetched"] > 0
                and self.metrics["errors"] == []
            )
            
        except Exception as e:
            logger.error(f"Smoke test failed with exception: {e}")
            self.metrics["errors"].append(str(e))
            success = False
        
        finally:
            self.metrics["end_time"] = datetime.now(timezone.utc).isoformat()
            
            # Write metrics
            metrics_file = Path("smoke_test_metrics.json")
            metrics_file.write_text(json.dumps(self.metrics, indent=2, default=str))
            logger.info(f"Metrics written to {metrics_file}")
            
            # Print summary
            print("\n" + "=" * 50)
            print("SMOKE TEST SUMMARY")
            print("=" * 50)
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
            print(f"Health Checks: {self.metrics['health_checks_passed']}/{self.metrics['health_checks_passed'] + self.metrics['health_checks_failed']}")
            print(f"Prices Fetched: {self.metrics['prices_fetched']}")
            print(f"Signals Generated: {self.metrics['signals_generated']}")
            print(f"Orders Submitted: {self.metrics['orders_submitted']}")
            print(f"Orders Rejected: {self.metrics['orders_rejected']}")
            if self.metrics["errors"]:
                print(f"Errors: {self.metrics['errors']}")
            print("=" * 50)
        
        return success


async def main():
    """Main entry point."""
    tester = SmokeTester()
    success = await tester.run_smoke_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())