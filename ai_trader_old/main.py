#!/usr/bin/env python3
"""
AI Trading System - Minimal Viable Trading Bot
A pragmatic implementation using existing working components
"""

# Standard library imports
import asyncio
from datetime import UTC, datetime
from decimal import Decimal
import logging
import os
from pathlib import Path
import signal
import sys
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("trading_bot.log")],
)
logger = logging.getLogger(__name__)

# Third-party imports
# Import existing working components
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# Import Alpaca for trading
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# Local imports
from main.data_pipeline.storage.database_adapter import AsyncDatabaseAdapter


class MinimalTradingBot:
    """Minimal viable trading bot using existing components"""

    def __init__(self):
        self.running = False
        self.config = None
        self.db_adapter = None
        self.trading_client = None
        self.data_client = None
        self.positions = {}
        self.active_symbols = []

    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🚀 Initializing Minimal Trading Bot...")

            # Load configuration
            logger.info("Loading configuration...")
            self.config = self._load_config()

            # Initialize database
            logger.info("Connecting to database...")
            await self._init_database()

            # Initialize Alpaca clients
            logger.info("Initializing Alpaca clients...")
            self._init_alpaca_clients()

            # Load active symbols
            logger.info("Loading active symbols...")
            await self._load_active_symbols()

            logger.info(f"✅ Bot initialized successfully with {len(self.active_symbols)} symbols")
            return True

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False

    def _load_config(self) -> dict:
        """Load configuration from environment and files"""
        config = {
            "database": {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", 5432)),
                "database": os.getenv("DB_NAME", "ai_trader"),
                "user": os.getenv("DB_USER", "zachwade"),
                "password": os.getenv("DB_PASSWORD", ""),
            },
            "alpaca": {
                "api_key": os.getenv("ALPACA_API_KEY"),
                "secret_key": os.getenv("ALPACA_SECRET_KEY"),
                "base_url": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
                "feed": "iex",  # or 'sip' for paid data
            },
            "trading": {
                "max_position_size": Decimal(os.getenv("MAX_POSITION_SIZE", "10000")),
                "max_positions": 10,
                "stop_loss_pct": 0.02,  # 2% stop loss
                "take_profit_pct": 0.05,  # 5% take profit
            },
            "symbols": {
                "max_symbols": 10,  # Start with top 10 liquid symbols
                "min_volume": 1000000,  # Minimum daily volume
                "min_price": 10.0,  # Minimum stock price
            },
        }

        # Validate required credentials
        if not config["alpaca"]["api_key"] or not config["alpaca"]["secret_key"]:
            raise ValueError("Missing Alpaca API credentials in environment")

        return config

    async def _init_database(self):
        """Initialize database connection"""
        db_config = self.config["database"]

        # Build connection string
        if db_config["password"]:
            conn_str = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        else:
            conn_str = f"postgresql://{db_config['user']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

        # Create adapter
        self.db_adapter = AsyncDatabaseAdapter({"connection_string": conn_str})

        # Test connection
        test_query = "SELECT COUNT(*) as count FROM companies WHERE is_active = true"
        result = await self.db_adapter.fetch_one(test_query)
        logger.info(f"Database connected. Active companies: {result['count']}")

    def _init_alpaca_clients(self):
        """Initialize Alpaca trading and data clients"""
        alpaca_config = self.config["alpaca"]

        # Trading client for orders
        self.trading_client = TradingClient(
            api_key=alpaca_config["api_key"],
            secret_key=alpaca_config["secret_key"],
            paper=("paper" in alpaca_config["base_url"]),
        )

        # Data client for market data
        self.data_client = StockHistoricalDataClient(
            api_key=alpaca_config["api_key"], secret_key=alpaca_config["secret_key"]
        )

        # Test connection
        account = self.trading_client.get_account()
        logger.info(
            f"Alpaca connected. Account: ${float(account.cash):.2f} cash, ${float(account.portfolio_value):.2f} total"
        )

    async def _load_active_symbols(self):
        """Load most liquid symbols from database"""
        query = """
            SELECT symbol, name, exchange, market_cap
            FROM companies
            WHERE is_active = true
                AND layer1_qualified = true
                AND exchange IN ('NYSE', 'NASDAQ')
            ORDER BY market_cap DESC NULLS LAST
            LIMIT %s
        """

        rows = await self.db_adapter.fetch_all(query, self.config["symbols"]["max_symbols"])

        self.active_symbols = [row["symbol"] for row in rows]
        logger.info(f"Loaded symbols: {', '.join(self.active_symbols)}")

    async def fetch_latest_prices(self) -> dict[str, Decimal]:
        """Fetch latest prices for active symbols"""
        if not self.active_symbols:
            return {}

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=self.active_symbols)
            quotes = self.data_client.get_stock_latest_quote(request)

            prices = {}
            for symbol, quote in quotes.items():
                # Use midpoint of bid/ask
                if quote.bid_price and quote.ask_price:
                    prices[symbol] = Decimal(str((quote.bid_price + quote.ask_price) / 2))
                else:
                    prices[symbol] = Decimal(str(quote.ask_price or quote.bid_price or 0))

            return prices

        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return {}

    async def generate_signals(self, prices: dict[str, Decimal]) -> list[dict]:
        """Generate simple trading signals based on momentum"""
        signals = []

        for symbol, current_price in prices.items():
            if current_price <= 0:
                continue

            # Get previous close from database
            query = """
                SELECT close
                FROM market_data_1h
                WHERE symbol = %s
                    AND interval = '1day'
                ORDER BY timestamp DESC
                LIMIT 1
            """

            result = await self.db_adapter.fetch_one(query, symbol)

            if result and result["close"]:
                prev_close = Decimal(str(result["close"]))
                change_pct = ((current_price - prev_close) / prev_close) * 100

                # Simple momentum signal
                if change_pct > 1.0:  # Up more than 1%
                    signals.append(
                        {
                            "symbol": symbol,
                            "action": "buy",
                            "price": current_price,
                            "reason": f"Momentum up {change_pct:.2f}%",
                            "confidence": min(0.8, 0.5 + abs(change_pct) * 0.1),
                        }
                    )
                elif change_pct < -1.0:  # Down more than 1%
                    signals.append(
                        {
                            "symbol": symbol,
                            "action": "sell",
                            "price": current_price,
                            "reason": f"Momentum down {change_pct:.2f}%",
                            "confidence": min(0.8, 0.5 + abs(change_pct) * 0.1),
                        }
                    )

        return signals

    async def execute_trade(self, signal: dict) -> str | None:
        """Execute a trade based on signal"""
        try:
            symbol = signal["symbol"]
            action = signal["action"]

            # Check existing position
            positions = self.trading_client.get_all_positions()
            has_position = any(p.symbol == symbol for p in positions)

            # Skip if we already have a position and signal is buy
            if has_position and action == "buy":
                logger.info(f"Already have position in {symbol}, skipping buy")
                return None

            # Skip if we don't have position and signal is sell
            if not has_position and action == "sell":
                logger.info(f"No position in {symbol}, skipping sell")
                return None

            # Calculate position size
            account = self.trading_client.get_account()
            max_position = min(
                float(self.config["trading"]["max_position_size"]),
                float(account.cash) * 0.1,  # Max 10% of cash per position
            )

            quantity = int(max_position / float(signal["price"]))

            if quantity <= 0:
                logger.warning(f"Position size too small for {symbol}")
                return None

            # Create order
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if action == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )

            # Submit order
            order = self.trading_client.submit_order(order_data)

            logger.info(f"✅ Order submitted: {action.upper()} {quantity} {symbol} @ market")

            # Store in database
            await self._record_trade(order, signal)

            return order.id

        except Exception as e:
            logger.error(f"❌ Trade execution failed for {signal['symbol']}: {e}")
            return None

    async def _record_trade(self, order: Any, signal: dict):
        """Record trade in database"""
        try:
            query = """
                INSERT INTO trades (
                    order_id, symbol, side, quantity,
                    price, signal_reason, confidence,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            await self.db_adapter.execute(
                query,
                order.id,
                order.symbol,
                str(order.side),
                order.qty,
                float(signal["price"]),
                signal["reason"],
                signal["confidence"],
                datetime.now(UTC),
            )
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")

    async def monitor_positions(self):
        """Monitor and manage existing positions"""
        try:
            positions = self.trading_client.get_all_positions()

            for position in positions:
                symbol = position.symbol
                entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price or 0)
                unrealized_pl_pct = float(position.unrealized_plpc or 0) * 100

                # Check stop loss
                if unrealized_pl_pct <= -self.config["trading"]["stop_loss_pct"] * 100:
                    logger.warning(f"🛑 Stop loss triggered for {symbol}: {unrealized_pl_pct:.2f}%")
                    await self._close_position(symbol, "stop_loss")

                # Check take profit
                elif unrealized_pl_pct >= self.config["trading"]["take_profit_pct"] * 100:
                    logger.info(f"💰 Take profit triggered for {symbol}: {unrealized_pl_pct:.2f}%")
                    await self._close_position(symbol, "take_profit")

                else:
                    logger.info(f"Position {symbol}: P&L {unrealized_pl_pct:.2f}%")

        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")

    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            position = self.trading_client.get_position(symbol)

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=abs(int(position.qty)),
                side=OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )

            order = self.trading_client.submit_order(order_data)
            logger.info(f"Position closed: {symbol} for {reason}")

        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")

    async def trading_loop(self):
        """Main trading loop"""
        self.running = True
        loop_count = 0

        while self.running:
            try:
                loop_count += 1
                logger.info(f"\n{'='*50}")
                logger.info(
                    f"Trading Loop #{loop_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

                # 1. Fetch latest prices
                prices = await self.fetch_latest_prices()
                logger.info(f"Fetched prices for {len(prices)} symbols")

                # 2. Generate signals
                signals = await self.generate_signals(prices)
                logger.info(f"Generated {len(signals)} signals")

                # 3. Execute high-confidence signals
                for signal in signals:
                    if signal["confidence"] >= 0.7:
                        await self.execute_trade(signal)

                # 4. Monitor existing positions
                await self.monitor_positions()

                # 5. Log account status
                account = self.trading_client.get_account()
                logger.info(
                    f"Account: Cash ${float(account.cash):.2f}, Portfolio ${float(account.portfolio_value):.2f}"
                )

                # Sleep for 60 seconds (1 minute intervals)
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)  # Short pause on error

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading bot...")
        self.running = False

        # Close database connection
        if self.db_adapter:
            await self.db_adapter.close()

        logger.info("✅ Bot shutdown complete")


async def main():
    """Main entry point"""
    bot = MinimalTradingBot()

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        bot.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize bot
    if not await bot.initialize():
        logger.error("Failed to initialize bot")
        return 1

    # Start trading
    logger.info("🚀 Starting trading loop...")
    try:
        await bot.trading_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await bot.shutdown()

    return 0


if __name__ == "__main__":
    # Load environment variables from .env file
    # Third-party imports
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")

    # Run the bot
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
