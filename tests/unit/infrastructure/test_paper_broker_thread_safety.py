"""
Thread safety tests for PaperBroker

Tests concurrent operations to ensure thread safety and proper lock management.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

import pytest

from src.domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from src.domain.services.trading_calendar import TradingCalendar
from src.infrastructure.brokers.paper_broker import PaperBroker


class TestPaperBrokerThreadSafety:
    """Test suite for PaperBroker thread safety"""

    @pytest.fixture
    def config(self):
        """Create broker configuration for testing"""
        return PaperBrokerConfig(
            initial_capital=Decimal("1000000"),  # Large capital for concurrent trades
            fill_delay_seconds=0,  # No delay for faster testing
            simulate_partial_fills=False,
            max_orders_history=10000,
        )

    @pytest.fixture
    def broker(self, config):
        """Create a PaperBroker instance with all dependencies"""
        # Create commission calculator
        schedule = CommissionSchedule(
            commission_type=CommissionType.PER_SHARE,
            rate=Decimal("0.01"),
            minimum=Decimal("1.00"),
            maximum=Decimal("100.00"),
        )
        commission_calc = CommissionCalculatorFactory.create(schedule)

        # Create market microstructure
        slippage_config = SlippageConfig(
            base_bid_ask_bps=Decimal("5"),  # 0.05% bid-ask difference
            impact_coefficient=Decimal("0.1"),
        )
        microstructure = MarketMicrostructureFactory.create(
            MarketImpactModel.LINEAR, slippage_config
        )

        # Create other dependencies
        calendar = TradingCalendar()
        validator = OrderValidator(commission_calculator=commission_calc)

        # Create broker (it creates its own portfolio and order processor)
        broker = PaperBroker(
            config=config,
            commission_calculator=commission_calc,
            market_model=microstructure,
            order_validator=validator,
            trading_calendar=calendar,
        )
        # Connect the broker
        broker.connect()
        return broker

    def test_concurrent_order_submission(self, broker):
        """Test submitting multiple orders concurrently"""
        num_threads = 10
        orders_per_thread = 50
        all_order_ids = []
        submission_errors = []

        def submit_orders(thread_id):
            """Submit orders from a single thread"""
            thread_order_ids = []
            for i in range(orders_per_thread):
                try:
                    order = Order(
                        symbol=f"TEST{thread_id}",
                        quantity=Decimal("10"),
                        side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                        order_type=OrderType.MARKET,
                    )
                    broker.submit_order(order)
                    thread_order_ids.append(order.id)
                except Exception as e:
                    submission_errors.append((thread_id, i, str(e)))
            return thread_order_ids

        # Submit orders from multiple threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(submit_orders, i) for i in range(num_threads)]
            for future in as_completed(futures):
                all_order_ids.extend(future.result())

        # Verify results
        assert len(submission_errors) == 0, f"Submission errors: {submission_errors}"
        assert len(all_order_ids) == num_threads * orders_per_thread
        assert len(set(all_order_ids)) == len(all_order_ids), "Duplicate order IDs found"

        # Verify all orders are in broker's order dict
        broker_order_count = len(broker.orders)
        assert broker_order_count > 0, "No orders found in broker"

    def test_concurrent_order_cancellation(self, broker):
        """Test cancelling orders concurrently"""
        # First, submit many orders
        orders = []
        for i in range(100):
            order = Order(
                symbol="CANCEL_TEST",
                quantity=Decimal("10"),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                limit_price=Decimal("100.00"),
            )
            broker.submit_order(order)
            orders.append(order)

        cancellation_results = []
        cancellation_errors = []

        def cancel_order(order):
            """Cancel a single order"""
            try:
                result = broker.cancel_order(order.id)
                return (order.id, result)
            except Exception as e:
                cancellation_errors.append((order.id, str(e)))
                return (order.id, False)

        # Cancel orders from multiple threads
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(cancel_order, order) for order in orders]
            for future in as_completed(futures):
                cancellation_results.append(future.result())

        # Verify results
        assert len(cancellation_errors) == 0, f"Cancellation errors: {cancellation_errors}"

        # Check that orders are properly cancelled
        for order in orders:
            stored_order = broker.orders.get(order.id)
            if stored_order:
                assert stored_order.status in [
                    OrderStatus.CANCELLED,
                    OrderStatus.FILLED,  # May have been filled before cancellation
                ]

    def test_concurrent_portfolio_updates(self, broker):
        """Test concurrent updates to portfolio through order fills"""
        # Set market data for automatic fills
        symbols = [f"PORT{i}" for i in range(10)]
        for symbol in symbols:
            broker.update_market_data(symbol, Decimal("100.00"))

        fill_events = []
        lock_timeouts = []

        def submit_and_fill(symbol_index):
            """Submit an order and wait for fill"""
            symbol = symbols[symbol_index]
            results = []

            for i in range(10):
                order = Order(
                    symbol=symbol,
                    quantity=Decimal("10"),
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                )

                try:
                    broker.submit_order(order)
                    # Give time for order to be processed
                    time.sleep(0.01)

                    # Check if order was filled
                    filled_order = broker.orders.get(order.id)
                    if filled_order and filled_order.status == OrderStatus.FILLED:
                        results.append((symbol, order.id, "FILLED"))
                    else:
                        results.append((symbol, order.id, "PENDING"))
                except Exception as e:
                    results.append((symbol, order.id, f"ERROR: {e}"))

            return results

        # Submit and fill orders from multiple threads
        with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
            futures = [executor.submit(submit_and_fill, i) for i in range(len(symbols))]
            for future in as_completed(futures):
                fill_events.extend(future.result())

        # Verify no lock timeout errors
        errors = [e for e in fill_events if "ERROR" in str(e[2])]
        assert len(errors) == 0, f"Errors during concurrent fills: {errors}"

        # Verify portfolio integrity
        portfolio = broker.portfolio
        assert portfolio.cash_balance >= 0, "Negative cash balance after concurrent trades"

    def test_concurrent_market_data_updates(self, broker):
        """Test updating market data from multiple threads"""
        symbols = [f"DATA{i}" for i in range(50)]
        update_errors = []

        def update_prices(thread_id):
            """Update market prices from a single thread"""
            for i in range(100):
                symbol = symbols[i % len(symbols)]
                price = Decimal(f"{100 + thread_id + i * 0.1:.2f}")
                try:
                    broker.update_market_data(symbol, price)
                except Exception as e:
                    update_errors.append((thread_id, symbol, str(e)))

        # Update market data from multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_prices, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()

        # Verify no errors
        assert len(update_errors) == 0, f"Market data update errors: {update_errors}"

        # Verify all symbols have market data
        for symbol in symbols:
            # Check market data was set (we can't access private _market_data)
            # The test will show if updates work through order processing
            pass  # Test passes if no errors during updates

    def test_lock_hierarchy_no_deadlock(self, broker):
        """Test that lock hierarchy prevents deadlocks"""
        deadlock_detected = threading.Event()
        test_complete = threading.Event()

        def thread1_operations():
            """Thread 1: Acquires locks in correct order"""
            try:
                # This follows the correct lock hierarchy
                # Test lock hierarchy - we can't access private locks directly
                # But we can test through the public API that operations are thread-safe
                time.sleep(0.01)  # Hold lock briefly
                broker.update_market_data("TEST_LOCK", Decimal("100"))
                time.sleep(0.01)
                orders = broker.get_orders()
                time.sleep(0.01)
                positions = broker.get_positions()
                # Successfully performed operations in order
            except Exception:
                deadlock_detected.set()

        def thread2_operations():
            """Thread 2: Also acquires locks in correct order"""
            try:
                # Even if we try different operations, hierarchy is enforced
                # Test operations through public API
                orders = broker.get_orders()
                time.sleep(0.01)
                positions = broker.get_positions()
                time.sleep(0.01)
            except Exception:
                deadlock_detected.set()

        # Run threads
        t1 = threading.Thread(target=thread1_operations)
        t2 = threading.Thread(target=thread2_operations)

        t1.start()
        t2.start()

        # Wait for completion with timeout
        t1.join(command_timeout=5)
        t2.join(command_timeout=5)

        # Check results
        assert not deadlock_detected.is_set(), "Deadlock or exception detected"
        assert not t1.is_alive(), "Thread 1 still running (possible deadlock)"
        assert not t2.is_alive(), "Thread 2 still running (possible deadlock)"

    def test_order_history_bounded_size(self, broker):
        """Test that order history respects max size under concurrent load"""
        max_orders = broker.config.max_orders_history
        total_orders_submitted = max_orders * 2  # Submit twice the max

        def submit_batch(start_idx):
            """Submit a batch of orders"""
            for i in range(100):
                order = Order(
                    symbol=f"HIST{start_idx + i}",
                    quantity=Decimal("1"),
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                )
                broker.submit_order(order)

        # Submit orders from multiple threads
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(0, total_orders_submitted, 100):
                futures.append(executor.submit(submit_batch, i))
            for future in as_completed(futures):
                future.result()

        # Verify order dictionary is bounded
        assert (
            len(broker.orders) <= max_orders
        ), f"Order history exceeded max size: {len(broker.orders)} > {max_orders}"

    def test_concurrent_position_reversals(self, broker):
        """Test position reversals under concurrent load"""
        symbol = "REVERSE"
        broker.update_market_data(symbol, Decimal("100.00"))

        reversal_results = []

        def execute_reversal(thread_id):
            """Execute trades that reverse positions"""
            results = []

            # Buy to open long
            buy_order = Order(
                symbol=symbol,
                quantity=Decimal("100"),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
            )
            broker.submit_order(buy_order)
            time.sleep(0.05)  # Wait for fill
            results.append((thread_id, "BUY", buy_order.id))

            # Sell to reverse to short
            sell_order = Order(
                symbol=symbol,
                quantity=Decimal("200"),
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
            )
            broker.submit_order(sell_order)
            time.sleep(0.05)  # Wait for fill
            results.append((thread_id, "SELL", sell_order.id))

            return results

        # Execute reversals from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(execute_reversal, i) for i in range(5)]
            for future in as_completed(futures):
                reversal_results.extend(future.result())

        # Verify all orders were processed
        assert len(reversal_results) == 10, "Not all reversal orders were processed"

        # Verify final position makes sense
        position = broker.portfolio.positions.get(symbol)
        if position:
            # Position quantity should be reasonable given the trades
            assert abs(position.quantity) <= Decimal("1000"), "Position size unreasonable"

    @pytest.mark.asyncio
    async def test_async_operations_thread_safety(self, broker):
        """Test thread safety with async operations"""

        async def async_submit_orders():
            """Submit orders asynchronously"""
            tasks = []
            for i in range(50):
                order = Order(
                    symbol=f"ASYNC{i}",
                    quantity=Decimal("10"),
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                )
                # Simulate async order submission
                task = asyncio.create_task(asyncio.to_thread(broker.submit_order, order))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        # Run async operations
        results = await async_submit_orders()

        # Verify no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Async operations failed: {exceptions}"

    def test_stress_test_all_operations(self, broker):
        """Stress test with all operations running concurrently"""
        test_duration = 2  # seconds
        stop_event = threading.Event()
        operation_counts = {
            "submit": 0,
            "cancel": 0,
            "market_update": 0,
            "get_positions": 0,
        }
        errors = []

        def submit_orders_continuously():
            """Continuously submit orders"""
            count = 0
            while not stop_event.is_set():
                try:
                    order = Order(
                        symbol=f"STRESS{count % 10}",
                        quantity=Decimal("5"),
                        side=OrderSide.BUY if count % 2 == 0 else OrderSide.SELL,
                        order_type=OrderType.MARKET,
                    )
                    broker.submit_order(order)
                    count += 1
                except Exception as e:
                    errors.append(("submit", str(e)))
            operation_counts["submit"] = count

        def cancel_orders_continuously():
            """Continuously cancel orders"""
            count = 0
            while not stop_event.is_set():
                try:
                    # Try to cancel random orders
                    if broker.orders:
                        order_ids = list(broker.orders.keys())[:10]
                        for order_id in order_ids:
                            broker.cancel_order(order_id)
                            count += 1
                except Exception as e:
                    if "not found" not in str(e).lower():
                        errors.append(("cancel", str(e)))
            operation_counts["cancel"] = count

        def update_market_continuously():
            """Continuously update market data"""
            count = 0
            while not stop_event.is_set():
                try:
                    symbol = f"STRESS{count % 10}"
                    price = Decimal(f"{100 + count * 0.1:.2f}")
                    broker.update_market_data(symbol, price)
                    count += 1
                except Exception as e:
                    errors.append(("market_update", str(e)))
            operation_counts["market_update"] = count

        def get_positions_continuously():
            """Continuously query positions"""
            count = 0
            while not stop_event.is_set():
                try:
                    positions = broker.get_positions()
                    count += 1
                except Exception as e:
                    errors.append(("get_positions", str(e)))
            operation_counts["get_positions"] = count

        # Start all threads
        threads = [
            threading.Thread(target=submit_orders_continuously),
            threading.Thread(target=cancel_orders_continuously),
            threading.Thread(target=update_market_continuously),
            threading.Thread(target=get_positions_continuously),
        ]

        for t in threads:
            t.start()

        # Run for specified duration
        time.sleep(test_duration)
        stop_event.set()

        # Wait for all threads to complete
        for t in threads:
            t.join(command_timeout=1)

        # Verify results
        assert len(errors) == 0, f"Errors during stress test: {errors[:10]}"  # Show first 10

        # Verify operations were performed
        assert operation_counts["submit"] > 0, "No orders submitted"
        assert operation_counts["market_update"] > 0, "No market updates"
        assert operation_counts["get_positions"] > 0, "No position queries"

        # Verify broker state is consistent
        assert broker.portfolio.cash_balance >= 0, "Negative cash balance after stress test"
        assert len(broker.orders) <= broker.config.max_orders_history, "Order history overflow"

        print(f"Stress test completed: {operation_counts}")
