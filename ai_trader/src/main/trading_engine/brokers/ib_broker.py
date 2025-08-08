"""
Interactive Brokers implementation using ib_insync.
Provides connection to IB TWS/Gateway for live and paper trading.
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
import pandas as pd

# Import base interface and common models
from .broker_interface import BrokerInterface
from main.models.common import (
    Order, Position, AccountInfo, MarketData,
    OrderStatus, OrderType, OrderSide, TimeInForce
)
from main.config.config_manager import get_config
from main.utils.core import ErrorHandlingMixin

logger = logging.getLogger(__name__)

# Try to import ib_insync - will be optional
try:
    from ib_insync import IB, Stock, Order as IBOrder, Contract, util
    IB_AVAILABLE = True
except ImportError:
    logger.warning("ib_insync not installed. IB broker will not be available.")
    IB_AVAILABLE = False
    # Create placeholder types for type hints when ib_insync is not available
    Contract = type('Contract', (), {})
    IB = type('IB', (), {})
    Stock = type('Stock', (), {})
    IBOrder = type('IBOrder', (), {})
    util = type('util', (), {})


class IBBroker(BrokerInterface, ErrorHandlingMixin):
    """
    Interactive Brokers broker implementation.
    
    Features:
    - Live and paper trading support
    - Real-time market data streaming
    - Advanced order types
    - Portfolio margin support
    - Multi-currency support
    - Options and futures trading
    """
    
    def __init__(self, config: Any = None):
        """
        Initialize IB broker.
        
        Args:
            config: Configuration object
        """
        if config is None:
            config = get_config()
        super().__init__(config)
        ErrorHandlingMixin.__init__(self)
        
        if not IB_AVAILABLE:
            raise ImportError("ib_insync package is required for IB broker. Install with: pip install ib_insync")
        
        # IB configuration
        self.host = config.get('brokers.ib.host', '127.0.0.1')
        self.port = config.get('brokers.ib.port', 7497)  # 7497 for paper, 7496 for live
        self.client_id = config.get('brokers.ib.client_id', 1)
        self.account_id = config.get('brokers.ib.account_id')
        
        # IB connection
        self.ib = IB()
        
        # Contract cache and position cache via MarketDataCache
        from main.utils.cache import MemoryBackend
        from main.utils.cache import CacheType
        self.cache = get_global_cache()
        
        # Order mapping (our order ID -> IB order)
        self._order_map: Dict[str, IBOrder] = {}
        self._ib_order_map: Dict[int, Order] = {}  # IB order ID -> our order
        
        # Subscribed symbols for market data
        self._subscribed_symbols: Set[str] = set()
        self._market_data_subscriptions: Dict[str, Any] = {}
        
        # Account info cache
        self._account_info_cache: Optional[AccountInfo] = None
        self._last_account_update = datetime.min
        
        logger.info(f"IBBroker initialized for {self.host}:{self.port}")
    
    async def _create_contract(self, symbol: str) -> Contract:
        """
        Create IB contract for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            IB Contract object
        """
        cached_contract = await self.cache.get(CacheType.CUSTOM, f"ib_contract:{symbol}")
        if cached_contract:
            return cached_contract
        
        # For now, assume US stocks
        # Could be extended to support options, futures, forex, etc.
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Qualify the contract to get full details
        self.ib.qualifyContracts(contract)
        
        await self.cache.set(CacheType.CUSTOM, f"ib_contract:{symbol}", contract, 3600)  # Cache for 1 hour
        return contract
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert our order type to IB order type."""
        type_map = {
            OrderType.MARKET: 'MKT',
            OrderType.LIMIT: 'LMT',
            OrderType.STOP: 'STP',
            OrderType.STOP_LIMIT: 'STP LMT',
            OrderType.TRAILING_STOP: 'TRAIL'
        }
        return type_map.get(order_type, 'MKT')
    
    def _convert_time_in_force(self, tif: TimeInForce) -> str:
        """Convert our TIF to IB TIF."""
        tif_map = {
            TimeInForce.DAY: 'DAY',
            TimeInForce.GTC: 'GTC',
            TimeInForce.IOC: 'IOC',
            TimeInForce.FOK: 'FOK',
            TimeInForce.GTX: 'GTX',
            TimeInForce.OPG: 'OPG',
            TimeInForce.CLS: 'MOC'  # Market on close
        }
        return tif_map.get(tif, 'DAY')
    
    def _convert_order_status(self, ib_status: str) -> OrderStatus:
        """Convert IB order status to our status."""
        status_map = {
            'PendingSubmit': OrderStatus.PENDING,
            'PendingCancel': OrderStatus.PENDING,
            'PreSubmitted': OrderStatus.PENDING,
            'Submitted': OrderStatus.SUBMITTED,
            'Filled': OrderStatus.FILLED,
            'Cancelled': OrderStatus.CANCELLED,
            'Inactive': OrderStatus.REJECTED
        }
        return status_map.get(ib_status, OrderStatus.UNKNOWN)
    
    def _create_ib_order(self, order: Order) -> IBOrder:
        """
        Create IB order from our order.
        
        Args:
            order: Our order object
            
        Returns:
            IB Order object
        """
        ib_order = IBOrder()
        
        # Basic order attributes
        ib_order.action = 'BUY' if order.side == OrderSide.BUY else 'SELL'
        ib_order.totalQuantity = order.quantity
        ib_order.orderType = self._convert_order_type(order.order_type)
        ib_order.tif = self._convert_time_in_force(order.time_in_force)
        
        # Price attributes
        if order.limit_price:
            ib_order.lmtPrice = order.limit_price
        if order.stop_price:
            ib_order.auxPrice = order.stop_price  # Stop price in IB
        
        # Trailing stop
        if order.order_type == OrderType.TRAILING_STOP:
            if order.trail_percent:
                ib_order.trailingPercent = order.trail_percent
            elif order.trail_price:
                ib_order.trailStopPrice = order.trail_price
        
        # Additional attributes
        if order.client_order_id:
            ib_order.orderRef = order.client_order_id
        
        # Smart routing
        ib_order.algoStrategy = 'Adaptive'
        ib_order.algoParams = [
            ('adaptivePriority', 'Normal')
        ]
        
        return ib_order
    
    async def connect(self) -> bool:
        """Connect to IB TWS/Gateway."""
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            
            if not self.ib.isConnected():
                raise ConnectionError("Failed to connect to IB")
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Request initial account data
            self.ib.reqAccountUpdates(subscribe=True, account=self.account_id or '')
            
            # Request positions
            self.ib.reqPositions()
            
            self._connected = True
            logger.info(f"Connected to IB at {self.host}:{self.port}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            return False
    
    def _setup_event_handlers(self):
        """Set up IB event handlers."""
        # Order events
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details
        
        # Position events
        self.ib.positionEvent += self._on_position
        
        # Account events
        self.ib.accountValueEvent += self._on_account_value
        
        # Error events
        self.ib.errorEvent += self._on_error
        
        # Market data events
        self.ib.pendingTickersEvent += self._on_pending_tickers
    
    def _on_order_status(self, trade):
        """Handle order status updates from IB."""
        ib_order = trade.order
        
        # Find our order
        our_order = self._ib_order_map.get(ib_order.orderId)
        if not our_order:
            logger.warning(f"Received status for unknown order: {ib_order.orderId}")
            return
        
        # Update order status
        our_order.status = self._convert_order_status(trade.orderStatus.status)
        our_order.filled_quantity = trade.orderStatus.filled
        our_order.remaining_quantity = trade.orderStatus.remaining
        
        if trade.orderStatus.avgFillPrice:
            our_order.average_fill_price = trade.orderStatus.avgFillPrice
        
        # Update timestamps
        if our_order.status == OrderStatus.FILLED:
            our_order.filled_at = datetime.now(timezone.utc)
        elif our_order.status == OrderStatus.CANCELLED:
            our_order.cancelled_at = datetime.now(timezone.utc)
        
        # Store status message
        if trade.orderStatus.whyHeld:
            our_order.status_message = trade.orderStatus.whyHeld
        
        # Trigger callbacks
        self._trigger_order_callbacks(our_order)
        
        logger.info(f"Order {our_order.broker_order_id} status: {our_order.status}")
    
    def _on_exec_details(self, trade, fill):
        """Handle execution details from IB."""
        # Find our order
        our_order = self._ib_order_map.get(fill.execution.orderId)
        if not our_order:
            return
        
        # Update commission if available
        if fill.commissionReport:
            our_order.commission = fill.commissionReport.commission
        
        logger.info(f"Execution: {fill.execution.side} {fill.execution.shares} "
                   f"{fill.contract.symbol} @ {fill.execution.price}")
    
    def _on_position(self, position):
        """Handle position updates from IB."""
        if position.position == 0:
            # Position closed - schedule cache deletion
            asyncio.create_task(self.cache.delete(CacheType.POSITIONS, position.contract.symbol))
            return
        
        # Create/update position
        pos = Position(
            symbol=position.contract.symbol,
            quantity=position.position,
            average_price=position.avgCost / position.contract.multiplier,
            current_price=0.0,  # Will be updated from market data
            market_value=position.position * position.avgCost,
            unrealized_pnl=0.0,  # Will be calculated
            realized_pnl=0.0  # IB doesn't provide this directly
        )
        
        # Cache the position
        asyncio.create_task(self.cache.set(CacheType.POSITIONS, position.contract.symbol, pos, 300))  # Cache for 5 minutes
    
    def _on_account_value(self, value):
        """Handle account value updates from IB."""
        # Cache account values for building AccountInfo
        # This would need more sophisticated handling for full implementation
        pass
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle errors from IB."""
        logger.error(f"IB Error {errorCode}: {errorString}")
        
        # Handle order errors
        if reqId > 0 and reqId in self._ib_order_map:
            our_order = self._ib_order_map[reqId]
            our_order.status = OrderStatus.REJECTED
            our_order.status_message = f"Error {errorCode}: {errorString}"
            self._trigger_order_callbacks(our_order)
    
    async def _on_pending_tickers(self, tickers):
        """Handle market data updates."""
        for ticker in tickers:
            if ticker.contract.symbol in self._subscribed_symbols:
                # Create market data
                market_data = MarketData(
                    symbol=ticker.contract.symbol,
                    timestamp=datetime.now(timezone.utc),
                    bid=ticker.bid if not util.isNan(ticker.bid) else 0.0,
                    ask=ticker.ask if not util.isNan(ticker.ask) else 0.0,
                    last=ticker.last if not util.isNan(ticker.last) else 0.0,
                    volume=ticker.volume if not util.isNan(ticker.volume) else 0,
                    open=ticker.open if not util.isNan(ticker.open) else 0.0,
                    high=ticker.high if not util.isNan(ticker.high) else 0.0,
                    low=ticker.low if not util.isNan(ticker.low) else 0.0,
                    close=ticker.close if not util.isNan(ticker.close) else 0.0
                )
                
                # Trigger callbacks
                self._trigger_market_data_callbacks(market_data)
                
                # Update position current prices
                cached_pos = await self.cache.get(CacheType.POSITIONS, ticker.contract.symbol)
                if cached_pos:
                    cached_pos.current_price = market_data.last
                    cached_pos.market_value = cached_pos.quantity * market_data.last
                    cached_pos.unrealized_pnl = (market_data.last - cached_pos.average_price) * cached_pos.quantity
                    # Update the cache with modified position
                    await self.cache.set(CacheType.POSITIONS, ticker.contract.symbol, cached_pos, 300)
    
    async def disconnect(self) -> None:
        """Disconnect from IB."""
        if self.ib.isConnected():
            # Unsubscribe from all market data
            for ticker in self._market_data_subscriptions.values():
                self.ib.cancelMktData(ticker)
            
            # Cancel account updates
            self.ib.reqAccountUpdates(subscribe=False)
            
            # Disconnect
            self.ib.disconnect()
        
        self._connected = False
        logger.info("Disconnected from IB")
    
    async def submit_order(self, order: Order) -> Order:
        """
        Submit order to IB.
        
        Args:
            order: Order to submit
            
        Returns:
            Updated order with broker order ID
        """
        try:
            # Validate order
            self.validate_order(order)
            
            # Create IB contract and order
            contract = await self._create_contract(order.symbol)
            ib_order = self._create_ib_order(order)
            
            # Place order with IB
            trade = self.ib.placeOrder(contract, ib_order)
            
            # Store order mapping
            order.broker_order_id = str(trade.order.orderId)
            self._order_map[order.order_id] = trade.order
            self._ib_order_map[trade.order.orderId] = order
            
            # Update order
            order.submitted_at = datetime.now(timezone.utc)
            order.status = OrderStatus.SUBMITTED
            
            # Store in our orders dict
            self._orders[order.broker_order_id] = order
            
            logger.info(f"Submitted order to IB: {order.broker_order_id}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to submit order to IB: {e}")
            order.status = OrderStatus.REJECTED
            order.status_message = str(e)
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.
        
        Args:
            order_id: Broker order ID to cancel
            
        Returns:
            True if cancellation request submitted
        """
        if order_id not in self._orders:
            logger.warning(f"Order {order_id} not found")
            return False
        
        our_order = self._orders[order_id]
        ib_order_id = int(order_id)
        
        try:
            # Cancel with IB
            self.ib.cancelOrder(self._order_map.get(our_order.order_id))
            
            logger.info(f"Cancellation requested for order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def modify_order(self, order_id: str, 
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          quantity: Optional[float] = None) -> Order:
        """
        Modify existing order.
        
        Args:
            order_id: Broker order ID to modify
            limit_price: New limit price
            stop_price: New stop price
            quantity: New quantity
            
        Returns:
            Modified order
        """
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")
        
        our_order = self._orders[order_id]
        ib_order = self._order_map.get(our_order.order_id)
        
        if not ib_order:
            raise ValueError(f"IB order not found for {order_id}")
        
        # Apply modifications to IB order
        if limit_price is not None:
            ib_order.lmtPrice = limit_price
            our_order.limit_price = limit_price
        if stop_price is not None:
            ib_order.auxPrice = stop_price
            our_order.stop_price = stop_price
        if quantity is not None:
            ib_order.totalQuantity = quantity
            our_order.quantity = quantity
        
        # Get contract
        contract = await self._create_contract(our_order.symbol)
        
        # Modify order with IB
        self.ib.placeOrder(contract, ib_order)
        
        our_order.modified_at = datetime.now(timezone.utc)
        
        logger.info(f"Modified order {order_id}")
        return our_order
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        # Request fresh position data
        self.ib.reqPositions()
        
        # Wait a bit for updates
        await asyncio.sleep(0.1)
        
        # Get all positions from cache
        # In a full implementation, we'd need to track position keys
        # For now, return empty dict as we don't have a way to enumerate cache keys for positions
        return {}
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        positions = await self.get_positions()
        return positions.get(symbol)
    
    async def get_orders(self, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get orders, optionally filtered by status."""
        # Request fresh order data
        self.ib.reqAllOpenOrders()
        
        # Wait a bit for updates
        await asyncio.sleep(0.1)
        
        orders = list(self._orders.values())
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        return orders
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get specific order by ID."""
        return self._orders.get(order_id)
    
    async def get_recent_orders(self) -> List[Order]:
        """Get recently executed orders."""
        # Request completed orders for today
        self.ib.reqCompletedOrders(apiOnly=True)
        
        # Wait for response
        await asyncio.sleep(0.5)
        
        # Return orders from our cache
        recent_orders = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for order in self._orders.values():
            if order.submitted_at and order.submitted_at > cutoff:
                recent_orders.append(order)
        
        return sorted(recent_orders, key=lambda o: o.submitted_at or datetime.min, reverse=True)
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        # Request fresh account data
        account_values = self.ib.accountValues(account=self.account_id or '')
        account_summary = self.ib.accountSummary(account=self.account_id or '')
        
        # Parse account values
        values_dict = {item.tag: item.value for item in account_values}
        summary_dict = {item.tag: item.value for item in account_summary}
        
        # Build AccountInfo
        return AccountInfo(
            account_id=self.account_id or self.ib.client.clientId,
            buying_power=float(values_dict.get('BuyingPower', 0)),
            cash=float(values_dict.get('CashBalance', 0)),
            portfolio_value=float(values_dict.get('NetLiquidation', 0)),
            pattern_day_trader=values_dict.get('PatternDayTrader') == 'true',
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            trade_suspended_by_user=False,
            multiplier=float(values_dict.get('Leverage-S', 1.0)),
            shorting_enabled=True,
            equity=float(values_dict.get('Equity', 0)),
            last_equity=float(values_dict.get('EquityWithLoanValue', 0)),
            initial_margin=float(values_dict.get('InitMarginReq', 0)),
            maintenance_margin=float(values_dict.get('MaintMarginReq', 0)),
            daytrade_count=int(summary_dict.get('DayTradesRemaining', 0)),
            balance_asof=datetime.now(timezone.utc)
        )
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol."""
        contract = await self._create_contract(symbol)
        
        # Get snapshot data
        ticker = self.ib.reqMktData(contract, '', snapshot=True)
        
        # Wait for data
        await asyncio.sleep(0.5)
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            bid=ticker.bid if not util.isNan(ticker.bid) else 0.0,
            ask=ticker.ask if not util.isNan(ticker.ask) else 0.0,
            last=ticker.last if not util.isNan(ticker.last) else 0.0,
            volume=ticker.volume if not util.isNan(ticker.volume) else 0,
            open=ticker.open if not util.isNan(ticker.open) else 0.0,
            high=ticker.high if not util.isNan(ticker.high) else 0.0,
            low=ticker.low if not util.isNan(ticker.low) else 0.0,
            close=ticker.close if not util.isNan(ticker.close) else 0.0
        )
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote for symbol."""
        market_data = await self.get_market_data(symbol)
        
        return {
            'symbol': symbol,
            'timestamp': market_data.timestamp.isoformat(),
            'bid': market_data.bid,
            'ask': market_data.ask,
            'bid_size': 100,  # Would need Level 2 data for real sizes
            'ask_size': 100,
            'last': market_data.last,
            'volume': market_data.volume
        }
    
    async def subscribe_market_data(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data."""
        for symbol in symbols:
            if symbol not in self._subscribed_symbols:
                contract = await self._create_contract(symbol)
                ticker = self.ib.reqMktData(contract, '', snapshot=False)
                
                self._subscribed_symbols.add(symbol)
                self._market_data_subscriptions[symbol] = ticker
                
                logger.info(f"Subscribed to market data for {symbol}")
    
    async def unsubscribe_market_data(self, symbols: List[str]) -> None:
        """Unsubscribe from market data."""
        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                ticker = self._market_data_subscriptions.get(symbol)
                if ticker:
                    self.ib.cancelMktData(ticker)
                
                self._subscribed_symbols.discard(symbol)
                self._market_data_subscriptions.pop(symbol, None)
                
                logger.info(f"Unsubscribed from market data for {symbol}")
    
    async def get_historical_data(self, 
                                 symbol: str,
                                 start: datetime,
                                 end: datetime,
                                 timeframe: str = "1Day") -> List[Dict[str, Any]]:
        """Get historical market data."""
        contract = await self._create_contract(symbol)
        
        # Convert timeframe
        bar_size_map = {
            "1Min": "1 min",
            "5Min": "5 mins",
            "15Min": "15 mins",
            "1Hour": "1 hour",
            "1Day": "1 day"
        }
        bar_size = bar_size_map.get(timeframe, "1 day")
        
        # Calculate duration
        duration_seconds = int((end - start).total_seconds())
        if duration_seconds < 86400:  # Less than 1 day
            duration = f"{duration_seconds} S"
        else:
            duration_days = duration_seconds // 86400
            duration = f"{duration_days} D"
        
        # Request historical data
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True
        )
        
        # Convert to our format
        result = []
        for bar in bars:
            result.append({
                'timestamp': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
        
        return result
    
    async def get_tradable_symbols(self) -> List[str]:
        """Get list of tradable symbols."""
        # This would require scanning IB's symbol database
        # For now, return empty list
        logger.warning("get_tradable_symbols not fully implemented for IB")
        return []
    
    async def is_tradable(self, symbol: str) -> bool:
        """Check if symbol is tradable."""
        try:
            contract = await self._create_contract(symbol)
            details = self.ib.reqContractDetails(contract)
            return len(details) > 0 and details[0].tradingClass != ''
        except (ConnectionError, TimeoutError, AttributeError, IndexError) as e:
            logger.debug(f"Failed to check if symbol {symbol} is tradable: {e}")
            return False