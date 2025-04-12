import json
from typing import Any, Dict, List, Tuple
import math
import jsonpickle
import numpy as np
from collections import deque

# Import required classes from datamodel
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId

# Logger boilerplate
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()

# Trader algorithm class
class Trader:
    def __init__(self):
        # Mean reversion parameters for SQUID_INK
        self.squid_ma_window = 22         # Window for calculating moving average
        self.squid_entry_threshold = 2.68  # Z-score threshold for entry
        self.squid_exit_threshold = 0.53   # Z-score threshold for exit
        self.squid_stop_loss_threshold = 4.53  # Z-score threshold for stop loss
        self.squid_price_stop_loss_pct = 0.005349  # Price-based stop-loss percentage (0.5%)
        self.squid_trend_window = 73      # Window for trend detection
        self.squid_trend_threshold = 0.029  # Trend strength to avoid trading against
        self.squid_trade_cooldown = 2     # Minimum ticks between trades
        self.squid_volatility_lookback = 74  # Period for volatility calculation
        
        # Price history 
        self.price_history = {
            "SQUID_INK": deque(maxlen=300)  # Store up to 300 price points
        }
        
        # Trade state tracking
        self.trade_state = {
            "SQUID_INK": {
                "positions": [],  # For tracking individual positions
                "last_trade_time": 0
            }
        }
    
    def calculate_ema_z_score(self, current_price: float, prices: List[float], period: int) -> Tuple[float, float]:
        """
        Compute z-score using exponentially weighted mean and standard deviation.
        Both mean and variance are computed with same decay (alpha = 2 / (period + 1)).
        """
        if len(prices) < period:
            return 0.0, None

        alpha = 2 / (period + 1)
        ema = prices[0]
        ew_var = 0.0

        # Step through prices to compute EMA and exponentially weighted variance
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
            deviation = price - ema
            ew_var = alpha * (deviation ** 2) + (1 - alpha) * ew_var

        std_dev = math.sqrt(ew_var) if ew_var > 0 else 0.0001  # Avoid division by zero

        z_score = (current_price - ema) / std_dev
        return z_score, ema
    
    def detect_trend(self, prices: List[float], period: int) -> float:
        """Detect trend using linear regression slope"""
        if len(prices) < period:
            return 0
        
        # Simple linear regression slope estimation
        x = np.array(range(period))
        y = np.array(prices[-period:])
        
        # Calculate slope using least squares formula
        n = len(x)
        slope = (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / (n * np.sum(x*x) - np.sum(x)**2)
        
        # Normalize by average price to get percentage move
        avg_price = np.mean(y)
        normalized_slope = slope * period / avg_price if avg_price != 0 else 0
        
        return normalized_slope
    
    def calculate_volatility(self, prices: List[float], period: int) -> float:
        """Calculate price volatility as standard deviation of returns"""
        if len(prices) < period + 1:
            return 0.01  # Default volatility estimate
        
        # Calculate percentage returns
        returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))][-period:]
        
        # Standard deviation of returns
        return np.std(returns) if len(returns) > 0 else 0.01
    
    def calculate_order_book_imbalance(self, order_depth: OrderDepth) -> float:
        """Calculate order book imbalance as a ratio of buy orders to sell orders"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
        
        total_buy_volume = sum(order_depth.buy_orders.values())
        total_sell_volume = abs(sum(order_depth.sell_orders.values()))
        
        if total_buy_volume + total_sell_volume == 0:
            return 0.0
        
        imbalance = (total_buy_volume - total_sell_volume) / (total_buy_volume + total_sell_volume)
        
        return imbalance
    
    def squid_ink_mean_reversion(
        self, order_depth: OrderDepth, symbol: str, position: int, position_limit: int, timestamp: int
    ) -> List[Order]:
        """Mean reversion strategy for SQUID_INK"""
        orders = []
        
        # Safety check for order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
            
        # Get current market data
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Calculate order book imbalance
        order_book_imbalance = self.calculate_order_book_imbalance(order_depth)
        
        # Update price history
        self.price_history["SQUID_INK"].append(mid_price)
        prices = list(self.price_history["SQUID_INK"])
        
        # Need enough history
        if len(prices) < self.squid_ma_window:
            logger.print(f"Building price history: {len(prices)}/{self.squid_ma_window}")
            return []
        
        # Calculate z-score using exponentially weighted method
        z_score, ema = self.calculate_ema_z_score(mid_price, prices, self.squid_ma_window)
        
        # Detect trend 
        trend = self.detect_trend(prices, self.squid_trend_window)
        
        # Calculate volatility for position sizing
        volatility = self.calculate_volatility(prices, self.squid_volatility_lookback)
        
        # Log market metrics
        logger.print(f"SQUID_INK: price={mid_price:.2f}, EMA={ema:.2f}, z-score={z_score:.2f}, trend={trend:.4f}, vol={volatility:.4f}")
        logger.print(f"SQUID_INK: order book imbalance: {order_book_imbalance:.4f}, spread: {spread}")
        
        # Get trade state
        trade_state = self.trade_state["SQUID_INK"]
        cooldown_elapsed = timestamp - trade_state["last_trade_time"] >= self.squid_trade_cooldown
        
        # Execute mean reversion trades
        if cooldown_elapsed:
            # Only trade if not against a strong trend
            trade_with_trend = (trend < self.squid_trend_threshold and z_score > self.squid_entry_threshold) or \
                               (trend > -self.squid_trend_threshold and z_score < -self.squid_entry_threshold)
                               
            if trade_with_trend:
                if z_score > self.squid_entry_threshold and position > -position_limit * 0.8:
                    # Price is too high - SELL signal
                    sell_size = self.calc_position_size(abs(z_score), position_limit, volatility)
                    sell_quantity = min(sell_size, position_limit + position)
                    
                    if sell_quantity > 0:
                        orders.append(Order(symbol, best_bid, -sell_quantity))
                        logger.print(f"SQUID_INK: SELLING {sell_quantity} at {best_bid} (z-score: {z_score:.2f}, mean reversion)")
                        
                        # Record trade
                        trade_state["positions"].append({
                            "type": "SHORT",
                            "entry_price": best_bid,
                            "quantity": sell_quantity,
                            "entry_time": timestamp,
                            "z_score": z_score,
                            "target_price": ema  # Target the EMA
                        })
                        trade_state["last_trade_time"] = timestamp
                        
                elif z_score < -self.squid_entry_threshold and position < position_limit * 0.8:
                    # Price is too low - BUY signal
                    buy_size = self.calc_position_size(abs(z_score), position_limit, volatility)
                    buy_quantity = min(buy_size, position_limit - position)
                    
                    if buy_quantity > 0:
                        orders.append(Order(symbol, best_ask, buy_quantity))
                        logger.print(f"SQUID_INK: BUYING {buy_quantity} at {best_ask} (z-score: {z_score:.2f}, mean reversion)")
                        
                        # Record trade
                        trade_state["positions"].append({
                            "type": "LONG",
                            "entry_price": best_ask,
                            "quantity": buy_quantity,
                            "entry_time": timestamp,
                            "z_score": z_score,
                            "target_price": ema  # Target the EMA
                        })
                        trade_state["last_trade_time"] = timestamp
        
        # Manage existing positions - take profit, stop loss, or reduce risk
        if trade_state["positions"]:
            # Check each position
            new_positions = []
            for pos in trade_state["positions"]:
                if pos["type"] == "LONG":
                    # STOP LOSS CONDITION - z-score moved more negative or price dropped too much
                    if z_score < -self.squid_stop_loss_threshold or mid_price < pos["entry_price"] * (1 - self.squid_price_stop_loss_pct):
                        # Sell to close the long position - stop loss
                        sell_quantity = pos["quantity"]
                        orders.append(Order(symbol, best_bid, -sell_quantity))
                        logger.print(f"SQUID_INK: STOP LOSS - Selling {sell_quantity} at {best_bid} (z-score: {z_score:.2f})")
                    
                    # PROFIT TAKING - Price reverted toward mean 
                    elif mid_price >= pos["target_price"] or z_score >= -self.squid_exit_threshold:
                        # Sell to close the long position - take profit
                        sell_quantity = pos["quantity"]
                        orders.append(Order(symbol, best_bid, -sell_quantity))
                        logger.print(f"SQUID_INK: PROFIT TAKING - Selling {sell_quantity} at {best_bid} (target reached)")
                    else:
                        # Keep the position
                        new_positions.append(pos)
                
                elif pos["type"] == "SHORT":
                    # STOP LOSS CONDITION - z-score moved more positive or price increased too much
                    if z_score > self.squid_stop_loss_threshold or mid_price > pos["entry_price"] * (1 + self.squid_price_stop_loss_pct):
                        # Buy to close the short position - stop loss
                        buy_quantity = pos["quantity"]
                        orders.append(Order(symbol, best_ask, buy_quantity))
                        logger.print(f"SQUID_INK: STOP LOSS - Buying {buy_quantity} at {best_ask} (z-score: {z_score:.2f})")
                    
                    # PROFIT TAKING - Price reverted toward mean
                    elif mid_price <= pos["target_price"] or z_score <= self.squid_exit_threshold:
                        # Buy to close the short position - take profit
                        buy_quantity = pos["quantity"]
                        orders.append(Order(symbol, best_ask, buy_quantity))
                        logger.print(f"SQUID_INK: PROFIT TAKING - Buying {buy_quantity} at {best_ask} (target reached)")
                    else:
                        # Keep the position
                        new_positions.append(pos)
            
            # Update positions list
            trade_state["positions"] = new_positions
        
        # Manage extreme positions - if we're close to position limit, actively reduce
        position_pct = abs(position) / position_limit
        if position_pct > 0.8:
            if position > 0:  # We have a large long position
                # Calculate how much to reduce
                reduce_quantity = int(position * 0.3)  # Reduce by 30%
                if reduce_quantity > 0:
                    orders.append(Order(symbol, best_bid, -reduce_quantity))
                    logger.print(f"SQUID_INK: RISK MANAGEMENT - Reducing long by {reduce_quantity} at {best_bid}")
            elif position < 0:  # We have a large short position
                # Calculate how much to reduce
                reduce_quantity = int(abs(position) * 0.3)  # Reduce by 30%
                if reduce_quantity > 0:
                    orders.append(Order(symbol, best_ask, reduce_quantity))
                    logger.print(f"SQUID_INK: RISK MANAGEMENT - Reducing short by {reduce_quantity} at {best_ask}")
        
        return orders

    def calc_position_size(self, signal_strength: float, position_limit: int, volatility: float) -> int:
        """Calculate position size based on signal strength and volatility"""
        # Base size (30% of position limit)
        base_size = int(position_limit * 0.3)
        
        # Adjust for signal strength (stronger signal = larger position)
        signal_adjustment = min(1.5, 1 + signal_strength/3)
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_factor = max(0.5, min(1.5, 1 / (volatility * 20)))
        
        # Calculate final size
        size = int(base_size * signal_adjustment * volatility_factor)
        
        # Cap at 80% of position limit
        capped_size = min(size, int(position_limit * 0.8))
        
        return max(1, capped_size)  # Ensure at least 1 unit
        
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """Main method called by the trading platform"""
        result: Dict[Symbol, List[Order]] = {}
        
        # Position limits
        position_limits = {
            "SQUID_INK": 50
        }
        
        timestamp = state.timestamp
        
        # If we have saved state, restore it
        if state.traderData:
            try:
                saved_data = jsonpickle.decode(state.traderData)
                
                # Restore price history
                for symbol, history in saved_data.get("price_history", {}).items():
                    if symbol in self.price_history:
                        self.price_history[symbol] = deque(history, maxlen=self.price_history[symbol].maxlen)
                    
                # Restore trade state
                if "trade_state" in saved_data:
                    self.trade_state = saved_data["trade_state"]
            except Exception as e:
                logger.print(f"Error restoring saved state: {e}")
        
        # SQUID_INK orders with mean reversion strategy
        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_orders = self.squid_ink_mean_reversion(
                state.order_depths["SQUID_INK"],
                "SQUID_INK",
                squid_position,
                position_limits["SQUID_INK"],
                timestamp
            )
            if squid_orders:
                result["SQUID_INK"] = squid_orders

        # Update trader data for persistence
        trader_data = {
            "price_history": {k: list(v) for k, v in self.price_history.items()},
            "trade_state": self.trade_state
        }

        traderData = jsonpickle.encode(trader_data)
        
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData