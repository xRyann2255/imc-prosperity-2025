import json
from typing import Any, Dict, List, Tuple
import math
import jsonpickle
import numpy as np
import string
from collections import deque
import pandas as pd
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
        self.has_plotted = False
        self.final_timestamp = 10000
        # Mean reversion parameters
        self.squid_ma_long = 15  # Long-term moving average period (mean to revert to)
        self.squid_z_threshold = 2.1 # Z-score threshold for entry

        # Price history for each product
        self.price_history = {
            "RAINFOREST_RESIN": deque(maxlen=50),
            "KELP": deque(maxlen=50),
            "SQUID_INK": deque(maxlen=300)
        }
        
        # Trade state tracking
        self.trade_state = {
            "SQUID_INK": {
                "positions": [],
                "last_trade_time": 0,
                "trade_cooldown": 1
            }
        }
        
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60
        }

        self.spread_threshold_b1 = 10    # Minimum spread (price units) to trigger an aggressive trade for Basket1
        self.spread_threshold_b2 = 8     # For Basket2
        self.calibration_factor = 5      # Determines trade size per unit excess spread
        self.safety_margin_b1 = 2        # Used for passive orders around fair value for Basket1
        self.safety_margin_b2 = 1.5      # For Basket2
        self.cushion_fraction = 0.95     # Use only up to 95% of available position capacity
        
        self.basket1_margin = 10  # Margin for basket1 arbitrage
        self.basket2_margin = 10  # Margin for basket2 arbitrage
        
    def calculate_mid_price(self, order_depth):
        """Calculate the mid price from the order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        return (best_bid + best_ask) / 2
    
    def calculate_moving_average(self, prices, period):
        """Calculate moving average of the given period"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    def calculate_exponential_moving_average(self, prices: List[float], period: int) -> float:
        """Calculate the exponential moving average (EMA) using the given decay (alpha)."""
        decay = 2 / (period + 1)
        if not prices:
            return None
        ema = prices[0]
        for price in prices[1:]:
            ema = decay * price + (1 - decay) * ema
        return ema
    
    def calculate_z_score(self, current_price, prices, period):
        """Calculate z-score (deviation from mean in standard deviations)"""
        if len(prices) < period:
            return 0
            
        # Calculate moving average
        ma = self.calculate_exponential_moving_average(prices, period)
        
        # Calculate standard deviation
        variance = sum((p - ma) ** 2 for p in prices[-period:]) / period
        std_dev = math.sqrt(variance) if variance > 0 else 1  # Avoid division by zero
        
        # Calculate z-score
        z_score = (current_price - ma) / std_dev
        
        return z_score, ma
    
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
            prev_ema = ema
            ema = alpha * price + (1 - alpha) * ema
            deviation = price - prev_ema
            ew_var = alpha * (deviation ** 2) + (1 - alpha) * ew_var

        std_dev = math.sqrt(ew_var) if ew_var > 0 else 1.0  # Avoid division by zero

        z_score = (current_price - ema) / std_dev
        return z_score, ema

    def rainforest_resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        width: int,
        position: int,
        position_limit: int,
    ) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Safety check for order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        # Select levels from the order depth
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(bbf) if bbf else fair_value - 2

        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask < fair_value:
            quantity = min(best_ask_amount, position_limit - position)
            if quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                buy_order_volume += quantity

        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        if best_bid > fair_value:
            quantity = min(best_bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", best_bid, -quantity))
                sell_order_volume += quantity

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            "RAINFOREST_RESIN",
            buy_order_volume,
            sell_order_volume,
            fair_value,
            1,
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))

        return orders

    def clear_position_order(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
        width: int,
    ) -> Tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, method="mid_price", min_vol=0) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            if (
                len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) == 0
                or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) == 0
            ):
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
                return mid_price

    def kelp_orders(
        self,
        order_depth: OrderDepth,
        timespan: int,
        width: float,
        kelp_take_width: float,
        position: int,
        position_limit: int,
    ) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Safety check for order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        # Dynamic volume filter based on order book averages
        avg_sell_volume = sum(abs(order_depth.sell_orders[price]) for price in order_depth.sell_orders) / len(order_depth.sell_orders)
        avg_buy_volume = sum(abs(order_depth.buy_orders[price]) for price in order_depth.buy_orders) / len(order_depth.buy_orders)
        vol = math.floor(min(avg_sell_volume, avg_buy_volume))
        logger.print("Volume filter set to: ", vol)
        fair_value = self.kelp_fair_value(order_depth, method="mid_price_with_vol_filter", min_vol=vol)
        
        # Update KELP price history
        if fair_value is not None:
            self.price_history["KELP"].append(fair_value)

        # Determine resting order prices
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(bbf) if bbf else fair_value - 2

        best_ask = min(order_depth.sell_orders.keys())
        best_ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask < fair_value:
            quantity = min(best_ask_amount, position_limit - position)
            if quantity > 0:
                orders.append(Order("KELP", best_ask, quantity))
                buy_order_volume += quantity

        best_bid = max(order_depth.buy_orders.keys())
        best_bid_amount = order_depth.buy_orders[best_bid]
        if best_bid > fair_value:
            quantity = min(best_bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order("KELP", best_bid, -quantity))
                sell_order_volume += quantity

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 1
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("KELP", int(bbbf + 1), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", int(baaf - 1), -sell_quantity))

        return orders
    
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
        orders: List[Order] = []
        
        # Safety check for order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
            
        # Get current market data
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        order_book_imbalance = self.calculate_order_book_imbalance(order_depth)
    
        # Update price history
        self.price_history["SQUID_INK"].append(mid_price)
        prices = list(self.price_history["SQUID_INK"])
        
        # Need at least enough history to calculate metrics
        if len(prices) < self.squid_ma_long:
            logger.print(f"Building price history for SQUID_INK: {len(prices)}/{self.squid_ma_long}")
            return []
        
        # Calculate moving averages
        ma_long = self.calculate_exponential_moving_average(prices, self.squid_ma_long)
        
        # Calculate z-score - how far current price deviates from long-term mean
        z_score, _ = self.calculate_ema_z_score(mid_price, prices, self.squid_ma_long)
        _, ema = self.calculate_ema_z_score(mid_price, prices, self.squid_ma_long)
 
        logger.print(f"SQUID_INK price history: {prices[-10:]}")
        logger.print(f"SQUID_INK order book imbalance: {order_book_imbalance}")
        # Log current metrics
        logger.print(f"SQUID_INK: price={mid_price}, MA(long)={ma_long:.2f}, z-score={z_score:.2f}, position={position}")
        
        # Get trade state
        trade_state = self.trade_state["SQUID_INK"]
        cooldown_elapsed = timestamp - trade_state["last_trade_time"] >= trade_state["trade_cooldown"]
        
        # Mean reversion trading logic
        if cooldown_elapsed:
            # Strong mean reversion signal - price significantly above long-term mean while trade imbalance is negative
            if z_score > self.squid_z_threshold and position > -position_limit * 0.8:
                # Calculate size based on signal strength - stronger signal = larger position
                signal_strength = min(abs(z_score) / 3.0, 1.0)  # Scale between 0-1
                size = max(1, int(position_limit * signal_strength * 0.8))  # Use up to 20% of position limit
                sell_quantity = min(size, position_limit + position)
                
                if sell_quantity > 0:
                    # Sell at current bid (aggressive) to ensure execution
                    orders.append(Order(symbol, best_bid, -sell_quantity))
                    logger.print(f"SQUID_INK: SELLING {sell_quantity} at {best_bid} (z-score: {z_score:.2f}, mean reversion)")
                    
                    # Record trade
                    trade_state["positions"].append({
                        "type": "SHORT",
                        "entry_price": best_bid,
                        "quantity": sell_quantity,
                        "entry_time": timestamp,
                        "z_score": z_score,
                        "target_price": ema  # Target the long-term mean
                    })
                    trade_state["last_trade_time"] = timestamp
            
            # Strong mean reversion signal - price significantly below long-term mean
            elif z_score < -self.squid_z_threshold and position < position_limit * 0.8:
                # Calculate size based on signal strength
                signal_strength = min(abs(z_score) / 3.0, 1.0)  # Scale between 0-1
                size = max(1, int(position_limit * signal_strength * 0.8))  # Use up to 20% of position limit
                buy_quantity = min(size, position_limit - position)
                
                if buy_quantity > 0:
                    # Buy at current ask (aggressive) to ensure execution
                    orders.append(Order(symbol, best_ask, buy_quantity))
                    logger.print(f"SQUID_INK: BUYING {buy_quantity} at {best_ask} (z-score: {z_score:.2f}, mean reversion)")
                    
                    # Record trade
                    trade_state["positions"].append({
                        "type": "LONG",
                        "entry_price": best_ask,
                        "quantity": buy_quantity,
                        "entry_time": timestamp,
                        "z_score": z_score,
                        "target_price": ema  # Target the long-term mean
                    })
                    trade_state["last_trade_time"] = timestamp
        
        # Manage existing positions - take profit or reduce risk
        if trade_state["positions"]:
            # Check each position
            new_positions = []
            for pos in trade_state["positions"]:
                if pos["type"] == "LONG":
                    # Price has reverted toward mean - take profit
                    if mid_price >= pos["target_price"] or z_score >= -0.5:
                        # Sell to close the long position
                        sell_quantity = pos["quantity"]
                        orders.append(Order(symbol, best_bid, -sell_quantity))
                        logger.print(f"SQUID_INK: PROFIT TAKING - Selling {sell_quantity} at {best_bid} (target reached)")
                    else:
                        # Keep the position
                        new_positions.append(pos)
                
                elif pos["type"] == "SHORT":
                    # Price has reverted toward mean - take profit
                    if mid_price <= pos["target_price"] or z_score <= 0.5:
                        # Buy to close the short position
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
    
    def plot_squid_ink_price_with_short_ma(self):
        prices = list(self.price_history["SQUID_INK"])
        if not prices:
            print("No price history to plot.")
            return

        ticks = list(range(len(prices)))
        short_period = self.squid_ma_long

        # Recompute the full EMA series using your logic
        decay = 2 / (short_period + 1)
        ema_series = [prices[0]]
        for price in prices[1:]:
            ema = decay * price + (1 - decay) * ema_series[-1]
            ema_series.append(ema)

    def calculate_mid_price(self, order_depth):
        """Calculate the mid price from an order book."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    def get_fair_values(self, state):
        """Compute fair values for Basket1 and Basket2 using mid-prices of their components."""
        mid_c = self.calculate_mid_price(state.order_depths["CROISSANTS"])
        mid_j = self.calculate_mid_price(state.order_depths["JAMS"])
        mid_d = self.calculate_mid_price(state.order_depths["DJEMBES"])
        # Basket1: 6 CROISSANTS + 3 JAMS + 1 DJEMBES
        fair_b1 = 6 * mid_c + 3 * mid_j + mid_d
        # Basket2: 4 CROISSANTS + 2 JAMS
        fair_b2 = 4 * mid_c + 2 * mid_j
        return fair_b1, fair_b2

    def aggressive_trade_basket1(self, state):
        """
        Execute aggressive (immediate) trades for Basket1 if its market price
        deviates strongly from the fair value.
        """
        orders = {"PICNIC_BASKET1": [], "CROISSANTS": [], "JAMS": [], "DJEMBES": []}
        fair_b1, _ = self.get_fair_values(state)
        # Obtain best prices from order depths
        depth_b1 = state.order_depths["PICNIC_BASKET1"]
        depth_c  = state.order_depths["CROISSANTS"]
        depth_j  = state.order_depths["JAMS"]
        depth_d  = state.order_depths["DJEMBES"]

        best_bid_b1 = max(depth_b1.buy_orders.keys())
        best_ask_b1 = min(depth_b1.sell_orders.keys())
        best_bid_c  = max(depth_c.buy_orders.keys())
        best_ask_c  = min(depth_c.sell_orders.keys())
        best_bid_j  = max(depth_j.buy_orders.keys())
        best_ask_j  = min(depth_j.sell_orders.keys())
        best_bid_d  = max(depth_d.buy_orders.keys())
        best_ask_d  = min(depth_d.sell_orders.keys())

        # Compute current mid price of Basket1 and its spread relative to fair value
        mid_b1 = (best_bid_b1 + best_ask_b1) / 2
        spread_b1 = mid_b1 - fair_b1

        # Get current positions
        pos_b1 = state.position.get("PICNIC_BASKET1", 0)
        pos_c  = state.position.get("CROISSANTS", 0)
        pos_j  = state.position.get("JAMS", 0)
        pos_d  = state.position.get("DJEMBES", 0)
        lim_b1 = self.position_limits["PICNIC_BASKET1"]
        lim_c  = self.position_limits["CROISSANTS"]
        lim_j  = self.position_limits["JAMS"]
        lim_d  = self.position_limits["DJEMBES"]

        # If Basket1 is significantly overpriced, trade aggressively.
        if spread_b1 > self.spread_threshold_b1:
            # Calculate desired trade quantity based on excess spread
            raw_qty = int((spread_b1 - self.spread_threshold_b1) / self.calibration_factor) + 1

            # Calculate available capacity:
            avail_b1 = int((pos_b1 + lim_b1) * self.cushion_fraction)  # number of baskets we can sell short
            avail_c  = int((lim_c - pos_c) // 6 * self.cushion_fraction)
            avail_j  = int((lim_j - pos_j) // 3 * self.cushion_fraction)
            avail_d  = int((lim_d - pos_d) * self.cushion_fraction)
            capacity = min(avail_b1, avail_c, avail_j, avail_d)

            trade_qty = min(raw_qty, capacity)
            if trade_qty > 0:
                # Sell Basket1 at best bid; buy components at best ask.
                orders["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", best_bid_b1, -trade_qty))
                orders["CROISSANTS"].append(Order("CROISSANTS", best_ask_c, trade_qty * 6))
                orders["JAMS"].append(Order("JAMS", best_ask_j, trade_qty * 3))
                orders["DJEMBES"].append(Order("DJEMBES", best_ask_d, trade_qty))
        # If Basket1 is significantly underpriced, do the reverse.
        elif spread_b1 < -self.spread_threshold_b1:
            raw_qty = int((-spread_b1 - self.spread_threshold_b1) / self.calibration_factor) + 1
            avail_b1 = int((lim_b1 - pos_b1) * self.cushion_fraction)
            avail_c  = int(((pos_c + lim_c) // 6) * self.cushion_fraction)
            avail_j  = int(((pos_j + lim_j) // 3) * self.cushion_fraction)
            avail_d  = int((pos_d + lim_d) * self.cushion_fraction)
            capacity = min(avail_b1, avail_c, avail_j, avail_d)
            trade_qty = min(raw_qty, capacity)
            if trade_qty > 0:
                orders["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", best_ask_b1, trade_qty))
                orders["CROISSANTS"].append(Order("CROISSANTS", best_bid_c, -trade_qty * 6))
                orders["JAMS"].append(Order("JAMS", best_bid_j, -trade_qty * 3))
                orders["DJEMBES"].append(Order("DJEMBES", best_bid_d, -trade_qty))
        return orders

    def passive_trade_basket1(self, state):
        """
        Post passive limit orders for Basket1 around its fair value.
        These orders will provide liquidity and potentially capture the spread.
        """
        orders = {"PICNIC_BASKET1": []}
        fair_b1, _ = self.get_fair_values(state)
        depth_b1 = state.order_depths["PICNIC_BASKET1"]
        best_bid = max(depth_b1.buy_orders.keys())
        best_ask = min(depth_b1.sell_orders.keys())
        # We choose prices between best_bid and best_ask anchored by fair value.
        # For a passive buy order, post slightly below fair value; for a passive sell, slightly above.
        buy_price = min(math.floor(fair_b1 - self.safety_margin_b1), best_bid)
        sell_price = max(math.ceil(fair_b1 + self.safety_margin_b1), best_ask)
        pos_b1 = state.position.get("PICNIC_BASKET1", 0)
        lim_b1 = self.position_limits["PICNIC_BASKET1"]
        # Calculate available capacity conservatively.
        avail_buy = int((lim_b1 - pos_b1) * self.cushion_fraction)
        avail_sell = int((pos_b1 + lim_b1) * self.cushion_fraction)
        if avail_buy > 0:
            orders["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", buy_price, avail_buy))
        if avail_sell > 0:
            orders["PICNIC_BASKET1"].append(Order("PICNIC_BASKET1", sell_price, -avail_sell))
        return orders

    def aggressive_trade_basket2(self, state):
        """
        Execute aggressive trades for Basket2 (comprised of 4 CROISSANTS and 2 JAMS) when mispriced.
        """
        orders = {"PICNIC_BASKET2": [], "CROISSANTS": [], "JAMS": []}
        _, fair_b2 = self.get_fair_values(state)
        depth_b2 = state.order_depths["PICNIC_BASKET2"]
        depth_c  = state.order_depths["CROISSANTS"]
        depth_j  = state.order_depths["JAMS"]

        best_bid_b2 = max(depth_b2.buy_orders.keys())
        best_ask_b2 = min(depth_b2.sell_orders.keys())
        best_bid_c  = max(depth_c.buy_orders.keys())
        best_ask_c  = min(depth_c.sell_orders.keys())
        best_bid_j  = max(depth_j.buy_orders.keys())
        best_ask_j  = min(depth_j.sell_orders.keys())

        mid_b2 = (best_bid_b2 + best_ask_b2) / 2
        spread_b2 = mid_b2 - fair_b2

        pos_b2 = state.position.get("PICNIC_BASKET2", 0)
        pos_c  = state.position.get("CROISSANTS", 0)
        pos_j  = state.position.get("JAMS", 0)
        lim_b2 = self.position_limits["PICNIC_BASKET2"]
        lim_c  = self.position_limits["CROISSANTS"]
        lim_j  = self.position_limits["JAMS"]

        if spread_b2 > self.spread_threshold_b2:
            raw_qty = int((spread_b2 - self.spread_threshold_b2) / self.calibration_factor) + 1
            avail_b2 = int((pos_b2 + lim_b2) * self.cushion_fraction)
            avail_c = int((lim_c - pos_c) // 4 * self.cushion_fraction)
            avail_j = int((lim_j - pos_j) // 2 * self.cushion_fraction)
            capacity = min(avail_b2, avail_c, avail_j)
            trade_qty = min(raw_qty, capacity)
            if trade_qty > 0:
                orders["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_bid_b2, -trade_qty))
                orders["CROISSANTS"].append(Order("CROISSANTS", best_ask_c, trade_qty * 4))
                orders["JAMS"].append(Order("JAMS", best_ask_j, trade_qty * 2))
        elif spread_b2 < -self.spread_threshold_b2:
            raw_qty = int((-spread_b2 - self.spread_threshold_b2) / self.calibration_factor) + 1
            avail_b2 = int((lim_b2 - pos_b2) * self.cushion_fraction)
            avail_c = int(((pos_c + lim_c) // 4) * self.cushion_fraction)
            avail_j = int(((pos_j + lim_j) // 2) * self.cushion_fraction)
            capacity = min(avail_b2, avail_c, avail_j)
            trade_qty = min(raw_qty, capacity)
            if trade_qty > 0:
                orders["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", best_ask_b2, trade_qty))
                orders["CROISSANTS"].append(Order("CROISSANTS", best_bid_c, -trade_qty * 4))
                orders["JAMS"].append(Order("JAMS", best_bid_j, -trade_qty * 2))
        return orders

    def passive_trade_basket2(self, state):
        """
        Post passive limit orders for Basket2 around its fair value.
        """
        orders = {"PICNIC_BASKET2": []}
        _, fair_b2 = self.get_fair_values(state)
        depth_b2 = state.order_depths["PICNIC_BASKET2"]
        best_bid = max(depth_b2.buy_orders.keys())
        best_ask = min(depth_b2.sell_orders.keys())
        buy_price = min(math.floor(fair_b2 - self.safety_margin_b2), best_bid)
        sell_price = max(math.ceil(fair_b2 + self.safety_margin_b2), best_ask)
        pos_b2 = state.position.get("PICNIC_BASKET2", 0)
        lim_b2 = self.position_limits["PICNIC_BASKET2"]
        avail_buy = int((lim_b2 - pos_b2) * self.cushion_fraction)
        avail_sell = int((pos_b2 + lim_b2) * self.cushion_fraction)
        if avail_buy > 0:
            orders["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", buy_price, avail_buy))
        if avail_sell > 0:
            orders["PICNIC_BASKET2"].append(Order("PICNIC_BASKET2", sell_price, -avail_sell))
        return orders

    def merge_orders(self, *orders_list):
        """Merge multiple order dictionaries into a single dictionary."""
        merged = {}
        for orders in orders_list:
            for sym, ord_list in orders.items():
                if sym not in merged:
                    merged[sym] = []
                merged[sym].extend(ord_list)
        return merged

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        
        # Position limits
        position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60
        }
        

        timestamp = state.timestamp
        
        
        # RAINFOREST_RESIN orders
        # if "RAINFOREST_RESIN" in state.order_depths:
        #     rainforest_resin_position = state.position.get("RAINFOREST_RESIN", 0)
            
        #     # Update price history for RAINFOREST_RESIN
        #     if state.order_depths["RAINFOREST_RESIN"].buy_orders and state.order_depths["RAINFOREST_RESIN"].sell_orders:
        #         mid_price = self.calculate_mid_price(state.order_depths["RAINFOREST_RESIN"])
        #         if mid_price:
        #             self.price_history["RAINFOREST_RESIN"].append(mid_price)
                
        #     rainforest_resin_orders = self.rainforest_resin_orders(
        #         state.order_depths["RAINFOREST_RESIN"],
        #         10000,  # Fixed fair value for RAINFOREST_RESIN
        #         2,      # Width
        #         rainforest_resin_position,
        #         position_limits["RAINFOREST_RESIN"]
        #     )
        #     result["RAINFOREST_RESIN"] = rainforest_resin_orders

        # # KELP orders
        # if "KELP" in state.order_depths:
        #     kelp_position = state.position.get("KELP", 0)
        #     kelp_orders = self.kelp_orders(
        #         state.order_depths["KELP"],
        #         10,     # Timespan
        #         4,      # Width
        #         1.35,   # Take width
        #         kelp_position,
        #         position_limits["KELP"]
        #     )
        #     result["KELP"] = kelp_orders

        # # SQUID_INK orders with mean reversion strategy
        # if "SQUID_INK" in state.order_depths:
        #     squid_position = state.position.get("SQUID_INK", 0)
        #     squid_orders = self.squid_ink_mean_reversion(
        #         state.order_depths["SQUID_INK"],
        #         "SQUID_INK",
        #         squid_position,
        #         position_limits["SQUID_INK"],
        #         timestamp
        #     )
        #     if squid_orders:
        #         result["SQUID_INK"] = squid_orders
        
        orders_b1_aggressive = self.aggressive_trade_basket1(state)
        orders_b1_passive = self.passive_trade_basket1(state)
        orders_b2_aggressive = self.aggressive_trade_basket2(state)
        orders_b2_passive = self.passive_trade_basket2(state)

        result = self.merge_orders(orders_b1_aggressive, orders_b1_passive,
                                           orders_b2_aggressive, orders_b2_passive)
        
        # Update our trader data for persistence
        trader_data = {
            "price_history": {k: list(v) for k, v in self.price_history.items()},
            "trade_state": self.trade_state
        }

        traderData = jsonpickle.encode(trader_data)

        ##
        #if timestamp > self.final_timestamp and not self.has_plotted:
           # print(f"Plotting at final tick {state.timestamp}")
          #  self.plot_squid_ink_price_with_short_ma()
           # self.has_plotted = True#

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData