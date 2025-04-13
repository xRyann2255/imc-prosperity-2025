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

    def basket1_arbitrage_order(self, state: TradingState) -> List[Order]:
        """Hybrid: post limit orders around fair value + multi-level immediate arbitrage if extremely profitable."""
        basket_orders: List[Order] = []
        croissant_orders: List[Order] = []
        jam_orders: List[Order] = []
        djembe_orders: List[Order] = []

        required_symbols = ["PICNIC_BASKET1", "CROISSANTS", "JAMS", "DJEMBES"]
        for sym in required_symbols:
            if sym not in state.order_depths:
                return basket_orders, croissant_orders, jam_orders, djembe_orders
            depth = state.order_depths[sym]
            if not depth.buy_orders or not depth.sell_orders:
                return basket_orders, croissant_orders, jam_orders, djembe_orders

        depth_basket = state.order_depths["PICNIC_BASKET1"]
        depth_croissant = state.order_depths["CROISSANTS"]
        depth_jam = state.order_depths["JAMS"]
        depth_djembe = state.order_depths["DJEMBES"]

        # Best bid/ask for each
        b_bid = max(depth_basket.buy_orders.keys())
        b_ask = min(depth_basket.sell_orders.keys())
        c_bid = max(depth_croissant.buy_orders.keys())
        c_ask = min(depth_croissant.sell_orders.keys())
        j_bid = max(depth_jam.buy_orders.keys())
        j_ask = min(depth_jam.sell_orders.keys())
        d_bid = max(depth_djembe.buy_orders.keys())
        d_ask = min(depth_djembe.sell_orders.keys())

        # Mid-prices for components
        c_mid = (c_bid + c_ask) / 2
        j_mid = (j_bid + j_ask) / 2
        d_mid = (d_bid + d_ask) / 2

        # Basket1 fair value
        fair_value_b1 = 6 * c_mid + 3 * j_mid + d_mid

        # We define two different margins:
        # 1) 'aggressive_margin': if the top-of-book price is beyond this margin, we cross the spread for immediate fill
        # 2) 'safety_margin': used for posting limit orders around fair_value
        #    i.e. we place limit buy at (fair_value - safety_margin) and limit sell at (fair_value + safety_margin)
        # These can be distinct from self.basket1_margin. Tune as needed.
        aggressive_margin = 5.0      # how many ticks or absolute price difference triggers immediate cross
        safety_margin = 2.0          # how far from fair_value we place limit orders
        # Also consider your position-limits conservatism:
        cushion_fraction = 0.95      # only use 95% of the limit to keep a bit of buffer

        pos_basket = state.position.get("PICNIC_BASKET1", 0)
        pos_croissant = state.position.get("CROISSANTS", 0)
        pos_jam = state.position.get("JAMS", 0)
        pos_djembe = state.position.get("DJEMBES", 0)

        limit_basket = self.position_limits["PICNIC_BASKET1"]
        limit_croissant = self.position_limits["CROISSANTS"]
        limit_jam = self.position_limits["JAMS"]
        limit_djembe = self.position_limits["DJEMBES"]

        ####################################
        # PART A: Attempt immediate arbitrage if extremely profitable at top-of-book
        ####################################
        # Overpriced scenario: current best bid >> fair_value + aggressive_margin => SELL basket, BUY components
        if (b_bid - fair_value_b1) > aggressive_margin:
            # multi-level iteration
            basket_bids = sorted(depth_basket.buy_orders.keys(), reverse=True)
            c_asks = sorted(depth_croissant.sell_orders.keys())
            j_asks = sorted(depth_jam.sell_orders.keys())
            d_asks = sorted(depth_djembe.sell_orders.keys())

            i_b = i_c = i_j = i_d = 0
            while i_b < len(basket_bids) and i_c < len(c_asks) and i_j < len(j_asks) and i_d < len(d_asks):
                pb = basket_bids[i_b]
                pc = c_asks[i_c]
                pj = j_asks[i_j]
                pd = d_asks[i_d]

                local_margin = pb - (6 * pc + 3 * pj + pd)
                if local_margin <= aggressive_margin:
                    break

                # Volumes
                vb = depth_basket.buy_orders[pb]
                vc = -depth_croissant.sell_orders[pc]
                vj = -depth_jam.sell_orders[pj]
                vd = -depth_djembe.sell_orders[pd]

                max_by_book = min(vb, vc // 6, vj // 3, vd)
                # Position-limited
                # SELL basket => pos_basket - n >= -limit => n <= pos_basket + limit
                n_basket = int((pos_basket + limit_basket) * cushion_fraction)
                # BUY croissants => pos_croissant + 6n <= limit_croissant
                n_croissant = int(((limit_croissant - pos_croissant) // 6) * cushion_fraction)
                n_jam = int(((limit_jam - pos_jam) // 3) * cushion_fraction)
                n_djembe = int((limit_djembe - pos_djembe) * cushion_fraction)
                max_by_pos = min(n_basket, n_croissant, n_jam, n_djembe)

                qty = min(max_by_book, max_by_pos)
                if qty <= 0:
                    # Advance whichever is exhausted
                    advanced = False
                    if vb <= 0:
                        i_b += 1
                        advanced = True
                    if vc < 6:
                        i_c += 1
                        advanced = True
                    if vj < 3:
                        i_j += 1
                        advanced = True
                    if vd < 1:
                        i_d += 1
                        advanced = True
                    if not advanced:
                        # position-limited
                        break
                    continue

                # Execute trade
                basket_orders.append(Order("PICNIC_BASKET1", pb, -qty))
                croissant_orders.append(Order("CROISSANTS", pc, qty * 6))
                jam_orders.append(Order("JAMS", pj, qty * 3))
                djembe_orders.append(Order("DJEMBES", pd, qty))

                depth_basket.buy_orders[pb] -= qty
                depth_croissant.sell_orders[pc] += qty * 6
                depth_jam.sell_orders[pj] += qty * 3
                depth_djembe.sell_orders[pd] += qty

                pos_basket -= qty
                pos_croissant += qty * 6
                pos_jam += qty * 3
                pos_djembe += qty

                if depth_basket.buy_orders[pb] <= 0:
                    i_b += 1
                if depth_croissant.sell_orders[pc] >= 0:
                    i_c += 1
                if depth_jam.sell_orders[pj] >= 0:
                    i_j += 1
                if depth_djembe.sell_orders[pd] >= 0:
                    i_d += 1

        # Underpriced scenario: best ask << fair_value - aggressive_margin => BUY basket, SELL components
        elif (fair_value_b1 - b_ask) > aggressive_margin:
            basket_asks = sorted(depth_basket.sell_orders.keys())
            c_bids = sorted(depth_croissant.buy_orders.keys(), reverse=True)
            j_bids = sorted(depth_jam.buy_orders.keys(), reverse=True)
            d_bids = sorted(depth_djembe.buy_orders.keys(), reverse=True)

            i_b = i_c = i_j = i_d = 0
            while i_b < len(basket_asks) and i_c < len(c_bids) and i_j < len(j_bids) and i_d < len(d_bids):
                pb = basket_asks[i_b]
                pc = c_bids[i_c]
                pj = j_bids[i_j]
                pd = d_bids[i_d]

                local_margin = (6 * pc + 3 * pj + pd) - pb
                if local_margin <= aggressive_margin:
                    break

                vb = -depth_basket.sell_orders[pb]
                vc = depth_croissant.buy_orders[pc]
                vj = depth_jam.buy_orders[pj]
                vd = depth_djembe.buy_orders[pd]

                max_by_book = min(vb, vc // 6, vj // 3, vd)

                n_basket = int((limit_basket - pos_basket) * cushion_fraction)
                n_croissant = int(((pos_croissant + limit_croissant) // 6) * cushion_fraction)
                n_jam = int(((pos_jam + limit_jam) // 3) * cushion_fraction)
                n_djembe = int((pos_djembe + limit_djembe) * cushion_fraction)
                max_by_pos = min(n_basket, n_croissant, n_jam, n_djembe)

                qty = min(max_by_book, max_by_pos)
                if qty <= 0:
                    advanced = False
                    if vb <= 0:
                        i_b += 1; advanced = True
                    if vc < 6:
                        i_c += 1; advanced = True
                    if vj < 3:
                        i_j += 1; advanced = True
                    if vd < 1:
                        i_d += 1; advanced = True
                    if not advanced:
                        break
                    continue

                basket_orders.append(Order("PICNIC_BASKET1", pb, qty))
                croissant_orders.append(Order("CROISSANTS", pc, -qty * 6))
                jam_orders.append(Order("JAMS", pj, -qty * 3))
                djembe_orders.append(Order("DJEMBES", pd, -qty))

                depth_basket.sell_orders[pb] += qty
                depth_croissant.buy_orders[pc] -= qty * 6
                depth_jam.buy_orders[pj] -= qty * 3
                depth_djembe.buy_orders[pd] -= qty

                pos_basket += qty
                pos_croissant -= qty * 6
                pos_jam -= qty * 3
                pos_djembe -= qty

                if depth_basket.sell_orders[pb] >= 0:
                    i_b += 1
                if depth_croissant.buy_orders[pc] <= 0:
                    i_c += 1
                if depth_jam.buy_orders[pj] <= 0:
                    i_j += 1
                if depth_djembe.buy_orders[pd] <= 0:
                    i_d += 1

        ####################################
        # PART B: Place passive orders around fair_value (post-only)
        ####################################
        # Because we might not get immediate fills if margin is small, we place limit orders that “make the market”
        # so we collect spread if the market trades through us.
        # We'll place a buy around (fair_value_b1 - safety_margin) and a sell around (fair_value_b1 + safety_margin).
        # Only place these if we still have room under position limits.

        # 1) Limit BUY below fair_value
        buy_price = math.floor(fair_value_b1 - safety_margin)
        # Check if our buy_price is inside the best ask => we risk crossing the spread unintentionally
        best_ask_basket = min(depth_basket.sell_orders.keys())
        if buy_price >= best_ask_basket:
            # ensure we stay behind best ask
            buy_price = best_ask_basket - 1

        # how many baskets can we buy? keep cushion
        can_buy_baskets = int((limit_basket - pos_basket) * cushion_fraction)
        if can_buy_baskets > 0:
            basket_orders.append(Order("PICNIC_BASKET1", buy_price, can_buy_baskets))

        # 2) Limit SELL above fair_value
        sell_price = math.ceil(fair_value_b1 + safety_margin)
        best_bid_basket = max(depth_basket.buy_orders.keys())
        if sell_price <= best_bid_basket:
            sell_price = best_bid_basket + 1

        can_sell_baskets = int((pos_basket + limit_basket) * cushion_fraction)
        if can_sell_baskets > 0:
            basket_orders.append(Order("PICNIC_BASKET1", sell_price, -can_sell_baskets))

        return basket_orders, croissant_orders, jam_orders, djembe_orders


    def basket2_arbitrage_order(self, state: TradingState) -> List[Order]:
        """Same 'hybrid' approach, but for PICNIC_BASKET2 (4 croissants + 2 jams)."""
        basket_orders: List[Order] = []
        croissant_orders: List[Order] = []
        jam_orders: List[Order] = []

        required_symbols = ["PICNIC_BASKET2", "CROISSANTS", "JAMS"]
        for sym in required_symbols:
            if sym not in state.order_depths:
                return basket_orders, croissant_orders, jam_orders
            depth = state.order_depths[sym]
            if not depth.buy_orders or not depth.sell_orders:
                return basket_orders, croissant_orders, jam_orders

        depth_basket = state.order_depths["PICNIC_BASKET2"]
        depth_croissant = state.order_depths["CROISSANTS"]
        depth_jam = state.order_depths["JAMS"]

        # Best bids/asks
        b_bid = max(depth_basket.buy_orders.keys())
        b_ask = min(depth_basket.sell_orders.keys())
        c_bid = max(depth_croissant.buy_orders.keys())
        c_ask = min(depth_croissant.sell_orders.keys())
        j_bid = max(depth_jam.buy_orders.keys())
        j_ask = min(depth_jam.sell_orders.keys())

        # Mid-prices
        c_mid = (c_bid + c_ask) / 2
        j_mid = (j_bid + j_ask) / 2
        fair_value_b2 = 4 * c_mid + 2 * j_mid

        # Similar margin approach
        aggressive_margin = 3.0
        safety_margin = 1.5
        cushion_fraction = 0.95

        pos_basket = state.position.get("PICNIC_BASKET2", 0)
        pos_croissant = state.position.get("CROISSANTS", 0)
        pos_jam = state.position.get("JAMS", 0)

        limit_basket = self.position_limits["PICNIC_BASKET2"]
        limit_croissant = self.position_limits["CROISSANTS"]
        limit_jam = self.position_limits["JAMS"]

        # PART A: immediate cross if highly profitable
        if (b_bid - fair_value_b2) > aggressive_margin:
            # Overpriced => SELL basket, BUY components
            b_bids = sorted(depth_basket.buy_orders.keys(), reverse=True)
            c_asks = sorted(depth_croissant.sell_orders.keys())
            j_asks = sorted(depth_jam.sell_orders.keys())
            i_b = i_c = i_j = 0
            while i_b < len(b_bids) and i_c < len(c_asks) and i_j < len(j_asks):
                pb = b_bids[i_b]
                pc = c_asks[i_c]
                pj = j_asks[i_j]
                local_margin = pb - (4 * pc + 2 * pj)
                if local_margin <= aggressive_margin:
                    break

                vb = depth_basket.buy_orders[pb]
                vc = -depth_croissant.sell_orders[pc]
                vj = -depth_jam.sell_orders[pj]
                max_by_book = min(vb, vc // 4, vj // 2)

                n_basket = int((pos_basket + limit_basket) * cushion_fraction)
                n_croissant = int(((limit_croissant - pos_croissant) // 4) * cushion_fraction)
                n_jam = int(((limit_jam - pos_jam) // 2) * cushion_fraction)
                max_by_pos = min(n_basket, n_croissant, n_jam)
                qty = min(max_by_book, max_by_pos)
                if qty <= 0:
                    advanced = False
                    if vb <= 0:
                        i_b += 1; advanced = True
                    if vc < 4:
                        i_c += 1; advanced = True
                    if vj < 2:
                        i_j += 1; advanced = True
                    if not advanced:
                        break
                    continue

                basket_orders.append(Order("PICNIC_BASKET2", pb, -qty))
                croissant_orders.append(Order("CROISSANTS", pc, qty * 4))
                jam_orders.append(Order("JAMS", pj, qty * 2))

                depth_basket.buy_orders[pb] -= qty
                depth_croissant.sell_orders[pc] += qty * 4
                depth_jam.sell_orders[pj] += qty * 2

                pos_basket -= qty
                pos_croissant += qty * 4
                pos_jam += qty * 2

                if depth_basket.buy_orders[pb] <= 0:
                    i_b += 1
                if depth_croissant.sell_orders[pc] >= 0:
                    i_c += 1
                if depth_jam.sell_orders[pj] >= 0:
                    i_j += 1

        elif (fair_value_b2 - b_ask) > aggressive_margin:
            # Underpriced => BUY basket, SELL components
            b_asks = sorted(depth_basket.sell_orders.keys())
            c_bids = sorted(depth_croissant.buy_orders.keys(), reverse=True)
            j_bids = sorted(depth_jam.buy_orders.keys(), reverse=True)
            i_b = i_c = i_j = 0
            while i_b < len(b_asks) and i_c < len(c_bids) and i_j < len(j_bids):
                pb = b_asks[i_b]
                pc = c_bids[i_c]
                pj = j_bids[i_j]
                local_margin = (4 * pc + 2 * pj) - pb
                if local_margin <= aggressive_margin:
                    break

                vb = -depth_basket.sell_orders[pb]
                vc = depth_croissant.buy_orders[pc]
                vj = depth_jam.buy_orders[pj]
                max_by_book = min(vb, vc // 4, vj // 2)

                n_basket = int((limit_basket - pos_basket) * cushion_fraction)
                n_croissant = int(((pos_croissant + limit_croissant) // 4) * cushion_fraction)
                n_jam = int(((pos_jam + limit_jam) // 2) * cushion_fraction)
                qty = min(max_by_book, n_basket, n_croissant, n_jam)

                if qty <= 0:
                    advanced = False
                    if vb <= 0:
                        i_b += 1; advanced = True
                    if vc < 4:
                        i_c += 1; advanced = True
                    if vj < 2:
                        i_j += 1; advanced = True
                    if not advanced:
                        break
                    continue

                basket_orders.append(Order("PICNIC_BASKET2", pb, qty))
                croissant_orders.append(Order("CROISSANTS", pc, -qty * 4))
                jam_orders.append(Order("JAMS", pj, -qty * 2))

                depth_basket.sell_orders[pb] += qty
                depth_croissant.buy_orders[pc] -= qty * 4
                depth_jam.buy_orders[pj] -= qty * 2

                pos_basket += qty
                pos_croissant -= qty * 4
                pos_jam -= qty * 2

                if depth_basket.sell_orders[pb] >= 0:
                    i_b += 1
                if depth_croissant.buy_orders[pc] <= 0:
                    i_c += 1
                if depth_jam.buy_orders[pj] <= 0:
                    i_j += 1

        # PART B: place passive orders around fair_value
        buy_price = math.floor(fair_value_b2 - safety_margin)
        if buy_price >= b_ask:
            buy_price = b_ask - 1
        can_buy_baskets = int((limit_basket - pos_basket) * cushion_fraction)
        if can_buy_baskets > 0:
            basket_orders.append(Order("PICNIC_BASKET2", buy_price, can_buy_baskets))

        sell_price = math.ceil(fair_value_b2 + safety_margin)
        best_basket_bid = max(depth_basket.buy_orders.keys())
        if sell_price <= best_basket_bid:
            sell_price = best_basket_bid + 1
        can_sell_baskets = int((pos_basket + limit_basket) * cushion_fraction)
        if can_sell_baskets > 0:
            basket_orders.append(Order("PICNIC_BASKET2", sell_price, -can_sell_baskets))

        return basket_orders, croissant_orders, jam_orders
        

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
        
        basket1_orders, croissant_orders, jam_orders, djembe_orders = self.basket1_arbitrage_order(state)
        if basket1_orders and croissant_orders and jam_orders and djembe_orders:
            logger.print("Executing basket1 arbitrage orders")
            result["PICNIC_BASKET1"] = basket1_orders
            result["CROISSANTS"] = croissant_orders
            result["JAMS"] = jam_orders
            result["DJEMBES"] = djembe_orders

        basket2_orders, croissant_orders, jam_orders = self.basket2_arbitrage_order(state)
        if basket2_orders and croissant_orders and jam_orders:
            logger.print("Executing basket2 arbitrage orders")
            result["PICNIC_BASKET2"] = basket2_orders
            result["CROISSANTS"] = croissant_orders
            result["JAMS"] = jam_orders
        
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