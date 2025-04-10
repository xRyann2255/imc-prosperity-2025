import json
from typing import Any, Dict, List, Tuple
import math
import jsonpickle

import numpy as np
import string
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
        # Price history for each product
        self.price_history = {
            "RAINFOREST_RESIN": deque(maxlen=50),
            "KELP": deque(maxlen=50),
            "SQUID_INK": deque(maxlen=50)
        }
        
        # Trade state tracking
        self.trade_state = {
            "SQUID_INK": {
                "positions": [],
                "last_trade_time": 0,
                "trade_cooldown": 2
            }
        }
        
        
        # Mean reversion parameters
        self.squid_ma_short = 5   # Short-term moving average period
        self.squid_ma_long = 20   # Long-term moving average period (mean to revert to)
        self.squid_z_threshold = 1.5  # Z-score threshold for entry
        
        # Configure EMA decay factors for SQUID_INK
        self.squid_ema_short_decay = 2 / (self.squid_ma_short + 1)   # For short-term EMA
        self.squid_ema_long_decay = 2 / (self.squid_ma_long + 1)     # For long-term EMA
        
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
    
    def calculate_exponential_moving_average(self, prices: List[float], decay: float) -> float:
        """Calculate the exponential moving average (EMA) using the given decay (alpha)."""
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
            orders.append(Order("KELP", bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", baaf - 1, -sell_quantity))

        return orders

    def squid_ink_mean_reversion(
        self, order_depth: OrderDepth, symbol: str, position: int, position_limit: int, timestamp: int
    ) -> List[Order]:
        """Inverted mean reversion strategy for SQUID_INK"""
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        trade_imbalance = 0
        for price, volume in order_depth.buy_orders.items():
            if price < best_bid:
                trade_imbalance += volume
        for price, volume in order_depth.sell_orders.items():
            if price > best_ask:
                trade_imbalance -= volume

        self.price_history["SQUID_INK"].append(mid_price)
        prices = list(self.price_history["SQUID_INK"])

        if len(prices) < self.squid_ma_long:
            logger.print(f"Building price history for SQUID_INK: {len(prices)}/{self.squid_ma_long}")
            return []

        ma_short = self.calculate_exponential_moving_average(prices, self.squid_ma_short)
        ma_long = self.calculate_exponential_moving_average(prices, self.squid_ma_long)
        z_score, _ = self.calculate_ema_z_score(mid_price, prices, self.squid_ma_long)

        logger.print(f"SQUID_INK: price={mid_price}, MA(short)={ma_short:.2f}, MA(long)={ma_long:.2f}, z-score={z_score:.2f}, position={position}")

        trade_state = self.trade_state["SQUID_INK"]
        cooldown_elapsed = timestamp - trade_state["last_trade_time"] >= trade_state["trade_cooldown"]

        if cooldown_elapsed:
            if z_score > self.squid_z_threshold and position < position_limit * 0.8:
                signal_strength = min(abs(z_score) / 3.0, 1.0)
                size = max(1, int(position_limit * signal_strength * 0.8))
                buy_quantity = min(size, position_limit - position)

                if buy_quantity > 0:
                    orders.append(Order(symbol, best_ask, buy_quantity))  # Inverted: Buy on high z
                    logger.print(f"SQUID_INK (inverted): BUY {buy_quantity} at {best_ask} (z-score: {z_score:.2f})")

                    trade_state["positions"].append({
                        "type": "LONG",  # Inverted type
                        "entry_price": best_ask,
                        "quantity": buy_quantity,
                        "entry_time": timestamp,
                        "z_score": z_score,
                        "target_price": ma_long
                    })
                    trade_state["last_trade_time"] = timestamp

            elif z_score < -self.squid_z_threshold and position > -position_limit * 0.8:
                signal_strength = min(abs(z_score) / 3.0, 1.0)
                size = max(1, int(position_limit * signal_strength * 0.8))
                sell_quantity = min(size, position_limit + position)

                if sell_quantity > 0:
                    orders.append(Order(symbol, best_bid, -sell_quantity))  # Inverted: Sell on low z
                    logger.print(f"SQUID_INK (inverted): SELL {sell_quantity} at {best_bid} (z-score: {z_score:.2f})")

                    trade_state["positions"].append({
                        "type": "SHORT",  # Inverted type
                        "entry_price": best_bid,
                        "quantity": sell_quantity,
                        "entry_time": timestamp,
                        "z_score": z_score,
                        "target_price": ma_long
                    })
                    trade_state["last_trade_time"] = timestamp

        if trade_state["positions"]:
            new_positions = []
            for pos in trade_state["positions"]:
                if pos["type"] == "LONG":
                    if mid_price <= pos["target_price"] or z_score <= 0.5:  # Inverted: take profit on reversion downward
                        sell_quantity = pos["quantity"]
                        orders.append(Order(symbol, best_bid, -sell_quantity))
                        logger.print(f"SQUID_INK (inverted): PROFIT - SELL {sell_quantity} at {best_bid}")
                    else:
                        new_positions.append(pos)

                elif pos["type"] == "SHORT":
                    if mid_price >= pos["target_price"] or z_score >= -0.5:  # Inverted: take profit on reversion upward
                        buy_quantity = pos["quantity"]
                        orders.append(Order(symbol, best_ask, buy_quantity))
                        logger.print(f"SQUID_INK (inverted): PROFIT - BUY {buy_quantity} at {best_ask}")
                    else:
                        new_positions.append(pos)

            trade_state["positions"] = new_positions

        position_pct = abs(position) / position_limit
        if position_pct > 0.8:
            if position > 0:
                reduce_quantity = int(position * 0.3)
                if reduce_quantity > 0:
                    orders.append(Order(symbol, best_bid, -reduce_quantity))
                    logger.print(f"SQUID_INK (inverted): REDUCE LONG {reduce_quantity} at {best_bid}")
            elif position < 0:
                reduce_quantity = int(abs(position) * 0.3)
                if reduce_quantity > 0:
                    orders.append(Order(symbol, best_ask, reduce_quantity))
                    logger.print(f"SQUID_INK (inverted): REDUCE SHORT {reduce_quantity} at {best_ask}")

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        
        # Position limits
        position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        
        timestamp = state.timestamp
        
        # RAINFOREST_RESIN orders
        if "RAINFOREST_RESIN" in state.order_depths:
            rainforest_resin_position = state.position.get("RAINFOREST_RESIN", 0)
            
            # Update price history for RAINFOREST_RESIN
            if state.order_depths["RAINFOREST_RESIN"].buy_orders and state.order_depths["RAINFOREST_RESIN"].sell_orders:
                mid_price = self.calculate_mid_price(state.order_depths["RAINFOREST_RESIN"])
                if mid_price:
                    self.price_history["RAINFOREST_RESIN"].append(mid_price)
                
            rainforest_resin_orders = self.rainforest_resin_orders(
                state.order_depths["RAINFOREST_RESIN"],
                10000,  # Fixed fair value for RAINFOREST_RESIN
                2,      # Width
                rainforest_resin_position,
                position_limits["RAINFOREST_RESIN"]
            )
            result["RAINFOREST_RESIN"] = rainforest_resin_orders

        # KELP orders
        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"],
                10,     # Timespan
                4,      # Width
                1.35,   # Take width
                kelp_position,
                position_limits["KELP"]
            )
            result["KELP"] = kelp_orders

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

        # Update our trader data for persistence
        trader_data = {
            "price_history": {k: list(v) for k, v in self.price_history.items()},
            "trade_state": self.trade_state
        }

        traderData = jsonpickle.encode(trader_data)

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData