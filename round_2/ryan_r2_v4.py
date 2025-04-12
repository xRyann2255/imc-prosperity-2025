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
        
        # Position limits
        self.LIMIT = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        
        # Price history for each product
        self.price_history = {
            "RAINFOREST_RESIN": deque(maxlen=50),
            "KELP": deque(maxlen=50),
            "SQUID_INK": deque(maxlen=300)
        }
        
        # SQUID_INK parameters with improved settings
        self.SQUID_PARAMS = {
            "ema_period": 20,          # EMA period for fair value
            "z_score_entry": 2.5,      # Z-score threshold to enter position
            "z_score_exit": 0.5,       # Z-score threshold to exit position
            "stop_loss_multiplier": 1.5, # Exit if z-score moves 1.5x worse from entry
            "max_position_hold": 30,   # Maximum ticks to hold a position
            "trend_period": 40,        # Period for trend detection
            "trend_threshold": 0.7,    # Correlation threshold for trend detection
            "min_history": 30,         # Minimum price history required
            "base_order_size": 15,     # Base position size
            "max_order_size": 40,      # Maximum position size
            "dynamic_sizing": True,    # Whether to use dynamic position sizing
            "limit_order_edge": 0.1,   # Edge for limit orders vs market orders
            "lookback": 50             # Period for standard deviation calculation
        }
        
        # Initialize trader object for state persistence
        self.traderObject = {}
        
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
        if not prices or len(prices) < 2:
            return prices[0] if prices else None
            
        decay = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = decay * price + (1 - decay) * ema
        return ema
    
    def calculate_trend_strength(self, prices: List[float], period: int) -> float:
        """
        Calculate trend strength using correlation coefficient between price and time
        Returns a value between -1 and 1, where:
        - Values close to 1 indicate a strong uptrend
        - Values close to -1 indicate a strong downtrend
        - Values close to 0 indicate no clear trend
        """
        if len(prices) < period:
            return 0.0
            
        recent_prices = prices[-period:]
        time_indices = list(range(len(recent_prices)))
        
        # Calculate means
        mean_price = sum(recent_prices) / len(recent_prices)
        mean_time = sum(time_indices) / len(time_indices)
        
        # Calculate covariance and variances
        covariance = sum((p - mean_price) * (t - mean_time) for p, t in zip(recent_prices, time_indices))
        variance_price = sum((p - mean_price) ** 2 for p in recent_prices)
        variance_time = sum((t - mean_time) ** 2 for t in time_indices)
        
        # Calculate correlation coefficient
        if variance_price > 0 and variance_time > 0:
            correlation = covariance / (math.sqrt(variance_price) * math.sqrt(variance_time))
            return correlation
        return 0.0
    
    def calculate_z_score(self, current_price, prices, period):
        """Calculate z-score (deviation from mean in standard deviations)"""
        if len(prices) < period:
            return 0, None
            
        # Use sample mean of recent prices (not EMA)
        recent_prices = prices[-period:]
        mean_price = sum(recent_prices) / len(recent_prices)
        
        # Calculate standard deviation based on deviations from mean
        variance = sum((p - mean_price) ** 2 for p in recent_prices) / period
        std_dev = math.sqrt(variance) if variance > 0 else 1  # Avoid division by zero
        
        # Calculate z-score
        z_score = (current_price - mean_price) / std_dev
        
        return z_score, mean_price
    
    def calculate_dynamic_order_size(self, z_score: float, base_size: int, max_size: int) -> int:
        """
        Calculate order size based on z-score magnitude:
        - Higher absolute z-score = larger position (more conviction)
        - Capped at max_size
        """
        # Scale size linearly with z-score magnitude
        abs_z = abs(z_score)
        z_factor = min(abs_z / self.SQUID_PARAMS["z_score_entry"], 2.0)  # Cap at 2x base size
        
        # Calculate size and ensure it's at least the base size
        size = max(base_size, int(base_size * z_factor))
        
        # Cap at max size
        return min(size, max_size)
    
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
    
    # --- IMPROVED MEAN REVERSION ALGORITHM FOR SQUID_INK ---
    
    def squid_ink_mean_reversion(
        self,
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        trader_data: Dict,
        timestamp: int
    ) -> List[Order]:
        """
        Improved mean reversion strategy for SQUID_INK:
        1. Calculate current mid price
        2. Compute EMA for trend detection
        3. Calculate z-score using proper sample mean and standard deviation
        4. Check for trend strength to avoid trading against strong trends
        5. Enter positions when z-score exceeds threshold (in opposite direction)
        6. Use dynamic position sizing based on z-score magnitude
        7. Multiple exit conditions: target profit, stop loss, time-based exit
        8. Position validation to ensure consistency
        9. Use limit orders when appropriate for better execution
        """
        orders: List[Order] = []
        
        # Safety check for order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        # Initialize SQUID_INK data in trader_data if not present
        if "squid_data" not in trader_data:
            trader_data["squid_data"] = {
                "in_position": False,
                "position_type": None,  # "long" or "short"
                "entry_price": None,
                "entry_z_score": None,
                "fair_value": None,
                "entry_timestamp": None,  # For time-based exit
                "tick_count": 0         # For debugging
            }
        
        squid_data = trader_data["squid_data"]
        squid_data["tick_count"] = squid_data.get("tick_count", 0) + 1
            
        # Calculate current mid price and book data
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        current_price = (best_bid + best_ask) / 2
        bid_volume = sum(order_depth.buy_orders.values())
        ask_volume = sum(abs(v) for v in order_depth.sell_orders.values())
        
        # Update price history
        self.price_history["SQUID_INK"].append(current_price)
        prices = list(self.price_history["SQUID_INK"])
        
        # If not enough history, just return
        if len(prices) < self.SQUID_PARAMS["min_history"]:
            logger.print(f"SQUID_INK: Not enough price history ({len(prices)}/{self.SQUID_PARAMS['min_history']})")
            return orders
        
        # Calculate EMA for trend detection
        ema = self.calculate_exponential_moving_average(prices, self.SQUID_PARAMS["ema_period"])
        
        # Calculate trend strength
        trend_strength = self.calculate_trend_strength(prices, self.SQUID_PARAMS["trend_period"])
        
        # Calculate standard deviation and z-score using proper statistics
        lookback = min(self.SQUID_PARAMS["lookback"], len(prices))
        z_score, mean_price = self.calculate_z_score(current_price, prices, lookback)
        
        logger.print(f"SQUID_INK: Price={current_price:.2f}, Mean={mean_price:.2f}, EMA={ema:.2f}, Z-score={z_score:.2f}, Trend={trend_strength:.2f}")
        logger.print(f"SQUID_INK: Position={position}, In position={squid_data['in_position']}, Tick={squid_data['tick_count']}")
        
        # --- POSITION VALIDATION ---
        # Check if our position tracking matches the actual position
        if squid_data["in_position"] and squid_data["position_type"] == "long" and position <= 0:
            logger.print(f"SQUID_INK: Position tracking mismatch! Tracked long but actual {position}")
            squid_data["in_position"] = False
            squid_data["position_type"] = None
        elif squid_data["in_position"] and squid_data["position_type"] == "short" and position >= 0:
            logger.print(f"SQUID_INK: Position tracking mismatch! Tracked short but actual {position}")
            squid_data["in_position"] = False
            squid_data["position_type"] = None
        
        # Strategy logic
        if not squid_data["in_position"]:
            # --- ENTRY LOGIC ---
            
            # Only trade if trend strength isn't too high (avoid fighting strong trends)
            strong_trend = abs(trend_strength) > self.SQUID_PARAMS["trend_threshold"]
            wrong_direction = (z_score > 0 and trend_strength > 0) or (z_score < 0 and trend_strength < 0)
            
            if abs(z_score) > self.SQUID_PARAMS["z_score_entry"] and not (strong_trend and wrong_direction):
                # If price is very high (positive z-score), go short
                if z_score > 0:
                    # Calculate dynamic order size if enabled
                    if self.SQUID_PARAMS["dynamic_sizing"]:
                        order_size = self.calculate_dynamic_order_size(z_score, 
                                                                     self.SQUID_PARAMS["base_order_size"], 
                                                                     self.SQUID_PARAMS["max_order_size"])
                    else:
                        order_size = self.SQUID_PARAMS["base_order_size"]
                        
                    order_size = min(order_size, position_limit + position)
                    
                    if order_size > 0:
                        # Try to improve on best bid with limit order if possible
                        improved_price = best_bid + self.SQUID_PARAMS["limit_order_edge"]
                        
                        # But make sure we're still below our fair value
                        if improved_price < mean_price:
                            sell_price = improved_price
                        else:
                            sell_price = best_bid  # Default to market order
                            
                        orders.append(Order("SQUID_INK", sell_price, -order_size))
                        logger.print(f"SQUID_INK: ENTRY SHORT {order_size} @ {sell_price}, z-score={z_score:.2f}")
                        
                        # Update position tracking
                        squid_data["in_position"] = True
                        squid_data["position_type"] = "short"
                        squid_data["entry_price"] = sell_price
                        squid_data["entry_z_score"] = z_score
                        squid_data["fair_value"] = mean_price
                        squid_data["entry_timestamp"] = timestamp
                
                # If price is very low (negative z-score), go long
                elif z_score < 0:
                    # Calculate dynamic order size if enabled
                    if self.SQUID_PARAMS["dynamic_sizing"]:
                        order_size = self.calculate_dynamic_order_size(z_score, 
                                                                     self.SQUID_PARAMS["base_order_size"], 
                                                                     self.SQUID_PARAMS["max_order_size"])
                    else:
                        order_size = self.SQUID_PARAMS["base_order_size"]
                        
                    order_size = min(order_size, position_limit - position)
                    
                    if order_size > 0:
                        # Try to improve on best ask with limit order if possible
                        improved_price = best_ask - self.SQUID_PARAMS["limit_order_edge"]
                        
                        # But make sure we're still above our fair value
                        if improved_price > mean_price:
                            buy_price = improved_price
                        else:
                            buy_price = best_ask  # Default to market order
                            
                        orders.append(Order("SQUID_INK", buy_price, order_size))
                        logger.print(f"SQUID_INK: ENTRY LONG {order_size} @ {buy_price}, z-score={z_score:.2f}")
                        
                        # Update position tracking
                        squid_data["in_position"] = True
                        squid_data["position_type"] = "long"
                        squid_data["entry_price"] = buy_price
                        squid_data["entry_z_score"] = z_score
                        squid_data["fair_value"] = mean_price
                        squid_data["entry_timestamp"] = timestamp
        else:
            # --- EXIT LOGIC with multiple conditions ---
            
            # Calculate time in position
            time_in_position = timestamp - squid_data["entry_timestamp"] if squid_data["entry_timestamp"] else 0
            max_hold_time_reached = time_in_position >= self.SQUID_PARAMS["max_position_hold"]
            
            if squid_data["position_type"] == "long":
                # For long positions
                entry_z = squid_data["entry_z_score"]
                
                # Target exit: price has reverted significantly toward or past mean
                target_exit = (z_score > -self.SQUID_PARAMS["z_score_exit"]) or (z_score > 0)
                
                # Stop loss: z-score moved even further negative (price dropped more)
                stop_loss = z_score < entry_z * self.SQUID_PARAMS["stop_loss_multiplier"]
                
                if target_exit or stop_loss or max_hold_time_reached:
                    exit_reason = "TARGET" if target_exit else "STOP" if stop_loss else "TIME"
                    
                    # Close long position
                    improved_price = best_bid + self.SQUID_PARAMS["limit_order_edge"]
                    
                    # Use market order for stop loss or time-based exit
                    if exit_reason in ["STOP", "TIME"]:
                        sell_price = best_bid
                    else:
                        sell_price = improved_price
                    
                    # Calculate our current long position size
                    long_position = position if position > 0 else 0
                    if long_position > 0:
                        orders.append(Order("SQUID_INK", sell_price, -long_position))
                        logger.print(f"SQUID_INK: EXIT LONG {exit_reason} {long_position} @ {sell_price}, z-score={z_score:.2f}")
                        
                        # Reset position tracking
                        squid_data["in_position"] = False
                        squid_data["position_type"] = None
                        squid_data["entry_price"] = None
                        squid_data["entry_z_score"] = None
                        squid_data["fair_value"] = None
                        squid_data["entry_timestamp"] = None
            
            elif squid_data["position_type"] == "short":
                # For short positions
                entry_z = squid_data["entry_z_score"]
                
                # Target exit: price has reverted significantly toward or past mean
                target_exit = (z_score < self.SQUID_PARAMS["z_score_exit"]) or (z_score < 0)
                
                # Stop loss: z-score moved even further positive (price rose more)
                stop_loss = z_score > entry_z * self.SQUID_PARAMS["stop_loss_multiplier"]
                
                if target_exit or stop_loss or max_hold_time_reached:
                    exit_reason = "TARGET" if target_exit else "STOP" if stop_loss else "TIME"
                    
                    # Close short position
                    improved_price = best_ask - self.SQUID_PARAMS["limit_order_edge"]
                    
                    # Use market order for stop loss or time-based exit
                    if exit_reason in ["STOP", "TIME"]:
                        buy_price = best_ask
                    else:
                        buy_price = improved_price
                    
                    # Calculate our current short position size
                    short_position = -position if position < 0 else 0
                    if short_position > 0:
                        orders.append(Order("SQUID_INK", buy_price, short_position))
                        logger.print(f"SQUID_INK: EXIT SHORT {exit_reason} {short_position} @ {buy_price}, z-score={z_score:.2f}")
                        
                        # Reset position tracking
                        squid_data["in_position"] = False
                        squid_data["position_type"] = None
                        squid_data["entry_price"] = None
                        squid_data["entry_z_score"] = None
                        squid_data["fair_value"] = None
                        squid_data["entry_timestamp"] = None
        
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        
        # Load trader data if available
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        
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
                self.LIMIT["RAINFOREST_RESIN"]
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
                self.LIMIT["KELP"]
            )
            result["KELP"] = kelp_orders

        # SQUID_INK orders with improved mean reversion strategy
        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_orders = self.squid_ink_mean_reversion(
                state.order_depths["SQUID_INK"],
                squid_position,
                self.LIMIT["SQUID_INK"],
                trader_data,
                timestamp
            )
            if squid_orders:
                result["SQUID_INK"] = squid_orders

        # Update price history in trader data for persistence
        trader_data["price_history"] = {k: list(v) for k, v in self.price_history.items()}

        traderData = jsonpickle.encode(trader_data)
        
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData