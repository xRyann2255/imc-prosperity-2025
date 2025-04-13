import sys
import math
import json
import jsonpickle
from collections import deque
from typing import Any, Dict, List, Tuple

# Import required classes from the competition's datamodel
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]],
              conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions,
                "",
                "",
            ])
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json([
                self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                self.compress_orders(orders),
                conversions,
                self.truncate(trader_data, max_item_length),
                self.truncate(self.logs, max_item_length),
            ])
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
        for symbol, depth in order_depths.items():
            compressed[symbol] = [depth.buy_orders, depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for trade_list in trades.values():
            for trade in trade_list:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_obs = {}
        for product, obs in observations.conversionObservations.items():
            conversion_obs[product] = [
                obs.bidPrice, obs.askPrice, obs.transportFees,
                obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex
            ]
        return [observations.plainValueObservations, conversion_obs]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        compressed = []
        for order_list in orders.values():
            for order in order_list:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[:max_length-3] + "..."

logger = Logger()

#################################
# Trader class – Basket Arbitrage Only
#################################
class Trader:
    def __init__(self):
        # Position limits (as provided)
        self.position_limits: Dict[str, int] = {
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60
        }
        # Tracking open arbitrage state per basket:
        self.open_position: Dict[str, bool] = {"PICNIC_BASKET1": False, "PICNIC_BASKET2": False}
        self.position_side: Dict[str, int] = {"PICNIC_BASKET1": 0, "PICNIC_BASKET2": 0}  # -1: short, +1: long
        self.trade_size_factor: Dict[str, int] = {"PICNIC_BASKET1": 1, "PICNIC_BASKET2": 1}
        self.max_size_factor: Dict[str, int] = {"PICNIC_BASKET1": 41, "PICNIC_BASKET2": 62}
        self.current_units: Dict[str, int] = {"PICNIC_BASKET1": 0, "PICNIC_BASKET2": 0}
        self.entry_diff: Dict[str, float] = {"PICNIC_BASKET1": 0.0, "PICNIC_BASKET2": 0.0}
        self.entry_time: Dict[str, int] = {"PICNIC_BASKET1": 0, "PICNIC_BASKET2": 0}
        
        # Set a minimal profit difference threshold very low to trigger trades 
        self.min_profit_diff = 0.01  
        # Stop loss parameters: if the trade is held too long or mispricing worsens, force exit.
        self.stop_loss_time = 10000   # in ms
        self.stop_loss_diff = 10      # adverse diff (price units)

    def arbitrage_basket(self, state: TradingState, basket_symbol: str) -> List[Order]:
        """
        Simple arbitrage strategy using top-of-book prices.
        For PICNIC_BASKET1, components are:
             6 CROISSANTS, 3 JAMS, 1 DJEMBES.
        For PICNIC_BASKET2:
             4 CROISSANTS, 2 JAMS.
        Entry is triggered if the basket's price is mismatched with its components by any positive margin.
        Exit is triggered when the mispricing disappears or a stop loss time is reached.
        Martingale doubling is applied if the exit results in a loss.
        """
        orders: List[Order] = []
        
        # Define components for each basket
        if basket_symbol == "PICNIC_BASKET1":
            comp_ratio = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
        elif basket_symbol == "PICNIC_BASKET2":
            comp_ratio = {"CROISSANTS": 4, "JAMS": 2}
        else:
            logger.print(f"Unknown basket symbol: {basket_symbol}")
            return orders

        # Check all required order depths are present.
        for sym in [basket_symbol] + list(comp_ratio.keys()):
            if sym not in state.order_depths:
                logger.print(f"{basket_symbol}: Missing order depth for {sym}")
                return orders

        # Retrieve top-of-book prices for basket.
        depth_b = state.order_depths[basket_symbol]
        if not depth_b.buy_orders or not depth_b.sell_orders:
            return orders
        basket_bid = max(depth_b.buy_orders.keys())
        basket_ask = min(depth_b.sell_orders.keys())
        basket_bid_vol = depth_b.buy_orders[basket_bid]
        basket_ask_vol = depth_b.sell_orders[basket_ask]

        # Retrieve top-of-book prices for each component.
        comp_best_ask = {}
        comp_best_bid = {}
        comp_ask_vol = {}
        comp_bid_vol = {}
        for comp, ratio in comp_ratio.items():
            depth_c = state.order_depths[comp]
            if not depth_c.buy_orders or not depth_c.sell_orders:
                return orders
            comp_best_ask[comp] = min(depth_c.sell_orders.keys())
            comp_best_bid[comp] = max(depth_c.buy_orders.keys())
            comp_ask_vol[comp] = depth_c.sell_orders[comp_best_ask[comp]]
            comp_bid_vol[comp] = depth_c.buy_orders[comp_best_bid[comp]]
        
        # Compute synthetic prices for one basket unit.
        synthetic_cost = sum(comp_best_ask[c] * comp_ratio[c] for c in comp_ratio)
        synthetic_value = sum(comp_best_bid[c] * comp_ratio[c] for c in comp_ratio)
        diff_over = basket_bid - synthetic_cost      # potential profit if basket is overpriced.
        diff_under = synthetic_value - basket_ask     # potential profit if basket is underpriced.

        logger.print(f"{basket_symbol}: diff_over={diff_over:.2f}, diff_under={diff_under:.2f}")
        
        # ***************************
        # ENTRY: No open position – enter if mispricing is positive
        # ***************************
        if not self.open_position[basket_symbol]:
            if diff_over > self.min_profit_diff and basket_bid_vol > 0:
                # Short basket arbitrage entry:
                trade_units = self.trade_size_factor[basket_symbol]
                # Limit by available volume on basket side:
                trade_units = min(trade_units, basket_bid_vol)
                # Limit each component:
                for comp, ratio in comp_ratio.items():
                    trade_units = min(trade_units, comp_ask_vol[comp] // ratio)
                if trade_units <= 0:
                    logger.print(f"{basket_symbol}: Not enough volume for SHORT entry.")
                else:
                    logger.print(f"{basket_symbol}: Enter SHORT arbitrage for {trade_units} units (diff_over={diff_over:.2f}).")
                    orders.append(Order(basket_symbol, basket_bid, -trade_units))
                    for comp, ratio in comp_ratio.items():
                        orders.append(Order(comp, comp_best_ask[comp], trade_units * ratio))
                    self.open_position[basket_symbol] = True
                    self.position_side[basket_symbol] = -1
                    self.current_units[basket_symbol] = trade_units
                    self.entry_diff[basket_symbol] = diff_over
                    self.entry_time[basket_symbol] = state.timestamp
            elif diff_under > self.min_profit_diff and basket_ask_vol > 0:
                # Long basket arbitrage entry:
                trade_units = self.trade_size_factor[basket_symbol]
                trade_units = min(trade_units, basket_ask_vol)
                for comp, ratio in comp_ratio.items():
                    trade_units = min(trade_units, comp_bid_vol[comp] // ratio)
                if trade_units <= 0:
                    logger.print(f"{basket_symbol}: Not enough volume for LONG entry.")
                else:
                    logger.print(f"{basket_symbol}: Enter LONG arbitrage for {trade_units} units (diff_under={diff_under:.2f}).")
                    orders.append(Order(basket_symbol, basket_ask, trade_units))
                    for comp, ratio in comp_ratio.items():
                        orders.append(Order(comp, comp_best_bid[comp], -trade_units * ratio))
                    self.open_position[basket_symbol] = True
                    self.position_side[basket_symbol] = 1
                    self.current_units[basket_symbol] = trade_units
                    self.entry_diff[basket_symbol] = -diff_under
                    self.entry_time[basket_symbol] = state.timestamp
            else:
                logger.print(f"{basket_symbol}: No entry conditions met (diffs: {diff_over:.2f} / {diff_under:.2f}).")
        
        # ***************************
        # EXIT: If a position is already open, exit when mispricing vanishes or stops
        # ***************************
        else:
            open_side = self.position_side[basket_symbol]
            close_trade = False
            if open_side == -1 and diff_over <= 0:
                close_trade = True
            if open_side == 1 and diff_under <= 0:
                close_trade = True
            # Force exit if held for too long.
            held_time = state.timestamp - self.entry_time[basket_symbol] if self.entry_time[basket_symbol] else 0
            if held_time > self.stop_loss_time:
                logger.print(f"{basket_symbol}: Time stop reached ({held_time}ms).")
                close_trade = True

            if close_trade:
                trade_units = self.current_units[basket_symbol]
                if open_side == -1:
                    trade_units = min(trade_units, basket_ask_vol)
                    for comp, ratio in comp_ratio.items():
                        trade_units = min(trade_units, comp_bid_vol[comp] // ratio)
                    if trade_units > 0:
                        logger.print(f"{basket_symbol}: Exiting SHORT arbitrage for {trade_units} units.")
                        orders.append(Order(basket_symbol, basket_ask, trade_units))
                        for comp, ratio in comp_ratio.items():
                            orders.append(Order(comp, comp_best_bid[comp], -trade_units * ratio))
                elif open_side == 1:
                    trade_units = min(trade_units, basket_bid_vol)
                    for comp, ratio in comp_ratio.items():
                        trade_units = min(trade_units, comp_ask_vol[comp] // ratio)
                    if trade_units > 0:
                        logger.print(f"{basket_symbol}: Exiting LONG arbitrage for {trade_units} units.")
                        orders.append(Order(basket_symbol, basket_bid, -trade_units))
                        for comp, ratio in comp_ratio.items():
                            orders.append(Order(comp, comp_best_ask[comp], trade_units * ratio))
                
                # Determine if trade lost money (stop loss) – simple criteria:
                trade_loss = False
                if held_time > self.stop_loss_time:
                    trade_loss = True
                if open_side == -1 and diff_over > 0:
                    trade_loss = True
                if open_side == 1 and diff_under > 0:
                    trade_loss = True
                if trade_loss:
                    self.trade_size_factor[basket_symbol] = min(self.trade_size_factor[basket_symbol] * 2,
                                                                 self.max_size_factor[basket_symbol])
                    logger.print(f"{basket_symbol}: Trade loss. Doubling next trade size to {self.trade_size_factor[basket_symbol]}.")
                else:
                    self.trade_size_factor[basket_symbol] = 1
                    logger.print(f"{basket_symbol}: Profit taken. Resetting trade size to 1.")
                self.current_units[basket_symbol] = 0
                self.open_position[basket_symbol] = False
                self.position_side[basket_symbol] = 0

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """
        Main entry point for basket arbitrage only.
        This strategy ignores RAINFOREST_RESIN, KELP, and SQUID_INK.
        """
        result: Dict[Symbol, List[Order]] = {}
        # Process PICNIC_BASKET1 arbitrage
        b1_orders = self.arbitrage_basket(state, "PICNIC_BASKET1")
        for order in b1_orders:
            result.setdefault(order.symbol, []).append(order)
        # Update state positions virtually for basket2
        if b1_orders:
            net_changes: Dict[str, int] = {}
            for order in b1_orders:
                net_changes[order.symbol] = net_changes.get(order.symbol, 0) + order.quantity
            for sym, change in net_changes.items():
                state.position[sym] = state.position.get(sym, 0) + change
        # Process PICNIC_BASKET2 arbitrage
        b2_orders = self.arbitrage_basket(state, "PICNIC_BASKET2")
        for order in b2_orders:
            result.setdefault(order.symbol, []).append(order)

        conversions = 0
        trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
