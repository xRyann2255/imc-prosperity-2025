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
        self.kelp_prices = []
        self.kelp_vwap = []
        self.squid_ink_cache = []  # Cache for recent mid prices (SQUID_INK)

    def predict_squid_ink_price(self, cache: List[float]) -> int:
        coef = [-0.01591316, 0.07101847, 0.12484902, 0.82004215]
        intercept = 0.00047196049877129553
        nxt_price = intercept + sum(c * coef[i] for i, c in enumerate(cache))
        return int(round(nxt_price))

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if buy == 0:
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask

        return tot_vol, best_val

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

        # Select levels from the order depth (these variables are not used in the rest but retained for similarity)
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
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

        # Dynamic volume filter based on order book averages
        if order_depth.sell_orders and order_depth.buy_orders:
            avg_sell_volume = sum(abs(order_depth.sell_orders[price]) for price in order_depth.sell_orders) / len(order_depth.sell_orders)
            avg_buy_volume = sum(abs(order_depth.buy_orders[price]) for price in order_depth.buy_orders) / len(order_depth.buy_orders)
            vol = math.floor(min(avg_sell_volume, avg_buy_volume))
            print("Volume filter set to: ", vol)
            fair_value = self.kelp_fair_value(order_depth, method="mid_price_with_vol_filter", min_vol=vol)
        else:
            print("Uh oh why am I here?")
            fair_value = self.kelp_fair_value(order_depth, method="mid_price")

        # Determine resting order prices
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(bbf) if bbf else fair_value - 2

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
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

    def squid_ink_orders(
        self, order_depth: OrderDepth, symbol: str, position: int, position_limit: int
    ) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Ensure the order book has both sides populated.
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return []

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        # Update the rolling cache for SQUID_INK mid prices.
        self.squid_ink_cache.append(mid_price)
        if len(self.squid_ink_cache) < 4:
            return []  # Wait for sufficient data points.
        if len(self.squid_ink_cache) > 4:
            self.squid_ink_cache.pop(0)

        predicted_price = self.predict_squid_ink_price(self.squid_ink_cache)

        # Determine resting order price levels similarly to KELP.
        aaf = [price for price in order_depth.sell_orders.keys() if price > predicted_price + 1]
        baaf = min(aaf) if aaf else predicted_price + 2
        bbf = [price for price in order_depth.buy_orders.keys() if price < predicted_price - 1]
        bbbf = max(bbf) if bbf else predicted_price - 2

        # Immediate order execution (market-taking) if conditions are met.
        best_ask_amount = -order_depth.sell_orders[best_ask]
        if best_ask < predicted_price:
            quantity = min(best_ask_amount, position_limit - position)
            if quantity > 0:
                orders.append(Order(symbol, best_ask, quantity))
                buy_order_volume += quantity

        best_bid_amount = order_depth.buy_orders[best_bid]
        if best_bid > predicted_price:
            quantity = min(best_bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order(symbol, best_bid, -quantity))
                sell_order_volume += quantity

        # Clear any adverse positions.
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, symbol, buy_order_volume, sell_order_volume, predicted_price, 1
        )

        # Place additional resting orders.
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(symbol, bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(symbol, baaf - 1, -sell_quantity))

        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        # RAINFOREST_RESIN strategy parameters
        rainforest_resin_fair_value = 10000
        rainforest_resin_width = 2
        rainforest_resin_position_limit = 50

        # KELP strategy parameters
        kelp_make_width = 4
        kelp_take_width = 1.35
        kelp_position_limit = 50
        kelp_timespan = 10

        # SQUID_INK strategy parameters
        squid_position_limit = 50

        # RAINFOREST_RESIN orders
        if "RAINFOREST_RESIN" in state.order_depths:
            rainforest_resin_position = state.position.get("RAINFOREST_RESIN", 0)
            rainforest_resin_orders = self.rainforest_resin_orders(
                state.order_depths["RAINFOREST_RESIN"],
                rainforest_resin_fair_value,
                rainforest_resin_width,
                rainforest_resin_position,
                rainforest_resin_position_limit,
            )
            result["RAINFOREST_RESIN"] = rainforest_resin_orders

        # KELP orders
        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"],
                kelp_timespan,
                kelp_make_width,
                kelp_take_width,
                kelp_position,
                kelp_position_limit,
            )
            result["KELP"] = kelp_orders

        # SQUID_INK orders (now passing position and limit)
        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_orders = self.squid_ink_orders(
                state.order_depths["SQUID_INK"],
                "SQUID_INK",
                squid_position,
                squid_position_limit,
            )
            if squid_orders:
                result["SQUID_INK"] = squid_orders

        traderData = jsonpickle.encode(
            {
                "kelp_prices": self.kelp_prices,
                "kelp_vwap": self.kelp_vwap,
            }
        )

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
