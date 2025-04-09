from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
import math


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 1.35,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 2,
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50}
        self.kelp_prices = []

    def take_best_orders(
        self,
        product: str,
        fair_value: float,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: float,
        ask: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(v for p, v in order_depth.buy_orders.items() if p >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -sent_quantity))
                sell_order_volume += sent_quantity

        if position_after_take < 0:
            clear_quantity = sum(-v for p, v in order_depth.sell_orders.items() if p <= fair_for_bid)
            clear_quantity = min(clear_quantity, -position_after_take)
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, sent_quantity))
                buy_order_volume += sent_quantity

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            avg_sell = sum(abs(v) for v in order_depth.sell_orders.values()) / len(order_depth.sell_orders)
            avg_buy = sum(abs(v) for v in order_depth.buy_orders.values()) / len(order_depth.buy_orders)
            min_vol = math.floor(min(avg_buy, avg_sell))
            valid_asks = [p for p, v in order_depth.sell_orders.items() if abs(v) >= min_vol]
            valid_bids = [p for p, v in order_depth.buy_orders.items() if abs(v) >= min_vol]
            if valid_asks and valid_bids:
                return (min(valid_asks) + max(valid_bids)) / 2
        return (min(order_depth.sell_orders) + max(order_depth.buy_orders)) / 2

    def take_orders(self, product, order_depth, fair_value, take_width, position):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position, 0, 0
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product, order_depth, fair_value, clear_width, position, buy_order_volume, sell_order_volume):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(self, product, order_depth, fair_value, position, buy_order_volume, sell_order_volume, disregard_edge, join_edge, default_edge, manage_position=False, soft_position_limit=0):
        orders: List[Order] = []
        asks_above_fair = [p for p in order_depth.sell_orders if p > fair_value + disregard_edge]
        bids_below_fair = [p for p in order_depth.buy_orders if p < fair_value - disregard_edge]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair:
            ask = best_ask_above_fair if abs(best_ask_above_fair - fair_value) <= join_edge else best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair:
            bid = best_bid_below_fair if abs(fair_value - best_bid_below_fair) <= join_edge else best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(product, orders, bid, ask, position, buy_order_volume, sell_order_volume)
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        for product in [Product.RAINFOREST_RESIN, Product.KELP]:
            if product not in self.params or product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            fair_value = self.params[product]["fair_value"] if product == Product.RAINFOREST_RESIN else self.kelp_fair_value(order_depth)

            take_orders, buy_volume, sell_volume = self.take_orders(
                product, order_depth, fair_value, self.params[product]["take_width"], position
            )
            clear_orders, buy_volume, sell_volume = self.clear_orders(
                product, order_depth, fair_value, self.params[product]["clear_width"], position, buy_volume, sell_volume
            )
            make_orders, _, _ = self.make_orders(
                product,
                order_depth,
                fair_value,
                position,
                buy_volume,
                sell_volume,
                self.params[product]["disregard_edge"],
                self.params[product]["join_edge"],
                self.params[product]["default_edge"],
                product == Product.RAINFOREST_RESIN,
                self.params[product].get("soft_position_limit", 0),
            )
            result[product] = take_orders + clear_orders + make_orders

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
