from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Tuple
import string
import jsonpickle
import numpy as np
import math
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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
                    self.compress_state(state, self.truncate(
                        "state.traderData", max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    # self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
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

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
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

    def compress_observations(self, observations: Observation) -> list[Any]:
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

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
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


class Trader:
    def __init__(self) -> None:
        test = 0

    def agent_basket1_basket2(self, traderData: dict,
                              pb1_order_depth: OrderDepth, pb2_order_depth: OrderDepth, d_order_depth: OrderDepth,
                              pb1_position: int, pb2_position: int, d_position: int,
                              pb1_position_limit: int, pb2_position_limit: int, d_position_limit: int,
                              spread_avg: float, spread_std: float, z_thresh: float, exit_thesh: float, timeframe: int, std_scaling: float, std_bandwidth: float
                              ):

        pb1_best_ask = min(pb1_order_depth.sell_orders.keys())
        pb1_best_bid = max(pb1_order_depth.buy_orders.keys())
        pb1_mid_price = (pb1_best_ask + pb1_best_bid) / 2

        pb2_best_ask = min(pb2_order_depth.sell_orders.keys())
        pb2_best_bid = max(pb2_order_depth.buy_orders.keys())
        pb2_mid_price = (pb2_best_ask + pb2_best_bid) / 2

        d_best_ask = min(d_order_depth.sell_orders.keys())
        d_best_bid = max(d_order_depth.buy_orders.keys())
        d_mid_price = (d_best_ask + d_best_bid) / 2

        pb1_quantity, pb2_quantity, d_quantity = 0, 0, 0
        pb1_price, pb2_price, d_price = 0, 0, 0
        spread = pb1_mid_price - (1.5 * pb2_mid_price + d_mid_price)

        traderData["pb1_pb2_spread"].append(spread)
        length = len(traderData["pb1_pb2_spread"])
        add = 0
        if length > timeframe:
            traderData["pb1_pb2_spread"].pop(0)
            std = np.std(traderData["pb1_pb2_spread"])
            traderData["pb1_pb2_std"] = std
            add = std_scaling * std - std_bandwidth

        if spread > spread_avg + spread_std * z_thresh + add:
            pb1_limit = min(pb1_position_limit + pb1_position,
                            abs(pb1_order_depth.buy_orders[pb1_best_bid]))  # sell
            pb2_limit = min(pb2_position_limit-pb2_position,
                            abs(pb2_order_depth.sell_orders[pb2_best_ask])) / 1.5  # buy
            d_limit = min(d_position_limit-d_position,
                          abs(d_order_depth.sell_orders[d_best_ask]))  # buy

            limit = math.floor(min(pb1_limit, pb2_limit, d_limit))
            limit = (limit // 2) * 2
            pb1_quantity, pb2_quantity, d_quantity = -limit, 1.5 * limit, limit
            pb1_price, pb2_price, d_price = pb1_best_bid, pb2_best_ask, d_best_ask

        elif spread < spread_avg - spread_std * z_thresh - add:
            pb1_limit = min(pb1_position_limit - pb1_position,
                            abs(pb1_order_depth.sell_orders[pb1_best_ask]))  # buy
            pb2_limit = min(pb2_position_limit+pb2_position,
                            abs(pb2_order_depth.buy_orders[pb2_best_bid])) / 1.5  # sell
            d_limit = min(d_position_limit+d_position,
                          abs(d_order_depth.buy_orders[d_best_bid]))  # sell

            limit = math.floor(min(pb1_limit, pb2_limit, d_limit))
            limit = (limit // 2) * 2
            pb1_quantity, pb2_quantity, d_quantity = limit, -1.5 * limit, -limit
            pb1_price, pb2_price, d_price = pb1_best_ask, pb2_best_bid, d_best_bid

        elif pb1_position > 0 and spread > spread_avg - (spread_std * z_thresh + add) * exit_thesh:
            pb1_limit = min(abs(pb1_position), abs(
                pb1_order_depth.buy_orders[pb1_best_bid]))  # sell
            pb2_limit = min(abs(pb2_position), abs(
                pb2_order_depth.sell_orders[pb2_best_ask])) / 1.5  # buy
            d_limit = min(abs(d_position), abs(
                d_order_depth.sell_orders[d_best_ask]))  # buy

            limit = math.floor(min(pb1_limit, pb2_limit, d_limit))
            limit = (limit // 2) * 2
            pb1_quantity, pb2_quantity, d_quantity = -limit, 1.5 * limit, limit
            pb1_price, pb2_price, d_price = pb1_best_bid, pb2_best_ask, d_best_ask

        elif pb1_position < 0 and spread < spread_avg + (spread_std * z_thresh + add) * exit_thesh:
            pb1_limit = min(abs(pb1_position), abs(
                pb1_order_depth.sell_orders[pb1_best_ask]))  # buy
            pb2_limit = min(abs(pb2_position), abs(
                pb2_order_depth.buy_orders[pb2_best_bid])) / 1.5  # sell
            d_limit = min(abs(d_position), abs(
                d_order_depth.buy_orders[d_best_bid]))  # sell

            limit = math.floor(min(pb1_limit, pb2_limit, d_limit))
            limit = (limit // 2) * 2
            pb1_quantity, pb2_quantity, d_quantity = limit, -1.5 * limit, -limit
            pb1_price, pb2_price, d_price = pb1_best_ask, pb2_best_bid, d_best_bid

        return pb1_price, pb1_quantity, pb2_price, pb2_quantity, d_price, d_quantity

    def basket_orders(self,
                      order_depths: OrderDepth, traderData: dict,
                      pb1_position_limit: int, pb2_position_limit: int, d_position_limit: int, j_position_limit: int, c_position_limit: int,
                      pb1_pb2_spread_avg: float, pb1_pb2_spread_std: float, pb1_pb2_z_thresh: float, pb1_pb2_exit_thesh: float, pb1_pb2_timeframe: int, pb1_pb2_std_scaling: float, pb1_pb2_std_bandwidth: float
                      ) -> Tuple[List[Order], List[Order], List[Order], List[Order], List[Order]]:
        pb1_orders: List[Order] = []
        pb2_orders: List[Order] = []
        d_orders: List[Order] = []
        j_orders: List[Order] = []
        c_orders: List[Order] = []

        if "PICNIC_BASKET1" in order_depths and "PICNIC_BASKET2" in order_depths and "DJEMBES" in order_depths:
            new_pb1_position_limit = 60
            new_pb2_position_limit = 1.5 * new_pb1_position_limit
            new_d_position_limit = new_pb1_position_limit

            pb1_position = traderData["pb1_pb2_vol_pb1"]
            pb2_position = traderData["pb1_pb2_vol_pb2"]
            d_position = traderData["pb1_pb2_vol_d"]

            pb1_price2, pb1_quantity2, pb2_price2, pb2_quantity2, d_price2, d_quantity2 = self.agent_basket1_basket2(
                traderData,
                order_depths["PICNIC_BASKET1"], order_depths["PICNIC_BASKET2"], order_depths["DJEMBES"],
                pb1_position, pb2_position, d_position,
                new_pb1_position_limit, new_pb2_position_limit, new_d_position_limit,
                pb1_pb2_spread_avg, pb1_pb2_spread_std, pb1_pb2_z_thresh, pb1_pb2_exit_thesh, pb1_pb2_timeframe, pb1_pb2_std_scaling, pb1_pb2_std_bandwidth
            )
            traderData["pb1_pb2_vol_pb1"] = pb1_position + pb1_quantity2
            traderData["pb1_pb2_vol_pb2"] = pb2_position + pb2_quantity2
            traderData["pb1_pb2_vol_d"] = d_position + d_quantity2

        pb1_orders.append(
            Order("PICNIC_BASKET1", pb1_price2, round(pb1_quantity2)))
        pb2_orders.append(
            Order("PICNIC_BASKET2", pb2_price2, round(pb2_quantity2)))
        d_orders.append(Order("DJEMBES", d_price2, round(d_quantity2)))

        return pb1_orders, pb2_orders, d_orders, j_orders, c_orders

    def run(self, state: TradingState):
        result = {}
# -------HYPER_PARAMETERS:------------------

    # Round 2: Pair Trading

        # PICNIC_BASKET1 -> pb1
        pb1_position_limit = 60

        # PICNIC_BASKET2 -> pb2
        pb2_position_limit = 100

        # DJEMBES -> d
        d_position_limit = 60

        # JAMS -> j
        j_position_limit = 350

        # CROISSANTS -> c
        c_position_limit = 250

        # Pairs:

        pb1_pb2_spread_avg = 3.408
        pb1_pb2_spread_std = 93.506
        pb1_pb2_z_thresh = 0.75
        pb1_pb2_exit_thesh = -1
        pb1_pb2_timeframe = 300
        pb1_pb2_std_scaling = 3
        pb1_pb2_std_bandwidth = 40


# -------load traderData--------------------
        if state.traderData == "":
            trader_data = {
                "pb1_pb2_vol_pb1": 0,
                "pb1_pb2_vol_pb2": 0,
                "pb1_pb2_vol_d": 0,
                "tic": 0,
                "pb1_pb2_std": 0,
                "pb1_pb2_spread": [],

            }
            state.traderData = trader_data
        else:
            state.traderData = jsonpickle.loads(state.traderData)

# -------orders for products----------------

        pb1_orders, pb2_orders, d_orders, j_orders, c_orders = self.basket_orders(
            state.order_depths, state.traderData,
            pb1_position_limit, pb2_position_limit, d_position_limit, j_position_limit, c_position_limit,
            pb1_pb2_spread_avg, pb1_pb2_spread_std, pb1_pb2_z_thresh, pb1_pb2_exit_thesh, pb1_pb2_timeframe, pb1_pb2_std_scaling, pb1_pb2_std_bandwidth
        )
        result["PICNIC_BASKET1"] = pb1_orders
        result["PICNIC_BASKET2"] = pb2_orders
        result["DJEMBES"] = d_orders
        result["JAMS"] = j_orders
        result["CROISSANTS"] = c_orders

        traderData = jsonpickle.encode(state.traderData)
        conversions = 1
        logger.flush(state, result, conversions, state.traderData)
        return result, conversions, traderData
