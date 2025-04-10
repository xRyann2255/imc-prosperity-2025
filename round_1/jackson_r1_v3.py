import json
from typing import Any, Dict, List
import typing
import jsonpickle
import numpy as np
import math
import string
import pandas as pd

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions, "", ""]))

        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
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
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
                      for p, o in observations.conversionObservations.items()}
        return [observations.plainValueObservations, conversion]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        return value if len(value) <= max_length else value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.kelp_prices = []
        self.kelp_vwap = []
        self.ink_order_book = []
        self.ink_prices = []


    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        baaf = min([p for p in order_depth.sell_orders if p > fair_value + 1], default=fair_value + 2)
        bbbf = max([p for p in order_depth.buy_orders if p < fair_value - 1], default=fair_value - 2)

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                qty = min(ask_amount, position_limit - position)
                if qty > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, qty))
                    buy_order_volume += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                qty = min(bid_amount, position_limit + position)
                if qty > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -qty))
                    sell_order_volume += qty

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, width)

        buy_qty = position_limit - (position + buy_order_volume)
        if buy_qty > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_qty))

        sell_qty = position_limit + (position - sell_order_volume)
        if sell_qty > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_qty))

        return orders

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str,
                             buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int) -> typing.Tuple[int, int]:
        pos_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_limit = position_limit - (position + buy_order_volume)
        sell_limit = position_limit + (position - sell_order_volume)

        if pos_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            qty = min(order_depth.buy_orders[fair_for_ask], pos_after_take, sell_limit)
            if qty > 0:
                orders.append(Order(product, fair_for_ask, -qty))
                sell_order_volume += qty

        if pos_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            qty = min(abs(order_depth.sell_orders[fair_for_bid]), abs(pos_after_take), buy_limit)
            if qty > 0:
                orders.append(Order(product, fair_for_bid, qty))
                buy_order_volume += qty

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, method="mid_price", min_vol=0) -> float:
        if method == "mid_price":
            return (min(order_depth.sell_orders) + max(order_depth.buy_orders)) / 2
        elif method == "mid_price_with_vol_filter":
            valid_asks = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= min_vol]
            valid_bids = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= min_vol]
            if valid_asks and valid_bids:
                return (min(valid_asks) + max(valid_bids)) / 2
            return self.kelp_fair_value(order_depth, method="mid_price")

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, width: float, take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_vol, sell_vol = 0, 0

        if order_depth.sell_orders and order_depth.buy_orders:
            avg_sell = sum(abs(v) for v in order_depth.sell_orders.values()) / len(order_depth.sell_orders)
            avg_buy = sum(abs(v) for v in order_depth.buy_orders.values()) / len(order_depth.buy_orders)
            vol = math.floor(min(avg_buy, avg_sell))
            fair = self.kelp_fair_value(order_depth, method="mid_price_with_vol_filter", min_vol=vol)
        else:
            fair = self.kelp_fair_value(order_depth, method="mid_price")

        baaf = min([p for p in order_depth.sell_orders if p > fair + 1], default=fair + 2)
        bbbf = max([p for p in order_depth.buy_orders if p < fair - 1], default=fair - 2)

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            ask_amt = -order_depth.sell_orders[best_ask]
            if best_ask < fair:
                qty = min(ask_amt, position_limit - position)
                if qty > 0:
                    orders.append(Order("KELP", best_ask, qty))
                    buy_vol += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            bid_amt = order_depth.buy_orders[best_bid]
            if best_bid > fair:
                qty = min(bid_amt, position_limit + position)
                if qty > 0:
                    orders.append(Order("KELP", best_bid, -qty))
                    sell_vol += qty

        buy_vol, sell_vol = self.clear_position_order(orders, order_depth, position, position_limit, "KELP", buy_vol, sell_vol, fair, 1)

        if (qty := position_limit - (position + buy_vol)) > 0:
            orders.append(Order("KELP", bbbf + 1, qty))
        if (qty := position_limit + (position - sell_vol)) > 0:
            orders.append(Order("KELP", baaf - 1, -qty))

        return orders

    def ink_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        mid_price = (min(order_depth.sell_orders) + max(order_depth.buy_orders)) / 2
        self.ink_prices.append(mid_price)
        self.ink_order_book.append(order_depth)

        fair = self.meta_learning_fair_value(order_depth, k=2, timespan=100, fallback_value=mid_price)
        baaf = int(min([p for p in order_depth.sell_orders if p > fair + 1], default=fair + 2))
        bbbf = int(max([p for p in order_depth.buy_orders if p < fair - 1], default=fair - 2))

        buy_vol, sell_vol = 0, 0
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            ask_amt = -order_depth.sell_orders[best_ask]
            if best_ask < fair:
                qty = min(ask_amt, position_limit - position)
                if qty > 0:
                    orders.append(Order("SQUID_INK", best_ask, qty))
                    buy_vol += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            bid_amt = order_depth.buy_orders[best_bid]
            if best_bid > fair:
                qty = min(bid_amt, position_limit + position)
                if qty > 0:
                    orders.append(Order("SQUID_INK", best_bid, -qty))
                    sell_vol += qty

        buy_vol, sell_vol = self.clear_position_order(orders, order_depth, position, position_limit, "SQUID_INK", buy_vol, sell_vol, fair, 1)

        if (qty := position_limit - (position + buy_vol)) > 0:
            orders.append(Order("SQUID_INK", bbbf + 1, qty))
        if (qty := position_limit + (position - sell_vol)) > 0:
            orders.append(Order("SQUID_INK", baaf - 1, -qty))

        return orders

    def meta_learning_fair_value(self, order_depth: OrderDepth, k: int, timespan: int, fallback_value: float) -> float:
        if len(self.ink_prices) < timespan:
            return fallback_value

        recent_midprices = self.ink_prices[-timespan:]
        recent_books = self.ink_order_book[-timespan:]

        y = np.log(recent_midprices[k:]) - np.log(recent_midprices[:-k])
        x = [self.preprocess_order_book(self.order_to_df(order, num_level=5)) for order in recent_books[:-k]]
        x = pd.concat(x)

        train_size = int(0.9 * len(x))
        if train_size < k:
            return fallback_value

        X_train = x[:train_size].values
        y_train = y[:train_size]
        X_train_aug = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        coeffs = np.linalg.lstsq(X_train_aug, y_train, rcond=None)[0]

        cur_df = self.order_to_df(order_depth, num_level=5)
        cur_feat = self.preprocess_order_book(cur_df).iloc[-1].values.reshape(1, -1)
        cur_aug = np.hstack((np.ones((1, 1)), cur_feat))
        pred_log_return = cur_aug.dot(coeffs)[0]

        return fallback_value * np.exp(pred_log_return)

    def preprocess_order_book(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    def order_to_df(self, order_depth: OrderDepth, num_level=5) -> pd.DataFrame:
        buy = sorted(order_depth.buy_orders.items(), reverse=True)[:num_level]
        sell = sorted(order_depth.sell_orders.items())[:num_level]

        data = {}
        for i in range(num_level):
            data[f"buy_price_{i}"] = buy[i][0] if i < len(buy) else 0
            data[f"buy_vol_{i}"] = buy[i][1] if i < len(buy) else 0
            data[f"sell_price_{i}"] = sell[i][0] if i < len(sell) else 0
            data[f"sell_vol_{i}"] = sell[i][1] if i < len(sell) else 0

        return pd.DataFrame([data])


    def run(self, state: TradingState) -> tuple:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 1

        if "RAINFOREST_RESIN" in state.order_depths:
            pos = state.position.get("RAINFOREST_RESIN", 0)
            result["RAINFOREST_RESIN"] = self.rainforest_resin_orders(state.order_depths["RAINFOREST_RESIN"], 10000, 2, pos, 50)

        if "KELP" in state.order_depths:
            pos = state.position.get("KELP", 0)
            result["KELP"] = self.kelp_orders(state.order_depths["KELP"], 10, 4, 1.35, pos, 50)

        if "SQUID_INK" in state.order_depths:
            pos = state.position.get("SQUID_INK", 0)
            result["SQUID_INK"] = self.ink_orders(state.order_depths["SQUID_INK"], pos, 50)

        traderData = jsonpickle.encode({"kelp_prices": self.kelp_prices, "kelp_vwap": self.kelp_vwap})
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
