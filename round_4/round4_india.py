import numpy as np
import math
from math import log, sqrt, exp, erf
from statistics import *
from datamodel import *
from abc import ABC, abstractmethod
from typing import Deque, Optional, Any, List, Tuple
from collections import deque
import statistics
from enum import IntEnum

# JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

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
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for trade_list in trades.values() for t in trade_list
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv_obs = {
            p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sunlightIndex, o.sugarPrice]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conv_obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for order_list in orders.values() for o in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        for i in range(min(len(value), max_length), 0, -1):
            candidate = value[:i] + ("..." if i < len(value) else "")
            if len(json.dumps(candidate)) <= max_length:
                return candidate
        return ""

logger = Logger()

class Signal(IntEnum):
    NEUTRAL = 0
    SELL = 1
    BUY = 2
    DO_NOTHING = -1
    BUY_SOS = 3
    SELL_SOS = 4

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.orders: list[Order] = []

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState, *args) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        if price is None:
            return
        self.orders.append(Order(self.symbol, round(price), quantity))

    def sell(self, price: int, quantity: int) -> None:
        if price is None:
            return
        self.orders.append(Order(self.symbol, round(price), -quantity))

    def save(self):
        return None

    def load(self, data, *args) -> None:
        self.args = args
        pass

    # @abstractmethod
    def get_mid_price(state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if len(buy_orders) > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        if len(sell_orders) > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        
        if len(buy_orders) == 0 and len(sell_orders) > 0:
            return popular_sell_price
        if len(sell_orders) == 0 and len(buy_orders) > 0:
            return popular_buy_price
        elif len(sell_orders) == len(buy_orders) == 0:
            return None

        return (popular_buy_price + popular_sell_price) / 2

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, window_size:int = 10, soft_position_limit: float = 0.5, price_alt:int = 1) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = window_size
        self.soft_position_limit = soft_position_limit
        self.price_alt = price_alt
        
    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        if true_value is None:
            return

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - self.price_alt if position > self.limit * self.soft_position_limit else true_value
        min_sell_price = true_value + self.price_alt if position < self.limit * -self.soft_position_limit else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0 and sell_orders:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self):
        return list(self.window)

    def load(self, data, *args) -> None:
        self.args = args
        self.window = deque(data)

class SignalStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, *args) -> None:
        super().__init__(symbol, limit)
        self.args = args
        self.signal = Signal.NEUTRAL

    @abstractmethod
    def get_signal(self, state: TradingState):
        raise NotImplementedError()

    def act(self, state: TradingState, *args) -> None:
        new_signal = self.get_signal(state)
        if new_signal == Signal.DO_NOTHING:
            return
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]
            
        buy_price = self.get_buy_price(order_depth)
        sell_price = self.get_sell_price(order_depth)

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                if buy_price is None:
                    return
                self.buy(buy_price, -position)
            elif position > 0:
                if sell_price is None:
                    return
                self.sell(sell_price, position)
        elif self.signal == Signal.SELL:
            if sell_price is None:
                return
            self.sell(sell_price, self.limit + position)
        elif self.signal == Signal.BUY:
            if buy_price is None:
                return
            self.buy(buy_price, self.limit - position)


    def get_buy_price(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

    def get_sell_price(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

    def save(self):
        return self.signal.value

    def load(self, data, *args) -> None:
        self.args = args
        self.signal = Signal(data)
        
class MeanReversionStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, limit: int, window_size: int = 10, z_score_threshold: float = 2.0) -> None:
        super().__init__(symbol, limit)
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.prices = deque(maxlen=window_size)

    def get_signal(self, state: TradingState):
        order_depth = state.order_depths[self.symbol]
        price = self.get_buy_price(order_depth)

        self.prices.append(price)

        if len(self.prices) < self.window_size:
            return Signal.NEUTRAL

        mean_price = statistics.mean(self.prices)
        std_dev = statistics.stdev(self.prices)

        if std_dev == 0:
            return Signal.NEUTRAL

        z_score = (price - mean_price) / std_dev

        if z_score > self.z_score_threshold:
            return Signal.BUY
        elif z_score < -self.z_score_threshold:
            return Signal.SELL

        return Signal.NEUTRAL
    
    def save(self):
        return {
            "window_size": self.window_size,
            "z_score_threshold": self.z_score_threshold,
            "prices": list(self.prices),}
        
    def load(self, data, *args) -> None:
        self.args = args
        self.window_size = data["window_size"]
        self.z_score_threshold = data["z_score_threshold"]
        self.prices = deque(data["prices"], maxlen=self.window_size)

class ResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10000

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buys = sorted(order_depth.buy_orders.items(), reverse=True)[:3]
        sells = sorted(order_depth.sell_orders.items())[:3]

        buy_vwap = sum(p * v for p, v in buys) / sum(v for _, v in buys) if buys else 0
        sell_vwap = sum(p * -v for p, v in sells) / sum(-v for _, v in sells) if sells else 0

        return round((buy_vwap + sell_vwap) / 2) if buy_vwap and sell_vwap else None
    
class SquidInkStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.LIMIT = 50
        self.window_size = 10
        self.ink_history = deque(maxlen=self.window_size)
        self.last_price = None

    def get_fair_value(self, state: TradingState, traderObject: dict) -> float:
        order_depth = state.order_depths[self.symbol]
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= 15
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= 15
            ]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mmmid_price = (best_ask + best_bid) / 2 if traderObject.get("SQUID_INK_last_price") is None else traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("SQUID_INK_last_price") is not None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                if len(self.ink_history) >= self.window_size:
                    returns = np.diff(list(self.ink_history)) / list(self.ink_history)[:-1]
                    volatility = np.std(returns)
                    dynamic_beta = -0.15 * (1 + volatility * 100)
                    pred_returns = last_returns * dynamic_beta
                else:
                    pred_returns = last_returns * -0.15
                fair = mmmid_price + mmmid_price * pred_returns
            else:
                fair = mmmid_price
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None

    def act(self, state: TradingState) -> None:
        import jsonpickle
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}

        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            self.ink_history.append((best_bid + best_ask) / 2)

        fair = self.get_fair_value(state, traderObject)
        if fair is None:
            return
        take_width = 1
        clear_width = 0
        disregard_edge = 2
        join_edge = 1
        default_edge = 2
        soft_position_limit = 20

        buy_orders = []
        sell_orders = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            vol = -order_depth.sell_orders[best_ask]
            if best_ask <= fair - take_width and vol <= 15:
                qty = min(vol, self.LIMIT - position)
                if qty > 0:
                    buy_orders.append(Order(self.symbol, best_ask, qty))
                    buy_order_volume += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            vol = order_depth.buy_orders[best_bid]
            if best_bid >= fair + take_width and vol <= 15:
                qty = min(vol, self.LIMIT + position)
                if qty > 0:
                    sell_orders.append(Order(self.symbol, best_bid, -qty))
                    sell_order_volume += qty

        net_pos = position + buy_order_volume - sell_order_volume
        fair_bid = round(fair - clear_width)
        fair_ask = round(fair + clear_width)
        buy_qty = self.LIMIT - (position + buy_order_volume)
        sell_qty = self.LIMIT + (position - sell_order_volume)

        if net_pos > 0:
            clear_qty = sum(v for p, v in order_depth.buy_orders.items() if p >= fair_ask)
            sent_qty = min(sell_qty, min(clear_qty, net_pos))
            if sent_qty > 0:
                sell_orders.append(Order(self.symbol, fair_ask, -sent_qty))
                sell_order_volume += sent_qty

        if net_pos < 0:
            clear_qty = sum(-v for p, v in order_depth.sell_orders.items() if p <= fair_bid)
            sent_qty = min(buy_qty, min(clear_qty, -net_pos))
            if sent_qty > 0:
                buy_orders.append(Order(self.symbol, fair_bid, sent_qty))
                buy_order_volume += sent_qty

        asks_above_fair = [p for p in order_depth.sell_orders if p > fair + disregard_edge]
        bids_below_fair = [p for p in order_depth.buy_orders if p < fair - disregard_edge]

        best_ask_above = min(asks_above_fair) if asks_above_fair else None
        best_bid_below = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair + default_edge)
        if best_ask_above:
            ask = best_ask_above if abs(best_ask_above - fair) <= join_edge else best_ask_above - 1

        bid = round(fair - default_edge)
        if best_bid_below:
            bid = best_bid_below if abs(fair - best_bid_below) <= join_edge else best_bid_below + 1

        if position > soft_position_limit:
            ask -= 1
        elif position < -soft_position_limit:
            bid += 1

        if self.LIMIT - (position + buy_order_volume) > 0:
            buy_orders.append(Order(self.symbol, bid, self.LIMIT - (position + buy_order_volume)))
        if self.LIMIT + (position - sell_order_volume) > 0:
            sell_orders.append(Order(self.symbol, ask, -(self.LIMIT + (position - sell_order_volume))))

        self.orders.extend(buy_orders + sell_orders)

    def save(self):
        return {
            "ink_history": list(self.ink_history),
            "last_price": self.last_price
        }

    def load(self, data) -> None:
        if data:
            self.ink_history = deque(data.get("ink_history", []), maxlen=self.window_size)
            self.last_price = data.get("last_price", None)

class VolumeWeightedStrategy(KelpStrategy):
    def _init_(self, symbol: str, limit: int) -> None:
        super()._init_(symbol, limit)
        self.recent_trade_window = 30

    def get_true_value(self, state: TradingState) -> int:
        depth = state.order_depths[self.symbol]
        
        top_buys = sorted(depth.buy_orders.items(), reverse=True)[:3]
        top_sells = sorted(depth.sell_orders.items())[:3]

        total_buy_vol = sum(v for p, v in top_buys)
        total_sell_vol = sum(v for p, v in top_sells)
        
        weighted_buy = sum(p * v for p, v in top_buys) / total_buy_vol if total_buy_vol > 0 else 0
        weighted_sell = sum(p * v for p, v in top_sells) / total_sell_vol if total_sell_vol > 0 else 0

        fair_value = (weighted_buy + weighted_sell) / 2
        
        return round(fair_value)

class PicnicBasket1Strategy(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.components = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
        self.spread = 60  # Market making spread
        self.cooldown_ticks = 0
        self.cooldown_long = 0
        self.cooldown_short = 0

    def act(self, state: TradingState) -> None:
        if any(s not in state.order_depths for s in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"]):
            return

        # ----------------------
        # Step 1: Arbitrage logic
        # ----------------------
        croissant = self.get_mid_price(state, "CROISSANTS")
        jam = self.get_mid_price(state, "JAMS")
        djembe = self.get_mid_price(state, "DJEMBES")
        basket1 = self.get_mid_price(state, "PICNIC_BASKET1")

        diff1 = basket1 - 6 * croissant - 3 * jam - djembe
        long_threshold, short_threshold = {
            "CROISSANTS": (0, 20),
            "JAMS": (-22, 50),
            "DJEMBES": (-10, 35),
            "PICNIC_BASKET1": (-10, 70),
        }[self.symbol]

        if diff1 < long_threshold and self.cooldown_long == 0:
            self.go_long(state)
            self.cooldown_long = self.cooldown_ticks
        elif diff1 > short_threshold and self.cooldown_short == 0:
            self.go_short(state)
            self.cooldown_short = self.cooldown_ticks

        self.cooldown_long = max(0, self.cooldown_long - 1)
        self.cooldown_short = max(0, self.cooldown_short - 1)

        # ----------------------
        # Step 2: Market Making
        # ----------------------
        fair_value = 6 * croissant + 3 * jam + djembe
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        buy_price = int(fair_value - self.spread + 10)
        sell_price = int(fair_value + self.spread + 10)

        buy_volume = min(self.limit - position, 2)
        sell_volume = min(self.limit + position, 2)

        if buy_volume > 0 and self.cooldown_long == 0:
            self.buy(buy_price, buy_volume)

        if sell_volume > 0 and self.cooldown_short == 0:
            self.sell(sell_price, sell_volume)

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.sell_orders.keys())
        position = state.position.get(self.symbol, 0)
        if(position >= self.limit*0.8):
            return
        to_buy = int((self.limit*0.7 - position))
        if to_buy > 0:
            self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())
        position = state.position.get(self.symbol, 0)
        if(position <= -self.limit*0.8):
            return
        to_sell = int((self.limit*0.7 + position))
        if to_sell > 0:
            self.sell(price, to_sell)

class PicnicBasket2Strategy(Strategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.components = {"CROISSANTS": 4, "JAMS": 2}
        self.spread = 30  # Market making spread
        self.cooldown_ticks = 0
        self.cooldown_long = 0
        self.cooldown_short = 0

    def act(self, state: TradingState) -> None:
        if any(s not in state.order_depths for s in ["CROISSANTS", "JAMS", "PICNIC_BASKET2"]):
            return

        # ----------------------
        # Step 1: Arbitrage logic
        # ----------------------
        croissant = self.get_mid_price(state, "CROISSANTS")
        jam = self.get_mid_price(state, "JAMS")
        basket2 = self.get_mid_price(state, "PICNIC_BASKET2")

        diff2 = basket2 - 4 * croissant - 2 * jam
        long_threshold, short_threshold = (-47, 40)

        if diff2 < long_threshold and self.cooldown_long == 0:
            self.go_long(state)
            self.cooldown_long = self.cooldown_ticks
        elif diff2 > short_threshold and self.cooldown_short == 0:
            self.go_short(state)
            self.cooldown_short = self.cooldown_ticks

        self.cooldown_long = max(0, self.cooldown_long - 1)
        self.cooldown_short = max(0, self.cooldown_short - 1)

        # ----------------------
        # Step 2: Market Making
        # ----------------------
        fair_value = 4 * croissant + 2 * jam
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        buy_price = int(fair_value - self.spread + 10)
        sell_price = int(fair_value + self.spread + 10)

        buy_volume = min(self.limit - position, 2)
        sell_volume = min(self.limit + position, 2)

        if buy_volume > 0 and self.cooldown_long == 0:
            self.buy(buy_price, buy_volume)

        if sell_volume > 0 and self.cooldown_short == 0:
            self.sell(sell_price, sell_volume)

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
        return (popular_buy_price + popular_sell_price) / 2

    def go_long(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = max(order_depth.sell_orders.keys())
        position = state.position.get(self.symbol, 0)
        if position >= self.limit * 0.7:
            return
        to_buy = int((self.limit * 0.7 - position))
        if to_buy > 0:
            self.buy(price, to_buy)

    def go_short(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        price = min(order_depth.buy_orders.keys())
        position = state.position.get(self.symbol, 0)
        if position <= -self.limit * 0.7:
            return
        to_sell = int((self.limit * 0.7 + position))
        if to_sell > 0:
            self.sell(price, to_sell)

class MacronStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.lower_threshold = 43
        self.higher_threshold = 44  # criticalSunlightIndex threshold
        self.fair_a = 3.4464
        self.fair_b = -51.3392
        self.previous_sunlightIndex = None
        self.window = deque()
        self.window_size = 10
        self.soft_position_limit = 0.3
        self.price_alt = 2
        self.entered=False

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
    
        # Safety check for order book
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        # Dynamic volume filter based on order book averages
        avg_sell_volume = sum(abs(order_depth.sell_orders[price]) for price in order_depth.sell_orders) / len(order_depth.sell_orders)
        avg_buy_volume = sum(abs(order_depth.buy_orders[price]) for price in order_depth.buy_orders) / len(order_depth.buy_orders)
        vol = math.floor(min(avg_sell_volume, avg_buy_volume))

        if (
            len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= vol]) == 0
            or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= vol]) == 0
        ):
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return round(mid_price)
        else:
            best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= vol])
            best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= vol])
            mid_price = (best_ask + best_bid) / 2
            return round(mid_price)

    def get_fair_value(self, observation: ConversionObservation) -> float:
        return self.fair_a * observation.sugarPrice + self.fair_b

    def act(self, state: TradingState) -> None:
        obs = state.observations.conversionObservations[self.symbol]
        order_depth = state.order_depths[self.symbol]
        position = state.position.get(self.symbol, 0)
        csi = obs.sunlightIndex
        true_value = self.get_fair_value(obs)
        # true_value=self.get_true_value(state)
        self.price_alt = max(1, int(0.05 * true_value))

        # First time — no previous value to compare
        if self.previous_sunlightIndex is None:
            self.previous_sunlightIndex = csi
            return
        
        if csi>=43 and self.entered:
            to_buy = self.limit - position
            sell_orders = sorted(order_depth.sell_orders.items()) if order_depth.sell_orders else []
            for price, volume in sell_orders:
                if to_buy > 0:
                    quantity = min(to_buy, -volume)
                    self.buy(price, quantity)
                    to_buy -= quantity
            if to_buy==0:self.entered=False
        elif (csi >= self.higher_threshold and csi<=self.previous_sunlightIndex) or (csi > self.lower_threshold and csi > self.previous_sunlightIndex):
            # Above threshold: regression-based trading
            self.window.append(abs(position) == self.limit)
            if len(self.window) > self.window_size:
                self.window.popleft()

            soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
            hard_liquidate = len(self.window) == self.window_size and all(self.window)

            max_buy_price = true_value - self.price_alt if position > self.limit * self.soft_position_limit else true_value
            min_sell_price = true_value + self.price_alt if position < self.limit * -self.soft_position_limit else true_value

            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True) if order_depth.buy_orders else []
            sell_orders=sorted(order_depth.sell_orders.items()) if order_depth.sell_orders else []

            to_buy=self.limit-position
            to_sell=self.limit+position

            for price, volume in sell_orders:
                if to_buy > 0 and price <= max_buy_price:
                    quantity = min(to_buy, -volume)
                    self.buy(price, quantity)
                    to_buy -= quantity

            if to_buy > 0 and sell_orders:
                popular_price = max(sell_orders, key=lambda tup: tup[1])[0]
                if popular_price <= max_buy_price:
                    self.buy(popular_price, min(to_buy, 4))
                    to_buy -= min(to_buy, 4)

            if to_buy > 0 and hard_liquidate:
                quantity = to_buy // 2
                self.buy(true_value, quantity)
                to_buy -= quantity

            if to_buy > 0 and soft_liquidate:
                quantity = to_buy // 2
                self.buy(true_value - 2, quantity)
                to_buy -= quantity

            if to_buy > 0 and buy_orders:
                popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                price = min(max_buy_price, popular_buy_price + 1)
                self.buy(price, to_buy)

            # if to_buy > 0:
            #     self.buy(true_value + 1, to_buy)

            for price, volume in buy_orders:
                if to_sell > 0 and price >= min_sell_price:
                    quantity = min(to_sell, volume)
                    self.sell(price, quantity)
                    to_sell -= quantity

            if to_sell > 0 and buy_orders:
                popular_price = max(buy_orders, key=lambda tup: tup[1])[0]
                if popular_price >= min_sell_price:
                    self.sell(popular_price, min(to_sell, 4))
                    to_sell -= min(to_sell, 4)

            if to_sell > 0 and hard_liquidate:
                quantity = to_sell // 2
                self.sell(true_value, quantity)
                to_sell -= quantity

            if to_sell > 0 and soft_liquidate:
                quantity = to_sell // 2
                self.sell(true_value + 2, quantity)
                to_sell -= quantity

            if to_sell > 0 and sell_orders:
                popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
                price = max(min_sell_price, popular_sell_price - 1)
                self.sell(price, to_sell)

            # if to_sell > 0:
            #     self.sell(true_value - 1, to_sell)
        else:
            self.entered=True
            # Below threshold: track sunlight trend
            if self.previous_sunlightIndex > csi:
                # Sunlight dropping → BUY aggressively
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders)
                    buy_qty = self.limit - position
                    if buy_qty > 0:
                        self.buy(best_ask, buy_qty)
            elif self.previous_sunlightIndex < csi:
                # Sunlight rising → SELL if price high
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders)
                    sell_qty = self.limit + position
                    if sell_qty > 0:
                        self.sell(best_bid, sell_qty)

        # Update history
        self.previous_sunlightIndex = csi
        best_bid= max(order_depth.buy_orders) if order_depth.buy_orders else None
        ask_bid1_diff = -(obs.askPrice + obs.transportFees + obs.importTariff - best_bid)
        if ask_bid1_diff < 0:
            return min(10,max(position,0))
        return 0
    
    def run(self, state: TradingState,*args) -> None:
        self.orders = []
        conversions=self.act(state)
        return self.orders,conversions

    def save(self):
        return {
            "previous_sunlightIndex": self.previous_sunlightIndex,
            "window": list(self.window),
        }

    def load(self, data, *args) -> None:
        self.args = args
        self.previous_sunlightIndex = data.get("previous_sunlightIndex", None)
        self.window = deque(data.get("window", []), maxlen=self.window_size)

def get_tte(day: int, timestamp: int) -> float:
    """
    Compute time-to-expiry (TTE) in years.
    
    For day = 3:
      days_to_expiry = 8 - 3 - (timestamp/1_000_000)
                    = 5 - (timestamp/1_000_000)
    At timestamp = 0, TTE = 5/365 years; at timestamp = 1,000,000, TTE = 4/365 years.
    """
    days_to_expiry = 8 - day - (timestamp / 1_000_000)
    return days_to_expiry / 365.0

class IVMomentumEmaStrategy(Strategy):
    """
    Detects implied volatility rises/falls using an EMA(15) of IV and
    trades accordingly. Holds each position for HOLD_PERIOD timestamps.
    """

    def __init__(self,
                 symbol: str,
                 position_limit: int,
                 underlying_symbol: str = "VOLCANIC_ROCK",
                 iv_threshold: float = 0.02,  # e.g. ±2% change triggers a trade
                 hold_period: int = 10_000,   # hold for 10,000 timestamps
                 trade_size: int = 10,
                 ema_window: int = 15):
        """
        :param symbol: The option symbol to trade (e.g. "VOLCANIC_ROCK_VOUCHER_10000").
        :param position_limit: Max net position allowed for this symbol.
        :param underlying_symbol: Name of the underlying asset.
        :param iv_threshold: The absolute % change in the EMA of IV that triggers trades.
        :param hold_period: Number of timestamps we hold a position once opened.
        :param trade_size: Number of contracts to buy/sell when opening a position.
        :param ema_window: Window size for the IV EMA (e.g. 15).
        """
        super().__init__(symbol, position_limit)
        self.underlying_symbol = underlying_symbol

        # Trigger thresholds for percent changes in EMA
        self.iv_threshold = iv_threshold

        # How long to hold once a trade is opened
        self.HOLD_PERIOD = hold_period

        # How many contracts to trade each time we enter
        self.trade_size = trade_size

        # EMA window for smoothing IV
        self.EMA_WINDOW = ema_window
        self.alpha = 2.0 / (self.EMA_WINDOW + 1.0)

        # Current EMA of IV (starts empty)
        self.iv_ema: Optional[float] = None

        # Position side: +1 = long, -1 = short, 0 = flat
        self.position_side: int = 0

        # Timestamp when position was opened
        self.entry_timestamp: Optional[int] = None

    def act(self, state: TradingState) -> None:
        self.orders = []

        # --- Sanity checks on order book data ---
        if self.symbol not in state.order_depths:
            return
        opt_depth = state.order_depths[self.symbol]
        if not opt_depth.buy_orders or not opt_depth.sell_orders:
            return

        # Underlying check
        if self.underlying_symbol not in state.order_depths:
            return
        und_depth = state.order_depths[self.underlying_symbol]
        if not und_depth.buy_orders or not und_depth.sell_orders:
            return

        # --- Compute option & underlying mid-prices ---
        best_bid_opt = max(opt_depth.buy_orders.keys())
        best_ask_opt = min(opt_depth.sell_orders.keys())
        option_mid_price = (best_bid_opt + best_ask_opt) / 2.0

        best_bid_under = max(und_depth.buy_orders.keys())
        best_ask_under = min(und_depth.sell_orders.keys())
        S = (best_bid_under + best_ask_under) / 2.0

        # --- Time to expiry ---
        T = self.get_time_to_expiry(state.timestamp)
        if T <= 0:
            return

        # --- Compute implied volatility from market option mid-price ---
        iv_calc = self.implied_vol(option_mid_price, S, T)
        if iv_calc is None:
            return

        # --- Update EMA(15) of IV ---
        if self.iv_ema is None:
            # First time: just initialize it.
            self.iv_ema = iv_calc
            return
        else:
            old_ema = self.iv_ema
            self.iv_ema = self.alpha * iv_calc + (1.0 - self.alpha) * self.iv_ema
            # Percent change (IV returns) from old EMA to new EMA
            if old_ema != 0:
                iv_return = (self.iv_ema - old_ema) / abs(old_ema)
            else:
                iv_return = 0.0

        # --- If we have an open position, check hold duration ---
        current_position = state.position.get(self.symbol, 0)
        if self.position_side != 0 and self.entry_timestamp is not None:
            if (state.timestamp - self.entry_timestamp) >= self.HOLD_PERIOD:
                # Time to exit
                if self.position_side > 0:   # we are long => sell
                    self.sell(best_bid_opt, abs(current_position))
                else:                       # we are short => buy
                    self.buy(best_ask_opt, abs(current_position))
                # Reset to flat
                self.position_side = 0
                self.entry_timestamp = None
            return  # no further action if we already have a position

        # --- If we are flat, check if IV return triggers a trade ---
        if self.position_side == 0:
            # If IV return is above +threshold => buy
            if iv_return > self.iv_threshold:
                # Check capacity
                if current_position < self.limit:
                    qty = min(self.trade_size, self.limit - current_position)
                    best_ask_volume = abs(opt_depth.sell_orders[best_ask_opt])
                    qty = min(qty, best_ask_volume)
                    if qty > 0:
                        self.buy(best_ask_opt, qty)
                        self.entry_timestamp = state.timestamp
                        self.position_side = +1

            # If IV return is below -threshold => sell
            elif iv_return < -self.iv_threshold:
                if current_position > -self.limit:
                    qty = min(self.trade_size, self.limit + current_position)
                    best_bid_volume = abs(opt_depth.buy_orders[best_bid_opt])
                    qty = min(qty, best_bid_volume)
                    if qty > 0:
                        self.sell(best_bid_opt, qty)
                        self.entry_timestamp = state.timestamp
                        self.position_side = -1

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    # --------------------------------------------------------------------------------
    # Helpers: get_time_to_expiry, implied_vol, plus simple black-scholes / brentq
    # --------------------------------------------------------------------------------
    def get_time_to_expiry(self, timestamp: int) -> float:
        """
        Assume 1,000,000 timestamps = 1 day, fixed 5 days to expiry.
        Adjust to your environment as needed.
        """
        days_passed = timestamp / 1_000_000.0
        days_remaining = max(0, 5.0 - days_passed)
        return days_remaining / 365.0

    def bs_call_price(self, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
        """
        Simple Black–Scholes call formula with zero dividend yield.
        """
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

    def implied_vol(self, option_price: float, S: float, T: float,
                    K: float = 10000, r: float = 0.0) -> Optional[float]:
        """
        Approximate implied volatility via numeric root-finding, fixing strike=10000
        for this example. Adjust as needed for your environment.
        """
        intrinsic = max(S - K, 0)
        if option_price < intrinsic:
            return None

        def objective(sig: float) -> float:
            return self.bs_call_price(S, K, T, sig, r) - option_price

        try:
            vol = brentq(objective, 1e-6, 5.0, xtol=1e-6)
            return vol
        except:
            return None

class VolcanicOptionVolatilityStrategy(Strategy):
    """
    Trading strategy using a dynamically updated quadratic vol smile 
    from the global_vol_smile instance.
    Note: It no longer updates the global smile by itself.
    """
    def __init__(self,
                 symbol: str,
                 strike: int,
                 position_limit: int,
                 underlying_symbol: str = "VOLCANIC_ROCK",
                 entry_threshold: float = 5.0,
                 exit_threshold: float = 2.0,
                 expiration_days: float = 4.0,
                 trading_days_per_year: int = 365,
                 interest_rate: float = 0.0,
                 debug: bool = False):
        super().__init__(symbol, position_limit)
        self.strike = strike
        self.underlying_symbol = underlying_symbol
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.expiration_days = expiration_days
        self.trading_days_per_year = trading_days_per_year
        self.interest_rate = interest_rate
        self.debug = debug

        self.last_iv = None
        self.last_theoretical_price = None
        self.last_market_price = None
        self.last_moneyness = None
        self.last_poly_fit = None

    def get_time_to_expiry(self, timestamp: int) -> float:
        days_passed = timestamp / 1_000_000
        days_remaining = max(0, self.expiration_days - days_passed)
        return days_remaining / self.trading_days_per_year

    def bs_call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + (self.interest_rate + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm.cdf(d1) - K * math.exp(-self.interest_rate * T) * norm.cdf(d2)

    def implied_volatility(self, option_price: float, S: float, K: float, T: float) -> Optional[float]:
        intrinsic = max(S - K, 0)
        if option_price < intrinsic:
            return None
        def objective(sig: float):
            return self.bs_call_price(S, K, T, sig) - option_price
        try:
            vol = brentq(objective, 1e-6, 5.0, xtol=1e-6)
            return vol
        except:
            return None

    def predict_iv(self, m: float, smile: Tuple[float, float, float]) -> float:
        a, b, c = smile
        iv = a*(m**2) + b*m + c
        return max(iv, 1e-6)

    def act(self, state: TradingState) -> None:
        self.orders = []
        if (self.underlying_symbol not in state.order_depths or
            self.symbol not in state.order_depths):
            return

        underlying_depth = state.order_depths[self.underlying_symbol]
        option_depth = state.order_depths[self.symbol]
        if (not underlying_depth.buy_orders or not underlying_depth.sell_orders or
            not option_depth.buy_orders or not option_depth.sell_orders):
            return

        best_bid_under = max(underlying_depth.buy_orders)
        best_ask_under = min(underlying_depth.sell_orders)
        S = (best_bid_under + best_ask_under) / 2

        best_bid_opt = max(option_depth.buy_orders)
        best_ask_opt = min(option_depth.sell_orders)
        option_price = (best_bid_opt + best_ask_opt) / 2

        T = self.get_time_to_expiry(state.timestamp)
        if T <= 0:
            return

        iv_calc = self.implied_volatility(option_price, S, self.strike, T)
        if iv_calc is None:
            return

        m = math.log(S / self.strike) / math.sqrt(T)

        # DO NOT update the global smile here.
        # Instead, assume that Trader.run has already updated the global vol smile.

        smile = global_vol_smile.fit_smile()
        if smile is None:
            return

        self.last_poly_fit = smile

        iv_from_smile = self.predict_iv(m, smile)
        self.last_iv = iv_from_smile
        self.last_moneyness = m
        # print(f"IV CALC: {iv_calc}, IV from smile: {iv_from_smile}")

        theoretical_price = self.bs_call_price(S, self.strike, T, iv_from_smile)
        self.last_theoretical_price = theoretical_price
        self.last_market_price = option_price

        price_diff = option_price - theoretical_price
        # print(f"Price Difference: {price_diff}")
        current_position = state.position.get(self.symbol, 0)

        if abs(price_diff) > self.entry_threshold:
            if price_diff > 0:
                qty_to_sell = self.limit + current_position
                qty_to_sell = min(qty_to_sell, abs(option_depth.buy_orders[best_bid_opt]))
                if qty_to_sell > 0:
                    self.orders.append(Order(self.symbol, best_bid_opt, -qty_to_sell))
            else:
                qty_to_buy = self.limit - current_position
                qty_to_buy = min(qty_to_buy, abs(option_depth.sell_orders[best_ask_opt]))
                if qty_to_buy > 0:
                    self.orders.append(Order(self.symbol, best_ask_opt, qty_to_buy))
        elif current_position != 0 and abs(price_diff) < self.exit_threshold:
            if current_position > 0:
                self.orders.append(Order(self.symbol, best_bid_opt, -current_position))
            else:
                self.orders.append(Order(self.symbol, best_ask_opt, -current_position))

        if self.debug and self.orders:
            logger.print(
                f"[{self.symbol}] M={m:.3f}, IVfit={iv_from_smile:.3f}, IVcalc={iv_calc:.3f}, "
                f"Theor={theoretical_price:.2f}, Mkt={option_price:.2f}, Diff={price_diff:.2f}, "
                f"Pos={current_position}, Orders={self.orders}"
            )

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def save(self) -> dict:
        return {
            "last_poly_fit": self.last_poly_fit,
            "last_iv": self.last_iv,
            "last_moneyness": self.last_moneyness,
            "last_theoretical_price": self.last_theoretical_price,
            "last_market_price": self.last_market_price
        }

    def load(self, data: dict) -> None:
        if not data:
            return
        if "last_poly_fit" in data:
            self.last_poly_fit = tuple(data["last_poly_fit"]) if data["last_poly_fit"] else None
        self.last_iv = data.get("last_iv")
        self.last_moneyness = data.get("last_moneyness")
        self.last_theoretical_price = data.get("last_theoretical_price")
        self.last_market_price = data.get("last_market_price")

class VolSmSpreadArbStrategyDynamic(Strategy):
    def __init__(self, asset_a: str, asset_b: str, trade_qty: int, 
                 spread_threshold: int, exit_threshold: int) -> None:
        super().__init__("", 0)
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.trade_qty = trade_qty
        self.spread_threshold = spread_threshold  # in IV points
        self.exit_threshold = exit_threshold      # in IV points
        self.underlying_symbol = "VOLCANIC_ROCK"
        # Use a lookup so that each asset can have its own strike.
        self.strikes: Dict[str, int] = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        self.a = 0.237212
        self.b = 0.00371769
        self.c = 0.149142

    def get_time_to_expiry(self, timestamp: int) -> float:
        # Assume 1,000,000 ticks = 1 day and a fixed expiration of 5 days.
        days_passed = timestamp / 1_000_000.0
        days_remaining = max(0, 5.0 - days_passed)
        return days_remaining / 365

    def bs_call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + (0.0 + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm.cdf(d1) - K * math.exp(0.0 * T) * norm.cdf(d2)

    def implied_volatility(self, option_price: float, S: float, K: float, T: float) -> Optional[float]:
        intrinsic = max(S - K, 0)
        if option_price < intrinsic:
            return None
        def objective(sig: float):
            return self.bs_call_price(S, K, T, sig) - option_price
        try:
            vol = brentq(objective, 1e-6, 5.0, xtol=1e-6)
            return vol
        except Exception:
            return None

    def predict_iv_for_asset(self, state: 'TradingState', asset: str, strike: int) -> Optional[float]:
        """
        Compute the predicted (theoretical) IV for a given asset
        using the underlying mid-price, the asset's strike, and the global quadratic fit.
        """
        if self.underlying_symbol not in state.order_depths or asset not in state.order_depths:
            return None
        underlying_depth = state.order_depths[self.underlying_symbol]
        option_depth = state.order_depths[asset]
        if (not underlying_depth.buy_orders or not underlying_depth.sell_orders or
            not option_depth.buy_orders or not option_depth.sell_orders):
            return None
        best_bid_under = max(underlying_depth.buy_orders.keys())
        best_ask_under = min(underlying_depth.sell_orders.keys())
        S = (best_bid_under + best_ask_under) / 2.0
        T = self.get_time_to_expiry(state.timestamp)
        if T <= 0:
            return None
        m = math.log(S / strike) / math.sqrt(T)
        # fit = global_vol_smile.fit_smile()
        # if fit is None:
        #     return None
        # a, b, c = fit
        predicted_iv = self.a * (m ** 2) + self.b * m + self.c
        return predicted_iv

    def get_market_iv_for_asset(self, state: 'TradingState', asset: str, strike: int) -> Optional[float]:
        """
        Compute the market implied volatility for a given asset using its market price.
        """
        option_depth = state.order_depths.get(asset)
        if option_depth is None or not option_depth.buy_orders or not option_depth.sell_orders:
            return None
        # Compute mid-price for the option.
        best_bid = max(option_depth.buy_orders.keys())
        best_ask = min(option_depth.sell_orders.keys())
        option_price = (best_bid + best_ask) / 2.0
        # Get underlying mid-price and time-to-expiry.
        underlying_depth = state.order_depths.get(self.underlying_symbol)
        if underlying_depth is None or not underlying_depth.buy_orders or not underlying_depth.sell_orders:
            return None
        best_bid_under = max(underlying_depth.buy_orders.keys())
        best_ask_under = min(underlying_depth.sell_orders.keys())
        S = (best_bid_under + best_ask_under) / 2.0
        T = self.get_time_to_expiry(state.timestamp)
        if T <= 0:
            return None
        market_iv = self.implied_volatility(option_price, S, strike, T)
        return market_iv

    def get_best_bid_with_vol(self, state: 'TradingState', symbol: str) -> Optional[Tuple[float, int]]:
        order_depth = state.order_depths.get(symbol)
        if order_depth and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            volume = order_depth.buy_orders[best_bid]
            return best_bid, volume
        return None

    def get_best_ask_with_vol(self, state: 'TradingState', symbol: str) -> Optional[Tuple[float, int]]:
        order_depth = state.order_depths.get(symbol)
        if order_depth and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            volume = abs(order_depth.sell_orders[best_ask])
            return best_ask, volume
        return None

    def get_order_for_asset(self, state: 'TradingState', symbol: str) -> Optional[Order]:
        # For each asset, compute market IV and predicted (theoretical) IV.
        market_iv_a = self.get_market_iv_for_asset(state, self.asset_a, self.strikes[self.asset_a])
        market_iv_b = self.get_market_iv_for_asset(state, self.asset_b, self.strikes[self.asset_b])
        if market_iv_a is None or market_iv_b is None:
            return None
        predicted_iv_a = self.predict_iv_for_asset(state, self.asset_a, self.strikes[self.asset_a])
        predicted_iv_b = self.predict_iv_for_asset(state, self.asset_b, self.strikes[self.asset_b])
        if predicted_iv_a is None or predicted_iv_b is None:
            return None

        # Compute volatility spreads.
        market_vol_spread = market_iv_a - market_iv_b
        theoretical_vol_spread = predicted_iv_a - predicted_iv_b
        diff = market_vol_spread - theoretical_vol_spread
        # print(f"Dynamic Spread Arb: market_iv({self.asset_a})={market_iv_a:.4f}, "
        #       f"market_iv({self.asset_b})={market_iv_b:.4f}, "
        #       f"predicted_iv({self.asset_a})={predicted_iv_a:.4f}, "
        #       f"predicted_iv({self.asset_b})={predicted_iv_b:.4f}, "
        #       f"market_vol_spread={market_vol_spread:.4f}, theoretical_vol_spread={theoretical_vol_spread:.4f}, diff={diff:.4f}")
        
        # Get effective volumes (using options order depths from one side—example uses asset A's bid and asset B's ask).
        bid_tuple_a = self.get_best_bid_with_vol(state, self.asset_a)
        ask_tuple_b = self.get_best_ask_with_vol(state, self.asset_b)
        if bid_tuple_a is None or ask_tuple_b is None:
            return None
        best_bid_a, vol_a = bid_tuple_a
        best_ask_b, vol_b = ask_tuple_b
        effective_qty = min(self.trade_qty, vol_a, vol_b)
        if effective_qty <= 0:
            return None

        # Entry logic: if the market IV spread is wider than theoretical by the entry threshold.
        if diff > self.spread_threshold:
            if symbol == self.asset_a:
                # Sell asset A (typically high IV) at a price slightly below best bid.
                return Order(self.asset_a, int(round(best_bid_a)) - 1, -effective_qty)
            elif symbol == self.asset_b:
                # Buy asset B (low IV) at a price slightly above best ask.
                return Order(self.asset_b, int(round(best_ask_b)) + 1, effective_qty)
        elif diff < -self.spread_threshold:
            if symbol == self.asset_a:
                # Buy asset A at a price slightly above best bid.
                return Order(self.asset_a, int(round(best_bid_a)) + 1, effective_qty)
            elif symbol == self.asset_b:
                # Sell asset B at a price slightly below best ask.
                return Order(self.asset_b, int(round(best_ask_b)) - 1, -effective_qty)
        # Exit logic: if the difference is now within the exit threshold, flatten any position.
        elif abs(diff) < self.exit_threshold:
            current_pos = state.position.get(symbol, 0)
            if current_pos != 0:
                side_price = int(round(best_bid_a)) if symbol == self.asset_a else int(round(best_ask_b))
                return Order(symbol, side_price, -current_pos)
        return None

    def act(self, state: 'TradingState') -> None:
        orders: List[Order] = []
        order_a = self.get_order_for_asset(state, self.asset_a)
        order_b = self.get_order_for_asset(state, self.asset_b)
        if order_a is not None and order_b is not None:
            orders.append(order_a)
            orders.append(order_b)
        self.orders = orders

    def run(self, state: 'TradingState') -> List[Order]:
        self.act(state)
        return self.orders

    def save(self) -> dict:
        return None

    def load(self, data: dict) -> None:
        pass

class VolatilityHedger:
    """
    Hedges option positions using delta-neutral hedging.
    Instead of updating its own vol smile, it uses the global_vol_smile (updated once per tick).
    """
    def __init__(self,
                 underlying_symbol: str = "VOLCANIC_ROCK",
                 option_symbols: List[str] = None,
                 strikes: List[int] = None,
                 expiration_days: float = 5,
                 trading_days_per_year: int = 365,
                 interest_rate: float = 0.0,
                 hedge_threshold: float = 5.0,
                 max_hedge_size: int = 400,
                 debug: bool = False):
        self.underlying_symbol = underlying_symbol

        if option_symbols is None:
            option_symbols = [
                "VOLCANIC_ROCK_VOUCHER_9500",
                "VOLCANIC_ROCK_VOUCHER_9750",
                "VOLCANIC_ROCK_VOUCHER_10000",
                "VOLCANIC_ROCK_VOUCHER_10250",
                "VOLCANIC_ROCK_VOUCHER_10500"
            ]
        if strikes is None:
            strikes = [9500, 9750, 10000, 10250, 10500]

        self.option_symbols = option_symbols
        self.strikes = {sym: strike for sym, strike in zip(option_symbols, strikes)}
        self.expiration_days = expiration_days
        self.trading_days_per_year = trading_days_per_year
        self.interest_rate = interest_rate
        self.hedge_threshold = hedge_threshold
        self.max_hedge_size = max_hedge_size
        self.debug = debug

        self.last_hedge_delta = 0
        self.orders: List[Order] = []

    def get_time_to_expiry(self, timestamp: int) -> float:
        days_passed = timestamp / 1_000_000.0
        days_remaining = max(0, self.expiration_days - days_passed)
        return days_remaining / self.trading_days_per_year

    def bs_call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + (self.interest_rate + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return norm.cdf(d1)

    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        if self.underlying_symbol not in state.order_depths:
            return self.orders
        underlying_depth = state.order_depths[self.underlying_symbol]
        if not underlying_depth.buy_orders or not underlying_depth.sell_orders:
            return self.orders

        best_bid_under = max(underlying_depth.buy_orders.keys())
        best_ask_under = min(underlying_depth.sell_orders.keys())
        underlying_price = (best_bid_under + best_ask_under) / 2

        tte = self.get_time_to_expiry(state.timestamp)
        if tte <= 0:
            return self.orders

        total_delta = 0.0
        for sym in self.option_symbols:
            pos = state.position.get(sym, 0)
            if pos == 0:
                continue
            K = self.strikes[sym]
            # Use the global vol smile (assumed to be updated once per timestamp)
            # fit = global_vol_smile.fit_smile()
            # if fit is None:
            #     continue
            a, b, c = 0.237212, 0.00371769, 0.149142#fit
            m = math.log(K / underlying_price) / math.sqrt(tte)
            iv = a*(m**2) + b*m + c
            iv = max(iv, 1e-6)
            d = self.bs_call_delta(underlying_price, K, tte, iv)
            total_delta += pos * d

            if self.debug:
                logger.print(f"[{sym}] pos={pos}, strike={K}, iv={iv:.3f}, delta={d:.3f}, contrib={pos * d:.3f}")

        target_hedge = -round(total_delta)
        target_hedge = max(-self.max_hedge_size, min(self.max_hedge_size, target_hedge))
        current_hedge = state.position.get(self.underlying_symbol, 0)

        if abs(target_hedge - current_hedge) >= self.hedge_threshold:
            order_qty = target_hedge - current_hedge
            if order_qty > 0:
                self.orders.append(Order(self.underlying_symbol, best_ask_under, order_qty))
            elif order_qty < 0:
                self.orders.append(Order(self.underlying_symbol, best_bid_under, order_qty))
            if self.debug:
                logger.print(f"[VOL HEDGER] total_delta={total_delta:.3f}, target_hedge={target_hedge}, "
                             f"current_hedge={current_hedge}, order_qty={order_qty}")
        return self.orders

    def save(self) -> dict:
        return {"last_hedge_delta": self.last_hedge_delta}

    def load(self, data: dict) -> None:
        if data and "last_hedge_delta" in data:
            self.last_hedge_delta = data["last_hedge_delta"]

class GlobalVolSmile:
    """
    Maintains a rolling window of (moneyness, implied volatility) points
    in a single deque.
    """
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.smile_points: deque = deque(maxlen=window_size)

    def update(self, m: float, iv: float) -> None:
        """Append the (m, iv) point (no labels)."""
        self.smile_points.append((m, iv))

    def fit_smile(self) -> Optional[Tuple[float, float, float]]:
        """
        Fit a quadratic: iv = a*m^2 + b*m + c using the stored points.
        Returns (a, b, c) if enough points are available, else None.
        """
        if len(self.smile_points) < 5:
            return None
        pts = list(self.smile_points)
        m_vals = np.array([p[0] for p in pts])
        iv_vals = np.array([p[1] for p in pts])
        coeffs = np.polyfit(m_vals, iv_vals, 2)
        return coeffs[0], coeffs[1], coeffs[2]

    def get_base_iv(self) -> Optional[float]:
        """
        Returns the quadratic intercept (base IV) from the fitted smile.
        """
        fit = self.fit_smile()
        if fit is None:
            return None
        _, _, c = fit
        return c

class VolcanicRockStrategy:  #best till now
    def __init__(self):
        self.product = "VOLCANIC_ROCK"
        self.voucher_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
        self.voucher_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
        self.history = deque([
            10217.5, 10216.5, 10217.0, 10216.0, 10215.5,
            10216.0, 10218.5, 10219.0, 10218.0, 10217.0,
            10216.5, 10217.5, 10218.0, 10217.0, 10216.5,
            10217.0, 10216.0, 10215.0, 10215.5, 10216.5,
            10217.5, 10218.5, 10218.0, 10217.0, 10216.5,
            10216.0, 10215.5, 10216.0, 10217.0, 10218.0,
            10218.5, 10219.5, 10219.0, 10218.0, 10217.5,
            10217.0, 10216.5, 10216.0, 10216.5, 10217.5,
            10218.0, 10218.5, 10218.0, 10217.5, 10217.0,
            10216.0, 10215.5, 10215.0, 10214.5, 10215.5
        ], maxlen=50)
        self.z = 0

    def run(self, state: TradingState) -> List[Order]:
        orders = []
        rock_depth = state.order_depths.get(self.product)

        if rock_depth and rock_depth.buy_orders and rock_depth.sell_orders:
            rock_bid = max(rock_depth.buy_orders)
            rock_ask = min(rock_depth.sell_orders)
            rock_mid = (rock_bid + rock_ask) / 2
            self.history.append(rock_mid)

        if len(self.history) >= 50:
            recent = np.array(self.history)[-50:]
            mean = np.mean(recent)
            std = np.std(recent)
            self.z = (recent[-1] - mean) / std if std > 0 else 0

            threshold = 2.1
            position = state.position.get(self.product, 0)
            position_limit = 400

            # Buy signal
            if self.z < -threshold and rock_depth and rock_depth.sell_orders:
                best_ask = min(rock_depth.sell_orders)
                qty = -rock_depth.sell_orders[best_ask]
                buy_qty = min(qty, position_limit - position)
                if buy_qty > 0:
                    orders.append(Order(self.product, best_ask, buy_qty))

            # Sell signal
            elif self.z > threshold and rock_depth and rock_depth.buy_orders:
                best_bid = max(rock_depth.buy_orders)
                qty = rock_depth.buy_orders[best_bid]
                sell_qty = min(qty, position + position_limit)
                if sell_qty > 0:
                    orders.append(Order(self.product, best_bid, -sell_qty))

        # --- Delta Hedge ---
        # hedge_orders = self.hedge_with_vouchers(state)
        # orders += hedge_orders

        # orders.append()
        # for product in state.order_depths:
        #     if product == "VOLCANIC_ROCK_VOUCHER_9500":
        #         orders.append(self.trade_mean_reversion(state, product))
        #     elif product == "VOLCANIC_ROCK_VOUCHER_9750":
        #         orders.append(self.trade_mean_reversion(state, product))
        #     elif product == "VOLCANIC_ROCK_VOUCHER_10000":
        #         orders.append(self.trade_mean_reversion(state, product))
        #     elif product == "VOLCANIC_ROCK_VOUCHER_10250":
        #         orders.append(self.trade_mean_reversion(state, product))
        #     elif product == "VOLCANIC_ROCK_VOUCHER_10500":
        #         orders.append(self.trade_mean_reversion(state, product))

        return orders

    def trade_mean_reversion(self, state: TradingState, product: Symbol) -> List[Order]:
        orders = []
        position = state.position.get(product, 0)
        position_limit = self.limits[product]

        threshold = self.threshold[product]  # Reversion trigger

        product_depth = state.order_depths.get(product)

        if not product_depth:
            return orders
        z = getattr(self, "z", None)
        if z is None:
            return orders

        z=self.z
        # Z-score low → buy target product (expecting VOLCANIC_ROCK to rebound)
        if z < -threshold and product_depth.sell_orders:
            best_ask = min(product_depth.sell_orders)
            qty = -product_depth.sell_orders[best_ask]
            buy_qty = min(qty, position_limit - position)
            if buy_qty > 0:
                orders.append(Order(product, best_ask, buy_qty))

        # Z-score high → sell target product (expecting VOLCANIC_ROCK to fall)
        elif z > threshold and product_depth.buy_orders:
            best_bid = max(product_depth.buy_orders)
            qty = product_depth.buy_orders[best_bid]
            sell_qty = min(qty, position + position_limit)
            if sell_qty > 0:
                orders.append(Order(product, best_bid, -sell_qty))

        return orders

    def hedge_with_vouchers(self, state: TradingState) -> List[Order]:
        orders = []

        # Constants
        delta_rock = 1
        delta_10250 = 0.2
        delta_10500 = 0.1
        limit = 200

        # Positions
        pos_rock = state.position.get(self.product, 0)
        pos_10250 = state.position.get(self.voucher_10250, 0)
        pos_10500 = state.position.get(self.voucher_10500, 0)

        # Portfolio delta
        net_delta = (
            pos_rock * delta_rock +
            pos_10250 * delta_10250 +
            pos_10500 * delta_10500
        )

        # Want to reduce delta to ~0
        hedge_depths = [
            (self.voucher_10250, delta_10250, state.order_depths.get(self.voucher_10250), pos_10250),
            (self.voucher_10500, delta_10500, state.order_depths.get(self.voucher_10500), pos_10500)
        ]

        for symbol, delta, depth, pos in hedge_depths:
            if abs(net_delta) < 0.5 or not depth:
                continue

            max_hedge_qty = int(net_delta / delta)

            # Clamp to position limits
            max_hedge_qty = max(-limit - pos, min(limit - pos, max_hedge_qty))

            if max_hedge_qty > 0 and depth.sell_orders:
                best_ask = min(depth.sell_orders)
                ask_qty = -depth.sell_orders[best_ask]
                qty = min(max_hedge_qty, ask_qty)
                if qty > 0:
                    orders.append(Order(symbol, best_ask, qty))
                    net_delta -= qty * delta

            elif max_hedge_qty < 0 and depth.buy_orders:
                best_bid = max(depth.buy_orders)
                bid_qty = depth.buy_orders[best_bid]
                qty = min(-max_hedge_qty, bid_qty)
                if qty > 0:
                    orders.append(Order(symbol, best_bid, -qty))
                    net_delta += qty * delta

        return orders
    
    def save(self):
        pass

    def load(self, data):
        pass

global_vol_smile = GlobalVolSmile(window_size=500)

class Trader:
    def __init__(self) -> None:
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,   
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS":75
        }
        
        self.days_left = 8
        self.strike_price = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        }
        self.history = {
            "KELP": [],
            "SQUID_INK": [],
            "JAMS": [],
            "CROISSANTS": [],
            "DJEMBES": [],
            "VOLCANIC_ROCK": deque([
                10217.5, 10216.5, 10217.0, 10216.0, 10215.5,
                10216.0, 10218.5, 10219.0, 10218.0, 10217.0,
                10216.5, 10217.5, 10218.0, 10217.0, 10216.5,
                10217.0, 10216.0, 10215.0, 10215.5, 10216.5,
                10217.5, 10218.5, 10218.0, 10217.0, 10216.5,
                10216.0, 10215.5, 10216.0, 10217.0, 10218.0,
                10218.5, 10219.5, 10219.0, 10218.0, 10217.5,
                10217.0, 10216.5, 10216.0, 10216.5, 10217.5,
                10218.0, 10218.5, 10218.0, 10217.5, 10217.0,
                10216.0, 10215.5, 10215.0, 10214.5, 10215.5
                ],maxlen=50),
            "VOLCANIC_ROCK_VOUCHER_9500":[],
            "VOLCANIC_ROCK_VOUCHER_9750":[],
            "VOLCANIC_ROCK_VOUCHER_10000":[],
            "VOLCANIC_ROCK_VOUCHER_10250":[],
            "VOLCANIC_ROCK_VOUCHER_10500":[],
        }
        self.historical_data = {}

        self.strategies = {
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", self.limits["RAINFOREST_RESIN"], 10, 0.75, 1), #final
            "KELP": KelpStrategy("KELP", self.limits["KELP"]), #final
            "SQUID_INK": SquidInkStrategy("SQUID_INK", self.limits["SQUID_INK"]), #final
        
            # "CROISSANTS": TrendFollowingStrategy("CROISSANTS", self.limits["CROISSANTS"], 55, (7,3), -2, 0), #best
            # "DJEMBES": TrendFollowingStrategy("DJEMBES", self.limits["DJEMBES"], 100, 12.5, 0, 25), #best
            
            "PICNIC_BASKET1": PicnicBasket1Strategy("PICNIC_BASKET1", self.limits["PICNIC_BASKET1"]), #best
            "PICNIC_BASKET2": PicnicBasket2Strategy("PICNIC_BASKET2", self.limits["PICNIC_BASKET2"]), #best

            "VOLCANIC_ROCK": VolcanicRockStrategy() #best
        }

        self.threshold = {
            "VOLCANIC_ROCK_VOUCHER_9500": 2.2,
            "VOLCANIC_ROCK_VOUCHER_9750": 2.2,
            "VOLCANIC_ROCK_VOUCHER_10000": 2.2,
            "VOLCANIC_ROCK_VOUCHER_10250": 2.2,
            "VOLCANIC_ROCK_VOUCHER_10500": 2.2,
            "VOLCANIC_ROCK": 2.2,
        }

    def calculate_mean(self, values):
        return sum(values) / len(values)

    def calculate_std_dev(self, values, mean):
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def update_volcanic_rock_history(self, state: TradingState) -> List[Order]:
        orders = []
        # Update VOLCANIC_ROCK price history
        rock_depth = state.order_depths.get("VOLCANIC_ROCK")
        if rock_depth and rock_depth.buy_orders and rock_depth.sell_orders:
            rock_bid = max(rock_depth.buy_orders)
            rock_ask = min(rock_depth.sell_orders)
            rock_mid = (rock_bid + rock_ask) / 2
            self.history["VOLCANIC_ROCK"].append(rock_mid)

        rock_prices = np.array(self.history["VOLCANIC_ROCK"])

        # Only proceed if enough VOLCANIC_ROCK history
        if len(rock_prices) >= 50:
            recent = rock_prices[-50:]
            mean = np.mean(recent)
            std = np.std(recent)
            self.z = (rock_prices[-1] - mean) / std if std > 0 else 0
            
            threshold = self.threshold["VOLCANIC_ROCK"]
            position = state.position.get("VOLCANIC_ROCK", 0)
            position_limit = self.limits["VOLCANIC_ROCK"]
            product = "VOLCANIC_ROCK"

            # Z-score low → buy (expecting VOLCANIC_ROCK to rebound)
            if self.z < -threshold and rock_depth.sell_orders:
                best_ask = min(rock_depth.sell_orders)
                qty = -rock_depth.sell_orders[best_ask]
                buy_qty = min(qty, position_limit - position)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
            # Z-score high → sell (expecting VOLCANIC_ROCK to fall)
            elif self.z > threshold and rock_depth.buy_orders:
                best_bid = max(rock_depth.buy_orders)
                qty = rock_depth.buy_orders[best_bid]
                sell_qty = min(qty, position + position_limit)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
        
        # Always return orders (empty list if no conditions met or history too short)
        return orders
    
    def fair_value(self, state: TradingState, product: Symbol) -> int:
        if product == "RAINFOREST_RESIN":
            return 10000
        order_depth = state.order_depths[product]
        if order_depth.buy_orders and order_depth.sell_orders:
            return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) // 2
        return None
    #use mean reversion
    #Mean reversion
    def trade_mean_reversion(self, state: TradingState, product: Symbol) -> List[Order]:
        orders = []
        position = state.position.get(product, 0)
        position_limit = self.limits[product]

        threshold = self.threshold[product]  # Reversion trigger

        product_depth = state.order_depths.get(product)

        if not product_depth:
            return orders
        z = getattr(self, "z", None)
        if z is None:
            return orders

        z=self.z
        # Z-score low → buy target product (expecting VOLCANIC_ROCK to rebound)
        if z < -threshold and product_depth.sell_orders:
            best_ask = min(product_depth.sell_orders)
            qty = -product_depth.sell_orders[best_ask]
            buy_qty = min(qty, position_limit - position)
            if buy_qty > 0:
                orders.append(Order(product, best_ask, buy_qty))

        # Z-score high → sell target product (expecting VOLCANIC_ROCK to fall)
        elif z > threshold and product_depth.buy_orders:
            best_bid = max(product_depth.buy_orders)
            qty = product_depth.buy_orders[best_bid]
            sell_qty = min(qty, position + position_limit)
            if sell_qty > 0:
                orders.append(Order(product, best_bid, -sell_qty))

        return orders

    def update_global_vol_smile(self, state: TradingState) -> None:
        """
        Update the global volatility smile exactly once per timestamp.
        For each option of interest (here, all vouchers), use current market data
        to compute normalized moneyness and the implied volatility.
        """
        # For simplicity, assume underlying is "VOLCANIC_ROCK" and strikes are fixed.
        underlying_sym = "VOLCANIC_ROCK"
        strike_lookup = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        # Check required market data exists
        if underlying_sym not in state.order_depths:
            return
        underlying_depth = state.order_depths[underlying_sym]
        if not underlying_depth.buy_orders or not underlying_depth.sell_orders:
            return
        best_bid_under = max(underlying_depth.buy_orders)
        best_ask_under = min(underlying_depth.sell_orders)
        S = (best_bid_under + best_ask_under) / 2

        # Use same expiration parameters as in strategies.
        expiration_days = 5.0
        trading_days_per_year = 365
        t = state.timestamp / 1_000_000.0
        T = max(0, expiration_days - t) / trading_days_per_year
        if T <= 0:
            return

        # For each voucher symbol with available data, compute and update the smile.
        for sym, strike in strike_lookup.items():
            if sym not in state.order_depths:
                continue
            option_depth = state.order_depths[sym]
            if not option_depth.buy_orders or not option_depth.sell_orders:
                continue
            best_bid_opt = max(option_depth.buy_orders)
            best_ask_opt = min(option_depth.sell_orders)
            option_price = (best_bid_opt + best_ask_opt) / 2

            # Define a simple Black–Scholes call price function.
            def bs_call_price(S, K, T, sigma, r=0.0):
                if T <= 0 or sigma <= 0:
                    return max(S - K, 0)
                d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
                d2 = d1 - sigma * math.sqrt(T)
                return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

            def implied_volatility(option_price, S, K, T):
                intrinsic = max(S - K, 0)
                if option_price < intrinsic:
                    return None
                def objective(sig):
                    return bs_call_price(S, K, T, sig) - option_price
                try:
                    vol = brentq(objective, 1e-6, 5.0, xtol=1e-6)
                    return vol
                except:
                    return None

            iv_calc = implied_volatility(option_price, S, strike, T)
            if iv_calc is None:
                continue
            m = math.log(S / strike) / math.sqrt(T)
            global_vol_smile.update(m, iv_calc)
            # Optionally, print to confirm the update.
            # print(f"Updated global smile for {sym}: (m, iv) = ({m:.4f}, {iv_calc:.4f})")
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions=0

        old_data = json.loads(state.traderData) if state.traderData else {}
        new_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_data:
                strategy.load(old_data[symbol])
            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)
            new_data[symbol] = strategy.save()

        # orders["VOLCANIC_ROCK"]=self.update_volcanic_rock_history(state)
        for product in state.order_depths:
            if product == "VOLCANIC_ROCK_VOUCHER_9500":
                orders[product] = self.trade_mean_reversion(state, product)
            elif product == "VOLCANIC_ROCK_VOUCHER_9750":
                orders[product] = self.trade_mean_reversion(state, product)
            elif product == "VOLCANIC_ROCK_VOUCHER_10000":
                orders[product] = self.trade_mean_reversion(state, product)
            elif product == "VOLCANIC_ROCK_VOUCHER_10250":
                orders[product] = self.trade_mean_reversion(state, product)
            elif product == "VOLCANIC_ROCK_VOUCHER_10500":
                orders[product] = self.trade_mean_reversion(state, product)

        strategy=MacronStrategy("MAGNIFICENT_MACARONS",self.limits["MAGNIFICENT_MACARONS"])
        if "MAGNIFICENT_MACARONS" in old_data:
            strategy.load(old_data["MAGNIFICENT_MACARONS"])
        if "MAGNIFICENT_MACARONS" in state.order_depths:
            orders["MAGNIFICENT_MACARONS"],conversions = strategy.run(state)
        new_data["MAGNIFICENT_MACARONS"] = strategy.save()
        trader_data = json.dumps(new_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data