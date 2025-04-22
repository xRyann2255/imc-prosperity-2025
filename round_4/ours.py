import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List, Optional, Tuple, TypeAlias
import math
import numpy as np

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

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

        return value[:max_length - 3] + "..."

logger = Logger()

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

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

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

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

        if to_buy > 0:
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

        if to_sell > 0:
            popular_sell_price = max(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    # def load(self, data: JSON) -> None:
    #     self.window = deque(data)

class ResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        return 10_000

class KelpStrategy(MarketMakingStrategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)

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

class JamsSwingStrategy(Strategy):
    def __init__(self, symbol: str, limit: int) -> None:
        super().__init__(symbol, limit)
        self.jams_price_history = []
        self.trade_counter = {}
        # Configurable parameters (from your original swing code)
        self.COMPONENT_BUY_ZSCORE = -2.5
        self.COMPONENT_SELL_ZSCORE = 1.0
        self.COMPONENT_SHORT_BIAS = True
        self.POSITION_LIMIT = limit  # Should be 350 as per your config
        self.POSITION_RISK_FACTOR = 0.7
        self.BASE_TRADE_SIZE = 15

    def act(self, state: TradingState) -> None:
        product = self.symbol  # expected to be "JAMS"
        
        # Check order book exists and has both sides
        if product not in state.order_depths:
            return
        depth = state.order_depths[product]
        if not depth.buy_orders or not depth.sell_orders:
            return
        
        position = state.position.get(product, 0)
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Filter out low-spread situations
        if spread < 2:
            return

        # --- PRICE HISTORY TRACKING ---
        self.jams_price_history.append(mid_price)
        if len(self.jams_price_history) > 100:
            self.jams_price_history.pop(0)
        if len(self.jams_price_history) < 10:
            return

        # --- POSITION CAPACITY ---
        long_capacity = self.POSITION_LIMIT - position
        short_capacity = self.POSITION_LIMIT + position
        if long_capacity <= 0 and short_capacity <= 0:
            return

        # --- Z-SCORE & TREND CALCULATION ---
        recent_prices = self.jams_price_history[-30:] if len(self.jams_price_history) >= 30 else self.jams_price_history[:]
        price_mean = np.mean(recent_prices)
        price_std = np.std(recent_prices) if len(recent_prices) > 5 else 1
        z_score = (mid_price - price_mean) / max(price_std, 0.1)
        
        if len(self.jams_price_history) >= 15:
            recent_avg = np.mean(self.jams_price_history[-5:])
            older_avg = np.mean(self.jams_price_history[-15:-5])
            trend = recent_avg - older_avg
        else:
            trend = 0

        # --- TRADE SIZE CALCULATION ---
        position_ratio = abs(position) / self.POSITION_LIMIT
        risk_adjustment = max(0.3, 1 - position_ratio * self.POSITION_RISK_FACTOR)
        
        # Initialize trade counter if not done already
        if product not in self.trade_counter:
            self.trade_counter[product] = 0
        
        # --- BUY SIGNAL ---
        if z_score < self.COMPONENT_BUY_ZSCORE and long_capacity > 0 and (trend > 0 or not self.COMPONENT_SHORT_BIAS):
            intensity = min(2, max(1, abs(z_score) / 2))
            trade_size = int(self.BASE_TRADE_SIZE * intensity * risk_adjustment)
            trade_size = min(trade_size, long_capacity)
            trade_size = max(1, trade_size)
            # Place buy order at the best ask price
            self.buy(best_ask, trade_size)
            self.trade_counter[product] += 1

        # --- SELL SIGNAL ---
        elif (z_score > self.COMPONENT_SELL_ZSCORE or trend < -0.1) and short_capacity > 0:
            intensity = min(3, max(1, abs(z_score) + (1 if trend < 0 else 0)))
            trade_size = int(self.BASE_TRADE_SIZE * intensity * risk_adjustment)
            trade_size = min(trade_size, short_capacity)
            trade_size = max(1, trade_size)
            # Place sell order at the best bid price
            self.sell(best_bid, trade_size)
            self.trade_counter[product] += 1

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


class PFTrendHybridStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int, box_size: float = 10.0, reversal: int = 3,
                 window: int = 50, cooldown: int = 10):
        super().__init__(symbol, limit)
        self.box_size = box_size
        self.reversal = reversal
        self.window = window
        self.cooldown = cooldown

        self.prices: Deque[float] = deque(maxlen=self.window)
        self.last_trend: Optional[str] = None
        self.current_box: Optional[int] = None
        self.cooldown_timer = 0

    def get_box(self, price: float) -> int:
        return int(np.floor(price / self.box_size))

    def detect_trend(self, price: float) -> Optional[str]:
        new_box = self.get_box(price)

        if self.current_box is None:
            self.current_box = new_box
            return None

        box_diff = new_box - self.current_box

        if self.last_trend is None:
            if abs(box_diff) >= self.reversal:
                self.last_trend = 'up' if box_diff > 0 else 'down'
                self.current_box = new_box
        else:
            direction = 1 if self.last_trend == 'up' else -1
            if direction * box_diff > 0:
                self.current_box = new_box  # Continue trend
            elif direction * box_diff <= -self.reversal:
                # Reversal
                self.last_trend = 'down' if self.last_trend == 'up' else 'up'
                self.current_box = new_box

        return self.last_trend

    def act(self, state: TradingState) -> None:
        mid_price = Strategy.get_mid_price(state, self.symbol)
        if mid_price is None:
            return

        position = state.position.get(self.symbol, 0)
        self.prices.append(mid_price)

        if len(self.prices) < self.window:
            return  # Not enough data yet

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return

        trend = self.detect_trend(mid_price)

        if trend == 'up' and position < self.limit:
            self.buy(mid_price, self.limit - position)
            self.cooldown_timer = self.cooldown
        elif trend == 'down' and position > -self.limit:
            self.sell(mid_price, position + self.limit)
            self.cooldown_timer = self.cooldown

    def save(self) -> dict:
        return {
            "prices": list(self.prices),
            "last_trend": self.last_trend,
            "current_box": self.current_box,
            "cooldown_timer": self.cooldown_timer
        }

    def load(self, data: dict, *args) -> None:
        self.prices = deque(data.get("prices", []), maxlen=self.window)
        self.last_trend = data.get("last_trend", None)
        self.current_box = data.get("current_box", None)
        self.cooldown_timer = data.get("cooldown_timer", 0)

class PFTrendHybridStrategyV2(Strategy):
    """
    Hybrid of Point & Figure trend detection and configurable trend-following strategy.
    All parameters from TrendFollowingStrategy are supported.
    """
    def __init__(self, symbol: Symbol, limit: int, box_size: float = 10.0, reversal: int = 3,
                 window: int = 50, threshold= 10.0, bias: int = 0, cooldown: int = 10):
        super().__init__(symbol, limit)
        self.box_size = box_size
        self.reversal = reversal
        self.window = window
        self.threshold = threshold
        self.bias = bias
        self.cooldown = cooldown

        self.prices: Deque[float] = deque(maxlen=window)
        self.last_trend: Optional[str] = None
        self.current_box: Optional[int] = None
        self.cooldown_timer = 0

    def get_box(self, price: float) -> int:
        return int(np.floor(price / self.box_size))

    def detect_trend(self, price: float) -> Optional[str]:
        new_box = self.get_box(price)

        if self.current_box is None:
            self.current_box = new_box
            return None

        box_diff = new_box - self.current_box

        if self.last_trend is None:
            if abs(box_diff) >= self.reversal:
                self.last_trend = 'up' if box_diff > 0 else 'down'
                self.current_box = new_box
        else:
            direction = 1 if self.last_trend == 'up' else -1
            if direction * box_diff > 0:
                self.current_box = new_box  # Continue trend
            elif direction * box_diff <= -self.reversal:
                # Reversal
                self.last_trend = 'down' if self.last_trend == 'up' else 'up'
                self.current_box = new_box

        return self.last_trend

    def act(self, state: TradingState) -> None:
        mid_price = Strategy.get_mid_price(state, self.symbol)
        if mid_price is None:
            return

        position = state.position.get(self.symbol, 0)
        self.prices.append(mid_price)

        if len(self.prices) < self.window:
            return  # Not enough data yet

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return

        mean_price = sum(self.prices) / self.window
        if isinstance(self.threshold, tuple):
            upper_band = mean_price + self.threshold[0]
            lower_band = mean_price - self.threshold[1]
        else:
            upper_band = mean_price + self.threshold
            lower_band = mean_price - self.threshold

        trend = self.detect_trend(mid_price)

        if trend == 'up':
            if mid_price < upper_band and position < self.limit:
                self.buy(mid_price, self.limit - position)
                self.cooldown_timer = self.cooldown
        elif trend == 'down':
            if mid_price > lower_band and position > -self.limit:
                self.sell(mid_price, position + self.limit)
                self.cooldown_timer = self.cooldown
        else:
            # Optional fallback: bias-based trading
            if self.bias > 0:
                self.buy(mid_price, min(self.bias, self.limit - position))
            elif self.bias < 0:
                self.sell(mid_price, min(abs(self.bias), position + self.limit))

    def save(self) -> dict:
        return {
            "prices": list(self.prices),
            "last_trend": self.last_trend,
            "current_box": self.current_box,
            "cooldown_timer": self.cooldown_timer
        }

    def load(self, data: dict, *args) -> None:
        self.prices = deque(data.get("prices", []), maxlen=self.window)
        self.last_trend = data.get("last_trend", None)
        self.current_box = data.get("current_box", None)
        self.cooldown_timer = data.get("cooldown_timer", 0)

class PFTrendHybridStrategyV3(Strategy):
    """
    Advanced hybrid strategy with:
    - Trend following during breakouts
    - Mean reversion in choppy zones
    - Market making during neutral/low-volatility periods
    - Bias trading in neutral zones with cooldown
    """
    def __init__(
        self, symbol: Symbol, limit: int,
        box_size: float = 10.0, reversal: int = 3, window: int = 50,
        trend_threshold: float = 10.0, reversion_threshold: float = 2.0,
        vol_window: int = 20, vol_threshold: float = 5.0,
        mm_spread: float = 1.0, mm_size: int = 1,
        cooldown: int = 10, bias_cooldown: int = 2,
        bias: int = 0
    ):
        super().__init__(symbol, limit)
        self.box_size = box_size
        self.reversal = reversal
        self.window = window
        self.trend_threshold = trend_threshold
        self.reversion_threshold = reversion_threshold
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.mm_spread = mm_spread
        self.mm_size = mm_size
        self.cooldown = cooldown
        self.bias_cooldown = bias_cooldown
        self.bias = bias

        self.prices: Deque[float] = deque(maxlen=window)
        self.returns: Deque[float] = deque(maxlen=vol_window)
        self.last_trend: Optional[str] = None
        self.current_box: Optional[int] = None
        self.cooldown_timer = 0
        self.bias_timer = 0

    def get_box(self, price: float) -> int:
        return int(np.floor(price / self.box_size))

    def detect_trend(self, price: float) -> Optional[str]:
        new_box = self.get_box(price)

        if self.current_box is None:
            self.current_box = new_box
            return None

        box_diff = new_box - self.current_box

        if self.last_trend is None:
            if abs(box_diff) >= self.reversal:
                self.last_trend = 'up' if box_diff > 0 else 'down'
                self.current_box = new_box
        else:
            direction = 1 if self.last_trend == 'up' else -1
            if direction * box_diff > 0:
                self.current_box = new_box  # Continue trend
            elif direction * box_diff <= -self.reversal:
                self.last_trend = 'down' if self.last_trend == 'up' else 'up'
                self.current_box = new_box

        return self.last_trend

    def compute_volatility(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        return np.std(self.returns)

    def act(self, state: TradingState) -> None:
        mid_price = Strategy.get_mid_price(state, self.symbol)
        if mid_price is None:
            return

        position = state.position.get(self.symbol, 0)
        self.prices.append(mid_price)

        if len(self.prices) >= 2:
            self.returns.append(self.prices[-1] - self.prices[-2])

        if len(self.prices) < self.window:
            return

        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return

        mean_price = sum(self.prices) / self.window
        z_score = (mid_price - mean_price) / (np.std(self.prices) + 1e-6)
        vol = self.compute_volatility()
        trend = self.detect_trend(mid_price)

        # --- STRATEGY SELECTION LOGIC ---
        if vol > self.vol_threshold:
            self.market_make(mid_price, position)
            return

        if abs(z_score) > self.reversion_threshold:
            # Mean reversion mode
            if z_score > 0 and position > -self.limit:
                self.sell(mid_price, min(self.limit + position, self.limit))
                self.cooldown_timer = self.cooldown
            elif z_score < 0 and position < self.limit:
                self.buy(mid_price, min(self.limit - position, self.limit))
                self.cooldown_timer = self.cooldown
            return

        if trend == 'up' and position < self.limit:
            self.buy(mid_price, self.limit - position)
            self.cooldown_timer = self.cooldown
            self.bias_timer = 0
        elif trend == 'down' and position > -self.limit:
            self.sell(mid_price, position + self.limit)
            self.cooldown_timer = self.cooldown
            self.bias_timer = 0
        else:
            # Neutral zone: bias trading + market making
            if self.bias_timer == 0:
                if self.bias > 0 and position < self.limit:
                    self.buy(mid_price, min(self.bias, self.limit - position))
                elif self.bias < 0 and position > -self.limit:
                    self.sell(mid_price, min(abs(self.bias), position + self.limit))
                self.bias_timer = self.bias_cooldown
            else:
                self.bias_timer -= 1

            self.market_make(mid_price, position)

    def market_make(self, mid_price: float, position: int):
        bid = mid_price - self.mm_spread / 2
        ask = mid_price + self.mm_spread / 2
        if position + self.mm_size <= self.limit:
            self.buy(bid, self.mm_size)
        if position - self.mm_size >= -self.limit:
            self.sell(ask, self.mm_size)

    def save(self) -> dict:
        return {
            "prices": list(self.prices),
            "returns": list(self.returns),
            "last_trend": self.last_trend,
            "current_box": self.current_box,
            "cooldown_timer": self.cooldown_timer,
            "bias_timer": self.bias_timer
        }

    def load(self, data: dict, *args) -> None:
        self.prices = deque(data.get("prices", []), maxlen=self.window)
        self.returns = deque(data.get("returns", []), maxlen=self.vol_window)
        self.last_trend = data.get("last_trend", None)
        self.current_box = data.get("current_box", None)
        self.cooldown_timer = data.get("cooldown_timer", 0)
        self.bias_timer = data.get("bias_timer", 0)
        
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
        hedge_orders = self.hedge_with_vouchers(state)
        orders += hedge_orders

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

# Create a single global instance.

class Trader:
    def __init__(self) -> None:
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS" : 250,
            "JAMS" : 350,
            "DJEMBES" : 60,
            "PICNIC_BASKET1" : 60,
            "PICNIC_BASKET2" : 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }
        underlying_limits = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60
        }
        self.strike_price = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        }
        self.threshold = {
            "VOLCANIC_ROCK_VOUCHER_9500": 1.9,
            "VOLCANIC_ROCK_VOUCHER_9750": 1.9,
            "VOLCANIC_ROCK_VOUCHER_10000": 1.9,
            "VOLCANIC_ROCK_VOUCHER_10250": 1.9,
            "VOLCANIC_ROCK_VOUCHER_10500": 1.9,
            "VOLCANIC_ROCK": 1.9,
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

            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", self.limits["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", self.limits["KELP"]),
            # "CROISSANTS": BasketHedgeStrategy("CROISSANTS", limits["CROISSANTS"], {"PICNIC_BASKET1": 6, "PICNIC_BASKET2": 4}),
            "JAMS": JamsSwingStrategy("JAMS", self.limits["JAMS"]),
            # "DJEMBES": BasketHedgeStrategy("DJEMBES", limits["DJEMBES"], {"PICNIC_BASKET1": 1}),
            
            "PICNIC_BASKET1": PicnicBasket1Strategy("PICNIC_BASKET1", self.limits["PICNIC_BASKET1"]), #best
            "PICNIC_BASKET2": PicnicBasket2Strategy("PICNIC_BASKET2", self.limits["PICNIC_BASKET2"]), #best

            "SQUID_INK": SquidInkStrategy("SQUID_INK", self.limits["SQUID_INK"]), #final
            "VOLCANIC_ROCK": VolcanicRockStrategy() #best


        }
    def trade_mean_reversion(self, state: TradingState, product: Symbol) -> List[Order]:
        orders = []
        position = state.position.get(product, 0)
        position_limit = self.limits[product]

        threshold = self.threshold[product]

        product_depth = state.order_depths.get(product)
        if not product_depth:
            return orders

        # Track and update history for each voucher
        if product not in self.history:
            self.history[product] = deque(maxlen=50)
        if product_depth.buy_orders and product_depth.sell_orders:
            bid = max(product_depth.buy_orders)
            ask = min(product_depth.sell_orders)
            mid = (bid + ask) / 2
            self.history[product].append(mid)

        # Calculate z-score for the voucher
        recent = np.array(self.history[product])
        if len(recent) < 50:
            return orders
        mean = np.mean(recent[-50:])
        std = np.std(recent[-50:])
        z = (recent[-1] - mean) / std if std > 0 else 0

        # Buy logic
        if z < -threshold and product_depth.sell_orders:
            best_ask = min(product_depth.sell_orders)
            qty = -product_depth.sell_orders[best_ask]
            buy_qty = min(qty, position_limit - position)
            if buy_qty > 0:
                orders.append(Order(product, best_ask, buy_qty))

        # Sell logic
        elif z > threshold and product_depth.buy_orders:
            best_bid = max(product_depth.buy_orders)
            qty = product_depth.buy_orders[best_bid]
            sell_qty = min(qty, position + position_limit)
            if sell_qty > 0:
                orders.append(Order(product, best_bid, -sell_qty))

        return orders

    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        old_data = json.loads(state.traderData) if state.traderData else {}
        new_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_data:
                strategy.load(old_data[symbol])
            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)
            new_data[symbol] = strategy.save()
        
        for product in state.order_depths:
            if product == "VOLCANIC_ROCK_VOUCHER_9500":
                orders[product] = self.trade_mean_reversion(state, product)
            elif product == "VOLCANIC_ROCK_VOUCHER_9750":
                orders[product] = self.trade_mean_reversion(state, product)
            elif product == "VOLCANIC_ROCK_VOUCHER_10000":
                orders[product] = self.trade_mean_reversion(state, product)

        trader_data = json.dumps(new_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
