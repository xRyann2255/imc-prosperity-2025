from collections import deque
import json
from abc import abstractmethod
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias

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
                observation.sunlightIndex,
                observation.sugarPrice,
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


class MacaronStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, n_days: int, z_low: float, z_high: float,
                 profit_margin: int, sell_off_ratio: float,
                 low_trade_ratio: float,
                 entering_aggression: int, take_profit_aggression: int, clearing_aggression: int,
                 high_hold_duration: int, low_hold_duration: int, safety: int, storage_cost: float = 0.1, grad: float = -.5, csi: float = 44.75) -> None:
        super().__init__(symbol, limit)
        self.true_value_history = deque(maxlen=2000)
        self.n_days = n_days
        self.z_low = z_low
        self.z_high = z_high
        self.profit_margin = profit_margin
        self.sell_off_ratio = sell_off_ratio
        # The original hold_duration becomes a fallback parameter
        self.low_trade_ratio = low_trade_ratio
        self.entering_aggression = entering_aggression
        self.take_profit_aggression = take_profit_aggression
        self.clearing_aggression = clearing_aggression

        # New parameters for different hold durations based on z-score intensity.
        self.high_hold_duration = high_hold_duration
        self.low_hold_duration = low_hold_duration
        self.entry_hold_duration = 0  # Will be set at entry.

        self.entry_time = None
        self.entry_side = None
        self.entry_price = None
        self.true_value_estimate = None
        self.error_cov = 1.0
        self.process_variance = 7.0
        self.measurement_variance = 3.0
        self.measurement = None

        self.csi = csi
        self.grad = grad
        self.storage_cost = storage_cost
        self.window = deque()
        self.window_size = 10
        self.prev_csi = 0
        self.prev_grad = 0
        self.long = False
        self.safety = safety
        # self.in_position = False
        # self.entry_price: float | None = None
        # self.max_price_since_entry: float | None = None

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
        
    def act(self, state: TradingState) -> None:

        # Let's check the CSI
        current_csi = state.observations.conversionObservations[self.symbol].sunlightIndex

        
        if self.long and current_csi > self.prev_csi:
            self.prev_grad = current_csi - self.prev_csi
            self.prev_csi = current_csi
            # Time to short
            self.long = False
            order_depth = state.order_depths[self.symbol]
            buy_orders = sorted(order_depth.buy_orders.items())
            position = state.position.get(self.symbol, 0)
            best_bid = max(order_depth.sell_orders.keys())

            for price, volume in buy_orders:
                if position == -self.limit:
                    # We're as short as we can be
                    return
                else:
                    if price <= best_bid - self.safety:
                        quantity = min(position + self.limit, volume)
                        to_sell = min(quantity, position + self.limit)
                        self.sell(price, to_sell)
                        position -= to_sell

        elif current_csi >= self.csi + 1.5:
            self.prev_grad = current_csi - self.prev_csi
            self.prev_csi = current_csi
            # We market make
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
        
        elif self.prev_grad < 0 and self.prev_grad > self.grad:
            self.prev_grad = current_csi - self.prev_csi
            self.prev_csi = current_csi
            self.apply_mean_deviation(state)

        else:
            # We want to short
            self.prev_grad = current_csi - self.prev_csi
            self.prev_csi = current_csi
            # We want to long
            order_depth = state.order_depths[self.symbol]
            sell_orders = sorted(order_depth.sell_orders.items())
            position = state.position.get(self.symbol, 0)
            best_ask = min(order_depth.sell_orders.keys())

            for price, volume in sell_orders:
                if position == self.limit:
                    self.long = True
                    return
                else:
                    if price <= best_ask + self.safety:
                        quantity = min(self.limit - position, volume)
                        to_buy = min(quantity, self.limit - position)
                        self.buy(price, to_buy)
                        position += to_buy
       
    def apply_mean_deviation(self, state: TradingState) -> None:
        true_value = self.get_true_value_MD(state)
        current_time = state.timestamp
        current_position = state.position.get(self.symbol, 0)

        # Compute n-day returns from true_value_history.
        if len(self.true_value_history) >= self.n_days + 1:
            returns = []
            for i in range(self.n_days, len(self.true_value_history)):
                prev = self.true_value_history[i - self.n_days]
                curr = self.true_value_history[i]
                if prev != 0:
                    returns.append((curr - prev) / prev)
            if returns:
                avg_return = sum(returns) / len(returns)
                std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                latest_return = returns[-1]
                z_score = (latest_return - avg_return) / std_return if std_return > 0 else 0
            else:
                z_score = 0
        else:
            z_score = 0

        # ----- Entry Signal -----
        # For a short signal, we want to sell (i.e. go short) aggressively.
        if z_score > self.z_high:
            entry_sell_price = round(self.measurement - self.entering_aggression)
            self.sell(entry_sell_price, self.limit + current_position)
            if self.entry_time is None:
                self.entry_time = current_time
                self.entry_side = "short"
                self.entry_price = self.measurement
                self.entry_hold_duration = self.high_hold_duration  # High z -> long hold.
        elif z_score > self.z_low:
            entry_sell_price = round(self.measurement - self.take_profit_aggression)
            self.sell(entry_sell_price, round((self.limit + current_position) * self.low_trade_ratio))
            if self.entry_time is None:
                self.entry_time = current_time
                self.entry_side = "short"
                self.entry_price = self.measurement
                self.entry_hold_duration = self.low_hold_duration  # Lower z -> short hold.
        # For a long signal, we want to buy aggressively.
        elif z_score < -self.z_high:
            entry_buy_price = round(self.measurement + self.entering_aggression)
            self.buy(entry_buy_price, self.limit - current_position)
            if self.entry_time is None:
                self.entry_time = current_time
                self.entry_side = "long"
                self.entry_price = self.measurement
                self.entry_hold_duration = self.high_hold_duration  # High |z| -> long hold.
        elif z_score < -self.z_low:
            entry_buy_price = round(self.measurement + self.take_profit_aggression)
            self.buy(entry_buy_price, round((self.limit - current_position) * self.low_trade_ratio))
            if self.entry_time is None:
                self.entry_time = current_time
                self.entry_side = "long"
                self.entry_price = self.measurement
                self.entry_hold_duration = self.low_hold_duration  # Lower |z| -> short hold.

        # ----- Profit-Taking Logic -----
        if self.entry_side == "long" and self.measurement >= self.entry_price + self.profit_margin:
            profit_qty = int(current_position * self.sell_off_ratio)
            if profit_qty > 0:
                profit_sell_price = round(self.measurement) - self.take_profit_aggression
                self.sell(profit_sell_price, profit_qty)
        elif self.entry_side == "short" and self.measurement <= self.entry_price - self.profit_margin:
            profit_qty = int((-current_position) * self.sell_off_ratio)
            if profit_qty > 0:
                profit_buy_price = round(self.measurement) + self.take_profit_aggression
                self.buy(profit_buy_price, profit_qty)

        # ----- Clearing Logic -----
        # Use the duration stored at entry time. This provides a longer trading duration for high z score trade conditions
        # and a shorter holding period for lower z score trade conditions.
        if self.entry_time is not None and (current_time - self.entry_time >= self.entry_hold_duration):
            # We switch from measurement-based to best-bid / best-ask.
            order_depth = state.order_depths[self.symbol]
            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
            sell_orders = sorted(order_depth.sell_orders.items())

            if current_position > 0 and sell_orders:
                # For a long position, undercut the best ask
                best_ask = min(sell_orders, key=lambda tup: tup[0])[0]
                clearing_sell_price = round(best_ask - self.clearing_aggression)
                self.sell(clearing_sell_price, current_position)
                self.entry_time = None
                self.entry_side = None

            elif current_position < 0 and buy_orders:
                # For a short position, overcut the best bid
                best_bid = max(buy_orders, key=lambda tup: tup[0])[0]
                clearing_buy_price = round(best_bid + self.clearing_aggression)
                self.buy(clearing_buy_price, -current_position)
                self.entry_time = None
                self.entry_side = None

        if self.entry_time is None and current_time > 50000:
            # Determine bid and ask prices around the estimated measurement.
            order_depth = state.order_depths[self.symbol]
            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
            sell_orders = sorted(order_depth.sell_orders.items())

            position = state.position.get(self.symbol, 0)
            to_buy = self.limit - position
            to_sell = self.limit + position

            max_buy_price = self.measurement - 1 if position > self.limit * 0.33 else self.measurement
            min_sell_price = self.measurement + 1 if position < self.limit * -0.33 else self.measurement


            if to_buy > 0:
                popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
                price = min(max_buy_price, popular_buy_price + 1)
                self.buy(round(price), to_buy)


            if to_sell > 0:
                popular_sell_price = max(sell_orders, key=lambda tup: tup[1])[0]
                price = max(min_sell_price, popular_sell_price - 1)
                self.sell(round(price), to_sell)

    def get_true_value_MD(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = max(sell_orders, key=lambda tup: tup[1])[0]
        measurement = (popular_buy_price + popular_sell_price) / 2
        self.measurement = measurement

        if self.true_value_estimate is None:
            self.true_value_estimate = measurement
        else:
            self.error_cov += self.process_variance
            kalman_gain = self.error_cov / (self.error_cov + self.measurement_variance)
            self.true_value_estimate += kalman_gain * (measurement - self.true_value_estimate)
            self.error_cov = (1 - kalman_gain) * self.error_cov

        self.true_value_history.append(self.true_value_estimate)
        return round(self.true_value_estimate)

    
    def save(self) -> Any:
        return self.trader_data

    def load(self, data: Any) -> None:
        if data is not None:
            self.trader_data = data

class Trader:
    def __init__(self) -> None:
        limits = {"MAGNIFICENT_MACARONS": 75}
        # U should match the notebook output
        self.strategies = {"MACARON": MacaronStrategy("MAGNIFICENT_MACARONS", limits["MAGNIFICENT_MACARONS"], n_days=210,
                                                   z_high=2.0,
                                                   z_low=.2,
                                                   profit_margin=17,
                                                   sell_off_ratio=0.25,
                                                   entering_aggression=1.6,
                                                   take_profit_aggression=0,
                                                   clearing_aggression=1,
                                                   high_hold_duration=15000,
                                                   low_hold_duration=8000,
                                                   low_trade_ratio=0.5, safety=50, storage_cost=0.1, grad=-2, csi=47)}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        orders: dict[Symbol, list[Order]] = {}
        # load saved state
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
                for name, strat in self.strategies.items():
                    if name in saved:
                        strat.load(saved[name])
            except Exception as e:
                logger.print(f"Error loading data: {e}")
        # execute
        for name, strat in self.strategies.items():
            try:
                for o in strat.run(state):
                    orders.setdefault(o.symbol, []).append(o)
            except Exception as e:
                logger.print(f"Error {name}: {e}")
        # save state
        to_save: dict[str, Any] = {}
        for name, strat in self.strategies.items():
            try:
                to_save[name] = strat.save()
            except Exception as e:
                logger.print(f"Error saving {name}: {e}")
        td = json.dumps(to_save, separators=(",", ":"))
        # flush
        conversions = 0
        logger.flush(state, orders, conversions, td)
        return orders, conversions, td
