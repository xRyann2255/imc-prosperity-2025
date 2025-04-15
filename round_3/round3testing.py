import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Optional, Tuple, TypeAlias
import math
from scipy.stats import norm
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
                observation.sunlight,
                observation.humidity,
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
    
class SquidInkDeviationStrategy(MarketMakingStrategy):
    def __init__(self, symbol: str, limit: int, n_days: int,
                 z_low: float, z_high: float,
                 profit_margin: int, sell_off_ratio: float,
                 low_trade_ratio: float,
                 entering_aggression: int, take_profit_aggression: int, clearing_aggression: int,
                 high_hold_duration: int, low_hold_duration: int) -> None:
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

    def get_true_value(self, state: TradingState) -> int:
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

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
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

class BasketTradingStrategy(Strategy):
    def __init__(self,
                 name: str,
                 pb1_position_limit: int,
                 pb2_position_limit: int,
                 d_position_limit: int,
                 j_position_limit: int,
                 c_position_limit: int,
                 pb1_pb2_spread_avg: float,
                 pb1_pb2_spread_std: float,
                 pb1_pb2_z_thresh: float,
                 pb1_pb2_exit_thesh: float,
                 pb1_pb2_timeframe: int,
                 pb1_pb2_std_scaling: float,
                 pb1_pb2_std_bandwidth: float) -> None:
        # Call the base constructor with a dummy symbol (not used) and a limit of 0.
        super().__init__(name, 0)
        self.pb1_position_limit = pb1_position_limit      # PICNIC_BASKET1 limit
        self.pb2_position_limit = pb2_position_limit      # PICNIC_BASKET2 limit
        self.d_position_limit = d_position_limit          # DJEMBES limit
        self.j_position_limit = j_position_limit          # JAMS limit (if needed)
        self.c_position_limit = c_position_limit          # CROISSANTS limit (if needed)

        self.pb1_pb2_spread_avg = pb1_pb2_spread_avg
        self.pb1_pb2_spread_std = pb1_pb2_spread_std
        self.pb1_pb2_z_thresh = pb1_pb2_z_thresh
        self.pb1_pb2_exit_thesh = pb1_pb2_exit_thesh
        self.pb1_pb2_timeframe = pb1_pb2_timeframe
        self.pb1_pb2_std_scaling = pb1_pb2_std_scaling
        self.pb1_pb2_std_bandwidth = pb1_pb2_std_bandwidth

        # Trader data holds persistent internal state
        self.trader_data = {
            "pb1_pb2_vol_pb1": 0,
            "pb1_pb2_vol_pb2": 0,
            "pb1_pb2_vol_d": 0,
            "tic": 0,
            "pb1_pb2_std": 0,
            "pb1_pb2_spread": []
        }

    def act(self, state: TradingState) -> None:
        # Ensure that the required basket order depths exist in the state.
        if "PICNIC_BASKET1" not in state.order_depths or \
           "PICNIC_BASKET2" not in state.order_depths or \
           "DJEMBES" not in state.order_depths:
            return

        pb1_od = state.order_depths["PICNIC_BASKET1"]
        pb2_od = state.order_depths["PICNIC_BASKET2"]
        d_od   = state.order_depths["DJEMBES"]

        # Compute mid-prices for the basket components.
        pb1_best_ask = min(pb1_od.sell_orders.keys())
        pb1_best_bid = max(pb1_od.buy_orders.keys())
        pb1_mid_price = (pb1_best_ask + pb1_best_bid) / 2

        pb2_best_ask = min(pb2_od.sell_orders.keys())
        pb2_best_bid = max(pb2_od.buy_orders.keys())
        pb2_mid_price = (pb2_best_ask + pb2_best_bid) / 2

        d_best_ask = min(d_od.sell_orders.keys())
        d_best_bid = max(d_od.buy_orders.keys())
        d_mid_price = (d_best_ask + d_best_bid) / 2

        # Calculate the basket spread (using a weighted sum).
        spread = pb1_mid_price - (1.5 * pb2_mid_price + d_mid_price)
        self.trader_data["pb1_pb2_spread"].append(spread)
        add = 0
        if len(self.trader_data["pb1_pb2_spread"]) > self.pb1_pb2_timeframe:
            self.trader_data["pb1_pb2_spread"].pop(0)
            std = np.std(self.trader_data["pb1_pb2_spread"])
            self.trader_data["pb1_pb2_std"] = std
            add = self.pb1_pb2_std_scaling * std - self.pb1_pb2_std_bandwidth

        # Get current positions (defaulting to 0 if not present)
        pb1_position = state.position.get("PICNIC_BASKET1", 0)
        pb2_position = state.position.get("PICNIC_BASKET2", 0)
        d_position   = state.position.get("DJEMBES", 0)

        pb1_price = pb1_quantity = pb2_price = pb2_quantity = d_price = d_quantity = 0

        # Compare the computed spread with thresholds (plus an adjustment 'add')
        if spread > self.pb1_pb2_spread_avg + self.pb1_pb2_spread_std * self.pb1_pb2_z_thresh + add:
            # Signal: sell PICNIC_BASKET1, buy PICNIC_BASKET2 and DJEMBES.
            pb1_limit = min(self.pb1_position_limit + pb1_position, abs(pb1_od.buy_orders[pb1_best_bid]))
            pb2_limit = min(self.pb2_position_limit - pb2_position, abs(pb2_od.sell_orders[pb2_best_ask])) / 1.5
            d_limit   = min(self.d_position_limit - d_position, abs(d_od.sell_orders[d_best_ask]))
            limit = math.floor(min(pb1_limit, pb2_limit, d_limit))
            limit = (limit // 2) * 2
            pb1_quantity, pb2_quantity, d_quantity = -limit, 1.5 * limit, limit
            pb1_price, pb2_price, d_price = pb1_best_bid, pb2_best_ask, d_best_ask

        elif spread < self.pb1_pb2_spread_avg - self.pb1_pb2_spread_std * self.pb1_pb2_z_thresh - add:
            # Signal: buy PICNIC_BASKET1, sell PICNIC_BASKET2 and DJEMBES.
            pb1_limit = min(self.pb1_position_limit - pb1_position, abs(pb1_od.sell_orders[pb1_best_ask]))
            pb2_limit = min(self.pb2_position_limit + pb2_position, abs(pb2_od.buy_orders[pb2_best_bid])) / 1.5
            d_limit   = min(self.d_position_limit + d_position, abs(d_od.buy_orders[d_best_bid]))
            limit = math.floor(min(pb1_limit, pb2_limit, d_limit))
            limit = (limit // 2) * 2
            pb1_quantity, pb2_quantity, d_quantity = limit, -1.5 * limit, -limit
            pb1_price, pb2_price, d_price = pb1_best_ask, pb2_best_bid, d_best_bid

        elif pb1_position > 0 and spread > self.pb1_pb2_spread_avg - (self.pb1_pb2_spread_std * self.pb1_pb2_z_thresh + add) * self.pb1_pb2_exit_thesh:
            # Exit signal for a long position.
            pb1_limit = min(abs(pb1_position), abs(pb1_od.buy_orders[pb1_best_bid]))
            pb2_limit = min(abs(pb2_position), abs(pb2_od.sell_orders[pb2_best_ask])) / 1.5
            d_limit   = min(abs(d_position), abs(d_od.sell_orders[d_best_ask]))
            limit = math.floor(min(pb1_limit, pb2_limit, d_limit))
            limit = (limit // 2) * 2
            pb1_quantity, pb2_quantity, d_quantity = -limit, 1.5 * limit, limit
            pb1_price, pb2_price, d_price = pb1_best_bid, pb2_best_ask, d_best_ask

        elif pb1_position < 0 and spread < self.pb1_pb2_spread_avg + (self.pb1_pb2_spread_std * self.pb1_pb2_z_thresh + add) * self.pb1_pb2_exit_thesh:
            # Exit signal for a short position.
            pb1_limit = min(abs(pb1_position), abs(pb1_od.sell_orders[pb1_best_ask]))
            pb2_limit = min(abs(pb2_position), abs(pb2_od.buy_orders[pb2_best_bid])) / 1.5
            d_limit   = min(abs(d_position), abs(d_od.buy_orders[d_best_bid]))
            limit = math.floor(min(pb1_limit, pb2_limit, d_limit))
            limit = (limit // 2) * 2
            pb1_quantity, pb2_quantity, d_quantity = limit, -1.5 * limit, -limit
            pb1_price, pb2_price, d_price = pb1_best_ask, pb2_best_bid, d_best_bid

        # Add non-zero orders to this strategy’s list.
        if pb1_quantity != 0:
            self.orders.append(Order("PICNIC_BASKET1", pb1_price, int(round(pb1_quantity))))
        if pb2_quantity != 0:
            self.orders.append(Order("PICNIC_BASKET2", pb2_price, int(round(pb2_quantity))))
        if d_quantity != 0:
            self.orders.append(Order("DJEMBES", d_price, int(round(d_quantity))))

    def save(self) -> Any:
        return self.trader_data

    def load(self, data: Any) -> None:
        if data is not None:
            self.trader_data = data

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

class VolcanicRockVoucherSpreadArbStrategy(Strategy):
    def __init__(self, voucher_lower: Symbol, voucher_higher: Symbol, trade_qty: int, execution_threshold: int) -> None:
        """
        This strategy performs spread arbitrage.
        It examines two instruments:
          - voucher_lower (e.g. lower–strike voucher) and
          - voucher_higher (e.g. higher–strike voucher).
        It computes the executable spread as:
            best_bid(voucher_lower) - best_ask(voucher_higher)
        If the spread exceeds execution_threshold (e.g. 250 SeaShells), it:
          - Sells voucher_lower at its best bid.
          - Buys voucher_higher at its best ask.
        The number of units traded is the minimum of trade_qty and available volumes.
        """
        super().__init__("", 0)
        self.voucher_lower = voucher_lower
        self.voucher_higher = voucher_higher
        self.trade_qty = trade_qty
        self.execution_threshold = execution_threshold

    def get_best_bid_with_vol(self, state: TradingState, symbol: str) -> Optional[Tuple[float, int]]:
        order_depth = state.order_depths.get(symbol)
        if order_depth and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            volume = order_depth.buy_orders[best_bid]
            return best_bid, volume
        return None

    def get_best_ask_with_vol(self, state: TradingState, symbol: str) -> Optional[Tuple[float, int]]:
        order_depth = state.order_depths.get(symbol)
        if order_depth and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            volume = abs(order_depth.sell_orders[best_ask])
            return best_ask, volume
        return None

    def get_order_for_option(self, state: TradingState, option: Symbol) -> Optional[Order]:
        bid_tuple = self.get_best_bid_with_vol(state, self.voucher_lower)
        ask_tuple = self.get_best_ask_with_vol(state, self.voucher_higher)
        if bid_tuple is None or ask_tuple is None:
            return None
        best_bid_lower, vol_lower = bid_tuple
        best_ask_higher, vol_higher = ask_tuple
        spread = best_bid_lower - best_ask_higher
        # print(f"Spread Arbitrage Check: Best Bid {self.voucher_lower}: {best_bid_lower} (vol: {vol_lower}), Best Ask {self.voucher_higher}: {best_ask_higher} (vol: {vol_higher}), Spread: {spread}")
        if spread > self.execution_threshold:
            effective_qty = min(self.trade_qty, vol_lower, vol_higher)
            if effective_qty <= 0:
                return None
            if option == self.voucher_lower:
                return Order(self.voucher_lower, int(round(best_bid_lower)), -effective_qty)
            elif option == self.voucher_higher:
                return Order(self.voucher_higher, int(round(best_ask_higher)), effective_qty)
        return None

    def act(self, state: TradingState) -> None:
        orders: List[Order] = []
        order_lower = self.get_order_for_option(state, self.voucher_lower)
        order_higher = self.get_order_for_option(state, self.voucher_higher)
        if order_lower is not None and order_higher is not None:
            orders.append(order_lower)
            orders.append(order_higher)
        self.orders = orders

    def run(self, state: TradingState) -> List[Order]:
        self.act(state)
        return self.orders

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass
            

class VolcanicRockVoucherFlyArbStrategy(Strategy):
    def __init__(self, voucher_low: Symbol, voucher_mid: Symbol, voucher_high: Symbol,
                 strike_low: int, strike_mid: int, strike_high: int,
                 trade_qty: int) -> None:
        """
        This strategy looks for a butterfly spread arbitrage (fly).
        It uses three option instruments with strikes: strike_low < strike_mid < strike_high.
        A standard long butterfly position is:
          - Buy 1 unit of the low–strike option,
          - Sell 2 units of the mid–strike option,
          - Buy 1 unit of the high–strike option.
        The net entry cost is computed as:
            cost = best_ask(voucher_low) - 2 * best_bid(voucher_mid) + best_ask(voucher_high)
        The maximum payoff of a long butterfly is approximately:
            max_payoff = strike_mid - strike_low
        This strategy will enter:
          - A long butterfly (if cost < 0), or
          - A short (reverse) butterfly (if cost > max_payoff).
        In a long butterfly, you:
          Buy low, Sell 2× mid, Buy high.
        In a short butterfly, you:
          Sell low, Buy 2× mid, Sell high.
        Trade quantity is the number of butterfly spreads (legs’ trade sizes are scaled accordingly).
        """
        super().__init__("", 0)
        self.voucher_low = voucher_low
        self.voucher_mid = voucher_mid
        self.voucher_high = voucher_high
        self.strike_low = strike_low
        self.strike_mid = strike_mid
        self.strike_high = strike_high
        self.trade_qty = trade_qty
        # No separate payoff threshold parameter: the decision is based on comparing cost and max payoff.

    def get_best_ask_with_vol(self, state: TradingState, symbol: str) -> Optional[Tuple[float, int]]:
        order_depth = state.order_depths.get(symbol)
        if order_depth and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            volume = abs(order_depth.sell_orders[best_ask])
            return best_ask, volume
        return None

    def get_best_bid_with_vol(self, state: TradingState, symbol: str) -> Optional[Tuple[float, int]]:
        order_depth = state.order_depths.get(symbol)
        if order_depth and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            volume = order_depth.buy_orders[best_bid]
            return best_bid, volume
        return None

    def compute_fly_cost(self, state: TradingState) -> Optional[float]:
        """
        Compute the net entry cost for a long butterfly:
          cost = best_ask(voucher_low) - 2 * best_bid(voucher_mid) + best_ask(voucher_high)
        """
        low_tuple = self.get_best_ask_with_vol(state, self.voucher_low)
        mid_tuple = self.get_best_bid_with_vol(state, self.voucher_mid)
        high_tuple = self.get_best_ask_with_vol(state, self.voucher_high)
        if low_tuple is None or mid_tuple is None or high_tuple is None:
            return None
        best_ask_low, _ = low_tuple
        best_bid_mid, _ = mid_tuple
        best_ask_high, _ = high_tuple
        cost = best_ask_low - 2 * best_bid_mid + best_ask_high
        # print(f"Fly Cost: best_ask({self.voucher_low})={best_ask_low}, best_bid({self.voucher_mid})={best_bid_mid}, best_ask({self.voucher_high})={best_ask_high}, cost={cost}")
        return cost

    def compute_max_payoff(self) -> int:
        """
        For a long butterfly, the approximate maximum payoff is strike_mid - strike_low.
        """
        return self.strike_mid - self.strike_low

    def get_trade_quantity(self, state: TradingState) -> int:
        """
        Determine the number of butterfly spreads tradeable based on available volume.
        For the low and high legs, use best ask volumes.
        For the mid leg, use best bid volume (divided by 2, since 2 mid contracts are needed per spread).
        """
        low_tuple = self.get_best_ask_with_vol(state, self.voucher_low)
        mid_tuple = self.get_best_bid_with_vol(state, self.voucher_mid)
        high_tuple = self.get_best_ask_with_vol(state, self.voucher_high)
        if low_tuple is None or mid_tuple is None or high_tuple is None:
            return 0
        _, vol_low = low_tuple
        _, vol_mid = mid_tuple
        _, vol_high = high_tuple
        max_spreads = min(self.trade_qty, vol_low, vol_mid // 2, vol_high)
        return max_spreads

    def get_orders(self, state: TradingState) -> List[Order]:
        """
        Compute the butterfly orders:
         - If cost < 0, enter a long butterfly:
              Buy 1 low, Sell 2 mid, Buy 1 high.
         - If cost > max_payoff, enter a short (reverse) butterfly:
              Sell 1 low, Buy 2 mid, Sell 1 high.
        Only trade if an arbitrage signal exists.
        """
        cost = self.compute_fly_cost(state)
        if cost is None:
            return []
        max_payoff = self.compute_max_payoff()
        # print(f"Fly Arbitrage: cost={cost}, max_payoff={max_payoff}")
        # Only enter if cost is either negative (net credit long butterfly) OR cost > max_payoff (reverse butterfly)
        if not (cost < 0 or cost > max_payoff):
            return []
        qty = self.get_trade_quantity(state)
        if qty <= 0:
            return []
        orders = []
        # Get relevant prices:
        low_tuple = self.get_best_ask_with_vol(state, self.voucher_low)
        mid_tuple = self.get_best_bid_with_vol(state, self.voucher_mid)
        high_tuple = self.get_best_ask_with_vol(state, self.voucher_high)
        if low_tuple is None or mid_tuple is None or high_tuple is None:
            return []
        best_ask_low, _ = low_tuple
        best_bid_mid, _ = mid_tuple
        best_ask_high, _ = high_tuple
        if cost < 0:
            # Long butterfly spread:
            # Buy low, Sell 2 mid, Buy high.
            orders.append(Order(self.voucher_low, int(round(best_ask_low)), qty))
            orders.append(Order(self.voucher_mid, int(round(best_bid_mid)), -2 * qty))
            orders.append(Order(self.voucher_high, int(round(best_ask_high)), qty))
        elif cost > max_payoff:
            # Short (reverse) butterfly spread:
            # Sell low, Buy 2 mid, Sell high.
            orders.append(Order(self.voucher_low, int(round(best_ask_low)), -qty))
            orders.append(Order(self.voucher_mid, int(round(best_bid_mid)), 2 * qty))
            orders.append(Order(self.voucher_high, int(round(best_ask_high)), -qty))
        return orders

    def act(self, state: TradingState) -> None:
        self.orders = self.get_orders(state)

    def run(self, state: TradingState) -> List[Order]:
        self.act(state)
        return self.orders

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

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

class Long9500OptionUnderpricedStrategy:
    """
    This strategy targets only the 9500 strike option ("VOLCANIC_ROCK_VOUCHER_9500").
    
    When the market's option price implies an extremely low volatility (i.e. the 
    inverted Black–Scholes implied volatility is below a small threshold, here < 0.001),
    the strategy buys the option to build a long position up to a specified limit.
    
    In this implementation there is no delta hedging. All that is considered is the option
    pricing — if the implied volatility is near zero, we assume that the option is underpriced 
    (relative to its intrinsic value) and buy.
    """
    
    def __init__(self, option_symbol: str, underlying_symbol: str, position_limit: int, sigma_assumed: float) -> None:
        self.symbol = option_symbol          # e.g., "VOLCANIC_ROCK_VOUCHER_9500"
        self.underlying = underlying_symbol   # e.g., "VOLCANIC_ROCK" (unused here)
        self.limit = position_limit           # target long option position, e.g., +200
        self.sigma_assumed = sigma_assumed    # historical volatility (e.g. 0.22) to use in the inversion
        self.orders: List[Order] = []         # orders to be submitted by the strategy

    @staticmethod
    def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
        """Compute the Black–Scholes call option price (using zero interest rate)."""
        if T <= 0:
            return max(S - K, 0)
        d1 = (math.log(S/K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm.cdf(d1) - K * norm.cdf(d2)
    
    def implied_volatility(self, market_price: float, S: float, K: float, T: float, r: float = 0.0) -> float:
        """
        Invert the Black–Scholes formula to compute implied volatility.
        If the option's market price is nearly equal to its intrinsic value,
        return 0.
        """
        intrinsic = max(S - K, 0)
        if market_price - intrinsic < 0.1:  
            return 0.0
        try:
            vol = brentq(lambda sigma: self.bs_call_price(S, K, T, sigma) - market_price,
                         1e-6, 3.0, xtol=1e-6)
            return vol
        except Exception:
            return 0.0
    
    def buy(self, symbol: str, price: int, quantity: int) -> None:
        self.orders.append(Order(symbol, price, quantity))
        
    def sell(self, symbol: str, price: int, quantity: int) -> None:
        self.orders.append(Order(symbol, price, -quantity))
    
    def act(self, state: TradingState) -> None:
        # Ensure the option order depth is available.
        if self.symbol not in state.order_depths:
            return
        
        # OPTION SIDE: Get the option mid price.
        option_depth: OrderDepth = state.order_depths[self.symbol]
        if not option_depth.buy_orders or not option_depth.sell_orders:
            return
        best_bid_option = max(option_depth.buy_orders.keys())
        best_ask_option = min(option_depth.sell_orders.keys())
        option_mid_price = (best_bid_option + best_ask_option) / 2
        
        # UNDERLYING SIDE: Also get an estimate of the underlying mid price (for pricing purposes).
        if self.underlying not in state.order_depths:
            return
        underlying_depth: OrderDepth = state.order_depths[self.underlying]
        if not underlying_depth.buy_orders or not underlying_depth.sell_orders:
            return
        best_bid_underlying = max(underlying_depth.buy_orders.keys())
        best_ask_underlying = min(underlying_depth.sell_orders.keys())
        underlying_mid_price = (best_bid_underlying + best_ask_underlying) / 2
        
        # Compute dynamic time-to-expiry.
        tte = get_tte(3, state.timestamp)
        
        # Calculate implied volatility via inversion.
        imp_vol = self.implied_volatility(option_mid_price, underlying_mid_price, 9500, tte)
        # print(f"IMPLIED VOL for {self.symbol}: {imp_vol}")
        
        # Only trade if the implied volatility is effectively near zero.
        if imp_vol > 0.001:
            return
        
        # Build long option position if under limit.
        current_option_pos = state.position.get(self.symbol, 0)
        target_option_pos = self.limit
        option_trade_qty = target_option_pos - current_option_pos
        
        if option_trade_qty > 0:
            # Buy at the best ask price.
            self.buy(self.symbol, best_ask_option, option_trade_qty)
    
    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.act(state)
        return self.orders
    
    def save(self) -> JSON:
        return {}
    
    def load(self, data: JSON) -> None:
        pass

###############################################################################
# Now we add a separate function to hedge "VOLCANIC_ROCK" based on all option positions.
###############################################################################

def hedge_volcano_rock(state: TradingState) -> List[Order]:
    """
    This function aggregates the net delta exposure from all VOLCANIC_ROCK options 
    and issues a hedge order on the underlying "VOLCANIC_ROCK" to neutralize that exposure.
    
    It uses the Black–Scholes delta formula with the historical volatility for each option.
    The mapping of option symbols to their strikes and assumed volatilities is defined below.
    """
    orders = []
    underlying_symbol = "VOLCANIC_ROCK"
    
    # Make sure the underlying order depth exists.
    if underlying_symbol not in state.order_depths:
        return orders
    
    # Determine the underlying mid price.
    underlying_depth: OrderDepth = state.order_depths[underlying_symbol]
    if not underlying_depth.buy_orders or not underlying_depth.sell_orders:
        return orders
    best_bid_underlying = max(underlying_depth.buy_orders.keys())
    best_ask_underlying = min(underlying_depth.sell_orders.keys())
    underlying_mid_price = (best_bid_underlying + best_ask_underlying) / 2
    
    # Define the option instruments to consider with their strikes and assumed volatilities.
    option_data = {
        "VOLCANIC_ROCK_VOUCHER_9500": {"strike": 9500, "sigma": 0.22},
        "VOLCANIC_ROCK_VOUCHER_9750": {"strike": 9750, "sigma": 0.18},
        "VOLCANIC_ROCK_VOUCHER_10000": {"strike": 10000, "sigma": 0.15},
        "VOLCANIC_ROCK_VOUCHER_10250": {"strike": 10250, "sigma": 0.155},
        "VOLCANIC_ROCK_VOUCHER_10500": {"strike": 10500, "sigma": 0.16}
    }
    
    # Use the same TTE for all options.
    tte = get_tte(3, state.timestamp)
    
    # Aggregate net delta from all option positions.
    total_option_delta = 0.0
    for option_sym, data in option_data.items():
        pos = state.position.get(option_sym, 0)
        if pos == 0:
            continue
        strike = data["strike"]
        sigma = data["sigma"]
        # Compute d1 = [ln(S/K) + 0.5*sigma^2*T] / (sigma*sqrt(T))
        if underlying_mid_price <= 0 or tte <= 0:
            continue
        d1 = (math.log(underlying_mid_price / strike) + 0.5 * (sigma ** 2) * tte) / (sigma * math.sqrt(tte))
        delta = norm.cdf(d1)
        total_option_delta += pos * delta
        print(f"Option {option_sym}: pos = {pos}, strike = {strike}, sigma = {sigma}, delta = {delta}")
    
    # The target underlying position is the negative of the aggregated option delta.
    target_underlying_pos = -total_option_delta
    current_underlying_pos = state.position.get(underlying_symbol, 0)
    hedge_qty = target_underlying_pos - current_underlying_pos
    print(f"Total aggregated option delta: {total_option_delta}")
    print(f"Target underlying pos: {target_underlying_pos}, current: {current_underlying_pos}, hedge qty: {hedge_qty}")
    
    # Create hedge order on the underlying.
    if hedge_qty < 0:
        orders.append(Order(underlying_symbol, best_bid_underlying, int(round(hedge_qty))))
    elif hedge_qty > 0:
        orders.append(Order(underlying_symbol, best_ask_underlying, int(round(hedge_qty))))
    return orders

            
class Trader:
    def __init__(self) -> None:
        limits = {
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

        self.strategies = {
            # "VOLCANIC_ROCK_VOUCHER_ARB1": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_9500",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_9750",
            #     trade_qty=200,
            #     execution_threshold=250  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB2": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_9500",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10000",
            #     trade_qty=200,
            #     execution_threshold=500  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB3": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_9500",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10250",
            #     trade_qty=200,
            #     execution_threshold=750  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB4": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_9500",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10500",
            #     trade_qty=200,
            #     execution_threshold=1000  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB5": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_9750",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10000",
            #     trade_qty=200,
            #     execution_threshold=250  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB6": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_9750",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10250",
            #     trade_qty=200,
            #     execution_threshold=500  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB7": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_9750",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10500",
            #     trade_qty=200,
            #     execution_threshold=750  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB8": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_10000",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10250",
            #     trade_qty=200,
            #     execution_threshold=250  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB9": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_10000",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10500",
            #     trade_qty=200,
            #     execution_threshold=500  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB10": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_10250",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10500",
            #     trade_qty=200,
            #     execution_threshold=250  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB11": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_10500",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10250",
            #     trade_qty=200,
            #     execution_threshold=0  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB12": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_10250",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_10000",
            #     trade_qty=200,
            #     execution_threshold=0  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB13": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_10000",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_9750",
            #     trade_qty=200,
            #     execution_threshold=0  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB14": VolcanicRockVoucherSpreadArbStrategy(
            #     voucher_lower="VOLCANIC_ROCK_VOUCHER_9750",
            #     voucher_higher="VOLCANIC_ROCK_VOUCHER_9500",
            #     trade_qty=200,
            #     execution_threshold=0  
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB15" : VolcanicRockVoucherFlyArbStrategy(
            #     voucher_low="VOLCANIC_ROCK_VOUCHER_9750",
            #     voucher_mid="VOLCANIC_ROCK_VOUCHER_10000",
            #     voucher_high="VOLCANIC_ROCK_VOUCHER_10250",
            #     strike_low=9750,
            #     strike_mid=10000,
            #     strike_high=10250,
            #     trade_qty=200
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB16" : VolcanicRockVoucherFlyArbStrategy(
            #     voucher_low="VOLCANIC_ROCK_VOUCHER_9750",
            #     voucher_mid="VOLCANIC_ROCK_VOUCHER_10000",
            #     voucher_high="VOLCANIC_ROCK_VOUCHER_10250",
            #     strike_low=9500,
            #     strike_mid=9750,
            #     strike_high=10000,
            #     trade_qty=200
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB17" : VolcanicRockVoucherFlyArbStrategy(
            #     voucher_low="VOLCANIC_ROCK_VOUCHER_9750",
            #     voucher_mid="VOLCANIC_ROCK_VOUCHER_10000",
            #     voucher_high="VOLCANIC_ROCK_VOUCHER_10250",
            #     strike_low=10000,
            #     strike_mid=10250,
            #     strike_high=10500,
            #     trade_qty=200
            # ),
            "VOLCANIC_ROCK_VOUCHER_ARB18" : Long9500OptionUnderpricedStrategy(
                option_symbol="VOLCANIC_ROCK_VOUCHER_9500",
                underlying_symbol="VOLCANIC_ROCK",
                position_limit=80,
                sigma_assumed=0.22
            ),
            "VOLCANIC_ROCK_VOUCHER_ARB19" : Long9500OptionUnderpricedStrategy(
                option_symbol="VOLCANIC_ROCK_VOUCHER_9750",
                underlying_symbol="VOLCANIC_ROCK",
                position_limit=80,
                sigma_assumed=0.18
            ),
            "VOLCANIC_ROCK_VOUCHER_ARB20" : Long9500OptionUnderpricedStrategy(
                option_symbol="VOLCANIC_ROCK_VOUCHER_10000",
                underlying_symbol="VOLCANIC_ROCK",
                position_limit=80,
                sigma_assumed=0.15
            ),
            "VOLCANIC_ROCK_VOUCHER_ARB21" : Long9500OptionUnderpricedStrategy(
                option_symbol="VOLCANIC_ROCK_VOUCHER_10250",
                underlying_symbol="VOLCANIC_ROCK",
                position_limit=80,
                sigma_assumed=0.155
            ),
            "VOLCANIC_ROCK_VOUCHER_ARB22" : Long9500OptionUnderpricedStrategy(
                option_symbol="VOLCANIC_ROCK_VOUCHER_10500",
                underlying_symbol="VOLCANIC_ROCK",
                position_limit=80,
                sigma_assumed=0.16
            )
            # "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            # "KELP": KelpStrategy("KELP", limits["KELP"]),
            # "CROISSANTS": BasketHedgeStrategy("CROISSANTS", limits["CROISSANTS"], {"PICNIC_BASKET1": 6, "PICNIC_BASKET2": 4}),
            # "JAMS": JamsSwingStrategy("JAMS", limits["JAMS"]),
            # "DJEMBES": BasketHedgeStrategy("DJEMBES", limits["DJEMBES"], {"PICNIC_BASKET1": 1}),
            # "BASKET_TRADING": BasketTradingStrategy(
            #     "BASKET_TRADING",
            #     pb1_position_limit=60,
            #     pb2_position_limit=100,
            #     d_position_limit=60,
            #     j_position_limit=350,
            #     c_position_limit=250,
            #     pb1_pb2_spread_avg=3.408,
            #     pb1_pb2_spread_std=93.506,
            #     pb1_pb2_z_thresh=0.75,
            #     pb1_pb2_exit_thesh=-1,
            #     pb1_pb2_timeframe=300,
            #     pb1_pb2_std_scaling=3,
            #     pb1_pb2_std_bandwidth=40
            # ),
            # "SQUID_INK": SquidInkDeviationStrategy("SQUID_INK",
            #                                        limits["SQUID_INK"],
            #                                        n_days=210,
            #                                        z_high=2.5,
            #                                        z_low=1.5,
            #                                        profit_margin=9,
            #                                        sell_off_ratio=0.25,
            #                                        entering_aggression=1.6,
            #                                        take_profit_aggression=0,
            #                                        clearing_aggression=1,
            #                                        high_hold_duration=15000,
            #                                        low_hold_duration=8000,
            #                                        low_trade_ratio=0.5),


        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        orders: dict[Symbol, List[Order]] = {}
        # Run individual strategies.
        for key, strategy in self.strategies.items():
            strategy.load(json.loads(state.traderData) if state.traderData != "" else {})
            strat_orders = strategy.run(state)
            for order in strat_orders:
                orders.setdefault(order.symbol, []).append(order)
        # Now add the separate underlying hedge for VOLCANIC_ROCK based on all option positions.
        hedge_orders = hedge_volcano_rock(state)
        for order in hedge_orders:
            orders.setdefault(order.symbol, []).append(order)
        
        trader_data = json.dumps({key: strategy.save() for key, strategy in self.strategies.items()}, separators=(",", ":"))
        # (Assume logger.flush is defined elsewhere.)
        conversions = 0
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data