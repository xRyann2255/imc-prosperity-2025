import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Optional, Tuple, TypeAlias, Dict
import math
from scipy.stats import norm
from scipy.optimize import brentq
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

        # Add non-zero orders to this strategy's list.
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
        Trade quantity is the number of butterfly spreads (legs' trade sizes are scaled accordingly).
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
        logger.print(f"Option {option_sym}: pos = {pos}, strike = {strike}, sigma = {sigma}, delta = {delta}")
    
    # The target underlying position is the negative of the aggregated option delta.
    target_underlying_pos = -total_option_delta
    current_underlying_pos = state.position.get(underlying_symbol, 0)
    hedge_qty = target_underlying_pos - current_underlying_pos
    logger.print(f"Total aggregated option delta: {total_option_delta}")
    logger.print(f"Target underlying pos: {target_underlying_pos}, current: {current_underlying_pos}, hedge qty: {hedge_qty}")
    
    # Create hedge order on the underlying.
    if hedge_qty < 0:
        orders.append(Order(underlying_symbol, best_bid_underlying, int(round(hedge_qty))))
    elif hedge_qty > 0:
        orders.append(Order(underlying_symbol, best_ask_underlying, int(round(hedge_qty))))
    return orders

#######################################################################################
# NEW CODE: Volatility-based option trading strategy
#######################################################################################

class VolcanicOptionVolatilityStrategy(Strategy):
    """
    Trading strategy for Volcanic Rock Options based on volatility mispricing.
    
    This strategy:
    1. Maintains a volatility smile curve.
    2. For each option, computes its moneyness and expected volatility from the smile.
    3. Compares the theoretical price with the market price.
    4. Trades when the price difference (normalized as a z-score) exceeds a threshold.
    """
    
    def __init__(self, 
                 symbol: str,  # Option symbol (e.g., "VOLCANIC_ROCK_VOUCHER_9500")
                 strike: int,  # Strike price
                 position_limit: int,  # Maximum position size
                 underlying_symbol: str = "VOLCANIC_ROCK",
                 base_volatility: float = 0.20,  # Base volatility estimate
                 entry_z_threshold: float = 2.0,  # z score threshold for entry signal
                 exit_z_threshold: float = 1.0,   # z score threshold for exit signal
                 expiration_days: float = 4.0,  # Time to expiration in trading days
                 trading_days_per_year: int = 252,  # Trading days per year
                 interest_rate: float = 0.0,  # Risk-free rate
                 smile_window_size: int = 1500,  # Number of timestamps to keep in volatility smile history
                 vol_history_size: int = 300,  # Number of volatility observations to keep for each strike
                 debug: bool = False  # Whether to print debug information
                 ):
        super().__init__(symbol, position_limit)
        self.strike = strike
        self.underlying_symbol = underlying_symbol
        self.base_volatility = base_volatility
        self.entry_z_threshold = entry_z_threshold
        self.exit_z_threshold = exit_z_threshold
        self.expiration_days = expiration_days
        self.trading_days_per_year = trading_days_per_year
        self.interest_rate = interest_rate
        self.smile_window_size = smile_window_size
        self.vol_history_size = vol_history_size
        self.debug = debug
        
        # Initialize data structures
        self.vol_smile_history = deque(maxlen=smile_window_size)  # Store past smile curves
        self.implied_vols = deque(maxlen=vol_history_size)  # Store past IVs for this option
        
        # New: History for price differences (spreads) for dynamic thresholding (#2)
        self.price_difference_history = deque(maxlen=vol_history_size)
        
        # New z-score thresholds for dynamic entry/exit signals

        
        # Last computed values
        self.last_iv = None
        self.last_theoretical_price = None
        self.last_market_price = None
        self.last_moneyness = None
    
    def bs_call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate Black–Scholes price for a call option"""
        r = self.interest_rate
        
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)  # Return intrinsic value if time or volatility is zero
        
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def bs_call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate the delta of a call option using Black–Scholes"""
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        
        d1 = (math.log(S/K) + (self.interest_rate + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return norm.cdf(d1)
    
    def implied_volatility(self, option_price: float, S: float, K: float, T: float) -> Optional[float]:
        """
        Calculate implied volatility using the bisection method.
        Returns None if the calculation fails.
        """
        intrinsic = max(S - K, 0)
        
        # If price is very close to intrinsic, return a very low IV
        if abs(option_price - intrinsic) < 0.1:
            return 0.001
        
        # If price is less than intrinsic (shouldn't happen in theory), return None
        if option_price < intrinsic:
            return None
        
        # Define the objective function (difference between BS price and market price)
        def objective(sigma):
            return self.bs_call_price(S, K, T, sigma) - option_price
        
        try:
            # Use bisection method to find IV (bounds between 0.001 and 5.0)
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return iv
        except Exception as e:
            if self.debug:
                logger.print(f"Failed to compute IV: {e}")
            return None
    
    def get_time_to_expiry(self, timestamp: int) -> float:
        """Calculate time to expiry in years based on current timestamp"""
        days_passed = timestamp / 1_000_000  # Convert to days (assuming 1M ticks = 1 day)
        days_remaining = max(0, self.expiration_days - days_passed)
        return days_remaining / self.trading_days_per_year
    
    def update_volatility_smile(self, state: TradingState) -> None:
        """
        Update the volatility smile based on current market data.
        This captures the implied volatility across different strikes.
        """
        # Ensure underlying price is available
        if self.underlying_symbol not in state.order_depths:
            return
        
        underlying_depth = state.order_depths[self.underlying_symbol]
        if not underlying_depth.buy_orders or not underlying_depth.sell_orders:
            return
        
        underlying_bid = max(underlying_depth.buy_orders.keys())
        underlying_ask = min(underlying_depth.sell_orders.keys())
        underlying_price = (underlying_bid + underlying_ask) / 2
        
        # Calculate time to expiry
        tte = self.get_time_to_expiry(state.timestamp)
        if tte <= 0:
            return  # Skip if we're past expiration
        
        # Mapping of strike prices to their corresponding symbols
        strike_symbols = {
            9500: "VOLCANIC_ROCK_VOUCHER_9500",
            9750: "VOLCANIC_ROCK_VOUCHER_9750",
            10000: "VOLCANIC_ROCK_VOUCHER_10000",
            10250: "VOLCANIC_ROCK_VOUCHER_10250",
            10500: "VOLCANIC_ROCK_VOUCHER_10500"
        }
        
        # Collect implied volatilities for each strike
        smile_points = []
        
        for strike, symbol in strike_symbols.items():
            if symbol not in state.order_depths:
                continue
                
            depth = state.order_depths[symbol]
            if not depth.buy_orders or not depth.sell_orders:
                continue
                
            option_bid = max(depth.buy_orders.keys())
            option_ask = min(depth.sell_orders.keys())
            option_price = (option_bid + option_ask) / 2
            
            # Calculate moneyness (log(S/K))
            moneyness = np.log(underlying_price / strike)
            
            # Calculate implied volatility
            iv = self.implied_volatility(option_price, underlying_price, strike, tte)
            
            # Skip invalid IVs
            if iv is None:
                continue
                
            # Store the moneyness and IV
            smile_points.append((moneyness, iv))
            
            # Store IV for this option if it's our symbol
            if symbol == self.symbol:
                self.last_iv = iv
                self.last_market_price = option_price
                self.last_moneyness = moneyness
                self.implied_vols.append(iv)
        
        # Sort by moneyness and store the smile curve
        if smile_points:
            smile_points.sort()
            self.vol_smile_history.append((state.timestamp, smile_points))
    
    def get_current_smile(self) -> List[Tuple[float, float]]:
        """Get the most recent volatility smile curve"""
        if not self.vol_smile_history:
            return []
        
        return self.vol_smile_history[-1][1]
    
    def interpolate_volatility(self, moneyness: float, lambda_=0.005) -> Optional[float]:
        """
        Estimate implied volatility at a given moneyness using
        exponentially-weighted regression over historical smile points.
        
        moneyness: The moneyness value for which to estimate IV.
        lambda_: Decay factor for historical smile points. 0.005 is a good starting point.
        
        """
        if not self.vol_smile_history:
            # Fallback: use historical IVs or a default small value
            return np.mean(self.implied_vols) if self.implied_vols else 0.20

        smile_points = []
        weights = []

        # Decay factor: controls how quickly older smiles are discounted
        # smaller = smoother; increase for faster adaptation

        for i, (_, smile) in enumerate(reversed(self.vol_smile_history)):
            decay = np.exp(-lambda_ * i)
            for m, iv in smile:
                smile_points.append((m, iv))
                weights.append(decay)

        if not smile_points:
            return np.mean(self.implied_vols) if self.implied_vols else 0.20

        x = np.array([m for m, _ in smile_points])
        y = np.array([v for _, v in smile_points])
        w = np.array(weights)

        try:
            # --- First-pass unweighted fit ---
            coeffs_initial = np.polyfit(x, y, deg=2)
            y_pred = np.polyval(coeffs_initial, x)

            # --- Residual z-score filtering ---
            residuals = y - y_pred
            res_mean = np.mean(residuals)
            res_std = np.std(residuals) + 1e-6
            z_scores = np.abs((residuals - res_mean) / res_std)

            # --- Keep only points where |z| <= 2.5 ---
            keep = z_scores <= 2.5
            x_filtered = x[keep]
            y_filtered = y[keep]
            w_filtered = w[keep]

            if len(x_filtered) < 5:
                return np.mean(self.implied_vols) if self.implied_vols else 0.20

            # --- Second-pass weighted fit ---
            coeffs_final = np.polyfit(x_filtered, y_filtered, 2)
            estimated_iv = np.polyval(coeffs_final, moneyness)

            if 0.001 <= estimated_iv <= 5.0:
                return estimated_iv
            else:
                return np.mean(self.implied_vols) if self.implied_vols else 0.20

        except Exception as e:
            if self.debug:
                logger.print(f"Smile interpolation failed: {e}")
            return np.mean(self.implied_vols) if self.implied_vols else 0.20
    
    def act(self, state: TradingState) -> None:
        """Execute the volatility-based trading strategy"""
        self.orders = []
        
        # Update the volatility smile
        self.update_volatility_smile(state)
        
        # Skip if market data is incomplete
        if (self.underlying_symbol not in state.order_depths or 
            self.symbol not in state.order_depths):
            return
        
        underlying_depth = state.order_depths[self.underlying_symbol]
        option_depth = state.order_depths[self.symbol]
        
        if (not underlying_depth.buy_orders or not underlying_depth.sell_orders or
            not option_depth.buy_orders or not option_depth.sell_orders):
            return
        
        # Get current prices
        underlying_bid = max(underlying_depth.buy_orders.keys())
        underlying_ask = min(underlying_depth.sell_orders.keys())
        underlying_price = (underlying_bid + underlying_ask) / 2
        
        option_bid = max(option_depth.buy_orders.keys())
        option_ask = min(option_depth.sell_orders.keys())
        option_price = (option_bid + option_ask) / 2
        
        # Calculate time to expiry
        tte = self.get_time_to_expiry(state.timestamp)
        if tte <= 0:
            return  # Skip if expired
        
        # Calculate moneyness and get IV from smile curve
        moneyness = np.log(underlying_price / self.strike)
        iv_from_smile = self.interpolate_volatility(moneyness)
        if iv_from_smile is None:
            iv_from_smile = self.base_volatility
        
        # Calculate theoretical price using IV from smile
        theoretical_price = self.bs_call_price(underlying_price, self.strike, tte, iv_from_smile)
        self.last_theoretical_price = theoretical_price
        
        # Calculate price difference (spread)
        price_diff = option_price - theoretical_price
        
        # Update the rolling history of price differences (#2)
        self.price_difference_history.append(price_diff)
        # Compute rolling standard deviation of the spread
        spread_std = np.std(np.array(list(self.price_difference_history)))
        # Avoid division by zero in z-score calculation:
        if spread_std < 1e-6:
            spread_std = 1e-6
        z_score = price_diff / spread_std
        logger.print(f"Price diff: {price_diff:.2f}, Z-score: {z_score:.2f}, Std: {spread_std:.4f}")
        # Get current position
        current_position = state.position.get(self.symbol, 0)
        
        # Determine trading action based on z-score instead of raw price difference
        if abs(z_score) > self.entry_z_threshold:
            if z_score > 0:  # Option is overpriced – sell it
                qty_to_sell = min(
                    self.limit + current_position,  # Remaining short capacity
                    abs(option_depth.buy_orders[option_bid])  # Available volume at best bid
                )
                if qty_to_sell > 0:
                    self.sell(option_bid, qty_to_sell)
            else:  # Option is underpriced – buy it
                qty_to_buy = min(
                    self.limit - current_position,  # Remaining long capacity
                    abs(option_depth.sell_orders[option_ask])  # Available volume at best ask
                )
                if qty_to_buy > 0:
                    self.buy(option_ask, qty_to_buy)
        
        # Handle exit trades when the z-score reverts to near zero
        elif current_position != 0 and abs(z_score) < self.exit_z_threshold:
            if current_position > 0:  # Long position – sell to exit
                self.sell(option_bid, current_position)
            else:  # Short position – buy to exit
                self.buy(option_ask, -current_position)
        
        if self.debug and self.orders:
            logger.print(f"[{self.symbol}] Generated orders: {self.orders}")
            logger.print(f"  Moneyness: {moneyness:.4f}, IV from smile: {iv_from_smile:.4f}")
            logger.print(f"  Market price: {option_price}, Theoretical: {theoretical_price:.2f}, Price Diff: {price_diff:.2f}, Z-score: {z_score:.2f}")
    
    def save(self) -> JSON:
        """Save strategy state for persistence"""
        return {
            "implied_vols": list(self.implied_vols),
            "price_difference_history": list(self.price_difference_history),
            "last_iv": self.last_iv,
            "last_theoretical_price": self.last_theoretical_price,
            "last_market_price": self.last_market_price,
            "last_moneyness": self.last_moneyness,
            # We don't save vol_smile_history as it can be large and will be regenerated
        }
    
    def load(self, data: JSON) -> None:
        """Load strategy state from persistence"""
        if data is None:
            return
            
        if "implied_vols" in data and isinstance(data["implied_vols"], list):
            self.implied_vols = deque(data["implied_vols"], maxlen=self.vol_history_size)
        if "price_difference_history" in data and isinstance(data["price_difference_history"], list):
            self.price_difference_history = deque(data["price_difference_history"], maxlen=self.smile_window_size)
        if "last_iv" in data:
            self.last_iv = data["last_iv"]
        if "last_theoretical_price" in data:
            self.last_theoretical_price = data["last_theoretical_price"]
        if "last_market_price" in data:
            self.last_market_price = data["last_market_price"]
        if "last_moneyness" in data:
            self.last_moneyness = data["last_moneyness"]

class VolatilityHedger:
    """
    Hedges option positions using delta-neutral hedging.
    Calculates the aggregate delta exposure of all options and 
    offsets it with a position in the underlying.
    """
    
    def __init__(self, 
                 underlying_symbol: str = "VOLCANIC_ROCK",
                 option_symbols: List[str] = None,
                 strikes: List[int] = None,
                 base_volatilities: List[float] = None,
                 expiration_days: float = 4.0,
                 trading_days_per_year: int = 252,
                 interest_rate: float = 0.0,
                 hedge_threshold: float = 5.0,  # Minimum delta change to trigger rehedging
                 max_hedge_size: int = 400,  # Maximum hedge position size
                 debug: bool = False):
        
        self.underlying_symbol = underlying_symbol
        
        # Default values if not provided
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
            
        if base_volatilities is None:
            base_volatilities = [0.22, 0.18, 0.15, 0.155, 0.16]
        
        self.option_symbols = option_symbols
        self.strikes = {sym: strike for sym, strike in zip(option_symbols, strikes)}
        self.base_volatilities = {sym: vol for sym, vol in zip(option_symbols, base_volatilities)}
        
        self.expiration_days = expiration_days
        self.trading_days_per_year = trading_days_per_year
        self.interest_rate = interest_rate
        self.hedge_threshold = hedge_threshold
        self.max_hedge_size = max_hedge_size
        self.debug = debug
        
        self.last_hedge_delta = 0
        self.orders = []
    
    def bs_call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate Black–Scholes price for a call option"""
        r = self.interest_rate
        
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)  # Return intrinsic value if time or volatility is zero
        
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def bs_call_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate the delta of a call option using Black–Scholes"""
        if T <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        
        d1 = (np.log(S/K) + (self.interest_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)
    
    def implied_volatility(self, option_price: float, S: float, K: float, T: float) -> Optional[float]:
        """
        Calculate implied volatility using the bisection method.
        Returns None if the calculation fails.
        """
        intrinsic = max(S - K, 0)
        
        # If price is very close to intrinsic, return a very low IV
        if abs(option_price - intrinsic) < 0.1:
            return 0.001
        
        # If price is less than intrinsic, return None
        if option_price < intrinsic:
            return None
        
        # Define the objective function
        def objective(sigma):
            return self.bs_call_price(S, K, T, sigma) - option_price
        
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return iv
        except Exception as e:
            if self.debug:
                logger.print(f"Failed to compute IV in hedger: {e}")
            return None
    
    def get_time_to_expiry(self, timestamp: int) -> float:
        """Calculate time to expiry in years based on current timestamp"""
        days_passed = timestamp / 1_000_000  # Convert to days (assuming 1M ticks = 1 day)
        days_remaining = max(0, self.expiration_days - days_passed)
        return days_remaining / self.trading_days_per_year
    
    def run(self, state: TradingState) -> List[Order]:
        """Calculate hedge position and generate orders"""
        self.orders = []
        
        # Ensure underlying price is available
        if self.underlying_symbol not in state.order_depths:
            return self.orders
        
        underlying_depth = state.order_depths[self.underlying_symbol]
        if not underlying_depth.buy_orders or not underlying_depth.sell_orders:
            return self.orders
        
        # Get underlying price
        underlying_bid = max(underlying_depth.buy_orders.keys())
        underlying_ask = min(underlying_depth.sell_orders.keys())
        underlying_price = (underlying_bid + underlying_ask) / 2
        
        # Calculate time to expiry
        tte = self.get_time_to_expiry(state.timestamp)
        if tte <= 0:
            return self.orders
        
        # Calculate total delta exposure from all options using dynamic volatility inputs
        total_delta = 0.0
        
        for symbol in self.option_symbols:
            position = state.position.get(symbol, 0)
            if position == 0:
                continue
                
            strike = self.strikes[symbol]
            # Default volatility to the base volatility
            volatility = self.base_volatilities[symbol]
            
            # If market data for the option is available, compute dynamic IV
            if symbol in state.order_depths:
                depth = state.order_depths[symbol]
                if depth.buy_orders and depth.sell_orders:
                    option_bid = max(depth.buy_orders.keys())
                    option_ask = min(depth.sell_orders.keys())
                    option_price = (option_bid + option_ask) / 2
                    dynamic_vol = self.implied_volatility(option_price, underlying_price, strike, tte)
                    if dynamic_vol is not None:
                        volatility = dynamic_vol  # Use dynamic volatility
                
            # Calculate option delta using the (possibly updated) volatility
            delta = self.bs_call_delta(underlying_price, strike, tte, volatility)
            option_delta = position * delta
            total_delta += option_delta
            
            if self.debug:
                logger.print(f"[{symbol}] Position: {position}, Strike: {strike}, Vol: {volatility:.4f}, Delta: {delta:.4f}")
                logger.print(f"  Contribution to total delta: {option_delta:.4f}")
        
        # Calculate required hedge position (negative of total delta)
        target_hedge = -round(total_delta)
        
        # Limit the hedge size
        target_hedge = max(-self.max_hedge_size, min(self.max_hedge_size, target_hedge))
        
        # Get current underlying position
        current_hedge = state.position.get(self.underlying_symbol, 0)
        
        # Determine if rehedging is needed
        delta_change = abs(target_hedge - current_hedge)
        
        if delta_change >= self.hedge_threshold:
            order_qty = target_hedge - current_hedge
            
            if order_qty > 0:  # Buy underlying
                self.orders.append(Order(self.underlying_symbol, underlying_ask, order_qty))
            elif order_qty < 0:  # Sell underlying
                self.orders.append(Order(self.underlying_symbol, underlying_bid, order_qty))
            
            self.last_hedge_delta = target_hedge
            
            if self.debug:
                logger.print(f"[HEDGE] Total delta: {total_delta:.4f}, Target hedge: {target_hedge}")
                logger.print(f"  Current: {current_hedge}, Order: {order_qty}")
        
        return self.orders
    
    def save(self) -> JSON:
        return {"last_hedge_delta": self.last_hedge_delta}
    
    def load(self, data: JSON) -> None:
        if data is not None and "last_hedge_delta" in data:
            self.last_hedge_delta = data["last_hedge_delta"]
            
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

        # Initialize strategies with the new volatility-based options strategies
        self.strategies = {
            # Volatility-based option strategies
            # "VOLCANIC_ROCK_VOUCHER_9500_VOL": VolcanicOptionVolatilityStrategy(
            #     symbol="VOLCANIC_ROCK_VOUCHER_9500",
            #     strike=9500,
            #     position_limit=80,
            #     base_volatility=0.22,
            #     entry_z_threshold=2.0,  # Price difference threshold for entry
            #     exit_z_threshold=1.0,    # Price difference threshold for exit
            #     expiration_days=4.0,   # 4 trading days to expiration
            #     trading_days_per_year=252,
            #     interest_rate=0.0
            # ),
            # "VOLCANIC_ROCK_VOUCHER_9750_VOL": VolcanicOptionVolatilityStrategy(
            #     symbol="VOLCANIC_ROCK_VOUCHER_9750",
            #     strike=9750,
            #     position_limit=80,
            #     base_volatility=0.18,
            #     entry_z_threshold=2.0,
            #     exit_z_threshold=1.0,
            #     expiration_days=4.0,
            #     trading_days_per_year=252,
            #     interest_rate=0.0
            # ),
            "VOLCANIC_ROCK_VOUCHER_10000_VOL": VolcanicOptionVolatilityStrategy(
                symbol="VOLCANIC_ROCK_VOUCHER_10000",
                strike=10000,
                position_limit=80,
                base_volatility=0.15,
                entry_z_threshold=1.5, # CHANGE THESE TWO ESSENTIALLY
                exit_z_threshold=0.8, # CHANGE THESE TWO ESSENTIALLY
                expiration_days=4.0,
                trading_days_per_year=252,
                interest_rate=0.0
            ),
            "VOLCANIC_ROCK_VOUCHER_10250_VOL": VolcanicOptionVolatilityStrategy(
                symbol="VOLCANIC_ROCK_VOUCHER_10250",
                strike=10250,
                position_limit=80,
                base_volatility=0.155,
                entry_z_threshold=2.0,
                exit_z_threshold=1.0,
                expiration_days=4.0,
                trading_days_per_year=252,
                interest_rate=0.0
            ),
            "VOLCANIC_ROCK_VOUCHER_10500_VOL": VolcanicOptionVolatilityStrategy(
                symbol="VOLCANIC_ROCK_VOUCHER_10500",
                strike=10500,
                position_limit=80,
                base_volatility=0.16,
                entry_z_threshold=2.0,
                exit_z_threshold=1.0,
                expiration_days=4.0,
                trading_days_per_year=252,
                interest_rate=0.0
            ),
            
            # Delta hedger for the underlying
            "VOLCANIC_ROCK_HEDGER": VolatilityHedger(
                underlying_symbol="VOLCANIC_ROCK",
                option_symbols=[
                    "VOLCANIC_ROCK_VOUCHER_9500",
                    "VOLCANIC_ROCK_VOUCHER_9750",
                    "VOLCANIC_ROCK_VOUCHER_10000",
                    "VOLCANIC_ROCK_VOUCHER_10250",
                    "VOLCANIC_ROCK_VOUCHER_10500"
                ],
                strikes=[9500, 9750, 10000, 10250, 10500],
                base_volatilities=[0.22, 0.18, 0.15, 0.155, 0.16],
                expiration_days=4.0,
                trading_days_per_year=252,
                interest_rate=0.0,
                hedge_threshold=5.0,  # Minimum delta change to trigger rehedging
                max_hedge_size=400    # Maximum hedge position size
            ),
            
            # You can keep your previous strategies if desired, or comment them out
            # to focus only on the volatility-based strategies
            
            # "VOLCANIC_ROCK_VOUCHER_ARB18" : Long9500OptionUnderpricedStrategy(
            #     option_symbol="VOLCANIC_ROCK_VOUCHER_9500",
            #     underlying_symbol="VOLCANIC_ROCK",
            #     position_limit=80,
            #     sigma_assumed=0.22
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB19" : Long9500OptionUnderpricedStrategy(
            #     option_symbol="VOLCANIC_ROCK_VOUCHER_9750",
            #     underlying_symbol="VOLCANIC_ROCK",
            #     position_limit=80,
            #     sigma_assumed=0.18
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB20" : Long9500OptionUnderpricedStrategy(
            #     option_symbol="VOLCANIC_ROCK_VOUCHER_10000",
            #     underlying_symbol="VOLCANIC_ROCK",
            #     position_limit=80,
            #     sigma_assumed=0.15
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB21" : Long9500OptionUnderpricedStrategy(
            #     option_symbol="VOLCANIC_ROCK_VOUCHER_10250",
            #     underlying_symbol="VOLCANIC_ROCK",
            #     position_limit=80,
            #     sigma_assumed=0.155
            # ),
            # "VOLCANIC_ROCK_VOUCHER_ARB22" : Long9500OptionUnderpricedStrategy(
            #     option_symbol="VOLCANIC_ROCK_VOUCHER_10500",
            #     underlying_symbol="VOLCANIC_ROCK",
            #     position_limit=80,
            #     sigma_assumed=0.16
            # )
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        orders: dict[Symbol, List[Order]] = {}
        
        # Load trader data if available
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
                for key, strategy in self.strategies.items():
                    if key in trader_data:
                        strategy.load(trader_data[key])
            except Exception as e:
                logger.print(f"Error loading trader data: {e}")
        
        # Run individual strategies
        for key, strategy in self.strategies.items():
            try:
                strat_orders = strategy.run(state)
                for order in strat_orders:
                    if order.symbol not in orders:
                        orders[order.symbol] = []
                    orders[order.symbol].append(order)
            except Exception as e:
                logger.print(f"Error running strategy {key}: {e}")
        
        # Save trader data for persistence
        trader_data = {}
        for key, strategy in self.strategies.items():
            try:
                trader_data[key] = strategy.save()
            except Exception as e:
                logger.print(f"Error saving strategy {key}: {e}")
        
        trader_data_json = json.dumps(trader_data, separators=(",", ":"))
        
        # No conversions in this implementation
        conversions = 0
        
        # Log information before returning
        logger.flush(state, orders, conversions, trader_data_json)
        
        return orders, conversions, trader_data_json