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
        # print(self.trader_data)
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

    # def save(self) -> Any:
    #     return self.trader_data

    # def load(self, data: Any) -> None:
    #     if data is not None:
    #         self.trader_data = data

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

# Create a single global instance.
global_vol_smile = GlobalVolSmile(window_size=500)

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
            "VOLCANIC_ROCK_VOUCHER_9500": 1.8,
            "VOLCANIC_ROCK_VOUCHER_9750": 1.8,
            "VOLCANIC_ROCK_VOUCHER_10000": 1.8,
            "VOLCANIC_ROCK_VOUCHER_10250": 1.8,
            "VOLCANIC_ROCK_VOUCHER_10500": 1.8,
            "VOLCANIC_ROCK": 1.8,
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
            # "VOLCANIC_ROCK_VOUCHER_10250_VOL": VolcanicOptionVolatilityStrategy(
            #     symbol="VOLCANIC_ROCK_VOUCHER_10250",
            #     strike=10250,
            #     position_limit=80,
            #     underlying_symbol= "VOLCANIC_ROCK",
            #     window_size=500,
            #     entry_threshold=1.5,
            #     exit_threshold=0.5,
            #     expiration_days=5.0,
            #     trading_days_per_year=365,
            #     interest_rate=0.0
            # ),
            # "VOLCANIC_ROCK_SPREAD_ARB" : VolSmSpreadArbStrategyDynamic(
            #     asset_a="VOLCANIC_ROCK_VOUCHER_9500",
            #     asset_b="VOLCANIC_ROCK_VOUCHER_10500",
            #     trade_qty=200,
            #     spread_threshold=0.2,
            #     exit_threshold=0.1
            # ),
            # "VOLCANIC_ROCK_MOMENTUM" : IVMomentumEmaStrategy(
            #     symbol = "VOLCANIC_ROCK_VOUCHER_10000", 
            #     position_limit=200,
            #     underlying_symbol="VOLCANIC_ROCK",
            #     iv_threshold=0.005,
            #     hold_period=15000,
            #     trade_size=200, 
            #     ema_window=15
            # ),
            # "VOLCANIC_ROCK_HEDGER": VolatilityHedger(
            #     underlying_symbol="VOLCANIC_ROCK",
            #     option_symbols=[
            #         "VOLCANIC_ROCK_VOUCHER_9500",
            #         "VOLCANIC_ROCK_VOUCHER_9750",
            #         "VOLCANIC_ROCK_VOUCHER_10000",
            #         "VOLCANIC_ROCK_VOUCHER_10250",
            #         "VOLCANIC_ROCK_VOUCHER_10500"
            #     ],
            #     strikes=[9500, 9750, 10000, 10250, 10500],
            #     expiration_days=5.0,
            #     trading_days_per_year=365,
            #     interest_rate=0.0,
            #     hedge_threshold=30.0,  # Minimum delta change to trigger rehedging
            #     max_hedge_size=400    # Maximum hedge position size
            # ),
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", self.limits["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", self.limits["KELP"]),
            # "CROISSANTS": BasketHedgeStrategy("CROISSANTS", limits["CROISSANTS"], {"PICNIC_BASKET1": 6, "PICNIC_BASKET2": 4}),
            "JAMS": JamsSwingStrategy("JAMS", self.limits["JAMS"]),
            # "DJEMBES": BasketHedgeStrategy("DJEMBES", limits["DJEMBES"], {"PICNIC_BASKET1": 1}),
            "BASKET_TRADING": BasketTradingStrategy(
                "BASKET_TRADING",
                pb1_position_limit=60,
                pb2_position_limit=100,
                d_position_limit=60,
                j_position_limit=350,
                c_position_limit=250,

                pb1_pb2_spread_avg=3.408,
                pb1_pb2_spread_std=93.506,
                pb1_pb2_z_thresh=0.7, 
                pb1_pb2_exit_thesh=-2,
                pb1_pb2_timeframe=200,
                pb1_pb2_std_scaling=4,
                pb1_pb2_std_bandwidth=60
            ),
            "SQUID_INK": SquidInkDeviationStrategy("SQUID_INK",
                                                   self.limits["SQUID_INK"],
                                                   n_days=210,
                                                   z_high=2.5,
                                                   z_low=1.5,
                                                   profit_margin=9,
                                                   sell_off_ratio=0.25,
                                                   entering_aggression=1.6,
                                                   take_profit_aggression=0,
                                                   clearing_aggression=1,
                                                   high_hold_duration=15000,
                                                   low_hold_duration=8000,
                                                   low_trade_ratio=0.5),


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


    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        orders: dict[Symbol, List[Order]] = {}
        # self.update_global_vol_smile(state)
        # Run individual strategies.
        for key, strategy in self.strategies.items():
            strategy.load(json.loads(state.traderData) if state.traderData != "" else {})
            strat_orders = strategy.run(state)
            for order in strat_orders:
                orders.setdefault(order.symbol, []).append(order)
        orders["VOLCANIC_ROCK"]=self.update_volcanic_rock_history(state)
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

        trader_data = json.dumps({key: strategy.save() for key, strategy in self.strategies.items()}, separators=(",", ":"))
        # (Assume logger.flush is defined elsewhere.)
        conversions = 0
        # position = state.position.get("VOLCANIC_ROCK_VOUCHER_10250", 0)
        # orders['VOLCANIC_ROCK_VOUCHER_10250'] = [Order("VOLCANIC_ROCK_VOUCHER_10250", 500, 200 - position)]
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data