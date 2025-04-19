import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias
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

    def load(self, data: JSON) -> None:
        self.window = deque(data)

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

class PicnicBasketSpreadStrategy(Strategy):
    def __init__(
        self,
        symbol: str,
        limit: int,
        synthetic_components: list[Symbol],
        component_quantities: list[int],
        default_spread_mean: float,
        strong_entry_zscore: float,
        weak_entry_zscore: float,
        exit_zscore: float,
        spread_std_window: int,
        strong_target: int,
        weak_target: int
    ) -> None:
        """
        Parameters:
          - symbol: The basket product (e.g. "PICNIC_BASKET1")
          - limit: Maximum position limit.
          - synthetic_components: List of symbols for underlying components.
          - component_quantities: Quantities corresponding to each component.
          - default_spread_mean: The expected (historical) mean spread.
          - strong_entry_zscore: The z–score threshold for a strong signal.
          - weak_entry_zscore: The z–score threshold for a weak signal.
          - exit_zscore: If |z–score| falls below this, exit any open position.
          - spread_std_window: Number of recent spread observations for volatility estimation.
          - strong_target: Absolute target position for a strong signal.
          - weak_target: Absolute target position for a weak signal.
        Note: A positive target implies a long position; a negative target implies a short position.
              The direction is chosen based on the sign of the z–score.
        """
        super().__init__(symbol, limit)
        self.synthetic_components = synthetic_components
        self.component_quantities = component_quantities
        self.default_spread_mean = default_spread_mean
        self.strong_entry_zscore = strong_entry_zscore
        self.weak_entry_zscore = weak_entry_zscore
        self.exit_zscore = exit_zscore
        self.spread_std_window = spread_std_window
        self.strong_target = strong_target
        self.weak_target = weak_target
        self.spread_history = deque(maxlen=spread_std_window)
    
    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        """
        Compute the mid–price for the given symbol using the order depth.
        Uses the highest-volume buy order and lowest-volume sell order.
        """
        order_depth = state.order_depths[symbol]
        # Sort buy and sell orders by price, using the order volume as tie-breaker.
        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda tup: tup[1], reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda tup: tup[1])
        # Use the most popular prices.
        popular_buy_price = buy_orders[0][0] if buy_orders else 0.0
        popular_sell_price = sell_orders[0][0] if sell_orders else 0.0
        return (popular_buy_price + popular_sell_price) / 2

    def execute_spread_orders(self, target_position: int, current_position: int, order_depths: dict[Symbol, OrderDepth]) -> None:
        """
        Places orders to adjust the basket's position toward the target.
        Uses the best bid/ask from the basket's order book.
        """
        basket_order_depth = order_depths.get(self.symbol)
        if not basket_order_depth:
            return

        position_diff = target_position - current_position
        if position_diff > 0:
            # To increase a long position, buy at the best ask.
            if basket_order_depth.sell_orders:
                best_ask = min(basket_order_depth.sell_orders.keys())
            else:
                best_ask = self.get_mid_price(state, self.symbol)  # fallback
            self.buy(int(round(best_ask)), position_diff)
        elif position_diff < 0:
            # To increase a short position, sell at the best bid.
            if basket_order_depth.buy_orders:
                best_bid = max(basket_order_depth.buy_orders.keys())
            else:
                best_bid = self.get_mid_price(state, self.symbol)  # fallback
            self.sell(int(round(best_bid)), abs(position_diff))
    
    def act(self, state: TradingState) -> None:
        """
        Implements the combined adaptive modified z–score strategy.
        Spread is now calculated as:
        
            spread = basket_mid – (Σ (component_mid_price * weight))
        
        where basket_mid is for self.symbol and the synthetic price is the weighted sum
        of the underlying components. The z–score is then computed relative to a fixed mean (or tuned mean)
        and the rolling standard deviation.
        """
        if self.symbol not in state.order_depths:
            return

        # Compute basket mid–price.
        basket_mid = self.get_mid_price(state, self.symbol)
        # Compute synthetic price as weighted sum.
        synthetic_mid = 0.0
        for comp, qty in zip(self.synthetic_components, self.component_quantities):
            if comp in state.order_depths:
                synthetic_mid += self.get_mid_price(state, comp) * qty
        
        # Calculate spread.
        spread = basket_mid - synthetic_mid
        
        # Update the rolling spread history.
        self.spread_history.append(spread)
        if len(self.spread_history) < self.spread_std_window:
            return
        
        spread_std = np.std(self.spread_history)
        if spread_std == 0:
            return

        # Compute the modified z–score.
        zscore = (spread - self.default_spread_mean) / spread_std
        current_position = state.position.get(self.symbol, 0)

        # Exit condition: if the signal has weakened.
        if abs(zscore) < self.exit_zscore:
            if current_position != 0:
                self.execute_spread_orders(0, current_position, state.order_depths)
            return

        # Determine target position based on the strength and sign of the signal.
        if zscore > 0:
            # Positive zscore => basket is expensive; signal for short.
            if zscore >= self.strong_entry_zscore:
                target = -abs(self.strong_target)
            elif zscore >= self.weak_entry_zscore:
                target = -abs(self.weak_target)
            else:
                target = current_position
        else:
            # Negative zscore => basket is cheap; signal for long.
            if abs(zscore) >= self.strong_entry_zscore:
                target = abs(self.strong_target)
            elif abs(zscore) >= self.weak_entry_zscore:
                target = abs(self.weak_target)
            else:
                target = current_position

        # Additional exit: if current position exists and its sign opposes the new target, exit first.
        if current_position != 0 and (current_position * target < 0):
            self.execute_spread_orders(0, current_position, state.order_depths)
            return

        if target is not None and current_position != target:
            self.execute_spread_orders(target, current_position, state.order_depths)

    def save(self) -> JSON:
        return list(self.spread_history)

    def load(self, data: JSON) -> None:
        if data is not None:
            self.spread_history = deque(data, maxlen=self.spread_std_window)


class Trader:
    
    def __init__(self, args=None) -> None:
        if args is None:
            class DummyArgs:
                pass
            dummy = DummyArgs()
            # Set parameters directly as fixed default values.
            dummy.z_high = 2.4
            dummy.z_low = 1.3
            dummy.n_days = 150
            dummy.profit_margin = 9
            dummy.sell_off_ratio = 0.15
            dummy.low_trade_ratio = 0.6
            dummy.entering_aggression = 2
            dummy.take_profit_aggression = 0
            dummy.clearing_aggression = 1
            dummy.high_hold_duration = 15000
            dummy.low_hold_duration = 5000
            dummy.symbol = "SQUID_INK"
            dummy.limit = 50
            args = dummy

        self.args = args
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANT": 250,
            "JAM": 350,
            "DJEMBE": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }
        self.strategies = {
            # args.symbol: SquidInkDeviationStrategy(
            #     args.symbol,
            #     limits[args.symbol],
            #     n_days=args.n_days,
            #     z_low=args.z_low,
            #     z_high=args.z_high,
            #     profit_margin=args.profit_margin,
            #     sell_off_ratio=args.sell_off_ratio,
            #     low_trade_ratio=args.low_trade_ratio,
            #     entering_aggression=args.entering_aggression,
            #     take_profit_aggression=args.take_profit_aggression,
            #     clearing_aggression=args.clearing_aggression,
            #     high_hold_duration=args.high_hold_duration,
            #     low_hold_duration=args.low_hold_duration
            # ),
            # "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            # "KELP": KelpStrategy("KELP", limits["KELP"]),
            # "CROISSANT": SquidInkDeviationStrategy("CROISSANT", limits["CROISSANT"], args.n_days,
            #                                          z_high=args.z_high, z_low=args.z_low, profit_margin=args.profit_margin,
            #                                          sell_off_ratio=args.sell_off_ratio, entering_aggression=args.entering_aggression,
            #                                          take_profit_aggression=args.take_profit_aggression,
            #                                          clearing_aggression=args.clearing_aggression,
            #                                          high_hold_duration=args.high_hold_duration,
            #                                          low_hold_duration=args.low_hold_duration,
            #                                          low_trade_ratio=args.low_trade_ratio),
            # "JAM": SquidInkDeviationStrategy("JAM", limits["JAM"], args.n_days,
            #                                   z_high=args.z_high, z_low=args.z_low, profit_margin=args.profit_margin,
            #                                   sell_off_ratio=args.sell_off_ratio, entering_aggression=args.entering_aggression,
            #                                   take_profit_aggression=args.take_profit_aggression,
            #                                   clearing_aggression=args.clearing_aggression,
            #                                   high_hold_duration=args.high_hold_duration,
            #                                   low_hold_duration=args.low_hold_duration,
            #                                   low_trade_ratio=args.low_trade_ratio),
            # "DJEMBE": SquidInkDeviationStrategy("DJEMBE", limits["DJEMBE"], args.n_days,
            #                                      z_high=args.z_high, z_low=args.z_low, profit_margin=args.profit_margin,
            #                                      sell_off_ratio=args.sell_off_ratio, entering_aggression=args.entering_aggression,
            #                                      take_profit_aggression=args.take_profit_aggression,
            #                                      clearing_aggression=args.clearing_aggression,
            #                                      high_hold_duration=args.high_hold_duration,
            #                                      low_hold_duration=args.low_hold_duration,
            #                                      low_trade_ratio=args.low_trade_ratio),
            "PICNIC_BASKET1": PicnicBasketSpreadStrategy(
                symbol="PICNIC_BASKET1",
                limit=limits["PICNIC_BASKET1"],
                synthetic_components=["CROISSANTS", "JAMS", "DJEMBES"],
                component_quantities=[6, 3, 1],
                default_spread_mean=-48.76,
                strong_entry_zscore=1.6,
                weak_entry_zscore=1.8,
                exit_zscore=0.4,
                spread_std_window=20,
                strong_target=60,
                weak_target=30
            ),
            # "PICNIC_BASKET2": PicnicBasketSpreadStrategy(
            #     symbol="PICNIC_BASKET2",
            #     limit=limits["PICNIC_BASKET2"],
            #     synthetic_components=["CROISSANTS", "JAMS"],
            #     component_quantities=[4, 2],
            #     default_spread_mean=-30.24,
            #     strong_entry_zscore=1.8,
            #     weak_entry_zscore=1.2,
            #     exit_zscore=0.4,
            #     spread_std_window=20,
            #     strong_target=100,
            #     weak_target=40
            # ),
            
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        logger.print(state.position)
        conversions = 0
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))
            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)
            new_trader_data[symbol] = strategy.save()
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data
