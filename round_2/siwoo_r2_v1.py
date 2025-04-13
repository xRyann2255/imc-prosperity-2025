import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, TypeAlias
import math

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
        # Kalman filter initialization:
        self.true_value_estimate = None    # The estimated fair value
        self.error_cov = 1.0               # Initial error covariance
        self.process_variance = 2.0        # How much we expect the true value to change
        self.measurement_variance = 4    # How noisy the measurement is

    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        # Sort the order book.
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        
        # Determine a raw measurement as the average of the popular bid and ask prices.
        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = max(sell_orders, key=lambda tup: tup[1])[0]
        measurement = (popular_buy_price + popular_sell_price) / 2

        # Initialize filter state if needed.
        if self.true_value_estimate is None:
            self.true_value_estimate = measurement

        # Kalman prediction update (assume no change in state, but error increases).
        self.error_cov += self.process_variance

        # Kalman gain calculation.
        kalman_gain = self.error_cov / (self.error_cov + self.measurement_variance)

        # Kalman update: update the estimated true value using the measurement.
        self.true_value_estimate += kalman_gain * (measurement - self.true_value_estimate)

        # Update error covariance.
        self.error_cov = (1 - kalman_gain) * self.error_cov

        # Avellaneda–Stoikov position adjustment:
        current_position = state.position.get(self.symbol, 0)
        risk_aversion = 1.5  # Risk-aversion coefficient.
        T = 0.01           # Effective time constant (adjust as needed).
        adjustment = - current_position * risk_aversion * self.measurement_variance * T / 2

        return round(self.true_value_estimate + adjustment)
    
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
                
class BasketDeviationStrategy(Strategy):
    def __init__(self, basket_symbol: str, component_weights: dict, limit: int, 
                 window_size: int = 45, z_threshold: float = 2.0) -> None:
        super().__init__(basket_symbol, limit)
        self.component_weights = component_weights  # e.g., {"CROISSANTS":6, "JAM":3, "DJEMBES":1}
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.spread_history = deque(maxlen=window_size)
    
    def compute_market_mid(self, order_depth: OrderDepth) -> float:
        # Compute the mid price from basket's order book.
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2.0
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        else:
            return 0.0
    
    def compute_synthetic_mid(self, state: TradingState) -> float:
        # Compute a synthetic mid price by summing the weighted mid prices of each component.
        total_bid = 0.0
        total_ask = 0.0
        valid_bid = True
        valid_ask = True
        for comp, weight in self.component_weights.items():
            if comp in state.order_depths:
                od = state.order_depths[comp]
                if od.buy_orders:
                    comp_bid = max(od.buy_orders.keys())
                else:
                    valid_bid = False
                    comp_bid = 0
                if od.sell_orders:
                    comp_ask = min(od.sell_orders.keys())
                else:
                    valid_ask = False
                    comp_ask = 0
                total_bid += weight * comp_bid
                total_ask += weight * comp_ask
            else:
                valid_bid = False
                valid_ask = False
        if valid_bid and valid_ask:
            return (total_bid + total_ask) / 2.0
        elif valid_bid:
            return total_bid
        elif valid_ask:
            return total_ask
        else:
            return 0.0

    def get_true_value(self, state: TradingState) -> int:
        # For the basket, we use the synthetic mid as our base “true value.”
        synthetic_mid = self.compute_synthetic_mid(state)
        return round(synthetic_mid)
    
    def act(self, state: TradingState) -> None:
        # Ensure the basket order depth is available.
        if self.symbol not in state.order_depths:
            return

        basket_od = state.order_depths[self.symbol]
        basket_mid = self.compute_market_mid(basket_od)
        synthetic_mid = self.compute_synthetic_mid(state)
        
        # Compute the spread between the market basket and its synthetic value.
        spread = basket_mid - synthetic_mid
        self.spread_history.append(spread)
        
        # If we have enough history, compute z-score for the spread.
        if len(self.spread_history) >= self.window_size:
            mean_spread = sum(self.spread_history) / len(self.spread_history)
            std_spread = (sum((s - mean_spread) ** 2 for s in self.spread_history) / len(self.spread_history)) ** 0.5
            zscore = (spread - mean_spread) / std_spread if std_spread > 0 else 0
        else:
            zscore = 0
        
        # Base true value is the synthetic value.
        true_value = round(synthetic_mid)
        # Adjust the true value depending on the z-score:
        # If the basket is overpriced (positive zscore), be more aggressive on the sell side.
        if zscore > self.z_threshold:
            true_value -= abs(zscore)  # adjust downward by zscore amount
        elif zscore < -self.z_threshold:
            true_value += abs(zscore)  # adjust upward for an underpriced basket

        # Now apply a basic market-making approach similar to MarketMakingStrategy.
        order_depth = basket_od
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position
        
        # Process sell orders for buying.
        for price, volume in sell_orders:
            if to_buy > 0 and price <= true_value:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity
        
        # Process buy orders for selling.
        for price, volume in buy_orders:
            if to_sell > 0 and price >= true_value:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity
        
        # Fallback: if orders remain, add orders at adjusted prices.
        if to_buy > 0:
            if buy_orders:
                popular_buy = max(buy_orders, key=lambda tup: tup[1])[0]
                price = min(true_value, popular_buy + 1)
            else:
                price = true_value
            self.buy(price, to_buy)
        if to_sell > 0:
            if sell_orders:
                popular_sell = min(sell_orders, key=lambda tup: tup[1])[0]
                price = max(true_value, popular_sell - 1)
            else:
                price = true_value
            self.sell(price, to_sell)
    
    def save(self) -> Any:
        return list(self.spread_history)
    
    def load(self, data: Any) -> None:
        self.spread_history = deque(data, maxlen=self.window_size)

class Trader:
    
    def __init__(self, args=None) -> None:
        if args is None:
            class DummyArgs:
                pass
            dummy = DummyArgs()
            # Set parameters directly as fixed default values.
            dummy.z_high = 2.5
            dummy.z_low = 1.5
            dummy.n_days = 200
            dummy.profit_margin = 9
            dummy.sell_off_ratio = 0.15
            dummy.low_trade_ratio = 0.6
            dummy.entering_aggression = 2
            dummy.take_profit_aggression = 0
            dummy.clearing_aggression = 1.5
            dummy.high_hold_duration = 15000
            dummy.low_hold_duration = 8000
            dummy.symbol = "SQUID_INK"
            dummy.limit = 50
            args = dummy

        self.args = args
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100
        }
        self.strategies = {
            args.symbol: SquidInkDeviationStrategy(
                args.symbol,
                limits[args.symbol],
                n_days=args.n_days,
                z_low=args.z_low,
                z_high=args.z_high,
                profit_margin=args.profit_margin,
                sell_off_ratio=args.sell_off_ratio,
                low_trade_ratio=args.low_trade_ratio,
                entering_aggression=args.entering_aggression,
                take_profit_aggression=args.take_profit_aggression,
                clearing_aggression=args.clearing_aggression,
                high_hold_duration=args.high_hold_duration,
                low_hold_duration=args.low_hold_duration
            ),
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", limits["KELP"]),
            "CROISSANT": SquidInkDeviationStrategy("CROISSANTS", limits["CROISSANTS"], args.n_days,
                                                     z_high=args.z_high, z_low=args.z_low, profit_margin=args.profit_margin,
                                                     sell_off_ratio=args.sell_off_ratio, entering_aggression=args.entering_aggression,
                                                     take_profit_aggression=args.take_profit_aggression,
                                                     clearing_aggression=args.clearing_aggression,
                                                     high_hold_duration=args.high_hold_duration,
                                                     low_hold_duration=args.low_hold_duration,
                                                     low_trade_ratio=args.low_trade_ratio),
            "JAM": SquidInkDeviationStrategy("JAMS", limits["JAMS"], args.n_days,
                                              z_high=args.z_high, z_low=args.z_low, profit_margin=args.profit_margin,
                                              sell_off_ratio=args.sell_off_ratio, entering_aggression=args.entering_aggression,
                                              take_profit_aggression=args.take_profit_aggression,
                                              clearing_aggression=args.clearing_aggression,
                                              high_hold_duration=args.high_hold_duration,
                                              low_hold_duration=args.low_hold_duration,
                                              low_trade_ratio=args.low_trade_ratio),
            "DJEMBE": SquidInkDeviationStrategy("DJEMBES", limits["DJEMBES"], args.n_days,
                                                 z_high=args.z_high, z_low=args.z_low, profit_margin=args.profit_margin,
                                                 sell_off_ratio=args.sell_off_ratio, entering_aggression=args.entering_aggression,
                                                 take_profit_aggression=args.take_profit_aggression,
                                                 clearing_aggression=args.clearing_aggression,
                                                 high_hold_duration=args.high_hold_duration,
                                                 low_hold_duration=args.low_hold_duration,
                                                 low_trade_ratio=args.low_trade_ratio),
            "PICNIC_BASKET1": BasketDeviationStrategy("PICNIC_BASKET1",
                                                      {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                                                      limits["PICNIC_BASKET1"],
                                                      window_size=500,
                                                      z_threshold=5.0),
            "PICNIC_BASKET2": BasketDeviationStrategy("PICNIC_BASKET2",
                                                      {"CROISSANTS": 4, "JAMS": 2},
                                                      limits["PICNIC_BASKET2"],
                                                      window_size=500,
                                                      z_threshold=5.0),
            
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