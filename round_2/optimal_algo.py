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

# --- Updated BasketDeviationStrategy that enforces full hedgeability ---
class BasketDeviationStrategy(Strategy):
    def __init__(self, basket_symbol: str, component_weights: dict, limit: int,
                 window_size: int = 45, z_threshold: float = 2.0, underlying_limits: dict = None) -> None:
        """
        basket_symbol: The symbol for the basket product (e.g. "PICNIC_BASKET1")
        component_weights: Dict mapping underlying names to the per-basket quantity
                           e.g. {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
        limit: Position limit for the basket.
        window_size: Window size used for computing the rolling spread history.
        z_threshold: When the spread’s z-score crosses this value, true_value is adjusted.
        underlying_limits: A dict mapping each underlying product (e.g. "CROISSANTS") to its position limit.
        """
        super().__init__(basket_symbol, limit)
        self.component_weights = component_weights  
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.spread_history = deque(maxlen=window_size)
        # underlying_limits is required to know what hedge capacity is available.
        self.underlying_limits = underlying_limits if underlying_limits is not None else {}

    def compute_market_mid(self, order_depth: OrderDepth) -> float:
        # Compute the mid price from the basket's order book.
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
        # Compute synthetic mid price by summing the weighted mid of each component.
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
        # Use the synthetic mid as our base true value.
        synthetic_mid = self.compute_synthetic_mid(state)
        return round(synthetic_mid)
    
    def hedge_cap(self, state: TradingState) -> tuple[float, float]:
        """
        Compute the maximum additional basket units that can be bought (positive) or sold (negative)
        such that the new basket position remains hedgeable by the underlying products.
        For each underlying u in the basket:
          - Allowed basket exposure is max_exposure = (underlying_limit[u] / weight[u]).
          - If current basket position is P, then:
              For buying:  P + d  <= max_exposure   => d <= max_exposure - P
              For selling: P + d  >= -max_exposure  => d >= -max_exposure - P
        We return (allowed_buy, allowed_sell) where allowed_sell is a positive number representing
        the maximum baskets that can be sold (in absolute terms).
        """
        pos_current = state.position.get(self.symbol, 0)
        allowed_buy = 99999
        allowed_sell = 99999
        for comp, weight in self.component_weights.items():
            if comp not in self.underlying_limits:
                continue  # If not provided, skip capping for this component.
            limit_u = self.underlying_limits[comp]
            max_exposure = limit_u // weight  # Maximum baskets fully hedgeable via this underlying.
            # For buying baskets (increasing basket position):
            allowed_buy_for_u = max_exposure - pos_current
            # For selling baskets (decreasing basket position): we require pos_current + d >= -max_exposure,
            # so d >= -max_exposure - pos_current; the allowed magnitude is: (-max_exposure - pos_current) in absolute terms.
            allowed_sell_for_u = pos_current + max_exposure  # (since d is negative, its magnitude is capped by this value)
            logger.print(f"Underlying: {comp}, Limit: {limit_u}, Weight: {weight}, Allowed Buy: {allowed_buy_for_u}, Allowed Sell: {allowed_sell_for_u}")
            allowed_buy = min(allowed_buy, allowed_buy_for_u)
            allowed_sell = min(allowed_sell, allowed_sell_for_u)
        return allowed_buy, allowed_sell

    def act(self, state: TradingState) -> None:
        # Ensure the basket order depth is available.
        if self.symbol not in state.order_depths:
            return

        basket_od = state.order_depths[self.symbol]
        basket_mid = self.compute_market_mid(basket_od)
        synthetic_mid = self.compute_synthetic_mid(state)
        
        # Compute the spread between basket market mid and synthetic mid.
        spread = basket_mid - synthetic_mid
        self.spread_history.append(spread)
        
        # Compute rolling z-score if enough history is present.
        if len(self.spread_history) >= self.window_size:
            mean_spread = sum(self.spread_history) / len(self.spread_history)
            std_spread = (sum((s - mean_spread) ** 2 for s in self.spread_history) / len(self.spread_history)) ** 0.5
            zscore = (spread - mean_spread) / std_spread if std_spread > 0 else 0
        else:
            zscore = 0
        
        # Base true value is set to the synthetic mid.
        true_value = round(synthetic_mid)
        # Adjust the true value based on the z-score.
        if zscore > self.z_threshold:
            true_value -= abs(zscore)
        elif zscore < -self.z_threshold:
            true_value += abs(zscore)

        # Standard market-making approach based on current basket order depth.
        order_depth = basket_od
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        current_pos = state.position.get(self.symbol, 0)
        to_buy = self.limit - current_pos
        to_sell = self.limit + current_pos

        # Enforce hedgeability: cap the basket trade quantities so the resulting basket position can be hedged.
        allowed_buy, allowed_sell = self.hedge_cap(state)
        logger.print(f"Allowed Buy: {allowed_buy}, Allowed Sell: {allowed_sell}")
        # Only trade up to the hedge cap.
        to_buy = min(to_buy, allowed_buy)
        to_sell = min(to_sell, allowed_sell)
        
        # Process orders on the sell side to fill a basket buy order.
        for price, volume in sell_orders:
            if to_buy > 0 and price <= true_value:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        # Process orders on the buy side to fill a basket sell order.
        for price, volume in buy_orders:
            if to_sell > 0 and price >= true_value:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        # If remaining orders exist, send fallback orders.
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

# === Begin BasketHedgeStrategy ===
class BasketHedgeStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, basket_recipes: dict) -> None:
        """
        basket_recipes: A dict mapping basket symbols to the quantity of this underlying per basket.
          e.g. for CROISSANT: {"PICNIC_BASKET1": 6, "PICNIC_BASKET2": 4}
        """
        super().__init__(symbol, limit)
        self.basket_recipes = basket_recipes

    def get_target_position(self, state: TradingState) -> int:
        # Calculate the net exposure from basket positions.
        # For each basket, multiply its current position by the per-basket amount and sum them.
        # Then, the hedge target is the negative of that sum.
        net_exposure = 0
        for basket_sym, qty_per_basket in self.basket_recipes.items():
            basket_pos = state.position.get(basket_sym, 0)
            net_exposure += basket_pos * qty_per_basket
        return -net_exposure

    def act(self, state: TradingState) -> None:
        # Compute the target underlying position to fully hedge the basket exposure.
        target = self.get_target_position(state)
        # Clip target to within our position limits.
        target = min(max(target, -self.limit), self.limit)
        current = state.position.get(self.symbol, 0)
        delta = target - current

        if delta == 0:
            # Already perfectly hedged.
            return

        if self.symbol not in state.order_depths:
            return  # Nothing to trade on if the order book is absent.

        od = state.order_depths[self.symbol]
        logger.print(f"Symbol:{self.symbol} Delta: {delta}, Target: {target}, Current: {current}")
        # If delta > 0, we need to buy underlying; if delta < 0, we need to sell.
        if delta > 0:
            # Need to buy 'delta' units; sweep the sell side of the order book.
            sorted_sell = sorted(od.sell_orders.items())  # ascending prices
            remaining = delta
            for price, vol in sorted_sell:
                # In the sell orders, volume is negative.
                avail = -vol
                trade_qty = min(remaining, avail)
                if trade_qty > 0:
                    self.buy(price, trade_qty)
                    remaining -= trade_qty
                    logger.print(f"Buying {trade_qty} at {price}")
                    if remaining <= 0:
                        break
            # If liquidity is insufficient, place a fallback order.
            if remaining > 0:
                fallback_price = sorted_sell[0][0] if sorted_sell else self.get_fallback_price(od)
                self.buy(fallback_price, remaining)
        else:
            # delta < 0, need to sell abs(delta) units; sweep the buy side.
            remaining = abs(delta)
            sorted_buy = sorted(od.buy_orders.items(), reverse=True)  # descending prices
            for price, vol in sorted_buy:
                avail = vol  # volume is positive for buy orders.
                trade_qty = min(remaining, avail)
                if trade_qty > 0:
                    self.sell(price, trade_qty)
                    remaining -= trade_qty
                    logger.print(f"Selling {trade_qty} at {price}")
                    if remaining <= 0:
                        break
            if remaining > 0:
                fallback_price = sorted_buy[0][0] if sorted_buy else self.get_fallback_price(od)
                self.sell(fallback_price, remaining)

    def get_fallback_price(self, order_depth: OrderDepth) -> int:
        # Compute a mid price fallback if no suitable order book orders are available.
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return round((best_bid + best_ask) / 2)
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        else:
            return 0

    def get_true_value(self, state: TradingState) -> int:
        # For hedging, the true value isn't used in the same way.
        if self.symbol in state.order_depths:
            return self.get_fallback_price(state.order_depths[self.symbol])
        return 0

    def save(self) -> Any:
        # This strategy is stateless.
        return {}

    def load(self, data: Any) -> None:
        pass

class PicnicBasket1Strategy(Strategy):
    def __init__(self, symbol: Symbol, components, limit: int, spread=60) -> None:
        super().__init__(symbol, limit)
        self.components = components  # Components of the basket and their weights
        self.spread = spread  # Adjustable spread around fair value

    def act(self, state: TradingState) -> None:
        fair_value = self.compute_fair_value(state)
        if fair_value is not None:
            self.market_make(state, fair_value)

    def compute_fair_value(self, state: TradingState) -> float | None:
        total_value = 0
        for symbol, weight in self.components.items():
            mid_price = self.get_mid_price(state, symbol)
            if mid_price is None:
                return None
            total_value += mid_price * weight
        return total_value

    def get_mid_price(self, state: TradingState, symbol: Symbol) -> float | None:
        order_depth = state.order_depths.get(symbol)
        if not order_depth:
            return None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None

    def market_make(self, state: TradingState, fair_value: float) -> None:
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths.get(self.symbol, None)
        if not order_depth:
            return

        buy_price = int(fair_value - self.spread + 0)
        sell_price = int(fair_value + self.spread + 0)

        buy_volume = min(self.limit - position, 20)
        sell_volume = min(self.limit + position, 20)

        if buy_volume > 0:
            self.buy(buy_price, buy_volume)
        if sell_volume > 0:
            self.sell(sell_price, sell_volume)

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
            "PICNIC_BASKET2" : 100
        }
        underlying_limits = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60
        }

        self.strategies = {
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", limits["KELP"]),
            "CROISSANTS": BasketHedgeStrategy("CROISSANTS", limits["CROISSANTS"], {"PICNIC_BASKET1": 6, "PICNIC_BASKET2": 4}),
            "JAMS": BasketHedgeStrategy("JAMS", limits["JAMS"], {"PICNIC_BASKET1": 3, "PICNIC_BASKET2": 2}),
            "DJEMBES": BasketHedgeStrategy("DJEMBES", limits["DJEMBES"], {"PICNIC_BASKET1": 1}),
            "PICNIC_BASKET1": PicnicBasket1Strategy("PICNIC_BASKET1",
                                                    components={"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                                                    limit=limits["PICNIC_BASKET1"],
                                                    spread=60),
            "PICNIC_BASKET2": PicnicBasket1Strategy("PICNIC_BASKET2",
                                                      components={"CROISSANTS": 4, "JAMS": 2},
                                                      limit=limits["PICNIC_BASKET2"],
                                                      spread=60),
            "SQUID_INK": SquidInkDeviationStrategy("SQUID_INK",
                                                   limits["SQUID_INK"],
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