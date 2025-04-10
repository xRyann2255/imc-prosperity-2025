import json
from typing import Any, Dict, List, Tuple
import math
import jsonpickle
import numpy as np
import statistics
from collections import deque

# Import required classes from datamodel
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId

# Logger boilerplate
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
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
        return value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        # Common tracking variables
        self.day = 0
        self.timestamp = 0
        
        # Rainforest Resin tracking
        self.resin_prices = deque(maxlen=200)
        self.resin_volatility = 0
        self.resin_price_ewma = None
        
        # Kelp tracking
        self.kelp_prices = []
        self.kelp_vwap = []
        
        # Squid Ink tracking
        self.squid_prices = deque(maxlen=400)  # Store more history for pattern detection
        self.squid_order_books = deque(maxlen=50)  # Recent order books
        self.squid_model = SquidInkPredictor()
        
        # Algorithm state
        self.initialization_complete = False
        
    def update_timestamp(self, timestamp):
        """Track timestamp to detect day boundaries"""
        if timestamp < self.timestamp:
            self.day += 1
        self.timestamp = timestamp
        
    def get_order_book_features(self, order_depth: OrderDepth) -> dict:
        """Extract features from order book for ML model"""
        features = {}
        
        # Basic price features
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            features['mid_price'] = mid_price
            features['spread'] = spread
            features['spread_pct'] = spread / mid_price if mid_price != 0 else 0
            
            # Volume features
            bid_volume = sum(order_depth.buy_orders.values())
            ask_volume = sum(abs(v) for v in order_depth.sell_orders.values())
            
            features['bid_volume'] = bid_volume
            features['ask_volume'] = ask_volume
            
            # Imbalance metrics
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                features['volume_imbalance'] = (bid_volume - ask_volume) / total_volume
            else:
                features['volume_imbalance'] = 0
                
            # Weighted average prices
            if bid_volume > 0:
                features['wavg_bid'] = sum(price * qty for price, qty in order_depth.buy_orders.items()) / bid_volume
            else:
                features['wavg_bid'] = best_bid
                
            if ask_volume > 0:
                features['wavg_ask'] = sum(price * abs(qty) for price, qty in order_depth.sell_orders.items()) / ask_volume
            else:
                features['wavg_ask'] = best_ask
                
            # Price pressure
            features['price_pressure'] = features['wavg_bid'] - features['wavg_ask']
            
        return features

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate mid price from order book"""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None
        
    def calculate_resin_fair_value(self, order_depth: OrderDepth) -> float:
        """Calculate adaptive fair value for Rainforest Resin"""
        # Get current mid price
        mid_price = self.get_mid_price(order_depth)
        if mid_price is None:
            return 10000  # Default if no orders
            
        # Add to price history
        self.resin_prices.append(mid_price)
        
        # Calculate EWMA if we have enough data
        if self.resin_price_ewma is None and len(self.resin_prices) > 0:
            self.resin_price_ewma = sum(self.resin_prices) / len(self.resin_prices)
        elif self.resin_price_ewma is not None:
            alpha = 0.05  # Lower alpha for more stability
            self.resin_price_ewma = alpha * mid_price + (1 - alpha) * self.resin_price_ewma
        
        # Calculate recent volatility
        if len(self.resin_prices) >= 10:
            recent_prices = list(self.resin_prices)[-10:]
            self.resin_volatility = statistics.stdev(recent_prices)
        
        # Default fair value known to be around 10000
        base_fair_value = 10000
        
        # Adjust fair value based on EWMA with limited deviation
        if self.resin_price_ewma is not None:
            # Allow small adjustments but anchor to base value
            max_deviation = 10 + self.resin_volatility  # Allow more flexibility when volatile
            deviation = self.resin_price_ewma - base_fair_value
            capped_deviation = max(min(deviation, max_deviation), -max_deviation)
            fair_value = base_fair_value + capped_deviation
        else:
            fair_value = base_fair_value
            
        return fair_value
        
    def calculate_spread_width(self, volatility: float, base_width: int = 2) -> int:
        """Calculate adaptive spread width based on volatility"""
        if volatility < 1:
            return base_width
        else:
            # Increase width when volatility increases
            return base_width + min(int(volatility / 3), 3)
    
    def rainforest_resin_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        """Improved strategy for Rainforest Resin"""
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Calculate adaptive fair value and spread width
        fair_value = self.calculate_resin_fair_value(order_depth)
        width = self.calculate_spread_width(self.resin_volatility)
        
        # Find best prices to place orders
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + width], default=fair_value + width + 1)
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - width], default=fair_value - width - 1)

        # Opportunistic trading - take profitable orders
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            
            # More aggressive buying when price is well below fair value
            price_discount = (fair_value - best_ask) / fair_value
            aggression_factor = 1.0
            if price_discount > 0.003:  # > 0.3% discount
                aggression_factor = 1.0 + min(price_discount * 100, 1.0)  # Increase size based on discount
                
            if best_ask < fair_value:
                buy_qty = min(int(best_ask_amount * aggression_factor), position_limit - position)
                if buy_qty > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, buy_qty))
                    buy_order_volume += buy_qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            
            # More aggressive selling when price is well above fair value
            price_premium = (best_bid - fair_value) / fair_value
            aggression_factor = 1.0
            if price_premium > 0.003:  # > 0.3% premium
                aggression_factor = 1.0 + min(price_premium * 100, 1.0)
                
            if best_bid > fair_value:
                sell_qty = min(int(best_bid_amount * aggression_factor), position_limit + position)
                if sell_qty > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -sell_qty))
                    sell_order_volume += sell_qty

        # Position management - try to clear position at fair value
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", 
            buy_order_volume, sell_order_volume, fair_value, width
        )

        # Market making - provide liquidity with remaining position capacity
        # Dynamic spread based on position
        position_bias = position / position_limit if position_limit > 0 else 0
        
        # When heavily long, widen ask side; when heavily short, widen bid side
        bid_adjustment = max(-0.5, min(0.5, -position_bias))
        ask_adjustment = max(-0.5, min(0.5, position_bias))
        
        # Place resting orders
        buy_qty = position_limit - (position + buy_order_volume)
        if buy_qty > 0:
            # Adjust price based on position
            bid_price = int(bbbf + 1 + bid_adjustment)
            orders.append(Order("RAINFOREST_RESIN", bid_price, buy_qty))

        sell_qty = position_limit + (position - sell_order_volume)
        if sell_qty > 0:
            # Adjust price based on position
            ask_price = int(baaf - 1 + ask_adjustment)
            orders.append(Order("RAINFOREST_RESIN", ask_price, -sell_qty))

        return orders

    def kelp_fair_value(self, order_depth: OrderDepth, method="mid_price", min_vol=0) -> float:
        """Calculate fair value for Kelp"""
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            if (
                len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) == 0
                or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) == 0
            ):
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
                return mid_price

    def kelp_orders(
        self,
        order_depth: OrderDepth,
        timespan: int,
        width: float,
        kelp_take_width: float,
        position: int,
        position_limit: int,
    ) -> List[Order]:
        """Strategy for Kelp trading"""
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Dynamic volume filter based on order book averages
        if order_depth.sell_orders and order_depth.buy_orders:
            avg_sell_volume = sum(abs(order_depth.sell_orders[price]) for price in order_depth.sell_orders) / len(order_depth.sell_orders)
            avg_buy_volume = sum(abs(order_depth.buy_orders[price]) for price in order_depth.buy_orders) / len(order_depth.buy_orders)
            vol = math.floor(min(avg_sell_volume, avg_buy_volume))
            logger.print("Volume filter set to: ", vol)
            fair_value = self.kelp_fair_value(order_depth, method="mid_price_with_vol_filter", min_vol=vol)
        else:
            logger.print("Uh oh why am I here?")
            fair_value = self.kelp_fair_value(order_depth, method="mid_price")

        # Determine resting order prices
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(bbf) if bbf else fair_value - 2

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -quantity))
                    sell_order_volume += quantity

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 1
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("KELP", bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", baaf - 1, -sell_quantity))

        return orders

    def clear_position_order(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
        width: int,
    ) -> Tuple[int, int]:
        """Try to clear positions at fair value"""
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def find_squid_patterns(self):
        """Analyze squid ink price history for patterns"""
        if len(self.squid_prices) < 50:
            return None  # Not enough data yet
        
        # Try to find cyclic patterns using autocorrelation
        prices = list(self.squid_prices)
        
        # Normalize prices for better pattern detection
        mean_price = sum(prices) / len(prices)
        normalized = [p - mean_price for p in prices]
        
        # Calculate autocorrelation for different lags
        autocorr = {}
        for lag in range(1, min(30, len(normalized) // 3)):
            # Skip if not enough data for this lag
            if lag >= len(normalized):
                continue
                
            # Calculate autocorrelation for this lag
            correlation = 0
            for i in range(len(normalized) - lag):
                correlation += normalized[i] * normalized[i + lag]
            
            # Normalize correlation
            if sum(n*n for n in normalized[:-lag]) == 0:
                autocorr[lag] = 0
            else:
                autocorr[lag] = correlation / (sum(n*n for n in normalized[:-lag]) ** 0.5 * 
                                              sum(n*n for n in normalized[lag:]) ** 0.5)
        
        # Find lag with highest autocorrelation
        if not autocorr:
            return None
            
        best_lag = max(autocorr.items(), key=lambda x: x[1])
        
        # If autocorrelation is strong enough, return the pattern period
        if best_lag[1] > 0.3:  # Threshold for significance
            return best_lag[0]
        return None

    def squid_ink_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        """Advanced strategy for Squid Ink"""
        orders: List[Order] = []
        
        # Skip if order book is incomplete
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return []
            
        # Get current mid price and add to history
        mid_price = self.get_mid_price(order_depth)
        self.squid_prices.append(mid_price)
        
        # Extract order book features for prediction
        features = self.get_order_book_features(order_depth)
        self.squid_order_books.append(features)
        
        # Find patterns in historical data
        pattern_period = self.find_squid_patterns()
        
        # Make predictions
        predicted_price = None
        
        # Method 1: Use pattern-based prediction if a strong pattern is found
        if pattern_period and len(self.squid_prices) > pattern_period:
            # Use the pattern to predict the next price
            pattern_prediction = self.squid_prices[-pattern_period]
            confidence_pattern = 0.5  # Base confidence in pattern
        else:
            pattern_prediction = None
            confidence_pattern = 0
        
        # Method 2: Use linear model prediction
        if len(self.squid_prices) >= 4:
            # Get last 4 prices for model input
            recent_prices = list(self.squid_prices)[-4:]
            
            # Use trained model to make prediction
            model_prediction = self.squid_model.predict(recent_prices)
            confidence_model = 0.5  # Base confidence in model
        else:
            model_prediction = None
            confidence_model = 0
        
        # Method 3: Order book imbalance prediction
        if features:
            # Simple prediction based on volume imbalance
            imbalance = features.get('volume_imbalance', 0)
            # Predict price movement based on imbalance
            imbalance_adjustment = imbalance * 1.0  # Scale factor
            imbalance_prediction = mid_price * (1 + imbalance_adjustment/100)
            confidence_imbalance = 0.3  # Base confidence in imbalance
        else:
            imbalance_prediction = None
            confidence_imbalance = 0
        
        # Combine predictions with weights based on confidence
        total_confidence = confidence_pattern + confidence_model + confidence_imbalance
        if total_confidence > 0:
            predicted_price = 0
            if pattern_prediction:
                predicted_price += (confidence_pattern / total_confidence) * pattern_prediction
            if model_prediction:
                predicted_price += (confidence_model / total_confidence) * model_prediction
            if imbalance_prediction:
                predicted_price += (confidence_imbalance / total_confidence) * imbalance_prediction
            
            # Round to nearest integer
            predicted_price = int(round(predicted_price))
        else:
            # Fallback to current mid price if no predictions
            predicted_price = mid_price
        
        # Calculate confidence-based spread
        confidence = min(1.0, total_confidence)
        spread_width = 2 if confidence > 0.6 else 3  # Tighter spread with higher confidence
        
        # Trading logic similar to other products but with predicted price
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Determine price levels
        aaf = [price for price in order_depth.sell_orders.keys() if price > predicted_price + spread_width]
        baaf = min(aaf) if aaf else predicted_price + spread_width + 1
        bbf = [price for price in order_depth.buy_orders.keys() if price < predicted_price - spread_width]
        bbbf = max(bbf) if bbf else predicted_price - spread_width - 1
        
        # Market taking - more aggressive with higher confidence
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            
            # Adjust threshold based on confidence
            threshold = predicted_price - (1.0 - confidence) * spread_width
            
            if best_ask < threshold:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_ask, quantity))
                    buy_order_volume += quantity
        
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            
            # Adjust threshold based on confidence
            threshold = predicted_price + (1.0 - confidence) * spread_width
            
            if best_bid > threshold:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_bid, -quantity))
                    sell_order_volume += quantity
        
        # Position management
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "SQUID_INK", 
            buy_order_volume, sell_order_volume, predicted_price, spread_width
        )
        
        # Place resting orders - size based on confidence
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            # Size order based on confidence
            order_size = int(buy_quantity * (0.5 + 0.5 * confidence))
            if order_size > 0:
                orders.append(Order("SQUID_INK", bbbf + 1, order_size))
        
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            # Size order based on confidence
            order_size = int(sell_quantity * (0.5 + 0.5 * confidence))
            if order_size > 0:
                orders.append(Order("SQUID_INK", baaf - 1, -order_size))
        
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """Main trading loop"""
        # Update timestamp tracking
        self.update_timestamp(state.timestamp)
        
        result: Dict[Symbol, List[Order]] = {}
        
        # Position limits
        position_limit = 50  # Same for all products
        
        # RAINFOREST_RESIN orders
        if "RAINFOREST_RESIN" in state.order_depths:
            position = state.position.get("RAINFOREST_RESIN", 0)
            resin_orders = self.rainforest_resin_orders(
                state.order_depths["RAINFOREST_RESIN"],
                position,
                position_limit
            )
            result["RAINFOREST_RESIN"] = resin_orders

        # KELP orders
        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"],
                10,  # timespan
                4,   # width
                1.35,  # take_width
                kelp_position,
                position_limit
            )
            result["KELP"] = kelp_orders

        # SQUID_INK orders
        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_orders = self.squid_ink_orders(
                state.order_depths["SQUID_INK"],
                squid_position,
                position_limit
            )
            if squid_orders:
                result["SQUID_INK"] = squid_orders

        # Store state data
        trader_data = {
            "resin_prices": list(self.resin_prices)[-30:] if self.resin_prices else [],
            "resin_volatility": self.resin_volatility,
            "kelp_prices": self.kelp_prices[-30:] if self.kelp_prices else [],
            "squid_prices": list(self.squid_prices)[-30:] if self.squid_prices else [],
        }
        
        traderData = jsonpickle.encode(trader_data)
        conversions = 1
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData


class SquidInkPredictor:
    """Machine learning predictor for squid ink patterns"""
    def __init__(self):
        # Pre-trained coefficimport json
from typing import Any, Dict, List, Tuple
import math
import jsonpickle
import numpy as np
import statistics
from collections import deque

# Import required classes from datamodel
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
# Import the squid ink predictor
from squid_ink_predictor import SquidInkPredictor

# Logger boilerplate
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
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
        return value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        # Common tracking variables
        self.day = 0
        self.timestamp = 0
        
        # Rainforest Resin tracking
        self.resin_prices = deque(maxlen=200)
        self.resin_volatility = 0
        self.resin_price_ewma = None
        
        # Kelp tracking
        self.kelp_prices = []
        self.kelp_vwap = []
        
        # Squid Ink tracking
        self.squid_prices = deque(maxlen=400)  # Store more history for pattern detection
        self.squid_order_books = deque(maxlen=50)  # Recent order books
        
        # Initialize the squid ink predictor
        # Option 1: Use the hardcoded values (if you don't have the JSON file)
        self.squid_model = SquidInkPredictor()
        
        # Option 2: Load from the trained model file (uncomment if you have the JSON)
        # try:
        #     self.squid_model = SquidInkPredictor.from_json("squid_ink_model.json")
        #     logger.print("Loaded squid ink model from squid_ink_model.json")
        # except Exception as e:
        #     logger.print(f"Error loading model: {e}, using default model instead")
        #     self.squid_model = SquidInkPredictor()
        
        # Algorithm state
        self.initialization_complete = False
        
    def update_timestamp(self, timestamp):
        """Track timestamp to detect day boundaries"""
        if timestamp < self.timestamp:
            self.day += 1
        self.timestamp = timestamp
        
    def get_order_book_features(self, order_depth: OrderDepth) -> dict:
        """Extract features from order book for ML model"""
        features = {}
        
        # Basic price features
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            features['mid_price'] = mid_price
            features['spread'] = spread
            features['spread_pct'] = spread / mid_price if mid_price != 0 else 0
            
            # Volume features
            bid_volume = sum(order_depth.buy_orders.values())
            ask_volume = sum(abs(v) for v in order_depth.sell_orders.values())
            
            features['bid_volume'] = bid_volume
            features['ask_volume'] = ask_volume
            
            # Imbalance metrics
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                features['volume_imbalance'] = (bid_volume - ask_volume) / total_volume
            else:
                features['volume_imbalance'] = 0
                
            # Weighted average prices
            if bid_volume > 0:
                features['wavg_bid'] = sum(price * qty for price, qty in order_depth.buy_orders.items()) / bid_volume
            else:
                features['wavg_bid'] = best_bid
                
            if ask_volume > 0:
                features['wavg_ask'] = sum(price * abs(qty) for price, qty in order_depth.sell_orders.items()) / ask_volume
            else:
                features['wavg_ask'] = best_ask
                
            # Price pressure
            features['price_pressure'] = features['wavg_bid'] - features['wavg_ask']
            
        return features

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate mid price from order book"""
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return None
        
    def calculate_resin_fair_value(self, order_depth: OrderDepth) -> float:
        """Calculate adaptive fair value for Rainforest Resin"""
        # Get current mid price
        mid_price = self.get_mid_price(order_depth)
        if mid_price is None:
            return 10000  # Default if no orders
            
        # Add to price history
        self.resin_prices.append(mid_price)
        
        # Calculate EWMA if we have enough data
        if self.resin_price_ewma is None and len(self.resin_prices) > 0:
            self.resin_price_ewma = sum(self.resin_prices) / len(self.resin_prices)
        elif self.resin_price_ewma is not None:
            alpha = 0.05  # Lower alpha for more stability
            self.resin_price_ewma = alpha * mid_price + (1 - alpha) * self.resin_price_ewma
        
        # Calculate recent volatility
        if len(self.resin_prices) >= 10:
            recent_prices = list(self.resin_prices)[-10:]
            self.resin_volatility = statistics.stdev(recent_prices)
        
        # Default fair value known to be around 10000
        base_fair_value = 10000
        
        # Adjust fair value based on EWMA with limited deviation
        if self.resin_price_ewma is not None:
            # Allow small adjustments but anchor to base value
            max_deviation = 10 + self.resin_volatility  # Allow more flexibility when volatile
            deviation = self.resin_price_ewma - base_fair_value
            capped_deviation = max(min(deviation, max_deviation), -max_deviation)
            fair_value = base_fair_value + capped_deviation
        else:
            fair_value = base_fair_value
            
        return fair_value
        
    def calculate_spread_width(self, volatility: float, base_width: int = 2) -> int:
        """Calculate adaptive spread width based on volatility"""
        if volatility < 1:
            return base_width
        else:
            # Increase width when volatility increases
            return base_width + min(int(volatility / 3), 3)
    
    def rainforest_resin_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        """Improved strategy for Rainforest Resin"""
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Calculate adaptive fair value and spread width
        fair_value = self.calculate_resin_fair_value(order_depth)
        width = self.calculate_spread_width(self.resin_volatility)
        
        # Find best prices to place orders
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + width], default=fair_value + width + 1)
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - width], default=fair_value - width - 1)

        # Opportunistic trading - take profitable orders
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            
            # More aggressive buying when price is well below fair value
            price_discount = (fair_value - best_ask) / fair_value
            aggression_factor = 1.0
            if price_discount > 0.003:  # > 0.3% discount
                aggression_factor = 1.0 + min(price_discount * 100, 1.0)  # Increase size based on discount
                
            if best_ask < fair_value:
                buy_qty = min(int(best_ask_amount * aggression_factor), position_limit - position)
                if buy_qty > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, buy_qty))
                    buy_order_volume += buy_qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            
            # More aggressive selling when price is well above fair value
            price_premium = (best_bid - fair_value) / fair_value
            aggression_factor = 1.0
            if price_premium > 0.003:  # > 0.3% premium
                aggression_factor = 1.0 + min(price_premium * 100, 1.0)
                
            if best_bid > fair_value:
                sell_qty = min(int(best_bid_amount * aggression_factor), position_limit + position)
                if sell_qty > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -sell_qty))
                    sell_order_volume += sell_qty

        # Position management - try to clear position at fair value
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", 
            buy_order_volume, sell_order_volume, fair_value, width
        )

        # Market making - provide liquidity with remaining position capacity
        # Dynamic spread based on position
        position_bias = position / position_limit if position_limit > 0 else 0
        
        # When heavily long, widen ask side; when heavily short, widen bid side
        bid_adjustment = max(-0.5, min(0.5, -position_bias))
        ask_adjustment = max(-0.5, min(0.5, position_bias))
        
        # Place resting orders
        buy_qty = position_limit - (position + buy_order_volume)
        if buy_qty > 0:
            # Adjust price based on position
            bid_price = int(bbbf + 1 + bid_adjustment)
            orders.append(Order("RAINFOREST_RESIN", bid_price, buy_qty))

        sell_qty = position_limit + (position - sell_order_volume)
        if sell_qty > 0:
            # Adjust price based on position
            ask_price = int(baaf - 1 + ask_adjustment)
            orders.append(Order("RAINFOREST_RESIN", ask_price, -sell_qty))

        return orders

    def kelp_fair_value(self, order_depth: OrderDepth, method="mid_price", min_vol=0) -> float:
        """Calculate fair value for Kelp"""
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            if (
                len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) == 0
                or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) == 0
            ):
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
                return mid_price

    def kelp_orders(
        self,
        order_depth: OrderDepth,
        timespan: int,
        width: float,
        kelp_take_width: float,
        position: int,
        position_limit: int,
    ) -> List[Order]:
        """Strategy for Kelp trading"""
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Dynamic volume filter based on order book averages
        if order_depth.sell_orders and order_depth.buy_orders:
            avg_sell_volume = sum(abs(order_depth.sell_orders[price]) for price in order_depth.sell_orders) / len(order_depth.sell_orders)
            avg_buy_volume = sum(abs(order_depth.buy_orders[price]) for price in order_depth.buy_orders) / len(order_depth.buy_orders)
            vol = math.floor(min(avg_sell_volume, avg_buy_volume))
            logger.print("Volume filter set to: ", vol)
            fair_value = self.kelp_fair_value(order_depth, method="mid_price_with_vol_filter", min_vol=vol)
        else:
            logger.print("Uh oh why am I here?")
            fair_value = self.kelp_fair_value(order_depth, method="mid_price")

        # Determine resting order prices
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(bbf) if bbf else fair_value - 2

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -quantity))
                    sell_order_volume += quantity

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 1
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("KELP", bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", baaf - 1, -sell_quantity))

        return orders

    def clear_position_order(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
        width: int,
    ) -> Tuple[int, int]:
        """Try to clear positions at fair value"""
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders:
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders:
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def squid_ink_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        """Advanced strategy for Squid Ink using trained model"""
        orders: List[Order] = []
        
        # Skip if order book is incomplete
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return []
            
        # Get current mid price and add to history
        mid_price = self.get_mid_price(order_depth)
        self.squid_prices.append(mid_price)
        
        # Extract order book features for future use
        features = self.get_order_book_features(order_depth)
        self.squid_order_books.append(features)
        
        # Wait until we have enough price history for the model
        if len(self.squid_prices) < len(self.squid_model.coefficients):
            return []  # Not enough data yet
        
        # Get price prediction from trained model
        recent_prices = list(self.squid_prices)[-len(self.squid_model.coefficients):]
        predicted_price = self.squid_model.predict_next_price(recent_prices)
        
        # Get predicted direction and confidence
        price_direction = self.squid_model.predict_direction(list(self.squid_prices))
        confidence = self.squid_model.get_confidence(list(self.squid_prices))
        
        # Logging for debugging
        logger.print(f"SQUID_INK - Current: {mid_price}, Predicted: {predicted_price}, Direction: {price_direction}")
        
        # Trading logic similar to other products but with predicted price
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Dynamic spread based on confidence
        spread_width = max(1, int(3 * (1 - confidence)))
        
        # Determine price levels
        aaf = [price for price in order_depth.sell_orders.keys() if price > predicted_price + spread_width]
        baaf = min(aaf) if aaf else predicted_price + spread_width + 1
        bbf = [price for price in order_depth.buy_orders.keys() if price < predicted_price - spread_width]
        bbbf = max(bbf) if bbf else predicted_price - spread_width - 1
        
        # Market taking - more aggressive with higher confidence
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            
            # Adjust threshold based on confidence
            threshold = predicted_price - (1.0 - confidence) * spread_width
            
            if best_ask < threshold:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_ask, quantity))
                    buy_order_volume += quantity
        
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            
            # Adjust threshold based on confidence
            threshold = predicted_price + (1.0 - confidence) * spread_width
            
            if best_bid > threshold:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_bid, -quantity))
                    sell_order_volume += quantity
        
        # Position management
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "SQUID_INK", 
            buy_order_volume, sell_order_volume, predicted_price, spread_width
        )
        
        # Place resting orders - size based on confidence
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            # Size order based on confidence
            order_size = int(buy_quantity * (0.5 + 0.5 * confidence))
            if order_size > 0:
                orders.append(Order("SQUID_INK", bbbf + 1, order_size))
        
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            # Size order based on confidence
            order_size = int(sell_quantity * (0.5 + 0.5 * confidence))
            if order_size > 0:
                orders.append(Order("SQUID_INK", baaf - 1, -order_size))
        
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """Main trading loop"""
        # Update timestamp tracking
        self.update_timestamp(state.timestamp)
        
        result: Dict[Symbol, List[Order]] = {}
        
        # Position limits
        position_limit = 50  # Same for all products
        
        # RAINFOREST_RESIN orders
        if "RAINFOREST_RESIN" in state.order_depths:
            position = state.position.get("RAINFOREST_RESIN", 0)
            resin_orders = self.rainforest_resin_orders(
                state.order_depths["RAINFOREST_RESIN"],
                position,
                position_limit
            )
            result["RAINFOREST_RESIN"] = resin_orders

        # KELP orders
        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"],
                10,  # timespan
                4,   # width
                1.35,  # take_width
                kelp_position,
                position_limit
            )
            result["KELP"] = kelp_orders

        # SQUID_INK orders
        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_orders = self.squid_ink_orders(
                state.order_depths["SQUID_INK"],
                squid_position,
                position_limit
            )
            if squid_orders:
                result["SQUID_INK"] = squid_orders

        # Store state data
        trader_data = {
            "resin_prices": list(self.resin_prices)[-30:] if self.resin_prices else [],
            "resin_volatility": self.resin_volatility,
            "kelp_prices": self.kelp_prices[-30:] if self.kelp_prices else [],
            "squid_prices": list(self.squid_prices)[-30:] if self.squid_prices else [],
        }
        
        traderData = jsonpickle.encode(trader_data)
        conversions = 1
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderDataients from analysis (these could be updated over time)
        # The model seems to follow an AR(4) pattern based on the code
        self.coefficients = [-0.01591316, 0.07101847, 0.12484902, 0.82004215]
        self.intercept = 0.00047196049877129553
        
    def predict(self, prices):
        """Predict next price based on recent prices"""
        if len(prices) != 4:
            return prices[-1]  # Return last known price if we don't have exactly 4 inputs
            
        prediction = self.intercept + sum(c * p for c, p in zip(self.coefficients, prices))
        return prediction