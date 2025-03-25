from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import pandas as pd

class Trader:
    """
    A trading algorithm implementing market making strategies for two products:
    RAINFOREST_RESIN and KELP.
    
    The strategy makes decisions based on the current order book (order_depth), computed fair value,
    and position limits. It also maintains historical data for KELP to compute a volume‐weighted
    average price (VWAP) over a sliding window.
    """
    
    def __init__(self):
        """
        Initializes the Trader instance.
        
        Attributes:
            kelp_prices (list): A list of historical mid-price values for KELP.
            kelp_vwap (list): A list of dictionaries holding volume and VWAP data for KELP.
        """
        self.kelp_prices = []
        self.kelp_vwap = []
        self.kelp_order_book = []

    def rainforest_resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, 
                                  position: int, position_limit: int) -> List[Order]:
        """
        Generates orders for RAINFOREST_RESIN based on the current order book depth.
        
        The method places orders if the best ask is below the fair value (buy opportunity)
        or the best bid is above the fair value (sell opportunity). It also adds backup orders
        if there is remaining capacity within the position limit.
        
        Args:
            order_depth (OrderDepth): Current market depth for RAINFOREST_RESIN.
            fair_value (int): The predetermined fair value for RAINFOREST_RESIN.
            width (int): A parameter to adjust order placement (not heavily used in the snippet).
            position (int): Current position held.
            position_limit (int): Maximum allowable position.
            
        Returns:
            List[Order]: A list of Order objects to be submitted.
        """
        orders: List[Order] = []

        # Initialize cumulative order volumes for buys and sells.
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Determine backup prices based on market orders that are away from the fair value.
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        # If there are sell orders, attempt to buy if the best ask is below fair value.
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            # Note: sell_orders values are negative (indicating quantity available to sell).
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                # Buy only up to the maximum allowed by the position limit.
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity)) 
                    buy_order_volume += quantity

        # If there are buy orders, attempt to sell if the best bid is above fair value.
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                # Sell only up to the maximum allowed by the position limit.
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity
        
        # Use helper method to clear any excess position by adjusting orders.
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", 
            buy_order_volume, sell_order_volume, fair_value, 1)

        # If capacity remains, add a backup buy order at a price slightly above the best bid filter.
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))  # Backup Buy order

        # If capacity remains, add a backup sell order at a price slightly below the best ask filter.
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))  # Backup Sell order

        return orders
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, 
                             position_limit: int, product: str, buy_order_volume: int, 
                             sell_order_volume: int, fair_value: float, width: int) -> List[Order]:
        """
        Adjusts orders to reduce net position risk, ensuring the net position remains within limits.
        
        If the trader's net position (after existing orders) is too high (or too low),
        the method adds orders to clear the excess by selling (or buying) at prices close to fair value.
        
        Args:
            orders (List[Order]): The current list of orders.
            order_depth (OrderDepth): Current market depth.
            position (int): The current net position.
            position_limit (int): Maximum allowable position.
            product (str): Product symbol (e.g., "RAINFOREST_RESIN" or "KELP").
            buy_order_volume (int): Volume of buy orders already placed.
            sell_order_volume (int): Volume of sell orders already placed.
            fair_value (float): Estimated fair value.
            width (int): A parameter used to adjust the price rounding.
            
        Returns:
            Tuple[int, int]: Updated buy and sell order volumes.
        """
        # Calculate net position after placing the current orders.
        position_after_take = position + buy_order_volume - sell_order_volume
        
        # Round the fair value for appropriate bid/ask price levels.
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)  # Lower bound for buy clearance.
        fair_for_ask = math.ceil(fair_value)   # Upper bound for sell clearance.

        # Calculate remaining capacity for placing additional orders.
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        # If net long, try to sell to reduce the long position.
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        # If net short, try to buy to reduce the short position.
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume
    
    def kelp_fair_value(self, order_depth: OrderDepth, method="mid_price", min_vol=0) -> float:
        """
        Calculates the fair value for KELP using one of two methods:
        
        - "mid_price": Uses the simple average of the best ask and best bid.
        - "mid_price_with_vol_filter": Filters orders based on a minimum volume requirement.
          If not enough volume is present, falls back to the simple mid-price.
        
        Args:
            order_depth (OrderDepth): Current market depth for KELP.
            method (str): The method for calculating fair value.
            min_vol (int): Minimum volume threshold for the volume filter method.
            
        Returns:
            float: The computed fair value.
        """
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            # If not enough volume at any price level, default to the simple mid-price.
            if (len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) == 0 or
                len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) == 0):
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
            return mid_price
    
    def order_to_df(self, order_depth: OrderDepth, num_level=5) -> pd.DataFrame:
        # Create a numpy array from the order book data.
        # OrderDepth 
        #   - buy_orders: Dict[int, int] = {}
        #   - sell_orders: Dict[int, int] = {}
        # OrderDepth.buy_orders and OrderDepth.sell_orders are dictionaries with price levels as keys and volumes as values.
        # Sort the order book by price level and convert it to a pandas dataframe.
        # Starting from best bid and best ask, the price levels are sorted in descending and ascending order, respectively.
        # feature names are AskP1, AskV1, AskP2, AskV2, ..., AskP BidP1, BidV1, BidP2, BidV2, ... up to num_level levels.
        # If there are fewer than num_level levels, the remaining columns are filled with NaN values.
        # The resulting dataframe is returned.
        
        # Sort the order book by price level.
        sorted_buy = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
        sorted_sell = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        
        # Initialize lists to store the price and volume data.
        ask_prices = []
        ask_volumes = []
        bid_prices = []
        bid_volumes = []
        
        # Extract the price and volume data from the sorted order book.
        for i in range(num_level):
            if i < len(sorted_sell):
                ask_prices.append(sorted_sell[i][0])
                ask_volumes.append(sorted_sell[i][1])
            else:
                ask_prices.append(np.nan)
                ask_volumes.append(np.nan)
            if i < len(sorted_buy):
                bid_prices.append(sorted_buy[i][0])
                bid_volumes.append(sorted_buy[i][1])
            else:
                bid_prices.append(np.nan)
                bid_volumes.append(np.nan)
        
        # Create a pandas dataframe from the extracted data.
        data = {}
        for i in range(num_level):
            data[f"AskP{i+1}"] = [ask_prices[i]]
            data[f"AskV{i+1}"] = [ask_volumes[i]]
            data[f"BidP{i+1}"] = [bid_prices[i]]
            data[f"BidV{i+1}"] = [bid_volumes[i]]
        
        data['midprice'] = [(ask_prices[0] + bid_prices[0]) / 2]
            
        return pd.DataFrame(data)
    
    def preprocess_order_book(self, order_df:pd.DataFrame) -> pd.DataFrame:
        # preprocess the order book data
        # 1. take log on all prices
        # 2. represent price level by difference between price and midprice
        
        # Take the logarithm of all price levels (columns starting with "AskP" or "BidP").
        price_columns = [col for col in order_df.columns if 'P' in col] + ['midprice']
        order_df[price_columns] = np.log(order_df[price_columns])
        
        # Represent the price level as the difference between the price and the mid-price.
        for price_col in price_columns:
            if price_col != 'midprice':
                order_df[price_col] = order_df[price_col] - order_df['midprice']
        return order_df
        
        
    def meta_learning_fair_value(self, order_depth: OrderDepth, k: int, timespan: int, fallback_value: float) -> float:
        """
        Computes a fair value for KELP using a meta-learning approach based on historical mid-prices.
        It takes the last 'timespan' mid-price values (from t-n to t-1), splits them 90:10 in time 
        (ensuring the training data precedes the test data), fits a linear regression model on the training data,
        and then predicts the mid-price at time t+k.
        
        Args:
            k (int): The future offset (e.g., 1 for one time-step ahead).
            timespan (int): The number of historical data points to use.
            fallback_value (float): The fallback fair value (typically the current mid-price) if insufficient history exists.
            
        Returns:
            float: The predicted fair value for KELP.
        """
        import pandas as pd  # Ensure pandas is imported
        if len(self.kelp_prices) < timespan:
            return fallback_value

        # Assume self.kelp_order_book stores order book snapshots corresponding to self.kelp_prices.
        data = self.kelp_order_book[-timespan:-k]
        data = [self.order_to_df(order, num_level=5) for order in data]
        data = [self.preprocess_order_book(order) for order in data]
        x = pd.concat(data)  # shape: (timespan, num_level * 4 + 1)

        # y = log difference between midprices apart by k
        y = np.array(self.kelp_prices[-timespan:])
        y = np.log(y[k:]) - np.log(y[:-k])
        n = len(x)
        
        # 90:10 train/test split along time
        train_size = int(0.9 * n)
        if train_size < k:
            return fallback_value
        X_train = x[:train_size]
        y_train = y[:train_size]
        
        # TODO: Fit a linear regression model using numpy
        # Convert training data to numpy array and add intercept term.
        X_train_np = X_train.values  # shape: (train_size, num_features)
        X_train_aug = np.hstack((np.ones((X_train_np.shape[0], 1)), X_train_np))
        coeffs = np.linalg.lstsq(X_train_aug, y_train, rcond=None)[0]

        # Test on the last 10% of the data
        X_test = x[train_size:]
        y_test = y[train_size:]
        
        # TODO: evaluate the model using mean squared error and correlation coefficient
        X_test_np = X_test.values
        X_test_aug = np.hstack((np.ones((X_test_np.shape[0], 1)), X_test_np))
        y_pred = X_test_aug.dot(coeffs)
        mse = np.mean((y_pred - y_test) ** 2)
        r = np.corrcoef(y_pred, y_test)[0, 1]
        print(f"mse: {mse}, r: {r}")
        
        # TODO: Predict the mid-price at current time
        # Process the current order depth into a feature vector.
        cur_order_df = self.order_to_df(order_depth, num_level=5)
        final_x = self.preprocess_order_book(cur_order_df)
        # Use the last row of the processed current order book data.
        final_x = final_x.iloc[-1]
        final_x_np = final_x.values.reshape(1, -1)
        final_x_aug = np.hstack((np.ones((final_x_np.shape[0], 1)), final_x_np))
        # Predict the log difference for the current time
        predicted_log_diff = final_x_aug.dot(coeffs)[0]
        # Convert predicted log difference to a multiplicative factor: log(mid[t+k]) = log(mid[t]) + predicted_log_diff
        prediction = fallback_value * np.exp(predicted_log_diff)
        return prediction


    def kelp_orders(self, order_depth: OrderDepth, timespan: int, width: float, 
                    kelp_take_width: float, position: int, position_limit: int) -> List[Order]:
        """
        Generates orders for KELP based on current market depth and recent historical data.
        
        This method filters the order book by a minimum volume to reduce noise, computes a mid-price,
        and updates historical records for prices and VWAP. It then determines whether to take liquidity
        (buy or sell immediately) if the best ask or bid deviates significantly from the fair value.
        Finally, backup orders are placed if there is remaining capacity within the position limit.
        
        Args:
            order_depth (OrderDepth): Current market depth for KELP.
            timespan (int): Number of recent data points to retain for historical analysis.
            width (float): A parameter for adjusting order placement.
            kelp_take_width (float): The price deviation threshold to trigger immediate order execution.
            position (int): Current position held for KELP.
            position_limit (int): Maximum allowable position for KELP.
            
        Returns:
            List[Order]: A list of Order objects for KELP.
        """
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Ensure both sell and buy sides of the order book are present.
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    

            # Get the best ask and bid prices.
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            # Filter orders with volume at least 15 to avoid noise.
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            
            # Use filtered prices if available.
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            # Calculate a mid-price based on the filtered best ask and bid.
            mmmid_price = (mm_ask + mm_bid) / 2    
            self.kelp_prices.append(mmmid_price)

            # Calculate the volume-weighted average price (VWAP) based on best ask and bid volumes.
            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})
            
            self.kelp_order_book.append(order_depth)
            # Maintain a sliding window of historical data.
            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)
            if len(self.kelp_order_book) > timespan:
                self.kelp_order_book.pop(0)
            
            # Compute fair value based on historical data (weighted average) and then override with current mid-price.
            # fair_value = sum([x["vwap"] * x["vol"] for x in self.kelp_vwap]) / sum([x["vol"] for x in self.kelp_vwap])
            # fair_value = mmmid_price
            fair_value = self.meta_learning_fair_value(order_depth, 5, timespan, mmmid_price)
            # Define a small buffer to allow slight flexibility in order execution.
            buffer_price = 0.7

            # Take liquidity: if the best ask is sufficiently below fair value, buy.
            if best_ask <= fair_value - kelp_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_ask, quantity))
                        buy_order_volume += quantity
                        
            # Take liquidity: if the best bid is sufficiently above fair value, sell.
            if best_bid >= fair_value + kelp_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_bid, -1 * quantity))
                        sell_order_volume += quantity

            # Adjust orders to reduce excessive net position risk.
            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "KELP", 
                buy_order_volume, sell_order_volume, fair_value, 2)
            
            # Calculate backup order prices based on the market depth around fair value.
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
           
            # Place an additional buy order if there is remaining capacity.
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP", bbbf + 1, buy_quantity))  # Backup Buy order

            # Place an additional sell order if there is remaining capacity.
            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", baaf - 1, -sell_quantity))  # Backup Sell order

        return orders

    def run(self, state: TradingState):
        """
        The main execution function for the trading strategy.
        
        This method is called periodically with the latest market state. It processes
        market data for each product (if available), generates orders, and then returns:
          - The orders to be sent.
          - A conversion metric (placeholder in this case).
          - Serialized trader state for persistence between runs.
        
        Args:
            state (TradingState): The current market state including order depths and positions.
            
        Returns:
            tuple: A tuple containing:
                - result (dict): Mapping of product names to lists of Order objects.
                - conversions (int): A conversion metric (set as a placeholder value 1).
                - traderData (str): JSON-encoded string of trader's state (e.g., historical data).
        """
        result = {}

        # Parameters for RAINFOREST_RESIN
        rainforest_resin_fair_value = 10000  # Fixed fair value
        rainforest_resin_width = 2
        rainforest_resin_position_limit = 40

        # Parameters for KELP
        kelp_make_width = 4
        kelp_take_width = 1.35
        kelp_position_limit = 49
        kelp_timespan = 50
        
        # Optionally, previously stored trader state could be decoded here (currently commented out).
        # traderData = jsonpickle.decode(state.traderData)
        # self.kelp_prices = traderData["kelp_prices"]
        # self.kelp_vwap = traderData["kelp_vwap"]

        # Generate orders for RAINFOREST_RESIN if data is available.
        if "RAINFOREST_RESIN" in state.order_depths:
            rainforest_resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            rainforest_resin_orders = self.rainforest_resin_orders(
                state.order_depths["RAINFOREST_RESIN"], 
                rainforest_resin_fair_value, 
                rainforest_resin_width, 
                rainforest_resin_position, 
                rainforest_resin_position_limit)
            result["RAINFOREST_RESIN"] = rainforest_resin_orders

        # Generate orders for KELP if data is available.
        if "KELP" in state.order_depths:
            kelp_position = state.position["KELP"] if "KELP" in state.position else 0
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"], 
                kelp_timespan, 
                kelp_make_width, 
                kelp_take_width, 
                kelp_position, 
                kelp_position_limit)
            result["KELP"] = kelp_orders

        # Serialize the current trader state (historical data) for persistence.
        traderData = jsonpickle.encode({
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap
        })

        conversions = 1  # Placeholder conversion metric

        return result, conversions, traderData