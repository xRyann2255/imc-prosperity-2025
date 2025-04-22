from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Tuple
from collections import deque
import string
import jsonpickle
import numpy as np
import statistics
import math
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState




class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 20750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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
                    self.compress_state(state, self.truncate("state.traderData", max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    #self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

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

    def compress_observations(self, observations: Observation) -> list[Any]:
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

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self) -> None:
        self.position_limit = {
            "KELP" : 50,
            "RAINFOREST_RESIN" : 50,
            "SQUID_INK" : 50
        }
        
         
    def clear_position_order(self, orders: List[Order],
                            order_depth: OrderDepth, 
                            position: int, position_limit: int, 
                            product: str, buy_order_volume: int, 
                            sell_order_volume: int, 
                            fair_value: float, 
                            width: int                            
                                ) -> List[Order]:
        #calculate position after this iteration
        position_after_take = position + buy_order_volume - sell_order_volume

        min_fair_for_bid = math.floor(fair_value )
        max_fair_for_bid = math.floor(fair_value + width)

        min_fair_for_ask = math.ceil(fair_value - width)
        max_fair_for_ask = math.ceil(fair_value )
        
        fair_asks = reversed(list(range(min_fair_for_ask, max_fair_for_ask +1)))
        fair_bids = list(range(min_fair_for_bid, max_fair_for_bid +1))

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            for price in fair_asks:
                if price in order_depth.buy_orders.keys():
                    clear_quantity = min(order_depth.buy_orders[price], position_after_take)
                    sent_quantity = min(sell_quantity, clear_quantity)
                    orders.append(Order(product, price, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            for price in fair_bids:
                if price in order_depth.sell_orders.keys():
                    clear_quantity = min(abs(order_depth.sell_orders[price]), abs(position_after_take))
                    sent_quantity = min(buy_quantity, clear_quantity)
                    orders.append(Order(product, price, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)
                    break
    
        return buy_order_volume, sell_order_volume
    



    def rainforest_resin_orders(self, 
                                order_depth: OrderDepth, 
                                fair_value: int, 
                                width: int, 
                                position: int, 
                                position_limit: int
                                    ) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0
        #take mispriced orders
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1*order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_order_volume += quantity
        #neutral position
        buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, 0)
        #get best orders from bot to undercut
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + width]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - width]
        baaf = min(aaf) if len(aaf) > 0 else fair_value + width +1
        bbbf = max(bbf) if len(bbf) > 0 else fair_value - width -1
        #market make
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))  # Sell order

        return orders
    



    def kelp_orders(self, 
                    order_depth: OrderDepth, 
                    width: float, 
                    kelp_take_width: float, 
                    position: int, 
                    position_limit: int,
                    market_trades,
                    timestamp,
                    signal,
                    own_trades
                    
                        ) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        trades = [t for t in market_trades if t.timestamp == timestamp - 100]
        if trades:
            if any(trade.buyer == "Olivia" for trade in trades):
                signal = 1
                logger.print("found olivia buying!")

            if any(trade.seller == "Olivia" for trade in trades):
                signal = -1
                logger.print("found olivia selling!")

        my_trades = [t for t in own_trades if t.timestamp == timestamp - 100]
        if my_trades:
            if any(trade.buyer == "Olivia" for trade in my_trades):
                signal = 1
                logger.print("found olivia buying!")

            if any(trade.seller == "Olivia" for trade in my_trades):
                signal = -1
                logger.print("found olivia selling!")

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    
            #get actual midprice
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2    
            fair_value = mmmid_price

            #olivia things
            if signal == -1: #lower fairprice
                fair_value -= 0.5
            elif signal == 1: #higher fairprice
                fair_value += 0.5

            #take mispriced orders
            if best_ask <= fair_value - kelp_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 50:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_ask, quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + kelp_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 50:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_bid, -1 * quantity))
                        sell_order_volume += quantity

            #neutral position !!now based on olivia:
            

            #if signal == 0:
            if position + buy_order_volume -sell_order_volume < 37:
                buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 0)
            else: # we got too much stored -> negetivly impacts mm performance
                buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 0.5)


            #get best orders from bot to undercut
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + width]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - width]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + width +1
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - width -1
            #market make
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP", round(bbbf + 1), buy_quantity))  # Buy order

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", round(baaf - 1), -sell_quantity))  # Sell order

        return orders, signal



    def black_scholes_call(
            self, spot, strike, time_to_expiry, volatility
    ):
        d1 = (
            np.log(spot) - np.log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        call_price = spot * statistics.NormalDist(0,1).cdf(d1) - strike * statistics.NormalDist(0,1).cdf(d2)
        return call_price

    def implied_volatility(
            self, call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-6
    ):
        low_vol = 0.0001
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = self.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


    def z_score_intrinsic_value_scaling(self, underlying_price, strike, TTE, spot, volatility):

        #scaling = (1 + 0.1* np.square((underlying_price-strike)/400) + 0.4 * np.square(np.square((underlying_price-strike)/400)) )
        scaling = (1 + 0.3* np.square((underlying_price-strike)/500) + 0.2 * np.square(np.square((underlying_price-strike)/500)) )
        #scaling = self.gamma(spot, spot, TTE, volatility) / self.gamma(spot, strike, TTE, volatility)
        #scaling = pow(self.gamma(spot, spot, TTE, volatility) / self.gamma(spot, strike, TTE, volatility),0.262)
        exit = 0.3 - 0.1 * pow((underlying_price-strike)/300,2) - 0.1 * pow ((underlying_price-strike)/300, 6)
        return scaling, exit

    def voucher_order(
            self, order_depth: OrderDepth, moneyness_history, market_iv_history, residual_history, 
            underlying_price, strike, TTE, 
            product, position, position_limit,
            entry_thresh, exit_thresh
    ):
        orders: List[Order] = []

        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:  
            return orders
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        #moneyness
        moneyness = np.log(strike/underlying_price) / np.sqrt(TTE)
        
        #market iv
        market_iv = self.implied_volatility(mid_price, underlying_price, strike, TTE)
        
        #fit iv
        
        # old parabola from Xilos :)
        # a = 0.2550023371712128675
        # b = 0.014855875346974514 
        # c = 0.15026516778454865
        # tag 0-4 parabel 
        a = 0.26109938
        b = 0.01357543  
        c = 0.14924834
        # tag 4 parabel
        # a = 0.27628128
        # b = 0.01200458
        # c = 0.15012026
        coeff = [a,b,c]
        fit_iv = np.polyval(coeff, moneyness)

        #residual
        residual = market_iv - fit_iv
        residual_history.append(residual)
        

        if len(residual_history) < 20:
            
            return orders

        #zscore
        residual_mean = np.mean(residual_history)
        residual_stdev = np.std(residual_history)
        z_score = (residual - residual_mean) / residual_stdev

        #check mispriced vol and trade

        #new logic, look at market iv for buy/sell side seperate:
        
        scaling,exit = self.z_score_intrinsic_value_scaling(underlying_price, strike, TTE, underlying_price, market_iv)

        for price in order_depth.sell_orders.keys():
            ask_market_iv = self.implied_volatility(price,underlying_price,strike,TTE)
            ask_residual = ask_market_iv - fit_iv
            ask_z_score = (ask_residual - residual_mean) / residual_stdev
            if ask_z_score < -entry_thresh*scaling: #underpriced -> buy
                buy_quantity = min(position_limit - position, abs(order_depth.sell_orders[price]))
                orders.append(Order(product, price, buy_quantity))
            elif position < 0 and ask_z_score <-exit_thresh/scaling: #fair again -> buy
                buy_quantity = min(abs(position), abs(order_depth.sell_orders[price]))
                orders.append(Order(product, price, buy_quantity))

        #buy orders
        for price in order_depth.buy_orders.keys():
            buy_market_iv = self.implied_volatility(price,underlying_price,strike,TTE)
            buy_residual = buy_market_iv - fit_iv
            buy_z_score = (buy_residual - residual_mean) / residual_stdev
            if buy_z_score > entry_thresh*scaling: #overpriced -> sell
                sell_quantity = min(position_limit + position, abs(order_depth.buy_orders[price]))
                orders.append(Order(product, price, -sell_quantity))
            elif position > 0 and buy_z_score > exit_thresh/scaling: #fair again -> sell
                sell_quantity = min(abs(position), abs(order_depth.buy_orders[price]))
                orders.append(Order(product, price, -sell_quantity))

        return orders
    

    def macaron_order(
            self, order_depth: OrderDepth, position, position_limit, observations,
        ):
        orders: List[Order] = []

        away_ask = observations.askPrice
        away_bid = observations.bidPrice
        importT = observations.importTariff
        transport = observations.transportFees
        exportT = observations.exportTariff

        effective_away_ask = away_ask + importT + transport
        effective_away_bid = away_bid - exportT - transport

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        #import 
        #bot filled 60% bei away_bid + 0,5
        sell_price = max(int(away_bid+0.5), round(effective_away_ask+1))
        orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -10))

        return orders
    

    def basket_signal(
            self, order_depths: OrderDepth, positions, 
            market_trades, timestamp, signal
        ) -> dict[str, List[Order]] :
        orders: dict[str, List[Order]] = {}

        position_limits = {
            "PICNIC_BASKET1": 50,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 50,
        }

        signal_direction = {
            "PICNIC_BASKET1": 1,
            "PICNIC_BASKET2": 1,
            "CROISSANTS": 1,
            "JAMS": -1,
            "DJEMBES": -1,
        }

        trades = [t for t in market_trades if t.timestamp == timestamp - 100]
        logger.print("trades: "+ str(trades))
        if trades:
            if any(trade.buyer == "Olivia" for trade in trades):
                signal = 1

            if any(trade.seller == "Olivia" for trade in trades):
                signal = -1


        for product in order_depths:
            if "BASKET" in product or "JAM" in product or "CROIS" in product or "DJEM" in product:
                orders[product] = []
                sell_quantity, buy_quantity = 0,0
                sell_volume, buy_volume = 0,0

                position = positions[product] if product in positions else 0

                best_ask = min(order_depths[product].sell_orders.keys())
                best_bid = max(order_depths[product].buy_orders.keys())


                if best_ask:
                    if signal * signal_direction[product] == 1: # long
                        for price in order_depths[product].sell_orders.keys():
                            buy_quantity = min(position_limits[product] - position - buy_volume, abs(order_depths[product].sell_orders[price]))
                            buy_volume += buy_quantity
                            orders[product].append(Order(product, price, buy_quantity))

                if best_bid:
                    if signal * signal_direction[product] == -1: #short
                        for price in order_depths[product].buy_orders.keys():
                            sell_quantity = min(position_limits[product] + position - sell_volume, abs(order_depths[product].buy_orders[price]))
                            sell_volume += sell_quantity
                            orders[product].append(Order(product, price, -sell_quantity))

        return orders, signal
        
    def ink_signal(
            self, product, order_depth: OrderDepth, position, position_limit, market_trades, timestamp, signal
        ):
        orders: List[Order] = []
        
        trades = [t for t in market_trades if t.timestamp == timestamp - 100]
        if trades:
            if any(trade.buyer == "Olivia" for trade in trades):
                signal = 1

            if any(trade.seller == "Olivia" for trade in trades):
                signal = -1

        sell_quantity, buy_quantity = 0,0
        sell_volume, buy_volume = 0,0
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        if best_ask:
            if signal == 1: # long
                for price in order_depth.sell_orders.keys():
                    buy_quantity = min(position_limit - position - buy_volume, abs(order_depth.sell_orders[price]))
                    buy_volume += buy_quantity
                    orders.append(Order(product, price, buy_quantity))

                
        if best_bid:
            if signal == -1: #short
                for price in order_depth.buy_orders.keys():
                    sell_quantity = min(position_limit + position - sell_volume, abs(order_depth.buy_orders[price]))
                    sell_volume += sell_quantity
                    orders.append(Order(product, price, -sell_quantity))

        return orders, signal
    
    def take_free_voucher(
            self, product, position, position_limit
    ):
        orders: List[Order] = []

        #buy free oders
        buy_quantity = position_limit -position
        orders.append(Order(product, 0, buy_quantity))

        #luck hedge or free 1k?
        # if position > 0:
        #     sell_quantity = position
        #     orders.append(Order(product, 1, -sell_quantity))

        return orders

    def run(self, state: TradingState):
        result = {}
#-------HYPER_PARAMETERS:------------------
    # Round 1: MM
        # RAINFOREST_RESIN -> rr
        rr_position_limit = 50
        rr_fair_value = 10000  
        rr_width = 1
        
        # KELP -> k
        k_position_limit = 50
        k_make_width = 1
        k_take_width = 2
    
        # SQUID_INK -> sq
        sq_position_limit = 50
    # Round 2: Pair Trading

        # PICNIC_BASKET1 -> pb1
        pb1_position_limit = 60

        # PICNIC_BASKET2 -> pb2
        pb2_position_limit = 100

        # DJEMBES -> d
        d_position_limit = 60

        # JAMS -> j
        j_position_limit = 350

        # CROISSANTS -> c
        c_position_limit = 250

        # Pairs:
        pb2_synth_spread_avg = 30.235
        pb2_synth_spread_std = 59.849
        pb2_synth_z_thresh = 0.75
        pb2_synth_exit_thesh = -0.2
        pb2_synth_timeframe = 300
        pb2_synth_std_scaling = 2.5
        pb2_synth_std_bandwidth = 30


        pb1_pb2_spread_avg = 3.408
        pb1_pb2_spread_std = 93.506
        pb1_pb2_z_thresh = 0.75
        pb1_pb2_exit_thesh = -1
        pb1_pb2_timeframe = 300
        pb1_pb2_std_scaling = 3
        pb1_pb2_std_bandwidth = 53

    # Round 3: Option Pricing   

        # VOLCANIC_ROCK -> vr
        vr_position_limit = 400
        vr_entry_thresh = 16
        midprice_timeframe = 100
        

        # VOLCANIC_ROCK_VOUCHER -> vrv
        vrv_position_limit = 200

        # VOLCANIC_ROCK_VOUCHER_9500 -> v095
        v095_strike = 9500
        v095_spread_avg = 0
        v095_entry_thresh = 0.9
        v095_exit_thresh = -0.3

        # VOLCANIC_ROCK_VOUCHER_9750 -> v097
        v097_strike = 9750
        v097_spread_avg = 0
        v097_entry_thresh = 0.5
        v097_exit_thresh = 0.2

        # VOLCANIC_ROCK_VOUCHER_10000 -> v100
        v100_strike = 10000
        v100_spread_avg = 0
        v100_entry_thresh = 0.5
        v100_exit_thresh = 0.2

        # VOLCANIC_ROCK_VOUCHER_10250 -> v102
        v102_strike = 10250
        v102_spread_avg = 0
        v102_entry_thresh = 0.5
        v102_exit_thresh = 0.2

        # Options
        TTE = (3 - state.timestamp/1000000) / 365
        logger.print("TTE:" + str(TTE))
        r = 0  
        max_lookback = 20  

    # Round 4: Arb
        # MAGNIFICENT_MACARONS -> m
        m_position_limit = 75
        
#-------load traderData--------------------
        if state.traderData == "":
            trader_data = { "v095_moneyness_history" :  deque(maxlen = max_lookback),
                            "v095_market_iv_history" :  deque(maxlen = max_lookback),
                            "v095_residual_history" :   deque(maxlen = max_lookback),

                            "v097_moneyness_history" :  deque(maxlen = max_lookback),
                            "v097_market_iv_history" :  deque(maxlen = max_lookback),
                            "v097_residual_history" :   deque(maxlen = max_lookback),

                            "v100_moneyness_history" :  deque(maxlen = max_lookback),
                            "v100_market_iv_history" :  deque(maxlen = max_lookback),
                            "v100_residual_history" :   deque(maxlen = max_lookback),

                            "v102_moneyness_history" :  deque(maxlen = max_lookback),
                            "v102_market_iv_history" :  deque(maxlen = max_lookback),
                            "v102_residual_history" :   deque(maxlen = max_lookback),

                            "v105_moneyness_history" :  deque(maxlen = max_lookback),
                            "v105_market_iv_history" :  deque(maxlen = max_lookback),
                            "v105_residual_history" :   deque(maxlen = max_lookback),

                            "ink_signal": 0,
                            "croissants_signal" : 0,
                            "kelp_signal" : 0,

                           }
            state.traderData = trader_data            
        else:
            state.traderData = jsonpickle.loads(state.traderData)

#-------orders for products----------------

        c_market_trades = state.market_trades["CROISSANTS"] if "CROISSANTS" in state.market_trades else []
        result, state.traderData["croissants_signal"] = self.basket_signal( state.order_depths, state.position, 
                                                                                c_market_trades, state.timestamp, state.traderData["croissants_signal"])
        
        if "RAINFOREST_RESIN" in state.order_depths:
            rr_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            rr_orders = self.rainforest_resin_orders(state.order_depths["RAINFOREST_RESIN"], rr_fair_value, rr_width, rr_position, rr_position_limit)
            result["RAINFOREST_RESIN"] = rr_orders
        
        if "KELP" in state.order_depths:
            k_position = state.position["KELP"] if "KELP" in state.position else 0
            k_market_trades = state.market_trades["KELP"] if "KELP" in state.market_trades else []
            k_own_trades = state.own_trades["KELP"] if "KELP" in state.own_trades else []
            result["KELP"],state.traderData["kelp_signal"] = self.kelp_orders(state.order_depths["KELP"], k_make_width, k_take_width, k_position, k_position_limit,
                                                                                k_market_trades, state.timestamp, state.traderData["kelp_signal"], k_own_trades)            

        if "SQUID_INK" in state.order_depths:
            sq_position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
            sq_market_trades = state.market_trades["SQUID_INK"] if "SQUID_INK" in state.market_trades else []
            result["SQUID_INK"], state.traderData["ink_signal"] = self.ink_signal(  "SQUID_INK", state.order_depths["SQUID_INK"], sq_position, sq_position_limit,
                                                                                    sq_market_trades, state.timestamp, state.traderData["ink_signal"])           
 
        if "VOLCANIC_ROCK" in state.order_depths:
            vr_position = state.position["VOLCANIC_ROCK"] if "VOLCANIC_ROCK" in state.position else 0
            if len(state.order_depths["VOLCANIC_ROCK"].sell_orders) != 0 and len(state.order_depths["VOLCANIC_ROCK"].buy_orders) != 0:
                vr_best_ask = min(state.order_depths["VOLCANIC_ROCK"].sell_orders.keys())
                vr_best_bid = max(state.order_depths["VOLCANIC_ROCK"].buy_orders.keys())
                vr_mid_price = (vr_best_ask + vr_best_bid) / 2
                #iv smile
                if "VOLCANIC_ROCK_VOUCHER_10250" in state.order_depths:
                    v102_position = state.position["VOLCANIC_ROCK_VOUCHER_10250"] if "VOLCANIC_ROCK_VOUCHER_10250" in state.position else 0
                    result["VOLCANIC_ROCK_VOUCHER_10250"] = self.voucher_order(
                                                                            state.order_depths["VOLCANIC_ROCK_VOUCHER_10250"], state.traderData["v102_moneyness_history"],
                                                                            state.traderData["v102_market_iv_history"], state.traderData["v102_residual_history"], 
                                                                            vr_mid_price, v102_strike, TTE, "VOLCANIC_ROCK_VOUCHER_10250" , v102_position, vrv_position_limit,
                                                                            v102_entry_thresh, v102_exit_thresh,
                                                                        )

                if "VOLCANIC_ROCK_VOUCHER_10000" in state.order_depths:
                    v100_position = state.position["VOLCANIC_ROCK_VOUCHER_10000"] if "VOLCANIC_ROCK_VOUCHER_10000" in state.position else 0
                    result["VOLCANIC_ROCK_VOUCHER_10000"] = self.voucher_order(
                                                                            state.order_depths["VOLCANIC_ROCK_VOUCHER_10000"], state.traderData["v100_moneyness_history"],
                                                                            state.traderData["v100_market_iv_history"], state.traderData["v100_residual_history"], 
                                                                            vr_mid_price, v100_strike, TTE, "VOLCANIC_ROCK_VOUCHER_10000" , v100_position, vrv_position_limit,
                                                                            v100_entry_thresh, v100_exit_thresh
                                                                        )
                    
                
                if "VOLCANIC_ROCK_VOUCHER_9750" in state.order_depths:
                    v097_position = state.position["VOLCANIC_ROCK_VOUCHER_9750"] if "VOLCANIC_ROCK_VOUCHER_9750" in state.position else 0
                    result["VOLCANIC_ROCK_VOUCHER_9750"] = self.voucher_order(
                                                                            state.order_depths["VOLCANIC_ROCK_VOUCHER_9750"], state.traderData["v097_moneyness_history"],
                                                                            state.traderData["v097_market_iv_history"], state.traderData["v097_residual_history"], 
                                                                            vr_mid_price, v097_strike, TTE, "VOLCANIC_ROCK_VOUCHER_9750" , v097_position, vrv_position_limit,
                                                                            v097_entry_thresh, v097_exit_thresh
                                                                        )
                    
                # if "VOLCANIC_ROCK_VOUCHER_9500" in state.order_depths:
                #     v095_position = state.position["VOLCANIC_ROCK_VOUCHER_9500"] if "VOLCANIC_ROCK_VOUCHER_9500" in state.position else 0
                #     result["VOLCANIC_ROCK_VOUCHER_9500"] = self.voucher_order(
                #                                                             state.order_depths["VOLCANIC_ROCK_VOUCHER_9500"], state.traderData["v095_moneyness_history"],
                #                                                             state.traderData["v095_market_iv_history"], state.traderData["v095_residual_history"], 
                #                                                             vr_mid_price, v095_strike, TTE, "VOLCANIC_ROCK_VOUCHER_9500" , v095_position, vrv_position_limit,
                #                                                             v095_entry_thresh, v095_exit_thresh
                #                                                         )
        
        if "VOLCANIC_ROCK_VOUCHER_10500" in state.order_depths:
            v105_position = state.position["VOLCANIC_ROCK_VOUCHER_10500"] if "VOLCANIC_ROCK_VOUCHER_10500" in state.position else 0
            result["VOLCANIC_ROCK_VOUCHER_10500"]= self.take_free_voucher(  "VOLCANIC_ROCK_VOUCHER_10500", v105_position, vrv_position_limit)



        if "MAGNIFICENT_MACARONS" in state.order_depths:
            m_position = state.position["MAGNIFICENT_MACARONS"] if "MAGNIFICENT_MACARONS" in state.position else 0
            result["MAGNIFICENT_MACARONS"] = self.macaron_order(    state.order_depths["MAGNIFICENT_MACARONS"], m_position, m_position_limit,
                                                                    state.observations.conversionObservations["MAGNIFICENT_MACARONS"])


        traderData = jsonpickle.encode(state.traderData)
        conversions = 0
        if m_position < 0:
            conversions = min(abs(m_position),10)
        if m_position > 0:
            conversions = -min(abs(m_position),10)
        logger.flush(state, result, conversions, state.traderData)
        return result, conversions, traderData

    