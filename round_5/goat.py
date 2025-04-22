import json
import jsonpickle
import numpy as np
import statistics
import math
from collections import deque
from typing import List, Any

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


# Logger Template
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 20750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
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
                    self.compress_state(
                        state, self.truncate("state.traderData", max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    # self.truncate(trader_data, max_item_length),
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

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
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
        # Set position limits
        self.position_limit = {"KELP": 50, "RAINFOREST_RESIN": 50, "SQUID_INK": 50}

    def clear_position_order(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        pos: int,
        position_limit: int,
        product: str,
        buy_vol: int,
        sell_vol: int,
        fair: float,
        w: int,
    ) -> List[Order]:
        next_pos = pos + buy_vol - sell_vol

        # see how much we can buy/sell to get to the position limit
        buy_amt = position_limit - (pos + buy_vol)
        sell_amt = position_limit + (pos - sell_vol)

        # find fair prices
        min_fair_bid = math.floor(fair)
        max_fair_bid = math.floor(fair + w)
        min_fair_ask = math.ceil(fair - w)
        max_fair_ask = math.ceil(fair)

        fair_asks = reversed(list(range(min_fair_ask, max_fair_ask + 1)))
        fair_bids = list(range(min_fair_bid, max_fair_bid + 1))

        if next_pos > 0:
            for price in fair_asks:
                if price in order_depth.buy_orders.keys():
                    clear_quantity = min(order_depth.buy_orders[price], next_pos)
                    quantity = min(sell_amt, clear_quantity)
                    orders.append(Order(product, price, -abs(quantity)))
                    sell_vol += abs(quantity)

        if next_pos < 0:
            for price in fair_bids:
                if price in order_depth.sell_orders.keys():
                    clear_quantity = min(
                        abs(order_depth.sell_orders[price]), abs(next_pos)
                    )
                    quantity = min(buy_amt, clear_quantity)
                    orders.append(Order(product, price, abs(quantity)))
                    buy_vol += abs(quantity)
                    break

        return buy_vol, sell_vol

    def rainforest_resin_orders(
        self,
        order_depth: OrderDepth,
        fair: int,
        w: int,
        pos: int,
        pos_lim: int,
    ) -> List[Order]:
        orders: List[Order] = []

        buy_vol = 0
        sell_vol = 0

        # BUY
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amt = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair:
                quantity = min(best_ask_amt, pos_lim - pos)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_vol += quantity

        # SELL
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amt = order_depth.buy_orders[best_bid]
            if best_bid > fair:
                quantity = min(best_bid_amt, pos_lim + pos)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    sell_vol += quantity

        # CLEAR POS
        buy_vol, sell_vol = self.clear_position_order(
            orders,
            order_depth,
            pos,
            pos_lim,
            "RAINFOREST_RESIN",
            buy_vol,
            sell_vol,
            fair,
            0,
        )

        # some classic stuff
        aaf = [price for price in order_depth.sell_orders.keys() if price > fair + w]
        bbf = [price for price in order_depth.buy_orders.keys() if price < fair - w]
        baaf = min(aaf) if len(aaf) > 0 else fair + w + 1
        bbbf = max(bbf) if len(bbf) > 0 else fair - w - 1

        buy_q = pos_lim - (pos + buy_vol)
        if buy_q > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_q))

        sell_q = pos_lim + (pos - sell_vol)
        if sell_q > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_q))

        return orders

    def kelp_orders(
        self,
        order_depth: OrderDepth,
        w: float,
        kelp_take_width: float,
        pos: int,
        pos_lim: int,
        market_trades,
        timestamp,
        sig,
        ours,
    ) -> List[Order]:
        orders: List[Order] = []

        # Filter trades to include only those that occurred exactly 100 milliseconds before the current timestamp
        trades = []
        for trade in market_trades:
            if trade.timestamp == timestamp - 100:
                trades.append(trade)

        if trades:
            if any(trade.buyer == "Olivia" for trade in trades):
                sig = 1

            if any(trade.seller == "Olivia" for trade in trades):
                sig = -1

        # Same
        other = []
        for trade in ours:
            if trade.timestamp == timestamp - 100:
                other.append(trade)

        if other:
            if any(trade.buyer == "Olivia" for trade in other):
                sig = 1

            if any(trade.seller == "Olivia" for trade in other):
                sig = -1

        buy_vol = 0
        sell_vol = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            f_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= 15
            ]
            f_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= 15
            ]
            mm_ask = min(f_ask) if len(f_ask) > 0 else best_ask
            mm_bid = max(f_bid) if len(f_bid) > 0 else best_bid

            mmmid_price = (mm_ask + mm_bid) / 2
            fair_value = mmmid_price

            if sig == -1:
                fair_value -= 0.5
            elif sig == 1:
                fair_value += 0.5

            if best_ask <= fair_value - kelp_take_width:
                ask_amt = -1 * order_depth.sell_orders[best_ask]
                if ask_amt <= 50:
                    quantity = min(ask_amt, pos_lim - pos)
                    if quantity > 0:
                        orders.append(Order("KELP", best_ask, quantity))
                        buy_vol += quantity
            if best_bid >= fair_value + kelp_take_width:
                bid_amt = order_depth.buy_orders[best_bid]
                if bid_amt <= 50:
                    quantity = min(bid_amt, pos_lim + pos)
                    if quantity > 0:
                        orders.append(Order("KELP", best_bid, -1 * quantity))
                        sell_vol += quantity

            if pos + buy_vol - sell_vol < 37:
                buy_vol, sell_vol = self.clear_position_order(
                    orders,
                    order_depth,
                    pos,
                    pos_lim,
                    "KELP",
                    buy_vol,
                    sell_vol,
                    fair_value,
                    0,
                )
            else:
                buy_vol, sell_vol = self.clear_position_order(
                    orders,
                    order_depth,
                    pos,
                    pos_lim,
                    "KELP",
                    buy_vol,
                    sell_vol,
                    fair_value,
                    0.5,
                )

            # as before
            aaf = [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + w
            ]
            bbf = [
                price
                for price in order_depth.buy_orders.keys()
                if price < fair_value - w
            ]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + w + 1
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - w - 1

            # BUY
            buy_q = pos_lim - (pos + buy_vol)
            if buy_q > 0:
                orders.append(Order("KELP", round(bbbf + 1), buy_q))

            # SELL
            sell_q = pos_lim + (pos - sell_vol)
            if sell_q > 0:
                orders.append(Order("KELP", round(baaf - 1), -sell_q))  # Sell order

        return orders, sig

    def black_scholes(self, spot, strike, t_expiry, vol):
        d1 = (np.log(spot) - np.log(strike) + (0.5 * vol * vol) * t_expiry) / (
            vol * np.sqrt(t_expiry)
        )
        d2 = d1 - vol * np.sqrt(t_expiry)
        call = spot * statistics.NormalDist(0, 1).cdf(
            d1
        ) - strike * statistics.NormalDist(0, 1).cdf(d2)
        return call

    def impllied_vol(
        self,
        call,
        spot,
        strike,
        t_expiry,
        max_iter=200,
        tol=1e-6,
    ):
        low_vol = 0.0001
        high_vol = 1.0
        vol = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iter):
            price = self.black_scholes(spot, strike, t_expiry, vol)
            diff = price - call
            if abs(diff) < tol:
                break
            elif diff > 0:
                high_vol = vol
            else:
                low_vol = vol
            vol = (low_vol + high_vol) / 2.0
        return vol

    def z_scale(self, underlying_price, strike):

        # Now  scale
        scale = (
            1
            + 0.3 * np.square((underlying_price - strike) / 500)
            + 0.2 * np.square(np.square((underlying_price - strike) / 500))
        )
        return scale

    def voucher_calcs(
        self,
        order_depth: OrderDepth,
        residuals,
        underlying,
        strike,
        TTE,
        product,
        pos,
        pos_lim,
        entry_threshold,
        exit_threshold,
    ):
        orders: List[Order] = []

        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2

        moneyness = np.log(strike / underlying) / np.sqrt(TTE)

        m_iv = self.impllied_vol(mid_price, underlying, strike, TTE)

        x1 = 0.26109938
        x2 = 0.01357543
        x3 = 0.14924834

        coeff = [x1, x2, x3]
        fit = np.polyval(coeff, moneyness)

        residual = m_iv - fit
        residuals.append(residual)

        if len(residuals) < 20:
            return orders

        # calculate z-score
        residual_mean = np.mean(residuals)
        residual_stdev = np.std(residuals)

        scaling = self.z_scale(underlying, strike)

        # SELL
        for price in order_depth.sell_orders.keys():
            ask_market_iv = self.impllied_vol(price, underlying, strike, TTE)
            ask_residual = ask_market_iv - fit
            ask_z_score = (ask_residual - residual_mean) / residual_stdev
            if ask_z_score < -entry_threshold * scaling:
                buy_quantity = min(pos_lim - pos, abs(order_depth.sell_orders[price]))
                orders.append(Order(product, price, buy_quantity))
            elif pos < 0 and ask_z_score < -exit_threshold / scaling:
                buy_quantity = min(abs(pos), abs(order_depth.sell_orders[price]))
                orders.append(Order(product, price, buy_quantity))

        # BUY
        for price in order_depth.buy_orders.keys():
            buy_market_iv = self.impllied_vol(price, underlying, strike, TTE)
            buy_residual = buy_market_iv - fit
            buy_z_score = (buy_residual - residual_mean) / residual_stdev
            if buy_z_score > entry_threshold * scaling:
                sell_quantity = min(pos_lim + pos, abs(order_depth.buy_orders[price]))
                orders.append(Order(product, price, -sell_quantity))
            elif pos > 0 and buy_z_score > exit_threshold / scaling:
                sell_quantity = min(abs(pos), abs(order_depth.buy_orders[price]))
                orders.append(Order(product, price, -sell_quantity))

        return orders

    def macaron_order(
        self,
        observations,
    ):
        orders: List[Order] = []

        aask = observations.askPrice
        abid = observations.bidPrice
        import_tariff = observations.importTariff
        transport_fees = observations.transportFees

        eff = aask + import_tariff + transport_fees

        sell_at = max(int(abid + 0.5), round(eff + 1))
        orders.append(Order("MAGNIFICENT_MACARONS", sell_at, -10))

        return orders

    def basket_sig(
        self, order_depths: OrderDepth, positions, market_trades, timestamp, sig
    ) -> dict[str, List[Order]]:
        orders: dict[str, List[Order]] = {}

        pos_lims = {
            "PICNIC_BASKET1": 50,
            "PICNIC_BASKET2": 100,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 50,
        }

        s_dir = {
            "PICNIC_BASKET1": 1,
            "PICNIC_BASKET2": 1,
            "CROISSANTS": 1,
            "JAMS": -1,
            "DJEMBES": -1,
        }

        trades = []
        for trade in market_trades:
            if trade.timestamp == timestamp - 100:
                trades.append(trade)

        if trades:
            if any(trade.buyer == "Olivia" for trade in trades):
                sig = 1

            if any(trade.seller == "Olivia" for trade in trades):
                sig = -1

        for product in order_depths:
            if (
                "BASKET" in product
                or "JAM" in product
                or "CROIS" in product
                or "DJEM" in product
            ):
                orders[product] = []

                bask = min(order_depths[product].sell_orders.keys())
                bbid = max(order_depths[product].buy_orders.keys())
                position = positions[product] if product in positions else 0
                sell_q, buy_q = 0, 0
                sell_vol, buy_vol = 0, 0

                # LONG
                if bask:
                    if sig * s_dir[product] == 1:
                        for price in order_depths[product].sell_orders.keys():
                            buy_q = min(
                                pos_lims[product] - position - buy_vol,
                                abs(order_depths[product].sell_orders[price]),
                            )
                            buy_vol += buy_q
                            orders[product].append(Order(product, price, buy_q))

                # SHORT
                if bbid:
                    if sig * s_dir[product] == -1:
                        for price in order_depths[product].buy_orders.keys():
                            sell_q = min(
                                pos_lims[product] + position - sell_vol,
                                abs(order_depths[product].buy_orders[price]),
                            )
                            sell_vol += sell_q
                            orders[product].append(Order(product, price, -sell_q))

        return orders, sig

    def ink_signal(
        self,
        product,
        order_depth: OrderDepth,
        pos,
        pos_lim,
        m_tr,
        timestamp,
        sig,
    ):
        orders: List[Order] = []

        trades = [t for t in m_tr if t.timestamp == timestamp - 100]
        if trades:
            if any(trade.buyer == "Olivia" for trade in trades):
                sig = 1

            if any(trade.seller == "Olivia" for trade in trades):
                sig = -1

        sell_q, buy_q = 0, 0
        sell_v, buy_v = 0, 0
        bask = min(order_depth.sell_orders.keys())
        bbid = max(order_depth.buy_orders.keys())

        # LONG
        if bask:
            if sig == 1:
                for price in order_depth.sell_orders.keys():
                    buy_q = min(
                        pos_lim - pos - buy_v,
                        abs(order_depth.sell_orders[price]),
                    )
                    buy_v += buy_q
                    orders.append(Order(product, price, buy_q))

        # SHORT
        if bbid:
            if sig == -1:
                for price in order_depth.buy_orders.keys():
                    sell_q = min(
                        pos_lim + pos - sell_v,
                        abs(order_depth.buy_orders[price]),
                    )
                    sell_v += sell_q
                    orders.append(Order(product, price, -sell_q))

        return orders, sig

    def take_free_voucher(self, product, pos, pos_lim):
        orders: List[Order] = []

        buy_q = pos_lim - pos
        orders.append(Order(product, 0, buy_q))

        return orders

    def run(self, state: TradingState):
        result = {}
        # LIMITS
        rainforest_resin_pos_lim = 50
        kelp_pos_lim = 50
        squid_ink_pos_lim = 50

        # OTHER STUFF
        rainforest_resin_fair = 10000
        rainforest_resin_w = 1
        kelp_make_w = 1
        kelp_take_w = 2

        # Volcano
        volc_rv_pos_lim = 200
        v097_strike = 9750
        v100_strike = 10000
        v102_strike = 10250
        v097_entry_threshold = 0.5
        v100_entry_threshold = 0.5
        v102_entry_threshold = 0.5
        v097_exit_threshold = 0.2
        v100_exit_threshold = 0.2
        v102_exit_threshold = 0.2

        TTE = (3 - state.timestamp / 1000000) / 365
        lb_max = 20

        if state.traderData == "":
            trader_data = {
                "v095_moneyness_history": deque(maxlen=lb_max),
                "v095_market_iv_history": deque(maxlen=lb_max),
                "v095_residual_history": deque(maxlen=lb_max),
                "v097_moneyness_history": deque(maxlen=lb_max),
                "v097_market_iv_history": deque(maxlen=lb_max),
                "v097_residual_history": deque(maxlen=lb_max),
                "v100_moneyness_history": deque(maxlen=lb_max),
                "v100_market_iv_history": deque(maxlen=lb_max),
                "v100_residual_history": deque(maxlen=lb_max),
                "v102_moneyness_history": deque(maxlen=lb_max),
                "v102_market_iv_history": deque(maxlen=lb_max),
                "v102_residual_history": deque(maxlen=lb_max),
                "v105_moneyness_history": deque(maxlen=lb_max),
                "v105_market_iv_history": deque(maxlen=lb_max),
                "v105_residual_history": deque(maxlen=lb_max),
                "ink_signal": 0,
                "croissants_signal": 0,
                "kelp_signal": 0,
            }
            state.traderData = trader_data
        else:
            state.traderData = jsonpickle.loads(state.traderData)

        c_market_trades = (
            state.market_trades["CROISSANTS"]
            if "CROISSANTS" in state.market_trades
            else []
        )
        result, state.traderData["croissants_signal"] = self.basket_sig(
            state.order_depths,
            state.position,
            c_market_trades,
            state.timestamp,
            state.traderData["croissants_signal"],
        )

        if "RAINFOREST_RESIN" in state.order_depths:
            rainforest_resin_pos = (
                state.position["RAINFOREST_RESIN"]
                if "RAINFOREST_RESIN" in state.position
                else 0
            )
            rainforest_resin_orders = self.rainforest_resin_orders(
                state.order_depths["RAINFOREST_RESIN"],
                rainforest_resin_fair,
                rainforest_resin_w,
                rainforest_resin_pos,
                rainforest_resin_pos_lim,
            )
            result["RAINFOREST_RESIN"] = rainforest_resin_orders

        if "KELP" in state.order_depths:
            kelp_pos = state.position["KELP"] if "KELP" in state.position else 0
            kelp_m_tr = (
                state.market_trades["KELP"] if "KELP" in state.market_trades else []
            )
            kelp_ours = state.own_trades["KELP"] if "KELP" in state.own_trades else []
            result["KELP"], state.traderData["kelp_signal"] = self.kelp_orders(
                state.order_depths["KELP"],
                kelp_make_w,
                kelp_take_w,
                kelp_pos,
                kelp_pos_lim,
                kelp_m_tr,
                state.timestamp,
                state.traderData["kelp_signal"],
                kelp_ours,
            )

        if "SQUID_INK" in state.order_depths:
            squid_ink_pos = (
                state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
            )
            squid_ink_m_tr = (
                state.market_trades["SQUID_INK"]
                if "SQUID_INK" in state.market_trades
                else []
            )
            result["SQUID_INK"], state.traderData["ink_signal"] = self.ink_signal(
                "SQUID_INK",
                state.order_depths["SQUID_INK"],
                squid_ink_pos,
                squid_ink_pos_lim,
                squid_ink_m_tr,
                state.timestamp,
                state.traderData["ink_signal"],
            )

        if "VOLCANIC_ROCK" in state.order_depths:
            if (
                len(state.order_depths["VOLCANIC_ROCK"].sell_orders) != 0
                and len(state.order_depths["VOLCANIC_ROCK"].buy_orders) != 0
            ):
                volcanic_rock_bask = min(
                    state.order_depths["VOLCANIC_ROCK"].sell_orders.keys()
                )
                volcanic_rock_bbid = max(
                    state.order_depths["VOLCANIC_ROCK"].buy_orders.keys()
                )
                volcanic_rock_mid = (volcanic_rock_bask + volcanic_rock_bbid) / 2
                if "VOLCANIC_ROCK_VOUCHER_10250" in state.order_depths:
                    v102_position = (
                        state.position["VOLCANIC_ROCK_VOUCHER_10250"]
                        if "VOLCANIC_ROCK_VOUCHER_10250" in state.position
                        else 0
                    )
                    result["VOLCANIC_ROCK_VOUCHER_10250"] = self.voucher_calcs(
                        state.order_depths["VOLCANIC_ROCK_VOUCHER_10250"],
                        state.traderData["v102_residual_history"],
                        volcanic_rock_mid,
                        v102_strike,
                        TTE,
                        "VOLCANIC_ROCK_VOUCHER_10250",
                        v102_position,
                        volc_rv_pos_lim,
                        v102_entry_threshold,
                        v102_exit_threshold,
                    )

                if "VOLCANIC_ROCK_VOUCHER_10000" in state.order_depths:
                    v100_position = (
                        state.position["VOLCANIC_ROCK_VOUCHER_10000"]
                        if "VOLCANIC_ROCK_VOUCHER_10000" in state.position
                        else 0
                    )
                    result["VOLCANIC_ROCK_VOUCHER_10000"] = self.voucher_calcs(
                        state.order_depths["VOLCANIC_ROCK_VOUCHER_10000"],
                        state.traderData["v100_residual_history"],
                        volcanic_rock_mid,
                        v100_strike,
                        TTE,
                        "VOLCANIC_ROCK_VOUCHER_10000",
                        v100_position,
                        volc_rv_pos_lim,
                        v100_entry_threshold,
                        v100_exit_threshold,
                    )

                if "VOLCANIC_ROCK_VOUCHER_9750" in state.order_depths:
                    v097_position = (
                        state.position["VOLCANIC_ROCK_VOUCHER_9750"]
                        if "VOLCANIC_ROCK_VOUCHER_9750" in state.position
                        else 0
                    )
                    result["VOLCANIC_ROCK_VOUCHER_9750"] = self.voucher_calcs(
                        state.order_depths["VOLCANIC_ROCK_VOUCHER_9750"],
                        state.traderData["v097_residual_history"],
                        volcanic_rock_mid,
                        v097_strike,
                        TTE,
                        "VOLCANIC_ROCK_VOUCHER_9750",
                        v097_position,
                        volc_rv_pos_lim,
                        v097_entry_threshold,
                        v097_exit_threshold,
                    )

        if "VOLCANIC_ROCK_VOUCHER_10500" in state.order_depths:
            v105_position = (
                state.position["VOLCANIC_ROCK_VOUCHER_10500"]
                if "VOLCANIC_ROCK_VOUCHER_10500" in state.position
                else 0
            )
            result["VOLCANIC_ROCK_VOUCHER_10500"] = self.take_free_voucher(
                "VOLCANIC_ROCK_VOUCHER_10500", v105_position, volc_rv_pos_lim
            )

        if "MAGNIFICENT_MACARONS" in state.order_depths:
            m_position = (
                state.position["MAGNIFICENT_MACARONS"]
                if "MAGNIFICENT_MACARONS" in state.position
                else 0
            )
            result["MAGNIFICENT_MACARONS"] = self.macaron_order(
                state.observations.conversionObservations["MAGNIFICENT_MACARONS"],
            )

        traderData = jsonpickle.encode(state.traderData)
        conversions = 0
        if m_position < 0:
            conversions = min(abs(m_position), 10)
        if m_position > 0:
            conversions = -min(abs(m_position), 10)
        logger.flush(state, result, conversions, state.traderData)
        return result, conversions, traderData
