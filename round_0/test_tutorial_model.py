import unittest
from datamodel import OrderDepth, Order, TradingState
from tutorial_model import Trader
import random
import math

class TestTrader(unittest.TestCase):

    def setUp(self):
        self.trader = Trader()

    def generate_random_order_depth(self, fair_value, volatility, num_orders=10):
        order_depth = OrderDepth()
        for _ in range(num_orders):
            price = int(random.gauss(fair_value, volatility))
            volume = random.randint(1, 10)
            if price < fair_value:
                order_depth.sell_orders[price] = volume
            else:
                order_depth.buy_orders[price] = volume

        return order_depth

    def test_rainforest_resin_orders(self):
        fair_value = 10000
        volatility = 50
        position = 0
        position_limit = 40
        width = 2

        order_depth = self.generate_random_order_depth(fair_value, volatility)
        orders = self.trader.rainforest_resin_orders(order_depth, fair_value, width, position, position_limit)
        
        for order in orders:
            if order.quantity > 0:
                self.assertLess(order.price, fair_value)
            else:
                self.assertGreater(order.price, fair_value)

    def test_kelp_orders(self):
        fair_value = 10000
        volatility = 50
        position = 0
        position_limit = 49
        timespan = 50
        width = 4
        kelp_take_width = 1.35

        order_depth = self.generate_random_order_depth(fair_value, volatility)
        orders = self.trader.kelp_orders(order_depth, timespan, width, kelp_take_width, position, position_limit)
        
        for order in orders:
            if order.quantity > 0:
                self.assertLess(order.price, fair_value)
            else:
                self.assertGreater(order.price, fair_value)

    def test_run(self):
        fair_value = 10000
        volatility = 50
        timestamp = 0
        traderData = ""
        listings = {}
        own_trades = {}
        market_trades = {}
        position = {"RAINFOREST_RESIN": 0, "KELP": 0}
        observations = None

        order_depths = {
            "RAINFOREST_RESIN": self.generate_random_order_depth(fair_value, volatility),
            "KELP": self.generate_random_order_depth(fair_value, volatility)
        }

        state = TradingState(traderData, timestamp, listings, order_depths, own_trades, market_trades, position, observations)
        result, conversions, traderData = self.trader.run(state)

        self.assertIn("RAINFOREST_RESIN", result)
        self.assertIn("KELP", result)
        self.assertEqual(conversions, 1)
        self.assertIsInstance(traderData, str)
        
    def test_multiple_trading_states(self):
        fair_value = 10000
        kelp_fair_value = 100
        volatility = 50
        traderData = ""
        listings = {}
        own_trades = {}
        market_trades = {}
        position = {"RAINFOREST_RESIN": 0, "KELP": 0}
        observations = None

        for _ in range(1000):
            timestamp = random.randint(0, 100000)
            # geometric random walk model for fair value
            kelp_fair_value = math.exp(math.log(kelp_fair_value) + random.gauss(0, 0.1))
            
            order_depths = {
                "RAINFOREST_RESIN": self.generate_random_order_depth(fair_value, volatility),
                "KELP": self.generate_random_order_depth(kelp_fair_value, volatility)
            }

            state = TradingState(traderData, timestamp, listings, order_depths, own_trades, market_trades, position, observations)
            result, conversions, traderData = self.trader.run(state)

        self.assertIn("RAINFOREST_RESIN", result)
        self.assertIn("KELP", result)
        self.assertIsInstance(traderData, str)

if __name__ == '__main__':
    unittest.main()