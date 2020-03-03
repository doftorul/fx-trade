import random
import numpy as np
from abc import ABC, abstractmethod
import time

"""
there are a number of strategy types to inform the design of your algorithmic trading robot. 
These include strategies that take advantage of the following (or any combination thereof):

Macroeconomic news (e.g. non-farm payroll or interest rate changes)
Fundamental analysis (e.g. using revenue data or earnings release notes)
Statistical analysis (e.g. correlation or co-integration)
Technical analysis (e.g. moving averages)
The market microstructure (e.g. arbitrage or trade infrastructure)


Read more: Coding Your Own Algo Trading Robot
Investopedia https://www.investopedia.com/articles/active-trading/081315/how-code-your-own-algo-trading-robot.asp#ixzz5YcW8c3zA 
Follow us: Investopedia on Facebook
"""

class Strategy(ABC):
    def __init__(self, api, instrument, **kwargs):
        self.api = api
        self.instrument = instrument.name
        self.granularity = kwargs.get("granularity", 60)
        self.count = kwargs.get("count", 50)
        self.idle_time = min(kwargs.get("idle_time", 5), self.granularity/10) #assert idle time for polling the server is one order of magnitude lower than granularity

    
    def collect(self, granularity=None, count=None, price="MBA"):
        if not granularity: granularity = self.granularity
        if not count: count = self.count

        return self.api.get_history(
            self.instrument, 
            granularity, 
            count, 
            price=price
            )

    @abstractmethod
    def action(self, candles):
        pass
        

    # wait for new candles, then perform action
    def idle(self, instrument):
        # candles = self.retrieve_data(300, 48)
        while True:
            candles = self.collect()
            current_time = candles[-1]['time']
            print(f"current time {current_time}")
            if current_time != instrument.time:
                break
            time.sleep(self.idle_time)

        # .action method can use collect() or extract_prices to extract other relevant time series
        order_signal, actual_price = self.action(candles)
        # order_signal should be 1 [buy] , -1 [sell], 0 [hold]

        return current_time, order_signal, actual_price

    def extract_prices(self, candles, price_type='mid'):
        # price_type = price_type.capitalize()

        opens = []
        closes = []
        highs = []
        lows = []
        timestamps = []
        volumes = []

        for candle in candles:
            opens.append(float(candle[price_type]['o']))
            closes.append(float(candle[price_type]['c']))
            highs.append(float(candle[price_type]['h']))
            lows.append(float(candle[price_type]['l']))
            timestamps.append(candle['time'][:19]) #'2018-10-05T14:56:40'
            volumes.append(int(candle['volume']))

        return opens, closes, highs, lows, volumes

class Random(Strategy):
    def __init__(self, *args, **kwargs):
        super(Random, self).__init__(**kwargs)

    def action(self, candles):
        dice = random.random()

        if dice < 1./3:
            return 'buy'
        elif dice < 2./3:
            return 'sell'
        else:
            return 'hold'

class MACD(Strategy):
    def __init__(self, *args, **kwargs):
        super(MACD, self).__init__(*args, **kwargs)
        self.fast_ma_period = kwargs.get('fast_ma_period', 9)
        self.slow_ma_period = kwargs.get('slow_ma_period', 21)

    def construct_ma(self, candles, price_type = 'ask'):
        prices = np.array(self.extract_prices(candles, price_type=price_type)[1], dtype=float)

        length = len(candles)

        slow_ma = np.array([None]*length, dtype=float)
        fast_ma = np.array([None]*length, dtype=float)

        ret = np.cumsum(prices, dtype=float)

        n = self.fast_ma_period
        fast_ma[n:] = ret[n:] - ret[:-n]
        fast_ma /= n
        n = self.slow_ma_period
        slow_ma[n:] = ret[n:] - ret[:-n]
        slow_ma /= n

        return fast_ma, slow_ma

    def action(self, candles):
        fma_ask, sma_ask = self.construct_ma(candles, price_type='ask')
        fma_bid, sma_bid = self.construct_ma(candles, price_type='bid')

        if fma_ask[-1] > sma_ask[-1]:
            return 'buy'
        elif fma_bid[-1] < sma_bid[-1]:
            return 'sell'
        else:
            return 'hold'

class McGinleyDynamic(Strategy):
    def __init__(self, *args, **kwargs):
        super(McGinleyDynamic, self).__init__(*args, **kwargs)

    def construct_mgd(self, candles, price_type = 'bid', N=10):
        closes = np.array(self.extract_prices(candles, price_type=price_type)[1], dtype=float)
        mgd = np.array([0]*len(closes), dtype=float)
        mgd[:] = closes[:]

        for i in range(1, len(mgd)):
            mgd[i] = mgd[i - 1] + (closes[i] - mgd[i - 1])/(N*(closes[i]/mgd[i - 1])**4)

        return mgd

    def action(self, candles):
        mgd_bid = self.construct_mgd(candles, price_type='bid', N=10), self.construct_mgd(candles, price_type='ask', N=10)
        mgd_ask = self.construct_mgd(candles, price_type='ask', N=10), self.construct_mgd(candles, price_type='ask', N=10)

        d_mgd_bid = 3*mgd_bid[-1] - 4*mgd_bid[-2] + mgd_bid[-3]
        d_mgd_ask = 3*mgd_ask[-1] - 4*mgd_ask[-2] + mgd_ask[-3]

        if d_mgd_ask > 0:
            return 'buy'
        elif d_mgd_bid < 0:
            return 'sell'
        else:
            return 'hold'

class ParabolicSAR(Strategy):
    def __init__(self, *args, **kwargs):
        super(ParabolicSAR, self).__init__(*args, **kwargs)

        self.start = kwargs.get('start', 0.02)
        self.increment = kwargs.get('increment', 0.02)
        self.max_increment = kwargs.get('max_increment', 0.2)

    def construct_psar(self, candles, price_type = 'ask'):
        price_type = price_type.title()

        opens, closes, highs, lows = self.extract_prices(candles, price_type)

        length = len(candles)
        af = self.start
        inc = self.increment
        uptrend = True
        psar = closes[:]
        psar_uptrend = [None]*length
        psar_downtrend = [None]*length

        high_point = highs[0]
        low_point = lows[0]

        for i in range(2, length):
            if uptrend:
                psar[i] = psar[i - 1] + af*(high_point - psar[i - 1])
            else:
                psar[i] = psar[i - 1] + af*(low_point - psar[i - 1])

            reverse = False

            if uptrend:
                if lows[i] < psar[i]:
                    uptrend = False
                    reverse = True
                    psar[i] = high_point
                    low_point = lows[i]
                    af = self.start

            else:
                if highs[i] > psar[i]:
                    uptrend = True
                    reverse = True
                    psar[i] = low_point
                    high_point = highs[i]
                    af = self.start

            if not reverse:
                if uptrend:
                    if highs[i] > high_point:
                        high_point = highs[i]
                        af = min(af + inc, self.max_increment)

                    if lows[i - 1] < psar[i]:
                        psar[i] = lows[i - 1]

                    if lows[i - 2] < psar[i]:
                        psar[i] = lows[i - 2]

                else:
                    if lows[i] < low_point:
                        low_point = lows[i]
                        af = min(af + inc, self.max_increment)
                    if highs[i - 1] > psar[i]:
                        psar[i] = highs[i - 1]
                    if highs[i - 2] > psar[i]:
                        psar[i] = highs[i - 2]

            if uptrend:
                psar_uptrend[i] = psar[i]
            else:
                psar_downtrend[i] = psar[i]

        return psar_uptrend, psar_downtrend

    def action(self, candles):

        psar_uptrend, psar_downtrend = self.construct_psar(candles, 'ask')

        psar_uptrend = psar_uptrend[-3:]
        psar_downtrend = psar_downtrend[-3:]

        if not None in psar_uptrend:
            return 'buy'
        elif not None in psar_downtrend:
            return 'sell'

        return 'hold'

class DeepSense(Strategy):
    "DQN approach with DeepSense approximating Q function"
    def __init__(self, *args, **kwargs):
        super(DeepSense, self).__init__(*args, **kwargs)
        pass

    def action(self, candles):
        pass

class A2C(Strategy):
    "Asynchronous Actor-Critic approach"
    def __init__(self, *args, **kwargs):
        super(A2C, self).__init__(*args, **kwargs)
        pass

    def action(self, candles):
        pass

class RRL(Strategy):
    "Recurrent Neural Network basic approach"
    def __init__(self, *args, **kwargs):
        super(RRL, self).__init__(*args, **kwargs)
        pass

    def action(self, candles):
        pass