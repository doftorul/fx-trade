import random
import numpy as np
from abc import ABC, abstractmethod
import time
from fxtrade.strategy.interface import Strategy
import pandas as pd
from pyti.exponential_moving_average import exponential_moving_average as ema

"""
there are a number of strategy types to inform the design of your algorithmic trading robot. 
These include strategies that take advantage of the following (or any combination thereof):

Macroeconomic news (e.g. non-farm payroll or interest rate changes)
Fundamental analysis (e.g. using revenue data or earnings release notes)
Statistical analysis (e.g. correlation or co-integration)
Technical analysis (e.g. moving averages)
The market microstructure (e.g. arbitrage or trade infrastructure)


Investopedia https://www.investopedia.com/articles/active-trading/081315/how-code-your-own-algo-trading-robot.asp#ixzz5YcW8c3zA 
"""

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
        self.fast_ma_period = kwargs.get('fast_ma_period', 12)
        self.slow_ma_period = kwargs.get('slow_ma_period', 26)

    def construct_ma(self, candles, price_type = 'ask'):
        o, c, h, l, v  = self.extract_prices(candles, price_type=price_type)

        slow_ma = ema(c, self.slow_ma_period)
        fast_ma = ema(c, self.fast_ma_period)

        macd = slow_ma - fast_ma
        macd_ewm = np.array(pd.DataFrame(macd).ewm(span=9).mean().squeeze()) 

        return macd, macd_ewm

    def action(self, candles):
        macd, macd_ewm = self.construct_ma(candles, price_type='mid')

        if macd_ewm[-1] > macd[-1]:
            return 'buy'
        elif macd_ewm[-1] < macd[-1]:
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