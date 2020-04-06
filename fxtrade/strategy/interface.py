"""
Strategy interface
This module defines the interface to apply for strategies
"""
import importlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple
import warnings
import time

from pandas import DataFrame

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """
    Enum to distinguish between buy and sell signals
    """
    BUY = "buy"
    SELL = "sell"


def retrieve_strategy(name_strategy):
    _cls = getattr(
        importlib.import_module("fxtrade.strategy"), 
        name_strategy
        )
    return _cls

class Instrument:
    def __init__(self, name, time):
        self.name = name
        self.time = time
        self.units = 0

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
            price=price,
            complete=True
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
            # print(f"current time {current_time}")
            if current_time != instrument.time:
                break
            time.sleep(self.idle_time)

        # .action method can use collect() or extract_prices to extract other relevant time series
        order_signal = self.action(candles)
        # order_signal should be 1 [buy] , -1 [sell], 0 [hold]

        return current_time, order_signal

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
