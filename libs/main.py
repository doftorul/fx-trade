# -*- coding: utf-8 -*-
import time
import argparse
from datetime import datetime
import calendar
import logging
import autoinit
import sys

import slacker

from libs.broker import Broker
from libs.robot import Bot
from libs.strategy import *
from libs.portfolio import Portfolio

#transaction cost minimisation: commissions (fees brokerage, exchange and taxes), slippage, spread ask-bid

logging.basicConfig(
    filename="./agent.log",
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',
)

logger = logging.getLogger(__name__)

def process():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='simplebot')
    #parser.add_argument('--longMA', default=20, type=int,
    #                    help='period of the long movingaverage')
    #parser.add_argument('--shortMA', default=10, type=int,
    #                    help='period of the short movingaverage')
    parser.add_argument('--stopLoss', default=0.5, type=float,
                        help='stop loss value as a percentage of entryvalue')
    parser.add_argument('--takeProfit', default=0.5, type=float,
                        help='take profit value as a percentage of entryvalue')
    #parser.add_argument('--instrument', type=str, help='instrument')
    #parser.add_argument('--granularity', choices=granularities, required=True)
    #parser.add_argument('--units', type=int, required=True)

    args = parser.parse_args()


    api = Broker()
    
    #this prints available instruments
    api.account_instruments()

    ## this saves a file plot  
    # candles = api.instruments_candles(instrument="EUR_USD", granularity="M5", count=250)
    # plot_candlestick(candles)

    #granularities are decided by strategy 
    # strategy = DeepSense()
    # strategy = QNetwork()
    
    #TODO units need to be decided by the portfolio management class 
    # as well as instrument

    portfolio = Portfolio(api)    
    
    specs = portfolio.predict()


    for spec in specs:

        strategy = McGinleyDynamic(api, spec.instrument) 
        bot = Bot(api, strategy, spec)
        bot.trade()
    


if __name__ == "__main__":
    process()