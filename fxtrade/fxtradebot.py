"""
Freqtrade is the main module of this bot. It contains the class Freqtrade()
"""

import copy
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import itertools
import arrow
from requests.exceptions import RequestException

from fxtrade import (DependencyException, OperationalException,
                       TemporaryError, __version__, constants, persistence)
from fxtrade.data.converter import order_book_to_dataframe
from fxtrade.comm.rpc_manager import RPCManager
from fxtrade.comm.rpc import RPCMessageType
from fxtrade.state import State
from fxtrade.strategy.interface import retrieve_strategy, Instrument
from fxtrade.wallets import Portfolio
from fxtrade.exchange.oanda import Oanda
from fxtrade.persistence import Persistor
from multiprocessing import Process, Pool
from joblib import Parallel, delayed

# from pathos.multiprocessing import ProcessingPool as Pool# from libs.factory import DataFactory

logger = logging.getLogger(__name__)

signal2index = {
    'buy' : 1,
    'sell' : -1, 
    'hold' : 0
}

def unwrap_self(*arg, **kwarg):
    return FXTradeBot._process(*arg, **kwarg)

class FXTradeBot(object):
    """
    Freqtrade is the main class of the bot.
    This is from here the bot start its logic.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Init all variables and objects the bot needs to work
        :param config: configuration dict, you can use Configuration.get_config()
        to get the config dict.
        """

        logger.info(
            'Starting fxtrade %s',
            __version__,
        )

        # Init bot states
        self.state = State.STOPPED

        self.persistor = Persistor()

        # Init objects
        self.config = config
        #this should be a class that is initialised
        self.strategy = retrieve_strategy(self.config["strategy"]["name"])
        self.strategy_params = self.config["strategy"]["params"]

        self.stop_loss = self.config["edge"]["stop_loss"]
        self.take_profit = self.config["edge"]["take_profit"]

        self.rpc: RPCManager = RPCManager(self)

        self.exchange = Oanda(self.config, self.rpc)

        # Attach Dataprovider to Strategy baseclass
        # IStrategy.dp = self.dataprovider
        # Attach Wallets to Strategy baseclass
        # IStrategy.wallets = self.wallets

        self.pairlists = [Instrument(pair, "") for pair in self.config.get('exchange').get('pair_whitelist', [])]

        
        self.strategies = [
            self.strategy(
                api=self.exchange, 
                instrument=pair, 
                kwargs=self.strategy_params) for pair in self.pairlists
            ]
        
        # Initializing Edge only if enabled
        # TODO: EDGE COMBINED WITH PORTFOLIO MANAGEMENT STRATEGIES FOR FOREX
        self.portfolio = Portfolio(
            api=self.exchange, 
            pairlists=self.pairlists, 
            stake=self.config["stake_amount"]
            )

        # Set initial application state
        initial_state = self.config.get('initial_state')

        if initial_state:
            self.state = State[initial_state.upper()]
        else:
            self.state = State.STOPPED


    def cleanup(self) -> None:
        """
        Cleanup pending resources on an already stopped bot
        :return: None
        """
        logger.info('Cleaning up modules ...')
        self.rpc.cleanup()

    def worker(self, old_state: State = None, idle: int = constants.PROCESS_THROTTLE_SECS) -> State:
        """
        Trading routine that must be run at each loop
        :param old_state: the previous service state from the previous call
        :return: current service state
        """
        # Log state transition
        state = self.state
        if state != old_state:
            self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': f'{state.name.lower()}'
            })
            logger.info('Changing state to: %s', state.name)
            if state == State.RUNNING:
                self.rpc.startup_messages(self.config, self.pairlists)

        if state == State.STOPPED:
            time.sleep(1)
        elif state == State.RUNNING:

            funds = self.portfolio.update()

            self.rpc.send_msg({
                'type': RPCMessageType.IDLE_NOTIFICATION,
                'status': "Fetching candles | Gran. {} s".format(self.strategy_params["granularity"])
            })

            self_list = [self]*len(self.pairlists)
            results = Parallel(n_jobs= -1, backend="threading")(delayed(unwrap_self)(_self,p,s,f) for _self,p,s,f in zip(self_list, self.pairlists, self.strategies, funds))

            self.pairlists = [r[0] for r in results]
            closed_order_details = [r[1] for r in results]
            open_order_details = list(itertools.chain.from_iterable([r[2] for r in results]))
            decisions = [r[3] for r in results]

            closed_order_details = [c for c in closed_order_details if c]
            open_order_details = [o for o in open_order_details if o]

            # closed_order_details = sorted(closed_order_details, key=lambda k: datetime.strptime(k['time'],'%Y-%m-%dT%H:%M:%S'))
            # open_order_details = sorted(open_order_details, key=lambda k: datetime.strptime(k['time'], '%Y-%m-%dT%H:%M:%S'))
            closed_order_details = sorted(closed_order_details, key=lambda k: k['time'])
            open_order_details = sorted(open_order_details, key=lambda k: k['time'])
            decisions = sorted(decisions, key=lambda k: k['time'])


            self.persistor.store_opened(open_order_details)
            self.persistor.store_closed(closed_order_details)
            self.persistor.store_decisions(decisions)
            # updated_pairlist = []
            # for p, s, f in zip(self.pairlists, self.strategies, funds):
            #     updated_pair = self._process(p,s,f)
            #     updated_pairlist.append(updated_pair)
            #self.pairlists = updated_pairlist

        return state

    

    def _process(self, instrument, strategy, to_commit):
        """
        This is a trade iteration. It checks for new candles every 5 seconds, then performs
        an action. 
        """

        closed_order_details = {}
        open_order_details = []

        if instrument.units:
            to_commit = instrument.units

        try:
            current_time, order_signal = strategy.idle(instrument) #check if the time is updated (this includes a while loop, so remove the outer loop)
            instrument.time = current_time #updated every granularity occurs

            order_signal = signal2index[order_signal]

            order_signal_id = [2,0,1][order_signal] #1, -1, 0
            decision = {
                "position" : ['LONG', 'SHORT', 'HOLD'][order_signal_id],
                "time" : datetime.utcnow(),
                "instrument" : instrument.name  
            } 



            self.exchange.sync_with_oanda()
            current_position = self.exchange.order_book[instrument.name]['order_type']
            if current_position != order_signal:
                if current_position:
                    closed_order_details = self.exchange.close_order(instrument.name) #close an order if different order signal
                
                open_order_details = self.exchange.open_order(
                    instrument=instrument.name, 
                    units=order_signal*to_commit,
                    #price=actual_price, 
                    stop_loss=self.stop_loss, 
                    take_profit=self.take_profit
                    )
            else:
                message = '{} ({})'.format(instrument.name, ['Long', 'Short', 'Not traded'][order_signal_id])
                self.rpc.send_msg({
                'type': RPCMessageType.HOLD_NOTIFICATION,
                'status': message
            })
        except Exception as error:
            traceback.print_exc()
            logger.warning(f"Error: {error}, retrying in {constants.RETRY_TIMEOUT} seconds...")
            
            # tb = traceback.format_exc()
            # hint = 'Issue `/start` if you think it is safe to restart.'
            self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': f'OperationalException:\n```\n{error}```'
            })
            #time.sleep(constants.RETRY_TIMEOUT)

        self.exchange.sync_with_oanda()
        if not self.exchange.order_book[instrument.name]['order_type']:
            instrument.units = 0
        else:
            instrument.units = to_commit

        return instrument, closed_order_details, open_order_details, decision


    def close_all_orders(self):
        self.exchange.sync_with_oanda()
        closed_order_details = []

        for instrument in self.exchange.order_book:
            if self.exchange.order_book[instrument]['order_type']:
                closed_order_details.append(self.exchange.close_order(instrument))

        closed_order_details = sorted(closed_order_details, key=lambda k: k['time'])
        self.persistor.store_closed(closed_order_details)
        # self.rpc.send_msg({
        #     'type' : RPCMessageType.CUSTOM_NOTIFICATION,
        #     'status' : "*Closing* all orders..."
        # })

    # def something(self):

    #     self.rpc.send_msg({
    #         'type': RPCMessageType.BUY_NOTIFICATION,
    #         'exchange': self.exchange.name.capitalize(),
    #         'pair': pair_s,
    #         'market_url': pair_url,
    #         'limit': buy_limit_filled_price,
    #         'stake_amount': stake_amount,
    #         'stake_currency': stake_currency,
    #         'fiat_currency': fiat_currency
    #     })

    
    #     self.rpc.send_msg({
    #             'type': RPCMessageType.STATUS_NOTIFICATION,
    #             'status': f'Unfilled sell order for {trade.pair} cancelled {reason}'
    #         })

    #     return False