"""
This module contains class to define a RPC communications
"""
import logging
from abc import abstractmethod
from datetime import timedelta, datetime, date
from enum import Enum
from typing import Dict, Any, List, Optional

import numpy as np
from pandas import DataFrame

#  from fxtrade.persistence import Trade
from fxtrade.state import State
from fxtrade.persistence import Persistor


logger = logging.getLogger(__name__)


class RPCMessageType(Enum):
    STATUS_NOTIFICATION = 'status'
    WARNING_NOTIFICATION = 'warning'
    CUSTOM_NOTIFICATION = 'custom'
    BUY_NOTIFICATION = 'buy'
    SELL_NOTIFICATION = 'sell'
    IDLE_NOTIFICATION = 'idle'
    HOLD_NOTIFICATION = 'hold'

    def __repr__(self):
        return self.value


class RPCException(Exception):
    """
    Should be raised with a rpc-formatted message in an _rpc_* method
    if the required state is wrong, i.e.:

    raise RPCException('*Status:* `no active trade`')
    """
    def __init__(self, message: str) -> None:
        super().__init__(self)
        self.message = message

    def __str__(self):
        return self.message


class RPC(object):
    """
    RPC class can be used to have extra feature, like bot data, and access to DB data
    """
    # Bind _fiat_converter if needed in each RPC handler
    _fiat_converter = None

    def __init__(self, fxtrade) -> None:
        """
        Initializes all enabled rpc modules
        :param fxtrade: Instance of a fxtrade bot
        :return: None
        """
        self._fxtrade = fxtrade
        self._persistor = Persistor()

    @property
    def name(self) -> str:
        """ Returns the lowercase name of the implementation """
        return self.__class__.__name__.lower()

    @abstractmethod
    def cleanup(self) -> None:
        """ Cleanup pending module resources """

    @abstractmethod
    def send_msg(self, msg: Dict[str, str]) -> None:
        """ Sends a message to all registered rpc modules """


    def _rpc_report(self, report_date) -> List[List[Any]]:
        
        
        results = self._persistor.retrieve_closed_trade_by_date(report_date)

        return [
            [
                result.instrument,
                result.profit,
                result.balance
            ]
            for result in results
        ]

    def _rpc_profit(self, report_date) -> List[List[Any]]:
        
        
        results = self._persistor.retrieve_closed_trade_by_date(report_date)

        profits = {}

        for result in results:
            if result.instrument not in profits:
                profits[result.instrument] = 0.
            profits[result.instrument] += result.profit

        return [[k,v] for k,v in profits.items()]


    def _rpc_start(self) -> Dict[str, str]:
        """ Handler for start """
        if self._fxtrade.state == State.RUNNING:
            return {'status': 'already running'}

        self._fxtrade.state = State.RUNNING
        return {'status': 'starting trader ...'}

    def _rpc_stop(self) -> Dict[str, str]:
        """ Handler for stop """
        if self._fxtrade.state == State.RUNNING:
            self._fxtrade.state = State.STOPPED
            return {'status': 'stopping trader ...'}

        return {'status': 'already stopped'}

    def _rpc_close_all(self) -> Dict[str, str]:
        """ Handler for closing all trades """
        self._fxtrade.close_all_orders()
        return {'status' : "Closed all orders."}

    def _rpc_reload_conf(self) -> Dict[str, str]:
        """ Handler for reload_conf. """
        self._fxtrade.state = State.RELOAD_CONF
        return {'status': 'reloading config ...'}


    def _rpc_whitelist(self) -> Dict:
        """ Returns the currently active whitelist"""
        return [
            [p.name, p.units] for p in self._fxtrade.pairlists
        ]