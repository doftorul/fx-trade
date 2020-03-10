"""
This module contains class to define a RPC communications
"""
import logging
from abc import abstractmethod
from datetime import timedelta, datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, Any, List, Optional

import arrow
import sqlalchemy as sql
from numpy import mean, nan_to_num, NAN
from pandas import DataFrame

from fxtrade import TemporaryError, DependencyException
from fxtrade.misc import shorten_date
#  from fxtrade.persistence import Trade
from fxtrade.state import State
from fxtrade.strategy.interface import SellType

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


    def _rpc_report(
            self, timescale: int,
            stake_currency: str, fiat_display_currency: str) -> List[List[Any]]:
        today = datetime.utcnow().date()

        return [
            [
                key,
                '{value:.8f} {symbol}'.format(
                    value=float(value['amount']),
                    symbol=stake_currency
                ),
                '{value:.3f} {symbol}'.format(
                    value=self._fiat_converter.convert_amount(
                        value['amount'],
                        stake_currency,
                        fiat_display_currency
                    ) if self._fiat_converter else 0,
                    symbol=fiat_display_currency
                ),
                '{value} trade{s}'.format(
                    value=value['trades'],
                    s='' if value['trades'] < 2 else 's'
                ),
            ]
            for key, value in profit_days.items()
        ]


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
        res = {'method': self._fxtrade.pairlists.name,
               'length': len(self._fxtrade.pairlists.whitelist),
               'whitelist': self._fxtrade.active_pair_whitelist
               }
        return res
