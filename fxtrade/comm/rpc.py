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


    def _rpc_persisted_report(self, report_date) -> List[List[Any]]:
        
        
        results = self._persistor.retrieve_closed_trade_by_date(report_date)

        return [
            [
                result.instrument,
                result.profit,
                result.balance
            ]
            for result in results
        ]

    def _rpc_persisted_profit(self, report_date) -> List[List[Any]]:
        
        
        results = self._persistor.retrieve_closed_trade_by_date(report_date)

        profits = {}

        for result in results:
            if result.instrument not in profits:
                profits[result.instrument] = 0.
            profits[result.instrument] += result.profit

        return [[k,v] for k,v in profits.items()]


    def _rpc_persisted_decisions(self, report_date) -> List[List[Any]]:
        
        
        results = self._persistor.retrieve_decisions_by_date(report_date)

        decisions = {}

        for result in results:
            if result.instrument not in decisions:
                decisions[result.instrument] = {
                    "TOTAL" : 0,
                    "LONG" : 0.,
                    "SHORT" : 0.,
                    "HOLD" : 0.
                }
                
            decisions[result.instrument][result.position] += 1
            decisions[result.instrument]["TOTAL"] += 1


        decisions_triplets = []

        for instrument in decisions:

            decisions_triplets.append(
                [
                    instrument,
                    round(decisions[instrument]["LONG"] / decisions[instrument]["TOTAL"]*100),
                    round(decisions[instrument]["SHORT"] / decisions[instrument]["TOTAL"]*100),
                    round(decisions[instrument]["HOLD"] / decisions[instrument]["TOTAL"]*100),
                    str(int(decisions[instrument]["TOTAL"]))
                ]
            )

        return decisions_triplets


    def _rpc_closed_activity(self, report_date) -> List[List[Any]]:
        
        datestring = report_date.strftime("%Y-%m-%d")

        _from = datestring+"T00:00:00.000000000Z"
        _to = datestring+"T23:59:59.999999999Z"
        
        result = self._fxtrade.exchange.transactions_list(_from=_from, _to=_to)

        transactions = result["transactions"] if result else []


        closed_trades = []

        for transaction in transactions:
            if ((transaction.get("reason", "") == "MARKET_ORDER_TRADE_CLOSE") or
                ((transaction.get("type", "") == "ORDER_FILL") and
                 (transaction.get("reason", "") == "STOP_LOSS_ORDER")) or 
                ((transaction.get("type", "") == "ORDER_FILL") and
                 (transaction.get("reason", "") == "TAKE_PROFIT_ORDER"))):
                closed_trades.append(
                    [
                        transaction["instrument"],
                        transaction["pl"],
                        transaction["accountBalance"]

                    ]
                )
            if transaction.get("type", "") == "DAILY_FINANCING":
                closed_trades.append(
                    [
                        "DAILY_FEE",
                        transaction["financing"],
                        transaction["accountBalance"]

                    ]
                )

        return closed_trades

    def _rpc_open_trades(self) -> List[List[Any]]:

        open_trades = self._fxtrade.exchange.open_trades()

        open_trades_triplets = []
        for trade in open_trades:

            if trade["state"] == "OPEN":
                open_trades_triplets.append(
                    [
                        trade["instrument"],
                        trade["initialUnits"],
                        trade["unrealizedPL"]
                    ]
                )
        return open_trades_triplets

    def _rpc_closed_profit(self, report_date) -> List[List[Any]]:
        
        
        datestring = report_date.strftime("%Y-%m-%d")

        _from = datestring+"T00:00:00.000000000Z"
        _to = datestring+"T23:59:59.999999999Z"
        
        result = self._fxtrade.exchange.transactions_list(_from=_from, _to=_to)

        transactions = result["transactions"] if result else []

        profits = {}

        for transaction in transactions:
            if ((transaction.get("reason", "") == "MARKET_ORDER_TRADE_CLOSE") or
                ((transaction.get("type", "") == "ORDER_FILL") and
                 (transaction.get("reason", "") == "STOP_LOSS_ORDER")) or 
                ((transaction.get("type", "") == "ORDER_FILL") and
                 (transaction.get("reason", "") == "TAKE_PROFIT_ORDER"))):
                
                if transaction["instrument"] not in profits:
                    profits[transaction["instrument"]] = 0.
                profits[transaction["instrument"]] += float(transaction["pl"])

            if transaction.get("type", "") == "DAILY_FINANCING":
                if "DAILY_FEE" not in profits:
                    profits["DAILY_FEE"] = 0.
                profits["DAILY_FEE"] += float(transaction["financing"])

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