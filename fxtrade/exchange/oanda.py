from fxtrade import constants, OperationalException, DependencyException, TemporaryError
from fxtrade.comm.rpc import RPCMessageType, RPC, RPCException
from fxtrade.data.converter import parse_ticker_dataframe
from tabulate import tabulate
import arrow

import inspect
from random import randint

from datetime import datetime
from math import floor, ceil

from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.forexlabs as forexlabs
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    TakeProfitDetails,
    StopLossDetails,
    StopLossOrderRequest,
    TakeProfitOrderRequest
)
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import logging
from typing import List, Dict, Tuple, Any, Optional
from pandas import DataFrame
import numpy as np


logger = logging.getLogger(__name__)

granularity_dict = {
    5       : "S5", 
    10      : "S10",	
    15      : "S15",	
    30      : "S30",	
    60      : "M1",
    120     : "M2",
    240     : "M4",
    300     : "M5",
    600     : "M10",	
    900     : "M15",	
    1800    : "M30",	
    3600    : "H1",	
    7200    : "H2",	
    10800   : "H3",	
    14400   : "H4",	
    21600   : "H6",
    28800   : "H8",	
    43200   : "H12",	
    86400   : "D",
    604800  : "W",
    2592000 : "M"
}


class Oanda(object):
    """
    This class defines the Oanda API
    """

    _conf: Dict = {}
    _params: Dict = {}

    def __init__(self, config, rpc):
        """
        Initializes this module with the given config,
        it does basic validation whether the specified
        exchange and pairs are valid.
        :return: None
        """
        self._conf.update(config)
        self.account_id, token = \
            self._conf["exchange"]["active_account"], \
            self._conf["exchange"]["token"]
        env = config.get("environment", "practice")
        self._api = API(access_token=token, environment=env)
        self.rpc = rpc

        self._cached_ticker: Dict[str, Any] = {}

        # Holds last candle refreshed time of each pair
        self._pairs_last_refresh_time: Dict[Tuple[str, str], int] = {}

        # Holds all open sell orders for dry_run
        self._dry_run_open_orders: Dict[str, Any] = {}

        if config['dry_run']:
            logger.info('Instance is running with dry_run enabled')

        logger.info('Using Exchange Oanda')

        self.markets = self.account_instruments()
        self.pairs = self.validate_pairs(config['exchange']['pair_whitelist'])
        self.order_book = self.oanda_order_book()


    def validate_pairs(self, pairs: List[str]) -> None:
        """
        Checks if all given pairs are tradable on the current exchange.
        Raises OperationalException if one pair is not available.
        :param pairs: list of pairs
        :return: None
        """

        if not self.markets:
            logger.warning('Unable to validate pairs (assuming they are correct).')
        #     return
        _pairs = []
        for pair in pairs:
            if self.markets and pair not in self.markets:
                raise OperationalException(
                    'Pair {pair} is not available at {self.name}'
                    'Please remove {pair} from your whitelist.')
            else:
                _pairs.append(pair)

        return _pairs

    def send_request(self, request):
        """
        Sends request to oanda API.
        Returns oanda's response if success, 1 if error.
        """
        try:
            rv = self._api.request(request)
            return rv
        except V20Error as err:
            status_code = request.status_code
            print(status_code, err)
            self.rpc.send_msg({
                'type': RPCMessageType.WARNING_NOTIFICATION,
                'status': "Status code: {}\n {}".format(status_code, err)
            })
            return 1


    def account_changes(self, transaction_id):
        params = {
            'sinceTransactionID' : transaction_id
            }
        endpoint = accounts.AccountChanges(self.account_id, params)
        self.send_request(endpoint)
        return endpoint.response
    
    def account_configuration(self, params={}):
        # TODO:think about useful data configs and see 
        # http://developer.oanda.com/rest-live-v20/account-ep/
        endpoint = accounts.AccountConfiguration(self.account_id, params)
        self.send_request(endpoint)
        return endpoint.response
    
    def account_details(self):
        endpoint = accounts.AccountDetails(self.account_id)
        self.send_request(endpoint)
        return endpoint.response
        
    def account_instruments(self, only_currency=True, display=False):

        def leverage(margin):
            return str(int(1./float(margin)))+":1"

        def percentage(margin):
            return str(int(float(margin)*100))+"%"
    

        endpoint = accounts.AccountInstruments(self.account_id)
        self.send_request(endpoint)
        _ins = endpoint.response
        _table = []
        _markets = []
        if only_currency:
            for _in in _ins['instruments']:
                if _in["type"] == "CURRENCY":
                    _markets.append(_in["name"])
                    _table.append([_in["displayName"], _in["name"],percentage(_in["marginRate"]), leverage(_in["marginRate"]),_in["type"]])
                    _table = sorted(_table, key = lambda x: x[1])
        else:
            for _in in _ins['instruments']:
                _table.append([_in["displayName"], _in["name"],percentage(_in["marginRate"]), leverage(_in["marginRate"]),_in["type"]])
                _table = sorted(_table, key = lambda x: x[1])

        if display:
            print(tabulate(_table, headers=['Display name', 'Name', 'Margin rate', 'Leverage', 'type'], tablefmt="pipe"))
        return _markets

    def account_list(self):
        endpoint = accounts.AccountList()
        self.send_request(endpoint)
        return endpoint.response
    
    def account_summary(self):
        endpoint = accounts.AccountSummary(self.account_id)
        self.send_request(endpoint)
        return endpoint.response

    def get_balance(self):
        summary = self.account_summary()
        return summary["account"]["balance"]

    """Instrument info"""
    #see http://developer.oanda.com/rest-live-v20/instrument-ep/
    def get_history(self, instrument: str, granularity:str, count: int = 50, 
        _from=None, _to=None, price="MBA", complete=False
        ):

        #granularity input is given as a number  [seconds] 
        #or as a string following libs.utils.granularity_dict values
        """
        Returns a list of candles, each candle is a dictionary:
            {
                'complete': True, 
                'volume': 100, 
                'time': '2018-10-05T14:56:40.000000000Z',
                'mid': {
                    'o': '1.15258', 
                    'h': '1.15286', 
                    'l': '1.15246', 
                    'c': '1.15286'
                    }
            }
        """

        params = { 
            "granularity":  granularity if type(granularity)==str \
                else granularity_dict[granularity]
            }

        params['count'] = count
        params['price'] = price
        
        if _from and _to:
            params['from'] = _from
            params['to'] = _to
            params.pop("count")

        candles = []
        for req in InstrumentsCandlesFactory(instrument=instrument, params=params):
            single_req = self.send_request(req)
            candles += single_req['candles']
        
        if not complete:
            return candles
        else:
            return candles if candles[-1]["complete"] else candles[:-1]

    def instruments_order_book(self, instrument, params={}):
        endpoint = instruments.InstrumentsOrderBook(instrument, params)
        ob = self.send_request(endpoint)
        return ob["orderBook"]["buckets"]

    def instruments_position_book(self, instrument, params={}):
        endpoint = instruments.InstrumentsPositionBook(instrument, params)
        ob = self.send_request(endpoint)
        return ob["positionBook"]["buckets"]

    def oanda_order_book(self):
        """Synchronize open positions with this object's order_book"""
        order_book_oanda = self.open_positions()

        order_book = {}
        for pair in self.pairs:
            order_book[pair] = {
                'order_type': 0, 
                'tradeID': 0
            }
        for pos in order_book_oanda['positions']:
            try:
                trade_id = pos['long']['tradeIDs'][0]
                order_type = 1
            except KeyError:
                trade_id = pos['short']['tradeIDs'][0]
                order_type = -1
            order_book[pos['instrument']]['tradeID'] = trade_id
            order_book[pos['instrument']]['order_type'] = order_type

        return order_book

    def sync_with_oanda(self):
        self.order_book = self.oanda_order_book()

    """Orders"""

    def open_order(
        self, 
        instrument: str, 
        units: int, 
        stop_loss: float = 0, 
        take_profit: float = 0, 
        comment="", 
        strategy="", 
        other="",
        # price: float,
        ):

        # check if position is already open
        if units < 0:
            if self.order_book[instrument]['order_type'] is (not None and -1):
                self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': 'Short: {} (holding)'.format(instrument)
                })
                return 1
        elif units > 0:
            if self.order_book[instrument]['order_type'] is (not None and 1):
                self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': 'Long: {} (holding)'.format(instrument)
                })
                return 1
        else:
            self.rpc.send_msg({
                'type': RPCMessageType.STATUS_NOTIFICATION,
                'status': 'Nothing: {} | 0 units specified'.format(instrument)
                })
            return 1

        client_extensions = {
            "comment": comment,
            "tag": strategy,
            "other" : other
            }

        order_params = {
            "tradeClientExtensions" : client_extensions
        }

        sign = float(np.sign(units))
        rounding = 4 if instrument!="USD_JPY" else 2

        # if stop_loss:
        #     stop_loss_pips = round(price*(1-stop_loss*sign), rounding)
        #     order_params["stopLossOnFill"] = StopLossDetails(price=stop_loss).data
        # if take_profit:
        #     take_profit_pips = round(price*(1+take_profit*sign), rounding)
        #     order_params["takeProfitOnFill"] = TakeProfitDetails(price=take_profit).data
        

        mkt_order = MarketOrderRequest(
                    instrument=instrument,
                    units=units,
                    **order_params
                    )

            
        endpoint = orders.OrderCreate(self.account_id, mkt_order.data)
        request_data = self.send_request(endpoint)
        tradeID = request_data['orderCreateTransaction']['id']
        
        price = float(request_data['orderFillTransaction']['price'])
        tradeID = request_data['lastTransactionID']
        instrument = request_data['orderCreateTransaction']['instrument']

        rounding = 4 if instrument!="USD_JPY" else 2
        # define stop loss
        if stop_loss:
            stop_loss_pips = round(price*(1-stop_loss*sign), rounding)
            stop_loss_order = StopLossOrderRequest(
                tradeID=tradeID, 
                price=stop_loss_pips
                )
            stop_loss_endpoint = orders.OrderCreate(self.account_id, stop_loss_order.data)
            request_data = self.send_request(stop_loss_endpoint)
            tradeID = request_data['orderCreateTransaction']['tradeID']
        # define take profit
        if take_profit:
            take_profit_pips = round(price*(1+take_profit*sign), rounding)
            take_profit_order = TakeProfitOrderRequest(
                tradeID=tradeID, 
                price=take_profit_pips
                )
            take_profit_endpoint = orders.OrderCreate(self.account_id, take_profit_order.data)
            request_data = self.send_request(take_profit_endpoint)
            tradeID = request_data['orderCreateTransaction']['tradeID']
        
        # check if request was fulfilled and save its ID
        if request_data is not 1:
            self.order_book[instrument]['tradeID'] = tradeID
            self.order_book[instrument]['order_type'] = -1 if units < 0 else 1
            if units > 0:
                self.rpc.send_msg({
                'type': RPCMessageType.BUY_NOTIFICATION,
                'units' : units,
                'price' : price,
                'pair' : instrument,
                'take_profit' : take_profit_pips,
                'stop_loss' : stop_loss_pips
                })
            else:
                self.rpc.send_msg({
                'type': RPCMessageType.SELL_NOTIFICATION,
                'units' : units,
                'price' : price,
                'pair' : instrument,
                'take_profit' : take_profit_pips,
                'stop_loss' : stop_loss_pips
                })

            return 0
        else:
            return 1

    def list_orders(self):
        endpoint = orders.OrderList(self.account_id)
        self.send_request(endpoint)
        return endpoint.response


    def cancel_order(self, order_id):
        endpoint = orders.OrderCancel(self.account_id, order_id)
        self.send_request(endpoint)
        return endpoint.response

    """Positions"""

    def open_positions(self):
        endpoint = positions.OpenPositions(self.account_id)
        op = self.send_request(endpoint)
        return op

    def list_positions(self):                
        endpoint = positions.PositionList(self.account_id)
        op = self.send_request(endpoint)
        return op["positions"]
        
    def close_position(self, instrument, data={"longUnits": "ALL"}):
        endpoint = positions.PositionClose(self.account_id, instrument, data)
        self.send_request(endpoint)
        return endpoint.response

    def position_details(self, instrument):
        endpoint = positions.PositionDetails(self.account_id, instrument)
        self.send_request(endpoint)
        return endpoint.response


    """Prices"""

    #TODO not useful for the purpose of the bot
    """
    def pricing_stream(self, instruments):
        if type(instruments) == "list":
            _ins=","
            _ins = _ins.join(instruments)

            params = {
                "instruments" : _ins
            }
        else:
            params = {
                "instruments" : instruments
            }   
    """

    def pricing_info(self, instruments):
        if type(instruments) == "list":
            _ins=","
            _ins = _ins.join(instruments)

            params = {
                "instruments" : _ins
            }
        else:
            params = {
                "instruments" : instruments
            }  

        endpoint = pricing.PricingInfo(self.account_id, params)
        p = self.send_request(endpoint)
        return p["prices"]



    """Trades"""
    def open_trades(self):
        pass
    
    def trade_crcdo(self, trade_id, take_profit, stop_loss):
        pass

    def trade_update_extentions(self, trade_id, comment="", strategy="", other=""):

        clientExtensions = {
            "comment": comment,
            "tag": strategy,
            "other" : other
            }

    def close_order(self, instrument):

        # check if position exist
        if not self.order_book[instrument]['order_type']:
            print('Position {} does not exist'.format(instrument))
            return 1

        # create and send a request
        r = trades.TradeClose(accountID=self.account_id, tradeID=self.order_book[instrument]['tradeID'])
        request_data = self.send_request(r)

        # check if request was fulfilled and clear it
        if request_data is not 1:
            instrument = request_data['orderCreateTransaction']['instrument']
            self.order_book[instrument]['order_type'] = None
            self.order_book[instrument]['tradeID'] = None
            print('Closed: {}'.format(instrument))
            return 0
        else:
            return 1

    def trade_details(self, trade_id):
        pass

    def trades_list(self, params={}):
        pass