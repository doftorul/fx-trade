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
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import (
    MarketOrderRequest,
    TakeProfitDetails,
    StopLossDetails
)
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import logging
from libs.utils import granularity_dict
from typing import List, Dict, Tuple, Any, Optional
from pandas import DataFrame


from freqtrade import constants, OperationalException, DependencyException, TemporaryError
from freqtrade.data.converter import parse_ticker_dataframe

logger = logging.getLogger(__name__)


class Oanda(object):
    """
    This class defines the Oanda API
    """

    _conf: Dict = {}
    _params: Dict = {}

    def __init__(self, config: dict) -> None:
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

        self._cached_ticker: Dict[str, Any] = {}

        # Holds last candle refreshed time of each pair
        self._pairs_last_refresh_time: Dict[Tuple[str, str], int] = {}

        # Holds all open sell orders for dry_run
        self._dry_run_open_orders: Dict[str, Any] = {}

        if config['dry_run']:
            logger.info('Instance is running with dry_run enabled')

        logger.info('Using Exchange Oanda')

        self.markets = self.account_instruments()
        self.validate_pairs(config['exchange']['pair_whitelist'])


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

        for pair in pairs:
            if self.markets and pair not in self.markets:
                raise OperationalException(
                    'Pair {pair} is not available at {self.name}'
                    'Please remove {pair} from your whitelist.')

    def account_changes(self, transaction_id):
        params = {
            'sinceTransactionID' : transaction_id
            }
        endpoint = accounts.AccountChanges(self.account_id, params)
        self._api.request(endpoint)
        return endpoint.response
    
    def account_configuration(self, params={}):
        # TODO:think about useful data configs and see 
        # http://developer.oanda.com/rest-live-v20/account-ep/
        endpoint = accounts.AccountConfiguration(self.account_id, params)
        self._api.request(endpoint)
        return endpoint.response
    
    def account_details(self):
        endpoint = accounts.AccountDetails(self.account_id)
        self._api.request(endpoint)
        return endpoint.response
        
    def account_instruments(self, only_currency=True, display=False):

        def leverage(margin):
            return str(int(1./float(margin)))+":1"

        def percentage(margin):
            return str(int(float(margin)*100))+"%"
    

        endpoint = accounts.AccountInstruments(self.account_id)
        self._api.request(endpoint)
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
        self._api.request(endpoint)
        return endpoint.response
    
    def account_summary(self):
        endpoint = accounts.AccountSummary(self.account_id)
        self._api.request(endpoint)
        return endpoint.response

    def get_balance(self):
        summary = self.account_summary()
        return summary["account"]["balance"]

    """Instrument info"""
    #see http://developer.oanda.com/rest-live-v20/instrument-ep/
    def get_history(self, instrument: str, granularity:str, count: int = 50, 
        _from=None, _to=None, price="MBA" 
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

        params["count"] = count
        params['price'] = price
        
        if _from and _to:
            params['from'] = _from
            params['to'] = _to
            params.pop("count")

        candles = []
        for req in InstrumentsCandlesFactory(instrument=instrument, params=params):
            single_req = self._api.request(req)
            candles += single_req['candles']
        return candles

    def instruments_order_book(self, instrument, params={}):
        endpoint = instruments.InstrumentsOrderBook(instrument, params)
        ob = self._api.request(endpoint)
        return ob["orderBook"]["buckets"]

    def instruments_position_book(self, instrument, params={}):
        endpoint = instruments.InstrumentsPositionBook(instrument, params)
        ob = self._api.request(endpoint)
        return ob["positionBook"]["buckets"]

    """Orders"""

    def create_order(self, instrument: str, units: int, 
        stop_loss: float, take_profit: float, comment="", strategy="", other=""
        ):

        client_extensions = {
            "comment": comment,
            "tag": strategy,
            "other" : other
            }

        mkt_order = MarketOrderRequest(
                    instrument=instrument,
                    units=units,
                    takeProfitOnFill=TakeProfitDetails(price=take_profit).data,
                    stopLossOnFill=StopLossDetails(price=stop_loss).data,
                    tradeClientExtensions=client_extensions
                    )
            
        endpoint = orders.OrderCreate(self.account_id, mkt_order.data)
        order_created = self._api.request(endpoint)
        #trade_id = order_created["orderFillTransaction"]["tradeOpened"]["tradeID"]
        #batch_id = order_created["orderCreateTransaction"]["batchID"]
        return order_created

    def list_orders(self):
        endpoint = orders.OrderList(self.account_id)
        self._api.request(endpoint)
        return endpoint.response


    def cancel_order(self, order_id):
        endpoint = orders.OrderCancel(self.account_id, order_id)
        self._api.request(endpoint)
        return endpoint.response

    """Positions"""

    def open_positions(self):
        endpoint = positions.OpenPositions(self.account_id)
        op = self._api.request(endpoint)
        return op["positions"]

    def list_positions(self):                
        endpoint = positions.PositionList(self.account_id)
        op = self._api.request(endpoint)
        return op["positions"]
        
    def close_position(self, instrument, data={"longUnits": "ALL"}):
        endpoint = positions.PositionClose(self.account_id, instrument, data)
        self._api.request(endpoint)
        return endpoint.response

    def position_details(self, instrument):
        endpoint = positions.PositionDetails(self.account_id, instrument)
        self._api.request(endpoint)
        return endpoint.response


    """Prices"""

    """
    #TODO not useful for the purpose of the bot
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
        p = self._api.request(endpoint)
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

    def trade_close(self, trade_id, units=None):
        data = {}
        if units:
            data = {
                "units" : units
                }

    def trade_details(self, trade_id):
        pass

    def trades_list(self, params={}):
        pass