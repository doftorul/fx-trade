from libs.config import getconfig
from tabulate import tabulate

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


logging.basicConfig(
    filename="agent.log",
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',
)

#TODO: Adapth this class to Exchange class   
class Oanda(object):
    """Account"""
    def __init__(self):#, instrument, granularity, units, clargs):
        self.conf = getconfig()
        self.accountID, token = self.conf["conf"]["active_account"], self.conf["conf"]["token"]
        self.client = API(access_token=token, environment="practice")


    def account_changes(self, transaction_id):
        params = {
            'sinceTransactionID' : transaction_id
            }
        endpoint = accounts.AccountChanges(self.accountID, params)
        self.client.request(endpoint)
        return endpoint.response
    
    def account_configuration(self, params={}):
        # TODO:think about useful data configs and see 
        # http://developer.oanda.com/rest-live-v20/account-ep/
        endpoint = accounts.AccountConfiguration(self.accountID, params)
        self.client.request(endpoint)
        return endpoint.response
    
    def account_details(self):
        endpoint = accounts.AccountDetails(self.accountID)
        self.client.request(endpoint)
        return endpoint.response
        
    def account_instruments(self, only_currency=True):

        def leverage(margin):
            return str(int(1./float(margin)))+":1"

        def percentage(margin):
            return str(int(float(margin)*100))+"%"
    
        endpoint = accounts.AccountInstruments(self.accountID)
        self.client.request(endpoint)
        _ins = endpoint.response
        _table = []
        if only_currency:
            for _in in _ins['instruments']:
                if _in["type"] == "CURRENCY":
                    _table.append([_in["displayName"], _in["name"],percentage(_in["marginRate"]), leverage(_in["marginRate"]),_in["type"]])
                    _table = sorted(_table, key = lambda x: x[1])
        else:
            for _in in _ins['instruments']:
                _table.append([_in["displayName"], _in["name"],percentage(_in["marginRate"]), leverage(_in["marginRate"]),_in["type"]])
                _table = sorted(_table, key = lambda x: x[1])

        print(tabulate(_table, headers=['Display name', 'Name', 'Margin rate', 'Leverage', 'type'], tablefmt="pipe"))


    def account_list(self):
        endpoint = accounts.AccountList()
        self.client.request(endpoint)
        return endpoint.response
    
    def account_summary(self):
        endpoint = accounts.AccountSummary(self.accountID)
        self.client.request(endpoint)
        return endpoint.response

    """Instrument info"""
    #see http://developer.oanda.com/rest-live-v20/instrument-ep/
    def instruments_candles(
        self, instrument, granularity, count, 
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
            "granularity":  granularity if type(granularity)==str else granularity_dict[granularity], 
            }

        if count:
            params["count"] = count
        if price:
            params['price'] = price
        if _from:
            params['from'] = _from
        if _to:
            params['to'] = _to

        candles = []
        for r in InstrumentsCandlesFactory(instrument=instrument, params=params):
            rv = self.client.request(r)
            candles += rv['candles']
        return candles

    def instruments_order_book(self, instrument, params={}):
        endpoint = instruments.InstrumentsOrderBook(instrument, params)
        ob = self.client.request(endpoint)
        return ob["orderBook"]["buckets"]

    def instruments_position_book(self, instrument, params={}):
        endpoint = instruments.InstrumentsPositionBook(instrument, params)
        ob = self.client.request(endpoint)
        return ob["positionBook"]["buckets"]

    """Orders"""

    def create_order(
        self, instrument, units, stop_loss, 
        take_profit, comment="", strategy="", other=""
        ):

        clientExtensions = {
            "comment": comment,
            "tag": strategy,
            "other" : other
            }

        mktOrder = MarketOrderRequest(
                    instrument=instrument,
                    units=units,
                    takeProfitOnFill=TakeProfitDetails(price=take_profit).data,
                    stopLossOnFill=StopLossDetails(price=stop_loss).data,
                    tradeClientExtensions=clientExtensions
                    )
            
        endpoint = orders.OrderCreate(self.accountID, mktOrder.data)
        t = self.client.request(endpoint)
        trade_id = t["orderFillTransaction"]["tradeOpened"]["tradeID"]
        batch_id = t["orderCreateTransaction"]["batchID"]
        return trade_id, batch_id

    def list_orders(self):
        endpoint = orders.OrderList(self.accountID)
        self.client.request(endpoint)
        return endpoint.response


    def cancel_order(self, orderID):
        endpoint = orders.OrderCancel(self.accountID, orderID)
        self.client.request(endpoint)
        return endpoint.response

    """Positions"""

    def open_positions(self):
        endpoint = positions.OpenPositions(self.accountID)
        op = self.client.request(endpoint)
        return op["positions"]

    def list_positions(self):                
        endpoint = positions.PositionList(self.accountID)
        op = self.client.request(endpoint)
        return op["positions"]
        
    def close_position(self, instrument, data={"longUnits": "ALL"}):
        endpoint = positions.PositionClose(self.accountID, instrument, data)
        self.client.request(endpoint)
        return endpoint.response

    def position_details(self, instrument):
        endpoint = positions.PositionDetails(self.accountID, instrument)
        self.client.request(endpoint)
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

        endpoint = pricing.PricingInfo(self.accountID, params)
        p = self.client.request(endpoint)
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

        pass

    def trade_close(self, trade_id, units=None):
        data = {}
        if units:
            data = {"units" : units}

    def trade_details(self, trade_id):
        pass

    def trades_list(self, params={}):
        pass