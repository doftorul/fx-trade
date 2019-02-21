import sys
import json

from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20 import API

from config import getconfig
from utils import granularity_dict

class DataFactory(object):
    def __init__(self):
        self.conf = getconfig()
        self.accountID, token = self.conf["conf"]["active_account"], self.conf["conf"]["token"]
        self.client = API(access_token=token, environment="practice")
        self.factory = InstrumentsCandlesFactory
    
    def __call__(
        self, instrument, granularity, count, 
        _from=None, _to=None, price="MBA" 
        ):

        """
        granularity is [seconds]
        """

        params = { 
            "granularity": granularity_dict[granularity],
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
        for r in self.factory(instrument=instrument, params=params):
            rv = self.client.request(r)
            candles += rv['candles']

        return candles