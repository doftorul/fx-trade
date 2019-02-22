import sys
import json

from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20 import API

from libs.utils import GRANULARITIES

class DataFactory(object):
    def __init__(self, config: dict):
        self._conf = config
        self.account_id, token = \
            self._conf["exchange"]["active_account"], \
            self._conf["exchange"]["token"]
        env = config.get("environment", "practice")
        self._api = API(access_token=token, environment=env)
        self.factory = InstrumentsCandlesFactory
    
    def __call__(self, instrument: str, granularity:str, count: int = 50, 
        _from=None, _to=None, price="MBA" 
        ):

        """
        granularity input is given as a number  [seconds] 
        or as a string following libs.utils.granularity_dict values
        
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
                else GRANULARITIES[granularity]
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