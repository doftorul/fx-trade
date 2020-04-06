from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from oandapyV20 import API
import json
import datetime
import os
import argparse
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


"""

Usage

from factory import Retriever
dw = Downloader(token="", environment="")

data = dw("EUR_USD", 9, 13, save=True, granularity="S30")

where 9 and 13 are start and end days of the current month


"""

# api = API(access_token="39e41febacb7f696aff65ba23713a553-112e0e75a1018a1ffff575cc1c28d5b0", environment="practice")

SAVEPATH="candles"

def get_time_interval(weeks_ago=1, today=None):

    """
    
    returns d, a list of tuples with start dates and end dates

    """
    d = []

    if not today: today = datetime.datetime.now()
    weekday = today.weekday()  # 0 for monday, ..., 6 for sunday

    if weekday >= 4: #if we call this method on or after friday
        start_day = today - datetime.timedelta(days=weekday) # monday
        start_day = start_day.replace(hour=0, minute=0, second=0) #monday midnight
        end_day = start_day + datetime.timedelta(days=4) #friday
        if weekday > 4:
            end_day = end_day.replace(hour=21, minute=0, second=0) #friday night

        for i in range(weeks_ago):
            s = start_day - datetime.timedelta(days=7*i) 
            e = end_day - datetime.timedelta(days=7*i) 
            d.append((s,e))
    else:
        start_day = today - datetime.timedelta(days=weekday)
        end_day = start_day + datetime.timedelta(weekday)
        start_day = start_day.replace(hour=0, minute=0, second=0)

        d.append((start_day, end_day))  # append monday 00:00 and today's time

        e_day = start_day + datetime.timedelta(days=4) #start day is monday
        e_day = e_day.replace(hour=21, minute=0, second=0)
    
        for i in range(weeks_ago):
            s = start_day - datetime.timedelta(days=7*(i+1)) 
            e = e_day - datetime.timedelta(days=7*(i+1)) 
            d.append((s,e))

    return d


class Downloader(object):
    def __init__(self, token, environment):
        self.api = API(
            access_token=token,
            environment=environment
        )

    def __call__(self, instrument, _from, _to, granularity="M1", price="MBA", count=2500, save=False):
        # from_date = now.replace(day=_from[0], month=_from[1], hour=0, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        # to_date = now.replace(day=_to[0], month=_to[1], hour=21, minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%SZ")


        from_date = _from.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_date = _to.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "from" : from_date,
            "to" : to_date,
            "price" : price,
            "granularity" : granularity,
            "count" : count,
        }

        candles = []
        for req in InstrumentsCandlesFactory(instrument=instrument, params=params):
            single_req = self.api.request(req)
            candles += single_req['candles']

        list_candles = []
        for candle in tqdm(candles):
            for k in ["bid", "ask", "mid"]:
                for x in ["o", "c", "h", "l"]:
                    candle["{}_{}".format(k, x)] = candle[k][x]
                candle.pop(k)

            list_candles.append(
                [
                    float(candle["mid_o"]),   # middle open
                    float(candle["mid_c"]),   # middle close
                    float(candle["mid_h"]),   # middle high
                    float(candle["mid_l"]),   # middle low
                    float(candle["ask_c"]),   # ask close
                    float(candle["bid_c"]),   # bid close
                    candle["volume"]
                ]
            )

        overall_count = len(list_candles)

        if save:
            now = datetime.datetime.now()

            namefile = os.path.join(
                SAVEPATH,
                "{}_from{}_to{}{}_{}_{}.json".format(
                    instrument,
                    _from,
                    _to,
                    now.strftime("%B"),
                    granularity,
                    overall_count
                )
            )

            with open(namefile, "w") as jout:
                json.dump(list_candles, jout, indent=4)

            return list_candles, namefile

        return list_candles