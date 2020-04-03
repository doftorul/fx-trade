import torch
import torch.nn as nn
from fxtrade.optimize.agents.a2c_batched import A2C
# from fxtrade.optimize.agents.a2c_batched_small import A2C_small
from fxtrade.optimize.environment import CandlesBatched, TradingEnvironment
from fxtrade.data.factory import Downloader
from torch.utils.data import DataLoader
import json
import logging

CONFIGS = json.load(open("config.json", "r"))

logger = logging.getLogger('fxtrade')

if __name__ == "__main__":

    # datapath = "./fxtrade/optimize/data/candles/EUR_USD_5000_M1.json"

    downloader = Downloader(CONFIGS["exchange"]["token"], CONFIGS["exchange"]["environment"])

    instruments = CONFIGS["exchange"]["pair_whitelist"]
    candles = []

    for i in instruments:
        c = downloader(
            instrument=i,
            _from=23,
            _to=27
            )
        candles.append(c)


    for i in instruments:
        c = downloader(
            instrument=i,
            _from=16,
            _to=20
            )
        candles.append(c)

    dataset = CandlesBatched(candles, instrument=instruments*2)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    net = A2C()
    net.train(dataloader, epochs=1)

