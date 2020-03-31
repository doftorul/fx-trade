import torch
import torch.nn as nn
from fxtrade.optimize.agents.a2c_batched import A2C
from fxtrade.optimize.environment import CandlesBatched
from fxtrade.data.factory import Downloader
from torch.utils.data import DataLoader
import json

CONFIGS = json.load(open("config.json", "r"))

if __name__ == "__main__":

    # datapath = "./fxtrade/optimize/data/candles/EUR_USD_5000_M1.json"

    downloader = Downloader(CONFIGS["exchange"]["token"], CONFIGS["exchange"]["environment"])

    instrument = "EUR_USD"
    candles = downloader(
        instrument=instrument,
        _from=23,
        _to=27
        )

    dataset = CandlesBatched(candles, instrument=instrument)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    net = A2C()
    net.train(dataloader)

