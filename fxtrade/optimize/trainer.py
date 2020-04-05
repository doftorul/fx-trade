import torch
import torch.nn as nn
from fxtrade.optimize.agents.a2c_batched import A2C
# from fxtrade.optimize.agents.a2c_batched_small import A2C_small
from fxtrade.optimize.environment import CandlesBatched, TradingEnvironment
from fxtrade.data.factory import Downloader, get_time_interval
from torch.utils.data import DataLoader
import json
import logging

NEURAL_NET_DICTIONARY = {
    "A2C": A2C
    }


class Trainer(object):
    def __init__(self, config):
        self.net = NEURAL_NET_DICTIONARY[config["train"]["net"]]()
        self.epochs = config["train"]["epochs"]
        self.batch_size = config["train"]["batch_size"]
        self.downloader = Downloader(config["exchange"]["token"], config["exchange"]["environment"])
        self.weeks_training = config["train"]["weeks"]
        self.instruments = config["exchange"]["pair_whitelist"]

    def run(self):

        days_from_to = get_time_interval(self.weeks_training)

        candles = []
        instruments = []

        for (_from,_to) in days_from_to:
            for i in self.instruments:
                c = self.downloader(
                    instrument = i,
                    _from = _from,
                    _to = _to
                )
                candles.append(c)
                instruments.append(i)

        dataset = CandlesBatched(datapath=candles, instrument=instruments)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.net.train(dataloader=dataloader, epochs=self.epochs)




