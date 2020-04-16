import torch
import torch.nn as nn
from fxtrade.optimize.agents.a2c_batched import A2C
from fxtrade.optimize.agents.deepq import DQN
from fxtrade.optimize.agents.lstmsom import DeepMotorMap
from fxtrade.optimize.environment import (
    CandlesBatched, 
    TradingEnvironment, 
    TriangularArbitrage,
    SAVED_ARBITRAGE_DEFAULT_PATH,
    SAVED_CANDLES_DEFAULT_PATH
)

import pickle
from fxtrade.data.factory import Downloader, get_time_interval
from torch.utils.data import DataLoader
import json
import logging
import pandas as pd
logger = logging.getLogger('fxtrade')


NEURAL_NET_DICTIONARY = {
    "A2C": A2C,
    "DQN": DQN,
    "DeepMotorMap": DeepMotorMap
    }


class Trainer(object):
    def __init__(self, config, arbitrage=False):
        self.net = NEURAL_NET_DICTIONARY[config["train"]["net"]]()
        self.epochs = config["train"]["epochs"]
        self.batch_size = config["train"]["batch_size"]
        self.num_steps= config["train"]["num_steps"]
        self.window = config["train"]["window"]
        self.model_dir = config["train"]["weights_dir"]
        self.downloader = Downloader(config["exchange"]["token"], config["exchange"]["environment"])
        self.weeks_training = config["train"]["weeks"]
        self.instruments = config["exchange"]["pair_whitelist"]
        self.arbitrage = arbitrage
        if self.arbitrage: self.triplet = config["train"]["triplet"]

    def run(self, load=False):

        if self.arbitrage:

            currencies = self.triplet.split("_")

            instruments = [
                "{}_{}".format(currencies[0], currencies[1]),
                "{}_{}".format(currencies[1], currencies[2]),
                "{}_{}".format(currencies[0], currencies[2])  #traded currency
            ]

            if not load:
                candles = self.downloader.multi_assets_builder(
                    self.weeks_training, 
                    instruments=instruments, 
                    granularity="M1", 
                    price="M"
                    )

                with open(SAVED_ARBITRAGE_DEFAULT_PATH.format(self.triplet), "wb") as w_file:
                    pickle.dump(candles, w_file)
                
            else:

                with open(SAVED_ARBITRAGE_DEFAULT_PATH.format(self.triplet), "rb") as load_file:
                    candles = pickle.load(load_file)

            dataset = TriangularArbitrage(
                    data=candles, 
                    triplet=self.triplet,
                    window=self.window,
                    next_steps=self.num_steps,
                    instrument=instruments
                )

        else:
            if not load:
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
                
                dataset = CandlesBatched(
                    datapath=candles, 
                    window=self.window,
                    steps=self.num_steps,
                    instrument=instruments
                    )

            else:
                candles = json.load(open(load, "r"))
                instruments = []

                dataset = CandlesBatched(
                    datapath=candles, 
                    window=self.window,
                    steps=self.num_steps,
                    instrument=instruments
                )
            

        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
            )

        self.net.train(
            dataloader=dataloader, 
            epochs=self.epochs, 
            num_steps=self.num_steps, 
            window=self.window,
            save_dir=self.model_dir
            )




