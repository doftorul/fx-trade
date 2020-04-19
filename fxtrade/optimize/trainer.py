import torch
import torch.nn as nn
from fxtrade.optimize.agents import A2C, DQN, DeepMotorMap
from fxtrade.optimize.environment import (
    CandlesBatched, 
    TradingEnvironment, 
    TriangularArbitrage,
    GramianFieldDataset,
    SAVED_ARBITRAGE_DEFAULT_PATH,
    SAVED_CANDLES_DEFAULT_PATH
)

import pickle
from fxtrade.data.factory import Downloader, get_time_interval
from torch.utils.data import DataLoader
import json
import logging
import pandas as pd
import os
logger = logging.getLogger('fxtrade')


NEURAL_NET_DICTIONARY = {
    "A2C": A2C,
    "DQN": DQN,
    "DeepMotorMap": DeepMotorMap
    }


class Trainer(object):
    def __init__(self, config):
        self.net = NEURAL_NET_DICTIONARY[config["train"]["net"]](**config["train"].get("net_params", {}))
        self.epochs = config["train"]["epochs"]
        self.batch_size = config["train"]["batch_size"]
        self.num_steps= config["train"]["num_steps"]
        self.iters= config["train"]["iters"]
        self.window = config["train"]["window"]
        self.model_dir = config["train"]["weights_dir"]
        self.downloader = Downloader(config["exchange"]["token"], config["exchange"]["environment"])
        self.weeks_training = config["train"]["weeks"]
        self.instruments = config["exchange"]["pair_whitelist"]
        self.dataset = config["train"]["dataset"]
        self.triplet = config["train"].get("triplet","")

    def run(self):

        if self.dataset == "arbitrage":

            currencies = self.triplet.split("_")

            instruments = [
                "{}_{}".format(currencies[0], currencies[1]),
                "{}_{}".format(currencies[1], currencies[2]),
                "{}_{}".format(currencies[0], currencies[2])  #traded currency
            ]

            if not os.path.exists(SAVED_ARBITRAGE_DEFAULT_PATH.format(self.triplet)):
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

        elif self.dataset == "candles":
            if not os.path.exists(SAVED_CANDLES_DEFAULT_PATH.format("+".join(self.instruments))):
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
                    steps=self.iters,
                    instrument=instruments
                    )

                datacandles = {

                    "candles" : candles,
                    "instrument" :instruments
                }

                with open(SAVED_CANDLES_DEFAULT_PATH.format("+".join(self.instruments)), "wb") as w_file:
                    pickle.dump(datacandles, w_file)

            else:
                with open(SAVED_CANDLES_DEFAULT_PATH.format("+".join(self.instruments)), "rb") as load_file:
                    datacandles = pickle.load(load_file)

                candles = datacandles["candles"]
                instruments = datacandles["instrument"]

                dataset = CandlesBatched(
                    datapath=candles, 
                    window=self.window,
                    steps=self.iters,
                    instrument=instruments
                )
                
        elif self.dataset == "gramian":
            if not os.path.exists(SAVED_CANDLES_DEFAULT_PATH.format("+".join(self.instruments))):
                days_from_to = get_time_interval(self.weeks_training)

                candles = []
                instruments = []

                for i in self.instruments:    
                    for (_from,_to) in days_from_to:     
                        c = self.downloader(
                            instrument = i,
                            _from = _from,
                            _to = _to
                        )
                        candles.append(c)
                        instruments.append(i)
                
                dataset = GramianFieldDataset(
                    data=candles, 
                    window=self.window,
                    next_steps=self.num_steps,
                    instrument=instruments
                    )

                datacandles = {

                    "candles" : candles,
                    "instrument" :instruments
                }

                with open(SAVED_CANDLES_DEFAULT_PATH.format("+".join(self.instruments)), "wb") as w_file:
                    pickle.dump(datacandles, w_file)
                
            else:

                with open(SAVED_CANDLES_DEFAULT_PATH.format("+".join(self.instruments)), "rb") as load_file:
                    datacandles = pickle.load(load_file)

                candles = datacandles["candles"]
                instruments = datacandles["instrument"]


                dataset = GramianFieldDataset(
                    data=candles, 
                    window=self.window,
                    next_steps=self.num_steps,
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
            num_steps=self.iters, 
            window=self.window,
            save_dir=self.model_dir
            )




