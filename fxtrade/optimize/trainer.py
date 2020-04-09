import torch
import torch.nn as nn
from fxtrade.optimize.agents.a2c_batched import A2C
from fxtrade.optimize.agents.deepq import DQN
from fxtrade.optimize.agents.lstmppo import PPO
from fxtrade.optimize.environment import CandlesBatched, TradingEnvironment
from fxtrade.data.factory import Downloader, get_time_interval
from torch.utils.data import DataLoader
import json
import logging

logger = logging.getLogger('fxtrade')


NEURAL_NET_DICTIONARY = {
    "A2C": A2C,
    "DQN": DQN,
    "PPO": PPO
    }


class Trainer(object):
    def __init__(self, config):
        self.net = NEURAL_NET_DICTIONARY[config["train"]["net"]]()
        self.epochs = config["train"]["epochs"]
        self.batch_size = config["train"]["batch_size"]
        self.num_steps= config["train"]["num_steps"]
        self.window = config["train"]["window"]
        self.model_dir = config["train"]["weights_dir"]
        self.downloader = Downloader(config["exchange"]["token"], config["exchange"]["environment"])
        self.weeks_training = config["train"]["weeks"]
        self.instruments = config["exchange"]["pair_whitelist"]

    def run(self, load=False):

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




