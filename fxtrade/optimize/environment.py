import numpy as np
import json
from collections import namedtuple
import random
import itertools
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import os

import logging
logger = logging.getLogger(__name__)

Transition = namedtuple('Transition', ['state', 'next_state', 'profit'])

from fxtrade.data.indicators import add_features

class TradingEnvironment():
    def __init__(self, datapath, window=50, steps=10):

        with open(datapath, "r") as dp: 
            candles = json.load(dp) 

        pip_conversion = 10000 if "JPY" not in datapath else 100

        len_data = len(candles)
        self.steps = steps

        self.data = []
        for i in range(0, len_data-window):
            self.data.append(
                Transition(
                    candles[i:i+window], 
                    candles[i+1:i+window+1], 
                    round(pip_conversion*(candles[i+1:i+window+1][-1][4]-candles[i:i+window][-1][5]),1)
                )
            )
    
        self.batch_len = len(self.data)

        

        # state = time-series(
        # [
        #   mid open, 
        #   mid close, 
        #   mid high,
        #   mid low,
        #   bid close,
        #   ask close,
        #   volume
        #  ]
        # Example
        # [
        #     "1.12590",
        #     "1.12598",
        #     "1.12598",
        #     "1.12590",
        #     "1.12606",
        #     "1.12591",
        #     10
        # ],

        self.position = 0



    def step(self, action):

        transation = self.data[self.position]
        reward = transation.profit * action

        if not reward: reward = -1

        next_state = transation.next_state
        self.position += 1

        if self.position == self.batch_len-1:
            done = True
        else:
            done = False
        
        
        return next_state, reward, done

    
    def reset(self):
        self.position = random.randint(0,self.batch_len-self.steps)
        transation = self.data[self.position]
        return transation.state




class CandlesBatched(Dataset):
    def __init__(self, datapath, window=50, steps=5, instrument=None):

        if type(datapath) == str:
            with open(datapath, "r") as dp: 
                candles = json.load(dp)
            pip_conversion = 10000 if "JPY" not in datapath else 100
        else:
            candles = datapath
            # pip_conversion = 10000 if "JPY" not in instrument else 100
        self.steps = steps
        
        self.samples = []

        if instrument:
            for ins, c in zip(instrument, candles):

                logger.info("Creating batches for {}".format(ins))

                pip_conversion = 10000 if "JPY" not in ins else 100

                len_data = len(c)

                data = []
                for i in range(0, len_data-window):
                    data.append(
                        (
                            c[i:i+window], 
                            c[i+1:i+window+1], 
                            round(pip_conversion*(c[i+1:i+window+1][-1][4]-c[i:i+window][-1][5]),1)
                        )
                    )
            
                lendata = len(data)

                featured_data = []

                for d in tqdm(data):
                    featured_data.append([add_features(d[0]), add_features(d[1]), d[2]])

                
                
                for d in range(0, lendata-self.steps):
                    actual = featured_data[d:d+self.steps]
                    st = [a[0] for a in actual]
                    ns = [a[1] for a in actual]
                    p = [a[2] for a in actual]
                    self.samples.append(
                        [st, ns, p]
                    )
        else:
            len_data = len(candles)

            ## TODO: add spread (ask-bid)

            data = []
            for i in range(0, len_data-window):
                data.append(
                    (
                        candles[i:i+window], 
                        candles[i+1:i+window+1], 
                        round(pip_conversion*(candles[i+1:i+window+1][-1][4]-candles[i:i+window][-1][5]),1)
                    )
                )
        
            lendata = len(data)

            featured_data = []

            for d in tqdm(data):
                featured_data.append([add_features(d[0]), add_features(d[1]), d[2]])

            
            self.samples = []
            for d in range(0, lendata-self.steps):
                actual = featured_data[d:d+self.steps]
                st = [a[0] for a in actual]
                ns = [a[1] for a in actual]
                p = [a[2] for a in actual]
                self.samples.append(
                    [st, ns, p]
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state_stack = self.samples[idx][0]
        state_stack = list(itertools.chain.from_iterable(state_stack))
        state_stack = torch.tensor(state_stack)
        # print(state_stack.shape)
        nstate_stack = self.samples[idx][1]
        nstate_stack = list(itertools.chain.from_iterable(nstate_stack))
        nstate_stack = torch.tensor(nstate_stack)
        # print(nstate_stack.shape)
        profit_stack = self.samples[idx][2]
        profit_stack = torch.tensor(profit_stack)
        # print(profit_stack.shape)

        return {
            "state" : state_stack,
            "next_state" : nstate_stack,
            "profit" : profit_stack,
        }





# dataloader = DataLoader(dataset, batch_size=16, shuffle=True,