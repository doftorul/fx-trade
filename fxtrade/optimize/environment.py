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
from pyts.image import GramianAngularField
from fxtrade.optimize.agents.lstmsom import idx2signal

import logging
logger = logging.getLogger(__name__)

Transition = namedtuple('Transition', ['state', 'next_state', 'profit'])

#from fxtrade.data.indicators import add_features, compute_trend, normalize
from fxtrade.data.indicators import add_features, compute_trend, norm_by_latest_close_triplet
from sklearn.preprocessing import normalize
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

        self.timestamp = 0



    def step(self, action):

        transation = self.data[self.timestamp]
        reward = transation.profit * action

        if not reward: reward = -1

        next_state = transation.next_state
        self.timestamp += 1

        if self.timestamp == self.batch_len-1:
            done = True
        else:
            done = False
        
        
        return next_state, reward, done

    
    def reset(self):
        self.timestamp = random.randint(0,self.batch_len-self.steps)
        transation = self.data[self.timestamp]
        return transation.state


SAVED_CANDLES_DEFAULT_PATH = "fxtrade/data/candles/{}.pkl"

class CandlesBatched(Dataset):
    def __init__(self, datapath, window=50, steps=5, instrument=None, load=False, save=False):
        
        if not load and not os.path.exists(SAVED_CANDLES_DEFAULT_PATH):
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
                                # round(pip_conversion*(c[i+1:i+window+1][-1][5]-c[i:i+window][-1][4]),1) bidclose-askclose
                                # round(pip_conversion*(c[i+1:i+window+1][-1][1]-c[i:i+window][-1][1]),1) #midclose - midclose
                                (c[i+1:i+window+1][-1][1]/c[i:i+window][-1][1]) #midclose - midclose
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

            if save:
                with open(SAVED_CANDLES_DEFAULT_PATH, "w") as json_file:
                    json.dump(self.samples, json_file)
        else:
            self.samples = json.load(open(SAVED_CANDLES_DEFAULT_PATH, "r"))

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




SAVED_ARBITRAGE_DEFAULT_PATH = "fxtrade/data/arbitrage/{}.pkl"

# dataloader = DataLoader(dataset, batch_size=16, shuffle=True

class TriangularArbitrage(Dataset):
    def __init__(self, data, triplet="EUR_GBP_USD", window=100, next_steps=60, skip=20, instrument=None, load=False, save=True):

        self.signal2idx = {v:k for k,v in idx2signal.items()}
        
        currencies = triplet.split("_")


        self.samples = []


        prices = data["prices"]
        x_min = data["min"]
        x_max = data["max"]
            

        logger.info("Creating batches for {}".format(triplet))

        len_data = len(prices)

        # prices_normalized = normalize(prices).tolist()#, x_min=x_min, x_max=x_max)

        for i in range(0, len_data-window-next_steps, skip):
            self.samples.append(
                (
                    #norm_by_latest_close_triplet(prices[i:i+window]), 
                    prices[i:i+window], 
                    *compute_trend([p[2] for p in prices[i+window:i+window+next_steps]])#midclose - midclose
                    # prices[i+skip:i+window+skip], 
                )
            )


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state = torch.FloatTensor(self.samples[idx][0])
        # print(state_stack.shape)
        trend =  self.signal2idx[self.samples[idx][1]]
        reward = self.samples[idx][2]

        return {
            "state" : state,
            "trend" : trend,
            "profit" : reward,
        }



class GramianFieldDataset(Dataset):
    def __init__(self, data, window=100, next_steps=60, skip=20, instrument=None, load=False, save=True):

        self.signal2idx = {v:k for k,v in idx2signal.items()}

        self.samples = []
        self.signals = []

        self.gasf_dict = {
            ins : GramianAngularField(image_size=window, method='summation') for ins in instrument
        }


        for ins, c in zip(instrument, data):
            prices = [p[1] for p in c]  #close price

            # x_min = data["min"]
            # x_max = data["max"]
                
            self.gasf_dict[ins].fit(np.array(prices))

            len_data = len(prices)

            # prices_normalized = normalize(prices).tolist()#, x_min=x_min, x_max=x_max)

            for i in range(0, len_data-window-next_steps, skip):
                self.samples.append(
                    (
                        #norm_by_latest_close_triplet(prices[i:i+window]), 
                        self.gasf_dict[ins].transform(np.array([prices[i:i+window]])), 
                        self.gasf_dict[ins].transform(np.array([prices[i+skip:i+window+skip]])), 
                        *compute_trend([p for p in prices[i+window:i+window+next_steps]])#midclose - midclose
                        # prices[i+skip:i+window+skip], 
                    )
                )


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state = torch.FloatTensor(self.samples[idx][0])
        next_state = torch.FloatTensor(self.samples[idx][1])
        # print(state_stack.shape)
        trend =  self.signal2idx[self.samples[idx][2]]
        reward = torch.FloatTensor([self.samples[idx][3]])

        return {
            "state" : state,
            "next_state" : next_state,
            "trend" : trend,
            "profit" : reward,
        }