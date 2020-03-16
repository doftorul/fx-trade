import numpy as np

BUY = 1
SELL = -1
HOLD = 0

class Environment(object):
    def __init__(self, instrument, exchange):

        self.position = 0
        self.exchange = exchange
        self.data = self.exchange.get_history(instrument, 30, 480) #480 candles, 30 seconds granularity

        self.memory = []

        self.rewards = []
        self.epoch_reward = 0
        

    def step(self, action):

        return self.next_state, self.reward, self.done, {}

    
    def get_reward(self):

        return reward

    
    def reset(self):
        
        return self.observation

