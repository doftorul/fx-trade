from fxtrade.strategy.interface import Strategy
from fxtrade.optimize.agents import A2C

class DeepQNetwork(Strategy):
    "DQN approach with DeepSense approximating Q function"
    def __init__(self, *args, **kwargs):
        super(DeepQNetwork, self).__init__(*args, **kwargs)
        pass

    def action(self, candles):
        pass

class AdvantageActorCritic(Strategy):
    "Asynchronous Actor-Critic approach"
    def __init__(self, *args, **kwargs):
        super(AdvantageActorCritic, self).__init__(*args, **kwargs)
        pass

    def action(self, candles):
        pass