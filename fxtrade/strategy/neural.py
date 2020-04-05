import torch

from fxtrade.strategy.interface import Strategy
from fxtrade.optimize.agents import A2C, PolicyNetwork
from fxtrade.data.indicators import add_features

BUY = 1
SELL = -1
HOLD = 0

signals = torch.tensor([BUY, SELL, HOLD])  #1, -1, 0

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
        self.net = PolicyNetwork(
            num_features=kwargs.get("features", 14), 
            num_actions=kwargs.get("actions", 3)
            )
        self.net.load_state_dict("{}/A2C.pth".format(kwargs.get("weights_dir")))

    def action(self, candles):
        
        candles = add_features(self.extract_prices(candles))

        action = signals[torch.argmax(self.net(torch.tensor(candles)))].item()

        return ["hold", "buy", "sell"][action]