import torch
from torch.distributions import Categorical

from fxtrade.strategy.interface import Strategy
from fxtrade.optimize.agents import A2C, PolicyNetwork
from fxtrade.data.indicators import add_features

BUY = 1
SELL = -1
HOLD = 0

signals = [BUY, SELL, HOLD]  #1, -1, 0

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

        self.policy_type = kwargs.get("policy", "greedy")

    def extract_prices(self, candles):
        output = []
        for candle in candles:
            output.append(
                [
                    float(candle["mid"]["o"]),
                    float(candle["mid"]["c"]),
                    float(candle["mid"]["h"]),
                    float(candle["mid"]["l"]),
                    float(candle["ask"]["c"]),
                    float(candle["bid"]["c"]),
                    candle["volume"]
                ]
            )

        return output
    
    def action(self, candles):
        candles = add_features(self.extract_prices(candles))
        policy = self.net(torch.tensor(candles).unsqueeze(0))
        if self.policy_type == "greedy":
            sample = torch.argmax(policy).item()
        elif self.policy_type == "dist":
            sample = Categorical(policy).sample().item()
        else:
            raise Exception("No policy available for {}".format(self.policy_type))

        action = signals[sample]
        return ["hold", "buy", "sell"][action]