from fxtrade.optimize.agents.a2c_batched import A2C
# from fxtrade.optimize.agents.a2c_batched_small import A2C_small
from fxtrade.optimize.environment import CandlesBatched, TradingEnvironment
from fxtrade.data.factory import Downloader
from fxtrade.optimize.trainer import Trainer
from torch.utils.data import DataLoader
import json
import logging


if __name__ == "__main__":
    config = json.load(open("config.json", "r"))
    trainer = Trainer(config)
    trainer.run()



