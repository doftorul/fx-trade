import torch
import torch.nn as nn
from fxtrade.optimize.agents.a2c_batched import A2C
from fxtrade.optimize.environment import CandlesBatched
from torch.utils.data import DataLoader


if __name__ == "__main__":

    datapath = "./fxtrade/optimize/data/candles/EUR_USD_5000_M1.json"

    dataset = CandlesBatched(datapath)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    net = A2C()
    net.train(dataloader)

