"""
Inspired by 
Article
Deep LSTM with Reinforcement Learning Layer for Financial Trend Prediction in FX High Frequency Trading Systems
Francesco Rundo
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from fxtrade.optimize.agents.utils import weight_init

device = torch.device("cuda" if use_cuda else "cpu")

idx2signal = {
    0 : "sell",
    1 : "buy",
    2 : "hold"
}


def pairwise_squared_distances(x, w):
    '''                                                                                              
    Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3         
    Input: x is a bxNxd matrix y is an optional bxMxd matirx                                                             
    Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
    i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2                                                         
    '''                   

    if len(x.shape) < 3:
        x = x.unsqueeze(0)
        y = w.unsqueeze(0)
    else:
        batch_size = x.shape[0]
        y = torch.stack([w for i in range(batch_size)])

    x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
    y_t = y.permute(0,2,1).contiguous()
    y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    dist[dist != dist] = 0 # replace nan values with 0

    return torch.clamp(dist, 0.0, np.inf) 

class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    def __init__(self, m=30, n=30, dim=100, beta=0.15, sigma=None):
        super(SOM, self).__init__()
        self.m = m
        self.n = n
        self.dim = dim
        # self.niter = niter
        if beta is None:
            self.beta = 0.15
        else:
            self.beta = float(beta)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        self.weights = nn.Parameter(torch.randn(m*n, dim), requires_grad=False)
        self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))

        #randomly initialize output motor map
        self.output_layer = nn.Parameter(torch.LongTensor(m*n).random_(0, 3), requires_grad=False)

    # def get_weights(self):
    #     return self.weights

    # def get_locations(self):
    #     return self.locations

    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])

    def penalty(self, real_trend, lstm_trend, som_trend):

        lstm_error = torch.abs(real_trend - lstm_trend)
        som_error = torch.abs(real_trend - som_trend)

        if max(lstm_error, som_error) == 0:
            new_output = lstm_trend
            update_weights = True
            penalty = -1
        elif (max(lstm_error, som_error) > 0) and (lstm_error==0):
            new_output = lstm_trend
            update_weights = False
            penalty = 1
        elif (max(lstm_error, som_error) > 0) and (som_error==0):
            new_output = som_trend
            update_weights = True
            penalty = 0
        else:
            new_output = real_trend
            update_weights = False
            penalty = 0

        return penalty, new_output, update_weights

    def forward(self, x, lstm_trend, real_trend):

        # x should have shape [Batch size, 3, 100] or [3, 100] if single element
        dists = pairwise_squared_distances(x, self.weights) #Batch size x 3 (signals) x 900 (30x30 map)

        #sums up the three signals along first axis and square root
        dists = torch.sqrt(torch.sum(dists, 1))  #Batch size x 900 (30x30 map)

        values, bmu_index = torch.min(dists, 1) # Batch-size 1-D tensors

        #retrieve the (x,y) positions on weights map
        bmu_loc = self.locations[bmu_index,:]  # Batch size x 2 (x,y)
        bmu_loc = bmu_loc.squeeze()

        som_trend = self.output_layer[bmu_index.squeeze()]

        penalty, new_output, update_weights = self.penalty(real_trend, lstm_trend, som_trend)

        self.output_layer[bmu_index.squeeze()] = new_output

        if update_weights:

            #learning_rate_op = 1.0 - it/self.niter
            # TODO: add decay learning rate 
            # beta = self.beta * learning_rate_op

            x_currency = x.squeeze(0)[-1]

            stack_bmu_loc = torch.stack([bmu_loc for i in range(self.m*self.n)])
            diff_locs = (self.locations.float()-stack_bmu_loc.float())
            pow_diff_locs = torch.pow(diff_locs, 2)
            bmu_distance_squares = torch.sum(pow_diff_locs, 1)



            # neighbourhood needs to decrease with time (so self.beta should decrease with time)
            sigma_op = self.sigma * self.beta

            neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))

                        
            learning_rate = self.beta * neighbourhood_func

            learning_rate_multiplier = torch.stack([learning_rate[i:i+1].repeat(self.dim) for i in range(self.m*self.n)])
            
            delta = torch.mul(learning_rate_multiplier, (torch.stack([x_currency for i in range(self.m*self.n)]) - self.weights))                                         
            self.weights = torch.add(self.weights, delta)

        return penalty

    def predict(self, x):
        dists = pairwise_squared_distances(x, self.weights)
        dists = torch.sqrt(torch.sum(dists, 1))
        _, bmu_index = torch.min(dists, 1)
        return self.output_layer[bmu_index]



class DeepLSTM(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size=300, num_layers=2):
        super(DeepLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size = num_features,
            hidden_size = hidden_size,
            num_layers=num_layers,
            batch_first=True)

        #self.reduce = nn.Linear(num_layers, 1)

        self.fc = nn.Linear(hidden_size, num_actions)

        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):

        _, (h, _) = self.lstm(x)

        # h = num_layers x batch x hidden size:

        h = h.permute(1,2,0)
        # h =  batch x hidden size x num_layers
        # h = self.reduce(h)
        h = h[:,:,-1] #last layer hidden state


        # h = h.squeeze(-1)
        h = self.softmax(self.fc(h))

        return h


class DeepMotorMap(object):
    def __init__(self, state_dim=3, action_dim=3, lr=0.001, seq_len=100, beta=0.15, optimiser="Adam"):

        self.lstm = DeepLSTM(num_features=state_dim, num_actions=action_dim)
        self.lstm.apply(weight_init)

        optimiser_ = getattr(optim, optimiser)
        self.optimizer = optimiser_(self.lstm.parameters(), lr=lr)
        self.criterion = nn.NLLLoss()

        self.motormap = SOM(m=30, n=30, dim=seq_len, beta=beta)

        self.lstm.to(device)
        self.motormap.to(device)

    @staticmethod
    def ensure(d):
        if not os.path.exists(d):
            os.makedirs(d)

    def train(self, dataloader, epochs= 5, window=100, num_steps=None, save_dir=""):

        idx = 0

        losses = []
        penalties = []

        try:
            for epoch in epochs:
                
                mean_loss = 0.
                penalty = 0.

                for batch in tqdm(dataloader):

                    signals = batch["prices"].to(device)
                    real_trends = batch["trends"].to(device)

                    lstm_trends = self.lstm(signals)

                    self.optimizer.zero_grad()
                    loss = self.criterion(lstm_trends, real_trends)
                    loss.backward()
                    self.optimizer.step()

                    batch_size = signals.shape[0]
                    for j in range(batch_size):

                        penalty += self.motormap(
                            x=signals[j],
                            lstm_trend=lstm_trends.detach()[j],
                            real_trend=real_trends[j]
                        )

                    mean_loss += loss.item()/batch_size/len(dataloader)
                
                
                losses.append(mean_loss)
                penalties.append(penalty)
            
            
            self.ensure(save_dir)
            torch.save(
                self.motormap.state_dict(),
                '{}/DeepMotorMap/motormap.pth'.format(save_dir)
            )
            torch.save(
                self.lstm.state_dict(),
                '{}/DeepMotorMap/lstm.pth'.format(save_dir)
            )
            
        except:
            self.ensure(save_dir)
            torch.save(
                self.motormap.state_dict(),
                '{}/DeepMotorMap/motormap.pth'.format(save_dir)
            )
            torch.save(
                self.lstm.state_dict(),
                '{}/DeepMotorMap/lstm.pth'.format(save_dir)
            )
