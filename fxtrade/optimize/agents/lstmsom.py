"""
Inspired by 
Article
Deep LSTM with Reinforcement Learning Layer for Financial Trend Prediction in FX High Frequency Trading Systems
Francesco Rundo
"""
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from fxtrade.optimize.agents.utils import weight_init
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

idx2signal = {
    0 : "sell",
    1 : "buy",
    2 : "hold"
}

idx2sign = {
    0 : -1,
    1 : 1,
    2 : 0
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

    x = x.permute(0,2,1).contiguous()

    x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
    y_t = y.permute(0,2,1).contiguous()
    y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    dist[dist != dist] = 0 # replace nan values with 0

    return torch.clamp(dist, 0.0, np.inf) 

class SelfOrganizingMap(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    def __init__(self, m=30, n=30, dim=100, beta=0.15, sigma=None):
        super(SelfOrganizingMap, self).__init__()
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

        _, bmu_index = torch.min(dists, 1) # Batch-size 1-D tensors

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
            x_currency = x.permute(1,0)[-1]

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
            self.weights += delta

        return penalty

    def predict(self, x):
        dists = pairwise_squared_distances(x, self.weights)
        dists = torch.sqrt(torch.sum(dists, 1))
        _, bmu_index = torch.min(dists, 1)
        return self.output_layer[bmu_index]



class DeepLSTM(nn.Module):
    def __init__(self, num_features, num_actions, hidden_size=300, num_layers=2):
        super(DeepLSTM, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.GRU(
            input_size = num_features,
            hidden_size = hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=False)

        #self.reduce = nn.Linear(num_layers, 1)

        self.fc = nn.Linear(hidden_size, num_actions, bias=False)

        self.softmax = nn.Softmax()
    
    def forward(self, x):

        # print(x)
        # print(x.shape)
        # 1/0
        # print(x.shape)
        gru_out, h_out = self.lstm(x)


        h_out = h_out.view(-1, self.hidden_size)

        #h = h.permute(1,2,0)
        # h =  batch x hidden size x num_layers
        # h = self.reduce(h)
        #h = h[:,:,-1] #last layer hidden state


        #h = h.squeeze(-1)
        #h = self.softmax(self.fc(x))
        o = self.fc(h_out)

        return o


class DeepMotorMap(object):
    def __init__(self, state_dim=3, action_dim=3, lr=0.3, seq_len=100, beta=0.15, optimiser="Adam"):

        self.lstm = DeepLSTM(
            num_features=state_dim, 
            num_actions=action_dim,
            hidden_size=300,
            num_layers=1
            )
        self.motormap = SelfOrganizingMap(m=30, n=30, dim=seq_len, beta=beta)

        # self.lstm.apply(weight_init)
        self.lstm.to(device)
        self.motormap.to(device)

        optimiser_ = getattr(optim, optimiser)
        self.optimizer = optimiser_(self.lstm.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()


        self.writer = SummaryWriter()

        # TODO: Add tensorboard writer

    @staticmethod
    def ensure(d):
        if not os.path.exists(d):
            os.makedirs(d)

    def train(self, dataloader, epochs= 5, window=100, num_steps=None, save_dir=""):

        self.lstm.train()

        idx = 0

        losses = []
        penalties = []

        try:
            for epoch in range(epochs):
                
                mean_loss = 0.
                penalty = 0.

                for batch in tqdm(dataloader):

                    self.optimizer.zero_grad()


                    signals = Variable(batch["state"].to(device))

                    real_trends = Variable(batch["trend"].to(device).long())

                    profits = torch.abs(batch["reward"])
                    overall_profit = torch.sum(profits)

                    lstm_trends = self.lstm(signals)

                    lstm_trends_argmax = torch.argmax(lstm_trends, dim=1)

                    lstm_trends_argmax_signs = torch.tensor([idx2sign[l.item()] for l in lstm_trends_argmax])

                    lstm_profit = lstm_trends_argmax_signs*batch["reward"]
                    lstm_profit = torch.sum(lstm_profit)

                    # for x in self.lstm.parameters():
                    #     print(x.grad)
                    loss = self.criterion(lstm_trends, real_trends)
                    loss.backward()
                    # for x in self.lstm.parameters():
                    #     print(x.grad)
                    self.optimizer.step()

                    batch_size = signals.shape[0]
                    for j in range(batch_size):

                        penalty += self.motormap(
                            x=signals[j],
                            lstm_trend=lstm_trends_argmax.detach()[j],
                            real_trend=real_trends[j]
                        )

                    self.writer.add_scalar("losses/loss", loss.data, idx)
                    self.writer.add_scalar("rl/penalty", penalty, idx)

                    self.writer.add_scalars("profit/", {
                        "lstm" : lstm_profit.item(),
                        "real" : overall_profit.item()
                    }, idx)

                    self.writer.add_images("rl/motormap", self.motormap.output_layer.data.view(30,30)*127, idx, dataformats="HW")

                    print(real_trends.detach().numpy())
                    print(lstm_trends_argmax.detach().numpy())
                    print(lstm_trends.detach().numpy())

                    self.writer.add_histogram("train/actions", lstm_trends_argmax.detach().numpy(), idx)
                    self.writer.add_histogram("train/probs", lstm_trends.detach().numpy(), idx)
                    self.writer.add_histogram("train/real_trends", real_trends.detach().numpy(), idx)




                    idx += 1
                
                
                losses.append(mean_loss)
                penalties.append(penalty)
            
            
            self.ensure(save_dir)
            torch.save(
                self.motormap.state_dict(),
                '{}/motormap.pth'.format(save_dir)
            )
            torch.save(
                self.lstm.state_dict(),
                '{}/lstm.pth'.format(save_dir)
            )
            
        except:
            traceback.print_exc()
            self.ensure(save_dir)
            torch.save(
                self.motormap.state_dict(),
                '{}/motormap.pth'.format(save_dir)
            )
            torch.save(
                self.lstm.state_dict(),
                '{}/lstm.pth'.format(save_dir)
            )
