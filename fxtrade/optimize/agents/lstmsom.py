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

                        
            learning_rate = self.beta * neighbourhood_func.to(device)

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
        self.num_layers = num_layers

        self.lstm = nn.GRU(
            input_size = num_features,
            hidden_size = hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bias=False,
            # dropout=0.5
            )

        #self.reduce = nn.Linear(num_layers, 1)

        self.fc = nn.Linear(hidden_size*num_layers, num_actions, bias=False)

        self.softmax = nn.Softmax()
    
    def forward(self, x):
        _, h_out = self.lstm(x)


        h_out = h_out.view(-1, self.hidden_size*self.num_layers)

        # o = self.softmax(self.fc(h_out))
        o = self.fc(h_out)

        return o

class PolicyNetworkCNN(nn.Module):
    def __init__(self, num_actions, hidden_size=64, gru_layers=4):
        super(PolicyNetworkCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3), stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2,2), stride=2)

        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,2))
        self.fc2 = nn.Conv2d(in_channels=16, out_channels=num_actions, kernel_size=(5,5))

        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()

    def forward(self, state):

        x = self.activation(self.pool(self.bn1(self.conv1(state))))
        x = self.activation(self.pool(self.bn2(self.conv2(x))))
        x = self.activation(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        x = x.squeeze(2).squeeze(2)

        return x

class DeepMotorMap(object):
    def __init__(self, state_dim=3, action_dim=3, policy_lr=0.0001, 
    seq_len=100, beta=0.15, optimiser="SGD", conv=True, som=True):

        if not conv:
            self.classifier = DeepLSTM(
                num_features=state_dim, 
                num_actions=action_dim,
                hidden_size=300,
                num_layers=2
                )

            self.classifier_input = "signals"

        else:
            self.classifier = PolicyNetworkCNN(action_dim)
            self.classifier_input = "state"

        if som:
            self.motormap = SelfOrganizingMap(m=30, n=30, dim=seq_len, beta=beta)
            self.motormap.to(device)
        else:
            self.motormap = None

        # self.lstm.apply(weight_init)
        self.classifier.to(device)

        optimiser_ = getattr(optim, optimiser)
        self.optimizer = optimiser_(self.classifier.parameters(), lr=policy_lr)
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.NLLLoss()


        self.writer = SummaryWriter()

        # TODO: Add tensorboard writer

    @staticmethod
    def ensure(d):
        if not os.path.exists(d):
            os.makedirs(d)

    def train(self, dataloader, epochs= 5, window=100, num_steps=None, save_dir=""):

        self.classifier.train()

        idx = 0

        try:
            for epoch in range(epochs):
                
                penalty = 0.

                for batch in tqdm(dataloader):

                    self.optimizer.zero_grad()


                    
                    
                    state = batch[self.classifier_input].to(device)

                    real_trends = batch["trend"].to(device).long()

                    profits = torch.abs(batch["profit"])

                    overall_profit = torch.sum(profits)

                    lstm_trends = self.classifier(state)
                    lstm_trends_argmax = torch.argmax(lstm_trends, dim=1)

                    lstm_trends_argmax_signs = torch.tensor([idx2sign[l.item()] for l in lstm_trends_argmax])

                    lstm_profit = lstm_trends_argmax_signs*batch["profit"].squeeze()
                    
                    lstm_profit = torch.sum(lstm_profit)

                    # for x in self.lstm.parameters():
                    #     print(x.grad)
                    loss = self.criterion(lstm_trends,real_trends)
                    loss.backward()
                    # for x in self.lstm.parameters():
                    #     print(x.grad)
                    self.optimizer.step()

                    
                    
                    self.writer.add_scalar("losses/loss", loss.data, idx)
                    self.writer.add_scalars("profit/", {
                        "lstm" : lstm_profit.item(),
                        "real" : overall_profit.item()
                    }, idx)

                    if self.motormap is not None:
                        signals = batch["signals"].to(device)
                        batch_size = signals.shape[0]
                        for j in range(batch_size):

                            penalty += self.motormap(
                                x=signals[j],
                                lstm_trend=lstm_trends_argmax.detach()[j],
                                real_trend=real_trends[j]
                            )

                        self.writer.add_scalar("rl/penalty", penalty, idx)


                        self.writer.add_images("rl/motormap", self.motormap.output_layer.data.view(30,30)*127, idx, dataformats="HW")

                    # print(real_trends.detach().cpu().numpy())
                    # print(lstm_trends_argmax.detach().cpu().numpy())
                    # print(lstm_trends.detach().cpu().numpy())

                    self.writer.add_histogram("train/actions", lstm_trends_argmax.detach().cpu().numpy(), idx)
                    self.writer.add_histogram("train/probs", lstm_trends.detach().cpu().numpy(), idx)
                    self.writer.add_histogram("train/real_trends", real_trends.detach().cpu().numpy(), idx)




                    idx += 1
            
            
            self.ensure(save_dir)
            if self.motormap is not None:
                torch.save(
                    self.motormap.state_dict(),
                    '{}/motormap.pth'.format(save_dir)
                )
            torch.save(
                self.classifier.state_dict(),
                '{}/classifier.pth'.format(save_dir)
            )
            
        except:
            traceback.print_exc()
            self.ensure(save_dir)
            if self.motormap is not None:
                torch.save(
                    self.motormap.state_dict(),
                    '{}/motormap.pth'.format(save_dir)
                )
            torch.save(
                self.classifier.state_dict(),
                '{}/classifier.pth'.format(save_dir)
            )
