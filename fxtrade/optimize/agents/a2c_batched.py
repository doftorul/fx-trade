import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
import traceback

"""import losswise
losswise.set_api_key('SA04W2342')

session = losswise.Session(tag='FxTrade with A2c', params={'batch_size': 16}, track_git=False)
graph_loss = session.graph('loss', kind='min')
graph_values = session.graph('value', kind='min')
graph_profits = session.graph('rewards', kind='min')"""


# import multiprocessing
# torch.set_num_threads(multiprocessing.cpu_count())

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

BUY = 1
SELL = -1
HOLD = 0

signals = torch.tensor([BUY, SELL, HOLD])  #1, -1, 0


class PolicyNetwork(nn.Module):
    def __init__(self, num_features, num_actions):
        super(PolicyNetwork, self).__init__()

        self.gru1 = nn.GRU(input_size = num_features,
                            hidden_size = 64,
                            #dropout=0.5, 
                            batch_first=True)

        self.gru2 = nn.GRU(input_size = 64,
                            hidden_size = 32,
                            #dropout=0.5, 
                            batch_first=True)

        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, num_actions)

        self.softmax = nn.Softmax(dim=1)

        self.activation = nn.LeakyReLU(0.1)
        # self.activation = nn.Tanh()

    def forward(self, state):
        x = self.activation(self.gru1(state)[0])
        x = self.activation(self.gru2(x)[1])
        x = x.squeeze(0)
        x = self.activation(self.linear1(x))
        x = self.softmax(self.linear2(x))
        return x

class ValueNetwork(nn.Module):
    def __init__(self, num_features):
        super(ValueNetwork, self).__init__()

        self.gru1 = nn.GRU(input_size = num_features,
                            hidden_size = 64,
                            #dropout=0.5, 
                            batch_first=True)

        self.gru2 = nn.GRU(input_size = 64,
                            hidden_size = 32,
                            #dropout=0.5, 
                            batch_first=True)

        self.linear1 = nn.Linear(32, 32)
        self.linear2 = nn.Linear(32, 1)

        self.activation = nn.LeakyReLU(0.1)
        # self.activation = nn.Tanh()

    def forward(self, state):
        x = self.activation(self.gru1(state)[0])
        x = self.activation(self.gru2(x)[1])
        x = x.squeeze(0)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x

class A2C(object):
    def __init__(self, state_dim=14, action_dim=3, gamma=0.99, 
        optimiser="Adam", value_lr=1e-3,
        policy_lr=1e-4, load_dir="", conv=False, debug=False, 
        output_dir="tensorboard", write=True, save=True, test_every=1000, test_only=False):

        if test_every:
            self.test_every = test_every

        self.save = save
        self.write = write

        self.gamma = gamma
        self.critic = ValueNetwork(state_dim)
        self.actor = PolicyNetwork(state_dim, action_dim)

        # if load_dir: self.load_model(load_dir)

        self.test_only =test_only
        if not self.test_only:
            self.init_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(output_dir, self.init_time)
            self.ensure(output_dir)
            if self.write:
                self.writer = SummaryWriter(log_dir=self.output_dir)
                #self.dummy_input = torch.autograd.Variable(torch.rand(1, state_dim))
                #self.writer.add_graph(self.critic, self.dummy_input)
                #self.writer.add_graph(self.actor, self.dummy_input)
            optimiser_ = getattr(optim, optimiser)

            self.value_optimizer = optimiser_(self.critic.parameters(), lr=value_lr)
            self.policy_optimizer = optimiser_(self.actor.parameters(), lr=policy_lr)

        
    def __call__(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value

    def compute_returns(self, next_value, rewards, masks):

        # next_value : Batch_size
        #    rewards:  Batch_size x Steps
        #    masks: Batch_size x Steps

        R = next_value  # Batch_size
        # print(R)
        returns = torch.zeros((rewards.shape[0], rewards.shape[1]))
        # print(returns.shape)
        for step in reversed(range(rewards.shape[1])):
            R = rewards[:, step] + self.gamma * R * masks[:, step]
            returns[:, step] = R
        return returns # Batch_size x Steps

    def train(self, dataloader, epochs= 5, num_steps=5, window=50, save_dir=""):

        idx = 0

        for epoch in range(epochs):
            for batch in tqdm(dataloader):
                log_probs = []
                values    = []
                rewards   = []
                potential_profits   = []
                masks     = []
                entropy = 0

                for n in range(num_steps):
                    try:
                        state = batch["state"][:, n*window:(n+1)*window]
                        dist, value = self(state) # dist = Categorical, value: Batch_size x 1
                        #print(dist)
                        action = dist.sample()  # Batch_size
                        action_values = signals[action] # Batch_size

                        potential_profit = abs(batch["profit"][:, n])

                        reward = batch["profit"][:, n] * action_values # Batch_size

                        penalties_for_holding = torch.ones(reward.shape[0])*(-1)
                        reward = torch.where(reward == 0, penalties_for_holding, reward)

                        done = 0


                        log_prob = dist.log_prob(action)  # Batch_size
                        entropy += dist.entropy().mean()   # Batch_size
                        
                        log_probs.append(log_prob)  # list of Batch_size vectors to be stacked afterwards 
                        values.append(value.squeeze()) # list of to Batch_size vectors be stacked afterwards
                        rewards.append(torch.FloatTensor(reward)) # list of Batch_size vectors to be stacked afterwards
                        masks.append(torch.ones(log_prob.shape[0])-done) # list of Batch_size vectors to be stacked afterwards
                        potential_profits.append(potential_profit)
                        # state = next_state
                        next_state = batch["next_state"][:, n*window:(n+1)*window] # Batch_size x 50 (window) x 7 (features)
                    except:
                        traceback.print_exc()
                    # episode += 1

                # next_state = torch.FloatTensor(next_state).to(device) #next state is the next state element of the last transition
                _, next_value = self(next_state)
                # print(next_value.shape)
                next_value = next_value.squeeze(1) # value: Batch_size

                potential_profits = torch.stack(potential_profits).T # Batch_size x Steps
                rewards = torch.stack(rewards).T # Batch_size x Steps
                masks = torch.stack(masks).T # Batch_size x Steps

                returns = self.compute_returns(next_value, rewards, masks)
                
                log_probs = torch.stack(log_probs).T # Batch_size x Steps
                returns   = returns.detach() # Batch_size x Steps
                values    = torch.stack(values).T # Batch_size x Steps
                #print(values.mean())
                self.writer.add_scalar("train/mean_return", returns.mean().data, idx)
                self.writer.add_scalar("train/mean_value_function", values.mean().data, idx)

                # graph_values.append(idx, {"mean_return":returns.mean().data, "mean_value_function":values.mean().data})


                self.writer.add_scalar("train/batch_reward", rewards.sum().data, idx)
                self.writer.add_scalar("train/batch_potential_profits", potential_profits.sum().data, idx)
                # graph_profits.append(idx, {"batch_reward":rewards.sum().data, "batch_potential_profits":potential_profits.sum().data})      

                advantage = returns - values
                
                #self.writer.add_scalar("train/mean_advantage", advantage.mean().data, episode)
                actor_loss  = -(log_probs * advantage.detach()).mean()
                self.writer.add_scalar("losses/actor_loss", actor_loss.data, idx)
                critic_loss = advantage.pow(2).mean()
                self.writer.add_scalar("losses/critic_loss", critic_loss.data, idx)

                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
                self.writer.add_scalar("losses/loss", loss.data, idx)

                # graph_loss.append(idx, {"actor":actor_loss.data, "critic":critic_loss.data, "loss":loss.data, "entropy":entropy.data})
                #self.writer.add_scalars("train/losses", self.unwrapper(loss), episode)
                #self.writer.add_scalars("train/entropies", self.unwrapper(entropy), episode)

                self.value_optimizer.zero_grad()
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.value_optimizer.step()
                self.policy_optimizer.step()

                idx += 1

        #session.done()
        self.ensure(save_dir)
        torch.save(
            self.actor.state_dict(),
            '{}/A2C.pth'.format(save_dir)
        )

    
    def ensure(self, d):
        if not os.path.exists(d):
            os.makedirs(d)
