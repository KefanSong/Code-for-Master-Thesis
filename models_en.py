import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Normal


def mlp(input_size, hidden_sizes=(64, 64), activation='tanh'):

    if activation == 'tanh':
        activation = nn.Tanh
    elif activation == 'relu':
        activation = nn.ReLU
    elif activation == 'sigmoid':
        activation = nn.Sigmoid

    layers = []
    sizes = (input_size, ) + hidden_sizes
    for i in range(len(hidden_sizes)):
        layers += [nn.Linear(sizes[i], sizes[i+1]), activation()]
    return nn.Sequential(*layers)

# class MutilGaussianPolicy(nn.Module):
#     __init__():
#        shn =  SharedNet
#        mutil_task1 = GaussianPolicy()
#        mutil_task2 = GaussianPolicy()
       
#     forward(x,task2):
#        return mutil_task1(shn(x)) 
#


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation='tanh', log_std=-0.5):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.mlp_net = mlp(obs_dim, hidden_sizes, activation)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.logstd_layer = nn.Parameter(torch.ones(1, act_dim) * log_std)

        self.mean_layer.weight.data.mul_(0.1)
        self.mean_layer.bias.data.mul_(0.0)

    def forward(self, obs):
        
        out = self.mlp_net(obs)
        mean = self.mean_layer(out)
        if len(mean.size()) == 1:
            mean = mean.view(1, -1)
        logstd = self.logstd_layer.expand_as(mean)
        std = torch.exp(logstd)

        return mean, logstd, std

    def get_act(self, obs, deterministic = False):
        mean, _, std = self.forward(obs)
        if deterministic:
            return mean
        else:
            return torch.normal(mean, std)

    def logprob(self, obs, act):
        mean, _, std = self.forward(obs)
        normal = Normal(mean, std)
        return normal.log_prob(act).sum(-1, keepdim=True), mean, std



class Value(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64), activation='tanh'):
        super().__init__()

        self.obs_dim = obs_dim

        self.mlp_net = mlp(obs_dim, hidden_sizes, activation)
        self.v_head = nn.Linear(hidden_sizes[-1], 1)

        self.v_head.weight.data.mul_(0.1)
        self.v_head.bias.data.mul_(0.0)

    def forward(self, obs):
        mlp_out = self.mlp_net(obs)
        v_out = self.v_head(mlp_out)
        return v_out


class MutilValue(nn.Module):
    def __init__(self, obs_dim,num_tasks, hidden_sizes=(64, 64), activation='tanh'):
        super().__init__()
        # Individual policy networks for each task
        self.value_policies = nn.ModuleList([
            Value(obs_dim,hidden_sizes=hidden_sizes,activation=activation)
            for _ in range(num_tasks)
        ])
    def forward(self, obs,task_id):
        return self.value_policies[task_id](obs)

class MultiTaskGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, num_tasks, hidden_sizes=(64, 64), activation='tanh', log_std=-0.5):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_tasks = num_tasks

        # Shared network for feature extraction
        self.shared_net = mlp(obs_dim, hidden_sizes, activation)

        # Individual policy networks for each task
        self.task_policies = nn.ModuleList([
            GaussianPolicy(hidden_sizes[-1], act_dim, hidden_sizes, activation, log_std)
            for _ in range(num_tasks)
        ])

    def forward(self, obs, task_idx):
        shared_out = self.shared_net(obs)
        return self.task_policies[task_idx](shared_out)

    def logprob(self, obs, act, task_idx):
        mean, _, std = self.forward(obs, task_idx)
        normal = Normal(mean, std)
        return normal.log_prob(act).sum(-1, keepdim=True), mean, std

    def get_act(self, obs, task_idx, deterministic=False):
        mean, _, std = self.forward(obs, task_idx)
        if deterministic:
            return mean
        else:
            return torch.normal(mean, std)