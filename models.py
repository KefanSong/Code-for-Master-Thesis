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
        
        # for a second head for the second task
        self.mean_layer2 = nn.Linear(hidden_sizes[-1], act_dim) 
        self.logstd_layer2 = nn.Parameter(torch.ones(1, act_dim) * log_std)

        self.mean_layer2.weight.data.mul_(0.1)
        self.mean_layer2.bias.data.mul_(0.0)

    def forward(self, obs, task_id=0):

        out = self.mlp_net(obs)

        
        if task_id==1:
            
            mean = self.mean_layer(out)
            if len(mean.size()) == 1:
                mean = mean.view(1, -1)
            logstd = self.logstd_layer.expand_as(mean)
            std = torch.exp(logstd)
        else:
            
            mean = self.mean_layer2(out)
            if len(mean.size()) == 1:
                mean = mean.view(1, -1)
            logstd = self.logstd_layer2.expand_as(mean)
            std = torch.exp(logstd)
            
        return mean, logstd, std

    def get_act(self, obs, deterministic = False, task_id=0):
        mean, _, std = self.forward(obs, task_id)
        if deterministic:
            return mean
        else:
            return torch.normal(mean, std)

    def logprob(self, obs, act, task_id=0):
        mean, _, std = self.forward(obs, task_id)
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
    def __init__(self, obs_dim, hidden_sizes=(64, 64), activation='tanh'):
        super().__init__()

        # self.obs_dim = obs_dim

        # self.mlp_net = mlp(obs_dim, hidden_sizes, activation)
        # self.v_head = nn.Linear(hidden_sizes[-1], 1)

        # self.v_head.weight.data.mul_(0.1)
        # self.v_head.bias.data.mul_(0.0)
        self.value_nets = nn.ModuleList([Value(), Value()])

    def forward(self, obs, task_id):
        
        return self.value_nets[task_id](obs)
    


import torch
import torch.nn as nn
from torch.distributions import Normal

class SharedNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        layers = []
        prev_size = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation())
            prev_size = size
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SingleGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation=nn.Tanh, log_std=-0.5):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.mlp_net = SharedNet(obs_dim, hidden_sizes, activation)
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

    def get_act(self, obs, deterministic=False):
        mean, _, std = self.forward(obs)
        if deterministic:
            return mean
        else:
            return torch.normal(mean, std)

    def logprob(self, obs, act):
        mean, _, std = self.forward(obs)
        normal = Normal(mean, std)
        return normal.log_prob(act).sum(-1, keepdim=True), mean, std

#



class MultiGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dims, shared_hidden_sizes=(64, 64), task_hidden_sizes=(64, 64), activation=nn.Tanh, log_std=-0.5):
        super().__init__()
        self.shared_net = SharedNet(obs_dim, shared_hidden_sizes, activation)
        self.task_policies = nn.ModuleList([
            SingleGaussianPolicy(shared_hidden_sizes[-1], act_dim, task_hidden_sizes, activation, log_std) for act_dim in act_dims
            ])
        #[ task1_GaussianPolicy,task2_GaussianPolicy,....., ]

    def forward(self, obs, task_idx):
        shared_out = self.shared_net(obs)
        return self.task_policies[task_idx](shared_out)
    # added get_act under different task_id
    def get_act(self, obs, deterministic=False, task_id=1):
        mean, _, std = self.forward(obs, task_id)
        if deterministic:
            return mean
        else:
            return torch.normal(mean, std)
    def logprob(self, obs, act,task_idx):
        mean, _, std = self.forward(obs,task_idx)
        normal = Normal(mean, std)
        return normal.log_prob(act).sum(-1, keepdim=True), mean, std

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


