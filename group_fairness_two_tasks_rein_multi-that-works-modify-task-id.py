import argparse
# import gym
import gymnasium as gym
import torch.nn as nn
import time
from data_generator_Copy1 import DataGenerator
from models import GaussianPolicy, Value

from environment import get_threshold
from utils import *
from collections import deque

from big_foot_half_cheetah_v4 import BigFootHalfCheetahEnv
from collections import deque
from itertools import combinations
import wandb
import copy

# import os
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

wandb.login()
wandb.init(project="tmlr-100")

class FOCOPS:
    """
    Implement FOCOPS algorithm
    """
    def __init__(self,
                 env,
                 agent,
                 num_epochs,
                 mb_size,
                 c_gamma,
                 lam,
                 delta,
                 eta,
                 nu,
                 nu_lr,
                 nu_max,
                 cost_lim,
                 l2_reg,
                 score_queue,
                 cscore_queue,
                 logger,
                 subgroup,
                 num_subgroups,
                 epsilon,
                 initial_nu_value
                 ):


        self.env = env
        self.agent = agent
        self.policy = agent.policy
        self.value_net = agent.value_net
        self.cvalue_net = agent.cvalue_net

        self.pi_optimizer = agent.pi_optimizer
        self.vf_optimizer = agent.vf_optimizer
        self.cvf_optimizer = agent.cvf_optimizer

        self.pi_loss = None
        self.vf_loss = None
        self.cvf_loss = None

        self.num_epochs = num_epochs
        self.mb_size = mb_size

        self.c_gamma = c_gamma
        self.lam = lam
        self.delta = delta
        self.eta = eta
        self.cost_lim = cost_lim

        self.nu = nu
        self.nu_lr = nu_lr
        self.nu_max = nu_max

        self.l2_reg = l2_reg

        self.logger = logger
        self.score_queue = score_queue
        self.cscore_queue = cscore_queue
        
        self.subgroup = subgroup
        self.num_subgroups = num_subgroups
        self.epsilon = epsilon
        self.nu = np.ones((self.num_subgroups - 1, 2)) * initial_nu_value
        

    def update_params(self, rollout, dtype, device,return_diff, other_group_return, group_id, task_id):

        # for debugging purpose
        # print('policy_id for task', task_id, id(self.policy))
        print('updating group: ', group_id, ' task: ', task_id )

        # Convert data to tensor
        obs = torch.Tensor(rollout['states']).to(dtype).to(device)
        act = torch.Tensor(rollout['actions']).to(dtype).to(device)
        vtarg = torch.Tensor(rollout['v_targets']).to(dtype).to(device).detach()
        adv = torch.Tensor(rollout['advantages']).to(dtype).to(device).detach()
        cvtarg = torch.Tensor(rollout['cv_targets']).to(dtype).to(device).detach()
        cadv = torch.Tensor(rollout['c_advantages']).to(dtype).to(device).detach()

        # Get log likelihood, mean, and std of current policy
        old_logprob, old_mean, old_std = self.policy.logprob(obs, act, task_id)
        old_logprob, old_mean, old_std = to_dytype_device(dtype, device, old_logprob, old_mean, old_std)
        old_logprob, old_mean, old_std = graph_detach(old_logprob, old_mean, old_std)


        # Store in TensorDataset for minibatch updates
        dataset = torch.utils.data.TensorDataset(obs, act, vtarg, adv, cvtarg, cadv,
                                                 old_logprob, old_mean, old_std)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.mb_size, shuffle=True)
        # avg_cost = rollout['avg_cost']
        avg_cost = rollout['avg_return']

        #change to return

        # # Update nu
        # self.nu[0] += self.nu_lr * (avg_cost - self.cost_lim)
        
        # for z in range(self.num_subgroups - 1):
        #     self.nu[z][0] -= self.nu_lr * (self.epsilon - return_diff[z])
        #     self.nu[z][0] = max(min(self.nu[z][0], self.nu_max), 0)
        #     self.nu[z][1] -= self.nu_lr * (self.epsilon + return_diff[z])
        #     self.nu[z][1] = max(min(self.nu[z][1], self.nu_max), 0)

        for z in range(self.num_subgroups - 1):
            self.nu[z][0] -= self.nu_lr * (self.epsilon + other_group_return - avg_cost)
            self.nu[z][0] = max(min(self.nu[z][0], self.nu_max), 0)
            self.nu[z][1] -= self.nu_lr * (self.epsilon - other_group_return + avg_cost)
            self.nu[z][1] = max(min(self.nu[z][1], self.nu_max), 0)


        
        # if self.nu[0] < 0:
        #     self.nu[0] = 0
        # elif self.nu[0] > self.nu_max:
        #     self.nu[0] = self.nu_max

        # # two constraint's modification
        # avg_cost_stack = np.array((avg_cost, -avg_cost))

        # self.nu += self.nu_lr * (avg_cost_stack - self.cost_lim)
        # self.nu = np.clip(self.nu, a_min=0, a_max= self.nu_max)


        for epoch in range(self.num_epochs):

            for _, (obs_b, act_b, vtarg_b, adv_b, cvtarg_b, cadv_b,
                    old_logprob_b, old_mean_b, old_std_b) in enumerate(loader):


                # Update reward critic
                mse_loss = nn.MSELoss()
                vf_pred = self.value_net(obs_b)
                self.vf_loss = mse_loss(vf_pred, vtarg_b)
                # weight decay
                for param in self.value_net.parameters():
                    self.vf_loss += param.pow(2).sum() * self.l2_reg
                self.vf_optimizer.zero_grad()
                self.vf_loss.backward()
                self.vf_optimizer.step()

                # Update cost critic
                cvf_pred = self.cvalue_net(obs_b)
                self.cvf_loss = mse_loss(cvf_pred, cvtarg_b)
                # weight decay
                for param in self.cvalue_net.parameters():
                    self.cvf_loss += param.pow(2).sum() * self.l2_reg
                self.cvf_optimizer.zero_grad()
                self.cvf_loss.backward()
                self.cvf_optimizer.step()


                # Update policy
                logprob, mean, std = self.policy.logprob(obs_b, act_b, task_id)
                kl_new_old = gaussian_kl(mean, std, old_mean_b, old_std_b)
                ratio = torch.exp(logprob - old_logprob_b)

                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b)) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)
                adv_coefficient = 1.0
                for z in range(self.num_subgroups - 1):
                    adv_coefficient += (-self.nu[z][0] + self.nu[z][1])
                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b - self.nu[0] * cadv_b[:,0,:])) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)
                self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * adv_b * adv_coefficient) \
                * (kl_new_old.detach() <= self.eta).type(dtype)


                # # two constraint's update for pi:
                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b - torch.sum(torch.tensor(self.nu.reshape(1, 2, 1)) * cadv_b, dim=1))) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)
                
                ## original focops main update with scalar nu
                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b)) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)
                self.pi_loss = self.pi_loss.mean()
                self.pi_optimizer.zero_grad()
                self.pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.pi_optimizer.step()


            # Early stopping
            logprob, mean, std = self.policy.logprob(obs, act, task_id)
            kl_val = gaussian_kl(mean, std, old_mean, old_std).mean().item()
            if kl_val > self.delta:
                println('Break at epoch {} because KL value {:.4f} larger than {:.4f}'.format(epoch + 1, kl_val, self.delta))
                break



        # Store everything in log
        self.logger.update('MinR', np.min(self.score_queue))
        self.logger.update('MaxR', np.max(self.score_queue))
        self.logger.update('AvgR', np.mean(self.score_queue))
        self.logger.update('MinC', np.min(self.cscore_queue))
        self.logger.update('MaxC', np.max(self.cscore_queue))
        self.logger.update('AvgC', np.mean(self.cscore_queue))
        # self.logger.update('nu', self.nu)

        # TO-DO: add task id as input to update params and wandb.

        wandb.log({"Group"+str(group_id)+"Task"+str(task_id)+"AvgR": np.mean(np.sort(self.score_queue))})
        
        # self.logger.update('nu0', self.nu[0])
        # self.logger.update('nu1', self.nu[1])

        # Save models
        self.logger.save_model('policy_params', self.policy.state_dict())
        self.logger.save_model('value_params', self.value_net.state_dict())
        self.logger.save_model('cvalue_params', self.cvalue_net.state_dict())
        self.logger.save_model('pi_optimizer', self.pi_optimizer.state_dict())
        self.logger.save_model('vf_optimizer', self.vf_optimizer.state_dict())
        self.logger.save_model('cvf_optimizer', self.cvf_optimizer.state_dict())
        self.logger.save_model('pi_loss', self.pi_loss)
        self.logger.save_model('vf_loss', self.vf_loss)
        self.logger.save_model('cvf_loss', self.cvf_loss)

        return rollout['avg_return']

envnames = ['HalfCheetah-v3',
    'BigFootHalfCheetah']

def make_envs(args):
    """
        make envs
    """

    
    
    class HalfCheetahRewardWrapper(gym.Wrapper):
        # TO-DO: add task_id as input to initalization
        def __init__(self, env, task_id):
            super().__init__(env)
            self.task_id = task_id
    
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            
            # TO-DO: if clause to append different task encoding to obs
            if self.task_id == 0:
                obs = np.append(obs, [0, 1, 2, 3])
                reward = reward - 2 * info['x_velocity']
            else:
                obs = np.append(obs, [4, 5, 6, 7])
            return obs, reward, terminated, truncated, info
        def reset(self, **kwargs):
            obs = self.env.reset(**kwargs)[0]


            # TO-DO: uncomment
            if self.task_id == 0:
                obs = np.append(obs, [0, 1, 2, 3])
            else:
                obs = np.append(obs, [4, 5, 6, 7])
            return obs
                

    
    envs = []
    tasks = []
    for t in range(2):
        env = gym.make('HalfCheetah-v4', render_mode='rgb_array')

        # env = gym.make('HalfCheetah-v4')
        env = HalfCheetahRewardWrapper(env, t)
        env = gym.wrappers.RecordVideo(env, video_folder='HC_T'+str(t), episode_trigger= lambda x: x % 50 == 0)
        # envname = 'HalfCheetah-v3'
        env.reset(seed=args.seed)
        tasks.append(env)
    envs.append(tasks)
    
    tasks = []
    for t in range(2):
        env = BigFootHalfCheetahEnv()

        env = HalfCheetahRewardWrapper(env, t)
        # env = gym.wrappers.RecordVideo(env, video_folder='BFHC_T'+str(t), episode_trigger = lambda x: x % 2 == 0)
        
        # envname = 'BigFootHalfCheetah'
        env.reset(seed=args.seed)
        tasks.append(env)
    envs.append(tasks)
    return envs

# def make_envs(args):
#     """
#         make envs
#     """
#     envs = []
#     tasks = []
#     for _ in range(2):
#         env = gym.make('HalfCheetah-v4')
#         # envname = 'HalfCheetah-v3'
#         env.reset(seed=args.seed)
#         tasks.append(env)
#     envs.append(tasks)
    
#     tasks = []
#     for _ in range(2):
#         env = BigFootHalfCheetahEnv()
#         # envname = 'BigFootHalfCheetah'
#         env.reset(seed=args.seed)
#         tasks.append(env)
#     envs.append(tasks)
#     return envs

class Agent:
    def __init__(self,env,envname,args,device, policy=None):

        # Add task encoding length
        obs_dim = env.observation_space.shape[0] + 4
        act_dim = env.action_space.shape[0]
    
        # policy = GaussianPolicy(obs_dim, act_dim, args.hidden_size, args.activation, args.logstd)
        if not policy:
            policy = GaussianPolicy(obs_dim, act_dim, args.hidden_size, nn.Tanh, args.logstd)
        policy.to(device)
            
        value_net = Value(obs_dim, args.hidden_size, args.activation)
        cvalue_net = Value(obs_dim, args.hidden_size, args.activation)
        
        value_net.to(device)
        cvalue_net.to(device)
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.policy = policy
        self.value_net = value_net
        self.cvalue_net = cvalue_net
        
        # Initialize optimizer
        self.pi_optimizer = torch.optim.Adam(policy.parameters(), args.pi_lr)
        self.vf_optimizer = torch.optim.Adam(value_net.parameters(), args.vf_lr)
        self.cvf_optimizer = torch.optim.Adam(cvalue_net.parameters(), args.cvf_lr)

        lr_lambda = lambda it: max(1.0 - it / args.max_iter_num, 0)
        self.pi_scheduler = torch.optim.lr_scheduler.LambdaLR(self.pi_optimizer, lr_lambda=lr_lambda)
        self.vf_scheduler = torch.optim.lr_scheduler.LambdaLR(self.vf_optimizer, lr_lambda=lr_lambda)
        self.cvf_scheduler = torch.optim.lr_scheduler.LambdaLR(self.cvf_optimizer, lr_lambda=lr_lambda)
        
        hyperparams = vars(args)
        
        self.running_stat = RunningStats(clip=5)
        self.score_queue = deque(maxlen=100)
        self.cscore_queue = deque(maxlen=100)
        self.logger = Logger(hyperparams, 1)
        # envname = envname[env]
        self.cost_lim = get_threshold(envname, constraint=args.constraint)
        

import numpy as np

def save_avg_returns(avg_return0s, avg_return1s, filename='avg_returns.npz'):
    """
    Save average returns for two subgroups to a NumPy .npz file.

    Parameters:
    avg_return0s (list): Average returns for subgroup 0.
    avg_return1s (list): Average returns for subgroup 1.
    filename (str): Name of the file to save the data to.
    """
    # Convert lists to NumPy arrays
    avg_return0s_array = np.array(avg_return0s)
    avg_return1s_array = np.array(avg_return1s)

    # Save arrays to .npz file
    np.savez(filename, avg_return0s=avg_return0s_array, avg_return1s=avg_return1s_array)

    print(f"Saved average returns to {filename}")


def train(args):

    # Initialize data type
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Initialize random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    #创建环境
    # Need two envs for each group, maybe in a two dimensional array
    envs = make_envs(args=args)
    num_subgroups = 2
    num_tasks = 2
    
    
    
    # Subgroup env setup
    fcpo = []
    data_gen = []
    agents = []
    for z in range(num_subgroups):
        # TO-DO: Need another loop for environments
        
        agents_tasks=[]
        # TO-DO: add task-encoding dimension for the shared policy
        obs_dim = envs[z][0].observation_space.shape[0] + 4
        act_dim = envs[z][0].action_space.shape[0]
        policy = GaussianPolicy(obs_dim, act_dim, args.hidden_size, nn.Tanh, args.logstd)
        for t in range(num_tasks):
            env = envs[z][t]
            envname = envnames[z]
            # print('envname',envname)
            agent = Agent(env,envname,args,device=device, policy=policy)
            # TO-DO: Assign the same policy of the agents from same group; perhaps in a different loop.
            agents_tasks.append(agent)
        agents.append(agents_tasks)

    # for z in range(num_subgroups):
    #     agents[z][1].policy = agents[z][0].policy

    #print('compare id:', agents[z][1].policy == agents[z][0].policy)


    for z in range(num_subgroups):
        # TO-DO: Need another loop for environments
        fcpo_tasks = []
        data_gen_tasks = []
        for t in range(num_tasks):

            # for some reason, deepcopy works
            # agent = copy.deepcopy(agents[z][t])
            agent = agents[z][t]
            # if t == 1:
            #     agent.policy = agents[z][0].policy
            env = copy.deepcopy(envs[z][t])
            envname = envnames[z]
            # agents.append(agent)
            #args.group_fairness_threshold
            fcpo_tasks.append(FOCOPS(env, agent,
                       args.num_epochs, args.mb_size,
                       args.c_gamma, args.lam, args.delta, args.eta,
                       args.nu, args.nu_lr, args.nu_max, agent.cost_lim,
                       args.l2_reg, agent.score_queue, agent.cscore_queue, agent.logger,
                       subgroup=z,num_subgroups=num_subgroups,
                       epsilon=args.epsilon,initial_nu_value=args.nu_init))
            
            
            # Add task encoding length to obs_dim
            data_gen_tasks.append(DataGenerator(agent.obs_dim, agent.act_dim, args.batch_size, args.max_eps_len))

        fcpo.append(fcpo_tasks)
        data_gen.append(data_gen_tasks)

    # ------------------------------------------------------- #
    #           Training Loop
    # ------------------------------------------------------- #
    start_time = time.time()
    # num tasks
    subgroup_returns = np.zeros((num_subgroups,num_tasks))
    cum_fair_violations = 0.
    # Queues for storing metrics
    running_cum_fair_return = deque(maxlen=100)
    running_fair_gap = deque(maxlen=100)
    running_subgroup_returns = [deque(maxlen=100) for _ in range(num_subgroups)]
    
    rollouts = []
    for z in range(num_subgroups):
        # TO-DO: add another loop for tasks, where rollouts is a two-dimensional list.
        # collecting rollouts for multi-tasks
        rollouts_tasks = []
        for t in range(num_tasks):
            env = envs[z][t]
            agent = fcpo[z][t].agent
            #data_gen[z].collect_data_and_targets(fcpo[z].agent, envs[z], writer)
            rollout = data_gen[z][t].run_traj(env, agent.policy, agent.value_net, agent.cvalue_net,
                                              agent.running_stat, agent.score_queue, agent.cscore_queue,
                                              args.gamma, args.c_gamma, args.gae_lam, args.c_gae_lam,
                                              dtype, device, args.constraint, t)
            rollouts_tasks.append(rollout)
        rollouts.append(rollouts_tasks)

    
    # avg_return0s = []
    # avg_return1s = []

    for iter in range(args.max_iter_num):
        # TO-DO: Add a loop for t.
        
        print("iter nums",iter)

        # commented out the extra statistics for now. 
        
        # # ---------- Log the statistics after one update --------------
        # for z in range(num_subgroups):
        #     subgroup_returns[z] = rollouts[z]['avg_return']

        # # Fairness related metrics
        # cum_fair_return = np.sum(subgroup_returns)
        # # maximum fair gap between different groups
        # gap_between_returns = []
        # for pair in combinations(range(num_subgroups), 2):
        #     gap_between_returns.append(abs(subgroup_returns[pair[0]] - subgroup_returns[pair[1]]))
        # fair_gap = max(gap_between_returns)
        

        # # extra statistics
        # fair_violation = float(fair_gap > args.epsilon)
        # cum_fair_violations += fair_violation
        
        # print("fair_return",cum_fair_return)
        # print("fair_gap",fair_gap)
        # print("cum_violations",cum_fair_violations)

        # commented out avg_returns, since we have wandb.
        # for z in range(num_subgroups):
        #     if z == 0:
        #         avg_return0s.append(subgroup_returns[z])
        #     if z == 1:
        #         avg_return1s.append(subgroup_returns[z])
        #    print(f"avg_return {z}", subgroup_returns[z])

        # # queue based plotting logic
        # # add the metrics in a queue
        # running_cum_fair_return.append(cum_fair_return)
        # running_fair_gap.append(fair_gap)

        # print(f"queue/fair_return", np.mean(running_cum_fair_return))
        # print(f"queue/fair_gap", np.mean(running_fair_gap))
    
        # ---------- Update the agents --------------
        # Note: can also update in a random update order
        for z0 in range(num_subgroups):
            
            print('compare id:', agents[z][1].policy == agents[z][0].policy)
            for t in range(num_tasks):
                
                
                
                return_diff = []
                # get diff of first wrt other
                for z1 in range(num_subgroups):
                    if z0 == z1:
                        continue
                    return_diff.append(rollouts[z0][t]['avg_return'] - rollouts[z1][t]['avg_return'])
                    other_group_return = rollouts[z1][t]['avg_return']
    
                if t == 1:
                    other_group_return = fcpo[z0][t].update_params(rollouts[z0][t], dtype, device,return_diff,other_group_return, z0, t)
                    agent = fcpo[z0][t].agent
                    #step
                    agent.pi_scheduler.step()
                    agent.vf_scheduler.step()
                    agent.cvf_scheduler.step()
                    
                    # Update time and running stat
                    agent.logger.update('time', time.time() - start_time)
                    agent.logger.update('running_stat', agent.running_stat)
        
                    # Save and print values
                    # agent.logger.dump()
                    env = envs[z0][t]
                    # agent = fcpo[z0].agent
    
                    # added task_id here
                    rollouts[z0][t] = data_gen[z0][t].run_traj(env, agent.policy, agent.value_net, agent.cvalue_net,
                                                  agent.running_stat, agent.score_queue, agent.cscore_queue,
                                                  args.gamma, args.c_gamma, args.gae_lam, args.c_gae_lam,
                                                  dtype, device, args.constraint, t)
        
    # save_avg_returns(avg_return0s, avg_return1s, filename='avg_returns.npz')

  
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FOCOPS Implementation')
    parser.add_argument('--epsilon',type=float, default=1000,
                       help='Maximum difference between the return of any two groups (Default: 1000)')
    parser.add_argument('--rounds-of-update',type=float, default=5000,
                       help='The number of times policy from each group take turn to update')
    
    parser.add_argument('--env-id', default='Humanoid-v3',
                        help='Name of Environment (default: Humanoid-v3')
    parser.add_argument('--constraint', default='velocity',
                        help='Constraint setting (default: velocity')
    parser.add_argument('--activation', default="tanh",
                        help='Activation function for policy/critic network (Default: tanh)')
    parser.add_argument('--hidden_size', type=float, default=(64, 64),
                        help='Tuple of size of hidden layers for policy/critic network (Default: (64, 64))')
    parser.add_argument('--logstd', type=float, default=-0.5,
                        help='Log std of Policy (Default: -0.5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for reward (Default: 0.99)')
    parser.add_argument('--c-gamma', type=float, default=0.99,
                        help='Discount factor for cost (Default: 0.99)')
    parser.add_argument('--gae-lam', type=float, default=0.95,
                        help='Lambda value for GAE for reward (Default: 0.95)')
    parser.add_argument('--c-gae-lam', type=float, default=0.95,
                        help='Lambda value for GAE for cost (Default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3,
                        help='L2 Regularization Rate (default: 1e-3)')
    parser.add_argument('--pi-lr', type=float, default=3e-4,
                        help='Learning Rate for policy (default: 3e-4)')
    parser.add_argument('--vf-lr', type=float, default=3e-4,
                        help='Learning Rate for value function (default: 3e-4)')
    parser.add_argument('--cvf-lr', type=float, default=3e-4,
                        help='Learning Rate for c-value function (default: 3e-4)')
    parser.add_argument('--lam', type=float, default=1.5,
                        help='Inverse temperature lambda (default: 1.5)')
    parser.add_argument('--delta', type=float, default=0.02,
                        help='KL bound (default: 0.02)')
    parser.add_argument('--eta', type=float, default=0.02,
                        help='KL bound for indicator function (default: 0.02)')
    # parser.add_argument('--nu', type=float, default=0,
    #                     help='Cost coefficient (default: 0)')
    parser.add_argument('--nu', type=float, default=[0, 0],
                        help='Cost coefficient (default: 0)')
    parser.add_argument('--nu_lr', type=float, default=0.01,
                        help='Cost coefficient learning rate (default: 0.01)')
    parser.add_argument('--nu_max', type=float, default=2.0,
                        help='Maximum cost coefficient (default: 2.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed (default: 0)')
    parser.add_argument('--max-eps-len', type=int, default=1000,
                        help='Maximum length of episode (default: 1000)')
    parser.add_argument('--mb-size', type=int, default=64,
                        help='Minibatch size per update (default: 64)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch Size per Update (default: 2048)')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of passes through each minibatch per update (default: 10)')
    parser.add_argument('--max-iter-num', type=int, default=500,
                        help='Number of Main Iterations (default: 500)')
    parser.add_argument("--nu-init", type=float, default=0,
                        help="the initial nu parameter")
    args = parser.parse_args()

    train(args)