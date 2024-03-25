import argparse
# import gym
import gymnasium as gym
import torch.nn as nn
import time
from data_generator_two_reward_functions import DataGenerator
from models import GaussianPolicy, Value
from environment import get_threshold
from utils import *
from collections import deque

from big_foot_half_cheetah_v4 import BigFootHalfCheetahEnv

import pickle

class FOCOPS:
    """
    Implement FOCOPS algorithm
    """
    def __init__(self,
                 env,
                 policy_net,
                 value_net_list,
                 cvalue_net,
                 pi_optimizer,
                 vf_optimizer_list,
                 cvf_optimizer,
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
                 logger):


        self.env = env

        self.policy = policy_net
        self.value_net_list = value_net_list
        self.cvalue_net = cvalue_net

        self.pi_optimizer = pi_optimizer
        self.vf_optimizer_list = vf_optimizer_list
        self.cvf_optimizer = cvf_optimizer

        self.pi_loss = None
        self.vf_loss_list = [None]*len(self.value_net_list)
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


    def update_params(self, rollout, dtype, device):

        # Convert data to tensor
        obs = torch.Tensor(rollout['states']).to(dtype).to(device)
        act = torch.Tensor(rollout['actions']).to(dtype).to(device)
        vtarg = torch.Tensor(rollout['v_targets']).to(dtype).to(device).detach()
        adv = torch.Tensor(rollout['advantages']).to(dtype).to(device).detach()
        cvtarg = torch.Tensor(rollout['cv_targets']).to(dtype).to(device).detach()
        cadv = torch.Tensor(rollout['c_advantages']).to(dtype).to(device).detach()



        # Get log likelihood, mean, and std of current policy
        old_logprob, old_mean, old_std = self.policy.logprob(obs, act)
        old_logprob, old_mean, old_std = to_dytype_device(dtype, device, old_logprob, old_mean, old_std)
        old_logprob, old_mean, old_std = graph_detach(old_logprob, old_mean, old_std)


        # Store in TensorDataset for minibatch updates
        dataset = torch.utils.data.TensorDataset(obs, act, vtarg, adv, cvtarg, cadv,
                                                 old_logprob, old_mean, old_std)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.mb_size, shuffle=True)
        avg_cost = rollout['avg_cost']
        
        print('avg_cost: ', avg_cost)

        # stack average cost for two constraints
        avg_cost_stack = np.array((avg_cost[0], -avg_cost[0], avg_cost[1], -avg_cost[1]))


        # Update nu
        self.nu += self.nu_lr * (avg_cost_stack - self.cost_lim)
        # if self.nu < 0:
        #     self.nu = 0
        # elif self.nu > self.nu_max:
        #     self.nu = self.nu_max

        self.nu = np.clip(self.nu, a_min=0, a_max= self.nu_max)


        for epoch in range(self.num_epochs):

            for _, (obs_b, act_b, vtarg_b, adv_b, cvtarg_b, cadv_b,
                    old_logprob_b, old_mean_b, old_std_b) in enumerate(loader):


                # Update N reward critics
                # only update one value net for unit testing

                # potential bug for why the second agent is not learning... 
                for n in range(2):
                    value_net = self.value_net_list[n]
                    
                    mse_loss = nn.MSELoss()
                    vf_pred = value_net(obs_b)
                    self.vf_loss_list[n] = mse_loss(vf_pred, vtarg_b[:, n, :])
                    # weight decay
                    for param in value_net.parameters():
                        self.vf_loss_list[n] += param.pow(2).sum() * self.l2_reg
                    self.vf_optimizer_list[n].zero_grad()
                    self.vf_loss_list[n].backward()
                    self.vf_optimizer_list[n].step()

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
                logprob, mean, std = self.policy.logprob(obs_b, act_b)
                kl_new_old = gaussian_kl(mean, std, old_mean_b, old_std_b)
                ratio = torch.exp(logprob - old_logprob_b)

                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * adv_b[:, 0, :]*(1.0 - self.nu[0]+self.nu[1])) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)

                self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b[:, 0, :]*(1.0 - self.nu[0]+self.nu[1]) + adv_b[:, 1, :]*(1.0 - self.nu[2]+self.nu[3]))) \
                          * (kl_new_old.detach() <= self.eta).type(dtype)


                self.pi_loss = self.pi_loss.mean()
                self.pi_optimizer.zero_grad()
                self.pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                self.pi_optimizer.step()


            # Early stopping
            logprob, mean, std = self.policy.logprob(obs, act)
            kl_val = gaussian_kl(mean, std, old_mean, old_std).mean().item()
            if kl_val > self.delta:
                println('Break at epoch {} because KL value {:.4f} larger than {:.4f}'.format(epoch + 1, kl_val, self.delta))
                break



        # Store everything in log
        self.logger.update('MinR', np.min(self.score_queue[0]))
        self.logger.update('MaxR', np.max(self.score_queue[0]))
        self.logger.update('AvgR', np.mean(self.score_queue[0]))

        self.logger.update('AvgR2', np.mean(self.score_queue[1]))
        
        self.logger.update('MinC', np.min(self.cscore_queue))
        self.logger.update('MaxC', np.max(self.cscore_queue))
        self.logger.update('AvgC', np.mean(self.cscore_queue))
        self.logger.update('nu0', self.nu[0])
        self.logger.update('nu1', self.nu[1])


        # Save models
        self.logger.save_model('policy_params', self.policy.state_dict())
        # only saving the first value net, first vf optimizer and first value loss for now
        self.logger.save_model('value_params', self.value_net_list[0].state_dict())
        self.logger.save_model('cvalue_params', self.cvalue_net.state_dict())
        self.logger.save_model('pi_optimizer', self.pi_optimizer.state_dict())
        self.logger.save_model('vf_optimizer', self.vf_optimizer_list[0].state_dict())
        self.logger.save_model('cvf_optimizer', self.cvf_optimizer.state_dict())
        self.logger.save_model('pi_loss', self.pi_loss)
        self.logger.save_model('vf_loss', self.vf_loss_list[0])
        self.logger.save_model('cvf_loss', self.cvf_loss)




def train(args):
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    envname = 'BigFootHalfCheetah'
    # envs for different groups
    envs = []
    # testing one env for now
    envs.append(gym.make('HalfCheetah-v4'))
    envs.append(BigFootHalfCheetahEnv())

    # for different environments for different the groups, the observation dimension and action dimension are the same. 
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]

    # Initialize random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # env.seed(args.seed)
    
    for env in envs:
        env.reset(seed=args.seed)

    start_time = time.time()
    agents = []

    for z in range(args.num_groups):
        env = envs[z]

        # To-Do: change to group fairness constraint passed from args
        # Get constraint bounds
        # cost_lim = get_threshold(envname, constraint=args.constraint)
        cost_lim = np.inf

        # initialize new policy net, value nets and logger for each group
        # Initialize neural nets
        policy = GaussianPolicy(obs_dim, act_dim, args.hidden_size, args.activation, args.logstd)
        value_net_list = []
        for n in range(2):
            value_net_list.append(Value(obs_dim, args.hidden_size, args.activation))
        # value_net = Value(obs_dim, args.hidden_size, args.activation)
        cvalue_net = Value(obs_dim, args.hidden_size, args.activation)
        policy.to(device)
        
        for value_net in value_net_list:
            value_net.to(device)
        cvalue_net.to(device)
        
        # Initialize optimizer
        pi_optimizer = torch.optim.Adam(policy.parameters(), args.pi_lr)
        vf_optimizer_list = []
        for n in range(2):
            vf_optimizer_list.append(torch.optim.Adam(value_net_list[n].parameters(), args.vf_lr))
            # vf_optimizer_list.append(torch.optim.Adam(value_net.parameters(), args.vf_lr))
        cvf_optimizer = torch.optim.Adam(cvalue_net.parameters(), args.cvf_lr)
        
        # Initialize learning rate scheduler
        lr_lambda = lambda it: max(1.0 - it / args.max_iter_num, 0)
        pi_scheduler = torch.optim.lr_scheduler.LambdaLR(pi_optimizer, lr_lambda=lr_lambda)
        vf_scheduler_list = []
        for n in range(2):
            vf_scheduler_list.append(torch.optim.lr_scheduler.LambdaLR(vf_optimizer_list[n], lr_lambda=lr_lambda))
        cvf_scheduler = torch.optim.lr_scheduler.LambdaLR(cvf_optimizer, lr_lambda=lr_lambda)
        
        # Store hyperparameters for log
        hyperparams = vars(args)
        
        # Initialize RunningStat for state normalization, score queue, logger
        running_stat = RunningStats(clip=5)

        
        score_queue = [deque(maxlen=100), deque(maxlen=100)]
        cscore_queue = deque(maxlen=100)
        # logger = Logger(hyperparams)

        logger = Logger(hyperparams, z)
        # Initialize and train FOCOPS agent
        agents.append(FOCOPS(env, policy, value_net_list, cvalue_net,
                       pi_optimizer, vf_optimizer_list, cvf_optimizer,
                       args.num_epochs, args.mb_size,
                       args.c_gamma, args.lam, args.delta, args.eta,
                       args.nu, args.nu_lr, args.nu_max, cost_lim,
                       args.l2_reg, score_queue, cscore_queue, logger))

        


    # update one group at a time

    for k in range(200):
        for z in range(args.num_groups):
            print('====================')
            print('updating group: ', z)
            
    
            
            groups_returns = []
            # thresholds for two constraint functions
            b = np.zeros((4))
            
            for z1 in range(args.num_groups):
        
        
                agent = agents[z1]
                env = envs[z1]
                
                # agent.logger.save_model('iter', iter)
    
                # sampling 10 times more trajectories when estimating return for the fairness constraint.


                data_generator = DataGenerator(obs_dim, act_dim, args.batch_size*10, args.max_eps_len)
                rollouts = data_generator.run_traj(env, agent.policy, agent.value_net_list, agent.cvalue_net,
                                                  running_stat, agent.score_queue, agent.cscore_queue,
                                                  args.gamma, args.c_gamma, args.gae_lam, args.c_gae_lam,
                                                  dtype, device, args.constraint)
                groups_returns.append(rollouts['avg_return'])
    
                if z != z1:
                    # constraint threshold b when solving a CPO problem for a particular group z
                    # b = args.group_fairness_threshold + groups_returns[z1]
                    b[0] = args.group_fairness_threshold + groups_returns[z1][0]
                    b[1] = args.group_fairness_threshold - groups_returns[z1][0]
                    b[2] = args.group_fairness_threshold + groups_returns[z1][1]
                    b[3] = args.group_fairness_threshold - groups_returns[z1][1]
                
            print(groups_returns)
            
    
            # iterate through all other groups z1. 
            # for z1 in range(args.num_groups):
    
        
            
        
            # update policies, value functions for group z
            agent = agents[z]
            agent.cost_lim = b
            # agent.nu = nu_list[z]
            # agent.score_queue
            print('b: ', b)
            prev_scores = []



            for iter in range(args.max_iter_num):


                with open('score_queue1.pkl', 'wb') as f:
                    pickle.dump(agent.score_queue, f)
            
                env = envs[z]
                data_generator = DataGenerator(obs_dim, act_dim, args.batch_size*10, args.max_eps_len)
                rollouts = data_generator.run_traj(env, agent.policy, agent.value_net_list, agent.cvalue_net,
                                              running_stat, agent.score_queue, agent.cscore_queue,
                                              args.gamma, args.c_gamma, args.gae_lam, args.c_gae_lam,
                                              dtype, device, args.constraint)
                agent.update_params(rollouts, dtype, device)

                print('the other group return: ', groups_returns[1-z])
                print('current group return: ', rollouts['avg_return'])
                print('b: ', b)
                # Update learning rates
                pi_scheduler.step()
                for vf_scheduler in vf_scheduler_list:
                    vf_scheduler.step()
                cvf_scheduler.step()
            
                # Update time and running stat
                agent.logger.update('time', time.time() - start_time)
                agent.logger.update('running_stat', running_stat)
            
                # Save and print values
                agent.logger.dump()
    
                
                
                # stop training if the score queue stopped improving for 10 consecutive iterations.
                prev_scores.append(np.mean(agent.score_queue[0]))

                if iter >=10:
                    print('abs diff: ', np.abs(prev_scores[iter-10] - np.mean(agent.score_queue[0])))
                    if np.abs(prev_scores[iter-10] - np.mean(agent.score_queue[0])) < np.mean(agent.score_queue[0])/30:
                        break
                print('avg return: ', rollouts['avg_return'])
                print('len score queue: ', len(agent.score_queue))



        
        
        
    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FOCOPS Implementation')
    parser.add_argument('--group-fairness-threshold',type=float, default=100,
                       help='Maximum difference between the return of any two groups (Default: 100)')

    parser.add_argument('--num-groups', default=2,
                       help='The number of groups for group fairness (default: 2)')
    parser.add_argument('--env-id', default='Humanoid-v3',
                        help='Name of Environment (default: Humanoid-v3')
    parser.add_argument('--constraint', default='group fairness',
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
    parser.add_argument('--nu', type=float, default=[0, 0, 0, 0],
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
    args = parser.parse_args()

    train(args)
