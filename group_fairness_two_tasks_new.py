import argparse
# import gym
import gymnasium as gym
import torch.nn as nn
import time
from data_generator_two_tasks_debug import DataGenerator
from models import GaussianPolicy, Value
from environment import get_threshold
from utils import *
from collections import deque


from big_foot_half_cheetah_v4 import BigFootHalfCheetahEnv


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
                 logger,
                load_model=False,cur_diff=None,epsilon=None
                ):


        self.env = env


        self.policy = policy_net
        self.value_net_list = value_net_list
        # self.value_net = value_net
        self.cvalue_net = cvalue_net

        self.pi_optimizer = pi_optimizer
        self.vf_optimizer_list = vf_optimizer_list
        # self.vf_optimizer = vf_optimizer
        self.cvf_optimizer = cvf_optimizer

        self.pi_loss = None
        self.vf_loss_list = [None,None]
        self.cvf_loss = None

        if load_model:
            logger.load_model()
            
            self.policy.load_state_dict(logger.models['policy_params'])
            for i in range(2):
                self.value_net_list[i].load_state_dict(logger.models['value_params'][i])
                self.vf_optimizer_list[i].load_state_dict(logger.models['vf_optimizer'][i])
                self.vf_loss_list[i] = logger.models['vf_loss'][i]
                
            self.cvalue_net.load_state_dict(logger.models['cvalue_params'])

            self.pi_optimizer.load_state_dict(logger.models['pi_optimizer'])
            # self.vf_optimizer.load_state_dict(logger.models['vf_optimizer'])
            self.cvf_optimizer.load_state_dict(logger.models['cvf_optimizer'])


            self.pi_loss = logger.models['pi_loss']
            # self.vf_loss = logger.models['vf_loss']
            self.cvf_loss = logger.models['cvf_loss']

            

        
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
        if self.cur_diff is not None and self.epsilon is not None:
            self.cur_diff = cur_diff
            self.epsilon = epsilon

    def update_params(self, rollout, dtype, device):
        

        # Convert data to tensor
        obs0 = torch.Tensor(rollout['states']).to(dtype).to(device)
        obs1 = torch.Tensor(rollout['states2']).to(dtype).to(device)

        obs_list = []
        obs_list.append(obs0)
        obs_list.append(obs1)
        
        
        act = torch.Tensor(rollout['actions']).to(dtype).to(device)
        vtarg = torch.Tensor(rollout['v_targets']).to(dtype).to(device).detach()
        # adv = torch.Tensor(rollout['advantages']).to(dtype).to(device).detach()
        adv0, adv1 = rollout['advantages']
        adv0 =  torch.Tensor(adv0).to(dtype).to(device).detach()
        adv1 =  torch.Tensor(adv1).to(dtype).to(device).detach()
        cvtarg = torch.Tensor(rollout['cv_targets']).to(dtype).to(device).detach()
        cadv = torch.Tensor(rollout['c_advantages']).to(dtype).to(device).detach()


        # Get log likelihood, mean, and std of current policy

        old_logprob, old_mean, old_std = self.policy.logprob(obs0, act)
        old_logprob, old_mean, old_std = to_dytype_device(dtype, device, old_logprob, old_mean, old_std)
        old_logprob0, old_mean0, old_std0 = graph_detach(old_logprob, old_mean, old_std)

        old_logprob, old_mean, old_std = self.policy.logprob(obs1, act)
        old_logprob, old_mean, old_std = to_dytype_device(dtype, device, old_logprob, old_mean, old_std)
        old_logprob1, old_mean1, old_std1 = graph_detach(old_logprob, old_mean, old_std)

        old_mean_list = []
        old_mean_list.append(old_mean0)
        old_mean_list.append(old_mean1)

        old_std_list = []
        old_std_list.append(old_std0)
        old_std_list.append(old_std1)
        
        # Store in TensorDataset for minibatch updates
        # dataset = torch.utils.data.TensorDataset(obs, act, vtarg, adv, cvtarg, cadv,
        #                                          old_logprob, old_mean, old_std)
        dataset = torch.utils.data.TensorDataset(obs0, obs1, act, vtarg, adv0, adv1, cvtarg, cadv,
                                                 old_logprob0, old_mean0, old_std0, old_logprob1, old_mean1, old_std1)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.mb_size, shuffle=True)
        # avg_cost = rollout['avg_cost']
        avg_cost = rollout['avg_return']


        # # Update nu
        # self.nu[0] += self.nu_lr * (avg_cost - self.cost_lim)
        # if self.nu[0] < 0:
        #     self.nu[0] = 0
        # elif self.nu[0] > self.nu_max:
        #     self.nu[0] = self.nu_max


        avg_cost_stack = np.array((avg_cost[0], -avg_cost[0], avg_cost[1], -avg_cost[1]))
        # avg_cost_stack = np.array((avg_cost, -avg_cost))
        
        if self.cur_diff is not None and self.epsilon is not None:
            self.nu += self.nu_lr * (self.epsilon  -self.cur_diff)
        else:
            self.nu += self.nu_lr * (avg_cost_stack - self.cost_lim )


        self.nu = np.clip(self.nu, a_min=0, a_max= self.nu_max)
        


        for epoch in range(self.num_epochs):


            # for _, (obs_b, act_b, vtarg_b, adv_b, cvtarg_b, cadv_b,
            #         old_logprob_b, old_mean_b, old_std_b) in enumerate(loader):
            for _, (obs_b0,obs_b1, act_b, vtarg_b, adv_b0, adv_b1, cvtarg_b, cadv_b,
                    old_logprob_b0, old_mean_b0, old_std_b0, old_logprob_b1, old_mean_b1, old_std_b1) in enumerate(loader):

                obs_b_list = []
                obs_b_list.append(obs_b0)
                obs_b_list.append(obs_b1)

                old_logprob_b_list = []
                old_logprob_b_list.append(old_logprob_b0)
                old_logprob_b_list.append(old_logprob_b1)

                old_mean_b_list = []
                old_mean_b_list.append(old_mean_b0)
                old_mean_b_list.append(old_mean_b1)

                old_std_b_list = []
                old_std_b_list.append(old_std_b0)
                old_std_b_list.append(old_std_b1)

                adv_b_list = []
                adv_b_list.append(adv_b0)
                adv_b_list.append(adv_b1)


                # update n reward critic

                for h in range(2):
                    value_net = self.value_net_list[h]
                    
                    mse_loss = nn.MSELoss()
                    vf_pred = value_net(obs_b_list[h])
                    self.vf_loss_list[h] = mse_loss(vf_pred, vtarg_b[:, h, :])
                    # weight decay
                    for param in value_net.parameters():
                        self.vf_loss_list[h] += param.pow(2).sum() * self.l2_reg
                    self.vf_optimizer_list[h].zero_grad()
                    self.vf_loss_list[h].backward()
                    self.vf_optimizer_list[h].step()

                # value_net = self.value_net_list[n]
                
                # mse_loss = nn.MSELoss()
                # vf_pred = value_net(obs_b_list[n])
                # self.vf_loss_list[n] = mse_loss(vf_pred, vtarg_b[:, n, :])
                # # weight decay
                # for param in value_net.parameters():
                #     self.vf_loss_list[n] += param.pow(2).sum() * self.l2_reg
                # self.vf_optimizer_list[n].zero_grad()
                # self.vf_loss_list[n].backward()
                # self.vf_optimizer_list[n].step()


                # # Update reward critic
                # mse_loss = nn.MSELoss()
                # vf_pred = self.value_net(obs_b)
                # self.vf_loss = mse_loss(vf_pred, vtarg_b)
                # # weight decay
                # for param in self.value_net.parameters():
                #     self.vf_loss += param.pow(2).sum() * self.l2_reg
                # self.vf_optimizer.zero_grad()
                # self.vf_loss.backward()
                # self.vf_optimizer.step()

                # # Update cost critic
                # cvf_pred = self.cvalue_net(obs_b)
                # self.cvf_loss = mse_loss(cvf_pred, cvtarg_b)
                # # weight decay
                # for param in self.cvalue_net.parameters():
                #     self.cvf_loss += param.pow(2).sum() * self.l2_reg
                # self.cvf_optimizer.zero_grad()
                # self.cvf_loss.backward()
                # self.cvf_optimizer.step()


                # Update policy
                logprob, mean, std = self.policy.logprob(obs_b_list[0], act_b)
                kl_new_old = gaussian_kl(mean, std, old_mean_b_list[0], old_std_b_list[0])
                ratio = torch.exp(logprob - old_logprob_b_list[0])
                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b - self.nu * cadv_b[:,:,:])) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)

                
                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b - torch.sum(torch.tensor(self.nu.reshape(1, 2, 1)) * cadv_b, dim=1))) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)

                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b[:, 0, :]*(1.0 - self.nu[0]+self.nu[1]) + adv_b[:, 1, :]*(1.0 - self.nu[2]+self.nu[3]))) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)
                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b[:, 0, :] + adv_b[:, 1, :])) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)

                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b[:, 1, :])) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)
                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b1)) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)
                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b0*(1.0 - self.nu[0]+self.nu[1]) + adv_b1*(1.0 - self.nu[2]+self.nu[3]))) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)

                self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b_list[0])) \
                          * (kl_new_old.detach() <= self.eta).type(dtype)




                # self.pi_loss = self.pi_loss.mean()
                # self.pi_optimizer.zero_grad()
                # self.pi_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                # self.pi_optimizer.step()



                # logprob, mean, std = self.policy.logprob(obs_b_list[1], act_b)
                # kl_new_old = gaussian_kl(mean, std, old_mean_b_list[1], old_std_b_list[1])
                # ratio = torch.exp(logprob - old_logprob_b_list[1])
                
                # self.pi_loss = (kl_new_old - (1 / self.lam) * ratio * (adv_b_list[0])) \
                #           * (kl_new_old.detach() <= self.eta).type(dtype)




                # self.pi_loss = self.pi_loss.mean()
                # self.pi_optimizer.zero_grad()
                # self.pi_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40)
                # self.pi_optimizer.step()


            ## Early stopping
            logprob, mean, std = self.policy.logprob(obs_list[0], act)
            kl_val = gaussian_kl(mean, std, old_mean_list[0], old_std_list[0]).mean().item()
            if kl_val > self.delta:
                println('Break at epoch {} because KL value {:.4f} larger than {:.4f}'.format(epoch + 1, kl_val, self.delta))
                break

            # # Early stopping
            # logprob, mean, std = self.policy.logprob(obs_list[1], act)
            # kl_val = gaussian_kl(mean, std, old_mean_list[1], old_std_list[1]).mean().item()
            # if kl_val > self.delta:
            #     println('Break at epoch {} because KL value {:.4f} larger than {:.4f}'.format(epoch + 1, kl_val, self.delta))
            #     break



        # Store everything in log
        self.logger.update('MinR', np.min(self.score_queue))
        self.logger.update('MaxR', np.max(self.score_queue))
        
        # self.logger.update('AvgR', np.mean(self.score_queue))
        # get rid of the lowest 5 values when calculating the score
        # with open('score_queue.pkl', 'wb') as f:
        #     pickle.dump(self.score_queue, f)

        self.logger.update('AvgR', np.mean(np.sort(self.score_queue[0])))
        self.logger.update('AvgR2', np.mean(np.sort(self.score_queue[1])))
        self.logger.update('MinC', np.min(self.cscore_queue))
        self.logger.update('MaxC', np.max(self.cscore_queue))
        self.logger.update('AvgC', np.mean(self.cscore_queue))
        # self.logger.update('nu', self.nu)
        self.logger.update('nu0', self.nu[0])
        self.logger.update('nu1', self.nu[1])


        # Save models
        self.logger.save_model('policy_params', self.policy.state_dict())
        # self.logger.save_model('value_params', self.value_net.state_dict())
        self.logger.save_model('value_params', [self.value_net_list[i].state_dict() for i in range(2)])
        self.logger.save_model('cvalue_params', self.cvalue_net.state_dict())
        self.logger.save_model('pi_optimizer', self.pi_optimizer.state_dict())
        # self.logger.save_model('vf_optimizer', self.vf_optimizer.state_dict())
        self.logger.save_model('vf_optimizer', [self.vf_optimizer_list[i].state_dict() for i in range(2)])
        self.logger.save_model('cvf_optimizer', self.cvf_optimizer.state_dict())
        self.logger.save_model('pi_loss', self.pi_loss)
        self.logger.save_model('vf_loss', [self.vf_loss_list[i] for i in range(2)])
        self.logger.save_model('cvf_loss', self.cvf_loss)


def train(args, env_list, envname, load_model, cost_lim, group, score_queue, cscore_queue,cur_diff = None):
    if cur_diff is not None:
        # Initialize data type
        dtype = torch.float32
        torch.set_default_dtype(dtype)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize environment
        # env = BigFootHalfCheetahEnv()
        # envname = 'BigFootHalfCheetah'
        # env.reset(seed=args.seed)

        # env = gym.make('HalfCheetah-v4')
        # envname = 'HalfCheetah-v3'
        # env.reset(seed=args.seed)
        
        obs_dim = env_list[0].observation_space.shape[0] + 4
        act_dim = env_list[0].action_space.shape[0]

        # Initialize random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # env.seed(args.seed)

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
        # value_net.to(device)
        cvalue_net.to(device)

        # Initialize optimizer
        pi_optimizer = torch.optim.Adam(policy.parameters(), args.pi_lr)
        vf_optimizer_list = []
        for n in range(2):

            vf_optimizer_list.append(torch.optim.Adam(value_net_list[n].parameters(), args.vf_lr))
        # vf_optimizer = torch.optim.Adam(value_net.parameters(), args.vf_lr)
        cvf_optimizer = torch.optim.Adam(cvalue_net.parameters(), args.cvf_lr)

        # Initialize learning rate scheduler
        lr_lambda = lambda it: max(1.0 - it / args.max_iter_num, 0)
        pi_scheduler = torch.optim.lr_scheduler.LambdaLR(pi_optimizer, lr_lambda=lr_lambda)
        vf_scheduler_list = []
        for n in range(2):
            vf_scheduler_list.append(torch.optim.lr_scheduler.LambdaLR(vf_optimizer_list[n], lr_lambda=lr_lambda))
        # vf_scheduler = torch.optim.lr_scheduler.LambdaLR(vf_optimizer, lr_lambda=lr_lambda)
        cvf_scheduler = torch.optim.lr_scheduler.LambdaLR(cvf_optimizer, lr_lambda=lr_lambda)

        # Store hyperparameters for log
        hyperparams = vars(args)

        # Initialize RunningStat for state normalization, score queue, logger
        running_stats = [RunningStats(clip=5),RunningStats(clip=5)]
        # score_queue = deque(maxlen=100)
        # cscore_queue = deque(maxlen=100)
        logger = Logger(hyperparams, group)

        # Get constraint bounds
        # cost_lim = get_threshold(envname, constraint=args.constraint)
        # cost_lim = args.group_fairness_threshold

        # Initialize and train FOCOPS agent
        agent = FOCOPS(env_list, policy, value_net_list, cvalue_net,
                    pi_optimizer, vf_optimizer_list, cvf_optimizer,
                    args.num_epochs, args.mb_size,
                    args.c_gamma, args.lam, args.delta, args.eta,
                    args.nu, args.nu_lr, args.nu_max, cost_lim,
                    args.l2_reg, score_queue, cscore_queue, logger, load_model,
                    #cur_diff=None,epsilon=None
                    cur_diff=cur_diff,epsilon=args.group_fairness_threshold)


        start_time = time.time()

        for iter in range(args.max_iter_num):
            print('updating group: ', group)

            # Update iteration for model
            agent.logger.save_model('iter', iter)

            # Collect trajectories
            data_generator = DataGenerator(obs_dim, act_dim, args.batch_size, args.max_eps_len)
            rollout = data_generator.run_traj(env_list, agent.policy, agent.value_net_list, agent.cvalue_net,
                                            running_stats, agent.score_queue, agent.cscore_queue,
                                            args.gamma, args.c_gamma, args.gae_lam, args.c_gae_lam,
                                            dtype, device, args.constraint)

            # Update FOCOPS parameters
            agent.update_params(rollout, dtype, device)

            # Update learning rates
            pi_scheduler.step()
            for vf_scheduler in vf_scheduler_list:
                vf_scheduler.step()
            # vf_scheduler.step()
            cvf_scheduler.step()

            # Update time and running stat
            agent.logger.update('time', time.time() - start_time)
            agent.logger.update('running_stat', running_stats[0])

            # Save and print values
            agent.logger.dump()
        return agent.logger.log_data['AvgR'][-1],agent.logger.log_data['AvgR2'][-1], agent.score_queue, agent.cscore_queue
        
    else:
        # Initialize data type
        dtype = torch.float32
        torch.set_default_dtype(dtype)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize environment
        # env = BigFootHalfCheetahEnv()
        # envname = 'BigFootHalfCheetah'
        # env.reset(seed=args.seed)

        # env = gym.make('HalfCheetah-v4')
        # envname = 'HalfCheetah-v3'
        # env.reset(seed=args.seed)
        
        obs_dim = env_list[0].observation_space.shape[0] + 4
        act_dim = env_list[0].action_space.shape[0]

        # Initialize random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # env.seed(args.seed)

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
        # value_net.to(device)
        cvalue_net.to(device)

        # Initialize optimizer
        pi_optimizer = torch.optim.Adam(policy.parameters(), args.pi_lr)
        vf_optimizer_list = []
        for n in range(2):

            vf_optimizer_list.append(torch.optim.Adam(value_net_list[n].parameters(), args.vf_lr))
        # vf_optimizer = torch.optim.Adam(value_net.parameters(), args.vf_lr)
        cvf_optimizer = torch.optim.Adam(cvalue_net.parameters(), args.cvf_lr)

        # Initialize learning rate scheduler
        lr_lambda = lambda it: max(1.0 - it / args.max_iter_num, 0)
        pi_scheduler = torch.optim.lr_scheduler.LambdaLR(pi_optimizer, lr_lambda=lr_lambda)
        vf_scheduler_list = []
        for n in range(2):
            vf_scheduler_list.append(torch.optim.lr_scheduler.LambdaLR(vf_optimizer_list[n], lr_lambda=lr_lambda))
        # vf_scheduler = torch.optim.lr_scheduler.LambdaLR(vf_optimizer, lr_lambda=lr_lambda)
        cvf_scheduler = torch.optim.lr_scheduler.LambdaLR(cvf_optimizer, lr_lambda=lr_lambda)

        # Store hyperparameters for log
        hyperparams = vars(args)

        # Initialize RunningStat for state normalization, score queue, logger
        running_stats = [RunningStats(clip=5),RunningStats(clip=5)]
        # score_queue = deque(maxlen=100)
        # cscore_queue = deque(maxlen=100)
        logger = Logger(hyperparams, group)

        # Get constraint bounds
        # cost_lim = get_threshold(envname, constraint=args.constraint)
        # cost_lim = args.group_fairness_threshold
        # args.group_fairness_threshold
        # Initialize and train FOCOPS agent
        agent = FOCOPS(env_list, policy, value_net_list, cvalue_net,
                    pi_optimizer, vf_optimizer_list, cvf_optimizer,
                    args.num_epochs, args.mb_size,
                    args.c_gamma, args.lam, args.delta, args.eta,
                    args.nu, args.nu_lr, args.nu_max, cost_lim,
                    args.l2_reg, score_queue, cscore_queue, logger, load_model)


        start_time = time.time()

        for iter in range(args.max_iter_num):
            print('updating group: ', group)

            # Update iteration for model
            agent.logger.save_model('iter', iter)

            # Collect trajectories
            data_generator = DataGenerator(obs_dim, act_dim, args.batch_size, args.max_eps_len)
            rollout = data_generator.run_traj(env_list, agent.policy, agent.value_net_list, agent.cvalue_net,
                                            running_stats, agent.score_queue, agent.cscore_queue,
                                            args.gamma, args.c_gamma, args.gae_lam, args.c_gae_lam,
                                            dtype, device, args.constraint)

            # Update FOCOPS parameters
            agent.update_params(rollout, dtype, device)

            # Update learning rates
            pi_scheduler.step()
            for vf_scheduler in vf_scheduler_list:
                vf_scheduler.step()
            # vf_scheduler.step()
            cvf_scheduler.step()

            # Update time and running stat
            agent.logger.update('time', time.time() - start_time)
            agent.logger.update('running_stat', running_stats[0])

            # Save and print values
            agent.logger.dump()
        return agent.logger.log_data['AvgR'][-1],agent.logger.log_data['AvgR2'][-1], agent.score_queue, agent.cscore_queue



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FOCOPS Implementation')
    parser.add_argument('--group-fairness-threshold',type=float, default=1000,
                       help='Maximum difference between the return of any two groups (Default: 1000)')
    
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
    parser.add_argument('--nu', type=float, default=[0, 0, 0, 0],
                        help='Cost coefficient (default: 0)')
    # parser.add_argument('--nu', type=float, default=[0, 0],
    #                     help='Cost coefficient (default: 0)')
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


    # code for alternatively training two groups 
    
    AvgR = AvgR2 = 0
    
    
    envs = []
    
    env = gym.make('HalfCheetah-v4')
    envname = 'HalfCheetah-v3'
    env.reset(seed=args.seed)
    env2 = gym.make('HalfCheetah-v4')
    envname = 'HalfCheetah-v3'
    env2.reset(seed=args.seed+10)
    envs.append([env, env2])
    
    env = BigFootHalfCheetahEnv()
    envname = 'BigFootHalfCheetah'
    env.reset(seed=args.seed)
    
    env2 = BigFootHalfCheetahEnv()
    envname = 'BigFootHalfCheetah'
    env2.reset(seed=args.seed+10)
    envs.append([env, env2])


    score_queue_list = [[deque(maxlen=100), deque(maxlen=100)],[deque(maxlen=100), deque(maxlen=100)]]
    cscore_queue_list = [deque(maxlen=100), deque(maxlen=100)]

    # To-Do: set it to sampled return without updating the policy
    for z in range(2):
        env_list = envs[z]
        cost_lim = np.zeros(4)

        cost_lim[0] = args.group_fairness_threshold + AvgR
        cost_lim[1] = args.group_fairness_threshold - AvgR
        cost_lim[2] = args.group_fairness_threshold + AvgR2
        cost_lim[3] = args.group_fairness_threshold - AvgR2

        AvgR,AvgR2, score_queue_list[z], cscore_queue_list[z] = train(args, env_list, envname, False, cost_lim, z, score_queue_list[z], cscore_queue_list[z])
        
    
    diffs = [
        [0,0]
        
    ]
    for _ in range(1000):
        # for _ in range(2):
        # return_diff = []

        for z0 in range(2):
            for z1 in range(2):
                if z0 == z1:
                    continue
            
                ## agent1 0
                env_list = envs[z0]
        
                cost_lim[0] = args.group_fairness_threshold + AvgR
                cost_lim[1] = args.group_fairness_threshold - AvgR
                cost_lim[2] = args.group_fairness_threshold + AvgR2
                cost_lim[3] = args.group_fairness_threshold - AvgR2
                
                diffz0 = diffs[-1][z0]
                # task 
                AvgR_0,AvgR2_0, score_queue_list[z0], cscore_queue_list[z0] = train(args, env_list, envname, True, cost_lim, z, score_queue_list[z0], cscore_queue_list[z0],cur_diff=diffz0)
                #-----------
                
                ## agent2 
                env_list = envs[z1]
        
                cost_lim[0] = args.group_fairness_threshold + AvgR
                cost_lim[1] = args.group_fairness_threshold - AvgR
                cost_lim[2] = args.group_fairness_threshold + AvgR2
                cost_lim[3] = args.group_fairness_threshold - AvgR2
                
                diffz1 = diffs[-1][z1]
                # task 
                AvgR_1,AvgR2_1, score_queue_list[z1], cscore_queue_list[z1] = train(args, env_list, envname, True, cost_lim, z, score_queue_list[z1], cscore_queue_list[z1],cur_diff=diffz1)
                #-----------
                diff_1 = AvgR_0-AvgR_1
                diff_2 = AvgR2_0-AvgR2_1
                
                diffs.append([diff_1,diff_2])
                # cur_diff = (AvgR_0-AvgR_1,AvgR2_0-AvgR2_1)
                
                #
                # AvgR,AvgR, score_queue_list[z1], cscore_queue_list[z1] = train(args, env_list, envname, True, cost_lim, z, score_queue_list[z1], cscore_queue_list[z1],cur_diff=cur_diff)
                # return_diff.append(cur_diff)
        
    # return_diff
