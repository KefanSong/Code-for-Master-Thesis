import numpy as np
import torch
from utils import torch_to_numpy

import pickle

from utils import *
import copy

class DataGenerator:
    """
    A data generator used to collect trajectories for on-policy RL with GAE
    References:
        https://github.com/Khrylx/PyTorch-RL
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
        https://github.com/ikostrikov/pytorch-trpo
    """
    def __init__(self, obs_dim, act_dim, batch_size, max_eps_len):

        # Hyperparameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.max_eps_len = max_eps_len

        # Batch buffer
        self.obs_buf = np.zeros((batch_size, obs_dim), dtype=np.float32)
        self.obs_buf2 = np.zeros((batch_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((batch_size, act_dim),  dtype=np.float32)
        self.vtarg_buf = np.zeros((batch_size,2, 1), dtype=np.float32)
        self.adv_buf = np.zeros((batch_size,2, 1), dtype=np.float32)
        # self.vtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        # self.adv_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.adv_buf0 = np.zeros((batch_size, 1), dtype=np.float32)
        self.adv_buf1 = np.zeros((batch_size, 1), dtype=np.float32)
        self.cvtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cadv_buf = np.zeros((batch_size,2, 1), dtype=np.float32)

        # Episode buffer
        self.obs_eps = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.obs_eps2 = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.next_obs_eps = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.next_obs_eps2 = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.act_eps = np.zeros((max_eps_len, act_dim),  dtype=np.float32)
        # self.rew_eps = np.zeros((max_eps_len, 1),  dtype=np.float32)
        self.rew_eps = [np.zeros((max_eps_len, 1),  dtype=np.float32), np.zeros((max_eps_len, 1),  dtype=np.float32)]
        self.cost_eps = np.zeros((max_eps_len, 1), dtype=np.float32)
        self.eps_len = 0
        self.not_terminal = 1


        # Pointer
        self.ptr = 0

    def run_traj(self, env_list, policy, value_net_list, cvalue_net, running_stats,
                 score_queue_list, cscore_queue, gamma, c_gamma, gae_lam, c_gae_lam,
                 dtype, device, constraint):
        running_stat = running_stats[0]
        running_stat2 = running_stats[1]
        # running_stat2 = copy.deepcopy(running_stat)
        batch_idx = 0

        ret_hist = []
        ret_hist2 = []
        cost_ret_hist = []

        avg_eps_len = 0
        num_eps = 0

        env = env_list[0]

        env2 = copy.deepcopy(env)
        # env2 = env_list[1]



        while batch_idx < self.batch_size:

            
            obs = env.reset()[0]

            # multi-task id
            obs = np.append(obs, [0, 1, 2, 3])



            obs2 = env2.reset()[0]

            obs2 = np.append(obs2, [4, 5, 6, 7])

            
            if running_stat is not None:
                obs = running_stat.normalize(obs)
            if running_stat2 is not None:
                obs2 = running_stat2.normalize(obs2)
            ret_eps = 0
            ret_eps2 = 0
            cost_ret_eps = 0

            # with open('obs.pkl', 'wb') as f:
            #     pickle.dump(obs, f)
            



            for t in range(self.max_eps_len):



                
                act = policy.get_act(torch.Tensor(obs).to(dtype).to(device))
                act = torch_to_numpy(act).squeeze()
                next_obs, rew, done, truncated, info = env.step(act)
                next_obs = np.append(next_obs, [0, 1, 2, 3])


                # run in reverse direction
                # rew -= 2*info['x_velocity']


                
                # for debugging
                act2 = policy.get_act(torch.Tensor(obs2).to(dtype).to(device))
                act2 = torch_to_numpy(act2).squeeze()
                next_obs2, rew2, done2, _, _ = env2.step(act2)
                next_obs2 = np.append(next_obs2, [4, 5, 6, 7])


                
                # act2 = policy.get_act(torch.Tensor(obs2).to(dtype).to(device))
                # act2 = torch_to_numpy(act2).squeeze()
                # next_obs, rew, done, truncated, info = env2.step(act2)
                # next_obs = np.append(next_obs, [0, 1, 2, 3])

                
                # # for debugging
                # act2 = policy.get_act(torch.Tensor(obs2).to(dtype).to(device))
                # act2 = torch_to_numpy(act2).squeeze()
                # next_obs2, rew2, done2, _, _ = env.step(act2)
                # next_obs2 = np.append(next_obs2, [0, 1, 2, 3])

                # act = act2
                # rew_ph = rew.copy()

                


                # rew2 = -rew2
                # rew2 = rew2 - 0.3*np.abs(act).mean()


                # rew = rew2
                # rew2 = rew_ph
                
                


                # add another task: more energy efficient running
                # add a higher penalty for actions with larger magnitude. 
                
                # v = info['x_velocity']
                # rew2 = v - 0.3*np.abs(act).sum()

                
                cost_vector = [0]*2
                if constraint == 'velocity':
                    if 'y_velocity' not in info:
                        cost = np.abs(info['x_velocity'])
                    else:
                        cost = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
                elif constraint == 'circle':
                    cost = info['cost']
                elif constraint == 'group fairness':
                    cost = rew
                    cost_vector[0] = rew
                    cost_vector[1] = rew2
                elif constraint == 'inf':
                    cost = 10**10
                
                # ret_eps += rew
                # ret_eps2 += rew2

                ret_eps += rew
                ret_eps2 += rew2

                # cost_ret_eps += (c_gamma ** t) * cost
                cost_ret_eps += cost

                # with open('obs.pkl', 'wb') as f:
                #     pickle.dump(next_obs, f)

                # debug
                if running_stat2 is not None:
                    next_obs = running_stat2.normalize(next_obs)
                if running_stat is not None:
                    next_obs2 = running_stat.normalize(next_obs2)

                # Store in episode buffer
                self.obs_eps[t] = obs
                self.obs_eps2[t] = obs2
                self.act_eps[t] = act
                self.next_obs_eps[t] = next_obs
                self.next_obs_eps2[t] = next_obs2
                # self.rew_eps[t] = rew
                self.rew_eps[0][t] = rew
                self.rew_eps[1][t] = rew2


                
                self.cost_eps[t] = cost

                obs = next_obs
                obs2 = next_obs2

                self.eps_len += 1
                batch_idx += 1


                # Store return for score if only episode is terminal
                if done2 or t == self.max_eps_len - 1:
                    if done2:
                        self.not_terminal = 0
                    
                    # score_queue.append(ret_eps)
                    score_queue_list[0].append(ret_eps)
                    score_queue_list[1].append(ret_eps2)
                    cscore_queue.append(cost_ret_eps)

                    # for group fairness, collect performance of (each agent's) return
                    ret_hist.append(ret_eps)
                    ret_hist2.append(ret_eps2)
                    
                    cost_ret_hist.append(cost_ret_eps)

                    num_eps += 1
                    avg_eps_len += (self.eps_len - avg_eps_len) / num_eps

                # if done2 or t == self.max_eps_len - 1:
                #     if done2:
                #         self.not_terminal = 0
                #     score_queue_list[1].append(ret_eps2)
                #     ret_hist2.append(ret_eps2)
                    
                #     num_eps += 1
                #     avg_eps_len += (self.eps_len - avg_eps_len) / num_eps

                

                if done2 or batch_idx == self.batch_size:
                    break


            # Store episode buffer
            self.obs_eps, self.next_obs_eps = self.obs_eps[:self.eps_len], self.next_obs_eps[:self.eps_len]
            self.obs_eps2, self.next_obs_eps2 = self.obs_eps2[:self.eps_len], self.next_obs_eps2[:self.eps_len]
            # self.act_eps, self.rew_eps = self.act_eps[:self.eps_len], self.rew_eps[:self.eps_len]

            self.act_eps, self.rew_eps[0] = self.act_eps[:self.eps_len], self.rew_eps[0][:self.eps_len]
            self.rew_eps[1] = self.rew_eps[1][:self.eps_len]



            self.cost_eps = self.cost_eps[:self.eps_len]


            # Calculate advantage

            adv_eps_list = []
            vtarg_eps_list = []
            for n in range(2):
                adv_eps, vtarg_eps = self.get_advantage(value_net_list[n], gamma, gae_lam, dtype, device, mode='reward', reward_idx = n)
                
                
                adv_eps_list.append(adv_eps)
                vtarg_eps_list.append(vtarg_eps)

                
                
            # adv_eps, vtarg_eps = self.get_advantage(value_net, gamma, gae_lam, dtype, device, mode='reward')
            # cadv_eps, cvtarg_eps = self.get_advantage(cvalue_net, c_gamma, c_gae_lam, dtype, device, mode='cost')
            cadv_eps, cvtarg_eps = self.get_advantage(value_net_list[0], c_gamma, c_gae_lam, dtype, device, mode='cost')
            
            # try stacking multiple cost advantage estimates. 
            cadv_eps_stack = np.stack((cadv_eps, -cadv_eps), axis=1)


            vtarg_eps_stack = np.stack(vtarg_eps_list, axis=1)
            adv_eps_stack = np.stack(adv_eps_list, axis=1)

            # with open('adv_eps_list.pkl', 'wb') as f:
            #     pickle.dump(adv_eps_list, f)

            
            # Update batch buffer
            start_idx, end_idx = self.ptr, self.ptr + self.eps_len
            self.obs_buf[start_idx: end_idx], self.act_buf[start_idx: end_idx] = self.obs_eps, self.act_eps
            self.obs_buf2[start_idx: end_idx] = self.obs_eps2
            self.vtarg_buf[start_idx: end_idx], self.adv_buf[start_idx: end_idx] = vtarg_eps_stack, adv_eps_stack
            # self.vtarg_buf[start_idx: end_idx], self.adv_buf[start_idx: end_idx] = vtarg_eps, adv_eps
            self.cvtarg_buf[start_idx: end_idx], self.cadv_buf[start_idx: end_idx] = cvtarg_eps, cadv_eps_stack

            # debugging
            n = 0
            adv_eps, vtarg_eps = self.get_advantage(value_net_list[n], gamma, gae_lam, dtype, device, mode='reward', reward_idx = n)
            self.adv_buf0[start_idx: end_idx] = adv_eps
            n = 1
            adv_eps, vtarg_eps = self.get_advantage(value_net_list[n], gamma, gae_lam, dtype, device, mode='reward', reward_idx = n)
            self.adv_buf1[start_idx: end_idx] = adv_eps
            



            # Update pointer
            self.ptr = end_idx

            # Reset episode buffer and update pointer
            self.obs_eps = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.obs_eps2 = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.next_obs_eps = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.next_obs_eps2 = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.act_eps = np.zeros((self.max_eps_len, self.act_dim), dtype=np.float32)
            self.rew_eps = [np.zeros((self.max_eps_len, 1),  dtype=np.float32), np.zeros((self.max_eps_len, 1),  dtype=np.float32)]
            # self.rew_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.cost_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.eps_len = 0
            self.not_terminal = 1

        # for group fairness, calculate average return
        avg_ret = [np.mean(ret_hist), np.mean(ret_hist2)]
        
        avg_cost = np.mean(cost_ret_hist)
        std_cost = np.std(cost_ret_hist)

        # Normalize advantage functions
        # self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-6)
        self.adv_buf = (self.adv_buf - self.adv_buf.mean(axis=0)) / (self.adv_buf.std(axis=0) + 1e-6)
        self.cadv_buf = (self.cadv_buf - self.cadv_buf.mean(axis=0)) / (self.cadv_buf.std(axis=0) + 1e-6)

        # debugging
        self.adv_buf0 = (self.adv_buf0 - self.adv_buf0.mean()) / (self.adv_buf0.std() + 1e-6)
        self.adv_buf1 = (self.adv_buf1 - self.adv_buf1.mean()) / (self.adv_buf1.std() + 1e-6)
        
        # for group fairness, added avg return
        return {'states':self.obs_buf,'states2':self.obs_buf2, 'actions':self.act_buf,
                'v_targets': self.vtarg_buf,'advantages': [self.adv_buf0, self.adv_buf1],
                'cv_targets': self.cvtarg_buf, 'c_advantages': self.cadv_buf,
                'avg_cost': avg_cost, 'std_cost': std_cost, 'avg_eps_len': avg_eps_len, 'avg_return':avg_ret}
    


    def get_advantage(self, value_net, gamma, gae_lam, dtype, device, mode='reward', reward_idx = 0):
        gae_delta = np.zeros((self.eps_len, 1))
        adv_eps =  np.zeros((self.eps_len, 1))
        # Check if terminal state, if terminal V(S_T) = 0, else V(S_T)
        status = np.ones((self.eps_len, 1))
        status[-1] = self.not_terminal
        prev_adv = 0

        for t in reversed(range(self.eps_len)):
            # Get value for current and next state
            if reward_idx == 0:
                obs_tensor = torch.Tensor(self.obs_eps[t]).to(dtype).to(device)
                next_obs_tensor = torch.Tensor(self.next_obs_eps[t]).to(dtype).to(device)
            else:
                obs_tensor = torch.Tensor(self.obs_eps2[t]).to(dtype).to(device)
                next_obs_tensor = torch.Tensor(self.next_obs_eps2[t]).to(dtype).to(device)
            
            current_val, next_val = torch_to_numpy(value_net(obs_tensor), value_net(next_obs_tensor))

            # Calculate delta and advantage
            if mode == 'reward':
                gae_delta[t] = self.rew_eps[reward_idx][t] + gamma * next_val * status[t] - current_val
            elif mode =='cost':
                gae_delta[t] = self.cost_eps[t] + gamma * next_val * status[t] - current_val
            adv_eps[t] = gae_delta[t] + gamma * gae_lam * prev_adv

            # Update previous advantage
            prev_adv = adv_eps[t]


        # Get target for value function
        if reward_idx == 0:
            
            obs_eps_tensor = torch.Tensor(self.obs_eps).to(dtype).to(device)
        else:
            obs_eps_tensor = torch.Tensor(self.obs_eps2).to(dtype).to(device)
        vtarg_eps = torch_to_numpy(value_net(obs_eps_tensor)) + adv_eps



        return adv_eps, vtarg_eps
