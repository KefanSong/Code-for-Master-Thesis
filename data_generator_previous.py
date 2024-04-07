import numpy as np
import torch
from utils import torch_to_numpy


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
        self.act_buf = np.zeros((batch_size, act_dim),  dtype=np.float32)
        self.vtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.adv_buf = np.zeros((batch_size, 1), dtype=np.float32)
        self.cvtarg_buf = np.zeros((batch_size, 1), dtype=np.float32)
        # self.cadv_buf = np.zeros((batch_size, 1), dtype=np.float32)

        self.cadv_buf = np.zeros((batch_size,2, 1), dtype=np.float32)

        
        # Episode buffer
        self.obs_eps = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.next_obs_eps = np.zeros((max_eps_len, obs_dim),  dtype=np.float32)
        self.act_eps = np.zeros((max_eps_len, act_dim),  dtype=np.float32)
        self.rew_eps = np.zeros((max_eps_len, 1),  dtype=np.float32)
        # self.cost_eps = np.zeros((max_eps_len, 1), dtype=np.float32)
        self.cost_eps = [np.zeros((max_eps_len, 1), dtype=np.float32)]*2
        self.eps_len = 0
        self.not_terminal = 1


        # Pointer
        self.ptr = 0

    def run_traj(self, env, policy, value_net, cvalue_net, running_stat,
                 score_queue, cscore_queue, gamma, c_gamma, gae_lam, c_gae_lam,
                 dtype, device, constraint):

        batch_idx = 0

        ret_hist = []
        cost_ret_hist = [[]]*2

        avg_eps_len = 0
        num_eps = 0



        

        while batch_idx < self.batch_size:
            obs = env.reset()[0]
            if running_stat is not None:
                obs = running_stat.normalize(obs)
            ret_eps = 0
            # cost_ret_eps = 0

            cost_ret_eps_vector = [0]*2

            for t in range(self.max_eps_len):
                act = policy.get_act(torch.Tensor(obs).to(dtype).to(device))
                act = torch_to_numpy(act).squeeze()
                next_obs, rew, done, truncated, info = env.step(act)

                cost_vector = [0]*2
                if constraint == 'velocity':
                    if 'y_velocity' not in info:
                        cost = np.abs(info['x_velocity'])
                    else:
                        cost = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
                elif constraint == 'circle':
                    cost = info['cost']
                elif constraint == 'group fairness':
                    cost_vector[0] = rew
                    cost_vector[1] = -rew

                ret_eps += rew
                # cost_ret_eps += (c_gamma ** t) * cost
                # cost_ret_eps += cost

                cost_ret_eps_vector[0] += cost_vector[0]
                cost_ret_eps_vector[1] += cost_vector[1]

                if running_stat is not None:
                    next_obs = running_stat.normalize(next_obs)

                # Store in episode buffer
                self.obs_eps[t] = obs
                self.act_eps[t] = act
                self.next_obs_eps[t] = next_obs
                self.rew_eps[t] = rew
                # self.cost_eps[t] = cost
                self.cost_eps[0][t] = cost_vector[0]
                self.cost_eps[1][t] = cost_vector[1]

                
                obs = next_obs

                self.eps_len += 1
                batch_idx += 1

                # Store return for score if only episode is terminal
                if done or t == self.max_eps_len - 1:
                    if done:
                        self.not_terminal = 0
                    score_queue.append(ret_eps)
                    # use the first constraint for cscore for now.
                    cscore_queue.append(cost_ret_eps_vector[0])

                    # for group fairness, collect performance of (each agent's) return
                    ret_hist.append(ret_eps)
                    cost_ret_hist[0].append(cost_ret_eps_vector[0])
                    cost_ret_hist[1].append(cost_ret_eps_vector[1])

                    num_eps += 1
                    avg_eps_len += (self.eps_len - avg_eps_len) / num_eps

                if done or batch_idx == self.batch_size:
                    break

            # Store episode buffer
            self.obs_eps, self.next_obs_eps = self.obs_eps[:self.eps_len], self.next_obs_eps[:self.eps_len]
            self.act_eps, self.rew_eps = self.act_eps[:self.eps_len], self.rew_eps[:self.eps_len]
            # self.cost_eps = self.cost_eps[:self.eps_len]
            self.cost_eps[0] = self.cost_eps[0][:self.eps_len]
            self.cost_eps[1] = self.cost_eps[1][:self.eps_len]


            # Calculate advantage
            adv_eps, vtarg_eps = self.get_advantage(value_net, gamma, gae_lam, dtype, device, mode='reward')
            cadv_eps_vector = [0]*2
            cvtarg_eps_vector = [0]*2
            cadv_eps_vector[0], cvtarg_eps_vector[0] = self.get_advantage(cvalue_net, c_gamma, c_gae_lam, dtype, device, mode='cost',constraint_idx=0)
            # cadv_eps_vector[1], cvtarg_eps_vector[1] = self.get_advantage(cvalue_net, c_gamma, c_gae_lam, dtype, device, mode='cost',1)
            cadv_eps_vector[1] = cadv_eps_vector[0]
            
            # Update batch buffer
            start_idx, end_idx = self.ptr, self.ptr + self.eps_len
            self.obs_buf[start_idx: end_idx], self.act_buf[start_idx: end_idx] = self.obs_eps, self.act_eps
            self.vtarg_buf[start_idx: end_idx], self.adv_buf[start_idx: end_idx] = vtarg_eps, adv_eps
            # self.cvtarg_buf[start_idx: end_idx], self.cadv_buf[start_idx: end_idx] = cvtarg_eps, cadv_eps
            self.cvtarg_buf[start_idx: end_idx], self.cadv_buf[start_idx: end_idx, 0, :] = cvtarg_eps_vector[0], cadv_eps_vector[0]
            self.cadv_buf[start_idx: end_idx, 1, :] = cadv_eps_vector[1]


            # Update pointer
            self.ptr = end_idx

            # Reset episode buffer and update pointer
            self.obs_eps = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.next_obs_eps = np.zeros((self.max_eps_len, self.obs_dim), dtype=np.float32)
            self.act_eps = np.zeros((self.max_eps_len, self.act_dim), dtype=np.float32)
            self.rew_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            # self.cost_eps = np.zeros((self.max_eps_len, 1), dtype=np.float32)
            self.cost_eps = [np.zeros((self.max_eps_len, 1), dtype=np.float32)]*2
            self.eps_len = 0
            self.not_terminal = 1

        # for group fairness, calculate average return
        avg_ret = np.mean(ret_hist)

        # this need to be a vector.
        avg_cost_vector = std_cost_vector = [0]*2
        
        avg_cost_vector[0] = np.mean(cost_ret_hist[0])
        avg_cost_vector[1] = np.mean(cost_ret_hist[1])
        std_cost_vector[0] = np.std(cost_ret_hist[0])
        std_cost_vector[1] = np.std(cost_ret_hist[1])


        # Normalize advantage functions
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-6)
        # self.cadv_buf = (self.cadv_buf - self.cadv_buf.mean()) / (self.cadv_buf.std() + 1e-6)

        self.cadv_buf[:, 0, :] = (self.cadv_buf[:, 0, :] - self.cadv_buf[:, 0, :].mean()) / (self.cadv_buf[:, 0, :].std() + 1e-6)
        self.cadv_buf[:, 1, :] = (self.cadv_buf[:, 1, :] - self.cadv_buf[:, 1, :].mean()) / (self.cadv_buf[:, 1, :].std() + 1e-6)


        # for group fairness, added avg return
        return {'states':self.obs_buf, 'actions':self.act_buf,
                'v_targets': self.vtarg_buf,'advantages': self.adv_buf,
                'cv_targets': self.cvtarg_buf, 'c_advantages': self.cadv_buf,
                'avg_cost': avg_cost_vector, 'std_cost': std_cost_vector, 'avg_eps_len': avg_eps_len, 'avg_return':avg_ret}
    


    def get_advantage(self, value_net, gamma, gae_lam, dtype, device, mode='reward', constraint_idx=0):
        gae_delta = np.zeros((self.eps_len, 1))
        adv_eps =  np.zeros((self.eps_len, 1))
        # Check if terminal state, if terminal V(S_T) = 0, else V(S_T)
        status = np.ones((self.eps_len, 1))
        status[-1] = self.not_terminal
        prev_adv = 0

        for t in reversed(range(self.eps_len)):
            # Get value for current and next state
            obs_tensor = torch.Tensor(self.obs_eps[t]).to(dtype).to(device)
            next_obs_tensor = torch.Tensor(self.next_obs_eps[t]).to(dtype).to(device)
            current_val, next_val = torch_to_numpy(value_net(obs_tensor), value_net(next_obs_tensor))

            # Calculate delta and advantage
            if mode == 'reward':
                gae_delta[t] = self.rew_eps[t] + gamma * next_val * status[t] - current_val
            elif mode =='cost':
                gae_delta[t] = self.cost_eps[constraint_idx][t] + gamma * next_val * status[t] - current_val
            adv_eps[t] = gae_delta[t] + gamma * gae_lam * prev_adv

            # Update previous advantage
            prev_adv = adv_eps[t]

        # Get target for value function
        obs_eps_tensor = torch.Tensor(self.obs_eps).to(dtype).to(device)
        vtarg_eps = torch_to_numpy(value_net(obs_eps_tensor)) + adv_eps
        


        return adv_eps, vtarg_eps
