import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import imageio


from IPython.display import HTML, display
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import pickle

import os
#### If you encounter error related to "libiomp5md.dll", you can try the code in the next line.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================================
#  Define Basic Classes
#  - Finish get_state_values() function in class DQN
#  - Finish forward() function and get_state_values() function in DuelingDQN class for bonus
# ==================================================================

#### Definition of Standard Experience Replay Buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminated'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args): 
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#### Definition of Standard DQN network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.L1 = nn.Linear(n_observations, 64)
        self.L2 = nn.Linear(64, 64)
        self.L3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        return self.L3(x)
    
    # Get Q(s,a)
    def get_state_action_values(self, state_batch, action_batch): 
        q_values = self(state_batch)
        row_index = torch.arange(0, state_batch.shape[0])
        selected_actions_q_values = q_values[row_index, action_batch]
        return selected_actions_q_values
    
    # Get max_a Q(s,a) based on the Q(s,a) above, this will be used to calculate the target to learn.
    def get_state_values(self, state_batch): 
        #### TODO
        q_values = self(state_batch)
        max_q_values, _ = torch.max(q_values, dim=-1)
        return max_q_values

    def get_argmax_a(self, state):
        q_values = self(state)
        argmax_a = torch.argmax(q_values)
        return argmax_a

    def get_batch_argmax_a(self, state_batch):
        q_values = self(state_batch)
        argmax_a = torch.argmax(q_values, dim=-1)
        return argmax_a

#### Definition of Dueling Networkss
class DuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DuelingDQN, self).__init__()
        # self.feature_layer = nn.Linear(n_observations, 64)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        #### TODO
        return 

    # Get Q(s,a)
    def get_state_action_values(self, state_batch, action_batch): 
        q_values = self(state_batch)
        row_index = torch.arange(0, state_batch.shape[0])
        selected_actions_q_values = q_values[row_index, action_batch]
        return selected_actions_q_values
    
    # Get max_a Q(s,a)
    def get_state_values(self, state_batch): 
        #### TODO
        return 

#=========================================================================



# ===================================
#  Hyperparameters
# ===================================
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.003
EPS_DECAY = 1000
TAU = 0.002
LR = 1e-4
# ==================================


# =====================================================================
#  Initialize Environment and Networks
# =====================================================================
env = gym.make('LunarLander-v2')
#### Get the dimension of action and state
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]


#### Initilize DQN/DDQN Networks and optimizer
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())   
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)


#### Initilize Dueling Networks and optimizer
policy_net_duel = DuelingDQN(n_observations, n_actions).to(device)
target_net_duel = DuelingDQN(n_observations, n_actions).to(device)
target_net_duel.load_state_dict(policy_net_duel.state_dict())   
## Only update the parameter for the policy network
optimizer_duel = optim.AdamW(policy_net_duel.parameters(), lr=LR, amsgrad=True)

#### Initizalize Experience Replay Buffer
memory = ReplayMemory(10000)

# ======================================================================





# ===============================================================================================================
#  Define Main Algorithms
#  - Finish the update of optimize_model_DQN() amd optimize_model_DDQN()
#  - Finish the update of optimize_model_DN() for bonus 
# ===============================================================================================================

#### Implementation of vanilla DQN
def optimize_model_DQN():
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, terminateds = zip(*memory.sample(BATCH_SIZE))
    state_batch = torch.tensor(np.array(states), device=device, dtype=torch.float)
    action_batch = torch.tensor(actions, device=device)
    reward_batch = torch.tensor(rewards, device=device, dtype=torch.float)
    next_state_batch = torch.tensor(np.array(next_states), device=device, dtype=torch.float)
    terminated_batch = torch.tensor(terminateds, dtype=torch.int, device=device)
    
    #### TODO
    ## state_action_values = Q(s,a, \theta) 
    ## expected_state_action_values = r + \gamma max_a Q(s',a, \theta_{tar})
    # print(np.shape(state_batch), BATCH_SIZE)

    state_action_values = policy_net.get_state_action_values(state_batch, action_batch)

    # next states 
    max_q_values = target_net.get_state_values(next_state_batch).detach()
    expected_state_action_values = reward_batch + GAMMA*max_q_values*(1-terminated_batch)

    print(terminated_batch)

    
    loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#### Implementation of Double DQN
def optimize_model_DDQN():
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, terminateds = zip(*memory.sample(BATCH_SIZE))
    state_batch = torch.tensor(np.array(states), device=device, dtype=torch.float)
    action_batch = torch.tensor(actions, device=device)
    reward_batch = torch.tensor(rewards, device=device, dtype=torch.float)
    next_state_batch = torch.tensor(np.array(next_states), device=device, dtype=torch.float)
    terminated_batch = torch.tensor(terminateds, dtype=torch.int, device=device)

    #### TODO
    ## state_action_values = Q(s,a, \theta) 
    ## expected_state_action_values = r + \gamma Q(s',a', \theta_{tar}) where a' = argmax_a Q(s',a, \theta)
    state_action_values = policy_net.get_state_action_values(state_batch, action_batch)
    
    batch_argmax_a = policy_net.get_batch_argmax_a(state_batch)
    expected_state_action_values = target_net.get_state_action_values(next_state_batch, batch_argmax_a)

    loss = F.mse_loss(state_action_values, expected_state_action_values)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#### Implementation of Double DQN + Dueling Network
def optimize_model_DN():
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, terminateds = zip(*memory.sample(BATCH_SIZE))
    state_batch = torch.tensor(np.array(states), device=device, dtype=torch.float)
    action_batch = torch.tensor(actions, device=device)
    reward_batch = torch.tensor(rewards, device=device, dtype=torch.float)
    next_state_batch = torch.tensor(np.array(next_states), device=device, dtype=torch.float)
    terminated_batch = torch.tensor(terminateds, dtype=torch.int, device=device)

    #### TODO
    ## state_action_values = Q(s,a, \theta)
    ## expected_state_action_values = r + \gamma Q(s',a', \theta_{tar}) where a' = argmax_a Q(s',a, \theta)
    ## The same as DDQN except using policy_net_duel and target_net_duel
    state_action_values = 0
    expected_state_action_values = 0

    loss = F.mse_loss(state_action_values, expected_state_action_values)

    optimizer_duel.zero_grad()
    loss.backward()
    optimizer_duel.step()


# ==============================================================================================================================




# ===============================================================================================================
#  Main Train Loop
#  - Finish the epsilon greedy exploration
# ===============================================================================================================

#### Training Episodes
NUM_EPISODES = 2000

#### Training Loop. If the input algorithm == "DQN", it will utilize DQN to train. 
#### Similarly, if the input algorithm == "DDQN", it will utilize DDQN to train. If the input algorithm == "DN", it will utilize Dueling Networks to train
def train_models(algorithm):


    
    episode_returns = []
    average_returns = []
    for iteration in range(NUM_EPISODES):
    ## Choose action based on epsilon greedy
        current_episode_return = 0
        state, _ = env.reset()
        terminated = 0
        truncated = 0

        while not (terminated or truncated):
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * iteration / EPS_DECAY)
            if random.random() > eps:
                if algorithm == "DQN" or "DDQN":
                    #### TODO
                    #### Finish the action based on Algorthm 1 in Homework.

                    action = policy_net.get_argmax_a(torch.tensor(state)).item()
                if algorithm == "DN":
                    #### TODO
                    #### Finish the action based on Algorthm 2 in Homework (The same as Algorithm 1 but use different networks).
                    action = 0
            else:
                action = random.randrange(n_actions)


            next_state, reward, terminated, truncated, _ = env.step(action)
            memory.push(state, action, reward, next_state, terminated)
            current_episode_return += reward

            #### Update the target model
            if algorithm == "DQN" or "DDQN":
                for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
            
            if algorithm == "DN":
                for target_param, policy_param in zip(target_net_duel.parameters(), policy_net_duel.parameters()):
                    target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
            
            #### Determine whether an episode is terminated
            if terminated or truncated:
                if iteration % 20 == 0:
                    print('Episode {},  score: {}'.format(iteration, current_episode_return))
                
                episode_returns.append(current_episode_return)
                #### Store the average returns of 100 consecutive episodes
                if iteration < 100:
                    average_returns.append(np.average(episode_returns))
                else:
                    average_returns.append(np.average(episode_returns[iteration-100: iteration]))
                
                
            else: 
                state = next_state

            ## Choose your algorithm here
            if algorithm == "DQN":
                optimize_model_DQN()
            if algorithm == "DDQN":
                optimize_model_DDQN()
            if algorithm == "DN":
                optimize_model_DN()
    

    plt.title('Training with ' + algorithm)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_returns, label = "Episode Rewards")
    plt.plot(average_returns, label = "Average Rewards")
    plt.legend()
    plt.savefig("Training with " + algorithm)
    plt.show()



# ===============================================================================================================
#  Functions to generate videos for visulization 
#  (This part is only for playing. You do not need to upload any video for homework)
#  To play with your own policy, you can save your policy network and put it into generate_videos(policy) as policy
#  You also need to complete a line in generate_videos(policy) to play.
#  This line is the same as the line you finishes in train_models(algorithm), choosing the best action based on your model.
# ===============================================================================================================
def generate_videos(policy):
    env = gym.make('LunarLander-v2', render_mode = 'rgb_array')
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]

    frames = []

    state, _ = env.reset()
    img = env.render()
    frames.append(img)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if policy == "random":
            action = np.random.randint(0, n_actions)
        else:
            #### Change this line to choose the best action based on your own policy_net
            #### It is the same as the line you finishes in function "train_models(algorithm)"", which chooses the best action based on your model.
            action = 0
        next_state, _, terminated, truncated, _ = env.step(action)
        img = env.render()
        frames.append(img)
        if terminated or truncated:
            break
        state = next_state
    imageio.mimsave('lunar_lander_random.mp4', frames, fps=60)

# ===============================================================================================================



if __name__ == "__main__":
    #generate_videos("random")
    # train_models("DQN")
    train_models("DDQN")
    #train_models("DN")
