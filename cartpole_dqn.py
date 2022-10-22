import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import copy #

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from pickle import TRUE
import torch
from torch import nn
import copy
from collections import deque
import random

import gym
from tqdm import tqdm
import pylab 
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

#env = gym.make('CartPole-v0', render_mode='rgb_array').unwrapped
#env = gym.make('CartPole-v0').unwrapped


# observation_space = env.observation_space.shape[0]
# action_space = env.action_space.n
# n_actions = env.action_space.n
#DQN_Agent = DQN_Agent(observation_space, action_space)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(*[
            nn.Linear(input_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, output_dim), 
        ])
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, s):
        out = self.network(s)
        return out
    
def optimize_model(online_dqn, target_dqn, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # state_action_values = online_dqn(state_batch).gather(1, action_batch) # torch.gather()
    state_action_values = online_dqn(state_batch)[np.arange(len(state_batch)), action_batch]

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in online_dqn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    print(loss)
    
env = gym.make("CartPole-v0")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

num_episodes = 400
agent = DQN(obs_dim, action_dim)
target_agent = DQN(obs_dim, action_dim)
target_agent.load_state_dict(agent.state_dict())

optimiser = torch.optim.RMSprop(agent.parameters())

memory = ReplayMemory(10000)

episode_durations = []


steps_done = 0

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()

    # state, action, next_state, reward = env.reset()[0] ## ERROR HERE
    state = env.reset()[0]
    
    for t in count():
        # Select and perform an action
        # action = agent.select_action(obs, state)
        
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        rnd = np.random.uniform(0, 1)
        if rnd < eps_threshold:
            action = np.random.randint(action_dim)
        else:
            action = agent(torch.tensor(state[None, :])).max(dim=-1)[1].item()
        next_state, reward, done, _, _ = env.step(action)
        # reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(torch.tensor(state)[None, :], torch.tensor([action]), 
                    torch.tensor(next_state)[None, :], torch.tensor([reward]))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model(agent, target_agent, optimiser, memory)
        
        steps_done += 1
        if done:
            episode_durations.append(t + 1)
            # plot_durations(episode_durations)
            break
        
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_agent.load_state_dict(agent.state_dict())

print('Complete')
# env.render()
# env.close()
# plt.ioff()
# plt.show()
