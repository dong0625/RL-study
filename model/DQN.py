'''
Paper
    Playing Atari with Deep Reinforcement Learning
    Human-level control through deep reinforcement learning

Abstract
    The model is a convolutional neural network,
    trained with a variant of Q-learning
    whose input is raw pixels and
    whose output is a value function estimating future rewards.
'''

from base import Model
from utils.replay_buffer import ReplayBuffer

import torch
from torch import nn
from torch.functional import F

import random

class Qnetwork(nn.Module):
    def __init__(self, input_shape, output_space):
        super().__init__()

        self.input_shape = input_shape
        self.output_space = output_space

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
            )
        
        dummy = self.conv(torch.zeros(1, *input_shape)).view(-1)
        self.fc_input_size = dummy.shape.item()
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_space)
            )
        
    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out.view(-1, self.fc_input_size))
        return out
    
class DQN(Model):
    def __init__(self, config, env):
        super().__init__(config, env)

        self.batch_size = config.batch_size
        self.state_shape = env.state_shape
        self.action_space = env.action_space

        self.behavior_network = Qnetwork(self.state_shape, self.action_space)
        self.target_network = Qnetwork(self.state_shape, self.action_space)
        
        self.gamma = config.gamma
        self.epsilon = config.epsilon

        self.replay_buffer = ReplayBuffer(config.capacity, self.state_shape)

    def get_action(self, state):
        state = state.view(-1, *self.state_shape)

        if random.random() < self.epsilon:
            action = random.randrange(self.action_space)
            return action
        
    def get_value(self, states, actions=None):
        if actions == None:
            Q_values = self.behavior_network(states)
            return torch.max(Q_values, dim=-1)
        else:
            Q_values = self.behavior_network(states)
            return Q_values[:, actions]
    
    def get_loss(self, *frame):
        self.replay_buffer.store(*frame)

        batch = self.replay_buffer.sample(self.batch_size)
        if batch == None:
            return None
        
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            y = rewards + self.gamma * (1 - dones) * self.get_value(next_states)
        pred = self.get_value(states, actions)

        loss = F.mse_loss(pred, y)

        return loss