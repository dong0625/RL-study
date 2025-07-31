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

from model.base import Model
from utils.replay_buffer import ReplayBuffer

import torch
from torch import nn
from torch.functional import F

import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            nn.ReLU(),
            nn.Flatten()
            )

        dummy = self.conv(torch.zeros(1, *input_shape)).view(-1)
        self.fc_input_size = dummy.shape[0]
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_space)
            )
        
    def forward(self, x) -> torch.Tensor:
        out = self.conv(x)
        out = self.fc(out)
        return out
    
class DQN(Model):
    def __init__(self, config, env):
        super().__init__(config, env)

        self.batch_size = config.batch_size
        self.state_shape = env.state_shape
        self.action_space = env.action_space

        self.behavior_network = Qnetwork(self.state_shape, self.action_space)
        self.target_network = Qnetwork(self.state_shape, self.action_space)
        
        self.update_n = 0

        self.gamma = config.gamma
        self.eps = lambda n : \
            config.initial_eps + \
            (config.final_eps - config.initial_eps) * min(1, n / config.final_eps_step)

        self.replay_buffer = ReplayBuffer(config.capacity, self.state_shape, config.learning_starts)

    def get_action(self, state : torch.Tensor):
        if self.training and random.random() < self.eps(self.update_n):
            action = random.randrange(self.action_space)
            return action
        
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            Q_values = self.behavior_network(state)
            actions = torch.argmax(Q_values, dim=-1)
            return actions.item()
    
    def get_loss(self, *frame):
        self.replay_buffer.store(*frame)
        
        batch = self.replay_buffer.sample(self.batch_size)
        if batch == None:
            return torch.zeros(1, requires_grad=True)
        
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            target_Q = self.target_network(next_states)
            y = rewards + self.gamma * (1 - dones) * target_Q.max(dim=-1).values

        pred_Q = self.behavior_network(states)
        pred = torch.gather(pred_Q, 1, actions.unsqueeze(1)).squeeze(-1)

        loss = F.mse_loss(pred, y)

        self.update()
        return loss
    
    def update(self):
        if self.update_n % 10000 == 0:
            self.target_network.load_state_dict(self.behavior_network.state_dict())

        self.update_n += 1