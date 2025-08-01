'''
Paper
    Dueling Network Architectures for Deep Reinforcement Learnin

Abstract
    The main benefit of this factoring is
    to generalize learning across actions
    without imposing any change to the underlying reinforcement learning algorithm.
    
    Our results show that
    this architecture leads to better policy evaluation
    in the presence of many similar-valued actions
'''

from model.DQN import DQN, DQNAgent

from torch import nn

class DeulingDQN(DQN):
    def __init__(self, *args):
        super().__init__(*args)

        self.state_value_fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            )

    def forward(self, x):
        out = self.conv(x)
        state_value = self.state_value_fc(out)
        advantage = self.fc(out)
        Q_value = state_value + advantage
        return Q_value
    
class DeulingDQNAgent(DQNAgent):
    def __init__(self, *args):
        super().__init__(*args)

        self.behavior_network = DeulingDQN(self.state_shape, self.action_space)
        self.target_network = DeulingDQN(self.state_shape, self.action_space)