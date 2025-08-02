'''
Paper
    Deep Reinforcement Learning with Double Q-Learning

Abstract
    We show that the resulting algorithm not only reduces the observed overestimations, as hypothesized,
    but that this also leads to much better performance on several games.
'''

from model.DQN import DQNAgent

import torch
from torch.nn import functional as F

class DoubleDQN(DQNAgent):
    def get_loss(self, *frame):
        self.replay_buffer.store(*frame)
        
        batch = self.replay_buffer.sample(self.batch_size)
        if batch == None:
            return torch.zeros(1, requires_grad=True)
        
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            max_a = self.behavior_network(next_states).argmax(-1, keepdim=True)
            target_Q = self.target_network(next_states)
            target_Q = torch.gather(target_Q, 1, max_a).squeeze(-1)
            y = rewards + self.gamma * (1 - dones) * target_Q

        pred_Q = self.behavior_network(states)
        pred = torch.gather(pred_Q, 1, actions.unsqueeze(1)).squeeze(-1)

        loss = F.smooth_l1_loss(pred, y)

        self.update()
        return loss